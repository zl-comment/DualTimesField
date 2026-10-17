"""Meaningful invariants for the canonical historical dual-field adapter."""
import numpy as np
import pytest
import torch

from forecasting.full_v5.dual_history_model import (
    FIT_CUTOFF, HOUR, TRAIN_START, DualHistoryConfig, HistoricalDualField,
    fit_historical_dualfield, lowpass_price,
)


def inputs():
    rng = np.random.default_rng(2026)
    origins = np.array([TRAIN_START + 48 * HOUR, FIT_CUTOFF - 25 * HOUR,
                        FIT_CUTOFF - 24 * HOUR, FIT_CUTOFF + 10 * HOUR], dtype=np.int64)
    actual = (100 + 30 * rng.normal(size=(4, 24))).astype(np.float32)
    actual[:, 12] += 400
    z = rng.normal(scale=.3, size=(4, 24, 16)).astype(np.float32)
    lag = rng.normal(loc=100, scale=40, size=(4, 24, 3)).astype(np.float32)
    return origins, actual, z, lag


def small_config():
    return DualHistoryConfig(epochs=1, batch_size=2, hidden_dim=8,
                             num_atoms=3, num_frequencies=2)


def test_fit_strict_cutoff_and_future_values_cannot_affect_fit():
    args = inputs()
    model, audit = fit_historical_dualfield(*args, config=small_config())
    altered = [a.copy() for a in args]
    for a in altered[1:]:
        a[2:] = np.nan
    other, other_audit = fit_historical_dualfield(*altered, config=small_config())
    assert audit == other_audit
    assert audit['fit_rows'] == 2
    assert audit['latest_fit_label_available'] < FIT_CUTOFF
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)


def test_maturity_filters_before_forward_and_exact_boundary(monkeypatch):
    args = inputs()
    model, _ = fit_historical_dualfield(*args, config=small_config())
    bad = [a.copy() for a in args]
    for a in bad[1:]:
        a[3:] = np.nan
    seen = []
    old = model.forward

    def forward(price, z, lag_prices, **kwargs):
        seen.append(len(price))
        assert torch.isfinite(price).all()
        return old(price, z, lag_prices, **kwargs)

    monkeypatch.setattr(model, 'forward', forward)
    result = model.encode_matured(*bad, as_of=FIT_CUTOFF)
    assert result['indices'].tolist() == [0, 1, 2]
    assert sum(seen) == 3
    with pytest.raises(ValueError, match='not fitted'):
        model.encode_matured(*args, as_of=FIT_CUTOFF - 1)


def test_both_fields_and_live_gate_receive_gradient_without_shortcut():
    torch.set_num_threads(2)
    torch.manual_seed(9)
    model = HistoricalDualField(small_config())
    _, price, z, lag = inputs()
    price, z, lag = [torch.tensor(a[:2]) for a in (price, z, lag)]
    loss, terms = model.reconstruction_loss(price, z, lag)
    loss.backward()
    for parameter in (model.core.ctf.data_encoder[0].weight,
                      model.core.dgf.atoms.amplitude_head.weight,
                      model.core.dgf.atoms.gate_head.weight):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad
               for p in model.core.dgf.atoms.residual_net.parameters())
    model.zero_grad(set_to_none=True)
    output = model(price, z, lag)
    output['gates'].mean().backward()
    assert model.core.dgf.atoms.gate_head.weight.grad.abs().sum() > 0
    assert terms['live_gate'].requires_grad


def test_fields_are_additive_finite_nonzero_and_price_conditioned(tmp_path):
    args = inputs()
    model, audit = fit_historical_dualfield(*args, config=small_config())
    result = model.encode_matured(*args, as_of=FIT_CUTOFF)
    for key in ('trend', 'event', 'reconstruction'):
        assert result[key].shape == (3, 24)
        assert np.isfinite(result[key]).all()
        assert np.abs(result[key]).sum() > 0
    np.testing.assert_allclose(result['reconstruction'], result['trend'] + result['event'],
                               atol=3e-5, rtol=1e-5)
    changed = [a.copy() for a in args]
    changed[1][0] += np.linspace(-100., 200., 24)
    alternative = model.encode_matured(*changed, as_of=FIT_CUTOFF)
    assert not np.allclose(result['reconstruction'][0], alternative['reconstruction'][0])
    # Future curves, not just training rows, must not influence prior embeddings.
    changed[1][3] = 1e20
    causal = model.encode_matured(*changed, as_of=FIT_CUTOFF)
    np.testing.assert_array_equal(causal['reconstruction'], alternative['reconstruction'])
    path = tmp_path / 'encoder.pt'
    model.save_checkpoint(path, audit)
    restored, restored_audit = HistoricalDualField.load_checkpoint(path)
    restored_result = restored.encode_matured(*args, as_of=FIT_CUTOFF)
    assert restored_audit == audit
    np.testing.assert_array_equal(result['reconstruction'], restored_result['reconstruction'])


def test_background_prior_is_dc_plus_first_harmonic():
    t = torch.arange(24) * (2 * np.pi / 24)
    low = 2 + torch.sin(t)
    curves = (low + .5 * torch.cos(6 * t))[None]
    torch.testing.assert_close(lowpass_price(curves)[0], low, atol=1e-6, rtol=1e-6)


def test_rejects_bad_shapes_and_impossible_training_cutoff():
    args = inputs()
    with pytest.raises(ValueError, match='Expected actual'):
        fit_historical_dualfield(*args[:3], args[3][..., :2], config=small_config())
    with pytest.raises(ValueError, match='cutoff'):
        fit_historical_dualfield(*args, fit_cutoff=FIT_CUTOFF + HOUR,
                                 config=small_config())
