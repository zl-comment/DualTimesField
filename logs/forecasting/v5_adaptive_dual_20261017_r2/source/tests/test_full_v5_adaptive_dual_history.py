import numpy as np
import pytest
import torch

from forecasting.full_v5.adaptive_dual_history import (
    AdaptiveDualConfig, AdaptiveHistoricalDualField, FIT_CUTOFF,
    fit_adaptive_dualfield,
)
from forecasting.full_v5.dual_history_model import HOUR, TRAIN_START


def curves(n=4):
    t = torch.arange(24, dtype=torch.float32)
    price = (70 + 20 * torch.cos(2 * torch.pi * t / 24)).repeat(n, 1)
    for i in range(n):
        price[i] += 900 * torch.exp(-.5 * ((t - (3 + 4 * i)) / .8).square())
    return price, torch.zeros(n, 24, 16), torch.zeros(n, 24, 3)


def test_peaks_follow_curve_and_preserve_amplitude_and_mean():
    model = AdaptiveHistoricalDualField()
    price, z, lag = curves()
    output = model(price, z, lag, fixed=True)
    for i in range(4):
        assert abs(float(output['centers'][i, 0]) - (3 + 4 * i)) <= 1
        assert float(output['reconstruction'][i, 3 + 4 * i] * 300) > 800
    torch.testing.assert_close(output['reconstruction'].mean(1), price.mean(1) / 300,
                               atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output['trend'] + output['event'], output['reconstruction'])


def test_initial_adaptive_equals_fixed_control():
    model = AdaptiveHistoricalDualField()
    args = curves()
    adaptive, fixed = model(*args), model(*args, fixed=True)
    for key in ('trend', 'event', 'centers', 'widths', 'amplitude', 'reconstruction'):
        torch.testing.assert_close(adaptive[key], fixed[key], rtol=0, atol=0)


def test_masked_targets_never_affect_proposals_features_or_fit():
    model = AdaptiveHistoricalDualField()
    price, z, lag = curves()
    mask = torch.ones_like(price, dtype=torch.bool)
    mask[:, 2:8] = False
    changed = price.clone()
    changed[~mask] = float('nan')
    for fixed in (False, True):
        original = model(price, z, lag, observed_mask=mask, fixed=fixed)
        perturbed = model(changed, z, lag, observed_mask=mask, fixed=fixed)
        for key in ('trend', 'event', 'reconstruction', 'centers', 'widths', 'amplitude'):
            torch.testing.assert_close(original[key], perturbed[key], atol=0, rtol=0)
        pred = original['reconstruction']
        torch.testing.assert_close((pred * mask).sum(1), (price / 300 * mask).sum(1),
                                   rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match='at least one'):
        model(price, z, lag, observed_mask=torch.zeros_like(mask))


def test_geometry_receives_finite_nonzero_gradient():
    model = AdaptiveHistoricalDualField()
    loss = model.reconstruction_loss(*curves())
    loss.backward()
    gradient = model.geometry[-1].weight.grad
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    optimizer.step()
    output = model(*curves())
    assert bool((output['widths'] >= .25).all())
    assert bool((output['widths'] <= 1.5).all())
    assert not torch.equal(output['widths'], torch.full_like(output['widths'], .5))


def test_fitting_ignores_all_post_cutoff_targets():
    args = [a.numpy() for a in curves()]
    origins = np.array([TRAIN_START, TRAIN_START + HOUR, FIT_CUTOFF - 24 * HOUR,
                        FIT_CUTOFF], dtype=np.int64)
    config = AdaptiveDualConfig(epochs=1, batch_size=2)
    first, audit = fit_adaptive_dualfield(origins, *args, config=config)
    changed = args[0].copy()
    changed[2:] = np.nan
    second, _ = fit_adaptive_dualfield(origins, changed, *args[1:], config=config)
    assert audit['fit_rows'] == 2
    assert audit['latest_fit_label_available'] < FIT_CUTOFF
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match='July--August'):
        fit_adaptive_dualfield(origins, *args, cutoff=FIT_CUTOFF + HOUR, config=config)


def test_encode_checks_encoder_and_full_curve_release_before_forward():
    model = AdaptiveHistoricalDualField()
    args = [a.numpy() for a in curves()]
    origins = FIT_CUTOFF + np.arange(4) * HOUR
    args[0][1:] = np.nan
    result = model.encode_matured(origins, *args, as_of=FIT_CUTOFF + 24 * HOUR)
    np.testing.assert_array_equal(result['indices'], [0])
    assert result['reconstruction'].shape == (1, 24)
    assert np.isfinite(result['reconstruction']).all()
    empty = model.encode_matured(origins, *args, as_of=FIT_CUTOFF)
    assert empty['centers'].shape == (0, 6)
    with pytest.raises(ValueError, match='not fitted'):
        model.encode_matured(origins, *args, as_of=FIT_CUTOFF - 1)


def test_checkpoint_exact_replay(tmp_path):
    model = AdaptiveHistoricalDualField()
    args = curves()
    loss = model.reconstruction_loss(*args)
    loss.backward()
    torch.optim.Adam(model.parameters(), lr=.003).step()
    expected = model(*args)
    path = tmp_path / 'adaptive.pt'
    model.save_checkpoint(path, {'test': True})
    restored, audit = AdaptiveHistoricalDualField.load_checkpoint(path)
    assert audit == {'test': True}
    actual = restored(*args)
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)


def test_one_visible_point_is_finite_and_preserves_its_value():
    model = AdaptiveHistoricalDualField()
    price, z, lag = curves()
    mask = torch.zeros_like(price, dtype=torch.bool)
    mask[:, 5] = True
    output = model(price, z, lag, observed_mask=mask)
    assert bool(torch.isfinite(output['reconstruction']).all())
    torch.testing.assert_close(output['reconstruction'][:, 5], price[:, 5] / 300,
                               rtol=1e-6, atol=1e-6)


def test_physical_export_preserves_cancelling_fields_contract(monkeypatch):
    """Regression for TAS-sized opposing fields and a near-zero final price."""
    from forecasting.full_v5.dual_history_memory import _encoded_values

    model = AdaptiveHistoricalDualField()
    trend = torch.full((1, 24), 7.934988975524902, dtype=torch.float32)
    event = torch.full((1, 24), -7.917924404144287, dtype=torch.float32)
    reconstruction = trend + event
    actual = (reconstruction.numpy() * 300).copy()
    origins = np.array([FIT_CUTOFF], dtype=np.int64)
    z, lag = np.zeros((1, 24, 16), dtype=np.float32), np.zeros((1, 24, 3), dtype=np.float32)
    legacy = dict(indices=np.array([0]), trend=trend.numpy() * 300,
                  event=event.numpy() * 300, reconstruction=actual)
    with pytest.raises(ValueError, match='must equal trend plus event'):
        _encoded_values(legacy, 1)

    def controlled_forward(*args, **kwargs):
        return dict(trend=trend, event=event, reconstruction=reconstruction,
                    centers=torch.zeros(1, 6), widths=torch.ones(1, 6),
                    amplitude=torch.zeros(1, 6), gates=torch.ones(1, 6))

    monkeypatch.setattr(model, 'forward', controlled_forward)
    for fixed in (False, True):
        encoded = model.encode_matured(origins, actual, z, lag,
                                       as_of=FIT_CUTOFF + 24 * HOUR, fixed=fixed)
        assert encoded['trend'].dtype == np.float64
        assert encoded['event'].dtype == np.float64
        assert encoded['reconstruction'].dtype == np.float64
        checked = _encoded_values(encoded, 1)
        np.testing.assert_array_equal(checked['trend'] + checked['event'], checked['reconstruction'])
        np.testing.assert_allclose(checked['reconstruction'].mean(1), actual.mean(1), atol=1e-6, rtol=0)
