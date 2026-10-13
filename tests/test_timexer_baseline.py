"""Official TimeXer parity and canonical-input integration."""
from types import SimpleNamespace

import torch

from forecasting.aligned_inputs import INPUT_KEYS
from forecasting.timexer_baseline import build_timexer
from forecasting.vendor.timexer.model import Model


def example():
    torch.manual_seed(123)
    shapes = [(72, 2), (72, 10), (24, 10), (1,), (24, 11), (24, 1), (24, 3)]
    return {k: torch.randn(2, *s) for k, s in zip(INPUT_KEYS, shapes)}


def test_historical_path_matches_official_point_model_with_widened_head():
    torch.set_num_threads(2)
    batch = example()
    model = build_timexer(batch, d_model=16, n_heads=4, e_layers=2, d_ff=32,
                          dropout=0., include_future=False).eval()
    config = SimpleNamespace(**model.provenance['dimensions'])
    config.pred_len = 24 * 5
    official = Model(config).eval()
    official.load_state_dict(model.state_dict(), strict=True)
    history = batch['history_values']
    x = torch.cat([history[..., 1:], history[..., :1]], -1)
    expected = official(x, batch['history_calendar'], None, None).reshape(2, 24, 5)
    torch.testing.assert_close(model(batch), expected, rtol=0, atol=0)


def test_all_shared_input_blocks_reach_the_quantile_forecast():
    batch = {k: v.requires_grad_() for k, v in example().items()}
    model = build_timexer(batch, d_model=16, n_heads=4, e_layers=2, d_ff=32, dropout=0.)
    out = model(batch)
    assert out.shape == (2, 24, 5) and torch.isfinite(out).all()
    (out * torch.randn_like(out)).sum().backward()
    for key, value in batch.items():
        assert value.grad is not None and torch.isfinite(value.grad).all(), key
        assert value.grad.abs().sum() > 0, key
    assert model.future_embedding.value_embedding.weight.grad.abs().sum() > 0


def test_constant_price_and_empty_optional_exogenous_blocks_are_finite():
    batch = example()
    batch['history_values'][:] = 0
    for key in ('origin_context', 'future_exogenous', 'ctf_exogenous', 'quantile_exogenous'):
        batch[key] = batch[key][..., :0]
    model = build_timexer(batch, d_model=16, n_heads=4, e_layers=1, d_ff=32, dropout=0.)
    assert torch.isfinite(model(batch)).all()
