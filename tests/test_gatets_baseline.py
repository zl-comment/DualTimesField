"""Integration guarantees for the pinned official GateTS adaptation."""
import hashlib
from pathlib import Path

import pytest
import torch

from forecasting.gatets.model import build_gatets
from forecasting.gatets.vendor.new_moe import KroneckerGatingNetwork


def _batch():
    torch.manual_seed(42)
    return {
        "history_values": torch.randn(2, 72, 2),
        "history_calendar": torch.randn(2, 72, 10),
        "future_calendar": torch.randn(2, 24, 10),
        "future_exogenous": torch.randn(2, 24, 11),
        # FairWindows has already appended the quantile shortfalls here.
        "ctf_exogenous": torch.randn(2, 24, 5),
        "origin_context": torch.randn(2, 1),
    }


def test_vendored_source_matches_pinned_official_commit():
    source = Path(__file__).resolve().parents[1] / "forecasting/gatets/vendor/new_moe.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "0e7d9c23e55b3dcf9ee8984e1a21cb5e4b17353d9ee7cd4d5ffbf4c1b0d4bb54"
    )


@pytest.mark.parametrize("kind", ["gatets", "gatets_adapter"])
def test_quantile_forward_backward_and_checkpoint_roundtrip(kind):
    torch.set_num_threads(2)
    batch = _batch()
    model = build_gatets(kind, batch)
    moe = model.backbone.layers[0].moe_decoder
    assert isinstance(moe.gating, KroneckerGatingNetwork)
    assert len(moe.experts) == 16 and moe.top_k == 2
    quantiles = model(batch)
    assert quantiles.shape == (2, 24, 5)
    quantiles.square().mean().backward()
    assert torch.isfinite(quantiles).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert moe.gating.A.grad.abs().sum() > 0
    assert model.backbone.output_layer.weight.grad.abs().sum() > 0
    clone = build_gatets(kind, batch)
    clone.load_state_dict(model.state_dict())
    model.eval()
    clone.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(batch), clone(batch))


def test_information_boundary_and_adapter_sensitivity():
    torch.set_num_threads(2)
    batch = _batch()
    history_only = build_gatets("gatets", batch).eval()
    adapted = build_gatets("gatets_adapter", batch).eval()
    changed = {key: value.clone() for key, value in batch.items()}
    changed["future_exogenous"] += 3
    changed["ctf_exogenous"] -= 2
    changed["history_values"][..., 1] += 5
    with torch.no_grad():
        torch.testing.assert_close(history_only(batch), history_only(changed))
        assert not torch.allclose(adapted(batch), adapted(changed))
        # Labels never enter either model; they can be omitted or altered.
        poisoned = dict(batch, target_price=torch.full((2, 24, 1), float("nan")),
                        target_quantile=torch.full((2, 24, 1), float("nan")))
        torch.testing.assert_close(adapted(batch), adapted(poisoned))


def test_unknown_kind_fails_explicitly():
    with pytest.raises(ValueError, match="Unknown GateTS"):
        build_gatets("gatets_typo", _batch())
