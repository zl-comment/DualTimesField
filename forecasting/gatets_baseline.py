"""Pinned official GateTS with explicit AEMO quantile/output adaptations.

``gatets`` uses price history only. ``gatets_adapter`` additionally receives
all known inputs through the existing linear baseline adapter. Neither is a
reproduction of the paper's original dataset scores. See vendor provenance.
"""
from __future__ import annotations

import torch
from torch import nn

from .general_baselines import KnownInputAdapter, _known_future
from .vendor.gatets.new_moe import MoETransformerForecaster

UPSTREAM_REPOSITORY = "https://github.com/ruiite/MOE_time_series"
UPSTREAM_COMMIT = "a3b821fff6fda3bde8124a79564a8d33df9732f1"
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)
OFFICIAL_DEFAULTS = dict(
    d_model=48,
    n_heads=4,
    num_layers=1,
    ffn_dim=64,
    num_experts=16,
    top_k=2,
    use_decoder_expert=True,
    expert_n_heads=2,
    dropout=0.1,
    gating="kronecker",
)


class GateTSQuantileForecaster(nn.Module):
    """Unmodified official backbone plus a quantile-sized output projection."""

    def __init__(self, input_hours: int, output_hours: int, future_dim: int = 0,
                 history_exog_dim: int = 0, *, adapter: bool = False):
        super().__init__()
        self.input_hours = input_hours
        self.output_hours = output_hours
        self.backbone = MoETransformerForecaster(
            input_size=input_hours,
            target_size=output_hours * len(QUANTILES),
            **OFFICIAL_DEFAULTS,
        )
        self.adapter = (KnownInputAdapter(future_dim, history_exog_dim,
                                         input_hours, output_hours)
                        if adapter else None)
        self.provenance = {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "license": "CC-BY-4.0",
            "adaptation": "AEMO standardized raw-price quantile output",
            "inputs": "price history and known-input adapter" if adapter else "price history only",
            "official_defaults": dict(OFFICIAL_DEFAULTS),
            "expert_compute": "all experts evaluated; top-k output mask",
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        history = batch["history_values"]
        if history.ndim != 3 or history.shape[1] != self.input_hours:
            raise ValueError("GateTS history must have shape [batch, input_hours, variables]")
        forecast = self.backbone(history[..., :1]).reshape(
            history.shape[0], self.output_hours, len(QUANTILES)
        )
        if self.adapter is not None:
            forecast = forecast + self.adapter(batch)
        return forecast


def build_gatets(kind: str, example: dict[str, torch.Tensor]) -> nn.Module:
    """Build from a batched SequenceWindows/FairWindows example, without labels."""
    if kind not in {"gatets", "gatets_adapter"}:
        raise ValueError(f"Unknown GateTS baseline kind: {kind}")
    history = example["history_values"]
    calendar = example["future_calendar"]
    if history.ndim != 3 or calendar.ndim != 3:
        raise ValueError("GateTS expects batched history and future calendar tensors")
    use_adapter = kind == "gatets_adapter"
    future_dim = _known_future(example).shape[-1] if use_adapter else 0
    return GateTSQuantileForecaster(
        input_hours=history.shape[1], output_hours=calendar.shape[1],
        future_dim=future_dim, history_exog_dim=history.shape[-1] - 1,
        adapter=use_adapter,
    )
