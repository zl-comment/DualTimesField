"""Shared quantile loss and known-covariate adapter; no fitted preprocessing."""
from __future__ import annotations

import torch
from torch import nn

QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)


def _known_future(batch: dict) -> torch.Tensor:
    from ..baselines import _known_future as known_future

    return known_future(batch)


def pinball(quantile: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor(QUANTILES, device=quantile.device, dtype=quantile.dtype)
    error = target[..., None] - quantile
    return torch.maximum(levels * error, (levels - 1) * error).mean()


class QuantileBaseline(nn.Module):
    """Shared loss and prediction for models whose forward returns [B, H, Q]."""

    def loss(self, batch: dict) -> torch.Tensor:
        return pinball(self(batch), batch["target_price"][..., 0])

    @torch.no_grad()
    def predict(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        quantile = self(batch)
        return quantile[..., QUANTILES.index(0.5)], quantile


class KnownInputAdapter(nn.Module):
    """Linear map from known future inputs and exogenous history to the outputs."""

    def __init__(self, future_dim: int, history_exog_dim: int, input_hours: int, output_hours: int):
        super().__init__()
        self.output_hours = output_hours
        self.linear = nn.Linear(output_hours * future_dim + input_hours * history_exog_dim,
                                output_hours * len(QUANTILES))

    def forward(self, batch: dict) -> torch.Tensor:
        parts = [_known_future(batch).flatten(1), batch["history_values"][..., 1:].flatten(1)]
        return self.linear(torch.cat(parts, dim=1)).view(-1, self.output_hours, len(QUANTILES))
