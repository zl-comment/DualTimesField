"""本地实现。来源与任务适配见本目录 README.md。"""
from __future__ import annotations

import math
import torch
from torch import nn

from ..common.neural import QUANTILES, QuantileBaseline, KnownInputAdapter, _known_future, pinball


class ITransformerLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attention(x, x, x, need_weights=False)[0]))
        return self.norm2(x + self.dropout(self.ff(x)))


class ITransformer(QuantileBaseline):
    """Variates as tokens. History series, history calendar, and each known future
    series (24 hours) are embedded as separate tokens; the price token is projected."""

    def __init__(self, input_hours: int, output_hours: int, future_dim: int, history_exog_dim: int,
                 d_model: int = 128, heads: int = 8, layers: int = 2, d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.output_hours = output_hours
        self.embed_history = nn.Linear(input_hours, d_model)
        self.embed_future = nn.Linear(output_hours, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(ITransformerLayer(d_model, heads, d_ff, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.project = nn.Linear(d_model, output_hours * len(QUANTILES))

    def forward(self, batch: dict) -> torch.Tensor:
        values = batch["history_values"]
        # Non-stationary normalization of the history series (use_norm in iTransformer).
        mean = values.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(values.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        history = torch.cat([(values - mean) / std, batch["history_calendar"]], dim=-1)
        tokens = torch.cat(
            [self.embed_history(history.transpose(1, 2)), self.embed_future(_known_future(batch).transpose(1, 2))],
            dim=1,
        )
        x = self.dropout(tokens)
        for layer in self.layers:
            x = layer(x)
        out = self.project(self.norm(x)[:, 0]).view(-1, self.output_hours, len(QUANTILES))
        return out * std[:, :, :1] + mean[:, :, :1]
