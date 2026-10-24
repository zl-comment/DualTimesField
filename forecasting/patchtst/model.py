"""本地实现。来源与任务适配见本目录 README.md。

论文链接与引用 / References
--------------------------
PatchTST/RevIN 的本地参考实现，增加外生变量 adapter 和五分位输出；未通过与官方权重映射的等价验证。

BibTeX（可直接复制）: 本目录 CITATIONS.bib；逐项用途与核对来源: source.json。

[nie2023time] PatchTST 原论文
Nie, Yuqi; Nguyen, Nam H.; Sinthong, Phanwadee; Kalagnanam, Jayant (2023). A Time Series is Worth 64
Words: Long-term Forecasting with Transformers. ICLR.
https://arxiv.org/abs/2211.14730

[kim2022reversible] RevIN 组件原论文
Kim, Taesung; Kim, Jinhee; Tae, Yunwon; Park, Cheonbok; Choi, Jang-Ho; Choo, Jaegul (2022).
Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift.
ICLR.
https://openreview.net/forum?id=cGDAkQo1C0p
"""
from __future__ import annotations

import math
import torch
from torch import nn

from ..common.neural import QUANTILES, QuantileBaseline, KnownInputAdapter, _known_future, pinball


class RevIN(nn.Module):
    """Reversible instance normalization with a learnable affine map."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        mean = x.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        return (x - mean) / std * self.weight + self.bias, (mean, std)

    def denormalize(self, y: torch.Tensor, stats: tuple) -> torch.Tensor:
        mean, std = stats
        y = (y - self.bias) / (self.weight + self.eps * self.eps)
        return y * std[..., None] + mean[..., None]


class PatchTSTLayer(nn.Module):
    """Post-norm Transformer layer with BatchNorm and residual attention scores."""

    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(0.0)
        self.dropout = nn.Dropout(dropout)
        self.norm_attention = nn.BatchNorm1d(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm_ff = nn.BatchNorm1d(d_model)

    def _batch_norm(self, norm: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor, previous: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if previous is not None:
            scores = scores + previous
        attention = self.attention_dropout(torch.softmax(scores, dim=-1))
        context = (attention @ v).transpose(1, 2).reshape(batch, length, width)
        x = self._batch_norm(self.norm_attention, x + self.dropout(self.out(context)))
        x = self._batch_norm(self.norm_ff, x + self.dropout(self.ff(x)))
        return x, scores


class PatchTST(QuantileBaseline):
    """PatchTST/42 defaults: patch 16, stride 8, d_model 128, 16 heads, 3 layers."""

    def __init__(self, input_hours: int, output_hours: int, future_dim: int, history_exog_dim: int,
                 patch: int = 16, stride: int = 8, d_model: int = 128, heads: int = 16, layers: int = 3,
                 d_ff: int = 256, dropout: float = 0.2, adapter: bool = True):
        super().__init__()
        self.output_hours, self.patch, self.stride = output_hours, patch, stride
        patches = (input_hours - patch) // stride + 1 + 1  # "end" padding adds one patch
        self.revin = RevIN()
        self.embed = nn.Linear(patch, d_model)
        self.position = nn.Parameter(torch.empty(patches, d_model).uniform_(-0.02, 0.02))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(PatchTSTLayer(d_model, heads, d_ff, dropout) for _ in range(layers))
        self.head = nn.Linear(patches * d_model, output_hours * len(QUANTILES))
        self.adapter = KnownInputAdapter(future_dim, history_exog_dim, input_hours, output_hours) if adapter else None

    def forward(self, batch: dict) -> torch.Tensor:
        price, stats = self.revin.normalize(batch["history_values"][..., 0])
        price = torch.cat([price, price[:, -1:].repeat(1, self.stride)], dim=1)
        x = self.embed(price.unfold(1, self.patch, self.stride)) + self.position
        x, scores = self.dropout(x), None
        for layer in self.layers:
            x, scores = layer(x, scores)
        out = self.revin.denormalize(self.head(x.flatten(1)).view(-1, self.output_hours, len(QUANTILES)), stats)
        return out if self.adapter is None else out + self.adapter(batch)
