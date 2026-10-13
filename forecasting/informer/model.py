"""本地实现。来源与任务适配见本目录 README.md。"""
from __future__ import annotations

import math
import torch
from torch import nn

from ..common.neural import QUANTILES, QuantileBaseline, KnownInputAdapter, _known_future, pinball


class ProbAttention(nn.Module):
    """ProbSparse self-attention: only the top-u queries by sparsity measure attend."""

    def __init__(self, mask: bool, factor: int = 5):
        super().__init__()
        self.mask, self.factor = mask, factor

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q, k, v: [B, H, L, D]
        batch, heads, length_q, dim = q.shape
        length_k = k.shape[2]
        sample_k = min(self.factor * math.ceil(math.log(length_k)), length_k)
        top = min(self.factor * math.ceil(math.log(length_q)), length_q)
        index_sample = torch.randint(length_k, (length_q, sample_k), device=q.device)
        k_sample = k[:, :, index_sample, :]  # [B, H, Lq, sample, D]
        qk_sample = (q.unsqueeze(-2) @ k_sample.transpose(-2, -1)).squeeze(-2)
        sparsity = qk_sample.max(-1)[0] - qk_sample.sum(-1) / length_k
        chosen = sparsity.topk(top, sorted=False)[1]  # [B, H, top]
        b = torch.arange(batch, device=q.device)[:, None, None]
        h = torch.arange(heads, device=q.device)[None, :, None]
        scores = q[b, h, chosen] @ k.transpose(-2, -1) / math.sqrt(dim)
        if self.mask:
            context = v.cumsum(dim=-2)
            future = torch.ones(length_q, length_k, dtype=torch.bool, device=q.device).triu(1)
            scores = scores.masked_fill(future[chosen], -torch.inf)
        else:
            context = v.mean(dim=-2, keepdim=True).expand(batch, heads, length_q, dim).clone()
        context[b, h, chosen] = torch.softmax(scores, dim=-1) @ v
        return context


class FullAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1) @ v


class AttentionLayer(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, heads: int, mix: bool = False):
        super().__init__()
        self.attention, self.heads, self.mix = attention, heads, mix
        self.query, self.key, self.value = (nn.Linear(d_model, d_model) for _ in range(3))
        self.out = nn.Linear(d_model, d_model)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        batch, length, _ = queries.shape
        split = lambda x: x.view(batch, x.shape[1], self.heads, -1).transpose(1, 2)
        context = self.attention(split(self.query(queries)), split(self.key(keys)), split(self.value(values)))
        # context: [B, H, L, D]. Informer's "mix" reshapes without swapping heads and time back.
        context = context.reshape(batch, length, -1) if self.mix else context.transpose(1, 2).reshape(batch, length, -1)
        return self.out(context)


class DataEmbedding(nn.Module):
    """Token (circular convolution), sinusoidal position, and time-feature embeddings."""

    def __init__(self, c_in: int, mark_dim: int, d_model: int, dropout: float, max_length: int = 512):
        super().__init__()
        self.token = nn.Conv1d(c_in, d_model, 3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.token.weight, mode="fan_in", nonlinearity="leaky_relu")
        position = torch.arange(max_length)[:, None]
        frequency = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        table = torch.zeros(max_length, d_model)
        table[:, 0::2], table[:, 1::2] = torch.sin(position * frequency), torch.cos(position * frequency)
        self.register_buffer("position", table, persistent=False)
        self.time = nn.Linear(mark_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
        token = self.token(x.transpose(1, 2)).transpose(1, 2)
        return self.dropout(token + self.position[: x.shape[1]] + self.time(marks))


class DistilConv(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=2, padding_mode="circular")
        self.norm = nn.BatchNorm1d(d_model)
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(nn.functional.elu(self.norm(self.conv(x.transpose(1, 2))))).transpose(1, 2)


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float, factor: int):
        super().__init__()
        self.attention = AttentionLayer(ProbAttention(False, factor), d_model, heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attention(x, x, x)))
        return self.norm2(x + self.dropout(self.ff(x)))


class InformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float, factor: int):
        super().__init__()
        self.self_attention = AttentionLayer(ProbAttention(True, factor), d_model, heads, mix=True)
        self.cross_attention = AttentionLayer(FullAttention(), d_model, heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x)))
        x = self.norm2(x + self.dropout(self.cross_attention(x, memory, memory)))
        return self.norm3(x + self.dropout(self.ff(x)))


class Informer(QuantileBaseline):
    """Informer defaults: d_model 512, 8 heads, 2 encoder layers with distilling,
    1 decoder layer, d_ff 2048, ProbSparse factor 5, dropout 0.05; label length is
    half the input (36 of 72 hours, as 48 of 96 in the original setting).

    Encoder inputs: price, demand, and the gas-price context, with the calendar as
    time features. Decoder inputs: the last 36 history hours followed by the 24
    forecast hours, whose price and demand are zero as in Informer; the known
    future exogenous series and a forecast-hour flag are extra decoder channels,
    zero over the label hours.
    """

    def __init__(self, input_hours: int, output_hours: int, future_dim: int, history_exog_dim: int,
                 calendar_dim: int, context_dim: int, d_model: int = 512, heads: int = 8,
                 encoder_layers: int = 2, decoder_layers: int = 1, d_ff: int = 2048,
                 dropout: float = 0.05, factor: int = 5):
        super().__init__()
        self.output_hours, self.label_hours = output_hours, input_hours // 2
        self.calendar_dim = calendar_dim
        history_dim = 1 + history_exog_dim
        exog_dim = future_dim - calendar_dim
        self.encoder_embedding = DataEmbedding(history_dim + context_dim, calendar_dim, d_model, dropout)
        self.decoder_embedding = DataEmbedding(history_dim + exog_dim + 1, calendar_dim, d_model, dropout)
        self.encoder_layers = nn.ModuleList(
            InformerEncoderLayer(d_model, heads, d_ff, dropout, factor) for _ in range(encoder_layers)
        )
        self.distil = nn.ModuleList(DistilConv(d_model) for _ in range(encoder_layers - 1))
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_layers = nn.ModuleList(
            InformerDecoderLayer(d_model, heads, d_ff, dropout, factor) for _ in range(decoder_layers)
        )
        self.decoder_norm = nn.LayerNorm(d_model)
        self.project = nn.Linear(d_model, len(QUANTILES))

    def forward(self, batch: dict) -> torch.Tensor:
        values = batch["history_values"]
        steps = values.shape[1]
        context = batch.get("origin_context")
        encoder_inputs = values if context is None else torch.cat(
            [values, context[:, None, :].expand(-1, steps, -1)], dim=-1
        )
        x = self.encoder_embedding(encoder_inputs, batch["history_calendar"])
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if i < len(self.distil):
                x = self.distil[i](x)
        memory = self.encoder_norm(x)

        future = _known_future(batch)
        future_calendar, exog = future[..., : self.calendar_dim], future[..., self.calendar_dim:]
        label = values[:, -self.label_hours:]
        decoder_inputs = torch.cat([
            torch.cat([label, exog.new_zeros(label.shape[0], self.label_hours, exog.shape[-1] + 1)], dim=-1),
            torch.cat([exog.new_zeros(*exog.shape[:2], values.shape[-1]), exog,
                       exog.new_ones(*exog.shape[:2], 1)], dim=-1),
        ], dim=1)
        decoder_marks = torch.cat([batch["history_calendar"][:, -self.label_hours:], future_calendar], dim=1)
        y = self.decoder_embedding(decoder_inputs, decoder_marks)
        for layer in self.decoder_layers:
            y = layer(y, memory)
        return self.project(self.decoder_norm(y))[:, -self.output_hours:]
