"""General time-series forecasting baselines on the local datasets.

DLinear (Zeng et al., AAAI 2023), PatchTST (Nie et al., ICLR 2023),
iTransformer (Liu et al., ICLR 2024), and Informer (Zhou et al., AAAI 2021),
re-implemented from their papers and reference code for the 72-hour history
and 24-hour horizon of the AEMO task. All four are used in the "MS" setting of
their code bases (multivariate inputs, the price as the only target) and see
the same inputs as the dual-field model and the RE-Price baselines:

* history: price and demand (``history_values``) with their calendar;
* known future: calendar, PD PASA spare capacity, soft-saturated net load,
  and the DWGM gas-price context of the origin (``baselines._known_future``).

Informer and iTransformer take known inputs natively (decoder inputs and
time-feature embeddings; variate tokens). DLinear and PatchTST are
channel-independent and would otherwise forecast the price from its own
history only, so both add a linear known-input adapter: one linear map from
the flattened known future inputs and demand history to the 24 x 5 outputs,
added to the backbone's forecast.

Every model outputs the 0.05/0.10/0.50/0.90/0.95 quantiles of the
standardized asinh price, the target space of the dual-field model, and is
trained with the pinball loss; the median is the point forecast.
"""

from __future__ import annotations

import math

import torch
from torch import nn

QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)


def _known_future(batch: dict) -> torch.Tensor:
    from .baselines import _known_future as known_future

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


# ----------------------------------------------------------------------------- DLinear


class MovingAverage(nn.Module):
    def __init__(self, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.pool = nn.AvgPool1d(kernel, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L]; replicate the ends so the output keeps length L.
        front = x[:, :1].repeat(1, (self.kernel - 1) // 2)
        back = x[:, -1:].repeat(1, (self.kernel - 1) // 2)
        return self.pool(torch.cat([front, x, back], dim=1)[:, None])[:, 0]


class DLinear(QuantileBaseline):
    """Series decomposition with one linear layer per component (moving average 25)."""

    def __init__(self, input_hours: int, output_hours: int, future_dim: int, history_exog_dim: int,
                 kernel: int = 25):
        super().__init__()
        self.output_hours = output_hours
        self.decompose = MovingAverage(kernel)
        width = output_hours * len(QUANTILES)
        self.seasonal = nn.Linear(input_hours, width)
        self.trend = nn.Linear(input_hours, width)
        # DLinear initializes both maps to the mean of the window.
        for layer in (self.seasonal, self.trend):
            nn.init.constant_(layer.weight, 1.0 / input_hours)
        self.adapter = KnownInputAdapter(future_dim, history_exog_dim, input_hours, output_hours)

    def forward(self, batch: dict) -> torch.Tensor:
        price = batch["history_values"][..., 0]
        trend = self.decompose(price)
        out = self.seasonal(price - trend) + self.trend(trend)
        return out.view(-1, self.output_hours, len(QUANTILES)) + self.adapter(batch)


# ---------------------------------------------------------------------------- PatchTST


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
                 d_ff: int = 256, dropout: float = 0.2):
        super().__init__()
        self.output_hours, self.patch, self.stride = output_hours, patch, stride
        patches = (input_hours - patch) // stride + 1 + 1  # "end" padding adds one patch
        self.revin = RevIN()
        self.embed = nn.Linear(patch, d_model)
        self.position = nn.Parameter(torch.empty(patches, d_model).uniform_(-0.02, 0.02))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(PatchTSTLayer(d_model, heads, d_ff, dropout) for _ in range(layers))
        self.head = nn.Linear(patches * d_model, output_hours * len(QUANTILES))
        self.adapter = KnownInputAdapter(future_dim, history_exog_dim, input_hours, output_hours)

    def forward(self, batch: dict) -> torch.Tensor:
        price, stats = self.revin.normalize(batch["history_values"][..., 0])
        price = torch.cat([price, price[:, -1:].repeat(1, self.stride)], dim=1)
        x = self.embed(price.unfold(1, self.patch, self.stride)) + self.position
        x, scores = self.dropout(x), None
        for layer in self.layers:
            x, scores = layer(x, scores)
        out = self.head(x.flatten(1)).view(-1, self.output_hours, len(QUANTILES))
        return self.revin.denormalize(out, stats) + self.adapter(batch)


# ------------------------------------------------------------------------ iTransformer


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


# ---------------------------------------------------------------------------- Informer


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


def build(kind: str, example: dict) -> nn.Module:
    """Builds a model from one collated example batch."""
    input_hours = example["history_values"].shape[1]
    output_hours = example["future_calendar"].shape[1]
    future_dim = _known_future(example).shape[-1]
    history_exog_dim = example["history_values"].shape[-1] - 1
    common = dict(input_hours=input_hours, output_hours=output_hours, future_dim=future_dim,
                  history_exog_dim=history_exog_dim)
    if kind == "dlinear":
        return DLinear(**common)
    if kind == "patchtst":
        return PatchTST(**common)
    if kind == "itransformer":
        return ITransformer(**common)
    if kind == "informer":
        context = example.get("origin_context")
        return Informer(**common, calendar_dim=example["future_calendar"].shape[-1],
                        context_dim=0 if context is None else context.shape[-1])
    raise ValueError(kind)


# Chosen from {1e-4, 5e-4, 1e-3} by the lowest validation pinball loss on raw
# NSW1 prices, seed 2026, then fixed for every region, seed, and price treatment.
LEARNING_RATES = {"dlinear": 1e-3, "patchtst": 1e-4, "itransformer": 1e-3, "informer": 1e-3}
