"""Official TimeXer components adapted to the shared AEMO quantile task.

论文链接与引用 / References
--------------------------
固定官方主干，增加已知未来变量 token 与五分位头；作者列表以 NeurIPS 正式出版页为准。

BibTeX（可直接复制）: 本目录 CITATIONS.bib；逐项用途与核对来源: source.json。

[wang2024timexer] TimeXer 原论文
Wang, Yuxuan; Wu, Haixu; Dong, Jiaxiang; Qin, Guo; Zhang, Haoran; Liu, Yong; Qiu, Yunzhong; Wang,
Jianmin; Long, Mingsheng (2024). TimeXer: Empowering Transformers for Time Series Forecasting with
Exogenous Variables. Advances in Neural Information Processing Systems 37, 469–498.
https://proceedings.neurips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from ..aligned_scoring import QUANTILE_LEVELS
from .vendor.model import Model
from .vendor.layers import DataEmbedding_inverted


class TimeXerQuantileForecaster(Model):
    """Keep endogenous/global-token attention; supply available future tokens.

    Historical exogenous variables and calendar retain the official inverted
    embedding. Each future-known variable receives a separate horizon-length
    projection into that same cross-attention context. No future price labels
    enter either embedding. The official output head is widened to H * Q.
    """

    def __init__(self, input_hours, output_hours, history_dim, *, patch_len=24,
                 d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.1,
                 use_norm=True, include_future=True):
        if patch_len < 1 or input_hours % patch_len:
            raise ValueError('TimeXer history must be divisible by its positive patch length')
        if d_model % n_heads or d_model % 2:
            raise ValueError('TimeXer embedding width must be even and divisible by head count')
        config = SimpleNamespace(task_name='short_term_forecast', features='MS',
            seq_len=input_hours, pred_len=output_hours, use_norm=use_norm,
            patch_len=patch_len, enc_in=history_dim, d_model=d_model,
            n_heads=n_heads, e_layers=e_layers, d_ff=d_ff, dropout=dropout,
            embed='timeF', freq='h', factor=5, activation='gelu')
        super().__init__(config)
        self.head.linear = nn.Linear(self.head_nf, output_hours * len(QUANTILE_LEVELS))
        self.future_embedding = (DataEmbedding_inverted(output_hours, d_model,
                                 dropout=dropout) if include_future else None)
        source = json.loads((Path(__file__).parent / 'vendor/source_manifest.json').read_text())
        self.provenance = {
            'repository': source['repository'], 'commit': source['commit'],
            'adaptation': 'five_quantile_head_and_future_known_cross_attention_tokens',
            'pretrained': False, 'include_future': include_future,
            'dimensions': vars(config), 'unused_input_blocks': [] if include_future else [
                'future_calendar', 'future_exogenous', 'ctf_exogenous',
                'quantile_exogenous', 'origin_context'],
        }

    def forward(self, batch):
        history = batch['history_values']
        # The common schema has price first; official TimeXer MS has it last.
        x = torch.cat([history[..., 1:], history[..., :1]], dim=-1)
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            centered = x - means
            stdev = torch.sqrt(centered.var(1, keepdim=True, unbiased=False) + 1e-5)
            x = centered / stdev
        endogenous, n_vars = self.en_embedding(x[..., -1:].permute(0, 2, 1))
        exogenous = self.ex_embedding(x[..., :-1], batch['history_calendar'])
        if self.future_embedding is not None:
            future = torch.cat([batch[k] for k in (
                'future_calendar', 'future_exogenous', 'ctf_exogenous', 'quantile_exogenous')]
                + [batch['origin_context'][:, None, :].expand(-1, self.pred_len, -1)], dim=-1)
            exogenous = torch.cat([exogenous, self.future_embedding(future, None)], dim=1)
        encoded = self.encoder(endogenous, exogenous)
        encoded = encoded.reshape(-1, n_vars, encoded.shape[-2], encoded.shape[-1])
        out = self.head(encoded.permute(0, 1, 3, 2)).reshape(
            history.shape[0], self.pred_len, len(QUANTILE_LEVELS))
        if self.use_norm:
            out = out * stdev[..., -1:] + means[..., -1:]
        return out


def build_timexer(example, **parameters):
    return TimeXerQuantileForecaster(
        input_hours=example['history_values'].shape[1],
        output_hours=example['future_calendar'].shape[1],
        history_dim=example['history_values'].shape[-1], **parameters)
