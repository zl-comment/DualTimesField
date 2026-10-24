"""本地实现。来源与任务适配见本目录 README.md。

论文链接与引用 / References
--------------------------
本地无趋势分解的线性预测对照，含已知外生 adapter；引用说明其设计背景，不将本地完整模型标为单独论文的官方实现。

BibTeX（可直接复制）: 本目录 CITATIONS.bib；逐项用途与核对来源: source.json。

[zeng2023transformers] Linear/DLinear 对照设计的相关论文
Zeng, Ailing; Chen, Muxi; Zhang, Lei; Xu, Qiang (2023). Are Transformers Effective for Time Series
Forecasting?. Proceedings of the AAAI Conference on Artificial Intelligence, 37(9), 11121–11128.
https://doi.org/10.1609/aaai.v37i9.26317
"""
from __future__ import annotations

import torch
from torch import nn

from ..common.neural import QUANTILES, QuantileBaseline, KnownInputAdapter, _known_future, pinball


from ..dlinear.model import DLinear


class KnownLinear(QuantileBaseline):
    """Attribution variant: the known-input adapter alone, without price history."""

    def __init__(self, input_hours: int, output_hours: int, future_dim: int, history_exog_dim: int):
        super().__init__()
        self.adapter = KnownInputAdapter(future_dim, history_exog_dim, input_hours, output_hours)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.adapter(batch)


class LinearMSE(QuantileBaseline):
    """Attribution variant: the "linear" model with a separate point output trained
    by mean squared error in the target space, as the dual-field point head is;
    the quantiles keep the pinball loss."""

    def __init__(self, input_hours: int, output_hours: int, future_dim: int, history_exog_dim: int):
        super().__init__()
        self.output_hours = output_hours
        self.history = nn.Linear(input_hours, output_hours * (len(QUANTILES) + 1))
        nn.init.constant_(self.history.weight, 1.0 / input_hours)
        self.adapter = nn.Linear(output_hours * future_dim + input_hours * history_exog_dim,
                                 output_hours * (len(QUANTILES) + 1))

    def forward(self, batch: dict) -> torch.Tensor:
        known = torch.cat([_known_future(batch).flatten(1), batch["history_values"][..., 1:].flatten(1)], dim=1)
        out = self.history(batch["history_values"][..., 0]) + self.adapter(known)
        return out.view(-1, self.output_hours, len(QUANTILES) + 1)

    def loss(self, batch: dict) -> torch.Tensor:
        out, target = self(batch), batch["target_price"][..., 0]
        return pinball(out[..., 1:], target) + nn.functional.mse_loss(out[..., 0], target)

    @torch.no_grad()
    def predict(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        out = self(batch)
        return out[..., 0], out[..., 1:]


class Linear(DLinear):
    """Local non-decomposing DLinear control; keeps identical parameter layout."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs, decomposition=False)
