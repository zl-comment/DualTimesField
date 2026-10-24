"""本地实现。来源与任务适配见本目录 README.md。

论文链接与引用 / References
--------------------------
移动平均分解与线性预测的本地参考实现，增加外生 adapter 与五分位头；未通过官方权重映射等价验证。

BibTeX（可直接复制）: 本目录 CITATIONS.bib；逐项用途与核对来源: source.json。

[zeng2023transformers] DLinear 原论文
Zeng, Ailing; Chen, Muxi; Zhang, Lei; Xu, Qiang (2023). Are Transformers Effective for Time Series
Forecasting?. Proceedings of the AAAI Conference on Artificial Intelligence, 37(9), 11121–11128.
https://doi.org/10.1609/aaai.v37i9.26317
"""
from __future__ import annotations

import torch
from torch import nn

from ..common.neural import QUANTILES, QuantileBaseline, KnownInputAdapter, _known_future, pinball


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
                 kernel: int = 25, decomposition: bool = True, adapter: bool = True):
        super().__init__()
        self.output_hours = output_hours
        self.decompose = MovingAverage(kernel) if decomposition else None
        width = output_hours * len(QUANTILES)
        self.seasonal = nn.Linear(input_hours, width)
        self.trend = nn.Linear(input_hours, width)
        # DLinear initializes both maps to the mean of the window.
        for layer in (self.seasonal, self.trend):
            nn.init.constant_(layer.weight, 1.0 / input_hours)
        self.adapter = KnownInputAdapter(future_dim, history_exog_dim, input_hours, output_hours) if adapter else None

    def forward(self, batch: dict) -> torch.Tensor:
        price = batch["history_values"][..., 0]
        if self.decompose is None:
            out = self.seasonal(price)
        else:
            trend = self.decompose(price)
            out = self.seasonal(price - trend) + self.trend(trend)
        out = out.view(-1, self.output_hours, len(QUANTILES))
        return out if self.adapter is None else out + self.adapter(batch)
