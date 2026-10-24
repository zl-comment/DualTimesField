"""本地实现：所有共享输入展平后的两层 MLP。

论文链接与引用 / References
--------------------------
本地通用 MLP 对照，没有对应本结构的单一原论文。所列文献仅为反向传播的经典背景，不是该电价模型或其两层/GELU/分位头设计的出处。

BibTeX（可直接复制）: 本目录 CITATIONS.bib；逐项用途与核对来源: source.json。

[rumelhart1986learning] 前馈网络反向传播的经典背景文献
Rumelhart, David E.; Hinton, Geoffrey E.; Williams, Ronald J. (1986). Learning representations by
back-propagating errors. Nature, 323, 533–536.
https://doi.org/10.1038/323533a0
"""
import numpy as np
import torch
from torch import nn
from ..aligned_inputs import INPUT_KEYS
from ..aligned_scoring import QUANTILE_LEVELS


class PlainMLP(nn.Module):
    """Independent all-input control, with no dual-field implementation dependency."""
    def __init__(self, example, horizon, hidden=64):
        super().__init__()
        width = sum(int(np.prod(example[k].shape[1:])) for k in INPUT_KEYS)
        self.horizon = horizon
        self.network = nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                                     nn.Linear(hidden, horizon * len(QUANTILE_LEVELS)))

    def forward(self, batch):
        flat = torch.cat([batch[k].flatten(1) for k in INPUT_KEYS], dim=1)
        return self.network(flat).reshape(-1, self.horizon, len(QUANTILE_LEVELS))
