"""本地实现：所有共享输入展平后的两层 MLP。"""
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
