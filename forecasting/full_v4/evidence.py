"""Sequential observations plus nonlinear grouped future-condition evidence."""
import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)
        self.padding = 2 * dilation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(F.gelu(self.conv(F.pad(x, (self.padding, 0)))))


class EvidenceEncoder(nn.Module):
    def __init__(self, width=64, dropout=.1):
        super().__init__()
        self.history = nn.Sequential(nn.Conv1d(12, width, 1),
            *(CausalBlock(width, d, dropout) for d in (1, 2, 4)))
        # Future vector: calendar10, gas1, future11, netload1, shortfalls3.
        self.groups = ((13, 22), (11, 14, 23, 24, 25), (15, 17, 18, 19, 20),
                       (12,), tuple(range(11)), (16, 21))
        self.group_layers = nn.ModuleList(nn.Sequential(nn.Linear(len(g), 16), nn.GELU())
                                         for g in self.groups)
        self.future = nn.Sequential(nn.Linear(96, width), nn.GELU(), nn.Linear(width, width))
        self.attention = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.latent = nn.Sequential(nn.Linear(24 * width, 128), nn.GELU(), nn.Linear(128, 32), nn.Tanh())

    def forward(self, history, future):
        hist = self.history(history.transpose(1, 2)).transpose(1, 2)
        query = self.future(torch.cat([layer(future[..., list(g)])
                                      for g, layer in zip(self.groups, self.group_layers)], -1))
        attended, _ = self.attention(query, hist, hist, need_weights=False)
        tokens = self.norm(query + attended)
        return self.latent(tokens.flatten(1)), tokens
