"""Sparse candidate selection with a genuine null option, independently per field."""
import torch
from torch import nn


def sparsemax(logits, dim=-1):
    z = logits - logits.max(dim=dim, keepdim=True).values
    ordered = z.sort(dim=dim, descending=True).values
    k = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.ndim
    shape[dim] = -1
    k = k.reshape(shape)
    cumulative = ordered.cumsum(dim)
    support = 1 + k * ordered > cumulative
    size = support.sum(dim=dim, keepdim=True).clamp(min=1)
    threshold = (cumulative.gather(dim, size - 1) - 1) / size
    return (z - threshold).clamp(min=0)


class OptionalHistory(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.query = nn.Linear(width, 16)
        self.key = nn.Linear(26, 16)
        # query/key differences, historical deviations, age and reconstruction quality
        self.score = nn.Sequential(nn.Linear(16 + 4, 32), nn.GELU(), nn.Linear(32, 2))
        self.null = nn.Sequential(nn.Linear(width + 6, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, tokens, candidate_future, c0, d0, hc, hd, quality, ages, valid,
                *, forced_history=False, force_null=False):
        b, k, h = hc.shape
        valid = valid[:, :, None].expand(b, k, h)
        difference = (self.query(tokens)[:, None] - self.key(candidate_future)).square()
        misc = torch.stack((hc - c0[:, None], hd - d0[:, None],
                            quality[:, :, None].expand(b, k, h), ages[:, :, None].expand(b, k, h)), -1)
        logits = self.score(torch.cat((difference, misc), -1)) # B,K,H,2
        count = valid.sum(1).clamp(min=1)
        def moments(v):
            mean = torch.where(valid, v, 0.).sum(1) / count
            variance = torch.where(valid, (v - mean[:, None]).square(), 0.).sum(1) / count
            return mean, torch.sqrt(variance + 1e-8)
        cm, cs = moments(hc - c0[:, None]); dm, ds = moments(hd - d0[:, None])
        qm = torch.where(valid, quality[:, :, None], 0.).sum(1) / count
        summary = torch.stack((cm, cs, dm, ds, qm, count / k), -1)
        null = self.null(torch.cat((tokens, summary), -1))[:, None]
        if forced_history:
            null = torch.where(valid.any(1)[:, None, :, None], torch.full_like(null, -1e6), null)
        logits = torch.where(valid[..., None], logits, torch.full_like(logits, -1e6))
        weights = sparsemax(torch.cat((logits, null), 1), dim=1)
        if force_null:
            weights = torch.zeros_like(weights)
            weights[:, -1] = 1.
        real = weights[:, :-1]
        dc = (real[..., 0] * (hc - c0[:, None])).sum(1)
        dd = (real[..., 1] * (hd - d0[:, None])).sum(1)
        return dc, dd, 1 - weights[:, -1, :, 0], 1 - weights[:, -1, :, 1], weights
