"""Neural front end: historical reconstruction is separate from unknown targets."""
import torch
from torch import nn
from .fields import DualField, HistoricalEncoder
from .evidence import EvidenceEncoder
from .retrieval import OptionalHistory


class Frontend(nn.Module):
    def __init__(self, mode='full', width=64, dropout=.1, memory_dropout=.2):
        super().__init__()
        if mode not in ('full', 'current', 'forced'):
            raise ValueError(mode)
        self.mode, self.memory_dropout = mode, memory_dropout
        self.evidence = EvidenceEncoder(width, dropout)
        self.historical = HistoricalEncoder()
        self.fields = DualField()
        self.retrieval = OptionalHistory(width)

    def reconstruct(self, flat, curve, *, masked=False):
        z = self.historical(flat, curve, mask_probability=.2 if masked else 0.)
        return self.fields(z)

    def forward(self, batch, *, force_null=False):
        z, tokens = self.evidence(batch['history'], batch['future'])
        c0, d0, amp = self.fields(z)
        # All curves are centered around each origin's recent level, in fold sigma.
        zero = torch.zeros_like(c0)
        dc, dd, gc, gd = zero, zero, zero, zero
        regular = c0.diff(n=2, dim=-1).square().mean() * .01 + amp.abs().mean() * .001
        reconstruction = zero.sum()
        weights = torch.zeros((len(c0), 1, 24, 2), device=c0.device)
        weights[:, -1] = 1
        if self.mode != 'current' and not force_null:
            valid = batch['valid'].clone()
            if self.training and self.memory_dropout:
                valid &= (torch.rand(len(c0), 1, device=c0.device) >= self.memory_dropout)
            shape = batch['candidate_curve'].shape
            hc, hd, ha = self.reconstruct(batch['candidate_flat'].flatten(0, 1),
                                          batch['candidate_curve'].flatten(0, 1))
            hc, hd = hc.reshape(shape), hd.reshape(shape)
            error = (hc + hd - batch['candidate_curve']).abs().mean(-1)
            denom = valid.sum().clamp(min=1)
            reconstruction = (error * valid).sum() / denom
            regular = regular + .01 * (hc.diff(n=2, dim=-1).square().mean(-1) * valid).sum() / denom
            regular = regular + .001 * (ha.reshape(*shape[:2], -1).abs().mean(-1) * valid).sum() / denom
            dc, dd, gc, gd, weights = self.retrieval(tokens, batch['candidate_future'], c0, d0,
                hc, hd, error, batch['ages'], valid, forced_history=self.mode == 'forced')
            regular = regular + .001 * weights.diff(dim=2).abs().mean()
        return dict(P=c0 + d0 + dc + dd, P0=c0 + d0, C0=c0, D0=d0,
                    delta_C=dc, delta_D=dd, gC=gc, gD=gd, weights=weights,
                    regularizer=regular, reconstruction=reconstruction)


def forecast_loss(result, centered_target):
    return (.5 * (result['P'] - centered_target).abs().mean()
            + .1 * (result['P0'] - centered_target).abs().mean()
            + .2 * result['reconstruction'] + result['regularizer'])
