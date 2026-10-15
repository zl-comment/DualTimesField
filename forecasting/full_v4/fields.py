"""Shared conditional spline background and signed localized event decoder."""
import numpy as np
import torch
from torch import nn
from scipy.interpolate import BSpline


class DualField(nn.Module):
    def __init__(self, latent=32, bases=6, atoms=4):
        super().__init__()
        knots = np.r_[np.zeros(4), np.linspace(0, 1, bases - 2)[1:-1], np.ones(4)]
        basis = BSpline(knots, np.eye(bases), 3)(np.linspace(0, 1, 24))
        self.register_buffer('basis', torch.tensor(basis, dtype=torch.float32))
        self.register_buffer('time', torch.arange(24, dtype=torch.float32))
        self.coefficients = nn.Linear(latent, bases)
        self.events = nn.Linear(latent, 3 * atoms)
        self.atoms = atoms

    def forward(self, z):
        c = self.coefficients(z) @ self.basis.T
        amp, center, width = self.events(z).chunk(3, -1)
        center = 23 * center.sigmoid()
        width = .5 + 5.5 * width.sigmoid()
        d = (amp[..., None] * torch.exp(-.5 * ((self.time - center[..., None]) /
                                              width[..., None]).square())).sum(-2)
        return c, d, amp


class HistoricalEncoder(nn.Module):
    def __init__(self, latent=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1465 + 48, 128), nn.GELU(),
                                 nn.Linear(128, latent), nn.Tanh())

    def forward(self, flattened, curve, *, mask_probability=0.):
        mask = torch.rand_like(curve) >= mask_probability if self.training and mask_probability else torch.ones_like(curve, dtype=torch.bool)
        return self.net(torch.cat((flattened, torch.where(mask, curve, 0.), mask.to(curve.dtype)), -1))
