"""Four confirmation variants; leave the original six-arm screen unchanged."""
import torch
from torch import nn

from ..full_v5_joint.model import JointHistoryModel
from .architecture import build_model as build_screen_model

VARIANTS = ('balanced', 'location_only', 'identity_anchor', 'asymmetric_scale')


class AsymmetricScaleModel(JointHistoryModel):
    """One location and two positive scales, sharing the original hidden layer."""

    def __init__(self, width=24):
        super().__init__(width)
        # Build the entire reference first. Replacing the last layer must not
        # change any other parameter or the subsequent random stream.
        old = self.calibration[-1]
        with torch.random.fork_rng(devices=[]):
            head = nn.Linear(width, 3)
            with torch.no_grad():
                head.weight[:2].copy_(old.weight)
                head.bias[:2].copy_(old.bias)
                head.weight[2].copy_(old.weight[1])
                head.bias[2].copy_(old.bias[1])
        self.calibration[-1] = head

    def calibrate_anchor(self, baseline, features):
        raw = self.calibration(features)
        location = 150 * raw[..., 0].tanh()
        lower = 1 + .5 * raw[..., 1].tanh()
        upper = 1 + .5 * raw[..., 2].tanh()
        scale = torch.stack((lower, lower, torch.ones_like(lower), upper, upper), dim=-1)
        anchor = baseline[..., 2:3] + location[..., None] + scale * (baseline - baseline[..., 2:3])
        return anchor, location


def build_model(variant, seed=2026):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown confirmation variant: {variant}')
    if variant != 'asymmetric_scale':
        return build_screen_model(variant, seed)
    torch.manual_seed(seed)
    return AsymmetricScaleModel()
