"""Smooth, separately bounded location and tail scales; no new parameters."""
import torch

from ..full_v5_anchor_balance.confirmation_model import AsymmetricScaleModel

LOCATION_LIMIT = 30.0
SCALE_LIMIT = 0.1


class BoundedCalibrationModel(AsymmetricScaleModel):
    def calibrate_anchor(self, baseline, features):
        raw = self.calibration(features)
        # Retain the original slope at identity, then smoothly saturate at 1/5
        # of the original output limits. The objective and all weights stay the same.
        location = LOCATION_LIMIT * torch.tanh(150 * raw[..., 0].tanh() / LOCATION_LIMIT)
        lower = 1 + SCALE_LIMIT * torch.tanh(.5 * raw[..., 1].tanh() / SCALE_LIMIT)
        upper = 1 + SCALE_LIMIT * torch.tanh(.5 * raw[..., 2].tanh() / SCALE_LIMIT)
        scale = torch.stack((lower, lower, torch.ones_like(lower), upper, upper), dim=-1)
        anchor = baseline[..., 2:3] + location[..., None] + scale * (baseline - baseline[..., 2:3])
        return anchor, location


def build_model(variant, seed=2026):
    if variant not in ('asymmetric_scale', 'bounded_calibration'):
        raise ValueError('Unknown robustness model')
    torch.manual_seed(seed)
    cls = AsymmetricScaleModel if variant == 'asymmetric_scale' else BoundedCalibrationModel
    return cls()
