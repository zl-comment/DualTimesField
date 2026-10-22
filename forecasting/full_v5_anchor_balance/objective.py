"""Explicitly supervise the Anchor while preserving the legacy objective."""
from dataclasses import dataclass
import math

import torch

from ..full_v5_joint.model import JointHistoryModel, pinball, training_loss


@dataclass(frozen=True)
class BalanceWeights:
    anchor: float = .5
    median: float = .5
    identity: float = .01

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in
               (self.anchor, self.median, self.identity)):
            raise ValueError('Balance weights must be finite and nonnegative')


def balance_loss(output, targets, baseline, weights=BalanceWeights(), *,
                 reconstruction_weight=.05, gate_penalty=.001):
    """Return loss and weighted tensor terms; price differences use units of 300.

    Median pinball equals half MAE. Identity regularization penalizes all five
    quantile movements, so it constrains location and interval width together.
    Zero balance weights preserve the exact legacy loss computation.
    """
    legacy = training_loss(output, targets, reconstruction_weight=reconstruction_weight,
                           gate_penalty=gate_penalty)
    zero = legacy.new_zeros(())
    anchor = (weights.anchor * pinball(output['anchor'], targets) / 300
              if weights.anchor else zero)
    median = (weights.median * .5 *
              ((output['anchor'][..., 2] - targets).abs().mean() +
               (output['prediction'][..., 2] - targets).abs().mean()) / 300
              if weights.median else zero)
    identity = (weights.identity * ((output['anchor'] - baseline) / 300).square().mean()
                if weights.identity else zero)
    terms = dict(full_pinball=pinball(output['prediction'], targets) / 300,
                 reconstruction=reconstruction_weight * output['reconstruction_loss'],
                 gate_penalty=gate_penalty * output['gates'].mean(),
                 anchor_pinball=anchor, median_pinball=median, identity=identity)
    return legacy + anchor + median + identity, terms


class _IdentityAnchorModel(JointHistoryModel):
    """Avoid floating cancellation in B50 + (Bq - B50) for exact identity."""

    def forward(self, batch, **controls):
        output = super().forward(batch, **controls)
        output['anchor'] = batch['baseline']
        output['prediction'] = batch['baseline'] + output['correction'][..., None]
        return output


def build_model(variant, seed=2026):
    """Create matched initial weights; the runner chooses legacy/balanced loss.

    identity_anchor trains the historical model with an exactly unchanged tree
    Anchor. Rebuild using the same variant before loading its state dictionary.
    """
    if variant not in ('legacy', 'balanced', 'identity_anchor'):
        raise ValueError(f'Unknown Anchor variant: {variant}')
    torch.manual_seed(seed)
    model = (_IdentityAnchorModel() if variant == 'identity_anchor' else JointHistoryModel())
    if variant == 'identity_anchor':
        model.calibration.requires_grad_(False)
    return model
