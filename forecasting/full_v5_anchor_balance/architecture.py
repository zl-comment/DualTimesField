"""Matched-initialization Anchor architectures for predeclared comparisons."""
import torch
from torch import nn

from ..full_v5_joint.model import pinball
from .objective import build_model as build_reference_model

VARIANTS = ('balanced', 'identity_anchor', 'location_only', 'separate_heads',
            'staged', 'wide_head')


class _LocationOnlyHead(nn.Sequential):
    """Retain the original head and ignore its scale output, making s exactly 1."""

    def forward(self, features):
        raw = super().forward(features)
        return torch.cat((raw[..., :1], torch.zeros_like(raw[..., 1:2])), dim=-1)


def _head(hidden, outputs):
    head = nn.Sequential(nn.Linear(32, hidden), nn.GELU(), nn.Linear(hidden, outputs))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class _SeparateHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.location = _head(12, 1)
        self.scale = _head(12, 1)

    def forward(self, features):
        return torch.cat((self.location(features), self.scale(features)), dim=-1)


def build_model(variant, seed=2026):
    """Replace only calibration after constructing the complete original model.

    All non-calibration parameters and buffers are bit-identical for a given
    seed. All heads start at delta=0, scale=1. Staged shares balanced's exact
    architecture; call configure_stage explicitly when entering either stage.
    """
    if variant not in VARIANTS:
        raise ValueError(f'Unknown Anchor architecture: {variant}')
    model = build_reference_model('identity_anchor' if variant == 'identity_anchor' else 'balanced', seed)
    # Replacement initialization must not alter subsequent CPU random draws.
    # Start from the same post-reference-model RNG state and restore it on exit.
    with torch.random.fork_rng(devices=[]):
        if variant == 'location_only':
            model.calibration = _LocationOnlyHead(*model.calibration.children())
        elif variant == 'separate_heads':
            model.calibration = _SeparateHeads()
        elif variant == 'wide_head':
            model.calibration = _head(48, 2)
    return model


def configure_stage(model, variant, stage):
    """Set trainability and clear stale gradients; rebuild the optimizer afterward.

    These flags are not stored in a state_dict. On restore, reconstruct the
    variant, load weights and reapply the saved stage before resuming training.
    Returns the same model. 'joint' is available for all variants; 'anchor' and
    'history' are exclusive to staged, with the shared state encoder frozen in
    the latter so the Anchor cannot drift during historical training.
    """
    if variant not in VARIANTS:
        raise ValueError(f'Unknown Anchor architecture: {variant}')
    if stage not in ('joint', 'anchor', 'history') or (variant != 'staged' and stage != 'joint'):
        raise ValueError(f'Invalid training stage {stage!r} for {variant!r}')
    for name, parameter in model.named_parameters():
        anchor_parameter = name.startswith(('state.', 'calibration.'))
        trainable = (anchor_parameter if stage == 'anchor' else
                     not anchor_parameter if stage == 'history' else True)
        if variant == 'identity_anchor' and name.startswith('calibration.'):
            trainable = False
        parameter.requires_grad_(trainable)
        parameter.grad = None
    return model


def anchor_pretrain_loss(output, targets, baseline):
    """Standalone Anchor supervision; no history or reconstruction objective.

    The .5*MAE term is exactly one q50 pinball term, normalized by 300.
    """
    anchor = output['anchor']
    terms = dict(anchor_pinball=pinball(anchor, targets) / 300,
                 anchor_median=.5 * (anchor[..., 2] - targets).abs().mean() / 300,
                 identity=.01 * ((anchor - baseline) / 300).square().mean())
    return terms['anchor_pinball'] + terms['anchor_median'] + terms['identity'], terms
