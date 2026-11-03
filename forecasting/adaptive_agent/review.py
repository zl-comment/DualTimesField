"""Structural checks for submissions; no rule forcing recent-QS winners."""
from __future__ import annotations

import numpy as np


def validate_artifact(context, artifact):
    """Reject wrong snapshots, target axes and invalid numerical forecasts."""
    if artifact.context_id != context.identity():
        raise ValueError('Forecast belongs to a different context')
    if not artifact.candidate_id or not artifact.model_id or not artifact.input_hash:
        raise ValueError('Forecast identity is incomplete')
    q = np.asarray(artifact.quantiles)
    if q.shape != (24, 5) or not np.isfinite(q).all():
        raise ValueError('Expected finite 24 by 5 forecast')
    if np.any(np.diff(q, axis=1) < 0):
        raise ValueError('Crossing quantiles cannot be submitted')
    if not np.array_equal(artifact.target_time_unix, context.snapshot.target_time_unix):
        raise ValueError('Forecast target axis differs from the requested axis')
    return artifact


def review_criteria(candidate_id):
    return {
        'approve': 'Approve this candidate on the currently available evidence; this is not a guarantee of accuracy.',
        'continue': 'Evidence is insufficient or contradictory; continue gathering evidence if budget remains.',
        'default': 'Abstain from this proposal and submit the original frozen default.',
    }
