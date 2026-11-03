"""Registered quantile averaging; this is not exact mixture quantiles."""
from __future__ import annotations

import numpy as np
from .registry import digest

WEIGHTS = (.25, .5, .75)


def pool_quantiles(a, b, weight=.5):
    from .contracts import ForecastArtifact
    if weight not in WEIGHTS or isinstance(weight, bool):
        raise ValueError('Unregistered pooling weight')
    if a.context_id != b.context_id or not np.array_equal(a.target_time_unix, b.target_time_unix):
        raise ValueError('Cannot pool different contexts or target axes')
    if a.candidate_id == b.candidate_id:
        raise ValueError('Pool requires two distinct candidates')
    for artifact in (a, b):
        if artifact.quantiles.shape != (24, 5) or not np.isfinite(artifact.quantiles).all():
            raise ValueError('Invalid pooling source')
    identity = {'a': a.identity(), 'b': b.identity(), 'weight_a': weight,
                'recipe': 'sorted_same_quantile_weighted_sum_v1'}
    quantiles = weight * np.sort(a.quantiles, axis=-1) + (1 - weight) * np.sort(b.quantiles, axis=-1)
    return ForecastArtifact(
        candidate_id=f'pool:{a.candidate_id}:{b.candidate_id}:{weight:g}',
        context_id=a.context_id, quantiles=quantiles, target_time_unix=a.target_time_unix,
        model_id='quantile_pool', input_hash=digest(identity),
        provenance=dict(identity, source_candidates=[a.candidate_id, b.candidate_id],
                        exact_mixture_quantiles=False, training_performed=False))
