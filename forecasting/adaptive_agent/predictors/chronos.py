"""Frozen Chronos adapter: actual contiguous context selection, no fitting."""
from __future__ import annotations

import numpy as np


def make_input(context, profile_id, window_hours):
    if profile_id not in ('base', 'calendar', 'price_only') or window_hours not in (24, 48, 72):
        raise ValueError('Unregistered Chronos input configuration')
    snapshot = context.snapshot
    row = {'target': snapshot.price[-window_hours:].copy()}
    past, future = {}, {}
    if profile_id == 'base':
        past['demand_std'] = snapshot.demand_std[-window_hours:].copy()
    if profile_id in ('base', 'calendar'):
        for j in range(10):
            past[f'cal_{j}'] = snapshot.calendar_past[-window_hours:, j].copy()
            future[f'cal_{j}'] = snapshot.calendar_future[:, j].copy()
    if past:
        row['past_covariates'] = past
    if future:
        row['future_covariates'] = future
    return row


class FrozenChronosAdapter:
    def __init__(self, pipeline, identity):
        if not identity or not identity.get('sha256'):
            raise ValueError('Frozen Chronos weights need a verified SHA256 identity')
        self.pipeline = pipeline
        self.identity = dict(identity)

    def predict(self, context, profile_id, window_hours):
        from ...chronos2_zeroshot.model import predict_common_quantiles
        row = make_input(context, profile_id, window_hours)
        return np.sort(predict_common_quantiles(
            self.pipeline, [row], batch_size=1, context_length=window_hours)[0], axis=-1)
