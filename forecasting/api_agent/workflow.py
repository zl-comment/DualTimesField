"""Two bounded decisions over one origin; no inference access to evaluation labels.

All numerical forecasts come from frozen Chronos. Kev can request the recent-fit
diagnostic and select one registered covariate profile; it cannot write prices.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import urllib.error

import numpy as np

PROFILES = ('base', 'calendar', 'price_only')
LEVELS = np.array([.05, .1, .5, .9, .95])
SKILLS = Path(__file__).parent / 'skills'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Snapshot:
    region: str
    origin_unix: int
    target_time_unix: np.ndarray
    price: np.ndarray
    demand_std: np.ndarray
    calendar_past: np.ndarray
    calendar_future: np.ndarray
    preprocessing_hash: str

    def __post_init__(self):
        expected = {'price': (72,), 'demand_std': (72,), 'calendar_past': (72, 10),
                    'calendar_future': (24, 10), 'target_time_unix': (24,)}
        if self.region not in ('NSW1', 'QLD1', 'TAS1'):
            raise ValueError('Unknown region')
        for key, shape in expected.items():
            value = np.array(getattr(self, key), copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Invalid snapshot field: ' + key)
            value.setflags(write=False)
            object.__setattr__(self, key, value)
        if not np.array_equal(self.target_time_unix, self.origin_unix + 3600 * np.arange(24)):
            raise ValueError('Target axis must be the existing direct 24-hour axis')
        if not self.preprocessing_hash:
            raise ValueError('Missing frozen preprocessing identity')

    def identity(self):
        return digest({'region': self.region, 'origin': self.origin_unix,
                       'preprocessing': self.preprocessing_hash,
                       **{k: getattr(self, k).tolist() for k in (
                           'price', 'demand_std', 'calendar_past', 'calendar_future', 'target_time_unix')}})


def predictor_input(snapshot, profile, *, hindcast=False):
    if profile not in PROFILES:
        raise ValueError('Unregistered forecast profile')
    # Hindcast predicts the last 24 observed hours from the preceding 48.
    # Its labels are inside this origin's 72-hour history, never future targets.
    n = 48 if hindcast else 72
    past, future = {}, {}
    if profile == 'base':
        past['demand_std'] = snapshot.demand_std[:n]
    if profile in ('base', 'calendar'):
        future_calendar = snapshot.calendar_past[48:] if hindcast else snapshot.calendar_future
        for j in range(10):
            past[f'cal_{j}'] = snapshot.calendar_past[:n, j]
            future[f'cal_{j}'] = future_calendar[:, j]
    row = {'target': snapshot.price[:n]}
    if past:
        row['past_covariates'] = past
    if future:
        row['future_covariates'] = future
    return row


def quantile_loss(actual, quantiles):
    q = np.sort(np.asarray(quantiles), axis=-1)
    error = np.asarray(actual)[..., None] - q
    return float(2 * np.maximum(LEVELS * error, (LEVELS - 1) * error).mean())


def forecast_profiles(snapshot, pipeline):
    from ..chronos2_zeroshot.model import predict_common_quantiles
    forecasts, hindcasts = {}, {}
    for profile in PROFILES:
        # Chronos rejects heterogeneous covariate keys within one API batch.
        # Keep the future forecast and its hindcast together for each profile.
        rows = [predictor_input(snapshot, profile, hindcast=h) for h in (False, True)]
        raw = predict_common_quantiles(pipeline, rows, batch_size=32, context_length=72)
        forecasts[profile], hindcasts[profile] = raw[0], raw[1]
    return forecasts, hindcasts


def diagnostics(snapshot, hindcasts):
    y = snapshot.price[-24:]
    recent_fit = {}
    for name in PROFILES:
        q = np.sort(hindcasts[name], axis=-1)
        recent_fit[name] = {
            'qs_24h': quantile_loss(y, q),
            'qs_first_12h': quantile_loss(y[:12], q[:12]),
            'qs_last_12h': quantile_loss(y[12:], q[12:]),
            'mae_24h': float(np.abs(y - q[:, 2]).mean()),
            'coverage_90': float(((y >= q[:, 0]) & (y <= q[:, -1])).mean())}
    return {'diagnostic': 'hindcast_inside_visible_history',
            'history_window_hours': 72, 'hindcast_context_hours': 48,
            'hindcast_horizon_hours': 24,
            'hindcast_target_end_exclusive': snapshot.origin_unix,
            'limitation': 'One recent day; the lowest past error need not be best tomorrow.',
            'profiles': recent_fit}


def fixed_choice(evidence):
    scores = evidence['profiles']
    winner = min(PROFILES, key=lambda p: scores[p]['qs_24h'])
    # A fixed, preregistered guard; no fitting of a new numerical predictor.
    return winner if scores[winner]['qs_24h'] < .98 * scores['base']['qs_24h'] else 'base'


def decide(snapshot, evidence, client):
    fixed = fixed_choice(evidence)
    trace = {'snapshot_id': snapshot.identity(), 'fixed_choice': fixed,
             'calls': [], 'tool_calls': [], 'fallback': None}
    price = snapshot.price
    state = {'task': 'Select a frozen electricity-price forecast input profile.',
             'visible_history_hours': 72,
             'last_day_price_median': float(np.median(price[-24:])),
             'preceding_two_days_price_median': float(np.median(price[:48])),
             'last_day_price_range': float(np.ptp(price[-24:])),
             'preceding_two_days_price_range': float(np.ptp(price[:48])),
             'available_tool': 'inspect_recent_fit: compare three frozen profiles on the last observed day.',
             'no_future_outcomes_available': True}
    try:
        action, call = client.choose(state, (SKILLS / 'triage.txt').read_text(), {
            'inspect_recent_fit': 'Inspect numerical forecast errors for the last observed day.',
            'keep_base': 'Stop and retain the standard frozen forecast.'})
        trace['calls'].append(call)
        if action == 'keep_base':
            trace['selected_profile'] = 'base'
            return 'base', trace
        trace['tool_calls'].append({'tool': 'inspect_recent_fit', 'result': evidence})
        state['recent_fit'] = evidence
        choice, call = client.choose(state, (SKILLS / 'select_forecast.txt').read_text(), {
            'base': 'Use price, past demand, and ten known calendar features.',
            'calendar': 'Use price and ten calendar features; omit past demand.',
            'price_only': 'Use the 72 observed prices without covariates.'})
        trace['calls'].append(call)
        # Certainty about an option is not the probability of forecasting gain.
        # Numerical guard rejects a proposed change contradicted by recent fit.
        if choice != 'base' and (call['selected_probability'] < .55 or
                evidence['profiles'][choice]['qs_24h'] >= .98 * evidence['profiles']['base']['qs_24h']):
            trace['guard'] = 'insufficient_support_for_change'
            choice = 'base'
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError) as exc:
        if hasattr(exc, 'record'):
            trace['calls'].append(exc.record)
        choice = fixed
        trace['fallback'] = {'to': 'FIXED', 'reason': type(exc).__name__, 'detail': str(exc)[:500]}
    trace['selected_profile'] = choice
    return choice, trace
