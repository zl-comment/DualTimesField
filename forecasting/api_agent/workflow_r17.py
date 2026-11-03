"""Bounded tool-using decision loop over one frozen forecasting snapshot.

Kev chooses tools and an input profile; Python computes all evidence and prices.
No external retrieval, mutable cross-origin memory, training, or label access.
"""
from __future__ import annotations

import hashlib
import urllib.error

import numpy as np

from .workflow import PROFILES, SKILLS, digest, fixed_choice

RELEASE = 'r17-tool-agent-v1'
MAX_CALLS = 4
SKILL_FILE = SKILLS / 'tool_agent_v1.txt'
TOOL_DESCRIPTIONS = {
    'inspect_recent_fit': 'Compare all profiles on the last 24 already observed hours.',
    'inspect_history': 'Inspect changes in observed price and standardized demand over three days.',
    'inspect_forecast_disagreement': 'Inspect candidate forecast medians and interval widths; these are predictions, not outcomes.',
}


def validate_inputs(snapshot, evidence, forecasts):
    if evidence.get('hindcast_target_end_exclusive') != snapshot.origin_unix:
        raise ValueError('Recent-fit evidence belongs to a different origin')
    if set(forecasts) != set(PROFILES):
        raise ValueError('Expected exactly the registered frozen forecasts')
    for name in PROFILES:
        q = np.asarray(forecasts[name])
        if q.shape != (24, 5) or not np.isfinite(q).all() or np.any(np.diff(q, axis=1) < 0):
            raise ValueError('Invalid frozen forecast: ' + name)
    # Reject nonfinite/non-JSON evidence before it can enter a request or trace.
    digest(evidence)


def inspect_history(snapshot):
    days = []
    for start in (0, 24, 48):
        price = snapshot.price[start:start + 24]
        demand = snapshot.demand_std[start:start + 24]
        days.append({
            'start_unix': int(snapshot.origin_unix - (72 - start) * 3600),
            'end_exclusive_unix': int(snapshot.origin_unix - (48 - start) * 3600),
            'price_median': float(np.median(price)),
            'price_min': float(price.min()), 'price_max': float(price.max()),
            'price_mean_absolute_hourly_change': float(np.abs(np.diff(price)).mean()),
            'demand_standardized_mean': float(demand.mean()),
            'demand_standardized_range': float(np.ptp(demand)),
        })
    return {'observed_days_oldest_first': days,
            'limitation': 'Descriptive observed history; no claim about future events or causation.'}


def inspect_disagreement(forecasts):
    medians = np.stack([forecasts[p][:, 2] for p in PROFILES])
    spread = np.ptp(medians, axis=0)
    return {
        'kind': 'frozen_predictions_not_observations',
        'median_spread_mean': float(spread.mean()),
        'median_spread_max': float(spread.max()),
        'most_disagreed_horizon_index': int(spread.argmax()),
        'profiles': {p: {
            'forecast_median_mean': float(forecasts[p][:, 2].mean()),
            'forecast_median_min': float(forecasts[p][:, 2].min()),
            'forecast_median_max': float(forecasts[p][:, 2].max()),
            'interval_90_width_mean': float((forecasts[p][:, -1] - forecasts[p][:, 0]).mean()),
        } for p in PROFILES},
        'limitation': 'Agreement or narrow intervals do not establish accuracy.'}


def decide(snapshot, evidence, client, *, forecasts):
    validate_inputs(snapshot, evidence, forecasts)
    skill = SKILL_FILE.read_text()
    fixed = fixed_choice(evidence)
    trace = {'release': RELEASE, 'snapshot_id': snapshot.identity(),
             'fixed_choice': fixed, 'calls': [], 'tool_calls': [], 'fallback': None,
             'skill_sha256': hashlib.sha256(skill.encode()).hexdigest(),
             'forecast_fingerprint': digest({p: forecasts[p].tolist() for p in PROFILES})}
    state = {'task': 'Choose tools, inspect evidence, then commit a frozen forecast profile.',
             'visible_history_hours': 72, 'horizon_hours': 24,
             'history_end_exclusive_unix': int(snapshot.origin_unix),
             'default_profile': fixed, 'evidence': {},
             'no_future_outcomes_available': True}

    def finish(profile, reason):
        trace.update(selected_profile=profile, stop_reason=reason)
        return profile, trace

    try:
        for step in range(MAX_CALLS):
            criteria = {
                'finish_fixed': 'Commit the fixed recent-fit rule and stop.',
                'finish_base': 'Commit the original price + past demand + calendar profile and stop.',
            }
            if 'inspect_recent_fit' in state['evidence']:
                for p in PROFILES[1:]:
                    if evidence['profiles'][p]['clears_2pct_change_guard']:
                        criteria['finish_' + p] = 'Commit the supported ' + p + ' profile and stop.'
            if step < MAX_CALLS - 1:
                criteria.update({k: v for k, v in TOOL_DESCRIPTIONS.items()
                                 if k not in state['evidence']})
            # Fresh JSON-compatible state: previous call records cannot be mutated
            # when the next observation is added to the working state.
            request_state = {**state, 'evidence': dict(state['evidence']),
                             'remaining_decisions': MAX_CALLS - step}
            action, call = client.choose(request_state, skill, criteria)
            trace['calls'].append(call)
            if action not in criteria:
                raise ValueError('Unregistered or unavailable action')
            if action.startswith('finish_'):
                if call['selected_probability'] < .55:
                    trace['guard'] = 'uncertain_commit_uses_fixed'
                    return finish(fixed, 'uncertain_commit')
                profile = fixed if action == 'finish_fixed' else action[len('finish_'):]
                return finish(profile, 'agent_commit')
            if action == 'inspect_recent_fit':
                result = evidence
            elif action == 'inspect_history':
                result = inspect_history(snapshot)
            else:
                result = inspect_disagreement(forecasts)
            state['evidence'][action] = result
            trace['tool_calls'].append({'tool': action, 'version': '1.0.0', 'result': result})
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError) as exc:
        if hasattr(exc, 'record'):
            trace['calls'].append(exc.record)
        trace['fallback'] = {'to': 'FIXED', 'reason': type(exc).__name__, 'detail': str(exc)[:500]}
        return finish(fixed, 'failed_decision')
    raise AssertionError('Decision loop must stop within its call budget')
