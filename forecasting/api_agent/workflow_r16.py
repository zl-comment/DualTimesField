"""Route only conflicting recent-fit evidence to Kev; keep easy decisions in code."""
from __future__ import annotations

import urllib.error
import numpy as np

from .workflow import PROFILES, SKILLS, fixed_choice


def needs_kev(evidence):
    scores = evidence['profiles']
    eligible = [p for p in PROFILES if p != 'base' and scores[p]['clears_2pct_change_guard']]
    fixed = fixed_choice(evidence)
    if len(eligible) <= 1:
        return False, eligible, 'zero_or_one_supported_alternative'
    dominates = all(all(scores[fixed][metric] <= scores[p][metric]
                        for metric in ('qs_first_12h', 'qs_last_12h')) for p in eligible)
    return not dominates, eligible, 'clear_recent_fit_dominance' if dominates else 'conflicting_recent_fit'


def decide(snapshot, evidence, client):
    fixed = fixed_choice(evidence)
    use_kev, eligible, reason = needs_kev(evidence)
    trace = {'snapshot_id': snapshot.identity(), 'fixed_choice': fixed, 'calls': [],
             'tool_calls': [{'tool': 'inspect_recent_fit', 'result': evidence}],
             'routing': reason, 'fallback': None}
    if not use_kev:
        trace['selected_profile'] = fixed
        return fixed, trace
    p = snapshot.price
    state = {
        'task': 'Resolve a trade-off between frozen electricity forecast profiles.',
        'default_profile': fixed, 'recent_fit': evidence,
        'last_day_price_median': float(np.median(p[-24:])),
        'preceding_two_days_price_median': float(np.median(p[:48])),
        'last_day_price_range': float(np.ptp(p[-24:])),
        'preceding_two_days_price_range': float(np.ptp(p[:48])),
        'no_future_outcomes_available': True,
        'allowed_profiles': ['base', *eligible]}
    descriptions = {'base': 'Retain price, past demand, and calendar inputs.',
                    'calendar': 'Use price and calendar inputs without demand.',
                    'price_only': 'Use the observed price history without covariates.'}
    try:
        choice, call = client.choose(state, (SKILLS / 'resolve_tradeoff_v1.txt').read_text(),
                                     {p: descriptions[p] for p in ('base', *eligible)})
        trace['calls'].append(call)
        if call['selected_probability'] < .55:
            choice = fixed
            trace['guard'] = 'uncertain_decision_uses_fixed_default'
        if choice != 'base' and choice not in eligible:
            raise ValueError('Unsupported profile change')
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError) as exc:
        if hasattr(exc, 'record'):
            trace['calls'].append(exc.record)
        choice = fixed
        trace['fallback'] = {'to': 'FIXED', 'reason': type(exc).__name__, 'detail': str(exc)[:500]}
    trace['selected_profile'] = choice
    return choice, trace
