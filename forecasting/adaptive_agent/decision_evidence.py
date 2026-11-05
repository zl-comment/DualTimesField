"""Deterministic R21 evidence admission before Kev may change model input."""
from __future__ import annotations

import math


SCHEMA = 'r21-input-decision-evidence-v2'
MIN_SUPPORT = 5
MIN_WIN_FRACTION = 0.625


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _current_disagreement(observation, default_id, candidate_id):
    comparison = observation.get('current_forecast_comparison', {})
    for row in comparison.get('pairs', []):
        if {row.get('a'), row.get('b')} == {default_id, candidate_id}:
            value = row.get('median_absolute_difference')
            return float(value) if _finite(value) else None
    return None


def _regime_hypothesis(numerical_state, candidate_id):
    days = numerical_state.get('days_oldest_first', []) if isinstance(numerical_state, dict) else []
    means = [row.get('demand_std_mean') for row in days if isinstance(row, dict)]
    if len(means) != 3 or not all(_finite(value) for value in means):
        return {'available': False, 'reason': 'three_daily_demand_means_unavailable'}
    oldest, middle, latest = map(float, means)
    if candidate_id.endswith(':history_demand_last24_mean'):
        shift = latest - (oldest + middle) / 2
        contrast = 'latest_24h_minus_older_48h_mean'
    elif candidate_id.endswith(':history_demand_last48_mean'):
        shift = (middle + latest) / 2 - oldest
        contrast = 'recent_48h_mean_minus_oldest_24h'
    else:
        return {'available': False, 'reason': 'no_registered_regime_contrast'}
    return {'available': True, 'contrast': contrast, 'signed_shift': shift,
            'absolute_shift': abs(shift),
            'units': 'frozen_standardized_demand_daily_mean'}


def build_decision_evidence(observation, default_id, numerical_state):
    """Return the finite operation set that Kev is permitted to consider.

    Positive gain means the operated input had lower QS than identity on the
    same fully mature paired historical episode.  The thresholds are protocol
    constants rather than values emitted by the LLM.
    """
    candidates = observation.get('candidates', {})
    result = {
        'schema': SCHEMA,
        'default_candidate_id': default_id,
        'rule': {
            'minimum_common_support': MIN_SUPPORT,
            'minimum_paired_win_fraction': MIN_WIN_FRACTION,
            'requires_positive_mean_qs_gain': True,
            'requires_positive_median_qs_gain': True,
            'requires_positive_first12_qs_gain': True,
            'requires_positive_last12_qs_gain': True,
            'requires_nonzero_current_forecast_disagreement': True,
            'gain_definition': 'identity_qs_minus_operated_qs; positive favors operation',
        },
        'eligible_candidate_ids': [],
        'operations': {},
    }
    default = candidates.get(default_id, {})
    default_metrics = default.get('metrics') or {}
    for candidate_id in sorted(key for key in candidates if key != default_id):
        row = candidates[candidate_id]
        paired = row.get('paired_vs_default') or {}
        metrics = row.get('metrics') or {}
        support = row.get('sample_count')
        disagreement = _current_disagreement(observation, default_id, candidate_id)
        first_gain = (default_metrics.get('qs_first12') - metrics.get('qs_first12')
                      if _finite(default_metrics.get('qs_first12'))
                      and _finite(metrics.get('qs_first12')) else None)
        last_gain = (default_metrics.get('qs_last12') - metrics.get('qs_last12')
                     if _finite(default_metrics.get('qs_last12'))
                     and _finite(metrics.get('qs_last12')) else None)
        checks = {
            'common_support': isinstance(support, int) and support >= MIN_SUPPORT,
            'positive_mean_qs_gain': _finite(paired.get('mean_qs_gain'))
                                     and paired['mean_qs_gain'] > 0,
            'positive_median_qs_gain': _finite(paired.get('median_qs_gain'))
                                       and paired['median_qs_gain'] > 0,
            'paired_win_fraction': _finite(paired.get('win_fraction'))
                                   and paired['win_fraction'] >= MIN_WIN_FRACTION,
            'positive_first12_qs_gain': _finite(first_gain) and first_gain > 0,
            'positive_last12_qs_gain': _finite(last_gain) and last_gain > 0,
            'current_forecast_changes': _finite(disagreement) and disagreement > 0,
        }
        eligible = all(checks.values())
        operation = {
            'candidate_id': candidate_id,
            'eligible': eligible,
            'checks': checks,
            'common_support': support,
            'paired_qs': paired,
            'first12_qs_gain': first_gain,
            'last12_qs_gain': last_gain,
            'current_median_absolute_forecast_difference': disagreement,
            'current_demand_regime_hypothesis': _regime_hypothesis(numerical_state, candidate_id),
            'rejection_reasons': [key for key, passed in checks.items() if not passed],
        }
        result['operations'][candidate_id] = operation
        if eligible:
            result['eligible_candidate_ids'].append(candidate_id)
    result['decision'] = ('kev_may_choose_identity_or_eligible_operation'
                          if result['eligible_candidate_ids'] else 'protocol_commit_identity')
    return result
