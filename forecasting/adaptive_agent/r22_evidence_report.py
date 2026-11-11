"""Pure R22 structured evidence report over cutoff-visible bounded summaries."""
from __future__ import annotations

from collections import Counter
import json
import math
from numbers import Integral, Real

import numpy as np

from .contracts import plain


SCHEMA = 'r22-structured-evidence-v1'
DIAGNOSTIC_SCHEMA = 'r22-paired-diagnostic-v1'
BOOTSTRAP_SEED = 20261111
BOOTSTRAP_REPLICATES = 2000
MATERIAL_QS_LOSS_THRESHOLD = .01
MIN_SUPPORT = 5
MIN_WIN_FRACTION = .625
_FINGERPRINT_FIELDS = {
    'predictor_fingerprint', 'operation_admission_fingerprint',
    'skill_fingerprint', 'source_fingerprint',
}
_FORBIDDEN_KEYS = {'actual', 'actuals', 'label', 'labels', 'target', 'targets', 'rows'}
_BUCKETS = ('hours_1_6', 'hours_7_12', 'hours_13_18', 'hours_19_24')


def _finite(value):
    return (isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _exact_keys(value, expected, name):
    _require(isinstance(value, dict) and set(value) == set(expected),
             f'Invalid {name} fields')


def _walk_finite_and_label_free(value, path='report'):
    if isinstance(value, dict):
        for key, item in value.items():
            _require(isinstance(key, str) and key.lower() not in _FORBIDDEN_KEYS,
                     f'Forbidden label-bearing field at {path}.{key}')
            _walk_finite_and_label_free(item, f'{path}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_finite_and_label_free(item, f'{path}[{index}]')
    elif isinstance(value, Real) and not isinstance(value, bool):
        _require(math.isfinite(float(value)), f'Nonfinite number at {path}')


def _summary(values):
    values = np.asarray(values, dtype=float)
    _require(values.ndim == 1 and len(values) and np.isfinite(values).all(),
             'Summary requires finite values')
    return {'minimum': float(values.min()), 'median': float(np.median(values)),
            'maximum': float(values.max()), 'std': float(values.std())}


def _regime_contrast(numerical_state, candidate_id):
    days = numerical_state.get('days_oldest_first', [])
    means = [row.get('demand_std_mean') for row in days if isinstance(row, dict)]
    if len(means) != 3 or not all(_finite(value) for value in means):
        return {'available': False, 'contrast': 'unavailable', 'signed_shift': None,
                'absolute_shift': None,
                'units': 'frozen_standardized_demand_daily_mean'}
    oldest, middle, latest = map(float, means)
    if candidate_id.endswith(':history_demand_last24_mean'):
        shift = latest - (oldest + middle) / 2
        contrast = 'latest_24h_minus_older_48h_mean'
    elif candidate_id.endswith(':history_demand_last48_mean'):
        shift = (middle + latest) / 2 - oldest
        contrast = 'recent_48h_mean_minus_oldest_24h'
    else:
        return {'available': False, 'contrast': 'unregistered_operation_contrast',
                'signed_shift': None, 'absolute_shift': None,
                'units': 'frozen_standardized_demand_daily_mean'}
    return {'available': True, 'contrast': contrast, 'signed_shift': float(shift),
            'absolute_shift': float(abs(shift)),
            'units': 'frozen_standardized_demand_daily_mean'}


def _find_disagreement(diagnostic, default_id, candidate_id):
    for row in diagnostic['current_forecast_comparison']['pairs']:
        if {row.get('a'), row.get('b')} == {default_id, candidate_id}:
            fields = ('median_absolute_difference', 'q05_absolute_difference',
                      'q95_absolute_difference', 'width90_absolute_difference')
            _require(all(_finite(row.get(key)) and row[key] >= 0 for key in fields),
                     'Current forecast disagreement is incomplete or nonfinite')
            return {key: float(row[key]) for key in fields}
    raise ValueError('Missing current forecast disagreement pair')


def _current_regime(context, numerical_state, diagnostic):
    snapshot = context.snapshot
    price = np.asarray(snapshot.price, dtype=float)
    demand = np.asarray(snapshot.demand_std, dtype=float)
    target_axis = np.asarray(snapshot.target_time_unix)
    _require(price.shape == demand.shape == (72,) and np.isfinite(price).all()
             and np.isfinite(demand).all(), 'Current visible history is invalid')
    latest_price, older_price = price[-24:], price[:48]
    latest_demand, older_demand = demand[-24:], demand[:48]
    expected_axis = context.origin_unix + np.arange(24) * 3600
    duplicate_count = int(len(target_axis) - len(np.unique(target_axis)))
    horizon_coverage_count = int(np.sum(target_axis == expected_axis))
    exogenous = {}
    for name in ('future_exogenous', 'ctf_exogenous', 'quantile_exogenous'):
        available = name in context.model_inputs
        values = np.asarray(context.model_inputs[name]) if available else np.asarray([])
        staleness_key = name + '_staleness_hours'
        staleness = context.metadata.get(staleness_key)
        _require(staleness is None or (_finite(staleness) and staleness >= 0),
                 'Invalid exogenous staleness metadata')
        exogenous[name] = {
            'available': available,
            'value_count': int(values.size),
            'finite_count': int(np.isfinite(values).sum()),
            'coverage_fraction': float(np.isfinite(values).mean()) if values.size else 0.,
            'staleness_available': staleness is not None,
            'staleness_hours': float(staleness) if staleness is not None else None,
        }
    nonfinite_count = int((~np.isfinite(price)).sum() + (~np.isfinite(demand)).sum()
                          + sum(row['value_count'] - row['finite_count']
                                for row in exogenous.values()))
    missing_count = sum(not row['available'] for row in exogenous.values())
    data_quality = {
        'nonfinite_count': nonfinite_count,
        'missing_exogenous_block_count': int(missing_count),
        'duplicate_target_time_count': duplicate_count,
        'horizon_coverage_count': horizon_coverage_count,
        'expected_horizon_count': 24,
        'critical_failure_count': int(nonfinite_count + duplicate_count
                                      + (horizon_coverage_count != 24)),
        'valid': nonfinite_count == 0 and duplicate_count == 0 and horizon_coverage_count == 24,
    }
    candidates = diagnostic['current_forecast_comparison']['candidates']
    current_forecasts = {}
    for candidate_id, row in sorted(candidates.items()):
        fields = ('q05_mean', 'median_mean', 'q95_mean', 'width90_mean')
        _require(all(_finite(row.get(key)) for key in fields),
                 'Invalid current forecast summary')
        current_forecasts[candidate_id] = {key: float(row[key]) for key in fields}
    tags = numerical_state.get('state_tags', [])
    _require(isinstance(tags, list) and all(isinstance(tag, str) and tag for tag in tags),
             'Invalid registered state tags')
    return {
        'price_latest_24h': {
            'mean': float(latest_price.mean()), 'std': float(latest_price.std()),
            'range': float(np.ptp(latest_price)),
            'maximum_absolute_hourly_ramp': float(np.abs(np.diff(latest_price)).max()),
            'mean_change_against_older_48h': float(latest_price.mean() - older_price.mean()),
        },
        'demand_latest_24h': {
            'standardized_mean': float(latest_demand.mean()),
            'standardized_mean_change_against_older_48h': float(
                latest_demand.mean() - older_demand.mean()),
        },
        'state_tags': list(tags), 'exogenous_quality': exogenous,
        'data_quality': data_quality, 'current_forecasts': current_forecasts,
    }


def _cohort_quality(context, diagnostic, retrieval, current_regime):
    items = retrieval.get('items')
    _require(isinstance(items, list) and len(items) <= 8, 'Invalid retrieval items')
    episode_ids = [row.get('episode_id') for row in items]
    _require(all(isinstance(value, str) and value for value in episode_ids)
             and len(set(episode_ids)) == len(episode_ids), 'Invalid retrieval episode IDs')
    _require(diagnostic.get('episodes') == episode_ids, 'Retrieval/diagnostic cohort mismatch')
    distances = [row.get('distance') for row in items]
    origins = [row.get('origin_unix') for row in items]
    _require(all(_finite(value) and value >= 0 for value in distances),
             'Invalid cohort similarity distance')
    _require(all(_integer(value) and value < context.origin_unix for value in origins),
             'Invalid cohort origin')
    roles = Counter(row.get('data_role', 'unknown') for row in items)
    _require(all(isinstance(key, str) and key for key in roles), 'Invalid cohort data role')
    tags = {tag for row in items for tag in row.get('state_tags', [])}
    ages = [(context.origin_unix - value) / 86400 for value in origins]
    duplicate_day_count = len(origins) - len({value // 86400 for value in origins})
    support = diagnostic.get('sample_count')
    fraction = diagnostic.get('paired_support_fraction')
    _require(_integer(support) and 0 <= support <= len(items)
             and _finite(fraction) and 0 <= fraction <= 1, 'Invalid paired cohort support')
    overlap = any(role in {'train', 'selection', 'training_in_sample',
                           'model_selection_exposed', 'training_or_selection_overlap_unknown'}
                  for role in roles)
    data_quality_valid = (current_regime['data_quality']['valid']
                          and duplicate_day_count == 0 and not diagnostic.get('failures')
                          and diagnostic.get('unattempted_count') == 0)
    if not data_quality_valid or support == 0:
        grade = 'invalid'
    elif support < MIN_SUPPORT or fraction < .625:
        grade = 'weak'
    else:
        grade = 'sufficient'
    return {
        'common_paired_support': int(support),
        'paired_support_fraction': float(fraction),
        'similarity_distance': _summary(distances) if distances else None,
        'episode_age_days': _summary(ages) if ages else None,
        'unique_state_tag_count': len(tags),
        'duplicate_day_count': int(duplicate_day_count),
        'data_role_composition': dict(sorted(roles.items())),
        'development_overlap_warning': bool(overlap),
        'coverage_grade': grade, 'data_quality_valid': bool(data_quality_valid),
    }


def _operation_effects(diagnostic, numerical_state, default_id, cohort_quality):
    candidates = diagnostic.get('candidates', {})
    _require(default_id in candidates, 'Default candidate missing from diagnostic')
    operations, uncertainty, eligible = {}, {}, []
    for candidate_id in sorted(key for key in candidates if key != default_id):
        paired = candidates[candidate_id].get('paired_vs_default')
        _require(isinstance(paired, dict), 'Missing paired operation effect')
        support = paired.get('common_support')
        _require(_integer(support) and support == cohort_quality['common_paired_support'],
                 'Operation common support mismatch')
        scalar_fields = ('mean_qs_gain', 'median_qs_gain', 'minimum_qs_gain',
                         'maximum_qs_gain', 'qs_gain_std', 'win_fraction')
        _require(all(_finite(paired.get(key)) for key in scalar_fields),
                 'Nonfinite paired operation effect')
        horizon = paired.get('horizon_qs_gain')
        _require(isinstance(horizon, dict) and set(horizon) == set(_BUCKETS),
                 'Missing 6-hour QS bucket')
        for row in horizon.values():
            _require(isinstance(row, dict) and all(_finite(row.get(key)) for key in
                     ('mean', 'median', 'minimum', 'maximum', 'std')),
                     'Invalid horizon QS gain summary')
        disagreement = _find_disagreement(diagnostic, default_id, candidate_id)
        coverage_delta = paired.get('coverage90_delta')
        width_delta = paired.get('width90_delta')
        downside = paired.get('downside')
        for name, row in (('mae_gain', paired.get('mae_gain')),
                          ('coverage90_delta', coverage_delta), ('width90_delta', width_delta),
                          ('calibration_damage', paired.get('calibration_damage'))):
            _require(isinstance(row, dict) and all(_finite(row.get(key)) for key in
                     ('mean', 'median', 'minimum', 'maximum', 'std')), f'Invalid {name}')
        _require(isinstance(downside, dict)
                 and all(_finite(downside.get(key)) for key in
                         ('worst_paired_qs_loss', 'lower_tail_mean_qs_gain', 'loss_fraction',
                          'material_loss_fraction', 'material_qs_loss_threshold')),
                 'Invalid downside summary')
        _require(downside['material_qs_loss_threshold'] == MATERIAL_QS_LOSS_THRESHOLD,
                 'Unregistered material-loss threshold')
        _require(all(_integer(paired.get(key)) and paired[key] >= 0 for key in
                     ('win_count', 'tie_count', 'loss_count'))
                 and sum(paired[key] for key in ('win_count', 'tie_count', 'loss_count')) == support,
                 'Invalid paired win/tie/loss counts')
        bootstrap = paired.get('bootstrap_mean_qs_gain')
        leave_one_out = paired.get('leave_one_episode_out')
        _require(isinstance(bootstrap, dict)
                 and bootstrap.get('role') == 'small_sample_supporting_evidence_not_significance_test'
                 and bootstrap.get('seed') == BOOTSTRAP_SEED
                 and bootstrap.get('replicates') == BOOTSTRAP_REPLICATES
                 and bootstrap.get('percentiles') == [2.5, 97.5]
                 and isinstance(bootstrap.get('interval'), list)
                 and len(bootstrap['interval']) == 2
                 and all(_finite(value) for value in bootstrap['interval']),
                 'Invalid or unbound small-sample bootstrap')
        _require(isinstance(leave_one_out, dict)
                 and all(_finite(leave_one_out.get(key)) for key in
                         ('minimum_mean_qs_gain', 'maximum_mean_qs_gain'))
                 and isinstance(leave_one_out.get('sign_stable'), bool),
                 'Invalid leave-one-episode-out summary')
        similarity = paired.get('similarity_weighted')
        _require(isinstance(similarity, dict) and isinstance(similarity.get('available'), bool),
                 'Invalid similarity-weighted summary')
        if similarity['available']:
            _require(_finite(similarity.get('mean_qs_gain'))
                     and _finite(similarity.get('effective_support'))
                     and isinstance(similarity.get('sign_agreement_with_unweighted'), bool),
                     'Invalid available similarity-weighted summary')
        operation = {
            'candidate_id': candidate_id, 'common_support': int(support),
            'paired_qs_gain': {key: float(paired[key]) for key in scalar_fields},
            'win_tie_loss': {key: int(paired[key]) for key in
                             ('win_count', 'tie_count', 'loss_count')},
            'horizon_qs_gain': plain(horizon),
            'mae_gain': plain(paired['mae_gain']),
            'coverage90_delta': plain(coverage_delta),
            'width90_delta': plain(width_delta),
            'calibration_damage': plain(paired['calibration_damage']),
            'similarity_weighted': plain(similarity),
            'current_forecast': {
                'identity_candidate_id': default_id,
                'selected_candidate_id': candidate_id,
                'current_disagreement': disagreement,
            },
            'current_demand_regime_contrast': _regime_contrast(numerical_state, candidate_id),
            'downside': plain(downside),
        }
        reasons = []
        if support < MIN_SUPPORT:
            reasons.append('insufficient_common_support')
        if paired['mean_qs_gain'] <= 0:
            reasons.append('nonpositive_mean_qs_gain')
        if paired['median_qs_gain'] <= 0:
            reasons.append('nonpositive_median_qs_gain')
        if paired['win_fraction'] < MIN_WIN_FRACTION:
            reasons.append('weak_win_fraction')
        if any(horizon[key]['mean'] <= 0 for key in _BUCKETS):
            reasons.append('nonpositive_horizon_bucket_qs_gain')
        if disagreement['median_absolute_difference'] <= 0:
            reasons.append('zero_current_forecast_change')
        if cohort_quality['coverage_grade'] != 'sufficient':
            reasons.append('cohort_quality_not_sufficient')
        operation['eligible'] = not reasons
        operation['rejection_reasons'] = reasons
        operations[candidate_id] = operation
        lower, upper = bootstrap['interval']
        if (support >= 7 and lower > 0 and leave_one_out['sign_stable']
                and downside['material_loss_fraction'] == 0):
            confidence = 'strong_supporting_evidence'
        elif support >= MIN_SUPPORT and upper > 0:
            confidence = 'limited_supporting_evidence'
        else:
            confidence = 'weak_supporting_evidence'
        uncertainty[candidate_id] = {
            'bootstrap_mean_qs_gain': plain(bootstrap),
            'leave_one_episode_out': plain(leave_one_out),
            'confidence_grade': confidence,
            'confidence_reasons': [
                'at_most_eight_mature_episodes',
                'bootstrap_is_supporting_evidence_not_a_significance_claim',
                'historical_overlap_flagged_in_cohort_quality',
            ],
        }
        if operation['eligible']:
            eligible.append(candidate_id)
    return operations, uncertainty, eligible


def build_evidence_report(context, diagnostic, retrieval, numerical_state,
                          default_candidate_id, provenance):
    """Build a deterministic, label-free report from R22 bounded summaries.

    ``provenance`` must contain exactly the four immutable source fingerprints;
    context and cohort identities are derived rather than caller asserted.
    """
    _exact_keys(provenance, _FINGERPRINT_FIELDS, 'provenance input')
    _require(all(isinstance(provenance[key], str) and provenance[key]
                 for key in _FINGERPRINT_FIELDS), 'Invalid provenance fingerprint')
    _require(isinstance(default_candidate_id, str) and default_candidate_id,
             'Invalid default candidate ID')
    _require(isinstance(diagnostic, dict)
             and diagnostic.get('r22_aggregate_schema') == DIAGNOSTIC_SCHEMA
             and 'rows' not in diagnostic, 'R22 report requires bounded API-visible diagnostics')
    _require(isinstance(retrieval, dict) and retrieval.get('cohort_id') == diagnostic.get('cohort_id'),
             'Retrieval/diagnostic cohort identity mismatch')
    _require(numerical_state.get('context_id') == context.identity(),
             'Numerical state belongs to another context')
    current_regime = _current_regime(context, numerical_state, diagnostic)
    cohort_quality = _cohort_quality(context, diagnostic, retrieval, current_regime)
    effects, uncertainty, eligible = _operation_effects(
        diagnostic, numerical_state, default_candidate_id, cohort_quality)
    report = {
        'schema': SCHEMA,
        'provenance': {
            'context_id': context.identity(), 'region': context.region,
            'origin_unix': int(context.origin_unix), 'cutoff_unix': int(context.cutoff),
            'memory_watermark': int(context.memory_watermark),
            'cohort_id': diagnostic['cohort_id'],
            'retrieval_strategy': retrieval.get('strategy'),
            'episode_ids': list(diagnostic['episodes']),
            'warehouse_fingerprint': retrieval.get('warehouse_fingerprint'),
            **dict(provenance),
            'bootstrap_seed': BOOTSTRAP_SEED,
            'bootstrap_replicates': BOOTSTRAP_REPLICATES,
            'material_qs_loss_threshold': MATERIAL_QS_LOSS_THRESHOLD,
        },
        'current_regime': current_regime,
        'cohort_quality': cohort_quality,
        'operation_effects': effects,
        'uncertainty_downside': uncertainty,
        'eligible_candidate_ids': eligible,
    }
    validate_evidence_report(report)
    return report


def validate_evidence_report(report):
    """Reject schema drift, nonfinite values, hidden labels, or incoherent IDs."""
    _walk_finite_and_label_free(report)
    _exact_keys(report, {'schema', 'provenance', 'current_regime', 'cohort_quality',
                         'operation_effects', 'uncertainty_downside',
                         'eligible_candidate_ids'}, 'report')
    _require(report['schema'] == SCHEMA, 'Unknown R22 evidence schema')
    provenance_fields = {
        'context_id', 'region', 'origin_unix', 'cutoff_unix', 'memory_watermark',
        'cohort_id', 'retrieval_strategy', 'episode_ids', 'warehouse_fingerprint',
        *_FINGERPRINT_FIELDS, 'bootstrap_seed', 'bootstrap_replicates',
        'material_qs_loss_threshold',
    }
    _exact_keys(report['provenance'], provenance_fields, 'report provenance')
    provenance = report['provenance']
    _require(all(isinstance(provenance[key], str) and provenance[key]
                 for key in {'context_id', 'region', 'cohort_id', 'retrieval_strategy',
                             'warehouse_fingerprint', *_FINGERPRINT_FIELDS}),
             'Invalid report provenance identifier')
    _require(all(_integer(provenance[key]) for key in
                 ('origin_unix', 'cutoff_unix', 'memory_watermark',
                  'bootstrap_seed', 'bootstrap_replicates')),
             'Invalid report provenance integer')
    _require(provenance['origin_unix'] == provenance['cutoff_unix']
             and provenance['memory_watermark'] <= provenance['cutoff_unix'],
             'Invalid report cutoff provenance')
    _require(provenance['bootstrap_seed'] == BOOTSTRAP_SEED
             and provenance['bootstrap_replicates'] == BOOTSTRAP_REPLICATES
             and provenance['material_qs_loss_threshold'] == MATERIAL_QS_LOSS_THRESHOLD,
             'Unregistered report uncertainty configuration')
    _require(isinstance(provenance['episode_ids'], list)
             and all(isinstance(value, str) and value for value in provenance['episode_ids'])
             and len(set(provenance['episode_ids'])) == len(provenance['episode_ids'])
             and len(provenance['episode_ids']) <= 8, 'Invalid provenance episode IDs')
    current = report['current_regime']
    _exact_keys(current, {'price_latest_24h', 'demand_latest_24h', 'state_tags',
                          'exogenous_quality', 'data_quality', 'current_forecasts'},
                'current regime')
    _exact_keys(current['price_latest_24h'], {
        'mean', 'std', 'range', 'maximum_absolute_hourly_ramp',
        'mean_change_against_older_48h'}, 'current price regime')
    _exact_keys(current['demand_latest_24h'], {
        'standardized_mean', 'standardized_mean_change_against_older_48h'},
        'current demand regime')
    _exact_keys(current['data_quality'], {
        'nonfinite_count', 'missing_exogenous_block_count', 'duplicate_target_time_count',
        'horizon_coverage_count', 'expected_horizon_count', 'critical_failure_count', 'valid'},
        'current data quality')
    _require(isinstance(current['data_quality']['valid'], bool)
             and _integer(current['data_quality']['critical_failure_count']),
             'Invalid current data quality')
    operations = report['operation_effects']
    uncertainty = report['uncertainty_downside']
    _require(isinstance(operations, dict) and operations
             and set(operations) == set(uncertainty), 'Operation/uncertainty mismatch')
    eligible = report['eligible_candidate_ids']
    _require(isinstance(eligible, list) and len(set(eligible)) == len(eligible)
             and set(eligible) <= set(operations), 'Invalid eligible candidate IDs')
    cohort = report['cohort_quality']
    _exact_keys(cohort, {'common_paired_support', 'paired_support_fraction',
                         'similarity_distance', 'episode_age_days', 'unique_state_tag_count',
                         'duplicate_day_count', 'data_role_composition',
                         'development_overlap_warning', 'coverage_grade',
                         'data_quality_valid'}, 'cohort quality')
    _require(cohort.get('coverage_grade') in {'sufficient', 'weak', 'invalid'}
             and isinstance(cohort.get('data_quality_valid'), bool)
             and _integer(cohort.get('common_paired_support')), 'Invalid cohort quality')
    for candidate_id, operation in operations.items():
        required = {'candidate_id', 'common_support', 'paired_qs_gain', 'win_tie_loss',
                    'horizon_qs_gain', 'mae_gain', 'coverage90_delta', 'width90_delta',
                    'calibration_damage', 'similarity_weighted', 'current_forecast',
                    'current_demand_regime_contrast', 'downside', 'eligible',
                    'rejection_reasons'}
        _exact_keys(operation, required, 'operation effect')
        _require(operation['candidate_id'] == candidate_id
                 and operation['common_support'] == cohort['common_paired_support']
                 and isinstance(operation['eligible'], bool)
                 and operation['eligible'] == (candidate_id in eligible),
                 'Incoherent operation effect')
        disagreement = operation['current_forecast'].get('current_disagreement', {})
        _exact_keys(operation['current_forecast'], {
            'identity_candidate_id', 'selected_candidate_id', 'current_disagreement'},
            'operation current forecast')
        _require(operation['current_forecast']['selected_candidate_id'] == candidate_id
                 and isinstance(operation['current_forecast']['identity_candidate_id'], str),
                 'Invalid operation forecast candidate identity')
        _require(all(_finite(disagreement.get(key)) and disagreement[key] >= 0 for key in
                     ('median_absolute_difference', 'q05_absolute_difference',
                      'q95_absolute_difference', 'width90_absolute_difference')),
                 'Invalid operation current disagreement')
        downside = operation['downside']
        _require(all(_finite(downside.get(key)) for key in
                     ('material_loss_fraction', 'lower_tail_mean_qs_gain'))
                 and 0 <= downside['material_loss_fraction'] <= 1,
                 'Invalid operation downside')
        for name in ('coverage90_delta', 'width90_delta'):
            _require(all(_finite(operation[name].get(key)) for key in
                         ('mean', 'median', 'minimum', 'maximum', 'std')),
                     f'Invalid operation {name}')
        uncertainty_row = uncertainty[candidate_id]
        _exact_keys(uncertainty_row, {'bootstrap_mean_qs_gain', 'leave_one_episode_out',
                                      'confidence_grade', 'confidence_reasons'},
                    'operation uncertainty')
        _require(uncertainty_row['confidence_grade'] in {
            'strong_supporting_evidence', 'limited_supporting_evidence',
            'weak_supporting_evidence'}, 'Invalid confidence grade')
    try:
        json.dumps(report, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('R22 evidence report is not canonical JSON') from exc
    return report


build_r22_evidence_report = build_evidence_report
validate_r22_evidence_report = validate_evidence_report
