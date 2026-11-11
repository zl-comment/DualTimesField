"""R23 admission over the shared, validated structured evidence payload."""
from __future__ import annotations

from copy import deepcopy

from .r22_evidence_report import (
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, MATERIAL_QS_LOSS_THRESHOLD,
    MIN_SUPPORT, MIN_WIN_FRACTION, SCHEMA, build_evidence_report as build_shared_report,
    validate_evidence_report,
)


ADMISSION_SCHEMA = 'r23-state-aware-admission-v1'


def admission_contract():
    return {
        'schema': ADMISSION_SCHEMA,
        'payload_schema': SCHEMA,
        'minimum_common_support': MIN_SUPPORT,
        'minimum_win_fraction': MIN_WIN_FRACTION,
        'minimum_similarity_effective_support': MIN_SUPPORT,
        'requires_positive_mean_and_median_gain': True,
        'requires_positive_first12_and_last12_gain': True,
        'requires_positive_similarity_weighted_gain': True,
        'six_hour_buckets': 'review_evidence_not_individual_hard_vetoes',
        'risk_thresholds': 'unchanged_r22',
    }


def apply_admission(report):
    """Return a new report without changing historical estimates or input evidence."""
    validate_evidence_report(report)
    result = deepcopy(report)
    result['eligible_candidate_ids'] = []
    for candidate_id, effect in result['operation_effects'].items():
        paired = effect['paired_qs_gain']
        horizon = effect['horizon_qs_gain']
        similarity = effect['similarity_weighted']
        first12 = (horizon['hours_1_6']['mean'] + horizon['hours_7_12']['mean']) / 2
        last12 = (horizon['hours_13_18']['mean'] + horizon['hours_19_24']['mean']) / 2
        reasons = []
        if effect['common_support'] < MIN_SUPPORT:
            reasons.append('insufficient_common_support')
        if paired['mean_qs_gain'] <= 0:
            reasons.append('nonpositive_mean_qs_gain')
        if paired['median_qs_gain'] <= 0:
            reasons.append('nonpositive_median_qs_gain')
        if paired['win_fraction'] < MIN_WIN_FRACTION:
            reasons.append('weak_win_fraction')
        if first12 <= 0:
            reasons.append('nonpositive_first12_qs_gain')
        if last12 <= 0:
            reasons.append('nonpositive_last12_qs_gain')
        if not similarity.get('available'):
            reasons.append('similarity_evidence_unavailable')
        else:
            if similarity['effective_support'] < MIN_SUPPORT:
                reasons.append('insufficient_similarity_effective_support')
            if similarity['mean_qs_gain'] <= 0:
                reasons.append('nonpositive_similarity_weighted_qs_gain')
        if effect['current_forecast']['current_disagreement']['median_absolute_difference'] <= 0:
            reasons.append('zero_current_forecast_change')
        if result['cohort_quality']['coverage_grade'] != 'sufficient':
            reasons.append('cohort_quality_not_sufficient')
        effect['rejection_reasons'] = reasons
        effect['eligible'] = not reasons
        if not reasons:
            result['eligible_candidate_ids'].append(candidate_id)
    validate_evidence_report(result)
    return result


def build_evidence_report(context, diagnostic, retrieval, numerical_state,
                          default_candidate_id, provenance):
    report = build_shared_report(context, diagnostic, retrieval, numerical_state,
                                 default_candidate_id, provenance)
    return apply_admission(report)
