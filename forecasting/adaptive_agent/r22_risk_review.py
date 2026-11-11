"""Deterministic fail-closed risk review boundary for R22."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from types import MappingProxyType
from collections.abc import Mapping


REPORT_SCHEMA = 'r22-structured-evidence-v1'
PACKET_SCHEMA = 'r22-risk-review-packet-v1'
DECISION_SCHEMA = 'r22-risk-review-decision-v1'

APPROVE_SELECTED = 'approve_selected'
REVERT_TO_IDENTITY = 'revert_to_identity'
REVIEW_ACTIONS = (APPROVE_SELECTED, REVERT_TO_IDENTITY)
PROVENANCE_FIELDS = frozenset({
    'context_id', 'region', 'origin_unix', 'cutoff_unix', 'memory_watermark',
    'cohort_id', 'retrieval_strategy', 'episode_ids', 'warehouse_fingerprint',
    'predictor_fingerprint', 'operation_admission_fingerprint',
    'skill_fingerprint', 'source_fingerprint', 'bootstrap_seed',
    'bootstrap_replicates', 'material_qs_loss_threshold',
})


@dataclass(frozen=True)
class RiskThresholds:
    """Frozen host-side limits; review output cannot alter these values.

    Coverage damage is a fraction. Width inflation is the candidate-minus-
    identity mean width90 delta in absolute target-price units.
    """

    minimum_common_support: int = 5
    maximum_material_loss_fraction: float = 0.25
    minimum_lower_tail_mean_qs_gain: float = 0.0
    maximum_coverage90_damage: float = 0.005
    maximum_width90_mean_inflation_target_price_units: float = 0.10
    minimum_current_median_absolute_difference: float = 0.0


RISK_THRESHOLDS = RiskThresholds()


_REASON_CODE_DETAILS = {
    'risk_invalid_schema': ('hard_veto', 'The report or packet does not satisfy the frozen schema.'),
    'risk_nonfinite_value': ('hard_veto', 'The report contains a nonfinite numeric value.'),
    'risk_provenance_mismatch': ('hard_veto', 'Report provenance differs from the origin-bound expectation.'),
    'risk_candidate_not_admitted': ('hard_veto', 'The selected candidate is not evidence-admitted.'),
    'risk_missing_operation_effect': ('hard_veto', 'The selected candidate has no operation-effect record.'),
    'risk_insufficient_common_support': ('hard_veto', 'Paired support is below the frozen minimum.'),
    'risk_material_downside': ('hard_veto', 'Historical downside exceeds a frozen limit.'),
    'risk_calibration_damage': ('hard_veto', 'Coverage damage exceeds the frozen limit.'),
    'risk_width_inflation': ('hard_veto', 'Interval-width inflation exceeds the frozen limit.'),
    'risk_data_quality_failure': ('hard_veto', 'Current cutoff-visible data quality is invalid.'),
    'risk_missing_current_disagreement': ('hard_veto', 'Current forecast disagreement is incomplete.'),
    'risk_zero_current_disagreement': ('hard_veto', 'The selected forecast does not differ from identity.'),
    'risk_reviewer_uncertain': ('reviewer', 'The reviewer cannot justify accepting current risk.'),
    'risk_reviewer_conflicting_evidence': ('reviewer', 'Risk evidence is internally conflicting.'),
    'risk_reviewer_state_misalignment': ('reviewer', 'The candidate is not aligned with the current state.'),
    'risk_review_timeout': ('runtime', 'The bounded review call timed out.'),
    'risk_review_skill_unavailable': ('runtime', 'The registered review skill is unavailable.'),
    'risk_review_budget_exhausted': ('runtime', 'The review budget is exhausted.'),
    'risk_review_exception': ('runtime', 'The review call raised an exception.'),
    'risk_malformed_review_output': ('runtime', 'The review output is malformed or unregistered.'),
    'risk_review_approved': ('approval', 'The selected candidate passed deterministic and reviewer checks.'),
}
REASON_CODE_REGISTRY = MappingProxyType({
    code: MappingProxyType({'category': category, 'description': description})
    for code, (category, description) in _REASON_CODE_DETAILS.items()
})

_HARD_VETO_CODES = frozenset(
    code for code, detail in REASON_CODE_REGISTRY.items() if detail['category'] == 'hard_veto')
_REVIEWER_REVERT_CODES = frozenset(
    code for code, detail in REASON_CODE_REGISTRY.items() if detail['category'] == 'reviewer')
_RUNTIME_REVERT_CODES = frozenset(
    code for code, detail in REASON_CODE_REGISTRY.items() if detail['category'] == 'runtime')
_DISAGREEMENT_FIELDS = (
    'median_absolute_difference', 'q05_absolute_difference',
    'q95_absolute_difference', 'width90_absolute_difference',
)


def registered_reason_codes():
    """Return a JSON-safe copy of the immutable reason-code registry."""
    return {code: dict(detail) for code, detail in REASON_CODE_REGISTRY.items()}


def _is_finite_number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _is_nonnegative_finite(value):
    return _is_finite_number(value) and value >= 0


def _contains_nonfinite(value):
    if isinstance(value, Real) and not isinstance(value, bool):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_contains_nonfinite(key) or _contains_nonfinite(item)
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    ).hexdigest()


def _add(vetoes, code):
    if code not in vetoes:
        vetoes.append(code)


def evaluate_hard_vetoes(report, selected_candidate_id, identity_candidate_id,
                         expected_provenance, thresholds=RISK_THRESHOLDS):
    """Return every deterministic veto in stable protocol order.

    This function never trusts report-provided limits and never raises for an
    invalid report. Any malformed or incomplete input therefore closes to the
    identity candidate through :func:`build_review_packet`.
    """
    vetoes = []
    if thresholds != RISK_THRESHOLDS or not isinstance(thresholds, RiskThresholds):
        return ['risk_invalid_schema']
    if (not isinstance(report, Mapping)
            or not isinstance(selected_candidate_id, str) or not selected_candidate_id
            or not isinstance(identity_candidate_id, str) or not identity_candidate_id
            or selected_candidate_id == identity_candidate_id
            or not isinstance(expected_provenance, Mapping)
            or set(expected_provenance) != PROVENANCE_FIELDS):
        return ['risk_invalid_schema']
    if _contains_nonfinite(report) or _contains_nonfinite(expected_provenance):
        _add(vetoes, 'risk_nonfinite_value')

    required = {'schema', 'provenance', 'current_regime', 'cohort_quality',
                'operation_effects', 'uncertainty_downside', 'eligible_candidate_ids'}
    if report.get('schema') != REPORT_SCHEMA or not required.issubset(report):
        _add(vetoes, 'risk_invalid_schema')

    provenance = report.get('provenance')
    if not isinstance(provenance, Mapping) or set(provenance) != PROVENANCE_FIELDS:
        _add(vetoes, 'risk_invalid_schema')
    elif any(provenance[key] != expected_provenance[key] for key in PROVENANCE_FIELDS):
        _add(vetoes, 'risk_provenance_mismatch')

    eligible = report.get('eligible_candidate_ids')
    if (not isinstance(eligible, (list, tuple))
            or not all(isinstance(value, str) and value for value in eligible)
            or len(eligible) != len(set(eligible))):
        _add(vetoes, 'risk_invalid_schema')
        eligible = ()
    if selected_candidate_id not in eligible:
        _add(vetoes, 'risk_candidate_not_admitted')

    effects = report.get('operation_effects')
    if not isinstance(effects, Mapping):
        _add(vetoes, 'risk_invalid_schema')
        effect = None
    else:
        effect = effects.get(selected_candidate_id)
    if not isinstance(effect, Mapping):
        _add(vetoes, 'risk_missing_operation_effect')
        effect = {}

    cohort = report.get('cohort_quality')
    cohort_support = cohort.get('common_paired_support') if isinstance(cohort, Mapping) else None
    if (not isinstance(cohort, Mapping)
            or cohort.get('data_quality_valid') is not True
            or cohort.get('coverage_grade') != 'sufficient'):
        _add(vetoes, 'risk_data_quality_failure')
    operation_support = effect.get('common_support')
    if (not isinstance(cohort_support, Integral) or isinstance(cohort_support, bool)
            or not isinstance(operation_support, Integral) or isinstance(operation_support, bool)):
        _add(vetoes, 'risk_invalid_schema')
        _add(vetoes, 'risk_insufficient_common_support')
    elif (cohort_support < thresholds.minimum_common_support
          or operation_support < thresholds.minimum_common_support
          or operation_support != cohort_support):
        _add(vetoes, 'risk_insufficient_common_support')

    downside = effect.get('downside')
    material_fraction = (downside.get('material_loss_fraction')
                         if isinstance(downside, Mapping) else None)
    lower_tail_gain = (downside.get('lower_tail_mean_qs_gain')
                       if isinstance(downside, Mapping) else None)
    if (not _is_nonnegative_finite(material_fraction)
            or material_fraction > 1
            or not _is_finite_number(lower_tail_gain)):
        _add(vetoes, 'risk_invalid_schema')
        _add(vetoes, 'risk_material_downside')
    elif (material_fraction > thresholds.maximum_material_loss_fraction
          or lower_tail_gain < thresholds.minimum_lower_tail_mean_qs_gain):
        _add(vetoes, 'risk_material_downside')

    coverage_summary = effect.get('coverage90_delta')
    coverage_delta = (coverage_summary.get('mean')
                      if isinstance(coverage_summary, Mapping) else None)
    summary_fields = {'mean', 'median', 'minimum', 'maximum', 'std'}
    if (not isinstance(coverage_summary, Mapping)
            or not summary_fields.issubset(coverage_summary)
            or not all(_is_finite_number(coverage_summary[key]) for key in summary_fields)):
        _add(vetoes, 'risk_invalid_schema')
        _add(vetoes, 'risk_calibration_damage')
    elif coverage_delta < -thresholds.maximum_coverage90_damage:
        _add(vetoes, 'risk_calibration_damage')

    width_summary = effect.get('width90_delta')
    width_delta = width_summary.get('mean') if isinstance(width_summary, Mapping) else None
    if (not isinstance(width_summary, Mapping)
            or not summary_fields.issubset(width_summary)
            or not all(_is_finite_number(width_summary[key]) for key in summary_fields)):
        _add(vetoes, 'risk_invalid_schema')
        _add(vetoes, 'risk_width_inflation')
    elif width_delta > thresholds.maximum_width90_mean_inflation_target_price_units:
        _add(vetoes, 'risk_width_inflation')

    current_regime = report.get('current_regime')
    data_quality = (current_regime.get('data_quality')
                    if isinstance(current_regime, Mapping) else None)
    if (not isinstance(data_quality, Mapping)
            or data_quality.get('valid') is not True
            or data_quality.get('critical_failure_count') != 0):
        _add(vetoes, 'risk_data_quality_failure')

    current_forecast = effect.get('current_forecast')
    disagreement = (current_forecast.get('current_disagreement')
                    if isinstance(current_forecast, Mapping) else None)
    forecast_ids_valid = (
        isinstance(current_forecast, Mapping)
        and current_forecast.get('identity_candidate_id') == identity_candidate_id
        and current_forecast.get('selected_candidate_id') == selected_candidate_id
    )
    if (not forecast_ids_valid or not isinstance(disagreement, Mapping)
            or not all(field in disagreement for field in _DISAGREEMENT_FIELDS)):
        _add(vetoes, 'risk_missing_current_disagreement')
    elif not all(_is_nonnegative_finite(disagreement[field])
                 for field in _DISAGREEMENT_FIELDS):
        _add(vetoes, 'risk_invalid_schema')
        _add(vetoes, 'risk_missing_current_disagreement')
    elif disagreement['median_absolute_difference'] <= \
            thresholds.minimum_current_median_absolute_difference:
        _add(vetoes, 'risk_zero_current_disagreement')
    return vetoes


def build_review_packet(report, selected_candidate_id, identity_candidate_id,
                        expected_provenance):
    """Build a bounded, JSON-safe packet and bind deterministic veto results."""
    vetoes = evaluate_hard_vetoes(
        report, selected_candidate_id, identity_candidate_id, expected_provenance)
    try:
        safe_report = _json_copy(report)
    except (TypeError, ValueError, OverflowError):
        safe_report = None
        _add(vetoes, 'risk_invalid_schema')
    try:
        safe_provenance = _json_copy(expected_provenance)
    except (TypeError, ValueError, OverflowError):
        safe_provenance = None
        _add(vetoes, 'risk_invalid_schema')
    packet = {
        'schema': PACKET_SCHEMA,
        'report_schema': report.get('schema') if isinstance(report, Mapping) else None,
        'selected_candidate_id': selected_candidate_id,
        'identity_candidate_id': identity_candidate_id,
        'expected_provenance': safe_provenance,
        'thresholds': asdict(RISK_THRESHOLDS),
        'reason_codes': registered_reason_codes(),
        'allowed_actions': list(REVIEW_ACTIONS),
        'hard_vetoes': vetoes,
        'review_allowed': not vetoes,
        'report_embedded': safe_report is not None and safe_provenance is not None,
        'report': safe_report,
    }
    packet['packet_sha256'] = _digest(packet)
    return packet


def validate_review_packet(packet):
    """Return stable validation reason codes for a stored or transported packet."""
    if not isinstance(packet, Mapping):
        return ['risk_invalid_schema']
    try:
        body = {key: value for key, value in packet.items() if key != 'packet_sha256'}
        expected_hash = _digest(body)
    except (TypeError, ValueError, OverflowError):
        return ['risk_invalid_schema']
    if (packet.get('schema') != PACKET_SCHEMA
            or packet.get('thresholds') != asdict(RISK_THRESHOLDS)
            or packet.get('allowed_actions') != list(REVIEW_ACTIONS)
            or packet.get('reason_codes') != registered_reason_codes()
            or packet.get('packet_sha256') != expected_hash):
        return ['risk_invalid_schema']
    vetoes = packet.get('hard_vetoes')
    if (not isinstance(vetoes, list) or len(vetoes) != len(set(vetoes))
            or any(code not in _HARD_VETO_CODES for code in vetoes)
            or packet.get('review_allowed') is not (not vetoes)):
        return ['risk_invalid_schema']
    if packet.get('report_embedded') is not True:
        return ([] if ('risk_invalid_schema' in vetoes
                       or 'risk_nonfinite_value' in vetoes)
                else ['risk_invalid_schema'])
    rebuilt = evaluate_hard_vetoes(
        packet.get('report'), packet.get('selected_candidate_id'),
        packet.get('identity_candidate_id'), packet.get('expected_provenance'))
    return [] if rebuilt == vetoes else ['risk_invalid_schema']


def _decision(packet, requested_action, action, reason_code):
    identity = packet.get('identity_candidate_id') if isinstance(packet, Mapping) else None
    selected = packet.get('selected_candidate_id') if isinstance(packet, Mapping) else None
    return {
        'schema': DECISION_SCHEMA,
        'requested_action': requested_action,
        'action': action,
        'reason_code': reason_code,
        'selected_candidate_id': selected,
        'identity_candidate_id': identity,
        'final_candidate_id': selected if action == APPROVE_SELECTED else identity,
        'approved': action == APPROVE_SELECTED,
        'hard_vetoes': list(packet.get('hard_vetoes', []))
                         if isinstance(packet, Mapping) else [],
        'packet_sha256': packet.get('packet_sha256') if isinstance(packet, Mapping) else None,
    }


def fail_closed_decision(packet, reason_code):
    """Map a registered runtime failure to identity without consulting the reviewer."""
    packet_errors = validate_review_packet(packet)
    if packet_errors:
        return _decision(packet, None, REVERT_TO_IDENTITY, packet_errors[0])
    hard_vetoes = packet['hard_vetoes']
    if hard_vetoes:
        return _decision(packet, None, REVERT_TO_IDENTITY, hard_vetoes[0])
    if reason_code not in _RUNTIME_REVERT_CODES:
        reason_code = 'risk_malformed_review_output'
    return _decision(packet, None, REVERT_TO_IDENTITY, reason_code)


def validate_review_action(packet, response):
    """Validate the bounded reviewer response and return the final host decision."""
    packet_errors = validate_review_packet(packet)
    requested = response.get('action') if isinstance(response, Mapping) else None
    if packet_errors:
        return _decision(packet, requested, REVERT_TO_IDENTITY, packet_errors[0])
    if packet['hard_vetoes']:
        return _decision(packet, requested, REVERT_TO_IDENTITY, packet['hard_vetoes'][0])
    if (not isinstance(response, Mapping)
            or set(response) - {'action', 'reason_code'}
            or requested not in REVIEW_ACTIONS):
        return _decision(packet, requested, REVERT_TO_IDENTITY,
                         'risk_malformed_review_output')
    reason_code = response.get('reason_code')
    if requested == APPROVE_SELECTED:
        if reason_code not in (None, 'risk_review_approved'):
            return _decision(packet, requested, REVERT_TO_IDENTITY,
                             'risk_malformed_review_output')
        return _decision(packet, requested, APPROVE_SELECTED, 'risk_review_approved')
    if reason_code not in _REVIEWER_REVERT_CODES:
        return _decision(packet, requested, REVERT_TO_IDENTITY,
                         'risk_malformed_review_output')
    return _decision(packet, requested, REVERT_TO_IDENTITY, reason_code)
