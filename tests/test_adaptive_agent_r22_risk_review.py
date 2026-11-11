import math
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict

from forecasting.adaptive_agent.r22_risk_review import (
    APPROVE_SELECTED,
    REVERT_TO_IDENTITY,
    REASON_CODE_REGISTRY,
    RISK_THRESHOLDS,
    build_review_packet,
    evaluate_hard_vetoes,
    fail_closed_decision,
    registered_reason_codes,
    validate_review_action,
    validate_review_packet,
)


IDENTITY = 'lightgbm-input:identity'
SELECTED = 'lightgbm-input:history_demand_last48_mean'
PROVENANCE = {
    'context_id': 'context', 'region': 'TAS1', 'origin_unix': 1_700_000_000,
    'cutoff_unix': 1_700_000_000, 'memory_watermark': 1_699_999_999,
    'cohort_id': 'cohort', 'retrieval_strategy': 'state',
    'episode_ids': ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8'],
    'warehouse_fingerprint': 'warehouse-sha256',
    'predictor_fingerprint': 'predictor-sha256',
    'operation_admission_fingerprint': 'admission-sha256',
    'skill_fingerprint': 'skills-sha256', 'source_fingerprint': 'source-sha256',
    'bootstrap_seed': 20261111, 'bootstrap_replicates': 2000,
    'material_qs_loss_threshold': 0.01,
}


def summary(mean):
    return {'mean': mean, 'median': mean, 'minimum': mean,
            'maximum': mean, 'std': 0.0}


def valid_report():
    return {
        'schema': 'r22-structured-evidence-v1',
        'provenance': dict(PROVENANCE),
        'current_regime': {
            'data_quality': {'valid': True, 'critical_failure_count': 0},
        },
        'cohort_quality': {
            'common_paired_support': 8, 'coverage_grade': 'sufficient',
            'data_quality_valid': True,
        },
        'uncertainty_downside': {'available': True},
        'eligible_candidate_ids': [SELECTED],
        'operation_effects': {
            SELECTED: {
                'common_support': 8,
                'coverage90_delta': summary(-0.001),
                'width90_delta': summary(0.02),
                'downside': {
                    'material_loss_fraction': 0.125,
                    'lower_tail_mean_qs_gain': 0.01,
                },
                'current_forecast': {
                    'identity_candidate_id': IDENTITY,
                    'selected_candidate_id': SELECTED,
                    'current_disagreement': {
                        'median_absolute_difference': 0.4,
                        'q05_absolute_difference': 0.3,
                        'q95_absolute_difference': 0.5,
                        'width90_absolute_difference': 0.2,
                    },
                },
            },
        },
    }


def packet(report=None):
    return build_review_packet(
        valid_report() if report is None else report, SELECTED, IDENTITY, PROVENANCE)


class R22RiskReviewTests(unittest.TestCase):
    def test_thresholds_and_reason_registry_are_immutable_and_serializable(self):
        with self.assertRaises(FrozenInstanceError):
            RISK_THRESHOLDS.minimum_common_support = 1
        with self.assertRaises(TypeError):
            REASON_CODE_REGISTRY['new'] = {}
        self.assertEqual(asdict(RISK_THRESHOLDS)['minimum_common_support'], 5)
        self.assertEqual(
            asdict(RISK_THRESHOLDS)[
                'maximum_width90_mean_inflation_target_price_units'],
            0.10)
        self.assertEqual(registered_reason_codes()['risk_material_downside']['category'],
                         'hard_veto')

    def test_builds_and_validates_bound_packet(self):
        result = packet()
        self.assertTrue(result['review_allowed'])
        self.assertEqual(result['hard_vetoes'], [])
        self.assertEqual(validate_review_packet(result), [])
        changed = deepcopy(result)
        changed['thresholds']['minimum_common_support'] = 1
        self.assertEqual(validate_review_packet(changed), ['risk_invalid_schema'])

    def test_schema_and_nonfinite_values_fail_closed(self):
        report = valid_report()
        report['schema'] = 'unregistered'
        report['operation_effects'][SELECTED]['width90_delta']['mean'] = math.nan
        result = packet(report)
        self.assertFalse(result['review_allowed'])
        self.assertIn('risk_invalid_schema', result['hard_vetoes'])
        self.assertIn('risk_nonfinite_value', result['hard_vetoes'])
        self.assertIsNone(result['report'])
        self.assertEqual(validate_review_packet(result), [])
        decision = validate_review_action(result, {'action': APPROVE_SELECTED})
        self.assertEqual(decision['action'], REVERT_TO_IDENTITY)
        self.assertEqual(decision['reason_code'], 'risk_nonfinite_value')

    def test_provenance_mismatch_fails_closed(self):
        report = valid_report()
        report['provenance']['context_id'] = 'wrong'
        self.assertIn('risk_provenance_mismatch', packet(report)['hard_vetoes'])

    def test_candidate_must_be_admitted_and_have_effect(self):
        report = valid_report()
        report['eligible_candidate_ids'] = []
        report['operation_effects'] = {}
        vetoes = packet(report)['hard_vetoes']
        self.assertIn('risk_candidate_not_admitted', vetoes)
        self.assertIn('risk_missing_operation_effect', vetoes)

    def test_support_is_paired_sufficient_and_consistent(self):
        report = valid_report()
        report['operation_effects'][SELECTED]['common_support'] = 4
        self.assertIn('risk_insufficient_common_support', packet(report)['hard_vetoes'])
        report['operation_effects'][SELECTED]['common_support'] = 7
        self.assertIn('risk_insufficient_common_support', packet(report)['hard_vetoes'])

    def test_material_downside_limits_are_hard_vetoes(self):
        for field, value in (
                ('material_loss_fraction', 0.251),
                ('lower_tail_mean_qs_gain', -0.001)):
            with self.subTest(field=field):
                report = valid_report()
                report['operation_effects'][SELECTED]['downside'][field] = value
                self.assertIn('risk_material_downside', packet(report)['hard_vetoes'])

    def test_calibration_and_width_limits_are_hard_vetoes(self):
        report = valid_report()
        report['operation_effects'][SELECTED]['coverage90_delta']['mean'] = -0.006
        report['operation_effects'][SELECTED]['width90_delta']['mean'] = 0.11
        vetoes = packet(report)['hard_vetoes']
        self.assertIn('risk_calibration_damage', vetoes)
        self.assertIn('risk_width_inflation', vetoes)
        report = valid_report()
        report['operation_effects'][SELECTED]['coverage90_delta']['mean'] = -0.005
        report['operation_effects'][SELECTED]['width90_delta']['mean'] = 0.10
        vetoes = packet(report)['hard_vetoes']
        self.assertNotIn('risk_calibration_damage', vetoes)
        self.assertNotIn('risk_width_inflation', vetoes)

    def test_current_data_quality_failure_is_vetoed(self):
        report = valid_report()
        report['current_regime']['data_quality']['valid'] = False
        self.assertIn('risk_data_quality_failure', packet(report)['hard_vetoes'])

    def test_current_disagreement_must_be_complete_and_nonzero(self):
        report = valid_report()
        del report['operation_effects'][SELECTED]['current_forecast'][
            'current_disagreement'][
            'q95_absolute_difference']
        self.assertIn('risk_missing_current_disagreement', packet(report)['hard_vetoes'])
        report = valid_report()
        report['operation_effects'][SELECTED]['current_forecast'][
            'current_disagreement'][
            'median_absolute_difference'] = 0.0
        self.assertIn('risk_zero_current_disagreement', packet(report)['hard_vetoes'])

    def test_evaluate_never_accepts_caller_modified_thresholds(self):
        from forecasting.adaptive_agent.r22_risk_review import RiskThresholds

        vetoes = evaluate_hard_vetoes(
            valid_report(), SELECTED, IDENTITY, PROVENANCE,
            RiskThresholds(minimum_common_support=1))
        self.assertEqual(vetoes, ['risk_invalid_schema'])

    def test_approve_and_registered_revert_are_the_only_review_actions(self):
        review_packet = packet()
        approved = validate_review_action(
            review_packet, {'action': APPROVE_SELECTED})
        self.assertTrue(approved['approved'])
        self.assertEqual(approved['final_candidate_id'], SELECTED)
        reverted = validate_review_action(review_packet, {
            'action': REVERT_TO_IDENTITY,
            'reason_code': 'risk_reviewer_conflicting_evidence',
        })
        self.assertFalse(reverted['approved'])
        self.assertEqual(reverted['final_candidate_id'], IDENTITY)

    def test_malformed_or_candidate_switching_output_reverts(self):
        for response in (
                {'action': 'approve_another'},
                {'action': APPROVE_SELECTED, 'candidate_id': 'other'},
                {'action': REVERT_TO_IDENTITY, 'reason_code': 'invented_reason'},
                None):
            with self.subTest(response=response):
                result = validate_review_action(packet(), response)
                self.assertEqual(result['action'], REVERT_TO_IDENTITY)
                self.assertEqual(result['reason_code'], 'risk_malformed_review_output')
                self.assertEqual(result['final_candidate_id'], IDENTITY)

    def test_hard_veto_cannot_be_overridden_by_approval(self):
        report = valid_report()
        report['current_regime']['data_quality']['valid'] = False
        result = validate_review_action(packet(report), {'action': APPROVE_SELECTED})
        self.assertEqual(result['action'], REVERT_TO_IDENTITY)
        self.assertEqual(result['reason_code'], 'risk_data_quality_failure')

    def test_runtime_failures_are_registered_and_fail_closed(self):
        for reason in ('risk_review_timeout', 'risk_review_skill_unavailable',
                       'risk_review_budget_exhausted', 'risk_review_exception'):
            with self.subTest(reason=reason):
                result = fail_closed_decision(packet(), reason)
                self.assertEqual(result['final_candidate_id'], IDENTITY)
                self.assertEqual(result['reason_code'], reason)


if __name__ == '__main__':
    unittest.main()
