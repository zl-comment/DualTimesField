import unittest

import numpy as np

from forecasting.api_agent.workflow import Snapshot
from forecasting.adaptive_agent.contracts import ForecastArtifact, ForecastContext
from forecasting.adaptive_agent.input_policy_r22 import run
from forecasting.adaptive_agent.input_registry import InputForecastSpec
from forecasting.adaptive_agent.state import summarize_state


DEFAULT = 'lightgbm-input:identity'
OPERATED = 'lightgbm-input:history_demand_last48_mean'


def context():
    origin = 1_700_000_000
    snapshot = Snapshot('NSW1', origin, origin + 3600 * np.arange(24),
                        np.linspace(10, 30, 72), np.linspace(-1, 1, 72),
                        np.ones((72, 10)), np.ones((24, 10)), 'preprocessing')
    return ForecastContext(snapshot, {}, {'region': 'NSW1'})


def forecast(ctx, candidate_id, value):
    quantiles = np.tile(np.asarray([value - 2, value - 1, value, value + 1, value + 2]),
                        (24, 1))
    return ForecastArtifact(candidate_id, ctx.identity(), quantiles,
                            ctx.snapshot.target_time_unix, 'lightgbm',
                            candidate_id + '-input', {'candidate': candidate_id})


def summary(mean=0.0):
    return {'mean': mean, 'median': mean, 'minimum': mean, 'maximum': mean, 'std': 0.0}


class Client:
    timeout = 60

    def __init__(self, actions, fail_review=False):
        self.actions = list(actions)
        self.fail_review = fail_review
        self.calls = []

    def __copy__(self):
        return self

    def choose(self, state, instruction, criteria):
        self.calls.append((state['phase'], tuple(criteria), state))
        if self.fail_review and state['phase'] == 'independent_risk_review':
            raise TimeoutError('review timeout')
        action = self.actions.pop(0)
        return action, {'chosen': action}


class Registry:
    release_id = 'r22-test'
    default_id = DEFAULT
    model_identities = {'lightgbm': {'sha256': 'frozen-model'}}
    specs = {
        DEFAULT: InputForecastSpec('lightgbm', 'identity', 'identity-sha'),
        OPERATED: InputForecastSpec('lightgbm', 'history_demand_last48_mean', 'operation-sha'),
    }


class Host:
    def __init__(self, *, beneficial=True, width_delta=0.0):
        self.registry = Registry()
        self.beneficial = beneficial
        self.width_delta = width_delta
        self.artifacts = {}
        self.locked = False

    def default(self, ctx):
        self.artifacts[DEFAULT] = forecast(ctx, DEFAULT, 10)
        return self.artifacts[DEFAULT]

    def artifact(self, ctx, candidate_id):
        return self.artifacts[candidate_id]

    def initial_observation(self, ctx):
        return {'eligible_analog_count': 8, 'registered_models': ['lightgbm'],
                'numerical_state': summarize_state(ctx)}

    def actions(self, ctx, state):
        result = {}
        if not self.locked:
            result['retrieve_analogs:recent'] = {
                'description': 'recent', 'costs': {'episodes': 8}}
        if OPERATED not in self.artifacts:
            result['forecast:' + OPERATED] = {
                'description': 'registered mean counterfactual',
                'costs': {'candidates': 1}}
        if self.locked and len(self.artifacts) > 1:
            result['diagnose:' + '|'.join(self.artifacts)] = {
                'description': 'paired', 'costs': {'historical_evaluations': 16}}
        return result

    def execute(self, ctx, action, state):
        if action == 'retrieve_analogs:recent':
            self.locked = True
            items = [{
                'episode_id': 'e' + str(index),
                'origin_unix': ctx.origin_unix - (index + 2) * 86400,
                'outcome_available_at': ctx.origin_unix - (index + 1) * 86400,
                'state_tags': ['stable'], 'distance': 0.1 + index / 10,
                'metrics': {}, 'data_role': 'model_selection_exposed',
            } for index in range(8)]
            return {'status': 'available', 'candidate_ids': [],
                    'costs': {'episodes': 8},
                    'observation': {'status': 'available', 'strategy': 'recent',
                                    'items': items, 'support_count': 8,
                                    'eligible_count': 8, 'low_support': False,
                                    'warehouse_fingerprint': 'warehouse',
                                    'cohort_id': 'locked', 'cohort_locked': True,
                                    'cohort_size': 8}}
        if action.startswith('forecast:'):
            candidate_id = action[len('forecast:'):]
            self.artifacts[candidate_id] = forecast(ctx, candidate_id, 11)
            return {'status': 'ok', 'candidate_ids': [candidate_id],
                    'costs': {'candidates': 1},
                    'observation': {'candidate': candidate_id}}
        if action.startswith('diagnose:'):
            episodes = ['e' + str(index) for index in range(8)]
            gain = 0.5 if self.beneficial else -0.5
            horizon = {key: summary(gain) for key in
                       ('hours_1_6', 'hours_7_12', 'hours_13_18', 'hours_19_24')}
            paired = {
                'default_candidate_id': DEFAULT,
                'gain_definition': 'default_qs_minus_candidate_qs',
                'mean_qs_gain': gain, 'median_qs_gain': gain,
                'minimum_qs_gain': gain, 'maximum_qs_gain': gain,
                'qs_gain_std': 0.0, 'win_count': 8 if gain > 0 else 0,
                'tie_count': 0, 'loss_count': 0 if gain > 0 else 8,
                'win_fraction': 1.0 if gain > 0 else 0.0,
                'common_support': 8, 'horizon_qs_gain': horizon,
                'mae_gain': summary(gain), 'coverage90_delta': summary(0.0),
                'width90_delta': summary(self.width_delta),
                'calibration_damage': summary(0.0),
                'downside': {
                    'worst_paired_qs_loss': 0.0,
                    'lower_tail_mean_qs_gain': max(0.0, gain),
                    'loss_fraction': 0.0 if gain > 0 else 1.0,
                    'material_loss_fraction': 0.0 if gain > 0 else 1.0,
                    'material_qs_loss_threshold': 0.01,
                },
                'similarity_weighted': {
                    'available': True, 'mean_qs_gain': gain,
                    'effective_support': 8.0,
                    'sign_agreement_with_unweighted': True,
                },
                'bootstrap_mean_qs_gain': {
                    'role': 'small_sample_supporting_evidence_not_significance_test',
                    'seed': 20261111, 'replicates': 2000,
                    'percentiles': [2.5, 97.5], 'interval': [gain, gain],
                },
                'leave_one_episode_out': {
                    'minimum_mean_qs_gain': gain,
                    'maximum_mean_qs_gain': gain, 'sign_stable': True,
                },
            }
            candidates = {
                DEFAULT: {'sample_count': 8, 'paired_episode_ids': episodes,
                          'metrics': {'qs': 10.0}},
                OPERATED: {'sample_count': 8, 'paired_episode_ids': episodes,
                           'metrics': {'qs': 10.0 - gain},
                           'paired_vs_default': paired},
            }
            comparison = {
                'candidates': {
                    DEFAULT: {'q05_mean': 8.0, 'median_mean': 10.0,
                              'q95_mean': 12.0, 'width90_mean': 4.0},
                    OPERATED: {'q05_mean': 9.0, 'median_mean': 11.0,
                               'q95_mean': 13.0, 'width90_mean': 4.0},
                },
                'pairs': [{
                    'a': DEFAULT, 'b': OPERATED,
                    'median_absolute_difference': 1.0,
                    'q05_absolute_difference': 1.0,
                    'q95_absolute_difference': 1.0,
                    'width90_absolute_difference': 0.0,
                }],
            }
            return {'status': 'available', 'candidate_ids': [],
                    'costs': {'historical_evaluations': 16},
                    'observation': {
                        'status': 'available', 'r22_aggregate_schema': 'r22-paired-diagnostic-v1',
                        'episodes': episodes, 'paired_episode_ids': episodes,
                        'sample_count': 8, 'paired_support_fraction': 1.0,
                        'cohort_id': 'locked', 'cohort_locked': True,
                        'cohort_size': 8, 'candidates': candidates,
                        'failures': [], 'unattempted_count': 0,
                        'current_forecast_comparison': comparison,
                    }}
        raise ValueError(action)


class R22InputPolicyTests(unittest.TestCase):
    def test_stage_c_approves_selected_operation(self):
        client = Client(['retrieve_analogs:recent', 'use_operated_input',
                         'commit:' + OPERATED, 'approve_selected'])
        selected, trace = run(context(), Host(), client)
        self.assertEqual(selected.candidate_id, OPERATED)
        self.assertEqual(trace['final_commit_reason'], 'risk_review_approve')
        self.assertEqual(trace['stage_c_decisions'][0]['action'], 'approve_selected')
        review_state = client.calls[-1][2]
        self.assertEqual([row['action'] for row in review_state['observations']],
                         ['risk_review_packet'])
        self.assertEqual(review_state['action_history'], [])

    def test_stage_c_revert_commits_identity(self):
        client = Client(['retrieve_analogs:recent', 'use_operated_input',
                         'commit:' + OPERATED, 'revert_to_identity'])
        selected, trace = run(context(), Host(), client)
        self.assertEqual(selected.candidate_id, DEFAULT)
        self.assertEqual(trace['final_commit_reason'], 'risk_review_revert')
        self.assertEqual(trace['risk_review_decision']['reason_code'],
                         'risk_reviewer_uncertain')

    def test_hard_veto_skips_stage_c_and_reverts(self):
        client = Client(['retrieve_analogs:recent', 'use_operated_input',
                         'commit:' + OPERATED])
        selected, trace = run(context(), Host(width_delta=0.2), client)
        self.assertEqual(selected.candidate_id, DEFAULT)
        self.assertEqual(trace['stage_c_decisions'], [])
        self.assertIn('risk_width_inflation', trace['risk_vetoes'])
        self.assertEqual(trace['final_commit_reason'],
                         'hard_veto:risk_width_inflation')

    def test_review_timeout_fails_closed(self):
        client = Client(['retrieve_analogs:recent', 'use_operated_input',
                         'commit:' + OPERATED], fail_review=True)
        selected, trace = run(context(), Host(), client)
        self.assertEqual(selected.candidate_id, DEFAULT)
        self.assertEqual(trace['risk_review_decision']['reason_code'],
                         'risk_review_timeout')

    def test_rejected_report_commits_identity_without_selection_calls(self):
        client = Client(['retrieve_analogs:recent'])
        selected, trace = run(context(), Host(beneficial=False), client)
        self.assertEqual(selected.candidate_id, DEFAULT)
        self.assertEqual(trace['stop_reason'],
                         'identity_no_operation_passed_structured_evidence')
        self.assertEqual([row[0] for row in client.calls],
                         ['select_historical_cohort'])


if __name__ == '__main__':
    unittest.main()
