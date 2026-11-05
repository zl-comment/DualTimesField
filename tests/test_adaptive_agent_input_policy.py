import unittest

import numpy as np

from forecasting.api_agent.workflow import Snapshot
from forecasting.adaptive_agent.contracts import ForecastArtifact, ForecastContext
from forecasting.adaptive_agent.input_policy import run


def context():
    origin = 1_700_000_000
    snapshot = Snapshot('NSW1', origin, origin + 3600 * np.arange(24),
                        np.ones(72), np.ones(72), np.ones((72, 10)),
                        np.ones((24, 10)), 'preprocessing')
    return ForecastContext(snapshot, {}, {'region': 'NSW1'})


def forecast(ctx, candidate_id, value):
    return ForecastArtifact(candidate_id, ctx.identity(), np.full((24, 5), value),
                            ctx.snapshot.target_time_unix, 'lightgbm', candidate_id + '-input',
                            {'candidate': candidate_id})


class Client:
    timeout = 60

    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def __copy__(self):
        return self

    def choose(self, state, instruction, criteria):
        self.calls.append((state['phase'], tuple(criteria)))
        action = self.actions.pop(0)
        return action, {'chosen': action}


class Host:
    default_id = 'lightgbm-input:identity'

    def __init__(self, *, eligible=8, diagnostic_failures=False, beneficial=True):
        self.eligible = eligible
        self.diagnostic_failures = diagnostic_failures
        self.beneficial = beneficial
        self.artifacts = {}
        self.locked = False
        self.audit = []

    def default(self, ctx):
        self.artifacts[self.default_id] = forecast(ctx, self.default_id, 1)
        return self.artifacts[self.default_id]

    def artifact(self, ctx, candidate_id):
        return self.artifacts[candidate_id]

    def initial_observation(self, ctx):
        return {'eligible_analog_count': self.eligible, 'registered_models': ['lightgbm']}

    def actions(self, ctx, state):
        result = {'inspect_state': {'description': 'state', 'costs': {}}}
        if not self.locked:
            result['retrieve_analogs:recent'] = {'description': 'recent', 'costs': {'episodes': self.eligible}}
        alternate = 'lightgbm-input:history_demand_last48_mean'
        if alternate not in self.artifacts:
            result['forecast:' + alternate] = {'description': 'registered mean counterfactual',
                                                'costs': {'candidates': 1}}
        if self.locked and len(self.artifacts) > 1:
            result['diagnose:' + '|'.join(self.artifacts)] = {
                'description': 'paired',
                'costs': {'historical_evaluations': self.eligible * len(self.artifacts)}}
        return result

    def execute(self, ctx, action, state):
        if action == 'retrieve_analogs:recent':
            self.locked = True
            return {'status': 'available', 'candidate_ids': [], 'costs': {'episodes': self.eligible},
                    'observation': {'items': [{'episode_id': 'e' + str(i)}
                                              for i in range(self.eligible)]}}
        if action.startswith('forecast:'):
            candidate = action[len('forecast:'):]
            self.artifacts[candidate] = forecast(ctx, candidate, 2)
            return {'status': 'ok', 'candidate_ids': [candidate], 'costs': {'candidates': 1},
                    'observation': {'candidate': candidate}}
        if action.startswith('diagnose:'):
            ids = action[len('diagnose:'):].split('|')
            episodes = ['e' + str(i) for i in range(self.eligible)]
            rows = {}
            for key in ids:
                identity = key == self.default_id
                gain = 0.5 if self.beneficial else -0.5
                rows[key] = {
                    'paired_episode_ids': episodes, 'sample_count': self.eligible,
                    'metrics': {'qs': 10 if identity else 10 - gain,
                                'qs_first12': 5 if identity else 5 - gain / 2,
                                'qs_last12': 5 if identity else 5 - gain / 2},
                }
                if not identity:
                    rows[key]['paired_vs_default'] = {
                        'mean_qs_gain': gain, 'median_qs_gain': gain,
                        'win_fraction': 1.0 if gain > 0 else 0.0,
                    }
            failures = ([{'candidate_id': ids[-1], 'episode_id': episodes[-1]}]
                        if self.diagnostic_failures else [])
            return {'status': 'available', 'candidate_ids': [],
                    'costs': {'historical_evaluations': self.eligible * len(ids)},
                    'observation': {'episodes': episodes, 'paired_episode_ids': episodes,
                                    'sample_count': self.eligible, 'cohort_id': 'locked',
                                    'candidates': rows, 'failures': failures,
                                    'current_forecast_comparison': {'pairs': [{
                                        'a': self.default_id, 'b': ids[-1],
                                        'median_absolute_difference': 1.0}]}}}
        raise ValueError(action)


class InputPolicyTests(unittest.TestCase):
    def test_stage_a_and_stage_b_are_separate_and_candidate_is_diagnosed(self):
        client = Client(['retrieve_analogs:recent', 'use_operated_input',
                         'commit:lightgbm-input:history_demand_last48_mean'])
        selected, trace = run(context(), Host(), client)
        self.assertEqual(selected.candidate_id, 'lightgbm-input:history_demand_last48_mean')
        self.assertEqual([row['action'] for row in trace['stage_a_decisions']],
                         ['use_operated_input'])
        self.assertEqual([row['action'] for row in trace['stage_b_decisions']],
                         ['commit:lightgbm-input:history_demand_last48_mean'])
        self.assertEqual(len(trace['iterations']), 1)
        self.assertIn(selected.candidate_id, trace['iterations'][0]['candidate_ids'])

    def test_no_mature_cohort_submits_identity_without_api_call(self):
        client = Client([])
        selected, trace = run(context(), Host(eligible=0), client)
        self.assertEqual(selected.candidate_id, 'lightgbm-input:identity')
        self.assertEqual(client.calls, [])
        self.assertEqual(trace['stop_reason'], 'identity_no_mature_cohort')

    def test_diagnostic_failure_prevents_nonidentity_commit(self):
        client = Client(['retrieve_analogs:recent'])
        selected, trace = run(context(), Host(diagnostic_failures=True), client)
        self.assertEqual(selected.candidate_id, 'lightgbm-input:identity')
        self.assertEqual(trace['stop_reason'], 'identity_insufficient_paired_support')

    def test_deterministic_gate_rejects_historically_harmful_operation(self):
        client = Client(['retrieve_analogs:recent'])
        selected, trace = run(context(), Host(beneficial=False), client)
        self.assertEqual(selected.candidate_id, 'lightgbm-input:identity')
        self.assertEqual(trace['stop_reason'], 'identity_no_operation_passed_evidence_gate')
        self.assertEqual(len(client.calls), 1)
        operation = trace['decision_admission']['operations'][
            'lightgbm-input:history_demand_last48_mean']
        self.assertFalse(operation['eligible'])
        self.assertIn('positive_mean_qs_gain', operation['rejection_reasons'])


if __name__ == '__main__':
    unittest.main()
