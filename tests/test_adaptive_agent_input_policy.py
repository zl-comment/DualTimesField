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

    def __init__(self, *, eligible=4, diagnostic_failures=False):
        self.eligible = eligible
        self.diagnostic_failures = diagnostic_failures
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
            result['retrieve_analogs:recent'] = {'description': 'recent', 'costs': {'episodes': 4}}
        alternate = 'lightgbm-input:history_demand_last48_mean'
        if alternate not in self.artifacts:
            result['forecast:' + alternate] = {'description': 'registered mean counterfactual',
                                                'costs': {'candidates': 1}}
        if self.locked and len(self.artifacts) > 1:
            result['diagnose:' + '|'.join(self.artifacts)] = {
                'description': 'paired',
                'costs': {'historical_evaluations': 4 * len(self.artifacts)}}
        return result

    def execute(self, ctx, action, state):
        if action == 'retrieve_analogs:recent':
            self.locked = True
            return {'status': 'available', 'candidate_ids': [], 'costs': {'episodes': 4},
                    'observation': {'items': [{'episode_id': 'e' + str(i)} for i in range(4)]}}
        if action.startswith('forecast:'):
            candidate = action[len('forecast:'):]
            self.artifacts[candidate] = forecast(ctx, candidate, 2)
            return {'status': 'ok', 'candidate_ids': [candidate], 'costs': {'candidates': 1},
                    'observation': {'candidate': candidate}}
        if action.startswith('diagnose:'):
            ids = action[len('diagnose:'):].split('|')
            episodes = ['e' + str(i) for i in range(4)]
            rows = {key: {'paired_episode_ids': episodes, 'sample_count': 4} for key in ids}
            failures = ([{'candidate_id': ids[-1], 'episode_id': episodes[-1]}]
                        if self.diagnostic_failures else [])
            return {'status': 'available', 'candidate_ids': [],
                    'costs': {'historical_evaluations': 4 * len(ids)},
                    'observation': {'episodes': episodes, 'paired_episode_ids': episodes,
                                    'sample_count': 4, 'cohort_id': 'locked',
                                    'candidates': rows, 'failures': failures}}
        raise ValueError(action)


class InputPolicyTests(unittest.TestCase):
    def test_stage_a_and_stage_b_are_separate_and_candidate_is_diagnosed(self):
        client = Client(['retrieve_analogs:recent', 'experiment',
                         'forecast:lightgbm-input:history_demand_last48_mean',
                         'commit_now', 'commit:lightgbm-input:history_demand_last48_mean'])
        selected, trace = run(context(), Host(), client)
        self.assertEqual(selected.candidate_id, 'lightgbm-input:history_demand_last48_mean')
        self.assertEqual([row['action'] for row in trace['stage_a_decisions']],
                         ['experiment', 'commit_now'])
        self.assertEqual([row['action'] for row in trace['stage_b_decisions']],
                         ['forecast:lightgbm-input:history_demand_last48_mean'])
        self.assertEqual(len(trace['iterations']), 1)
        self.assertIn(selected.candidate_id, trace['iterations'][0]['candidate_ids'])

    def test_no_mature_cohort_submits_identity_without_api_call(self):
        client = Client([])
        selected, trace = run(context(), Host(eligible=0), client)
        self.assertEqual(selected.candidate_id, 'lightgbm-input:identity')
        self.assertEqual(client.calls, [])
        self.assertEqual(trace['stop_reason'], 'identity_no_mature_cohort')

    def test_diagnostic_failure_prevents_nonidentity_commit(self):
        client = Client(['retrieve_analogs:recent', 'experiment',
                         'forecast:lightgbm-input:history_demand_last48_mean', 'commit_now'])
        selected, trace = run(context(), Host(diagnostic_failures=True), client)
        self.assertEqual(selected.candidate_id, 'lightgbm-input:identity')
        self.assertEqual(trace['stop_reason'], 'identity_insufficient_paired_support')


if __name__ == '__main__':
    unittest.main()
