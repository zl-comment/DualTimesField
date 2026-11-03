"""Controller behavioral tests: evidence, genuine choices, budgets and audit."""
import dataclasses
import time
import unittest

import numpy as np

from forecasting.api_agent.workflow import Snapshot
from forecasting.adaptive_agent.contracts import BudgetLimits, ForecastArtifact, ForecastContext
from forecasting.adaptive_agent.controller import load_skill, load_skills, run


def context():
    origin = 1660000000
    snap = Snapshot('NSW1', origin, origin + np.arange(24) * 3600,
                    np.arange(72.), np.ones(72), np.ones((72, 10)),
                    np.ones((24, 10)), 'synthetic-fixture')
    return ForecastContext(snap)


def artifact(ctx, name='default', shift=0):
    return ForecastArtifact(name, ctx.identity(), np.tile(np.arange(5.), (24, 1)) + shift,
                            ctx.snapshot.target_time_unix, 'synthetic-frozen', 'input-' + name)


class Host:
    def __init__(self, ctx, evidence='ordinary'):
        self.artifacts = {'default': artifact(ctx), 'alternative': artifact(ctx, 'alternative', 10)}
        self.evidence = evidence
        self.executions = []
        self.specs = {
            'inspect_state': {'description': 'Inspect state'},
            'search_evidence': {'description': 'Search time-valid evidence'},
            'read_evidence:notice-1': {'description': 'Read notice'},
            'resolve_event:notice-1:outage': {'description': 'Classify quoted notice'},
            'retrieve_analogs:diverse': {'description': 'Retrieve diverse mature episodes', 'costs': {'episodes': 2}},
            'request_forecast:alternative': {'description': 'Request frozen alternative', 'costs': {'candidates': 1}},
            'compare_candidates': {'description': 'Compare on mature history', 'costs': {'historical_evaluations': 4}},
            'commit_pool:half': {'description': 'Materialize registered pool', 'costs': {'candidates': 1}},
        }

    def default(self, ctx):
        return self.artifacts['default']

    def actions(self, ctx, state):
        return self.specs

    def execute(self, ctx, action, state):
        self.executions.append(action)
        result = {'status': 'ok', 'observation': {'evidence': self.evidence, 'source_id': 'notice-1'},
                  'costs': self.specs[action].get('costs', {})}
        if action in ('request_forecast:alternative', 'commit_pool:half'):
            result['candidate_ids'] = ['alternative']
        return result

    def artifact(self, ctx, candidate_id):
        return self.artifacts[candidate_id]


class ScriptedClient:
    timeout = 999

    def __init__(self, actions):
        self.actions = iter(actions)
        self.requests = []

    def choose(self, state, instructions, criteria):
        self.requests.append((state, instructions, criteria, self.timeout))
        action = next(self.actions)
        if isinstance(action, Exception):
            raise action
        if callable(action):
            action = action(state, criteria)
        return action, {'selected_probability': 0.3, 'synthetic': True}


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.ctx = context()
        self.host = Host(self.ctx)

    def test_all_seven_versioned_skills_parse_and_hash(self):
        skills = load_skills()
        self.assertEqual(len(skills), 7)
        self.assertTrue(all(len(s['sha256']) == 64 and s['version'] == '1.0.0' for s in skills.values()))
        self.assertIn('24 hours', load_skill('plan_forecast'))

    def test_actual_tool_evidence_can_change_selected_forecast(self):
        def conditional(state, criteria):
            return ('commit:alternative' if state['observations'][0]['observation']['evidence'] == 'outage'
                    else 'commit_default')
        for evidence, expected in [('ordinary', 'default'), ('outage', 'alternative')]:
            host = Host(self.ctx, evidence)
            client = ScriptedClient(['search_evidence', 'request_forecast:alternative', conditional])
            result, trace = run(self.ctx, host, client)
            self.assertEqual(result.candidate_id, expected)
            self.assertIsNone(trace['fallback'])
            # No R17 0.55 gate overrides a valid typed decision.
            self.assertEqual(trace['calls'][-1]['record']['selected_probability'], .3)

    def test_all_tool_families_reachable_and_trace_snapshots_are_independent(self):
        for action in self.host.specs:
            client = ScriptedClient([action, 'commit_default'])
            _, trace = run(self.ctx, self.host, client)
            self.assertEqual(trace['tools'][0]['action'], action)
            self.assertEqual(client.requests[0][0]['observations'], [])
            self.assertEqual(len(client.requests[1][0]['observations']), 1)
            self.assertNotIn(action, client.requests[1][2])

    def test_candidate_and_episode_reservations_prevent_tool_execution(self):
        client = ScriptedClient(['commit_default'])
        run(self.ctx, self.host, client, BudgetLimits(candidates=1, episodes=1, historical_evaluations=1))
        offered = client.requests[0][2]
        self.assertNotIn('request_forecast:alternative', offered)
        self.assertNotIn('retrieve_analogs:diverse', offered)
        self.assertNotIn('compare_candidates', offered)
        self.assertEqual(self.host.executions, [])

    def test_api_budget_reserves_final_commit(self):
        client = ScriptedClient(['inspect_state', 'commit_default'])
        _, trace = run(self.ctx, self.host, client, BudgetLimits(api_calls=2))
        self.assertFalse(any(key in self.host.specs for key in client.requests[-1][2]))
        self.assertEqual(trace['costs']['api_calls'], 2)

    def test_review_has_separate_skill_and_charges_both_decisions(self):
        client = ScriptedClient(['request_forecast:alternative', 'review:alternative', 'approve'])
        artifact_out, trace = run(self.ctx, self.host, client)
        self.assertEqual(artifact_out.candidate_id, 'alternative')
        self.assertEqual(trace['costs']['api_calls'], 3)
        self.assertEqual(trace['stop_reason'], 'reviewed_commit')
        self.assertEqual(set(client.requests[-1][2]), {'approve', 'continue', 'default'})
        self.assertEqual(client.requests[-1][1], load_skill('review_and_commit'))

    def test_review_can_abstain_or_continue(self):
        for action, suffix in [('default', []), ('continue', ['commit_default'])]:
            client = ScriptedClient(['review:default', action] + suffix)
            output, trace = run(self.ctx, self.host, client)
            self.assertEqual(output.candidate_id, 'default')
            self.assertEqual(trace['reviews'][0]['outcome'], action)

    def test_api_error_preserves_valid_default_and_failed_request(self):
        client = ScriptedClient(['inspect_state', TimeoutError('injected timeout')])
        output, trace = run(self.ctx, self.host, client)
        self.assertEqual(output.candidate_id, 'default')
        self.assertEqual(trace['costs']['api_calls'], 2)
        self.assertIn('TimeoutError', trace['calls'][-1]['error'])
        self.assertEqual(len(trace['tools']), 1)

    def test_repeated_or_unregistered_or_unmaterialized_actions_rejected(self):
        for actions in (['inspect_state', 'inspect_state'], ['run_shell'], ['commit:alternative']):
            _, trace = run(self.ctx, self.host, ScriptedClient(actions))
            self.assertIsNotNone(trace['fallback'])
            self.assertEqual(trace['selected_candidate_id'], 'default')

    def test_transport_timeout_is_bounded_without_mutating_shared_client(self):
        client = ScriptedClient(['commit_default'])
        run(self.ctx, self.host, client, BudgetLimits(api_timeout_seconds=7))
        self.assertLessEqual(client.requests[0][3], 7)
        self.assertEqual(client.timeout, 999)

    def test_cross_context_candidate_cannot_be_submitted(self):
        self.host.artifacts['alternative'] = dataclasses.replace(self.host.artifacts['alternative'], context_id='wrong')
        output, trace = run(self.ctx, self.host, ScriptedClient(['request_forecast:alternative']))
        self.assertEqual(output.candidate_id, 'default')
        self.assertIn('different context', trace['fallback']['detail'])

    def test_invalid_default_fails_without_claiming_success(self):
        self.host.artifacts['default'] = dataclasses.replace(self.host.artifacts['default'], context_id='wrong')
        with self.assertRaises(ValueError):
            run(self.ctx, self.host, ScriptedClient([]))

    def test_oversized_evidence_does_not_get_silently_truncated(self):
        self.host.evidence = 'x' * 24000
        _, trace = run(self.ctx, self.host, ScriptedClient(['search_evidence']))
        self.assertEqual(trace['costs']['api_calls'], 1)
        self.assertIn('24000-byte', trace['fallback']['detail'])

    def test_default_load_counts_toward_origin_deadline(self):
        self.host.default = lambda ctx: time.sleep(1)
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            run(self.ctx, self.host, ScriptedClient([]), BudgetLimits(origin_deadline_seconds=.03))
        self.assertLess(time.monotonic() - started, .5)

    def test_blocking_tool_is_interrupted_and_falls_back(self):
        self.host.execute = lambda *args: time.sleep(1)
        started = time.monotonic()
        output, trace = run(self.ctx, self.host, ScriptedClient(['inspect_state']),
                            BudgetLimits(origin_deadline_seconds=.05))
        self.assertLess(time.monotonic() - started, .5)
        self.assertEqual(output.candidate_id, 'default')
        self.assertIn('DeadlineExceeded', trace['fallback']['detail'])
        self.assertEqual(trace['tools'][0]['status'], 'error')

    def test_initial_numerical_observation_visible_before_first_decision(self):
        self.host.initial_observation = lambda ctx: {'volatility': 5.0, 'evidence_available': False}
        client = ScriptedClient(['commit_default'])
        _, trace = run(self.ctx, self.host, client)
        self.assertEqual(client.requests[0][0]['observations'][0]['observation']['volatility'], 5.0)
        self.assertFalse(trace['initial_observation']['evidence_available'])

    def test_tool_budget_violation_retains_result_in_audit(self):
        self.host.execute = lambda *args: {'observation': {'sample_count': 100},
                                           'costs': {'historical_evaluations': 100}}
        _, trace = run(self.ctx, self.host, ScriptedClient(['compare_candidates']))
        self.assertIn('resource reservation', trace['fallback']['detail'])
        self.assertEqual(trace['tools'][0]['result']['costs']['historical_evaluations'], 100)
        self.assertEqual(trace['tools'][0]['reserved_costs']['historical_evaluations'], 4)


if __name__ == '__main__':
    unittest.main()
