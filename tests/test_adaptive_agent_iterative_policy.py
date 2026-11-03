"""R20 optional input revision must use real new forecasts and fixed-cohort diagnostics."""
import json
import time
import unittest
from unittest.mock import patch

import numpy as np

from forecasting.adaptive_agent.contracts import BudgetLimits
from forecasting.adaptive_agent.iterative_policy import compact_request_state, load_skills, run
from forecasting.adaptive_agent.iterative_runtime import IterativeToolHost
from test_adaptive_agent_runtime import ALTERNATIVE, DEFAULT, ORIGIN, Choices, context, episode, host


REVISION = 'chronos:base:48'


def iterative_host(episodes=None):
    if episodes is None:
        episodes = [episode('old-' + str(i), ORIGIN - (i + 2) * 86400) for i in range(3)]
    original = host(episodes=episodes)
    return IterativeToolHost(original.registry, original.evidence, original.analogs, original.feedback)


class IterativePolicyTests(unittest.TestCase):
    def test_stage_skills_are_separately_versioned(self):
        self.assertTrue(all(s['version'] == '3.0.0' for s in load_skills().values()))

    def test_diagnosis_can_cause_optional_input_revision_and_changed_forecast(self):
        runtime = iterative_host()
        client = Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE,
                          'forecast:' + REVISION, 'commit:' + REVISION])
        output, trace = run(context(), runtime, client)
        self.assertIsNone(trace['fallback'], trace)
        self.assertEqual(output.candidate_id, REVISION)
        self.assertFalse(np.array_equal(output.quantiles, runtime.artifact(context(), DEFAULT).quantiles))
        self.assertEqual(trace['costs']['historical_evaluations'], 9)
        self.assertEqual([i['incremental_historical_units'] for i in trace['iterations']], [6, 3])
        self.assertEqual(trace['iterations'][0]['cohort_ids'], trace['iterations'][1]['cohort_ids'])
        self.assertEqual(len(trace['revisions']), 1)
        self.assertEqual(trace['revisions'][0]['new_candidate_id'], REVISION)
        self.assertFalse(trace['revisions'][0]['protocol_required'])
        self.assertEqual(trace['costs']['api_calls'], 4)
        self.assertEqual(trace['costs']['tool_calls'], 5)
        latest = [o for o in client.requests[-1]['state']['observations'] if o['action'].startswith('diagnose:')]
        self.assertEqual(len(latest), 1)
        self.assertEqual(set(latest[0]['observation']['candidates']), {DEFAULT, ALTERNATIVE, REVISION})
        self.assertEqual(set(latest[0]['observation']['current_forecast_comparison']['candidates']),
                         {DEFAULT, ALTERNATIVE, REVISION})
        self.assertEqual(len([o for o in trace['final_state']['observations'] if o['action'].startswith('diagnose:')]), 2)

    def test_revision_is_optional_and_direct_default_commit_remains_legal(self):
        runtime = iterative_host()
        client = Choices(['retrieve_analogs:diverse', 'forecast:' + ALTERNATIVE, 'commit:' + DEFAULT])
        output, trace = run(context(), runtime, client)
        self.assertEqual(output.candidate_id, DEFAULT)
        self.assertEqual(trace['revisions'], [])
        self.assertEqual(trace['costs']['api_calls'], 3)
        self.assertIn('forecast:' + REVISION, client.requests[-1]['criteria'])

    def test_two_revisions_reach_real_adapter_with_changed_profiles_and_windows(self):
        runtime = iterative_host()
        second = 'chronos:calendar:24'
        client = Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE,
                          'forecast:' + REVISION, 'forecast:' + second, 'commit:' + second])
        output, trace = run(context(), runtime, client)
        self.assertIsNone(trace['fallback'], trace)
        current_calls = [(profile, window) for identity, profile, window in runtime.registry.adapters['chronos'].calls
                         if identity == context().identity()]
        self.assertEqual(current_calls, [('price_only', 24), ('base', 48), ('calendar', 24)])
        self.assertEqual(output.candidate_id, second)
        self.assertEqual([r['new_candidate_id'] for r in trace['revisions']], [REVISION, second])
        self.assertEqual([i['incremental_historical_units'] for i in trace['iterations']], [6, 3, 3])

    def test_max_shape_five_candidates_eight_episodes_fits_real_request_byte_limit(self):
        episodes = [episode('old-' + str(i), ORIGIN - (i + 2) * 86400) for i in range(8)]
        runtime = iterative_host(episodes)
        second, third = 'chronos:calendar:24', 'chronos:base:24'
        client = Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE, 'forecast:' + REVISION,
                          'forecast:' + second, 'forecast:' + third, 'commit:' + third])
        output, trace = run(context(), runtime, client)
        self.assertIsNone(trace['fallback'], trace)
        self.assertEqual(output.candidate_id, third)
        self.assertEqual(trace['costs']['candidates'], 5)
        self.assertEqual(trace['costs']['historical_evaluations'], 40)
        self.assertEqual(len(trace['revisions']), 3)
        self.assertEqual([i['incremental_historical_units'] for i in trace['iterations']], [16, 8, 8, 8])
        self.assertTrue(all(len(i['cohort_ids']) == 8 for i in trace['iterations']))
        self.assertFalse(any(key.startswith('forecast:') for key in client.requests[-1]['criteria']))
        latest = [o for o in client.requests[-1]['state']['observations'] if o['action'].startswith('diagnose:')]
        self.assertEqual(len(latest), 1)
        self.assertEqual(len(latest[0]['observation']['candidates']), 5)
        # The production choose boundary constructs and checks the complete
        # typed-choice body, including prompts, on every actual request.
        self.assertEqual(trace['costs']['api_calls'], 6)
        self.assertLessEqual(max(call['request_bytes'] for call in trace['calls']), 24000)

    def test_hist_candidate_tool_and_api_limits_filter_optional_revision(self):
        for limits in (BudgetLimits(historical_evaluations=6), BudgetLimits(candidates=2),
                       BudgetLimits(tool_calls=4), BudgetLimits(api_calls=4)):
            with self.subTest(limits=limits):
                client = Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE, 'commit:' + DEFAULT])
                _, trace = run(context(), iterative_host(), client, limits)
                self.assertIsNone(trace['fallback'], trace)
                self.assertFalse(any(k.startswith('forecast:') for k in client.requests[-1]['criteria']))

    def test_optional_feedback_retrieval_is_recorded_and_not_reoffered(self):
        client = Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE,
                          'inspect_feedback', 'commit:' + DEFAULT])
        _, trace = run(context(), iterative_host(), client)
        self.assertIsNone(trace['fallback'], trace)
        self.assertEqual(trace['optional_evidence_actions'], ['inspect_feedback'])
        self.assertNotIn('inspect_feedback', client.requests[-1]['criteria'])
        tool = [t for t in trace['tools'] if t['action'] == 'inspect_feedback'][0]
        self.assertFalse(tool['protocol_required'])
        self.assertEqual(tool['execution_mode'], 'llm_selected')

    def test_cannot_change_cohort_after_initial_retrieval(self):
        client = Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE, 'retrieve_analogs:recent'])
        output, trace = run(context(), iterative_host(), client)
        self.assertEqual(output.candidate_id, DEFAULT)
        self.assertIn('unavailable phase action', trace['fallback']['detail'])
        self.assertFalse(any(k.startswith('retrieve_analogs:') for k in client.requests[-1]['criteria']))

    def test_mutated_diagnostic_cohort_fails_closed(self):
        runtime = iterative_host()
        original = runtime.execute
        def mutate(ctx, action, state):
            result = original(ctx, action, state)
            if action.startswith('diagnose:'):
                result['observation']['episodes'] = ['different-cohort']
            return result
        runtime.execute = mutate
        output, trace = run(context(), runtime, Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE]))
        self.assertEqual(output.candidate_id, DEFAULT)
        self.assertIn('changed the fixed historical cohort', trace['fallback']['detail'])

    def test_candidate_diagnostic_cannot_use_a_different_successful_subset(self):
        for field, invalid in [('paired_episode_ids', []), ('sample_count', 2)]:
            with self.subTest(field=field):
                runtime = iterative_host()
                original = runtime.execute
                def mutate(ctx, action, state):
                    result = original(ctx, action, state)
                    if action.startswith('diagnose:'):
                        result['observation']['candidates'][ALTERNATIVE][field] = invalid
                    return result
                runtime.execute = mutate
                output, trace = run(context(), runtime, Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE]))
                self.assertEqual(output.candidate_id, DEFAULT)
                self.assertIn('same successful episode intersection', trace['fallback']['detail'])

    def test_new_candidate_is_not_committed_when_incremental_diagnostic_fails(self):
        runtime = iterative_host()
        original = runtime.execute
        def fail(ctx, action, state):
            if action.startswith('diagnose:') and REVISION in action:
                raise RuntimeError('injected incremental failure')
            return original(ctx, action, state)
        runtime.execute = fail
        client = Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE, 'forecast:' + REVISION])
        output, trace = run(context(), runtime, client)
        self.assertEqual(output.candidate_id, DEFAULT)
        self.assertIn('injected incremental failure', trace['fallback']['detail'])
        self.assertEqual(len(trace['iterations']), 1)

    def test_pool_after_revision_is_explicit_and_requires_review(self):
        runtime = iterative_host()
        client = Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE, 'forecast:' + REVISION,
                          f'pool:{DEFAULT}|{REVISION}|0.5', 'approve'])
        output, trace = run(context(), runtime, client)
        self.assertIsNone(trace['fallback'], trace)
        self.assertEqual(output.model_id, 'quantile_pool')
        self.assertEqual(trace['costs']['candidates'], 4)
        self.assertEqual(trace['costs']['historical_evaluations'], 9)
        self.assertEqual(trace['reviews'][-1]['outcome'], 'approve')

    def test_no_mature_cohort_disables_revision_and_requires_review(self):
        runtime = iterative_host(episodes=[])
        client = Choices(['forecast:' + ALTERNATIVE, 'commit:' + DEFAULT, 'approve'])
        with patch('forecasting.adaptive_agent.diagnostics.evaluate_recent_history', return_value={
                'logical_inference_units': 3, 'sample_count': 1, 'lightgbm_supported': False}):
            _, trace = run(context(), runtime, client)
        self.assertIsNone(trace['fallback'], trace)
        self.assertFalse(any(k.startswith('forecast:') for k in client.requests[1]['criteria']))
        self.assertFalse(trace['paired_diagnostic_executed'])
        self.assertEqual(trace['revisions'], [])
        self.assertEqual(trace['stop_reason'], 'reviewed_commit')

    def test_future_episodes_cannot_enter_revision_evidence(self):
        episodes = [episode('past', ORIGIN - 3 * 86400),
                    episode('future-canary', ORIGIN + 86400, actual=91234567.12345)]
        client = Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE,
                          'forecast:' + REVISION, 'commit:' + REVISION, 'approve'])
        _, trace = run(context(), iterative_host(episodes), client)
        self.assertIsNone(trace['fallback'], trace)
        self.assertNotIn('future-canary', json.dumps(client.requests))
        self.assertNotIn('91234567.12345', json.dumps(client.requests))
        self.assertEqual(trace['cohort_ids'], ['past'])
        self.assertEqual(trace['costs']['historical_evaluations'], 3)

    def test_continue_review_does_not_approve_a_reselected_candidate(self):
        runtime = iterative_host([episode('only', ORIGIN - 3 * 86400)])
        client = Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE,
                          'commit:' + ALTERNATIVE, 'continue', 'commit:' + ALTERNATIVE, 'approve'])
        output, trace = run(context(), runtime, client)
        self.assertIsNone(trace['fallback'], trace)
        self.assertEqual(output.candidate_id, ALTERNATIVE)
        self.assertEqual([r['outcome'] for r in trace['reviews']], ['continue', 'approve'])

    def test_tiny_api_budget_falls_back_without_unbudgeted_generation(self):
        runtime = iterative_host()
        client = Choices(['retrieve_analogs:state'])
        output, trace = run(context(), runtime, client, BudgetLimits(api_calls=1))
        self.assertEqual(output.candidate_id, DEFAULT)
        self.assertEqual(trace['costs']['api_calls'], 1)
        self.assertIn('Insufficient API budget', trace['fallback']['detail'])
        self.assertEqual(runtime.registry.adapters['chronos'].calls, [])

    def test_last_review_has_no_continue_that_would_exhaust_submission_budget(self):
        runtime = iterative_host([episode('only', ORIGIN - 3 * 86400)])
        client = Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE,
                          'commit:' + ALTERNATIVE, 'approve'])
        _, trace = run(context(), runtime, client, BudgetLimits(api_calls=4))
        self.assertIsNone(trace['fallback'], trace)
        self.assertNotIn('continue', client.requests[-1]['criteria'])

    def test_partial_diagnostic_cost_is_recorded_even_if_deadline_wrapper_raises(self):
        runtime = iterative_host()
        original = runtime.execute
        def partial(ctx, action, state):
            if not action.startswith('diagnose:'):
                return original(ctx, action, state)
            try:
                time.sleep(1)
            except TimeoutError:
                return {'status': 'partial_failure', 'observation': {'attempted': 1},
                        'candidate_ids': [], 'costs': {'historical_evaluations': 1}}
            raise AssertionError('Expected the actual deadline interrupt')
        runtime.execute = partial
        client = Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE])
        output, trace = run(context(), runtime, client, BudgetLimits(origin_deadline_seconds=.15))
        self.assertEqual(output.candidate_id, DEFAULT)
        self.assertIsNotNone(trace['fallback'])
        self.assertEqual(trace['costs']['historical_evaluations'], 1)
        self.assertEqual(trace['tools'][-1]['charged_costs_basis'], 'observed')
        self.assertEqual(trace['resource_accounting']['unknown_actual_cost_events'], [])

    def test_exception_without_result_is_charged_as_explicit_upper_bound(self):
        runtime = iterative_host()
        original = runtime.execute
        def failed(ctx, action, state):
            if action.startswith('diagnose:'):
                raise RuntimeError('lost tool result after possible inference')
            return original(ctx, action, state)
        runtime.execute = failed
        output, trace = run(context(), runtime, Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE]))
        self.assertEqual(output.candidate_id, DEFAULT)
        self.assertEqual(trace['costs']['historical_evaluations'], 6)
        self.assertEqual(trace['tools'][-1]['charged_costs_basis'], 'estimated_upper_bound')
        self.assertEqual(len(trace['resource_accounting']['unknown_actual_cost_events']), 1)


if __name__ == '__main__':
    unittest.main()
