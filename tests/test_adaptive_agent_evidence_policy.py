"""R19 required comparison coverage is separate from discretionary LLM choices."""
import time
import unittest
from unittest.mock import patch

import numpy as np

from forecasting.adaptive_agent.contracts import BudgetLimits
from forecasting.adaptive_agent.evidence_policy import load_skills, run
from test_adaptive_agent_runtime import (ALTERNATIVE, DEFAULT, ORIGIN, Choices,
                                         context, episode, host)


def enough_history():
    return [episode('old-' + str(i), ORIGIN - (i + 2) * 86400) for i in range(3)]


class EvidencePolicyTests(unittest.TestCase):
    def test_versioned_stage_skills_are_separate_from_r18(self):
        skills = load_skills()
        self.assertEqual(len(skills), 7)
        self.assertTrue(all(s['version'] == '2.0.0' for s in skills.values()))

    def test_paired_path_requires_comparison_but_can_keep_default(self):
        runtime = host(episodes=enough_history())
        client = Choices(['retrieve_analogs:diverse', 'forecast:' + ALTERNATIVE, 'commit:' + DEFAULT])
        output, trace = run(context(), runtime, client)
        self.assertIsNone(trace['fallback'], trace)
        self.assertEqual(output.candidate_id, DEFAULT)
        self.assertEqual(trace['release'], 'r19-evidence-first-agent-v1')
        self.assertEqual(trace['costs'], {'api_calls': 3, 'tool_calls': 3, 'candidates': 2,
                                         'episodes': 3, 'historical_evaluations': 6})
        self.assertEqual(set(client.requests[0]['criteria']), {'retrieve_analogs:' + x for x in
                                                               ('recent', 'seasonal', 'state', 'diverse')})
        self.assertTrue(all(k.startswith('forecast:chronos:') for k in client.requests[1]['criteria']))
        self.assertNotIn('commit_default', client.requests[-1]['criteria'])
        self.assertNotIn('abstain', client.requests[-1]['criteria'])
        self.assertEqual(trace['tools'][0]['execution_mode'], 'llm_selected')
        self.assertTrue(trace['tools'][0]['protocol_required'])
        self.assertEqual(trace['tools'][2]['execution_mode'], 'protocol_required')
        diag = trace['tools'][2]['result']['observation']
        self.assertEqual(set(diag['candidates']), {DEFAULT, ALTERNATIVE})
        self.assertEqual(diag['sample_count'], 3)
        self.assertEqual(set(diag['episodes']), {'old-0', 'old-1', 'old-2'})

    def test_alternative_is_not_overridden_by_low_choice_probability_or_loss_ranking(self):
        runtime = host(episodes=enough_history())
        client = Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE, 'commit:' + ALTERNATIVE])
        output, trace = run(context(), runtime, client)
        self.assertIsNone(trace['fallback'], trace)
        self.assertEqual(output.candidate_id, ALTERNATIVE)
        self.assertFalse(np.array_equal(output.quantiles, runtime.artifact(context(), DEFAULT).quantiles))
        self.assertEqual(trace['calls'][-1]['record']['selected_probability'], .4)

    def test_low_support_review_is_explicit_and_can_return_default(self):
        runtime = host()
        client = Choices(['retrieve_analogs:recent', 'forecast:' + ALTERNATIVE,
                          'commit:' + ALTERNATIVE, 'default'])
        output, trace = run(context(), runtime, client)
        self.assertIsNone(trace['fallback'], trace)
        self.assertEqual(output.candidate_id, DEFAULT)
        self.assertEqual(trace['stop_reason'], 'review_abstention')
        self.assertEqual(trace['costs']['api_calls'], 4)
        self.assertEqual(trace['calls'][-1]['phase'], 'review_prediction')

    def test_pool_requires_a_real_final_choice_and_preserves_registered_weight(self):
        runtime = host(episodes=enough_history())
        pool = f'pool:{DEFAULT}|{ALTERNATIVE}|0.25'
        client = Choices(['retrieve_analogs:seasonal', 'forecast:' + ALTERNATIVE, pool, 'approve'])
        output, trace = run(context(), runtime, client)
        self.assertIsNone(trace['fallback'], trace)
        self.assertEqual(output.model_id, 'quantile_pool')
        expected = .25 * runtime.artifact(context(), DEFAULT).quantiles + .75 * runtime.artifact(context(), ALTERNATIVE).quantiles
        np.testing.assert_array_equal(output.quantiles, expected)
        self.assertEqual(trace['tools'][-1]['action'], pool)
        self.assertEqual(trace['tools'][-1]['execution_mode'], 'llm_selected')
        self.assertFalse(trace['tools'][-1]['protocol_required'])
        self.assertEqual(trace['costs']['candidates'], 3)

    def test_no_history_uses_automatic_chronos_only_hindcast_without_fake_pairing(self):
        runtime = host(episodes=[])
        client = Choices(['forecast:' + ALTERNATIVE, 'commit:' + DEFAULT, 'approve'])
        with patch('forecasting.adaptive_agent.diagnostics.evaluate_recent_history', return_value={
                'logical_inference_units': 3, 'sample_count': 1,
                'diagnostic_kind': 'within_visible_history_hindcast', 'lightgbm_supported': False}):
            output, trace = run(context(), runtime, client)
        self.assertIsNone(trace['fallback'], trace)
        self.assertEqual(output.candidate_id, DEFAULT)
        self.assertEqual(trace['tools'][0]['action'], 'inspect_recent_fit')
        self.assertEqual(trace['tools'][0]['execution_mode'], 'protocol_required')
        self.assertEqual(trace['calls'][0]['phase'], 'select_alternative_input')
        self.assertFalse(trace['paired_diagnostic_executed'])
        unsupported = [o for o in trace['final_state']['observations'] if o['action'] == 'paired_diagnostics'][0]
        self.assertEqual(unsupported['status'], 'unsupported')
        self.assertFalse(unsupported['observation']['lightgbm_chronos_paired_error_available'])

    def test_only_existing_default_can_be_committed_after_required_generation(self):
        runtime = host(episodes=enough_history())
        # A stop action from the old free-loop protocol cannot bypass phase one.
        output, trace = run(context(), runtime, Choices(['commit_default']))
        self.assertEqual(output.candidate_id, DEFAULT)
        self.assertIn('unavailable phase action', trace['fallback']['detail'])
        self.assertEqual(trace['costs']['tool_calls'], 0)

    def test_failure_after_retrieval_is_not_hidden_and_defaults_safely(self):
        runtime = host(episodes=enough_history())
        client = Choices(['retrieve_analogs:state', TimeoutError('injected API failure')])
        output, trace = run(context(), runtime, client)
        self.assertEqual(output.candidate_id, DEFAULT)
        self.assertIn('injected API failure', trace['fallback']['detail'])
        self.assertEqual(trace['costs']['api_calls'], 2)
        self.assertEqual(trace['costs']['tool_calls'], 1)
        self.assertIn('error', trace['calls'][-1]['record'])

    def test_historical_reservation_rejects_oversized_pair_before_inference(self):
        runtime = host(episodes=enough_history())
        client = Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE])
        output, trace = run(context(), runtime, client, BudgetLimits(historical_evaluations=5))
        self.assertEqual(output.candidate_id, DEFAULT)
        self.assertIn('Paired diagnostic unavailable', trace['fallback']['detail'])
        self.assertEqual(trace['costs']['historical_evaluations'], 0)
        self.assertEqual(len(runtime.registry.adapters['lightgbm'].calls), 1)

    def test_required_review_cannot_be_bypassed_at_api_budget_boundary(self):
        runtime = host()
        client = Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE, 'commit:' + ALTERNATIVE])
        output, trace = run(context(), runtime, client, BudgetLimits(api_calls=3))
        self.assertEqual(output.candidate_id, DEFAULT)
        self.assertEqual(trace['costs']['api_calls'], 3)
        self.assertEqual(trace['fallback']['reason'], 'required_review_budget_exhausted')

    def test_continue_then_same_candidate_requires_a_new_positive_review(self):
        runtime = host()
        client = Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE,
                          'commit:' + ALTERNATIVE, 'continue', 'commit:' + ALTERNATIVE, 'approve'])
        output, trace = run(context(), runtime, client)
        self.assertIsNone(trace['fallback'], trace)
        self.assertEqual(output.candidate_id, ALTERNATIVE)
        self.assertEqual(trace['costs']['api_calls'], 6)
        self.assertEqual([r['outcome'] for r in trace['reviews']], ['continue', 'approve'])
        self.assertEqual(client.requests[4]['state']['reviewed_candidate_ids'], [])

    def test_continue_does_not_authorize_same_candidate_after_budget_exhaustion(self):
        runtime = host()
        client = Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE,
                          'commit:' + ALTERNATIVE, 'continue', 'commit:' + ALTERNATIVE])
        output, trace = run(context(), runtime, client, BudgetLimits(api_calls=5))
        self.assertEqual(output.candidate_id, DEFAULT)
        self.assertEqual(trace['costs']['api_calls'], 5)
        self.assertEqual(trace['fallback']['reason'], 'required_review_budget_exhausted')
        self.assertEqual(trace['final_state']['reviewed_candidate_ids'], [])

    def test_no_unregistered_evidence_or_future_episode_in_state(self):
        runtime = host(episodes=enough_history() + [episode('future-secret', ORIGIN + 3600, actual=91234567.12345)])
        client = Choices(['retrieve_analogs:state', 'forecast:' + ALTERNATIVE, 'commit:' + DEFAULT])
        _, trace = run(context(), runtime, client)
        import json
        sent = json.dumps(client.requests)
        self.assertNotIn('future-secret', sent)
        self.assertNotIn('91234567.12345', sent)
        self.assertNotIn('search_evidence', client.requests[0]['criteria'])
        self.assertTrue(all(not c['state']['future_actuals_available'] for c in client.requests))

    def test_tool_deadline_interrupts_and_records_failure(self):
        runtime = host(episodes=enough_history())
        runtime.execute = lambda *args: time.sleep(1)
        started = time.monotonic()
        output, trace = run(context(), runtime, Choices(['retrieve_analogs:state']),
                            BudgetLimits(origin_deadline_seconds=.05))
        self.assertLess(time.monotonic() - started, .5)
        self.assertEqual(output.candidate_id, DEFAULT)
        self.assertEqual(trace['tools'][0]['status'], 'error')
        self.assertIn('DeadlineExceeded', trace['fallback']['detail'])


if __name__ == '__main__':
    unittest.main()
