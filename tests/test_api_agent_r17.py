import unittest
from unittest.mock import Mock

import numpy as np

import test_api_agent_r15 as fixtures
from forecasting.api_agent.transport import KevRequestError
from forecasting.api_agent.workflow_r17 import decide, inspect_history


class ToolAgentTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = fixtures.snapshot()
        self.evidence = fixtures.AgentContractTests().evidence()
        self.forecasts = {p: np.tile(np.arange(5.), (24, 1)) + shift
                          for p, shift in [('base', 10), ('calendar', 2), ('price_only', 5)]}

    def run_agent(self, actions):
        client = Mock()
        client.choose.side_effect = [(a, {'selected_probability': .9}) for a in actions]
        result = decide(self.snapshot, self.evidence, client, forecasts=self.forecasts)
        return *result, client

    def test_tools_are_chosen_in_sequence_then_change_selected_forecast(self):
        selected, trace, client = self.run_agent([
            'inspect_history', 'inspect_recent_fit', 'inspect_forecast_disagreement', 'finish_price_only'])
        self.assertEqual(selected, 'price_only')
        self.assertNotEqual(selected, trace['fixed_choice'])
        self.assertEqual(len(trace['calls']), 4)
        self.assertEqual(len(trace['tool_calls']), 3)
        first, second, third, last = client.choose.call_args_list
        self.assertEqual(first.args[0]['evidence'], {})
        self.assertEqual(set(second.args[0]['evidence']), {'inspect_history'})
        self.assertNotIn('inspect_history', second.args[2])
        self.assertNotIn('finish_price_only', first.args[2])
        self.assertIn('finish_price_only', third.args[2])
        self.assertTrue(all(k.startswith('finish_') for k in last.args[2]))
        self.assertEqual(trace['tool_calls'][2]['result']['kind'], 'frozen_predictions_not_observations')

    def test_uninspected_or_repeated_actions_fall_back(self):
        for actions in (['finish_price_only'], ['inspect_history', 'inspect_history'], ['run_shell']):
            with self.subTest(actions=actions):
                selected, trace, _ = self.run_agent(actions)
                self.assertEqual(selected, 'calendar')
                self.assertEqual(trace['stop_reason'], 'failed_decision')

    def test_unsupported_alternative_not_offered(self):
        self.evidence['profiles']['price_only']['clears_2pct_change_guard'] = False
        selected, trace, client = self.run_agent(['inspect_recent_fit', 'finish_price_only'])
        self.assertNotIn('finish_price_only', client.choose.call_args_list[-1].args[2])
        self.assertEqual(selected, 'calendar')
        self.assertIsNotNone(trace['fallback'])

    def test_timeout_after_tool_preserves_trace_and_fixed_prediction(self):
        client = Mock()
        record = {'request': {'state': 'test'}, 'error': 'TimeoutError'}
        client.choose.side_effect = [
            ('inspect_recent_fit', {'selected_probability': .9}), KevRequestError(record)]
        selected, trace = decide(self.snapshot, self.evidence, client, forecasts=self.forecasts)
        self.assertEqual(selected, 'calendar')
        self.assertEqual(trace['calls'][-1], record)
        self.assertEqual(len(trace['tool_calls']), 1)

    def test_uncertain_final_choice_uses_fixed(self):
        client = Mock()
        client.choose.return_value = ('finish_base', {'selected_probability': .4})
        selected, trace = decide(self.snapshot, self.evidence, client, forecasts=self.forecasts)
        self.assertEqual(selected, 'calendar')
        self.assertEqual(trace['stop_reason'], 'uncertain_commit')

    def test_input_corruption_fails_before_any_api_call(self):
        for kind in ('wrong_origin', 'nan_forecast', 'crossed_quantiles'):
            with self.subTest(kind=kind):
                self.setUp()
                if kind == 'wrong_origin':
                    self.evidence['hindcast_target_end_exclusive'] += 3600
                elif kind == 'nan_forecast':
                    self.forecasts['base'][0, 0] = np.nan
                else:
                    self.forecasts['base'][0, 0] = 999
                client = Mock()
                with self.assertRaises(ValueError):
                    decide(self.snapshot, self.evidence, client, forecasts=self.forecasts)
                client.choose.assert_not_called()

    def test_history_tool_cannot_extend_past_origin(self):
        days = inspect_history(self.snapshot)['observed_days_oldest_first']
        self.assertEqual(days[0]['start_unix'], self.snapshot.origin_unix - 72 * 3600)
        self.assertEqual(days[-1]['end_exclusive_unix'], self.snapshot.origin_unix)
        for before, after in zip(days, days[1:]):
            self.assertEqual(before['end_exclusive_unix'], after['start_unix'])


if __name__ == '__main__':
    unittest.main()
