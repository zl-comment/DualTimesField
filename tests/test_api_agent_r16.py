import unittest
from unittest.mock import Mock

import test_api_agent_r15 as fixtures
from forecasting.api_agent.workflow_r16 import needs_kev, decide


class GatedRoutingTests(unittest.TestCase):
    def evidence(self):
        return fixtures.AgentContractTests().evidence()

    def test_clear_dominance_skips_api(self):
        evidence = self.evidence()
        client = Mock()
        selected, trace = decide(fixtures.snapshot(), evidence, client)
        self.assertEqual(selected, 'calendar')
        client.choose.assert_not_called()
        self.assertEqual(trace['routing'], 'clear_recent_fit_dominance')

    def test_conflicting_halves_get_one_decision(self):
        evidence = self.evidence()
        evidence['profiles']['price_only']['qs_last_12h'] = .1
        self.assertTrue(needs_kev(evidence)[0])
        client = Mock()
        client.choose.return_value = ('price_only', {'selected_probability': .9})
        selected, trace = decide(fixtures.snapshot(), evidence, client)
        self.assertEqual(selected, 'price_only')
        self.assertEqual(len(trace['calls']), 1)

    def test_uncertain_tradeoff_uses_strong_fixed_default(self):
        evidence = self.evidence()
        evidence['profiles']['price_only']['qs_last_12h'] = .1
        client = Mock()
        client.choose.return_value = ('base', {'selected_probability': .4})
        selected, trace = decide(fixtures.snapshot(), evidence, client)
        self.assertEqual(selected, 'calendar')
        self.assertIn('guard', trace)


if __name__ == '__main__':
    unittest.main()
