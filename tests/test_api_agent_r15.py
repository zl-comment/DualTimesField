import unittest
from dataclasses import fields
from unittest.mock import Mock
import urllib.error

import numpy as np

from forecasting.api_agent.transport import validate_choice, KevRequestError
from forecasting.api_agent.workflow import (Snapshot, PROFILES, decide, diagnostics,
                                             fixed_choice, predictor_input)


def snapshot():
    return Snapshot('NSW1', 1640959200, 1640959200 + 3600 * np.arange(24),
                    np.arange(72.), np.arange(72.) / 10, np.ones((72, 10)),
                    np.ones((24, 10)), 'frozen-preprocessing-sha')


class AgentContractTests(unittest.TestCase):
    def test_snapshot_excludes_labels_and_owns_readonly_arrays(self):
        s = snapshot()
        self.assertFalse({'actual', 'target', 'label'} & {f.name for f in fields(s)})
        with self.assertRaises(ValueError):
            s.price[0] = 1
        self.assertEqual(s.target_time_unix[0], s.origin_unix)

    def test_future_or_shifted_axis_rejected(self):
        s = snapshot()
        with self.assertRaises(ValueError):
            Snapshot(s.region, s.origin_unix, s.target_time_unix + 3600,
                     s.price, s.demand_std, s.calendar_past, s.calendar_future, 'sha')

    def test_hindcast_uses_only_first_48_observed_prices(self):
        s = snapshot()
        row = predictor_input(s, 'base', hindcast=True)
        np.testing.assert_equal(row['target'], np.arange(48.))
        self.assertEqual(row['future_covariates']['cal_0'].shape, (24,))
        self.assertNotIn('demand_std', row['future_covariates'])
        with self.assertRaises(ValueError):
            predictor_input(s, 'invented_profile')

    def test_api_rejects_invalid_or_invented_choices(self):
        criteria = {'base': '', 'calendar': ''}
        result = {'model': 'kev-latest', 'answers': {'action': {'type': 'choice',
                  'choice': 'base', 'probabilities': {'base': .8, 'calendar': .2}}}}
        self.assertEqual(validate_choice(result, criteria)[0], 'base')
        result['answers']['action']['choice'] = 'execute_shell'
        with self.assertRaises(ValueError):
            validate_choice(result, criteria)
        result['answers']['action']['choice'] = 'base'
        result['answers']['action']['probabilities']['base'] = float('nan')
        with self.assertRaises(ValueError):
            validate_choice(result, criteria)

    def evidence(self):
        s = snapshot()
        q = np.repeat(s.price[-24:, None], 5, axis=1)
        return diagnostics(s, {'base': q + 10, 'calendar': q + 2, 'price_only': q + 5})

    def test_transport_failure_keeps_same_origin_via_fixed_fallback(self):
        evidence = self.evidence()
        self.assertEqual(fixed_choice(evidence), 'calendar')
        client = Mock()
        client.choose.side_effect = urllib.error.URLError('timeout')
        selected, trace = decide(snapshot(), evidence, client)
        self.assertEqual(selected, 'calendar')
        self.assertEqual(trace['fallback']['to'], 'FIXED')

    def test_agent_can_change_real_profile_with_bounded_calls(self):
        client = Mock()
        client.choose.side_effect = [('inspect_recent_fit', {'selected_probability': .9}),
                                    ('calendar', {'selected_probability': .9})]
        selected, trace = decide(snapshot(), self.evidence(), client)
        self.assertEqual(selected, 'calendar')
        self.assertEqual(len(trace['calls']), 2)
        self.assertEqual(len(trace['tool_calls']), 1)
        self.assertNotIn('demand_std', predictor_input(snapshot(), selected)['past_covariates'])

    def test_guard_rejects_unsupported_change(self):
        client = Mock()
        client.choose.side_effect = [('inspect_recent_fit', {'selected_probability': .9}),
                                    ('calendar', {'selected_probability': .4})]
        selected, trace = decide(snapshot(), self.evidence(), client)
        self.assertEqual(selected, 'base')
        self.assertEqual(trace['guard'], 'insufficient_support_for_change')

    def test_failed_request_remains_in_audit_trace(self):
        client = Mock()
        record = {'request': {'state': 'current origin only'}, 'error': 'TimeoutError'}
        client.choose.side_effect = KevRequestError(record)
        selected, trace = decide(snapshot(), self.evidence(), client)
        self.assertEqual(selected, 'calendar')
        self.assertEqual(trace['calls'], [record])


if __name__ == '__main__':
    unittest.main()
