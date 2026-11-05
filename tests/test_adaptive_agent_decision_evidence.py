import unittest

from forecasting.adaptive_agent.decision_evidence import build_decision_evidence


DEFAULT = 'lightgbm-input:identity'
OPERATED = 'lightgbm-input:history_demand_last48_mean'


def observation(*, mean=.3, median=.2, wins=.75, first=.1, last=.2, disagreement=.4):
    return {
        'candidates': {
            DEFAULT: {'sample_count': 8, 'metrics': {'qs_first12': 2, 'qs_last12': 3}},
            OPERATED: {
                'sample_count': 8,
                'metrics': {'qs_first12': 2 - first, 'qs_last12': 3 - last},
                'paired_vs_default': {'mean_qs_gain': mean, 'median_qs_gain': median,
                                      'win_fraction': wins},
            },
        },
        'current_forecast_comparison': {'pairs': [
            {'a': DEFAULT, 'b': OPERATED, 'median_absolute_difference': disagreement}
        ]},
    }


class DecisionEvidenceTests(unittest.TestCase):
    state = {'days_oldest_first': [
        {'demand_std_mean': -.5}, {'demand_std_mean': .1}, {'demand_std_mean': .4}
    ]}

    def test_admits_only_consistent_paired_gain(self):
        result = build_decision_evidence(observation(), DEFAULT, self.state)
        self.assertEqual(result['eligible_candidate_ids'], [OPERATED])
        row = result['operations'][OPERATED]
        self.assertTrue(row['eligible'])
        self.assertAlmostEqual(row['current_demand_regime_hypothesis']['signed_shift'], .75)

    def test_rejects_aggregate_gain_when_episode_win_rate_is_weak(self):
        result = build_decision_evidence(observation(wins=.5), DEFAULT, self.state)
        self.assertEqual(result['eligible_candidate_ids'], [])
        self.assertEqual(result['operations'][OPERATED]['rejection_reasons'],
                         ['paired_win_fraction'])

    def test_rejects_horizon_inconsistency(self):
        result = build_decision_evidence(observation(last=-.01), DEFAULT, self.state)
        self.assertEqual(result['eligible_candidate_ids'], [])
        self.assertIn('positive_last12_qs_gain',
                      result['operations'][OPERATED]['rejection_reasons'])


if __name__ == '__main__':
    unittest.main()
