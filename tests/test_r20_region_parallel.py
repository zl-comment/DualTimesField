import copy
import unittest

from scripts.run_r20_region_parallel import REGIONS, combine_reports


class RegionalAggregationTests(unittest.TestCase):
    def setUp(self):
        self.counts = dict(zip(REGIONS, (2, 4, 6)))
        self.reports = {}
        for i, region in enumerate(REGIONS):
            count = self.counts[region]
            self.reports[region] = {
                'status': 'COMPLETE', 'origins': count, 'expected_origins': count,
                'manifest_id': region, 'policy': 'r20_iterative_input',
                'regions': {region: {'count': count, 'BASE': {'qs': i + 2.0}, 'AGENT': {'qs': i + 1.0}}},
                'api_calls': count * 3, 'api_failure_count': 0, 'fallback_count': 0,
                'failed_origins': 0, 'prediction_differs_from_base': count,
                'revision_count': 0, 'origins_with_revision': 0,
                'selected_candidates': {'lightgbm:full:72': count}}

    def test_equal_region_mean_and_additive_counts(self):
        result = combine_reports(self.reports, self.counts)
        self.assertEqual(result['region_equal_mean']['AGENT']['qs'], 2.0)
        self.assertEqual(result['origins'], 12)
        self.assertEqual(result['api_calls'], 36)
        self.assertEqual(result['selected_candidates'], {'lightgbm:full:72': 12})
        self.assertNotIn('origin_latency_seconds', result)

    def test_reject_incomplete_or_wrong_region_reports(self):
        changes = [('status', 'FAILED'), ('origins', 1), ('expected_origins', 1),
                   ('policy', 'r19_evidence_first'), ('regions', {})]
        for field, value in changes:
            with self.subTest(field=field):
                reports = copy.deepcopy(self.reports)
                reports['NSW1'][field] = value
                with self.assertRaises(ValueError):
                    combine_reports(reports, self.counts)

    def test_reject_missing_region(self):
        self.reports.pop('TAS1')
        with self.assertRaises(ValueError):
            combine_reports(self.reports, self.counts)

    def test_reject_incompatible_metrics(self):
        self.reports['NSW1']['regions']['NSW1']['AGENT'] = {'mae': 3.0}
        with self.assertRaises(ValueError):
            combine_reports(self.reports, self.counts)


if __name__ == '__main__':
    unittest.main()
