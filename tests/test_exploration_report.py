import tempfile
from pathlib import Path
import unittest

import numpy as np

from forecasting import exploration_evaluate as evaluate
from forecasting import exploration_report as report


class ExplorationReportTest(unittest.TestCase):
    def forecast(self, start=0, n=2, price=0):
        return evaluate.Forecast(np.full((n, 2), price, dtype=float),
            np.full((n, 2, 5), price, dtype=float), np.zeros((n, 2)),
            np.arange(start, start + n) * 3600, 0.)

    def test_stitch_matches_static_actual_and_rejects_missing_origins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = []
            for quarter in range(8):
                path = root / f'q{quarter}.npz'
                report.save_forecast(path, self.forecast(start=quarter*2), 2026, 'NSW1')
                parts.append(path)
            joined = report.stitch_forecasts(parts, root/'joined.npz', 2026, 'NSW1',
                reference=self.forecast(n=16), expected_origins=16)
            self.assertEqual(len(joined.origin_unix), 16)
            with self.assertRaisesRegex(ValueError, 'gaps/duplicates'):
                report.stitch_forecasts(parts[:3] + [parts[2]] + parts[4:], root/'bad.npz',
                                        2026, 'NSW1', expected_origins=16)
            reference = self.forecast(n=16)
            reference.actual[0, 0] = 1
            with self.assertRaisesRegex(ValueError, 'Actual prices differ'):
                report.stitch_forecasts(parts, root/'bad.npz', 2026, 'NSW1',
                                        reference=reference, expected_origins=16)

    def test_stitch_requires_all_eight_quarters(self):
        with self.assertRaisesRegex(ValueError, 'eight'):
            report.stitch_forecasts([], Path('/unused'), 2026, 'NSW1')

    def test_floor_selection_uses_separate_point_and_quantile_objectives(self):
        validation = self.forecast(price=-100)
        validation.quantile[:] = 0
        choice = report.select_floors(validation, [-10., -5., 5., 10.])
        self.assertIsNotNone(choice['point_floor'])
        self.assertIsNone(choice['quantile_floor'])
        changed_test = self.forecast(price=-99999)
        changed_test.actual[:] = 50000
        adjusted = report.apply_floors(changed_test, choice)
        self.assertTrue(np.all(adjusted.point == choice['point_floor']))
        np.testing.assert_array_equal(adjusted.quantile, changed_test.quantile)
        self.assertEqual(choice, report.select_floors(validation, [-10., -5., 5., 10.]))

    def test_floor_ties_prefer_no_processing_and_never_positive_floor(self):
        choice = report.select_floors(self.forecast(), [1., 2., 3.])
        self.assertIsNone(choice['point_floor'])
        self.assertIsNone(choice['quantile_floor'])
        self.assertTrue(all(level is None or level <= 0 for level in choice['candidate_floors']))

    def test_training_price_reader_excludes_validation_and_test_extremes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'prices.csv'
            path.write_text('date,price\n2020-01-01T00:00:00Z,-10\n2020-02-01T00:00:00Z,5\n2021-01-01T00:00:00Z,-99999\n')
            prices = report._training_prices(str(path), 'date', 'price', 'UTC',
                                             '2020-01-01', '2021-01-01', None)
            np.testing.assert_array_equal(prices, [-10, 5])

    def test_matrix_requires_all_regions_and_seeds(self):
        jobs = [{'protocol': 'static', 'variant': 'full', 'region': region, 'seed': seed}
                for region in report.REGIONS for seed in report.SEEDS]
        self.assertEqual(len(report.matrix_groups(jobs)[('static', 'full')]), 9)
        with self.assertRaisesRegex(ValueError, 'region/seed'):
            report.matrix_groups(jobs[:-1])
        with self.assertRaisesRegex(ValueError, 'region/seed'):
            report.matrix_groups(jobs[:-1]+[jobs[0]])
        rolling = [dict(job, protocol='rolling/q0') for job in jobs]
        with self.assertRaisesRegex(ValueError, 'quarters'):
            report.matrix_groups(rolling)

    def test_event_controls_are_not_price_forecast_sources(self):
        self.assertEqual(dict(report.matrix_groups([{'baseline_kind':'event_controls'}])), {})


if __name__ == '__main__':
    unittest.main()
