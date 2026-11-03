"""Fixed paired bootstrap: numerical invariants and complete-grid rejection."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.r20_paired_uncertainty import analyze, main, preregistration, validate_daily_grid


def fixture(delta=0., variable=True):
    rows = []
    for day in range(181):
        for j, region in enumerate(('NSW1', 'QLD1', 'TAS1')):
            reference = 20. + (day * .25 + j * 5 if variable else 0)
            treatment = reference + delta
            rows.append({'region': region, 'origin_unix': 1640959200 + day * 86400,
                         'r19_qs': reference, 'r20_qs': treatment,
                         'qs_delta_r20_minus_r19': treatment - reference})
    return {'status': 'COMPLETE_PAIRED_DEVELOPMENT_COMPARISON', 'split': 'selection',
            'seed': 2026, 'origins': 543, 'per_origin': rows}


class PairedUncertaintyTests(unittest.TestCase):
    def test_same_arms_have_exact_zero_difference_and_intervals(self):
        result = analyze(fixture())
        self.assertEqual(result['mean_qs_delta_r20_minus_r19'], 0)
        self.assertEqual(result['mean_qs_delta_percentile_95_interval'], [0., 0.])
        self.assertEqual(result['relative_improvement_percentile_95_interval'], [0., 0.])
        self.assertEqual(result['bootstrap_replicates_completed'], 2000)

    def test_constant_difference_survives_paired_resampling_of_variable_reference(self):
        data = fixture(delta=-2.)
        result = analyze(data)
        self.assertEqual(result['mean_qs_delta_r20_minus_r19'], -2.)
        self.assertEqual(result['mean_qs_delta_percentile_95_interval'], [-2., -2.])
        expected_reference = np.mean([row['r19_qs'] for row in data['per_origin']])
        self.assertAlmostEqual(result['relative_improvement_percent'], 200 / expected_reference)
        low, high = result['relative_improvement_percentile_95_interval']
        self.assertGreater(high, low, 'Relative uncertainty reflects the resampled reference denominator')
        self.assertGreater(low, 0)

    def test_constant_scores_have_known_exact_relative_interval(self):
        result = analyze(fixture(delta=-2., variable=False))
        self.assertEqual(result['mean_qs'], {'R19': 20., 'R20': 18.})
        self.assertEqual(result['relative_improvement_percent'], 10.)
        self.assertEqual(result['relative_improvement_percentile_95_interval'], [10., 10.])

    def test_seed_is_reproducible_and_input_order_does_not_change_day_pairing(self):
        data = fixture(delta=-1.)
        first = analyze(data)
        data['per_origin'].reverse()
        self.assertEqual(analyze(data), first)
        self.assertEqual(preregistration()['block_days'], 7)
        self.assertEqual(preregistration()['rng_seed'], 2026)
        self.assertEqual(preregistration()['percentiles'], [2.5, 97.5])

    def test_missing_duplicate_and_misaligned_region_days_are_rejected(self):
        missing = fixture(); missing['per_origin'].pop()
        duplicate = fixture(); duplicate['per_origin'][1] = deepcopy(duplicate['per_origin'][0])
        misaligned = fixture(); misaligned['per_origin'][10]['origin_unix'] += 3600
        irregular = fixture()
        for row in irregular['per_origin']:
            if row['origin_unix'] == 1640959200 + 10 * 86400:
                row['origin_unix'] += 3600
        wrong_count = fixture(); wrong_count['origins'] = 542
        for data in (missing, duplicate, misaligned, irregular, wrong_count):
            with self.subTest(kind=data['origins']), self.assertRaises(ValueError):
                validate_daily_grid(data)

    def test_incomplete_wrong_scope_and_corrupted_scores_are_rejected(self):
        pending = fixture(); pending['status'] = 'PREDICTING'
        test_split = fixture(); test_split['split'] = 'test'
        other_seed = fixture(); other_seed['seed'] = 2027
        nan = fixture(); nan['per_origin'][0]['r19_qs'] = float('nan')
        wrong_delta = fixture(); wrong_delta['per_origin'][0]['qs_delta_r20_minus_r19'] = 5
        negative = fixture(); negative['per_origin'][0]['r19_qs'] = -2
        for data in (pending, test_split, other_seed, nan, wrong_delta, negative):
            with self.subTest(status=data['status']), self.assertRaises(ValueError):
                analyze(data)

    def test_zero_reference_denominator_and_wrong_summary_are_rejected(self):
        data = fixture(variable=False)
        for row in data['per_origin']:
            row.update(r19_qs=0., r20_qs=0., qs_delta_r20_minus_r19=0.)
        with self.assertRaisesRegex(ValueError, 'zero mean'):
            analyze(data)
        data = fixture(); data['region_equal_mean'] = {'R19': {'qs': 1}, 'R20': {'qs': 1}}
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            analyze(data)

    def test_preregistration_does_not_analyze_or_open_a_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'preregister.json'
            with patch('scripts.r20_paired_uncertainty.analyze', side_effect=AssertionError('Results read')):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(['--preregister', '--output', str(output)]), 0)
            saved = json.loads(output.read_text())
            self.assertEqual(saved['status'], 'PREREGISTERED_NO_RESULTS_READ')
            self.assertFalse(saved['parameters']['preregister_mode_reads_comparison'])
            self.assertEqual(saved['parameters']['bootstrap_replicates'], 2000)
            with self.assertRaises(FileExistsError):
                main(['--preregister', '--output', str(output)])


if __name__ == '__main__':
    unittest.main()
