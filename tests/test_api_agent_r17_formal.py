import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import test_api_agent_r15 as fixtures
from forecasting.api_agent.formal_r17 import (candidate_batch, checked_shard, load_base,
    paired_block_interval, per_origin_qs, require_complete, write_json)
from forecasting.api_agent.pilot import sha_file
from forecasting.api_agent.workflow import quantile_loss


class FormalRunnerTests(unittest.TestCase):
    def test_base_loading_requires_no_actual_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'base.npz'
            meta = {k: k for k in ('protocol_id', 'split', 'region', 'input_hash', 'sample_hash', 'preprocessing_hash')}
            axes = {'origin_unix': np.array([0]), 'target_time_unix': np.arange(24)[None] * 3600}
            q = np.tile(np.arange(5.), (1, 24, 1))
            np.savez(path, metadata_json=json.dumps(meta), **axes, quantiles=q)
            np.testing.assert_array_equal(load_base({'path': str(path)}, meta, axes), q)
            axes['origin_unix'] = np.array([1])
            with self.assertRaisesRegex(ValueError, 'axis'):
                load_base({'path': str(path)}, meta, axes)

    def test_batch_reuses_base_and_keeps_each_origin_independent(self):
        s = fixtures.snapshot()
        base = np.tile(np.arange(5.), (2, 24, 1))
        calls = []
        def predict(pipeline, rows, **kwargs):
            calls.append(rows)
            return np.stack([np.full((24, 5), i) for i in range(len(rows))])
        with patch('forecasting.chronos2_zeroshot.model.predict_common_quantiles', side_effect=predict):
            f, h = candidate_batch([s, s], base, None, 32)
        self.assertIs(f['base'], base)
        self.assertEqual([len(rows) for rows in calls], [2, 4, 4])
        self.assertEqual(len(calls[0][0]['target']), 48)
        np.testing.assert_array_equal(f['calendar'][:, 0, 0], [0, 2])
        np.testing.assert_array_equal(h['calendar'][:, 0, 0], [1, 3])

    def test_shard_tampering_or_release_change_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            data = directory / 'data'
            data.write_text('valid')
            write_json(directory / 'complete.json', {'source_fingerprint': 'release',
                                                     'files': {'data': sha_file(data)}})
            self.assertTrue(checked_shard(directory, 'release'))
            with self.assertRaises(ValueError):
                checked_shard(directory, 'other-release')
            data.write_text('corrupt')
            with self.assertRaises(ValueError):
                checked_shard(directory, 'release')

    def test_missing_cell_blocks_scoring_before_label_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                require_complete(Path(tmp), {'seeds': [2026, 2027, 2028],
                    'regions': ['NSW1', 'QLD1', 'TAS1'], 'origins_per_region': 2, 'shard_size': 1}, 'x')

    def test_paired_loss_matches_shared_definition_even_with_crossing(self):
        y = np.arange(48.).reshape(2, 24)
        q = np.broadcast_to(np.array([8., 2., 9., 11., 4.]), (2, 24, 5))
        self.assertAlmostEqual(float(per_origin_qs(y, q).mean()), quantile_loss(y, q))
        result = paired_block_interval(np.full(48, -2.), block_hours=24, draws=100)
        self.assertEqual(result['percentile_95_interval'], [-2., -2.])


if __name__ == '__main__':
    unittest.main()
