import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('comparison', Path(__file__).parents[1] / 'scripts/compare_r12_main_table.py')
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def test_alignment_accepts_only_exact_float32_rounding():
    origins = np.array([0, 3600, 7200])
    times = origins[:, None] + np.arange(24) * 3600
    truth = np.full((3, 24), 117.5266666667)
    actual = truth.astype(np.float32).astype(float)
    indices, difference = comparison.align(origins, times, actual, origins[[0, 2]], truth[[0, 2]])
    assert indices.tolist() == [0, 2] and difference > 0
    actual[0, 0] += .01
    with pytest.raises(ValueError, match='labels differ'):
        comparison.align(origins, times, actual, origins[[0, 2]], truth[[0, 2]])


def test_alignment_rejects_missing_origins_and_shifted_targets():
    origins = np.array([0, 3600])
    times = origins[:, None] + np.arange(24) * 3600
    truth = np.zeros((2, 24))
    with pytest.raises(ValueError, match='missing requested'):
        comparison.align(origins, times, truth, np.array([7200]), truth[:1])
    with pytest.raises(ValueError, match='target times'):
        comparison.align(origins, times + 3600, truth, origins, truth)


def test_comparison_score_matches_repository_shared_scoring():
    from forecasting.aligned_scoring import make_prediction_artifact, score_predictions
    y = np.arange(24, dtype=float)[None, :]
    q = np.broadcast_to(np.array([30., -2., 5., 24., 10.]), (1, 24, 5)).copy()
    meta = dict(protocol_id='synthetic', split='test', region='NSW1', seed=2026,
                source='synthetic', input_hash='x', sample_hash='x', preprocessing_hash='x',
                output_type='quantile', prediction_space='raw_rrp')
    artifact = make_prediction_artifact(actual=y, quantiles=q, quantile_levels=comparison.TAUS,
        origin_unix=np.array([0]), target_time_unix=np.arange(24)[None, :] * 3600, metadata=meta)
    expected = score_predictions(artifact)
    actual = comparison.score(y, quantiles=q)
    for short, long in dict(mae='mae', qs='quantile_score_proxy', rmse='rmse_pooled',
                           c80='coverage_80', c90='coverage_90',
                           width80='interval_width_80', width90='interval_width_90').items():
        assert actual[short] == pytest.approx(expected[long], abs=1e-12)
    point = comparison.score(y, point=y + 2)
    assert point['mae'] == point['rmse'] == 2 and point['qs'] is None
