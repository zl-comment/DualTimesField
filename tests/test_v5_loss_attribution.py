"""Analytic checks for label-aware diagnostic bounds, not predictor training."""
import importlib.util
from pathlib import Path

import numpy as np


path = Path(__file__).resolve().parents[1] / 'scripts/diagnostics/v5_loss_attribution.py'
spec = importlib.util.spec_from_file_location('v5_attribution', path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_interval_oracle_rejects_wrong_sign_and_caps_unreachable_target():
    y = np.array([-5., 6., 1000., -1000.])
    base = np.zeros(4)
    low = np.array([0., 0., -150., -150.])
    high = np.array([10., 10., 150., 150.])
    np.testing.assert_array_equal(audit.interval_oracle(y, base, low, high), [0., 6., 150., -150.])


def test_interval_oracle_dominates_dense_gate_grid_even_with_saturation():
    rng = np.random.default_rng(2026)
    y, base = rng.normal(size=(2, 5, 24)) * 400
    trend = rng.uniform(-150, 150, (5, 24))
    event = rng.uniform(0, 150, (5, 24))
    low, high = np.minimum(trend, 0), np.clip(np.maximum(trend, 0) + event, -150, 150)
    best = np.abs(y-audit.interval_oracle(y, base, low, high))
    for gt in np.linspace(0, 1, 11):
        for ge in np.linspace(0, 1, 11):
            candidate = base + np.clip(gt*trend+ge*event, -150, 150)
            assert np.all(best <= np.abs(y-candidate) + 1e-9)


def test_subset_oracle_selects_whole_curve_not_different_subset_per_hour():
    base = np.zeros((1, 24))
    y = base.copy()
    y[0, 0] = 10
    payload = np.zeros((1, 4, 24))
    payload[0, 0, 0] = 10
    payload[0, 1, 1] = 20  # This distinct historical event is harmful.
    pred, subset = audit.subset_oracle(y, base, base, np.ones_like(base), payload)
    np.testing.assert_array_equal(pred, y)
    assert subset.tolist() == [1]
    # A single candidate has both a useful and harmful hour: cannot choose
    # only its useful hour. Either retain the entire curve or reject it.
    payload[0, 0, 1] = 30
    pred, subset = audit.subset_oracle(y, base, base, np.ones_like(base), payload)
    assert subset.tolist() == [0]
    np.testing.assert_array_equal(pred, base)


def test_band_contribution_is_population_weighted_not_conditional_mean():
    y = np.array([[0., 0., 0., 1000.]])
    before = np.zeros((1, 4, 5))
    after = before.copy()
    after[0, 3] = 100
    e = audit.effect(y, before, after, y > 300)
    assert e['conditional_mae_delta'] == -100
    assert e['mae_contribution'] == -25
