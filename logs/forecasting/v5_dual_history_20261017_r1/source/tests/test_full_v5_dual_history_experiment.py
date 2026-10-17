"""Real encoder-to-memory-to-forecast causality and input-scale checks."""
import numpy as np

from forecasting.full_v5.dual_history_experiment import price_lags
from forecasting.full_v5.dual_history_model import (
    DualHistoryConfig, FIT_CUTOFF, TRAIN_START, fit_historical_dualfield,
)
from forecasting.full_v5.dual_history_memory import build_corrections
from forecasting.full_v5.history_correction import apply_history_correction


def test_history_price_inverse_uses_existing_train_scale_and_column():
    values = np.arange(2 * 72 * 2).reshape(2, 72, 2).astype(float)
    meta = dict(price_transform="identity", target_mean=80., target_std=25.)
    cfg = dict(data=dict(history_columns=["demand", "price"], target_column="price"))
    out = price_lags(dict(history_values=values), meta, cfg)
    assert out.shape == (2, 24, 3)
    for k in range(3):
        np.testing.assert_array_equal(out[:, :, k], values[:, k * 24:(k + 1) * 24, 1] * 25 + 80)


def test_real_dualfield_pipeline_ignores_all_immature_targets():
    day = 86400
    origins = np.r_[TRAIN_START, FIT_CUTOFF - 2 * day,
                    FIT_CUTOFF + np.arange(30) * day].astype(np.int64)
    rng = np.random.default_rng(2026)
    actual = 80 + rng.normal(0, 30, (len(origins), 24))
    actual[:, 13] += 300
    z = rng.normal(0, .1, (len(origins), 24, 16))
    lag = rng.normal(80, 30, (len(origins), 24, 3))
    anchor = np.broadcast_to(np.array([20, 40, 80, 130, 180.]), (len(origins), 24, 5)).copy()
    model, _ = fit_historical_dualfield(origins, actual, z, lag,
        config=DualHistoryConfig(epochs=1, batch_size=2, hidden_dim=8, num_atoms=3, num_frequencies=2))
    before = build_corrections(origins, actual, anchor, z, lag, model)
    query = FIT_CUTOFF + 20 * day
    changed = actual.copy()
    changed[origins + day > query] = 1e6
    after = build_corrections(origins, changed, anchor, z, lag, model)
    keep = origins <= query
    for variant in before["corrections"]:
        np.testing.assert_array_equal(before["corrections"][variant][keep], after["corrections"][variant][keep])
    np.testing.assert_array_equal(before["matched_indices"][keep], after["matched_indices"][keep])
    q1 = apply_history_correction(anchor, before["corrections"]["dualfull"], [.5] * 4)
    q2 = apply_history_correction(anchor, after["corrections"]["dualfull"], [.5] * 4)
    np.testing.assert_array_equal(q1[keep], q2[keep])
    np.testing.assert_array_equal(apply_history_correction(anchor, before["corrections"]["dualfull"], [0] * 4), anchor)
    assert np.any(q1[keep] != anchor[keep])
    assert before["audit"]["maturity_violations"] == 0
