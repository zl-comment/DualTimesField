"""Calibration tool contracts: CP numerics unchanged, QRA exactness and errors."""
import numpy as np
import pytest

from forecasting.common.calibration import absolute_residual_cp, apply_qra, fit_qra


def test_cp_matches_hand_computed_type7_widths_and_keeps_q50_exact():
    levels = (0.05, 0.10, 0.50, 0.90, 0.95)
    calibration_actual = np.array([[10.0, 100.0], [12.0, 90.0], [14.0, 110.0]])
    calibration_point = np.array([[9.0, 95.0], [11.0, 100.0], [15.0, 100.0]])
    point = np.array([[10.0, 100.0]])
    out = absolute_residual_cp(point, calibration_point, calibration_actual, levels)
    residual_h0 = np.abs(np.array([1.0, 1.0, -1.0]))           # {1,1,1}
    residual_h1 = np.abs(np.array([5.0, -10.0, 10.0]))          # {5,10,10}
    for index, level in enumerate(levels):
        expected_h0 = np.quantile(residual_h0, abs(2 * level - 1), method="linear")
        expected_h1 = np.quantile(residual_h1, abs(2 * level - 1), method="linear")
        np.testing.assert_allclose(out[0, 0, index], 10.0 + expected_h0 * np.sign(level - 0.5))
        np.testing.assert_allclose(out[0, 1, index], 100.0 + expected_h1 * np.sign(level - 0.5))
    assert out[0, 0, 2] == 10.0 and out[0, 1, 2] == 100.0  # q50 centre untouched
    assert out.shape[-1] == 5
    np.testing.assert_array_equal(np.diff(out, axis=-1) >= 0, True)  # monotone bands


def test_cp_rejects_mismatched_or_nonfinite_inputs():
    levels = (0.05, 0.5, 0.95)
    point = np.zeros((2, 3))
    with pytest.raises(ValueError):
        absolute_residual_cp(point, np.zeros((2, 2)), np.zeros((2, 3)), levels)
    bad = np.zeros((2, 3)); bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        absolute_residual_cp(point, bad, np.zeros((2, 3)), levels)


def test_moved_cp_function_is_bitwise_identical_to_lasso_import_path():
    from forecasting.lasso.model import absolute_residual_cp as legacy

    rng = np.random.default_rng(7)
    actual, predicted = rng.normal(size=(40, 6)) + 50, rng.normal(size=(40, 6)) + 50
    point = rng.normal(size=(3, 6)) + 50
    levels = (0.05, 0.1, 0.5, 0.9, 0.95)
    np.testing.assert_array_equal(absolute_residual_cp(point, predicted, actual, levels),
                                  legacy(point, predicted, actual, levels))


def test_qra_recovers_exact_linear_relation_and_is_deterministic():
    levels = (0.05, 0.5, 0.95)
    rng = np.random.default_rng(3)
    features = rng.normal(size=(200, 4, 3))
    # Exact affine relation per (horizon, level): y = 2*q_level + 1.
    actual = 2.0 * features[..., 0] + 1.0
    first = fit_qra(features, actual, levels)
    second = fit_qra(features, actual, levels)
    assert first["alpha"] == 0.0 and first["solver"] == "highs"
    np.testing.assert_array_equal(first["coefficients"], second["coefficients"])
    np.testing.assert_allclose(first["coefficients"][:, :, 1], 2.0, atol=1e-6)
    np.testing.assert_allclose(first["coefficients"][:, :, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(first["coefficients"][:, :, 2:], 0.0, atol=1e-6)
    applied = apply_qra(first, features)
    np.testing.assert_allclose(applied, np.broadcast_to(actual[..., None], applied.shape), atol=1e-6)


def test_qra_design_and_apply_reject_shape_mismatch():
    levels = (0.05, 0.5, 0.95)
    rng = np.random.default_rng(11)
    features = rng.normal(size=(30, 2, 3))
    fit = fit_qra(features, rng.normal(size=(30, 2)), levels)
    with pytest.raises(ValueError):
        apply_qra(fit, rng.normal(size=(30, 2, 4)))
    with pytest.raises(ValueError):
        fit_qra(np.ones((30, 2, 3)), np.full((30, 2), np.nan), levels)
