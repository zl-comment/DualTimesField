"""Shared post-hoc calibration tools for aligned baselines.

``absolute_residual_cp`` is the exact formula previously defined in
``forecasting/lasso/model.py`` (moved here verbatim; the old import site
re-exports it, so existing callers and numerics are unchanged). ``fit_qra`` /
``apply_qra`` implement quantile regression averaging on frozen base quantile
forecasts for the 2026-10-22 baseline expansion.

Both tools are pure functions of their inputs: no split loading, no fitting of
preprocessing, and no access to evaluation labels.
"""
from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)


def absolute_residual_cp(point: np.ndarray, calibration_point: np.ndarray,
                         calibration_actual: np.ndarray,
                         quantile_levels: Any = DEFAULT_QUANTILES) -> np.ndarray:
    """Author-style symmetric CP, per horizon, with type-7 interpolation.

    This reproduces the absolute-residual formula, not a finite-sample coverage
    guarantee under overlapping hourly observations. Calibration must be OOS;
    its chronological checks are enforced by the calling experiment layers.
    Reference: lipiecki/PostForecasts.jl src/models/cp.jl at 589f940.
    """
    from ..aligned_baselines import _levels
    q = _levels(quantile_levels)
    point, predicted, actual = map(lambda x: np.asarray(x, dtype=float),
                                   (point, calibration_point, calibration_actual))
    if point.ndim != 2 or predicted.ndim != 2 or predicted.shape != actual.shape or not len(actual) or predicted.shape[1] != point.shape[1]:
        raise ValueError("CP requires matching nonempty [origin, horizon] calibration arrays")
    if not all(np.isfinite(x).all() for x in (point, predicted, actual)):
        raise ValueError("CP inputs must be finite")
    widths = np.quantile(np.abs(actual - predicted), np.abs(2 * q - 1), axis=0, method="linear").T
    return point[..., None] + widths[None, ...] * np.sign(q - 0.5)


def _qra_design(features: np.ndarray) -> np.ndarray:
    """Deterministic design matrix [1, q05, q10, q50, q90, q95] per horizon."""
    if features.ndim != 3 or features.shape[0] < 1 or features.shape[2] < 1:
        raise ValueError("QRA features must be [origin, horizon, base_level] arrays")
    ones = np.ones(features.shape[:2] + (1,), dtype=np.float64)
    return np.concatenate([ones, features.astype(np.float64)], axis=2)


def fit_qra(calibration_features: np.ndarray, calibration_actual: np.ndarray,
            quantile_levels: Any = DEFAULT_QUANTILES, *, alpha: float = 0.0,
            solver: str = "highs") -> dict[str, Any]:
    """Per-horizon, per-level linear quantile regression averaging (QRA).

    First-version local adaptation: ordinary (unregularized when alpha=0)
    linear pinball regression of the calibration labels on the frozen base
    model's quantile forecasts, solved as an exact linear program. Feature
    order is the ascending base level order; intercept first in the returned
    coefficients. Deterministic: fixed solver, no randomness. Solver failures
    are raised, never silently smoothed.
    """
    from ..aligned_baselines import _levels
    from sklearn.linear_model import QuantileRegressor

    q = _levels(quantile_levels)
    features, actual = np.asarray(calibration_features), np.asarray(calibration_actual)
    if actual.shape != features.shape[:2] or not np.isfinite(features).all() or not np.isfinite(actual).all():
        raise ValueError("QRA calibration features and actuals must be finite and aligned")
    design = _qra_design(features)
    n, h, _ = design.shape
    coefficients = np.zeros((h, len(q), design.shape[2]), dtype=np.float64)
    for horizon in range(h):
        x_h = design[:, horizon, :]
        for level_index, level in enumerate(q):
            model = QuantileRegressor(quantile=float(level), alpha=float(alpha),
                                      fit_intercept=False, solver=solver)
            model.fit(x_h, actual[:, horizon])
            weights = model.coef_.astype(np.float64)
            if not np.isfinite(weights).all():
                raise FloatingPointError(f"Non-finite QRA coefficients at horizon {horizon}, level {level}")
            coefficients[horizon, level_index] = weights
    return {"coefficients": coefficients, "alpha": float(alpha), "solver": solver,
            "feature_order": ["intercept", *[f"base_q{float(level):g}" for level in q]],
            "quantile_levels": q.tolist(), "calibration_rows": int(n),
            "design": "intercept_plus_ascending_base_quantiles"}


def apply_qra(fit: dict[str, Any], features: np.ndarray) -> np.ndarray:
    """Apply stored QRA coefficients to base quantile features [N, H, F] -> [N, H, Q]."""
    design = _qra_design(np.asarray(features))
    coefficients = np.asarray(fit["coefficients"], dtype=np.float64)
    if coefficients.shape[0] != design.shape[1] or coefficients.shape[2] != design.shape[2]:
        raise ValueError("QRA coefficients do not match the provided features")
    return np.einsum("nhf,hqf->nhq", design, coefficients)
