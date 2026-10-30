"""Metric definitions mirroring forecasting/aligned_scoring.py exactly."""
from __future__ import annotations

import numpy as np

from . import protocol


def score_arrays(quantiles: np.ndarray, actual: np.ndarray) -> dict:
    """quantiles [N,24,5], actual [N,24]; equal weight per cell, no clipping."""
    q = np.sort(np.asarray(quantiles, dtype=np.float64), axis=-1)
    y = np.asarray(actual, dtype=np.float64)
    mae = float(np.abs(q[:, :, 2] - y).mean())
    error = y[..., None] - q
    levels = np.asarray(protocol.QUANTILE_LEVELS)
    pinball = np.maximum(levels * error, (levels - 1) * error)
    coverage_80 = float(((y >= q[:, :, 1]) & (y <= q[:, :, 3])).mean())
    coverage_90 = float(((y >= q[:, :, 0]) & (y <= q[:, :, 4])).mean())
    return {
        "n_cells": int(y.size),
        "mae": mae,
        "quantile_score": float(2 * pinball.mean()),
        "pinball_by_quantile": {str(float(protocol.QUANTILE_LEVELS[i])): float(pinball[..., i].mean())
                                for i in range(len(protocol.QUANTILE_LEVELS))},
        "coverage_80": coverage_80,
        "coverage_90": coverage_90,
        "width_80": float((q[:, :, 3] - q[:, :, 1]).mean()),
        "width_90": float((q[:, :, 4] - q[:, :, 0]).mean()),
    }


def arm_metrics_with_failures(quantiles: np.ndarray, actual: np.ndarray,
                              exit_codes: np.ndarray, mask: np.ndarray) -> dict:
    """Score only successful cells; failures never score as successes."""
    success = (np.asarray(exit_codes) == 0) & np.asarray(mask, dtype=bool)
    result = {"n_origins": int(len(exit_codes)), "n_success": int(success.sum()),
              "n_fail": int((~success).sum())}
    if success.any():
        result.update(score_arrays(quantiles[success], actual[success]))
    return result
