"""Scoring, slices, distribution summaries and paired block bootstrap.

All headline metrics come from the public scorer (forecasting.aligned_scoring);
nothing here modifies it.  QS is twice the mean five-quantile pinball (a proxy,
not an exact CRPS); RMSE is reported both pooled and as a macro average of
regional RMSEs, named separately.
"""
from __future__ import annotations

import numpy as np

from ..aligned_scoring import QUANTILE_LEVELS, make_prediction_artifact, score_predictions
from . import schema

HOUR = 3600


def build_artifact(model_id: str, quantiles: np.ndarray, arrays: dict, metadata: dict,
                   segment: str) -> dict:
    """Wrap raw quantiles in the public prediction contract and score them."""
    quantiles = np.asarray(quantiles, dtype=np.float64)
    n = len(quantiles)
    meta = dict(metadata)
    meta.update(source=model_id, seed=2026, split=f"v5_{segment}",
                output_type="quantile", prediction_space="raw_rrp")
    artifact = make_prediction_artifact(
        actual=arrays["actual"], origin_unix=arrays["origin_unix"],
        target_time_unix=arrays["target_time_unix"], metadata=meta,
        quantiles=quantiles, quantile_levels=QUANTILE_LEVELS)
    return dict(artifact=artifact, scores=score_predictions(artifact))


def band_decomposition(q50: np.ndarray, actual: np.ndarray) -> dict:
    """Conditional MAE/bias per price band plus whole-sample contributions.

    contribution_band = mean over ALL cells of |q50 - Y| * 1[band], so the three
    contributions sum exactly to the overall MAE.
    """
    error = q50 - actual
    abs_error = np.abs(error)
    total = float(abs_error.mean())
    bands = {}
    masks = dict(negative=actual < 0, range_0_300=(actual >= 0) & (actual <= 300), above_300=actual > 300)
    for name, mask in masks.items():
        sel = mask
        bands[name] = dict(
            n_cells=int(sel.sum()),
            conditional_mae=float(abs_error[sel].mean()) if sel.any() else None,
            conditional_bias=float(error[sel].mean()) if sel.any() else None,
            mae_contribution=float((abs_error * sel).mean()),
            fraction=float(sel.mean()),
        )
    bands["mae_total"] = total
    bands["contribution_sum"] = float(sum(bands[name]["mae_contribution"] for name in masks))
    return bands


def coverage_diagnostics(quantiles: np.ndarray, actual: np.ndarray) -> dict:
    q = np.sort(np.asarray(quantiles, dtype=np.float64), axis=-1)
    out = {}
    for pct, lo, hi in ((80, 1, 3), (90, 0, 4)):
        out[f"coverage_{pct}"] = float(((actual >= q[..., lo]) & (actual <= q[..., hi])).mean())
        out[f"interval_width_{pct}"] = float((q[..., hi] - q[..., lo]).mean())
    return out


def spike_alarm_diagnostics(quantiles: np.ndarray, actual: np.ndarray,
                            threshold: float = 300.0) -> dict:
    """Descriptive q50>300 alarm statistics; no gate and no threshold search."""
    q = np.sort(np.asarray(quantiles, dtype=np.float64), axis=-1)
    alarm = q[..., 2] > threshold
    truth = actual > threshold
    tp = float((alarm & truth).sum())
    out = dict(
        threshold=threshold,
        recall=float(tp / truth.sum()) if truth.any() else None,
        precision=float(tp / alarm.sum()) if alarm.any() else None,
        false_alarm_cells=int((alarm & ~truth).sum()),
        alarm_cells=int(alarm.sum()),
        true_cells=int(truth.sum()),
        q95_ranking_score_note="q95 values are retained as the fixed ranking score for any alarm ordering",
    )
    return out


def horizon_group_metrics(quantiles: np.ndarray, actual: np.ndarray) -> dict:
    q = np.sort(np.asarray(quantiles, dtype=np.float64), axis=-1)
    error = q[..., 2] - actual
    out = {}
    for g, (lo, hi) in enumerate(schema.HORIZON_GROUPS):
        sel = slice(lo, hi + 1)
        residual = actual[:, sel, None] - q[:, sel]
        pinball = np.maximum(QUANTILE_LEVELS * residual, (QUANTILE_LEVELS - 1.0) * residual)
        out[f"group_{g}_h{lo}_{hi}"] = dict(
            mae=float(np.abs(error[:, sel]).mean()),
            qs=float(2 * pinball.mean()),
            rmse=float(np.sqrt(np.square(error[:, sel]).mean())),
        )
    return out


def macro_and_pooled_rmse(quantiles: np.ndarray, actual: np.ndarray, region: str,
                          accumulator: dict) -> dict:
    """Per-region RMSE plus the cross-region pooled RMSE, named separately."""
    q = np.sort(np.asarray(quantiles, dtype=np.float64), axis=-1)
    error = q[..., 2] - actual
    accumulator.setdefault("squared_sum", 0.0)
    accumulator.setdefault("n_cells", 0)
    accumulator["squared_sum"] += float(np.square(error).sum())
    accumulator["n_cells"] += error.size
    return dict(region=region, rmse_region=float(np.sqrt(np.square(error).mean())))


def pooled_rmse(accumulator: dict) -> float:
    return float(np.sqrt(accumulator["squared_sum"] / accumulator["n_cells"]))


def calibrator_diagnostics(parts: dict, rows: np.ndarray) -> dict:
    """delta/s summaries and boundary-touch rates for the fitted rows."""
    delta = np.asarray(parts["delta"], dtype=np.float64)[rows]
    s = np.asarray(parts["s"], dtype=np.float64)[rows]
    return dict(
        delta_mean=float(delta.mean()), delta_std=float(delta.std()),
        delta_abs_max=float(np.abs(delta).max()),
        delta_at_bound_fraction=float((np.abs(np.abs(delta) - schema.DELTA_SCALE)
                                       <= 1e-6 * schema.DELTA_SCALE).mean()),
        s_mean=float(s.mean()), s_std=float(s.std()),
        s_at_bound_fraction=float(((s <= schema.S_MIN + 1e-6) | (s >= schema.S_MAX - 1e-6)).mean()),
        s_min=float(s.min()), s_max=float(s.max()),
    )


def feature_distribution_summary(features: np.ndarray, rows: np.ndarray, name: str) -> dict:
    values = np.asarray(features, dtype=np.float64)[rows]
    quantiles = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    return dict(name=name, shape=list(values.shape),
                column_quantiles={str(q): np.quantile(values.reshape(-1, values.shape[-1]), q, axis=0).tolist()
                                  for q in quantiles})


# --- paired block bootstrap ------------------------------------------------
def per_origin_losses(quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """[N] per-origin mean over the 24 horizons of (MAE loss, QS loss)."""
    q = np.sort(np.asarray(quantiles, dtype=np.float64), axis=-1)
    y = np.asarray(actual, dtype=np.float64)
    mae = np.abs(q[..., 2] - y).mean(axis=1)
    residual = y[:, :, None] - q
    pinball = np.maximum(QUANTILE_LEVELS * residual, (QUANTILE_LEVELS - 1.0) * residual)
    qs = 2.0 * pinball.mean(axis=(1, 2))
    return np.stack([mae, qs], axis=1)


def _block_index(origins: np.ndarray, block_hours: int, t0: int | None = None):
    origins = np.asarray(origins, dtype=np.int64)
    if t0 is None:
        t0 = int(origins[0])
    return (origins - t0) // (block_hours * HOUR)


def paired_block_bootstrap(origins_by_region: dict, losses_ref: dict, losses_cand: dict,
                           block_hours: int = 168, n_boot: int = 9999, seed: int = 2026) -> dict:
    """Joint three-region paired bootstrap of mean-loss differences.

    Effect convention (fixed end-to-end): ``point_difference = mean(loss_ref) -
    mean(loss_cand)``, so a POSITIVE difference means the candidate has the
    LOWER loss (better).  losses_*: region -> [n_origins, 2] per-origin (MAE,
    QS) losses; origins_by_region: region -> [n_origins] origin unix times.
    All regions share the hourly origin grid, so one set of resampled 168h
    block ids is applied to every region; block means are macro-averaged
    across regions.  Each bootstrap draw renormalises by the total number of
    origins actually drawn (partial and tail blocks included), so a constant
    loss difference reproduces its exact value in every draw.
    """
    regions = sorted(losses_ref)
    if regions != sorted(losses_cand) or regions != sorted(origins_by_region):
        raise ValueError("Bootstrap requires identical region sets")
    reference = origins_by_region[regions[0]]
    n_origins = len(reference)
    diffs = []
    for region in regions:
        ref, cand = losses_ref[region], losses_cand[region]
        origins = np.asarray(origins_by_region[region], dtype=np.int64)
        if len(ref) != n_origins or len(cand) != n_origins or len(origins) != n_origins:
            raise ValueError(f"Region {region} origin grid mismatch")
        if not np.array_equal(origins, reference):
            raise ValueError("Regions must share the same origin grid for joint block sampling")
        diffs.append(np.asarray(ref) - np.asarray(cand))  # [n_origins, 2], ref - cand
    diff = np.stack(diffs, axis=0)                   # [R, N, 2]
    block_ids = _block_index(reference, block_hours)
    n_blocks = int(block_ids.max()) + 1
    block_means = np.zeros((n_blocks, diff.shape[0], diff.shape[2]), dtype=np.float64)
    block_counts = np.zeros(n_blocks, dtype=np.float64)
    for b in range(n_blocks):
        sel = block_ids == b
        block_means[b] = diff[:, sel].mean(axis=1)
        block_counts[b] = sel.sum()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    drawn_means = block_means[draws]                                # [n_boot, n_blocks, R, 2]
    drawn_counts = block_counts[draws]                              # [n_boot, n_blocks]
    weights = drawn_counts / drawn_counts.sum(axis=1, keepdims=True)
    boot = (drawn_means * weights[..., None, None]).sum(axis=1)     # [n_boot, R, 2]
    boot = boot.mean(axis=1)                                        # [n_boot, 2]
    point = diff.mean(axis=(0, 1))
    lower = np.quantile(boot, 0.025, axis=0)
    upper = np.quantile(boot, 0.975, axis=0)
    p_two = np.minimum(1.0, 2.0 * np.minimum((boot <= 0).mean(axis=0), (boot >= 0).mean(axis=0)))
    return dict(
        block_hours=block_hours, n_blocks=n_blocks, n_boot=n_boot, seed=seed,
        effect_convention="positive = candidate (second argument) better: mean(loss_ref) - mean(loss_cand)",
        point_difference=dict(mae=float(point[0]), qs=float(point[1])),
        ci95=dict(mae=[float(lower[0]), float(upper[0])], qs=[float(lower[1]), float(upper[1])]),
        p_value=dict(mae=float(p_two[0]), qs=float(p_two[1])),
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values over the pre-declared family."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.zeros(m)
    running = 0.0
    for rank, index in enumerate(order):
        value = (m - rank) * p_values[index]
        running = max(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def oracle_stacker_upper_bound(baseline_q: np.ndarray, v4_q: np.ndarray, actual: np.ndarray) -> dict:
    """In-segment per-horizon convex oracle; purely descriptive."""
    from .calibrator import fit_stacker
    rows = np.ones(len(actual), dtype=bool)
    oracle = fit_stacker(baseline_q, v4_q, actual, rows)
    q = oracle.predict(baseline_q, v4_q)
    mae = float(np.abs(np.sort(q, axis=-1)[..., 2] - actual).mean())
    return dict(weights={g: float(w) for g, w in oracle.weights.items()}, in_segment_mae=mae,
                note="in-segment oracle; descriptive upper bound only, not deployable")
