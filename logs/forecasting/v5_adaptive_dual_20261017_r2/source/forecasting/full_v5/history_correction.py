"""Exploratory history-only correction of a frozen Dfit C_roll model.

The reference model is never fitted here. Its out-of-sample residual library
starts on 2022-10-01, after its fitting cutoff. Each query reads only fully
matured 24-hour outcomes, and fitting the four shrinkage weights uses V1
labels available strictly before December. This is a new development split,
not a replacement for the original V5 acceptance gates.
"""
from __future__ import annotations

import numpy as np

from . import schema

HOUR = 3600
DAY = 24 * HOUR
HISTORY_START = schema.unix_timestamp("2022-10-01")
FIT_CUTOFF = schema.unix_timestamp("2022-12-01")
MAX_CORRECTION = 150.0


def _arrays(origins, actual, frozen_quantiles):
    origins = np.asarray(origins, dtype=np.int64)
    actual = np.asarray(actual, dtype=np.float64)
    quantiles = np.asarray(frozen_quantiles, dtype=np.float64)
    n = len(origins)
    if origins.ndim != 1 or np.any(np.diff(origins) <= 0):
        raise ValueError("Origins must be a strictly increasing vector")
    if actual.shape != (n, schema.HORIZONS) or quantiles.shape != (n, schema.HORIZONS, 5):
        raise ValueError("Expected actual [N,24] and frozen quantiles [N,24,5]")
    if not np.isfinite(actual).all() or not np.isfinite(quantiles).all():
        raise ValueError("Non-finite actual or frozen prediction")
    if np.any(np.diff(quantiles, axis=-1) < 0):
        raise ValueError("Frozen quantiles must be ordered")
    return origins, actual, quantiles


def compute_history_corrections(origins, actual, frozen_quantiles, z, *, history_start=None,
                                policy="state"):
    """Compute matched/recent corrections [N,24] without fitted parameters.

    Library records before ``history_start`` are excluded even if supplied.
    States retain the existing six fixed V5 coordinates. Same-hour candidates
    and full-window maturity match the original availability protocol.
    """
    origins, actual, quantiles = _arrays(origins, actual, frozen_quantiles)
    start = HISTORY_START if history_start is None else int(history_start)
    if policy not in ("state", "fresh28", "recent_consistent"):
        raise ValueError("Unknown fixed history policy")
    lookback = schema.LOOKBACK_DAYS if policy == "state" else schema.WINDOW_LONG_DAYS
    if start < HISTORY_START:
        raise ValueError("History cannot predate the frozen C_roll fitting cutoff")
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (len(origins), schema.HORIZONS, schema.NZ) or not np.isfinite(z).all():
        raise ValueError("Expected finite z [N,24,16]")
    state = z[:, :, list(schema.MATCH_Z_INDICES)]
    residual = actual - quantiles[:, :, 2]
    matched = np.zeros_like(actual)
    recent = np.zeros_like(actual)
    hours = (origins // HOUR + 10) % 24
    pool_sizes = []
    opened = 0
    samples = []
    guarded_cells = 0
    for i, t in enumerate(origins):
        lo = max(np.searchsorted(origins, start, side="left"),
                 np.searchsorted(origins, t - lookback * DAY, side="right"))
        hi = np.searchsorted(origins, t - schema.HORIZONS * HOUR, side="right")
        pool = np.arange(lo, hi, dtype=np.int64)
        pool = pool[hours[pool] == hours[i]]
        pool_sizes.append(len(pool))
        if len(pool) < schema.MATCH_MIN_CANDIDATES:
            continue
        opened += 1
        k = min(schema.MATCH_K, len(pool))
        eta = k / schema.MATCH_K
        recent[i] = np.clip(eta * np.median(residual[pool[-k:]], axis=0),
                            -MAX_CORRECTION, MAX_CORRECTION)
        for h in range(schema.HORIZONS):
            distances = np.sum((state[pool, h] - state[i, h]) ** 2, axis=1)
            chosen = pool[np.argsort(distances, kind="stable")[:k]]
            matched[i, h] = np.clip(eta * np.median(residual[chosen, h]),
                                    -MAX_CORRECTION, MAX_CORRECTION)
            if policy == "recent_consistent":
                # Similar states can belong to a stale residual regime. Require
                # directional support from recent mature errors; disagreement
                # selects the null branch, agreement caps extrapolation.
                proposed, reference = matched[i, h], recent[i, h]
                trusted = (np.sign(proposed) * min(abs(proposed), abs(reference))
                           if proposed * reference > 0 else 0.0)
                guarded_cells += int(trusted != proposed)
                matched[i, h] = trusted
            if h == 0 and len(samples) < 16 and opened % 96 == 1:
                samples.append(dict(query_origin=int(t),
                                    selected_origins=origins[chosen].tolist()))
    return dict(matched=matched, recent=recent, audit=dict(
        history_start=start, residual_definition="actual - frozen_C_roll_q50",
        policy=policy, lookback_days=lookback, guarded_cells=guarded_cells,
        queries=len(origins), open_queries=opened,
        open_fraction=opened / max(1, len(origins)),
        eligible_library_records=int(np.sum(origins >= start)),
        pool_size_max=max(pool_sizes, default=0), selection_samples=samples,
        max_abs_correction=MAX_CORRECTION))


def _loss(actual, quantiles):
    r = actual[..., None] - quantiles
    tau = np.asarray(schema.QUANTILE_LEVELS)
    return float(np.sum(np.maximum(tau * r, (tau - 1.0) * r)
                        * np.asarray(schema.LOSS_WEIGHTS)))


def _fit_scalar(actual, quantiles, correction):
    """Exact bounded convex pinball minimizer using weighted breakpoints."""
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    quantiles = np.asarray(quantiles, dtype=np.float64).reshape(-1, 5)
    correction = np.asarray(correction, dtype=np.float64).reshape(-1)
    active = correction != 0
    loss_zero = _loss(actual, quantiles)
    if not np.any(active):
        return 0.0, loss_zero, loss_zero
    d = correction[active, None]
    breaks = ((actual[active, None] - quantiles[active]) / d).ravel()
    mass = (np.abs(d) * np.asarray(schema.LOSS_WEIGHTS)[None, :]).ravel()
    tau = np.where(d > 0, np.asarray(schema.QUANTILE_LEVELS),
                   1.0 - np.asarray(schema.QUANTILE_LEVELS))
    threshold = float(np.sum(mass * tau.ravel()))
    order = np.argsort(breaks, kind="stable")
    pos = min(int(np.searchsorted(np.cumsum(mass[order]), threshold, side="left")),
              len(order) - 1)
    alpha = float(np.clip(breaks[order[pos]], 0.0, 1.0))
    loss_fit = _loss(actual, quantiles + alpha * correction[:, None])
    # Null is preferred for equal objectives, including flat intervals.
    if loss_fit >= loss_zero - 1e-12 * max(1.0, abs(loss_zero)):
        return 0.0, loss_zero, loss_zero
    return alpha, loss_zero, loss_fit


def fit_history_weights(origins, actual, frozen_quantiles, correction, *,
                        fit_start=None, fit_cutoff=None):
    """Fit four alpha in [0,1] on V1, never consuming December fit labels.

    The caller must provide history corrections created from the frozen
    pre-October model. Full arrays may be passed; row selection and strict
    target maturity are enforced here, rather than entrusted to the caller.
    """
    origins, actual, quantiles = _arrays(origins, actual, frozen_quantiles)
    start = HISTORY_START if fit_start is None else int(fit_start)
    cutoff = FIT_CUTOFF if fit_cutoff is None else int(fit_cutoff)
    if not HISTORY_START <= start < cutoff <= FIT_CUTOFF:
        raise ValueError("Correction fitting must stay within October-November V1")
    correction = np.asarray(correction, dtype=np.float64)
    if correction.shape != actual.shape or not np.isfinite(correction).all():
        raise ValueError("Expected finite correction [N,24]")
    if np.any(np.abs(correction) > MAX_CORRECTION):
        raise ValueError("History correction exceeds its fixed bound")
    rows = (origins >= start) & (origins + schema.HORIZONS * HOUR < cutoff)
    if not np.any(rows):
        raise ValueError("No fully matured V1 fitting rows")
    alpha, groups = [], []
    for lo, hi in schema.HORIZON_GROUPS:
        hs = slice(lo, hi + 1)
        a, before, after = _fit_scalar(actual[rows, hs], quantiles[rows, hs], correction[rows, hs])
        alpha.append(a)
        groups.append(dict(alpha=a, objective_zero=before, objective_fit=after))
    return dict(alpha=alpha, diagnostics=dict(
        fit_start=start, fit_cutoff=cutoff, fit_origin_count=int(rows.sum()),
        last_fit_origin=int(origins[rows][-1]),
        latest_fit_label_available=int(origins[rows][-1] + schema.HORIZONS * HOUR),
        groups=groups, solver="bounded weighted pinball breakpoints"))


def apply_history_correction(frozen_quantiles, correction, alpha):
    """Shift all quantiles equally; zero weights reproduce the base exactly."""
    quantiles = np.asarray(frozen_quantiles, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    if quantiles.ndim != 3 or quantiles.shape[1:] != (schema.HORIZONS, 5):
        raise ValueError("Expected frozen quantiles [N,24,5]")
    if correction.shape != quantiles.shape[:2] or alpha.shape != (4,):
        raise ValueError("Expected correction [N,24] and four weights")
    if not all(np.isfinite(a).all() for a in (quantiles, correction, alpha)):
        raise ValueError("Non-finite prediction input")
    if np.any((alpha < 0) | (alpha > 1)) or np.any(np.abs(correction) > MAX_CORRECTION):
        raise ValueError("Correction or shrinkage weight outside fixed bounds")
    if np.any(np.diff(quantiles, axis=-1) < 0):
        raise ValueError("Frozen quantiles must be ordered")
    per_horizon = alpha[[schema.horizon_group_of(h) for h in range(schema.HORIZONS)]]
    return quantiles + (correction * per_horizon)[..., None]
