"""Deterministic mature-residual controls; no fitting, routing or file access.

Candidate eligibility uses a complete 24-hour maturity barrier. R28 has the
strict 28-day lower boundary in the frozen protocol. Matched16 reproduces the
existing HistoryStore.record prefilter, including its inclusive 730-day lower
boundary, source-dtype distances, stable ordering and recent-first deduplication.
"""
from datetime import datetime, timedelta, timezone
import math

import numpy as np


HOUR = 3600
DAY = 24 * HOUR
CLIP = 30.
BETAS = (0., .25, .5, .75, 1.)
LEVELS = np.asarray([.05, .10, .50, .90, .95], dtype=np.float64)
MATCH_DIMENSIONS = (1, 2, 3, 6, 7, 8)
AEST = timezone(timedelta(hours=10))
SELECTION_START = int(datetime(2022, 1, 1, tzinfo=AEST).timestamp())
SELECTION_END = int(datetime(2022, 7, 1, tzinfo=AEST).timestamp())


def _origins(value):
    origins = np.asarray(value)
    if origins.ndim != 1 or origins.dtype.kind not in 'iu' or not len(origins):
        raise ValueError('Nonempty integer hourly origins are required')
    if np.any(np.diff(origins) <= 0) or np.any(origins % HOUR):
        raise ValueError('Origins must be unique, chronological and hourly aligned')
    return origins


def _library(origin, actual, baseline, query_rows, fold_id):
    origins = _origins(origin)
    actual, baseline, rows = (np.asarray(v) for v in (actual, baseline, query_rows))
    if actual.shape != (len(origins), 24) or baseline.shape != (len(origins), 24, 5):
        raise ValueError('Expected a shared [origin,24] target and five-quantile grid')
    if not np.isfinite(baseline).all() or np.any(np.diff(baseline, axis=-1) < 0):
        raise ValueError('Baseline quantiles must be finite and ordered')
    if rows.ndim != 1 or rows.dtype.kind not in 'iu' or np.any((rows < 0) | (rows >= len(origins))):
        raise ValueError('Query rows must be valid integer library indices')
    if len(np.unique(rows)) != len(rows):
        raise ValueError('Duplicate query rows are not permitted')
    folds = np.full(len(origins), -1, dtype=np.int16) if fold_id is None else np.asarray(fold_id)
    if folds.shape != origins.shape or folds.dtype.kind not in 'iu':
        raise ValueError('Fold identifiers must be one integer per library origin')
    # Do not inspect labels outside the actual mature candidate sets. This also
    # allows unreleased labels to remain unavailable without affecting forecasts.
    return origins, actual, baseline, rows, folds


def _reduce_candidates(origins, actual, baseline, query_rows, folds, candidates,
                       width, method):
    n = len(query_rows)
    ids = np.full((n, width), -1, dtype=np.int64)
    ages = np.full((n, width), -1., dtype=np.float64)
    candidate_folds = np.full((n, width), -1, dtype=np.int32)
    median = np.zeros((n, 24), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int32)
    for k, (row, selected) in enumerate(zip(query_rows, candidates)):
        selected = np.asarray(selected, dtype=np.int64)
        if (selected.ndim != 1 or len(selected) > width or len(np.unique(selected)) != len(selected)
                or np.any((selected < 0) | (selected >= len(origins)))):
            raise AssertionError('Invalid mature candidate identifiers')
        if np.any(origins[selected] + DAY > origins[row]):
            raise AssertionError('An immature target window entered a control')
        counts[k] = len(selected)
        if not len(selected):
            continue
        # Subtract after conversion: saved FP32 prices are not recalculated or
        # rounded through another FP32 subtraction before the robust median.
        residual = actual[selected].astype(np.float64)-baseline[selected, :, 2].astype(np.float64)
        if not np.isfinite(residual).all():
            raise ValueError('A released candidate has a nonfinite raw residual')
        median[k] = np.median(residual, axis=0)
        ids[k, :len(selected)] = selected
        ages[k, :len(selected)] = (origins[row]-origins[selected])/HOUR
        candidate_folds[k, :len(selected)] = folds[selected]
    return dict(method=method, median=median, candidate_ids=ids,
                candidate_age_hours=ages, candidate_fold_id=candidate_folds,
                counts=counts, fallback=counts == 0, query_rows=query_rows.copy(),
                query_origin=origins[query_rows].copy())


def rolling_residual_28d(origin, actual, baseline, query_rows, *, fold_id=None):
    """Per-lead median over t-28d < j and j+24h <= t, all origin hours."""
    origins, actual, baseline, rows, folds = _library(origin, actual, baseline, query_rows, fold_id)
    query = origins[rows]
    left = np.searchsorted(origins, query-28*DAY, side='right')
    right = np.searchsorted(origins, query-DAY, side='right')
    candidates = [np.arange(lo, hi, dtype=np.int64) for lo, hi in zip(left, right)]
    width = max((len(c) for c in candidates), default=0)
    return _reduce_candidates(origins, actual, baseline, rows, folds, candidates, width, 'R28')


def recent_same_hour_residual(origin, actual, baseline, query_rows, *, fold_id=None):
    """Median of the latest 16 mature origins with the same AEST origin hour.

    No additional lookback cap is introduced. A constant AEST UTC+10 offset
    means equality of UTC residues modulo one day is the same hour criterion.
    """
    origins, actual, baseline, rows, folds = _library(origin, actual, baseline, query_rows, fold_id)
    by_hour = {hour: np.flatnonzero(origins % DAY == hour)
               for hour in np.unique(origins[rows] % DAY)}
    candidates = []
    for row in rows:
        pool = by_hour[origins[row] % DAY]
        hi = np.searchsorted(origins[pool], origins[row]-DAY, side='right')
        candidates.append(pool[max(0, hi-16):hi])
    return _reduce_candidates(origins, actual, baseline, rows, folds, candidates, 16, 'Recent16')


def matched_raw_residual(origin, actual, baseline, z, query_rows, *, fold_id=None):
    """Original HistoryStore's 8-recent plus state-near prefilter, raw median.

    The prefilter below intentionally preserves the old operations and their
    order. No neural representation, historical weighting, gate or query label
    enters it. The selected IDs are returned so an independent implementation
    can compare origin identities rather than assuming identical array offsets.
    """
    origins, actual, baseline, rows, folds = _library(origin, actual, baseline, query_rows, fold_id)
    z = np.asarray(z)
    if z.shape != (len(origins), 24, 16) or z.dtype.kind != 'f':
        raise ValueError('Expected original floating-point [origin,24,16] state array')
    candidates = []
    dims = list(MATCH_DIMENSIONS)
    for row in rows:
        t = origins[row]
        hi = np.searchsorted(origins, t-DAY, side='right')
        lo = np.searchsorted(origins, t-730*DAY)
        pool = np.arange(lo, hi)
        pool = pool[origins[pool] % DAY == t % DAY]
        chosen = []
        if len(pool):
            if not np.isfinite(z[pool][:, :, dims]).all() or not np.isfinite(z[row][:, dims]).all():
                raise ValueError('Nonfinite known state in the matched candidate prefilter')
            dist = ((z[pool][:, :, dims]-z[row][:, dims])**2).mean(axis=(1, 2))
            if not np.isfinite(dist).all():
                raise ValueError('Nonfinite source-dtype state distance')
            chosen = list(pool[-min(len(pool), 8):])
            for candidate in pool[np.argsort(dist, kind='stable')]:
                if int(candidate) not in chosen:
                    chosen.append(int(candidate))
                if len(chosen) >= 16:
                    break
        candidates.append(np.asarray(chosen, dtype=np.int64))
    return _reduce_candidates(origins, actual, baseline, rows, folds, candidates, 16, 'Matched16')


def build_residual_controls(data, query_rows):
    """Build all three fixed candidates once; diagnostics never select beta."""
    args = (data['origin'], data['actual'], data['baseline'])
    kwargs = dict(fold_id=data.get('fold_id'))
    return dict(R28=rolling_residual_28d(*args, query_rows, **kwargs),
                Recent16=recent_same_hour_residual(*args, query_rows, **kwargs),
                Matched16=matched_raw_residual(*args, data['z'], query_rows, **kwargs))


def apply_shared_beta(baseline, median, beta, clip=CLIP):
    """One frozen scalar beta and clip=30; all five quantiles share the shift."""
    b, m = np.asarray(baseline), np.asarray(median, dtype=np.float64)
    if b.ndim < 2 or b.shape[-1] != 5 or m.shape != b.shape[:-1] or not b.size:
        raise ValueError('Baseline and residual median grids differ')
    if (not np.isfinite(b).all() or not np.isfinite(m).all()
            or np.any(np.diff(b, axis=-1) < 0)):
        raise ValueError('Finite medians and ordered baseline quantiles are required')
    if np.asarray(beta).ndim != 0 or not math.isfinite(float(beta)) or float(beta) not in BETAS:
        raise ValueError('A single beta from the frozen grid is required')
    if float(clip) != CLIP:
        raise ValueError('The frozen correction clip is 30')
    if float(beta) == 0:
        return b.copy()  # Exact values and dtype, including signed zeros.
    q = b.astype(np.float64)+float(beta)*np.clip(m, -CLIP, CLIP)[..., None]
    if not np.isfinite(q).all() or np.any(np.diff(q, axis=-1) < 0):
        raise AssertionError('A common residual shift changed quantile order')
    return q


def _losses(actual, prediction):
    y, q = np.asarray(actual, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.shape != q.shape[:-1] or not np.isfinite(y).all():
        raise ValueError('Selection labels must be finite and aligned')
    error = y[..., None]-q
    return dict(mae=float(np.abs(error[..., 2]).mean()),
                qs=float(2*np.maximum(LEVELS*error, (LEVELS-1)*error).mean()))


def select_shared_beta(selection_regions, cfg=None):
    """Select on purged 2022H1 only; accept no later-origin selection records.

    Each region supplies only origin/actual/baseline/median arrays for this
    selection window. R28 provides median; Recent16/Matched16 never call this
    function separately. Return numeric curve/choice, never fitted parameters.
    """
    cfg = {} if cfg is None else cfg
    regions = list(cfg.get('regions', selection_regions.keys()))
    if not regions or set(regions) != set(selection_regions):
        raise ValueError('Exactly one selection dataset per configured region is required')
    grid = tuple(float(v) for v in cfg.get('beta_grid', BETAS))
    if grid != BETAS or float(cfg.get('clip', CLIP)) != CLIP:
        raise ValueError('The beta grid and clip are fixed')
    # Numeric tolerance is supplied by the frozen protocol, never tuned using
    # this curve. It is not a statistical noninferiority margin.
    tolerance = float(cfg.get('selection_numeric_tolerance', 1e-10))
    tie_tolerance = float(cfg.get('objective_tie_tolerance', 1e-12))
    if tolerance != 1e-10 or tie_tolerance != 1e-12:
        raise ValueError('Expected frozen guard/tie tolerances 1e-10 and 1e-12')
    normalized = {}
    for region in regions:
        data = selection_regions[region]
        origins = _origins(data['origin'])
        if np.any(origins < SELECTION_START) or np.any(origins+DAY >= SELECTION_END):
            raise ValueError('Beta selection is restricted to strictly purged 2022H1 AEST origins')
        y, b, m = (np.asarray(data[key]) for key in ('actual', 'baseline', 'median'))
        if y.shape != (len(origins), 24) or b.shape != (len(origins), 24, 5) or m.shape != y.shape:
            raise ValueError('Selection labels, baseline and R28 medians must align')
        apply_shared_beta(b, m, 0.)
        normalized[region] = dict(actual=y, baseline=b, median=m)
    curve = []
    baseline_macro = None
    for beta in BETAS:
        regional = {region: _losses(data['actual'], apply_shared_beta(data['baseline'], data['median'], beta))
                    for region, data in normalized.items()}
        macro = {metric: float(np.mean([regional[r][metric] for r in regions])) for metric in ('mae', 'qs')}
        if beta == 0:
            baseline_macro = macro
        eligible = all(macro[m] <= baseline_macro[m]+tolerance for m in ('mae', 'qs'))
        terms = [macro[m]/baseline_macro[m] if baseline_macro[m] > 0
                 else 1. if macro[m] == 0 else math.inf for m in ('mae', 'qs')]
        objective = sum(terms)
        curve.append(dict(beta=beta, regional=regional, macro=macro, eligible=eligible,
            standardized_relative_loss_sum=objective if math.isfinite(objective) else None))
    eligible = [entry for entry in curve if entry['eligible'] and entry['standardized_relative_loss_sum'] is not None]
    minimum = min(entry['standardized_relative_loss_sum'] for entry in eligible)
    tied = [entry for entry in eligible if entry['standardized_relative_loss_sum'] <= minimum+tie_tolerance]
    best = min(tied, key=lambda entry: entry['beta'])
    positive_eligible = any(entry['eligible'] and entry['beta'] > 0 for entry in curve)
    return dict(beta=best['beta'], curve=curve, baseline_macro=baseline_macro,
        fallback=best['beta'] == 0, positive_beta_eligible=positive_eligible,
        reason='no_positive_beta_satisfies_both_macro_guards' if not positive_eligible
               else 'minimum_standardized_relative_loss_sum_then_smaller_beta',
        regions=regions, selection_start_unix=SELECTION_START, selection_end_exclusive_unix=SELECTION_END,
        selection_purge='origin + 24 hours < 2022-07-01 00:00 AEST',
        objective='macro_MAE(beta)/macro_MAE(B) + macro_QS(beta)/macro_QS(B)',
        region_weighting='equal regions; deterministic controls have no neural seed dimension',
        selection_numeric_tolerance=tolerance, objective_tie_tolerance=tie_tolerance,
        clip=CLIP, diagnostic_beta='Recent16 and Matched16 reuse this R28 beta unchanged')
