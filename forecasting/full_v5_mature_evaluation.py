"""Paired scoring and frozen screening of deterministic mature residual controls.

All labels enter only scoring. Day blocks sample AEST origin dates jointly
across regions. Loss sums and cell counts preserve the original cell-weighted
estimand when purging makes boundary days incomplete. Confidence intervals are
descriptive, do not treat neural seeds as independent samples, and never enter
the prespecified L0 screening decision.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from .full_v5_correction_audit.metrics import cell_losses, score, score_and_gain


DAY = 86400
AEST = 10*3600
BLOCKS = (7, 14, 28)
REPLICATES = 2000
CONFIDENCE = .95
BOOTSTRAP_SEED = 20261027
NUMERIC_TOLERANCE = 1e-10
REGIONAL_RELATIVE_LIMIT = .01
COVERAGE_LIMIT = .01
REQUIRED_INTEGRITY_CHECKS = (
    'source_hashes_verified', 'baseline_archive_reproduced', 'samples_aligned',
    'candidates_independently_verified', 'future_labels_invariant',
    'maturity_boundary_verified', 'purge_verified', 'quantiles_ordered',
    'beta_zero_exact', 'deterministic_simple_arms', 'old_files_unchanged',
)


def _validate(actual, quantiles):
    y, q = np.asarray(actual, dtype=np.float64), np.asarray(quantiles, dtype=np.float64)
    if not y.size or q.shape != (*y.shape, 5):
        raise ValueError('Nonempty actuals and a matching five-quantile grid are required')
    if not np.isfinite(y).all() or not np.isfinite(q).all() or np.any(np.diff(q, axis=-1) < 0):
        raise ValueError('Outcomes and quantiles must be finite; quantiles must be ordered')
    return y, q


def score_quantiles(actual, quantiles):
    """The archived project's raw-price MAE and 2*mean-five-pinball QS."""
    y, q = _validate(actual, quantiles)
    return score(y, q)


def paired_scores(actual, baseline, prediction, fallback=None):
    y, b = _validate(actual, baseline)
    _, q = _validate(y, prediction)
    result = dict(cells=int(y.size), empty=False, **score_and_gain(y, b, q))
    if fallback is not None:
        mask = np.asarray(fallback)
        if mask.dtype != bool or mask.shape not in (y.shape, y.shape[:1]):
            raise ValueError('Fallback flags must be boolean per-origin or per-cell')
        result['fallback_rate'] = float(mask.mean())
    return result


def paired_cell_losses(actual, baseline, prediction):
    """Return cell losses without flattening the origin and lead dimensions."""
    y, b = _validate(actual, baseline)
    _, q = _validate(y, prediction)
    mb, qb, pb = cell_losses(y, b)
    ma, qa, pa = cell_losses(y, q)
    result = dict(before_mae=mb, after_mae=ma, before_qs=qb, after_qs=qa)
    for name, delta in (('mae', ma-mb), ('qs', qa-qb)):
        result.update({f'{name}_delta': delta, f'{name}_benefit': np.maximum(-delta, 0),
                       f'{name}_harm': np.maximum(delta, 0)})
    for column, name in enumerate(('q05', 'q10', 'q50', 'q90', 'q95')):
        result[f'before_pinball_{name}'] = pb[..., column]
        result[f'after_pinball_{name}'] = pa[..., column]
    for label, quantiles in (('before', b), ('after', q)):
        for level, lower, upper in ((80, 1, 3), (90, 0, 4)):
            result[f'{label}_coverage{level}'] = ((y >= quantiles[..., lower]) &
                                                (y <= quantiles[..., upper])).astype(np.float64)
            result[f'{label}_width{level}'] = quantiles[..., upper]-quantiles[..., lower]
    result['median_delta'] = q[..., 2]-b[..., 2]
    return result


def daily_paired_losses(origins, actual, baseline, prediction):
    """Aggregate all leads within each AEST origin date using sums and counts."""
    origins = np.asarray(origins, dtype=np.int64)
    y = np.asarray(actual)
    if origins.ndim != 1 or y.ndim != 2 or y.shape != (len(origins), 24):
        raise ValueError('Expected unique origins and the matching [origin,24] evaluation grid')
    if np.any(np.diff(origins) <= 0):
        raise ValueError('Origins must be unique and chronological')
    cells = paired_cell_losses(y, baseline, prediction)
    local_date = (origins+AEST)//DAY
    dates, inverse, counts = np.unique(local_date, return_inverse=True, return_counts=True)
    result = dict(date_unix=dates*DAY-AEST, cells=counts.astype(np.int64)*24)
    for name, values in cells.items():
        origin_sum = np.asarray(values, dtype=np.float64).sum(axis=1)
        result[f'{name}_sum'] = np.bincount(inverse, weights=origin_sum, minlength=len(dates))
    return result


def _fixed_inference_configuration(cfg):
    cfg = {} if cfg is None else cfg
    expected = dict(bootstrap_replicates=REPLICATES, bootstrap_confidence=CONFIDENCE,
                    bootstrap_blocks_days=list(BLOCKS), bootstrap_seed=BOOTSTRAP_SEED)
    for key, value in expected.items():
        if cfg.get(key, value) != value:
            raise ValueError(f'{key} is fixed by the committed protocol')
    return expected


def paired_day_block_inference(regional_daily, cfg=None):
    """Joint non-circular moving-day blocks, 2000 draws and percentile 95% CIs.

    The same day-index matrix is used for all regions and both losses. Complete
    calendar blocks cannot jump across missing dates. Returned seed is a
    resampling RNG seed, never a neural-model replication dimension.
    """
    settings = _fixed_inference_configuration(cfg)
    if not isinstance(regional_daily, dict) or not regional_daily:
        raise ValueError('One deterministic paired day table per region is required')
    regions = list(cfg['regions']) if cfg is not None and 'regions' in cfg else sorted(regional_daily)
    if set(regional_daily) != set(regions) or len(set(regions)) != len(regions):
        raise ValueError('Region matrix differs from the fixed protocol')
    dates = np.asarray(regional_daily[regions[0]]['date_unix'], dtype=np.int64)
    if dates.ndim != 1 or not len(dates) or np.any(np.diff(dates) <= 0):
        raise ValueError('Daily dates must be nonempty, unique and chronological')
    count_values, before_values, after_values = [], [], []
    for region in regions:
        daily = regional_daily[region]
        if not np.array_equal(np.asarray(daily['date_unix']), dates):
            raise ValueError('Regional day grids differ; synchronized bootstrap is impossible')
        counts = np.asarray(daily['cells'], dtype=np.float64)
        before = np.stack([np.asarray(daily[f'before_{m}_sum'], dtype=np.float64) for m in ('mae', 'qs')], axis=-1)
        after = np.stack([np.asarray(daily[f'after_{m}_sum'], dtype=np.float64) for m in ('mae', 'qs')], axis=-1)
        if counts.shape != dates.shape or before.shape != (len(dates), 2) or after.shape != before.shape:
            raise ValueError('Day losses and counts must match the day grid')
        if not np.isfinite(counts).all() or np.any(counts <= 0) or np.any(counts != np.floor(counts)):
            raise ValueError('Every sampled date must contain a positive integer number of forecast cells')
        if not np.isfinite(before).all() or not np.isfinite(after).all() or np.any(before < 0) or np.any(after < 0):
            raise ValueError('Day loss sums must be finite and nonnegative')
        count_values.append(counts)
        before_values.append(before)
        after_values.append(after)
    counts = np.stack(count_values, axis=1)  # day, region
    before, after = np.stack(before_values, axis=1), np.stack(after_values, axis=1)
    delta = after-before
    point_region = delta.sum(axis=0)/counts.sum(axis=0)[:, None]
    point_macro = point_region.mean(axis=0)
    output = dict(method='synchronized_non_circular_moving_day_block_percentile',
        date_basis='AEST forecast-origin date; retain every lead within its origin date',
        estimand='within-region paired loss sums divided by cell counts, then equal-region macro',
        regions=regions, days=len(dates), **settings, blocks=[],
        point_region={region:{metric:float(point_region[r,m]) for m,metric in enumerate(('mae','qs'))}
                      for r,region in enumerate(regions)},
        point_macro={metric:float(point_macro[m]) for m,metric in enumerate(('mae','qs'))},
        decision_uses_confidence_interval=False,
        interval_scope='pointwise descriptive 95% intervals; no simultaneous or multiplicity-adjusted inference',
        limitations=['Overlapping origins remain dependent within date blocks; the day block length is prespecified.',
                     'Intervals are descriptive under temporal dependence and drift, not guarantees under arbitrary nonstationarity.',
                     'Regions share resampled dates; neural seeds are not resampled or treated as independent market data.'])
    for block in BLOCKS:
        starts = np.arange(max(len(dates)-block+1, 0))
        if len(starts):
            starts = starts[np.all(dates[starts[:, None]+np.arange(block)] == dates[starts, None]+DAY*np.arange(block), axis=1)]
        if not len(starts):
            output['blocks'].append(dict(block_days=block, primary=block==7, available=False,
                                         reason='No complete consecutive calendar block of the fixed length'))
            continue
        rng = np.random.default_rng(BOOTSTRAP_SEED+block)
        sampled_starts = rng.choice(starts, size=(REPLICATES, math.ceil(len(dates)/block)))
        indices = (sampled_starts[..., None]+np.arange(block)).reshape(REPLICATES, -1)[:, :len(dates)]
        # One shared matrix supplies each region: simultaneous shocks are not
        # erased by independently drawing dates for different markets.
        sampled_region = delta[indices].sum(axis=1)/counts[indices].sum(axis=1)[..., None]
        sampled_macro = sampled_region.mean(axis=1)
        limits_region = np.quantile(sampled_region, [.025, .975], axis=0, method='linear')
        limits_macro = np.quantile(sampled_macro, [.025, .975], axis=0, method='linear')
        output['blocks'].append(dict(block_days=block, primary=block==7, available=True,
            valid_block_starts=int(len(starts)), replicates=REPLICATES, rng_seed=BOOTSTRAP_SEED+block,
            common_day_indices_sha256=hashlib.sha256(indices.astype('<i8', copy=False).tobytes()).hexdigest(),
            regional={region:{metric:dict(estimate=float(point_region[r,m]),
                lower=float(limits_region[0,r,m]), upper=float(limits_region[1,r,m]))
                for m,metric in enumerate(('mae','qs'))} for r,region in enumerate(regions)},
            macro={metric:dict(estimate=float(point_macro[m]), lower=float(limits_macro[0,m]),
                              upper=float(limits_macro[1,m])) for m,metric in enumerate(('mae','qs'))}))
    return output


def apply_decision_rules(region_rows, beta, integrity, cfg, stage='L0'):
    """Apply the frozen R28-only loss/coverage screen; diagnostic arms cannot vote."""
    for key, fixed in (('numeric_tolerance', NUMERIC_TOLERANCE),
                       ('regional_relative_limit', REGIONAL_RELATIVE_LIMIT),
                       ('coverage_tolerance', COVERAGE_LIMIT)):
        if cfg.get(key, fixed) != fixed:
            raise ValueError(f'{key} is fixed by the committed protocol')
    if not math.isfinite(float(beta)) or float(beta) not in (0., .25, .5, .75, 1.):
        raise ValueError('Only the frozen shared beta grid is allowed')
    expected = list(cfg['regions'])
    if len(set(expected)) != len(expected) or not expected:
        raise ValueError('Expected a nonempty unique fixed region list')
    selected = [row for row in region_rows if row.get('arm') == 'R28']
    represented = [row.get('region') for row in selected]
    complete = len(selected)==len(expected) and set(represented)==set(expected)
    criteria = [dict(name='complete_single_R28_per_region', passed=complete,
                     expected_regions=expected, observed_regions=represented),
                dict(name='positive_shared_beta', beta=float(beta), passed=float(beta)>0)]
    required_integrity = cfg.get('required_integrity_checks', list(REQUIRED_INTEGRITY_CHECKS))
    if tuple(required_integrity) != REQUIRED_INTEGRITY_CHECKS:
        raise ValueError('All eleven integrity checks are fixed by the committed protocol')
    if not isinstance(integrity, dict):
        raise ValueError('Integrity evidence must be an explicit boolean mapping')
    integrity_complete = set(required_integrity) <= set(integrity)
    criteria.append(dict(name='complete_integrity_manifest', passed=integrity_complete,
                         required=list(required_integrity)))
    for key in required_integrity:
        criteria.append(dict(name=f'integrity/{key}', passed=integrity.get(key) is True))
    macro_before, macro_after = {}, {}
    for region in expected:
        rows = [row for row in selected if row.get('region') == region]
        row = rows[0] if len(rows)==1 else {}
        valid = (complete and not row.get('empty', True) and int(row.get('cells', 0))>0)
        for loss in ('mae', 'qs'):
            before, after = row.get(f'before_{loss}'), row.get(f'after_{loss}')
            finite = valid and before is not None and after is not None and np.isfinite(before) and np.isfinite(after) and before>=0 and after>=0
            criteria.append(dict(name='regional_loss_limit', region=region, metric=loss,
                before=before, after=after, relative_limit=REGIONAL_RELATIVE_LIMIT,
                passed=bool(finite and after <= before*(1+REGIONAL_RELATIVE_LIMIT)+NUMERIC_TOLERANCE)))
        for level in (80, 90):
            before, after = row.get(f'before_coverage{level}'), row.get(f'after_coverage{level}')
            finite = valid and before is not None and after is not None and np.isfinite(before) and np.isfinite(after) and 0<=before<=1 and 0<=after<=1
            worsening = float(abs(after-level/100)-abs(before-level/100)) if finite else None
            criteria.append(dict(name='regional_coverage_deviation', region=region, level=level,
                before=before, after=after, absolute_deviation_worsening=worsening,
                limit=COVERAGE_LIMIT, passed=worsening is not None and worsening<=COVERAGE_LIMIT+NUMERIC_TOLERANCE))
    for loss in ('mae', 'qs'):
        valid = complete and all(not row.get('empty', True) and int(row.get('cells',0))>0
            and all(np.isfinite(row.get(f'{arm}_{loss}', np.nan)) and row[f'{arm}_{loss}']>=0 for arm in ('before','after')) for row in selected)
        before = float(np.mean([row[f'before_{loss}'] for row in selected])) if valid else None
        after = float(np.mean([row[f'after_{loss}'] for row in selected])) if valid else None
        macro_before[loss], macro_after[loss] = before, after
        criteria.append(dict(name='macro_loss_nonincrease', metric=loss, before=before, after=after,
            passed=before is not None and after is not None and after<=before+NUMERIC_TOLERANCE))
    strict = any(macro_before[m] is not None and macro_after[m]<macro_before[m]-NUMERIC_TOLERANCE for m in ('mae','qs'))
    criteria.append(dict(name='at_least_one_macro_loss_strictly_better', passed=strict))
    passed = all(row['passed'] for row in criteria)
    return dict(stage=stage, primary_arm='R28', beta=float(beta), passed=passed,
        decision=('proceed_to_frozen_L1' if stage=='L0' else 'retain_R28_as_required_research_baseline') if passed else 'stop_current_v5_residual_route',
        macro_before=macro_before, macro_after=macro_after, criteria=criteria,
        failed_criteria=[row for row in criteria if not row['passed']],
        numerical_tolerance=NUMERIC_TOLERANCE, intervals_used_for_decision=False,
        diagnostic_arms_used_for_decision=False, statistical_noninferiority_claim=False)
