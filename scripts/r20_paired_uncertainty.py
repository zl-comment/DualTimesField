"""Preregistered paired daily uncertainty from a completed comparison JSON.

This script never opens forecasting caches, labels, model weights or APIs.
Both arms use the same resampled day indices; regions are averaged before
resampling to preserve within-day dependence between the three regions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

REGIONS = ('NSW1', 'QLD1', 'TAS1')
N_DAYS = 181
DAY_SECONDS = 86400
BLOCK_DAYS = 7
N_BOOTSTRAP = 2000
RNG_SEED = 2026


def preregistration():
    """Fixed parameters, deliberately not configurable through the CLI."""
    return {
        'analysis_id': 'r20-r19-paired-daily-circular-block-v1',
        'input_status': 'COMPLETE_PAIRED_DEVELOPMENT_COMPARISON',
        'split': 'selection', 'experiment_seed': 2026,
        'regions': list(REGIONS), 'origins_per_region': N_DAYS,
        'total_region_origins': len(REGIONS) * N_DAYS,
        'origin_interval_seconds': DAY_SECONDS,
        'aggregation_order': 'First equal-weight the three regions at each shared origin; then average the 181 days.',
        'estimand': 'mean QS delta = R20 minus R19; negative favors R20',
        'relative_improvement_percent': '-100 * mean paired QS delta / mean R19 QS',
        'bootstrap_method': 'circular moving-block bootstrap',
        'block_days': BLOCK_DAYS, 'bootstrap_replicates': N_BOOTSTRAP,
        'rng_seed': RNG_SEED, 'rng_algorithm': 'numpy.random.Generator(PCG64)',
        'blocks_per_replicate': math.ceil(N_DAYS / BLOCK_DAYS),
        'sample_construction': 'Uniform block starts on the 181-day grid; wrap modulo 181; concatenate blocks and truncate to 181 days.',
        'paired_resampling': 'Identical day indices for R19 and R20; no separate region or arm resampling.',
        'interval': 'two-sided 95% percentile', 'percentiles': [2.5, 97.5],
        'percentile_interpolation': 'linear',
        'relative_denominator': 'Recompute the R19 mean within each bootstrap replicate; reject zero denominators rather than dropping replicates.',
        'result_dependent_parameter_selection': False,
        'preregister_mode_reads_comparison': False,
    }


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(field + ' must be a finite number')
    return float(value)


def validate_daily_grid(comparison):
    parameters = preregistration()
    if comparison.get('status') != parameters['input_status']:
        raise ValueError('Only a completed paired comparison can be analyzed')
    if comparison.get('split') != 'selection' or comparison.get('seed') != 2026:
        raise ValueError('Preregistered scope is selection, experiment seed 2026')
    expected = len(REGIONS) * N_DAYS
    rows = comparison.get('per_origin')
    if comparison.get('origins') != expected or not isinstance(rows, list) or len(rows) != expected:
        raise ValueError('Expected exactly 543 region-origins: three regions times 181 days')
    grouped = {region: {} for region in REGIONS}
    for row in rows:
        region, origin = row.get('region'), row.get('origin_unix')
        if region not in grouped:
            raise ValueError('Unexpected or missing region in the preregistered three-region grid')
        if isinstance(origin, bool) or not isinstance(origin, int):
            raise ValueError('Origin must be integer Unix seconds')
        if origin in grouped[region]:
            raise ValueError('Duplicate region/origin; repeats are not independent days')
        reference, treatment = _number(row.get('r19_qs'), 'r19_qs'), _number(row.get('r20_qs'), 'r20_qs')
        if reference < 0 or treatment < 0:
            raise ValueError('Quantile scores cannot be negative')
        declared = _number(row.get('qs_delta_r20_minus_r19'), 'qs_delta_r20_minus_r19')
        if not math.isclose(declared, treatment - reference, rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError('Saved paired delta disagrees with the two scores')
        grouped[region][origin] = (reference, treatment)
    origins = sorted(grouped[REGIONS[0]])
    if len(origins) != N_DAYS or any(sorted(grouped[region]) != origins for region in REGIONS):
        raise ValueError('Each of the three regions must have the same complete 181-origin grid')
    if any(b - a != DAY_SECONDS for a, b in zip(origins, origins[1:])):
        raise ValueError('Origins must be exactly daily with no skipped or irregular intervals')
    # [day, region, arm]. All regions have equal weight, independent of their QS.
    scores = np.array([[grouped[region][origin] for region in REGIONS] for origin in origins], dtype=np.float64)
    daily = scores.mean(axis=1)
    if not np.isfinite(daily).all():
        raise ValueError('Daily aggregation overflowed')
    means = daily.mean(axis=0)
    if means[0] <= 0:
        raise ValueError('Relative improvement is undefined for zero mean R19 QS')
    declared_regions = comparison.get('regions')
    if declared_regions is not None:
        if set(declared_regions) != set(REGIONS) or any(declared_regions[r].get('count') != N_DAYS for r in REGIONS):
            raise ValueError('Region report denominators do not match the complete daily grid')
    declared_mean = comparison.get('region_equal_mean')
    if declared_mean is not None:
        for i, arm in enumerate(('R19', 'R20')):
            reported = _number(declared_mean[arm]['qs'], arm + ' reported mean QS')
            if not math.isclose(reported, float(means[i]), rel_tol=1e-12, abs_tol=1e-10):
                raise ValueError('Per-origin daily aggregation disagrees with comparison report')
    return np.asarray(origins, dtype=np.int64), daily


def analyze(comparison):
    origins, daily = validate_daily_grid(comparison)
    rng = np.random.Generator(np.random.PCG64(RNG_SEED))
    starts = rng.integers(0, N_DAYS, size=(N_BOOTSTRAP, math.ceil(N_DAYS / BLOCK_DAYS)))
    indices = ((starts[..., None] + np.arange(BLOCK_DAYS)) % N_DAYS).reshape(N_BOOTSTRAP, -1)[:, :N_DAYS]
    # Paired differences retain between-arm covariance for every resampled day.
    daily_delta = daily[:, 1] - daily[:, 0]
    reference_samples = daily[indices, 0].mean(axis=1)
    delta_samples = daily_delta[indices].mean(axis=1)
    if not np.isfinite(reference_samples).all() or np.any(reference_samples <= 0):
        raise ValueError('At least one replicate has an undefined nonpositive R19 denominator; no replicates were discarded')
    improvement_samples = -100 * delta_samples / reference_samples
    if not np.isfinite(delta_samples).all() or not np.isfinite(improvement_samples).all():
        raise ValueError('Nonfinite bootstrap results; no replicates were discarded')
    reference_mean, treatment_mean = daily.mean(axis=0)
    delta = float(daily_delta.mean())
    interval = lambda values: np.percentile(values, [2.5, 97.5], method='linear').tolist()
    return {
        'status': 'COMPLETE_DESCRIPTIVE_PAIRED_UNCERTAINTY',
        'preregistered_parameters': preregistration(),
        'grid': {'regions': list(REGIONS), 'days': N_DAYS, 'region_origins': N_DAYS * len(REGIONS),
                 'first_origin_unix': int(origins[0]), 'last_origin_unix': int(origins[-1]),
                 'origin_interval_seconds': DAY_SECONDS},
        'mean_qs': {'R19': float(reference_mean), 'R20': float(treatment_mean)},
        'mean_qs_delta_r20_minus_r19': delta,
        'mean_qs_delta_percentile_95_interval': interval(delta_samples),
        'relative_improvement_percent': -100 * delta / float(reference_mean),
        'relative_improvement_percentile_95_interval': interval(improvement_samples),
        'bootstrap_replicates_completed': N_BOOTSTRAP,
        'bootstrap_indices_sha256': hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
        'numpy_version': np.__version__,
        'input_manifest_ids': comparison.get('manifest_ids'),
        'labels_read': False,
        'limitations': [
            'This describes the completed development comparison, not an independent blind-test result or a causal effect of the LLM.',
            'Seven-day blocks preserve local serial dependence and shared-day dependence between regions; longer dependence and nonstationarity may remain.',
            'Circular wrapping joins the end of the development interval to its beginning as a resampling approximation.',
            'A percentile interval excluding zero does not establish an independently validated gain after development or model/skill selection.',
            'No block-length search, one-sided test, or outcome-dependent stopping is performed.',
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--comparison', type=Path, help='Completed 543-origin paired comparison JSON')
    mode.add_argument('--preregister', action='store_true', help='Write fixed parameters without reading comparison results')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.preregister:
        result = {'status': 'PREREGISTERED_NO_RESULTS_READ', 'parameters': preregistration(),
                  'script_sha256': script_sha, 'numpy_version': np.__version__, 'labels_read': False}
    else:
        raw = args.comparison.read_bytes()
        result = analyze(json.loads(raw))
        result.update(input_comparison=str(args.comparison.resolve()),
                      input_comparison_sha256=hashlib.sha256(raw).hexdigest(), script_sha256=script_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
