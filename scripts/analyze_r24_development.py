"""Paired R24a development evaluation against identity, R21, R22 and R23.

The R24a replay writes one manifest for all regions (flat origin layout); the
earlier arms use one directory per region. Both layouts are validated with the
same checksum and schema checks. Gate references follow the frozen R24
execution plan: R24a must improve QS and MAE against identity and R23 in every
region without regional RMSE regression or coverage90 loss beyond 0.005.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_r22_blocked_cv import (
    REGIONS, METRICS, _load_parent, _check_paired_runs, _load_aligned_data,
    _balanced_blocks, _trace_summary, _read_json,
)
from forecasting.adaptive_agent.replay import aggregate_scores, digest, sha_file

SYSTEMS = ('identity', 'r21', 'r22', 'r23', 'r24')
GATE_REFERENCES = ('identity', 'r23')
R24_POLICY = 'r24_iterative_loop_input'


def load_flat_arm(root, expected_policy=R24_POLICY, split='calibration'):
    """Load one all-regions replay written by the R24a CLI layout."""
    root = Path(root)
    manifest = _read_json(root / 'run_manifest.json')
    report = _read_json(root / 'report.json')
    progress = _read_json(root / 'progress.json')
    complete = _read_json(root / 'predictions_complete.json')
    if (root / 'source_changed.json').exists():
        raise ValueError('Replay source was invalidated')
    configuration = manifest.get('configuration', {})
    if (configuration.get('policy') != expected_policy
            or configuration.get('split') != split
            or tuple(configuration.get('regions', ())) != REGIONS):
        raise ValueError('Flat arm policy, split or region set is wrong')
    expected = manifest.get('expected_origins')
    if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
        raise ValueError('Flat arm expected_origins is invalid')
    if report.get('status') != 'COMPLETE' or progress.get('status') != 'COMPLETE':
        raise ValueError('Flat arm is not COMPLETE')
    counts = (report.get('origins'), report.get('expected_origins'),
              progress.get('completed'), complete.get('completed'),
              complete.get('expected'))
    if any(value != expected for value in counts):
        raise ValueError('Flat arm completion denominator mismatch')
    manifest_id = digest(manifest)
    if report.get('manifest_id') != manifest_id or complete.get('manifest_id') != manifest_id:
        raise ValueError('Flat arm completion identity mismatch')
    rows = {}
    for directory in sorted((root / 'origins').iterdir()):
        if not directory.is_dir() or directory.name.startswith('.'):
            continue
        if directory.is_symlink():
            raise ValueError('Origin directory cannot be a symlink')
        record = _read_json(directory / 'record.json')
        if record.get('manifest_id') != manifest_id:
            raise ValueError('Origin belongs to a different replay manifest')
        required = {'predictions.npz', 'trace.json', 'audit.json'}
        files = record.get('files', {})
        if set(files) not in (required, required | {'materialized_candidates.npz'}):
            raise ValueError('Unexpected committed origin file schema')
        for name, expected_sha in files.items():
            if sha_file(directory / name) != expected_sha:
                raise ValueError('Committed origin checksum mismatch')
        with np.load(directory / 'predictions.npz', allow_pickle=False) as bundle:
            if set(bundle.files) != {'base', 'agent', 'target_time_unix'}:
                raise ValueError('Unexpected prediction array schema')
            arrays = {name: bundle[name].copy() for name in bundle.files}
        origin = record.get('origin_unix')
        region = record.get('region')
        if region not in REGIONS or isinstance(origin, bool) or not isinstance(origin, int):
            raise ValueError('Invalid origin record region or timestamp')
        target = arrays['target_time_unix']
        if (target.shape != (24,) or target.dtype.kind not in 'iu'
                or not np.array_equal(target, origin + np.arange(24) * 3600)):
            raise ValueError('Prediction target axis differs from the 24-hour origin grid')
        for kind in ('base', 'agent'):
            values = arrays[kind]
            if values.shape != (24, 5) or not np.isfinite(values).all():
                raise ValueError('Prediction values must be finite 24x5 quantiles')
        trace = _read_json(directory / 'trace.json')
        if trace.get('selected_forecast_id') != record.get('agent', {}).get('identity'):
            raise ValueError('Trace and committed prediction identity disagree')
        key = (region, origin)
        if key in rows:
            raise ValueError('Duplicate replay origin')
        rows[key] = {'record': record, 'arrays': arrays, 'trace': trace}
    if len(rows) != expected:
        raise ValueError('Flat arm committed origins differ from the denominator')
    return manifest, rows


def score(data, positions):
    systems = {name: aggregate_scores(data['actual'][positions], data[name][positions])
               for name in SYSTEMS}
    return {
        'systems': systems,
        'deltas': {f'r24_minus_{name}': {
            metric: systems['r24'][metric] - systems[name][metric] for metric in METRICS
        } for name in SYSTEMS[:-1]},
    }


def region_equal(regions):
    return {section: {
        name: {metric: float(np.mean([row[section][name][metric] for row in regions.values()]))
               for metric in METRICS}
        for name in next(iter(regions.values()))[section]
    } for section in ('systems', 'deltas')}


def development_gates(regions):
    """Frozen R24a gate: per region, beat identity and R23 on QS and MAE."""
    gates = {}
    for region, evaluation in regions.items():
        comparisons = {}
        for reference in GATE_REFERENCES:
            delta = evaluation['deltas'][f'r24_minus_{reference}']
            comparisons[reference] = {
                'qs_improved': delta['qs'] < 0,
                'mae_improved': delta['mae'] < 0,
                'rmse_not_worse': delta['rmse'] <= 0,
                'coverage_loss_within_0_005': delta['coverage90'] >= -0.005,
            }
        gates[region] = {'comparisons': comparisons,
                         'passed': all(all(checks.values()) for checks in comparisons.values())}
    return {'regions': gates, 'all_regions_passed': all(row['passed'] for row in gates.values())}


def loop_summary(traces):
    from collections import Counter
    rounds, actions, stops = Counter(), Counter(), Counter()
    fallbacks, api_failures, nonidentity = 0, 0, 0
    latencies = []
    for trace in traces:
        loop = trace.get('loop') or {}
        rounds[len(loop.get('rounds') or [])] += 1
        stops[loop.get('stop_reason')] += 1
        for row in loop.get('rounds') or []:
            actions[row['action'].split(':')[0]] += 1
        fallbacks += bool(trace.get('fallback'))
        api_failures += sum(1 for call in trace.get('calls', [])
                            if call.get('record', {}).get('error'))
        nonidentity += trace.get('selected_candidate_id') != 'lightgbm-input:identity'
        latencies.append(trace.get('elapsed_seconds', 0.0))
    latencies = np.asarray(latencies, dtype=float)
    return {
        'loop_rounds_histogram': dict(sorted(rounds.items())),
        'loop_actions': dict(sorted(actions.items())),
        'loop_stop_reasons': dict(sorted(stops.items(),
                                         key=lambda item: (item[0] is None, str(item[0])))),
        'fallback_count': fallbacks,
        'api_failure_count': api_failures,
        'nonidentity_commit_count': nonidentity,
        'origin_latency_seconds': {
            'count': int(latencies.size),
            'p50': float(np.percentile(latencies, 50)) if latencies.size else None,
            'p95': float(np.percentile(latencies, 95)) if latencies.size else None,
        },
    }


def analyze(r24_root, r23_root, r22_root, r21_root, aligned_baseline_root, blocks=6):
    split = 'calibration'
    loaded = {name: _load_parent(path) for name, path in
              (('r21', r21_root), ('r22', r22_root), ('r23', r23_root))}
    manifests21, _, rows21 = loaded['r21']
    manifests23, _, rows23 = loaded['r23']
    origins = _check_paired_runs(manifests21, loaded['r22'][0], rows21, loaded['r22'][2], split)
    _check_paired_runs(manifests21, manifests23, rows21, rows23, split,
                       right_policy='r23_state_aware_input')
    manifest24, rows24 = load_flat_arm(r24_root, split=split)
    if {(region, origin) for region, origin in rows24} != set(rows21):
        raise ValueError('R24a origin schedule does not match the paired arms')
    for region in REGIONS:
        for field in ('operation_admission',):
            left = manifests21[region].get(field, {})
            right = manifest24.get(field, {})
            if left.get('sha256') != right.get('sha256'):
                raise ValueError(f'R24a {region} operation admission mismatch')
        for field in ('lightgbm',):
            if (manifests21[region].get('models', {}).get(field, {}).get(region)
                    != manifest24.get('models', {}).get(field, {}).get(region)):
                raise ValueError(f'R24a {region} frozen model mismatch')
    aligned = _load_aligned_data(aligned_baseline_root, split, manifests21, origins)
    data = {}
    for region in REGIONS:
        keys = [(region, origin) for origin in origins]
        data[region] = {
            'actual': aligned[region]['actual'],
            'identity': np.stack([rows21[key]['arrays']['base'] for key in keys]),
            **{name: np.stack([loaded[name][2][key]['arrays']['agent'] for key in keys])
               for name in ('r21', 'r22', 'r23')},
            'r24': np.stack([rows24[key]['arrays']['agent'] for key in keys]),
            'traces': [rows24[key]['trace'] for key in keys],
            'identity_candidates': [rows24[key]['record']['base']['candidate_id'] for key in keys],
            'final_candidates': [rows24[key]['record']['agent']['candidate_id'] for key in keys],
        }
    pooled = {region: score(values, slice(None)) for region, values in data.items()}
    block_results = []
    for number, positions in enumerate(_balanced_blocks(len(origins), blocks), start=1):
        evaluations = {region: score(values, positions) for region, values in data.items()}
        block_results.append({
            'block': number, 'origins_per_region': len(positions),
            'first_origin': origins[int(positions[0])], 'last_origin': origins[int(positions[-1])],
            'regions': evaluations, 'region_equal': region_equal(evaluations),
        })
    trace_summary = _trace_summary(data)
    trace_summary.pop('_rows', None)
    return {
        'status': 'COMPLETE_DEVELOPMENT_EVALUATION',
        'design': {'split': split, 'period': '2022H2', 'blocks': blocks,
                   'repeated_exposure': True, 'within_block_fitting': False,
                   'formal_test': False, 'traditional_test_comparison': False,
                   'gate_references': list(GATE_REFERENCES)},
        'origins_per_region': len(origins),
        'analysis_sources': {path.name: sha_file(path) for path in (
            Path(__file__), Path(__file__).with_name('analyze_r22_blocked_cv.py'))},
        'inputs': {'roots': {name: str(Path(path).resolve()) for name, path in
                             (('r21', r21_root), ('r22', r22_root), ('r23', r23_root),
                              ('r24', r24_root))},
                   'manifest_ids': {name: {region: digest(value[0][region]) for region in REGIONS}
                                    for name, value in
                                    (('r21', loaded['r21']), ('r22', loaded['r22']),
                                     ('r23', loaded['r23']))}
                   | {'r24': digest(manifest24)}},
        'pooled': {'regions': pooled, 'region_equal': region_equal(pooled)},
        'blocks': block_results, 'development_gates': development_gates(pooled),
        'trace_summary': trace_summary,
        'operations': {region: loop_summary(data[region]['traces']) for region in REGIONS},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('r24-root', 'r23-root', 'r22-root', 'r21-root',
                 'aligned-baseline-root', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--blocks', type=int, default=6)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.r24_root, args.r23_root, args.r22_root, args.r21_root,
                     args.aligned_baseline_root, args.blocks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output),
                      'all_regions_passed': result['development_gates']['all_regions_passed']}))


if __name__ == '__main__':
    main()
