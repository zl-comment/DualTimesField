"""Paired R23 development evaluation against identity, R21 and R22."""
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
    _balanced_blocks, _trace_summary,
)
from forecasting.adaptive_agent.replay import aggregate_scores, digest, sha_file


SYSTEMS = ('identity', 'r21', 'r22', 'r23')


def score(data, positions):
    systems = {name: aggregate_scores(data['actual'][positions], data[name][positions])
               for name in SYSTEMS}
    return {
        'systems': systems,
        'deltas': {f'r23_minus_{name}': {
            metric: systems['r23'][metric] - systems[name][metric] for metric in METRICS
        } for name in SYSTEMS[:-1]},
    }


def region_equal(regions):
    return {section: {
        name: {metric: float(np.mean([row[section][name][metric] for row in regions.values()]))
               for metric in METRICS}
        for name in next(iter(regions.values()))[section]
    } for section in ('systems', 'deltas')}


def development_gates(regions):
    gates = {}
    for region, evaluation in regions.items():
        comparisons = {}
        for reference in ('identity', 'r21'):
            delta = evaluation['deltas'][f'r23_minus_{reference}']
            comparisons[reference] = {
                'qs_improved': delta['qs'] < 0,
                'mae_improved': delta['mae'] < 0,
                'rmse_not_worse': delta['rmse'] <= 0,
                'coverage_loss_within_0_005': delta['coverage90'] >= -0.005,
            }
        gates[region] = {'comparisons': comparisons,
                        'passed': all(all(checks.values()) for checks in comparisons.values())}
    return {'regions': gates, 'all_regions_passed': all(row['passed'] for row in gates.values())}


def analyze(r23_root, r22_root, r21_root, aligned_baseline_root, blocks=6):
    split = 'calibration'
    loaded = {name: _load_parent(path) for name, path in
              (('r21', r21_root), ('r22', r22_root), ('r23', r23_root))}
    manifests21, _, rows21 = loaded['r21']
    manifests22, _, rows22 = loaded['r22']
    manifests23, reports23, rows23 = loaded['r23']
    origins = _check_paired_runs(manifests21, manifests22, rows21, rows22, split)
    _check_paired_runs(manifests21, manifests23, rows21, rows23, split,
                       right_policy='r23_state_aware_input')
    for region in REGIONS:
        for name in ('r22', 'r23'):
            manifest = loaded[name][0][region]
            for field in ('operation_admission',):
                left = manifests21[region].get(field, {})
                right = manifest.get(field, {})
                if left.get('sha256') != right.get('sha256'):
                    raise ValueError(f'{name} {region} operation admission mismatch')
            for field in ('lightgbm',):
                if manifests21[region].get('models', {}).get(field) != manifest.get('models', {}).get(field):
                    raise ValueError(f'{name} {region} frozen model mismatch')
    aligned = _load_aligned_data(aligned_baseline_root, split, manifests21, origins)
    data = {}
    for region in REGIONS:
        keys = [(region, origin) for origin in origins]
        data[region] = {
            'actual': aligned[region]['actual'],
            'identity': np.stack([rows21[key]['arrays']['base'] for key in keys]),
            **{name: np.stack([loaded[name][2][key]['arrays']['agent'] for key in keys])
               for name in ('r21', 'r22', 'r23')},
            'traces': [rows23[key]['trace'] for key in keys],
            'identity_candidates': [rows23[key]['record']['base']['candidate_id'] for key in keys],
            'final_candidates': [rows23[key]['record']['agent']['candidate_id'] for key in keys],
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
    trace_summary.pop('_rows')
    return {
        'status': 'COMPLETE_DEVELOPMENT_EVALUATION',
        'design': {'split': split, 'period': '2022H2', 'blocks': blocks,
                   'repeated_exposure': True, 'within_block_fitting': False,
                   'formal_test': False, 'traditional_test_comparison': False},
        'origins_per_region': len(origins),
        'analysis_sources': {path.name: sha_file(path) for path in
                             (Path(__file__), Path(__file__).with_name('analyze_r22_blocked_cv.py'))},
        'inputs': {'roots': {name: str(Path(path).resolve()) for name, path in
                             (('r21', r21_root), ('r22', r22_root), ('r23', r23_root))},
                   'manifest_ids': {name: {region: digest(value[0][region]) for region in REGIONS}
                                    for name, value in loaded.items()}},
        'pooled': {'regions': pooled, 'region_equal': region_equal(pooled)},
        'blocks': block_results, 'development_gates': development_gates(pooled),
        'trace_summary': trace_summary,
        'operations': {region: {key: reports23[region].get(key) for key in
                                ('fallback_count', 'api_failure_count', 'selected_candidates',
                                 'risk_commit_reasons', 'risk_review_count', 'api_calls',
                                 'origin_latency_seconds')} for region in REGIONS},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('r23-root', 'r22-root', 'r21-root', 'aligned-baseline-root', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--blocks', type=int, default=6)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.r23_root, args.r22_root, args.r21_root,
                     args.aligned_baseline_root, args.blocks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
