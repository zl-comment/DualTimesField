"""Paired R24b development evaluation: candidate ledger versus R24a and R23.

Reuses the flat-layout loader and loop summary from analyze_r24_development.
Gate references stay identity and R23 per the frozen R24 execution plan.
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
    _balanced_blocks, _trace_summary,
)
from scripts.analyze_r24_development import loop_summary, load_flat_arm
from forecasting.adaptive_agent.replay import aggregate_scores, digest, sha_file

SYSTEMS = ('identity', 'r21', 'r22', 'r23', 'r24a', 'r24b')
GATE_REFERENCES = ('identity', 'r23')


def score(data, positions):
    systems = {name: aggregate_scores(data['actual'][positions], data[name][positions])
               for name in SYSTEMS}
    return {
        'systems': systems,
        'deltas': {f'r24b_minus_{name}': {
            metric: systems['r24b'][metric] - systems[name][metric] for metric in METRICS
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
        for reference in GATE_REFERENCES:
            delta = evaluation['deltas'][f'r24b_minus_{reference}']
            comparisons[reference] = {
                'qs_improved': delta['qs'] < 0,
                'mae_improved': delta['mae'] < 0,
                'rmse_not_worse': delta['rmse'] <= 0,
                'coverage_loss_within_0_005': delta['coverage90'] >= -0.005,
            }
        gates[region] = {'comparisons': comparisons,
                         'passed': all(all(checks.values()) for checks in comparisons.values())}
    return {'regions': gates, 'all_regions_passed': all(row['passed'] for row in gates.values())}


def analyze(r24b_root, r24a_root, r23_root, r22_root, r21_root, aligned_baseline_root,
            blocks=6):
    split = 'calibration'
    loaded = {name: _load_parent(path) for name, path in
              (('r21', r21_root), ('r22', r22_root), ('r23', r23_root))}
    manifests21, _, rows21 = loaded['r21']
    manifests23, _, rows23 = loaded['r23']
    origins = _check_paired_runs(manifests21, loaded['r22'][0], rows21, loaded['r22'][2], split)
    _check_paired_runs(manifests21, manifests23, rows21, rows23, split,
                       right_policy='r23_state_aware_input')
    manifest24a, rows24a = load_flat_arm(r24a_root, expected_policy='r24_iterative_loop_input',
                                         split=split)
    manifest24b, rows24b = load_flat_arm(r24b_root, expected_policy='r24b_candidate_ledger_input',
                                         split=split)
    for rows in (rows24a, rows24b):
        if set(rows) != set(rows21):
            raise ValueError('R24 flat arm origin schedule does not match the paired arms')
    for region in REGIONS:
        for field in ('lightgbm',):
            for manifest in (manifest24a, manifest24b):
                if (manifests21[region].get('models', {}).get(field, {}).get(region)
                        != manifest.get('models', {}).get(field, {}).get(region)):
                    raise ValueError(f'Flat arm {region} frozen model mismatch')
    aligned = _load_aligned_data(aligned_baseline_root, split, manifests21, origins)
    data = {}
    for region in REGIONS:
        keys = [(region, origin) for origin in origins]
        data[region] = {
            'actual': aligned[region]['actual'],
            'identity': np.stack([rows21[key]['arrays']['base'] for key in keys]),
            **{name: np.stack([loaded[name][2][key]['arrays']['agent'] for key in keys])
               for name in ('r21', 'r22', 'r23')},
            'r24a': np.stack([rows24a[key]['arrays']['agent'] for key in keys]),
            'r24b': np.stack([rows24b[key]['arrays']['agent'] for key in keys]),
            'traces': [rows24b[key]['trace'] for key in keys],
            'identity_candidates': [rows24b[key]['record']['base']['candidate_id'] for key in keys],
            'final_candidates': [rows24b[key]['record']['agent']['candidate_id'] for key in keys],
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

    def ledger_summary(traces):
        rows = [t.get('candidate_ledger') or {} for t in traces]
        return {
            'origins_with_ledger': sum(bool(r) for r in rows),
            'final_matured_rows': [r.get('matured_candidate_rows') for r in rows][-1],
            'final_candidate_ids': sorted(
                (rows[-1].get('candidates') or {}).keys()) if rows[-1] else [],
            'nonidentity_commit_count': sum(
                t.get('selected_candidate_id') != 'lightgbm-input:identity' for t in traces),
        }

    return {
        'status': 'COMPLETE_DEVELOPMENT_EVALUATION',
        'design': {'split': split, 'period': '2022H2', 'blocks': blocks,
                   'repeated_exposure': True, 'within_block_fitting': False,
                   'formal_test': False, 'gate_references': list(GATE_REFERENCES)},
        'origins_per_region': len(origins),
        'analysis_sources': {path.name: sha_file(path) for path in (
            Path(__file__), Path(__file__).with_name('analyze_r22_blocked_cv.py'),
            Path(__file__).with_name('analyze_r24_development.py'))},
        'inputs': {'roots': {name: str(Path(path).resolve()) for name, path in
                             (('r21', r21_root), ('r22', r22_root), ('r23', r23_root),
                              ('r24a', r24a_root), ('r24b', r24b_root))},
                   'manifest_ids': {'r24a': digest(manifest24a), 'r24b': digest(manifest24b)}},
        'pooled': {'regions': pooled, 'region_equal': region_equal(pooled)},
        'blocks': block_results, 'development_gates': development_gates(pooled),
        'trace_summary': trace_summary,
        'operations': {region: {**loop_summary(data[region]['traces']),
                                'candidate_ledger': ledger_summary(data[region]['traces'])}
                       for region in REGIONS},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('r24b-root', 'r24a-root', 'r23-root', 'r22-root', 'r21-root',
                 'aligned-baseline-root', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--blocks', type=int, default=6)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.r24b_root, args.r24a_root, args.r23_root, args.r22_root,
                     args.r21_root, args.aligned_baseline_root, args.blocks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output),
                      'all_regions_passed': result['development_gates']['all_regions_passed']}))


if __name__ == '__main__':
    main()
