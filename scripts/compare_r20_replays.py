"""Compare completed paired replay arms, including immutable prediction checks."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forecasting.adaptive_agent.replay import aggregate_scores, digest, sha_file
from forecasting.adaptive_agent.contracts import ForecastArtifact


def read_arm(path):
    path = Path(path)
    manifest = json.loads((path / 'run_manifest.json').read_text())
    report = json.loads((path / 'report.json').read_text())
    complete = json.loads((path / 'predictions_complete.json').read_text())
    if report['status'] != 'COMPLETE' or report['origins'] != report['expected_origins']:
        raise ValueError('Both arms must be complete without omitted origins')
    progress = json.loads((path / 'progress.json').read_text())
    if progress.get('status') != 'COMPLETE' or (path / 'source_changed.json').exists():
        raise ValueError('Replay is incomplete or its frozen source was invalidated')
    expected = manifest['expected_origins']
    if (report['origins'] != expected or complete.get('completed') != expected
            or complete.get('expected') != expected or progress.get('completed') != expected):
        raise ValueError('Run/report/completion denominator mismatch')
    if report['manifest_id'] != digest(manifest) or complete['manifest_id'] != digest(manifest):
        raise ValueError('Completion identity mismatch')
    rows = {}
    for directory in sorted((path / 'origins').iterdir()):
        if not directory.is_dir() or directory.name.startswith('.'):
            continue
        record = json.loads((directory / 'record.json').read_text())
        if record['manifest_id'] != digest(manifest):
            raise ValueError('Origin belongs to a different run')
        required = {'predictions.npz', 'trace.json', 'audit.json'}
        if set(record['files']) not in (required, required | {'materialized_candidates.npz'}):
            raise ValueError('Unexpected committed file manifest')
        if directory.is_symlink():
            raise ValueError('Origin directory cannot be a symlink')
        for name, expected in record['files'].items():
            if sha_file(directory / name) != expected:
                raise ValueError('Committed origin checksum mismatch')
        with np.load(directory / 'predictions.npz', allow_pickle=False) as bundle:
            arrays = {name: bundle[name].copy() for name in bundle.files}
        if set(arrays) != {'base', 'agent', 'target_time_unix'}:
            raise ValueError('Unexpected prediction array schema')
        if not np.array_equal(arrays['target_time_unix'], record['origin_unix'] + np.arange(24) * 3600):
            raise ValueError('Prediction target axis differs from origin')
        for kind in ('base', 'agent'):
            entry = record[kind]
            artifact = ForecastArtifact(entry['candidate_id'], record['context_id'], arrays[kind],
                arrays['target_time_unix'], entry['model_id'], entry['input_hash'], entry['provenance'])
            if artifact.identity() != entry['identity'] or np.any(np.diff(arrays[kind], axis=-1) < 0):
                raise ValueError('Frozen prediction identity or quantile ordering invalid')
        key = (record['region'], record['origin_unix'])
        if key in rows:
            raise ValueError('Duplicate origin')
        trace = json.loads((directory / 'trace.json').read_text())
        if trace.get('selected_forecast_id') != record['agent']['identity'] or trace.get('context_id') != record['context_id']:
            raise ValueError('Trace and prediction identity disagree')
        rows[key] = {'record': record, 'arrays': arrays, 'trace': trace, 'directory': directory}
    if len(rows) != report['origins']:
        raise ValueError('Saved origins do not match report denominator')
    schedule = sorted((record['origin_unix'], record['region'], record['index']) for record in
                      (row['record'] for row in rows.values()))
    if digest(schedule) != manifest['schedule_sha256']:
        raise ValueError('Saved schedule differs from frozen manifest')
    if {key[0] for key in rows} != set(manifest['configuration']['regions']):
        raise ValueError('Region coverage differs from requested denominator')
    return manifest, report, rows


def compare(reference, treatment, baseline_root):
    lm, lr, left = read_arm(reference)
    rm, rr, right = read_arm(treatment)
    if set(left) != set(right):
        raise ValueError('Origin schedules must match exactly')
    left_config = {k: v for k, v in lm['configuration'].items() if k != 'policy'}
    right_config = {k: v for k, v in rm['configuration'].items() if k != 'policy'}
    if left_config != right_config:
        raise ValueError('Comparison configuration differs beyond the intended policy change')
    # Source differs for a frozen earlier R19 smoke; models and inputs cannot.
    for key in ('inputs', 'models', 'budgets', 'evidence', 'packages', 'schedule_sha256'):
        if lm.get(key) != rm.get(key):
            raise ValueError('Comparison frozen identities differ: ' + key)
    keys = sorted(left)
    maximum_difference = 0.0
    changed_between_arms = 0
    regions = {}
    point_rows = []
    for region in sorted({key[0] for key in keys}):
        region_keys = [key for key in keys if key[0] == region]
        directory = baseline_root / 'data_cache' / region / lm['configuration']['split']
        actual_path = directory / 'actual.npy'
        declared = lm.get('inputs', {}).get(region, {}).get('declared_actual_sha256')
        if not declared or sha_file(actual_path) != declared:
            raise ValueError('Evaluation label identity differs')
        actual_all = np.load(actual_path, mmap_mode='r', allow_pickle=False)
        origins = np.load(directory / 'origin_unix.npy', mmap_mode='r', allow_pickle=False)
        targets = np.load(directory / 'target_time_unix.npy', mmap_mode='r', allow_pickle=False)
        for name in ('origin_unix.npy', 'target_time_unix.npy'):
            if sha_file(directory / name) != lm['inputs'][region]['files'][name]:
                raise ValueError('Scoring cache target/origin identity changed')
        truths, a, b, defaults = [], [], [], []
        for key in region_keys:
            l, r = left[key], right[key]
            for field in ('index', 'context_id', 'origin_unix', 'region'):
                if l['record'][field] != r['record'][field]:
                    raise ValueError('Origin context mismatch: ' + field)
            if not np.array_equal(l['arrays']['target_time_unix'], r['arrays']['target_time_unix']):
                raise ValueError('Target axes mismatch')
            difference = float(np.max(np.abs(l['arrays']['base'] - r['arrays']['base'])))
            maximum_difference = max(maximum_difference, difference)
            if difference != 0:
                raise ValueError('Frozen default predictions differ')
            changed_between_arms += int(not np.array_equal(l['arrays']['agent'], r['arrays']['agent']))
            index = l['record']['index']
            if int(origins[index]) != key[1] or not np.array_equal(targets[index], l['arrays']['target_time_unix']):
                raise ValueError('Scoring cache index does not map to the saved target')
            y = np.asarray(actual_all[index]).reshape(24)
            ql, qr = l['arrays']['agent'], r['arrays']['agent']
            ls, rs = aggregate_scores(y, ql), aggregate_scores(y, qr)
            point_rows.append({'region': region, 'origin_unix': key[1],
                'r19_candidate': l['record']['agent']['candidate_id'],
                'r20_candidate': r['record']['agent']['candidate_id'],
                'r19_qs': ls['qs'], 'r20_qs': rs['qs'], 'qs_delta_r20_minus_r19': rs['qs']-ls['qs']})
            truths.append(y); a.append(ql); b.append(qr); defaults.append(l['arrays']['base'])
        y = np.stack(truths)
        regions[region] = {'count': len(region_keys), 'BASE': aggregate_scores(y, np.stack(defaults)),
                          'R19': aggregate_scores(y, np.stack(a)), 'R20': aggregate_scores(y, np.stack(b))}
    equal = {arm: {metric: float(np.mean([row[arm][metric] for row in regions.values()]))
                   for metric in next(iter(regions.values()))[arm]} for arm in ('BASE', 'R19', 'R20')}
    deltas = np.array([row['qs_delta_r20_minus_r19'] for row in point_rows])
    for report, arm in ((lr, 'R19'), (rr, 'R20')):
        for region, recomputed in regions.items():
            for output_kind, kind in (('BASE', 'BASE'), ('AGENT', arm)):
                for metric, value in recomputed[kind].items():
                    if not np.isclose(report['regions'][region][output_kind][metric], value, rtol=1e-12, atol=1e-10):
                        raise ValueError('Recomputed scores disagree with frozen run report')
        for output_kind, kind in (('BASE', 'BASE'), ('AGENT', arm)):
            for metric, value in equal[kind].items():
                if not np.isclose(report['region_equal_mean'][output_kind][metric], value, rtol=1e-12, atol=1e-10):
                    raise ValueError('Recomputed regional aggregation differs from run report')
    def costs(report, rows):
        traces = [row['trace'] for row in rows.values()]
        return {**{key: report.get(key) for key in ('api_calls', 'api_failure_count', 'fallback_count',
                    'prediction_differs_from_base', 'origin_latency_seconds', 'selected_candidates',
                    'iteration_count', 'revision_count', 'origins_with_revision',
                    'llm_selected_tools', 'protocol_required_tools', 'optional_evidence_action_count')},
                'trace_revision_count': sum(len(t.get('revisions', [])) for t in traces),
                'trace_origins_with_revision': sum(bool(t.get('revisions')) for t in traces),
                'iteration_count': sum(len(t.get('iterations', [])) for t in traces),
                'revision_count': sum(len(t.get('revisions', [])) for t in traces),
                'origins_with_revision': sum(bool(t.get('revisions')) for t in traces),
                'optional_evidence_action_count': sum(len(t.get('optional_evidence_actions', [])) for t in traces),
                'unknown_actual_cost_events': sum(len(t.get('resource_accounting', {}).get('unknown_actual_cost_events', []))
                                                 for t in traces),
                'resource_totals_basis': 'Charged budget units; missing tool results can use explicit reserved upper bounds, not exact observed execution.',
                'resource_totals': {key: sum(t.get('costs', {}).get(key, 0) for t in traces)
                    for key in ('api_calls', 'tool_calls', 'candidates', 'episodes', 'historical_evaluations')}}
    return {'status': 'COMPLETE_PAIRED_DEVELOPMENT_COMPARISON', 'reference': str(reference),
            'treatment': str(treatment), 'manifest_ids': {'R19': digest(lm), 'R20': digest(rm)},
            'origins': len(keys), 'split': lm['configuration']['split'], 'seed': lm['configuration']['seed'],
            'default_max_abs_difference': maximum_difference, 'changed_between_arms': changed_between_arms,
            'regions': regions, 'region_equal_mean': equal,
            'qs_improvement_percent_vs_r19': (100 * (equal['R19']['qs']-equal['R20']['qs'])/equal['R19']['qs']
                                               if equal['R19']['qs'] != 0 else None),
            'fairness': {'same_source_fingerprint': lm['source']['sha256'] == rm['source']['sha256'],
                         'source_ids': {'R19': lm['source']['sha256'], 'R20': rm['source']['sha256']},
                         'matching': ['schedule', 'input_files', 'model_weights', 'budgets', 'evidence', 'packages'],
                         'run_metrics_reproduced': True,
                         'source_difference_limit': 'If sources differ, this is a version comparison; common-code equivalence is not proven merely by matching models.'},
            'paired_origins': {'r20_lower_qs': int((deltas < 0).sum()), 'r20_higher_qs': int((deltas > 0).sum()),
                               'equal_qs': int((deltas == 0).sum())},
            'costs': {'R19': costs(lr, left), 'R20': costs(rr, right)}, 'per_origin': point_rows,
            'limitations': ['Split=' + lm['configuration']['split'] + '; this paired policy comparison does not by itself establish a main-table ranking.',
                'Whole-policy comparison does not isolate the causal contribution of the LLM or any individual skill.',
                'Independent sequential feedback ledgers; policy choices may alter later feedback.',
                'Pointwise win counts are descriptive, not independent statistical significance.'],
            'comparison_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--treatment', type=Path, required=True)
    p.add_argument('--baseline-root', type=Path,
                   default=Path('/root/workspace/Code/DualTimesField/outputs/forecasting/aligned_baselines_20261012'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.reference, args.treatment, args.baseline_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_origin',)}, indent=2))


if __name__ == '__main__':
    main()
