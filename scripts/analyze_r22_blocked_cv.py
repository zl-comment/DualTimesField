"""Analyze paired R21/R22 replays as fixed-policy blocked development evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.adaptive_agent.replay import aggregate_scores, digest, sha_file


REGIONS = ('NSW1', 'QLD1', 'TAS1')
METRICS = ('qs', 'mae', 'rmse', 'coverage90', 'width90')
SYSTEMS = ('identity', 'r21', 'r22')
TARGETED_OPERATIONS = {
    'NSW1_keep_recent_24h_demand': ('NSW1', 'history_demand_last24_mean'),
    'TAS1_keep_recent_48h_demand': ('TAS1', 'history_demand_last48_mean'),
}


def _read_json(path):
    return json.loads(Path(path).read_text())


def _validate_complete(root, expected_region):
    root = Path(root)
    manifest = _read_json(root / 'run_manifest.json')
    report = _read_json(root / 'report.json')
    progress = _read_json(root / 'progress.json')
    complete = _read_json(root / 'predictions_complete.json')
    if (root / 'source_changed.json').exists():
        raise ValueError(f'Replay source was invalidated: {root}')
    if report.get('status') != 'COMPLETE' or progress.get('status') != 'COMPLETE':
        raise ValueError(f'Replay is not COMPLETE: {root}')
    expected = manifest.get('expected_origins')
    if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
        raise ValueError(f'Manifest expected_origins is invalid: {root}')
    counts = (report.get('origins'), report.get('expected_origins'),
              progress.get('completed'), complete.get('completed'), complete.get('expected'))
    if any(value != expected for value in counts):
        raise ValueError(f'Replay completion denominator mismatch: {root}')
    manifest_id = digest(manifest)
    if report.get('manifest_id') != manifest_id or complete.get('manifest_id') != manifest_id:
        raise ValueError(f'Replay completion identity mismatch: {root}')
    configured = tuple(manifest.get('configuration', {}).get('regions', ()))
    if configured != (expected_region,):
        raise ValueError(f'Replay must contain only {expected_region}: {root}')
    return manifest, report, manifest_id


def _load_arm(root, expected_region):
    root = Path(root)
    manifest, report, manifest_id = _validate_complete(root, expected_region)
    rows = {}
    origin_root = root / 'origins'
    for directory in sorted(origin_root.iterdir()):
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
        for name, expected in files.items():
            if sha_file(directory / name) != expected:
                raise ValueError('Committed origin checksum mismatch')
        with np.load(directory / 'predictions.npz', allow_pickle=False) as bundle:
            if set(bundle.files) != {'base', 'agent', 'target_time_unix'}:
                raise ValueError('Unexpected prediction array schema')
            arrays = {name: bundle[name].copy() for name in bundle.files}
        origin = record.get('origin_unix')
        region = record.get('region')
        if region != expected_region or isinstance(origin, bool) or not isinstance(origin, int):
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
        selected_id = record.get('agent', {}).get('identity')
        if trace.get('selected_forecast_id') != selected_id:
            raise ValueError('Trace and committed prediction identity disagree')
        key = (region, origin)
        if key in rows:
            raise ValueError('Duplicate replay origin')
        rows[key] = {'record': record, 'arrays': arrays, 'trace': trace}
    if len(rows) != manifest['expected_origins']:
        raise ValueError('Committed origins differ from replay denominator')
    schedule = sorted((row['record']['origin_unix'], row['record']['region'],
                       row['record']['index']) for row in rows.values())
    if manifest.get('schedule_sha256') and digest(schedule) != manifest['schedule_sha256']:
        raise ValueError('Committed origins differ from manifest schedule')
    return manifest, report, rows


def _load_parent(root):
    root = Path(root)
    manifests, reports, rows = {}, {}, {}
    for region in REGIONS:
        manifest, report, region_rows = _load_arm(root / region, region)
        manifests[region], reports[region] = manifest, report
        overlap = set(rows).intersection(region_rows)
        if overlap:
            raise ValueError('Duplicate region/origin across replay directories')
        rows.update(region_rows)
    return manifests, reports, rows


def _check_paired_runs(r21_manifests, r22_manifests, r21_rows, r22_rows, split):
    if set(r21_rows) != set(r22_rows):
        raise ValueError('R21 and R22 origin schedules must match exactly')
    for region in REGIONS:
        r21_manifest, r22_manifest = r21_manifests[region], r22_manifests[region]
        expected_policies = ((r21_manifest, 'R21', 'r21_single_lightgbm_input'),
                             (r22_manifest, 'R22', 'r22_state_aligned_input'))
        for manifest, label, expected_policy in expected_policies:
            configuration = manifest.get('configuration', {})
            if configuration.get('split') != split:
                raise ValueError(f'{label} {region} split does not match --split')
            if configuration.get('policy') != expected_policy:
                raise ValueError(f'{label} {region} root has the wrong replay policy')
        for field in ('seed', 'origins_per_region', 'regions', 'split'):
            if (r21_manifest['configuration'].get(field)
                    != r22_manifest['configuration'].get(field)):
                raise ValueError(f'R21/R22 {region} configuration mismatch: {field}')
        left_input = r21_manifest.get('inputs', {}).get(region, {})
        right_input = r22_manifest.get('inputs', {}).get(region, {})
        for field in ('declared_actual_sha256', 'files'):
            if left_input.get(field) != right_input.get(field):
                raise ValueError(f'R21/R22 aligned input declaration mismatch for {region}: {field}')
    for label, manifests in (('R21', r21_manifests), ('R22', r22_manifests)):
        reference = manifests[REGIONS[0]]['configuration']
        for region in REGIONS[1:]:
            for field in ('seed', 'origins_per_region', 'split', 'policy'):
                if manifests[region]['configuration'].get(field) != reference.get(field):
                    raise ValueError(f'{label} regional configuration mismatch: {field}')
    for key in r21_rows:
        left, right = r21_rows[key], r22_rows[key]
        for field in ('index', 'region', 'origin_unix'):
            if left['record'].get(field) != right['record'].get(field):
                raise ValueError('R21/R22 origin context mismatch: ' + field)
        if not np.array_equal(left['arrays']['target_time_unix'], right['arrays']['target_time_unix']):
            raise ValueError('R21/R22 target axes must match exactly')
        if not np.array_equal(left['arrays']['base'], right['arrays']['base']):
            raise ValueError('R21/R22 identity forecasts must match exactly')
    schedules = {
        region: [origin for row_region, origin in sorted(r21_rows) if row_region == region]
        for region in REGIONS
    }
    reference = schedules[REGIONS[0]]
    if not reference or any(schedules[region] != reference for region in REGIONS[1:]):
        raise ValueError('All three regions must use the same ordered origins')
    return reference


def _verify_declared_input(manifest, region, directory, name):
    declaration = manifest.get('inputs', {}).get(region, {})
    if name == 'actual.npy':
        expected = declaration.get('declared_actual_sha256')
    else:
        expected = declaration.get('files', {}).get(name)
    if expected is not None and sha_file(directory / name) != expected:
        raise ValueError(f'Aligned baseline {name} differs from replay manifest for {region}')


def _load_aligned_data(root, split, manifests, ordered_origins):
    root = Path(root)
    aligned = {}
    for region in REGIONS:
        directory = root / 'data_cache' / region / split
        for name in ('origin_unix.npy', 'target_time_unix.npy', 'actual.npy'):
            _verify_declared_input(manifests[region], region, directory, name)
        cache_origins = np.load(directory / 'origin_unix.npy', allow_pickle=False)
        cache_targets = np.load(directory / 'target_time_unix.npy', allow_pickle=False)
        actual_all = np.load(directory / 'actual.npy', allow_pickle=False)
        if (cache_origins.ndim != 1 or cache_targets.shape != (len(cache_origins), 24)
                or actual_all.shape != (len(cache_origins), 24)
                or cache_origins.dtype.kind not in 'iu' or cache_targets.dtype.kind not in 'iu'):
            raise ValueError(f'Aligned baseline arrays have invalid shape for {region}')
        positions = {int(origin): index for index, origin in enumerate(cache_origins)}
        if len(positions) != len(cache_origins):
            raise ValueError(f'Aligned baseline origins are duplicated for {region}')
        try:
            indices = np.asarray([positions[origin] for origin in ordered_origins], dtype=int)
        except KeyError as exc:
            raise ValueError(f'Replay origin is absent from aligned baseline for {region}') from exc
        expected_targets = np.stack([
            manifest_row_target(origin) for origin in ordered_origins
        ])
        selected_targets = np.asarray(cache_targets[indices])
        if not np.array_equal(selected_targets, expected_targets):
            raise ValueError(f'Aligned baseline target axes differ for {region}')
        actual = np.asarray(actual_all[indices]).copy()
        if not np.isfinite(actual).all():
            raise ValueError(f'Aligned baseline actuals are nonfinite for {region}')
        aligned[region] = {'actual': actual, 'target_time_unix': selected_targets}
    return aligned


def manifest_row_target(origin):
    return int(origin) + np.arange(24, dtype=np.int64) * 3600


def _score(actual, identity, r21, r22):
    predictions = {'identity': identity, 'r21': r21, 'r22': r22}
    systems = {name: aggregate_scores(actual, predictions[name]) for name in SYSTEMS}
    deltas = {
        'r22_minus_r21': {metric: systems['r22'][metric] - systems['r21'][metric]
                          for metric in METRICS},
        'r22_minus_identity': {metric: systems['r22'][metric] - systems['identity'][metric]
                              for metric in METRICS},
    }
    return {'systems': systems, 'deltas': deltas}


def _mean_evaluations(region_rows):
    result = {'systems': {}, 'deltas': {}}
    for system in SYSTEMS:
        result['systems'][system] = {
            metric: float(np.mean([row['systems'][system][metric] for row in region_rows.values()]))
            for metric in METRICS
        }
    for comparison in ('r22_minus_r21', 'r22_minus_identity'):
        result['deltas'][comparison] = {
            metric: float(np.mean([row['deltas'][comparison][metric]
                                   for row in region_rows.values()]))
            for metric in METRICS
        }
    return result


def _balanced_blocks(length, blocks):
    if isinstance(blocks, bool) or not isinstance(blocks, int) or not 1 <= blocks <= length:
        raise ValueError('--blocks must be positive and no greater than origins per region')
    result = [values for values in np.array_split(np.arange(length), blocks) if len(values)]
    if max(map(len, result)) - min(map(len, result)) > 1:
        raise AssertionError('Balanced block construction failed')
    return result


def _stage_c(trace, identity_candidate, final_candidate):
    events = trace.get('stage_c_decisions', [])
    if isinstance(events, dict):
        events = [events]
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise ValueError('R22 stage_c_decisions must be a list of objects')
    final_reason = trace.get('final_commit_reason')
    if final_reason is not None and not isinstance(final_reason, str):
        raise ValueError('R22 final_commit_reason must be a string')
    if len(events) > 1:
        raise ValueError('R22 may record at most one Stage C decision')
    if events:
        event = events[0]
        action = event.get('action')
        candidate = event.get('selected_candidate_id')
        if action not in ('approve_selected', 'revert_to_identity') or not isinstance(candidate, str):
            raise ValueError('Invalid R22 Stage C action or selected candidate')
        if action == 'approve_selected' and final_candidate != candidate:
            raise ValueError('Stage C approval did not commit its selected candidate')
        if action == 'revert_to_identity' and final_candidate != identity_candidate:
            raise ValueError('Stage C reversion did not commit identity')
        reason = final_reason or event.get('reason_code') or action
        return {'disposition': action, 'review_candidate': candidate, 'reason': reason}
    if isinstance(final_reason, str) and final_reason.startswith('hard_veto:'):
        if final_candidate != identity_candidate:
            raise ValueError('Hard veto did not commit identity')
        packet = trace.get('risk_review_packet', {})
        candidate = (trace.get('review_candidate_id')
                     or trace.get('stage_b_selected_candidate_id')
                     or trace.get('proposed_candidate_id')
                     or (packet.get('selected_candidate_id') if isinstance(packet, dict) else None))
        return {'disposition': 'hard_veto_revert', 'review_candidate': candidate,
                'reason': final_reason}
    if final_candidate == identity_candidate and isinstance(final_reason, str) \
            and final_reason.startswith('risk_review_'):
        packet = trace.get('risk_review_packet', {})
        candidate = (trace.get('review_candidate_id')
                     or trace.get('stage_b_selected_candidate_id')
                     or trace.get('proposed_candidate_id')
                     or (packet.get('selected_candidate_id') if isinstance(packet, dict) else None))
        return {'disposition': 'revert_to_identity', 'review_candidate': candidate,
                'reason': final_reason}
    if final_candidate != identity_candidate:
        raise ValueError('Nonidentity R22 commit is missing Stage C approval')
    return {'disposition': 'identity_bypass', 'review_candidate': None,
            'reason': final_reason or 'identity_bypass'}


def _operation_matches(candidate_id, operation_id):
    return isinstance(candidate_id, str) and candidate_id.split(':')[-1] == operation_id


def _trace_summary(region_data):
    dispositions, reasons = Counter(), Counter()
    final_candidates, review_candidates = Counter(), Counter()
    trace_rows = []
    for region in REGIONS:
        data = region_data[region]
        for position, trace in enumerate(data['traces']):
            final_candidate = data['final_candidates'][position]
            identity_candidate = data['identity_candidates'][position]
            stage_c = _stage_c(trace, identity_candidate, final_candidate)
            dispositions[stage_c['disposition']] += 1
            reasons[stage_c['reason']] += 1
            final_candidates[final_candidate] += 1
            if stage_c['review_candidate'] is not None:
                review_candidates[stage_c['review_candidate']] += 1
            trace_rows.append({'region': region, 'position': position,
                               'stage_c': stage_c})
    return {
        'stage_c': {
            'approve_selected': dispositions['approve_selected'],
            'revert_to_identity': dispositions['revert_to_identity'],
            'hard_veto_revert': dispositions['hard_veto_revert'],
            'identity_bypass': dispositions['identity_bypass'],
            'total_reverts': (dispositions['revert_to_identity']
                              + dispositions['hard_veto_revert']),
        },
        'final_commit_reasons': dict(sorted(reasons.items())),
        'review_candidate_counts': dict(sorted(review_candidates.items())),
        'final_candidate_counts': dict(sorted(final_candidates.items())),
        '_rows': trace_rows,
    }


def _slice_result(name, region, operation_id, region_data, trace_rows):
    matched = []
    dispositions = Counter()
    for row in trace_rows:
        if row['region'] != region:
            continue
        position = row['position']
        candidate = row['stage_c']['review_candidate']
        if candidate is None:
            candidate = region_data[region]['final_candidates'][position]
        if _operation_matches(candidate, operation_id):
            matched.append(position)
            dispositions[row['stage_c']['disposition']] += 1
    result = {'region': region, 'operation_id': operation_id,
              'selection_basis': 'Stage C review candidate, falling back to final candidate when no review occurred',
              'count': len(matched), 'stage_c_dispositions': dict(sorted(dispositions.items()))}
    if matched:
        index = np.asarray(matched, dtype=int)
        data = region_data[region]
        result['evaluation'] = _score(data['actual'][index], data['identity'][index],
                                      data['r21'][index], data['r22'][index])
        result['first_origin'] = int(data['origins'][index[0]])
        result['last_origin'] = int(data['origins'][index[-1]])
    else:
        result['evaluation'] = None
        result['first_origin'] = None
        result['last_origin'] = None
    return name, result


def analyze(r22_root, r21_root, aligned_baseline_root, split, blocks=6):
    r21_manifests, _, r21_rows = _load_parent(r21_root)
    r22_manifests, _, r22_rows = _load_parent(r22_root)
    ordered_origins = _check_paired_runs(
        r21_manifests, r22_manifests, r21_rows, r22_rows, split)
    aligned = _load_aligned_data(
        aligned_baseline_root, split, r21_manifests, ordered_origins)
    block_indices = _balanced_blocks(len(ordered_origins), blocks)
    region_data = {}
    for region in REGIONS:
        keys = [(region, origin) for origin in ordered_origins]
        left = [r21_rows[key] for key in keys]
        right = [r22_rows[key] for key in keys]
        targets = np.stack([row['arrays']['target_time_unix'] for row in left])
        if not np.array_equal(targets, aligned[region]['target_time_unix']):
            raise ValueError(f'Replay and aligned target axes differ for {region}')
        region_data[region] = {
            'origins': np.asarray(ordered_origins, dtype=np.int64),
            'actual': aligned[region]['actual'],
            'identity': np.stack([row['arrays']['base'] for row in left]),
            'r21': np.stack([row['arrays']['agent'] for row in left]),
            'r22': np.stack([row['arrays']['agent'] for row in right]),
            'traces': [row['trace'] for row in right],
            'identity_candidates': [row['record']['base']['candidate_id'] for row in right],
            'final_candidates': [row['record']['agent']['candidate_id'] for row in right],
        }
    pooled_regions = {
        region: _score(data['actual'], data['identity'], data['r21'], data['r22'])
        for region, data in region_data.items()
    }
    block_rows = []
    for block_number, index in enumerate(block_indices, start=1):
        regions = {
            region: _score(data['actual'][index], data['identity'][index],
                           data['r21'][index], data['r22'][index])
            for region, data in region_data.items()
        }
        block_rows.append({
            'block': block_number, 'position_start': int(index[0]),
            'position_stop_exclusive': int(index[-1] + 1), 'origins_per_region': len(index),
            'first_origin': int(ordered_origins[index[0]]),
            'last_origin': int(ordered_origins[index[-1]]),
            'regions': regions, 'region_equal': _mean_evaluations(regions),
        })
    traces = _trace_summary(region_data)
    trace_rows = traces.pop('_rows')
    targeted = dict(_slice_result(name, region, operation, region_data, trace_rows)
                    for name, (region, operation) in TARGETED_OPERATIONS.items())
    return {
        'status': 'COMPLETE_FIXED_POLICY_BLOCKED_EVALUATION',
        'inputs': {'r22_root': str(Path(r22_root).resolve()),
                   'r21_root': str(Path(r21_root).resolve()),
                   'aligned_baseline_root': str(Path(aligned_baseline_root).resolve()),
                   'split': split,
                   'manifest_ids': {
                       'r21': {region: digest(r21_manifests[region]) for region in REGIONS},
                       'r22': {region: digest(r22_manifests[region]) for region in REGIONS},
                   }},
        'evaluation_design': {
            'name': 'fixed-policy blocked evaluation',
            'repeated_exposure': True,
            'independent_blind_test': False,
            'within_block_fitting': False,
            'blocks': blocks,
            'construction': ('Per region, the common origins are ordered chronologically and split into '
                             'contiguous balanced blocks; block sizes differ by at most one origin.'),
            'interpretation': ('Descriptive repeated-exposure development evaluation of already fixed R21 '
                               'and R22 policies. No policy or threshold is fitted inside a block, and these '
                               'results must not be presented as an independent blind test.'),
        },
        'regions': list(REGIONS), 'origins_per_region': len(ordered_origins),
        'pooled': {'regions': pooled_regions,
                   'region_equal': _mean_evaluations(pooled_regions)},
        'blocks': block_rows,
        'trace_summary': traces,
        'targeted_slices': targeted,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r22-root', type=Path, required=True)
    parser.add_argument('--r21-root', type=Path, required=True)
    parser.add_argument('--aligned-baseline-root', type=Path, required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--blocks', type=int, default=6)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.r22_root, args.r21_root, args.aligned_baseline_root,
                     args.split, args.blocks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output),
                      'origins_per_region': result['origins_per_region'],
                      'blocks': len(result['blocks'])}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
