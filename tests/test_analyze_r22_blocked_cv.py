"""Synthetic end-to-end tests for the R22 fixed-policy blocked evaluator."""
import json
from pathlib import Path

import numpy as np
import pytest

from forecasting.adaptive_agent.replay import digest, sha_file
from scripts import analyze_r22_blocked_cv as blocked


REGIONS = ('NSW1', 'QLD1', 'TAS1')
IDENTITY = 'lightgbm-input:identity'
KEEP24 = 'lightgbm-input:history_demand_last24_mean'
KEEP48 = 'lightgbm-input:history_demand_last48_mean'


def quantiles(center):
    center = np.asarray(center)
    return np.stack([center - 4, center - 2, center, center + 2, center + 4], axis=-1)


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def write_cache(root, split, origins):
    for region_index, region in enumerate(REGIONS):
        directory = root / 'data_cache' / region / split
        directory.mkdir(parents=True)
        target = origins[:, None] + np.arange(24) * 3600
        actual = np.stack([np.full(24, 20 + region_index * 10 + index, dtype=float)
                           for index in range(len(origins))])
        np.save(directory / 'origin_unix.npy', origins)
        np.save(directory / 'target_time_unix.npy', target)
        np.save(directory / 'actual.npy', actual)


def input_declarations(cache_root, split):
    result = {}
    for region in REGIONS:
        directory = cache_root / 'data_cache' / region / split
        result[region] = {
            'declared_actual_sha256': sha_file(directory / 'actual.npy'),
            'files': {name: sha_file(directory / name)
                      for name in ('origin_unix.npy', 'target_time_unix.npy')},
        }
    return result


def r22_trace(region, index, final_candidate):
    trace = {'selected_candidate_id': final_candidate, 'risk_vetoes': [],
             'stage_c_decisions': []}
    if region == 'NSW1' and index == 0:
        trace.update(stage_c_decisions=[{'phase': 'risk_review', 'action': 'approve_selected',
                                        'selected_candidate_id': KEEP24}],
                     final_commit_reason='risk_review_approve')
    elif region == 'NSW1' and index == 1:
        trace.update(stage_c_decisions=[{'phase': 'risk_review', 'action': 'revert_to_identity',
                                        'selected_candidate_id': KEEP24}],
                     final_commit_reason='risk_review_revert')
    elif region == 'NSW1' and index == 2:
        trace.update(risk_vetoes=['material_downside'],
                     stage_b_selected_candidate_id=KEEP24,
                     final_commit_reason='hard_veto:material_downside')
    elif region == 'TAS1' and index == 0:
        trace.update(stage_c_decisions=[{'phase': 'risk_review', 'action': 'approve_selected',
                                        'selected_candidate_id': KEEP48}],
                     final_commit_reason='risk_review_approve')
    else:
        trace['final_commit_reason'] = 'agent_commit_identity'
    return trace


def write_replay(root, cache_root, split, origins, policy, region):
    inputs = input_declarations(cache_root, split)
    schedule = [(int(origin), region, index) for index, origin in enumerate(origins)]
    manifest = {
        'configuration': {'policy': policy, 'split': split, 'regions': [region],
                          'origins_per_region': str(len(origins)), 'seed': 2026},
        'inputs': {region: inputs[region]}, 'expected_origins': len(schedule),
        'schedule_sha256': digest(schedule),
    }
    manifest_id = digest(manifest)
    root.mkdir(parents=True)
    origin_root = root / 'origins'
    origin_root.mkdir()
    write_json(root / 'run_manifest.json', manifest)
    cache_directory = cache_root / 'data_cache' / region / split
    actual = np.load(cache_directory / 'actual.npy', allow_pickle=False)
    targets = np.load(cache_directory / 'target_time_unix.npy', allow_pickle=False)
    for index, origin in enumerate(origins):
        identity = quantiles(actual[index] + 2)
        r21 = quantiles(actual[index] + 1)
        final_candidate = IDENTITY
        agent = r21 if policy == 'r21_single_lightgbm_input' else identity
        if policy == 'r22_state_aligned_input':
            if region == 'NSW1' and index == 0:
                final_candidate, agent = KEEP24, quantiles(actual[index])
            elif region == 'TAS1' and index == 0:
                final_candidate, agent = KEEP48, quantiles(actual[index])
        directory = origin_root / f'{int(origin):012d}_{region}'
        directory.mkdir()
        with (directory / 'predictions.npz').open('wb') as stream:
            np.savez_compressed(stream, base=identity, agent=agent,
                                target_time_unix=targets[index])
        trace = ({'selected_candidate_id': final_candidate}
                 if policy == 'r21_single_lightgbm_input'
                 else r22_trace(region, index, final_candidate))
        trace['selected_forecast_id'] = f'{policy}:{region}:{index}:agent'
        write_json(directory / 'trace.json', trace)
        write_json(directory / 'audit.json', {'synthetic': True})
        files = {name: sha_file(directory / name)
                 for name in ('predictions.npz', 'trace.json', 'audit.json')}
        record = {
            'manifest_id': manifest_id, 'region': region, 'origin_unix': int(origin),
            'index': index, 'context_id': f'{region}:{origin}',
            'base': {'candidate_id': IDENTITY, 'identity': f'{policy}:{region}:{index}:base'},
            'agent': {'candidate_id': final_candidate,
                      'identity': trace['selected_forecast_id']},
            'files': files,
        }
        write_json(directory / 'record.json', record)
    expected = len(schedule)
    write_json(root / 'report.json', {'status': 'COMPLETE', 'manifest_id': manifest_id,
                                      'origins': expected, 'expected_origins': expected})
    write_json(root / 'progress.json', {'status': 'COMPLETE', 'completed': expected,
                                        'total': expected})
    write_json(root / 'predictions_complete.json', {'manifest_id': manifest_id,
                                                     'completed': expected, 'expected': expected})


@pytest.fixture
def synthetic_runs(tmp_path):
    split = 'calibration'
    origins = np.asarray([1_700_000_000 + day * 86400 for day in range(4)], dtype=np.int64)
    cache = tmp_path / 'aligned'
    write_cache(cache, split, origins)
    r21, r22 = tmp_path / 'r21', tmp_path / 'r22'
    for region in REGIONS:
        write_replay(r21 / region, cache, split, origins,
                     'r21_single_lightgbm_input', region)
        write_replay(r22 / region, cache, split, origins,
                     'r22_state_aligned_input', region)
    return r21, r22, cache, split


def test_blocked_analysis_scores_balanced_blocks_and_risk_trace(synthetic_runs, monkeypatch):
    r21, r22, cache, split = synthetic_runs
    original_load = blocked.np.load
    actual_reads = []

    def counted_load(path, *args, **kwargs):
        if Path(path).name == 'actual.npy':
            actual_reads.append(Path(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(blocked.np, 'load', counted_load)
    result = blocked.analyze(r22, r21, cache, split, blocks=2)
    assert len(actual_reads) == 3
    assert result['status'] == 'COMPLETE_FIXED_POLICY_BLOCKED_EVALUATION'
    assert result['evaluation_design']['repeated_exposure'] is True
    assert result['evaluation_design']['independent_blind_test'] is False
    assert result['evaluation_design']['within_block_fitting'] is False
    assert set(result['inputs']['manifest_ids']['r21']) == set(REGIONS)
    assert set(result['inputs']['manifest_ids']['r22']) == set(REGIONS)
    assert [row['origins_per_region'] for row in result['blocks']] == [2, 2]
    assert result['pooled']['region_equal']['deltas']['r22_minus_r21']['qs'] > 0
    assert result['pooled']['region_equal']['deltas']['r22_minus_identity']['qs'] < 0
    assert set(result['pooled']['regions']['NSW1']['systems']) == {'identity', 'r21', 'r22'}
    stage_c = result['trace_summary']['stage_c']
    assert stage_c == {'approve_selected': 2, 'revert_to_identity': 1,
                       'hard_veto_revert': 1, 'identity_bypass': 8, 'total_reverts': 2}
    assert result['trace_summary']['final_commit_reasons']['risk_review_approve'] == 2
    assert result['targeted_slices']['NSW1_keep_recent_24h_demand']['count'] == 3
    assert result['targeted_slices']['TAS1_keep_recent_48h_demand']['count'] == 1


def test_cli_writes_json_without_claiming_blind_test(synthetic_runs, tmp_path):
    r21, r22, cache, split = synthetic_runs
    output = tmp_path / 'analysis' / 'blocked.json'
    assert blocked.main(['--r22-root', str(r22), '--r21-root', str(r21),
                         '--aligned-baseline-root', str(cache), '--split', split,
                         '--blocks', '2', '--output', str(output)]) == 0
    saved = json.loads(output.read_text())
    assert saved['evaluation_design']['name'] == 'fixed-policy blocked evaluation'
    assert 'must not be presented as an independent blind test' in saved['evaluation_design']['interpretation']
    with pytest.raises(FileExistsError):
        blocked.main(['--r22-root', str(r22), '--r21-root', str(r21),
                      '--aligned-baseline-root', str(cache), '--split', split,
                      '--blocks', '2', '--output', str(output)])


def test_incomplete_or_misaligned_replays_fail_closed(synthetic_runs):
    r21, r22, cache, split = synthetic_runs
    progress_path = r22 / 'NSW1' / 'progress.json'
    progress = json.loads(progress_path.read_text())
    progress['status'] = 'PREDICTING'
    write_json(progress_path, progress)
    with pytest.raises(ValueError, match='not COMPLETE'):
        blocked.analyze(r22, r21, cache, split, blocks=2)


def test_tampered_prediction_target_axis_fails_closed(synthetic_runs):
    r21, r22, cache, split = synthetic_runs
    directory = sorted((r22 / 'NSW1' / 'origins').iterdir())[0]
    with np.load(directory / 'predictions.npz', allow_pickle=False) as bundle:
        base, agent = bundle['base'].copy(), bundle['agent'].copy()
        target = bundle['target_time_unix'].copy()
    target[0] += 1
    with (directory / 'predictions.npz').open('wb') as stream:
        np.savez_compressed(stream, base=base, agent=agent, target_time_unix=target)
    record = json.loads((directory / 'record.json').read_text())
    record['files']['predictions.npz'] = sha_file(directory / 'predictions.npz')
    write_json(directory / 'record.json', record)
    with pytest.raises(ValueError, match='target axis'):
        blocked.analyze(r22, r21, cache, split, blocks=2)
