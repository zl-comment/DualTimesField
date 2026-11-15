"""Synthetic tests for the paired R24a development evaluator."""
import json

import numpy as np
import pytest

from forecasting.adaptive_agent.replay import digest, sha_file
from scripts.analyze_r24_development import development_gates, load_flat_arm
from test_analyze_r22_blocked_cv import (
    REGIONS, write_cache, write_json, write_replay, input_declarations,
)
from pathlib import Path


def write_flat_replay(root, cache, split, origins, policy='r24_iterative_loop_input'):
    """One manifest for all regions; origin bundles live under a flat origins/."""
    inputs = input_declarations(cache, split)
    schedule = []
    for region in REGIONS:
        schedule += [(int(origin), region, index) for index, origin in enumerate(origins)]
    manifest = {
        'configuration': {'policy': policy, 'split': split, 'regions': list(REGIONS),
                          'origins_per_region': str(len(origins)), 'seed': 2026},
        'inputs': inputs, 'expected_origins': len(schedule),
        'schedule_sha256': digest(schedule),
    }
    manifest_id = digest(manifest)
    (root / 'origins').mkdir(parents=True)
    write_json(root / 'run_manifest.json', manifest)
    targets = np.load(cache / 'data_cache' / REGIONS[0] / split / 'target_time_unix.npy')
    actual = {region: np.load(cache / f'data_cache/{region}/{split}/actual.npy')
              for region in REGIONS}
    for region in REGIONS:
        for index, origin in enumerate(origins):
            identity = np.stack([actual[region][index] - 2, actual[region][index] - 1,
                                 actual[region][index], actual[region][index] + 1,
                                 actual[region][index] + 2], axis=-1)
            directory = root / 'origins' / f'{int(origin):012d}_{region}'
            directory.mkdir()
            with (directory / 'predictions.npz').open('wb') as stream:
                np.savez_compressed(stream, base=identity, agent=identity,
                                    target_time_unix=targets[index])
            trace = {'selected_candidate_id': 'lightgbm-input:identity',
                     'selected_forecast_id': f'flat:{region}:{index}:agent',
                     'stop_reason': 'agent_commit_identity', 'loop': {'rounds': [
                         {'round': 1, 'action': 'diagnose_one:x', 'productive': True}],
                         'stop_reason': 'agent_stop'}, 'calls': [], 'fallback': None,
                     'selected_candidate_id_backup': None}
            write_json(directory / 'trace.json', trace)
            write_json(directory / 'audit.json', {'synthetic': True})
            write_json(directory / 'record.json', {
                'manifest_id': manifest_id, 'region': region, 'origin_unix': int(origin),
                'index': index, 'context_id': f'{region}:{origin}',
                'base': {'candidate_id': 'lightgbm-input:identity',
                         'identity': f'flat:{region}:{index}:base'},
                'agent': {'candidate_id': 'lightgbm-input:identity',
                          'identity': trace['selected_forecast_id']},
                'files': {name: sha_file(directory / name) for name in
                          ('predictions.npz', 'trace.json', 'audit.json')}})
    write_json(root / 'report.json', {'status': 'COMPLETE', 'manifest_id': manifest_id,
                                      'origins': len(schedule),
                                      'expected_origins': len(schedule)})
    write_json(root / 'progress.json', {'status': 'COMPLETE', 'completed': len(schedule),
                                        'total': len(schedule)})
    write_json(root / 'predictions_complete.json', {'manifest_id': manifest_id,
                                                    'completed': len(schedule),
                                                    'expected': len(schedule)})


@pytest.fixture
def flat_arm(tmp_path):
    split = 'calibration'
    origins = np.arange(4, dtype=np.int64) * 86400 + 1656597600
    cache = tmp_path / 'cache'
    write_cache(cache, split, origins)
    root = tmp_path / 'r24'
    write_flat_replay(root, cache, split, origins)
    return root, cache


def test_flat_loader_accepts_and_counts_all_regions(flat_arm):
    root, _ = flat_arm
    manifest, rows = load_flat_arm(root)
    assert len(rows) == 12
    assert {region for region, _ in rows} == set(REGIONS)
    assert manifest['configuration']['policy'] == 'r24_iterative_loop_input'


def test_flat_loader_rejects_checksum_tampering(flat_arm):
    root, _ = flat_arm
    target = sorted((root / 'origins').iterdir())[0] / 'trace.json'
    trace = json.loads(target.read_text())
    trace['stop_reason'] = 'risk_review_approved_commit'
    target.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match='checksum mismatch'):
        load_flat_arm(root)


def test_flat_loader_rejects_wrong_policy(flat_arm, tmp_path):
    root, cache = flat_arm
    split = 'calibration'
    origins = np.arange(4, dtype=np.int64) * 86400 + 1656597600
    other = tmp_path / 'other'
    write_flat_replay(other, cache, split, origins, policy='r23_state_aware_input')
    with pytest.raises(ValueError, match='policy'):
        load_flat_arm(other)


def _evaluation(deltas):
    return {'deltas': deltas}


def test_development_gates_require_identity_and_r23_improvement():
    def delta(qs, mae, rmse=0.0, coverage=0.0):
        return {'qs': qs, 'mae': mae, 'rmse': rmse, 'coverage90': coverage}

    good = _evaluation({'r24_minus_identity': delta(-.1, -.1),
                        'r24_minus_r23': delta(-.05, -.05)})
    bad_qs = _evaluation({'r24_minus_identity': delta(-.1, -.1),
                          'r24_minus_r23': delta(.05, -.05)})
    bad_coverage = _evaluation({'r24_minus_identity': delta(-.1, -.1, 0.0, -.01),
                                'r24_minus_r23': delta(-.05, -.05)})
    gates = development_gates({'NSW1': good, 'QLD1': good, 'TAS1': good})
    assert gates['all_regions_passed']
    gates = development_gates({'NSW1': good, 'QLD1': bad_qs, 'TAS1': good})
    assert not gates['all_regions_passed']
    assert not gates['regions']['QLD1']['comparisons']['r23']['qs_improved']
    gates = development_gates({'NSW1': bad_coverage, 'QLD1': good, 'TAS1': good})
    assert not gates['all_regions_passed']
    assert not gates['regions']['NSW1']['comparisons']['identity']['coverage_loss_within_0_005']
