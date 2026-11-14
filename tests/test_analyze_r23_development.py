import json

import numpy as np
import pytest

from scripts.analyze_r23_development import analyze, development_gates, METRICS
from forecasting.adaptive_agent.replay import sha_file
from test_analyze_r22_blocked_cv import write_cache, write_replay, REGIONS


def fixture(tmp_path):
    cache = tmp_path / 'cache'
    origins = np.arange(6, dtype=np.int64) * 86400 + 1656597600
    write_cache(cache, 'calibration', origins)
    roots = {name: tmp_path / name for name in ('r21', 'r22', 'r23')}
    policies = {'r21': 'r21_single_lightgbm_input', 'r22': 'r22_state_aligned_input',
                'r23': 'r23_state_aware_input'}
    for name, root in roots.items():
        for region in REGIONS:
            write_replay(root / region, cache, 'calibration', origins, policies[name], region)
            if name == 'r23':
                for path in (root / region / 'origins').glob('*/trace.json'):
                    trace = json.loads(path.read_text())
                    trace.update(stage_c_decisions=[], risk_vetoes=[],
                                 final_commit_reason='agent_commit_identity')
                    path.write_text(json.dumps(trace))
                    record_path = path.parent / 'record.json'
                    record = json.loads(record_path.read_text())
                    record['files']['trace.json'] = sha_file(path)
                    record_path.write_text(json.dumps(record))
    return roots, cache


def test_paired_four_arm_development_analysis(tmp_path):
    roots, cache = fixture(tmp_path)
    result = analyze(roots['r23'], roots['r22'], roots['r21'], cache)
    assert result['origins_per_region'] == 6
    assert result['design']['formal_test'] is False
    assert len(result['blocks']) == 6
    for region in REGIONS:
        row = result['pooled']['regions'][region]
        assert set(row['systems']) == {'identity', 'r21', 'r22', 'r23'}
        assert row['deltas']['r23_minus_identity']['qs'] == 0
        assert row['deltas']['r23_minus_r21']['qs'] > 0
    assert not result['development_gates']['all_regions_passed']


def test_missing_or_incomplete_region_cannot_be_scored(tmp_path):
    roots, cache = fixture(tmp_path)
    path = roots['r23'] / 'QLD1' / 'progress.json'
    record = json.loads(path.read_text())
    record['status'] = 'RUNNING'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='not COMPLETE'):
        analyze(roots['r23'], roots['r22'], roots['r21'], cache)


def test_corrupt_committed_predictions_rejected(tmp_path):
    roots, cache = fixture(tmp_path)
    path = next((roots['r23'] / 'TAS1' / 'origins').glob('*/predictions.npz'))
    path.write_bytes(b'invalid')
    with pytest.raises(ValueError, match='checksum mismatch'):
        analyze(roots['r23'], roots['r22'], roots['r21'], cache)


def test_one_losing_region_blocks_development_promotion():
    regions = {region: {'deltas': {f'r23_minus_{reference}': {
        metric: (0.0 if metric == 'coverage90' else -0.1) for metric in METRICS
    } for reference in ('identity', 'r21')}} for region in REGIONS}
    assert development_gates(regions)['all_regions_passed']
    regions['QLD1']['deltas']['r23_minus_r21']['mae'] = 0.01
    result = development_gates(regions)
    assert not result['all_regions_passed']
    assert result['regions']['NSW1']['passed']
    assert not result['regions']['QLD1']['passed']
