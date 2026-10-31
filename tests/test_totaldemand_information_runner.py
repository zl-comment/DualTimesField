"""Runner boundaries: excluded bridge, numeric year window, origin-key score join."""
from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    'information_runner', Path(__file__).parents[1] / 'scripts/run_totaldemand_information.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_bridge_excluded_once_without_reordering_native_quantiles():
    native = np.broadcast_to([4., -2., 3., 1., 0.], (2, 1, 25, 5)).copy()
    native[:, :, 0] = 999999
    native[:, :, -1] = [-10, -9, -8, -7, -6]
    scored = runner.scoring_horizon(native, 2)
    assert scored.shape == (2, 24, 5)
    np.testing.assert_array_equal(scored[0, 0], [4, -2, 3, 1, 0])
    np.testing.assert_array_equal(scored[0, -1], [-10, -9, -8, -7, -6])
    assert not (scored == 999999).any()
    with pytest.raises(ValueError, match='shape'):
        runner.scoring_horizon(native[:, :, :24], 2)


def test_numeric_price_window_excludes_other_years(tmp_path):
    path = tmp_path / 'prices.csv'
    path.write_text('delivery_start_aest,available_at_aest,region,rrp_aud_per_mwh,total_demand_mw\n'
        '2022-12-27T23:00:00+10:00,2022-12-28T00:00:00+10:00,NSW1,earlier,earlier\n'
        '2022-12-28T23:00:00+10:00,2022-12-29T00:00:00+10:00,NSW1,2,3\n'
        '2023-12-31T23:00:00+10:00,2024-01-01T00:00:00+10:00,NSW1,4,5\n'
        '2024-01-01T00:00:00+10:00,2024-01-01T01:00:00+10:00,NSW1,later,later\n')
    frame = runner.load_price_window(path, '2022-12-28', '2024-01-01')
    assert frame['rrp_aud_per_mwh'].tolist() == [2, 4]
    assert frame['total_demand_mw'].tolist() == [3, 5]


def test_score_joins_origins_instead_of_array_positions(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    (root / 'predictions').mkdir(parents=True)
    (root / 'labels').mkdir()
    dates = np.datetime64('2023-01-01') + np.arange(20)
    origins = 1672495200 + np.arange(20) * 86400
    regions = ['NSW1', 'QLD1', 'TAS1']
    manifest = {'execution_commit': 'synthetic', 'origins': origins.tolist(),
                'dates': dates.astype(str).tolist(), 'source_complete': [[True] * 3] * 20,
                'resources': {}}
    for r, region in enumerate(regions):
        y = np.broadcast_to((np.arange(20) + r * 20)[:, None], (20, 24)).copy()
        a = np.repeat((y + 4)[..., None], 5, axis=-1)
        b = np.repeat((y + 3)[..., None], 5, axis=-1)
        runner.save_npz(root / 'predictions' / f'{region}.npz', A=a[::-1], B=b[::-1],
                        source_complete=np.ones(20, dtype=bool), origin_unix=origins[::-1],
                        target_time_unix=(origins[:, None] + np.arange(24) * 3600)[::-1])
        order = np.roll(np.arange(20), 3)
        runner.save_npz(root / 'labels' / f'{region}.npz', actual=y[order], origin_unix=origins[order])
    runner.write_json(root / 'predicted.json', {'files': {
        p.name: runner.sha256(p) for p in (root / 'predictions').iterdir()}})
    monkeypatch.setattr(runner, 'prepared', lambda cfg, include_labels=False: (root, manifest))
    archive = tmp_path / 'archive'
    runner.score({'regions': regions, 'archive_root': str(archive), 'seed': 2026,
                  'candidate_id': 'synthetic'})
    result = json.loads((archive / 'metrics.json').read_text())
    assert result['macro']['A']['mae'] == 4
    assert result['macro']['B']['mae'] == 3
    assert result['decision'] == 'PASS_INFORMATION'
    with np.load(archive / 'paired_predictions.npz') as payload:
        np.testing.assert_array_equal(payload['origin_unix'], origins)
        np.testing.assert_array_equal(payload['actual'][:, 0, 0], np.arange(20))
