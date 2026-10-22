import copy
import json

import numpy as np
import pytest

from forecasting.full_v5_anchor_balance import aligned_final as e
from forecasting.full_v5_joint.protocol import DAY, HOUR, timestamp


def test_latest_training_cutoffs_purge_boundaries_and_exclude_h2():
    cfg = e.configuration(['cpu'] * 3)
    origins = np.arange(timestamp('2017-01-01'), timestamp('2023-01-01'), HOUR)
    rows = e.training_rows(origins, cfg)
    assert origins[rows['train'][0]] == timestamp('2018-01-01')
    assert origins[rows['train'][-1]] + DAY < timestamp('2022-01-01')
    assert origins[rows['selection'][0]] == timestamp('2022-01-01')
    assert origins[rows['selection'][-1]] + DAY < timestamp('2022-07-01')
    assert not np.intersect1d(rows['train'], rows['selection']).size
    with pytest.raises(ValueError, match='Final-test origins'):
        e.training_rows(np.append(origins, timestamp('2023-01-01')), cfg)


def test_latest_configuration_and_all_nine_selection_barrier(tmp_path):
    cfg = e.configuration(['cpu'] * 3)
    assert cfg['seeds'] == [2026, 2027, 2028]
    assert cfg['tree_seed'] == 2026
    assert cfg['variants'] == ['asymmetric_scale']
    assert cfg['main_output'] == 'guarded'
    assert cfg['epochs'] == 20 and cfg['batch_size'] == 128
    assert len(e.expected_keys(cfg)) == 9
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_text('frozen checkpoint')
    record = dict(epochs_completed=20, checkpoints={str(checkpoint): e.c.base.digest(checkpoint)})
    completed = {key: copy.deepcopy(record) for key in e.expected_keys(cfg)}
    e.verify_barrier(completed, cfg)
    key = next(iter(completed))
    incomplete = dict(completed)
    del incomplete[key]
    with pytest.raises(ValueError, match='All nine'):
        e.verify_barrier(incomplete, cfg)
    completed[key]['epochs_completed'] = 19
    with pytest.raises(ValueError, match='epoch budget'):
        e.verify_barrier(completed, cfg)
    completed[key]['epochs_completed'] = 20
    checkpoint.write_text('changed')
    with pytest.raises(ValueError, match='checkpoint changed'):
        e.verify_barrier(completed, cfg)


def test_incomplete_selection_cannot_open_test_cache(tmp_path, monkeypatch):
    (tmp_path / 'selection_complete.json').write_text('{}')
    monkeypatch.setattr(e.c, 'configure_device', lambda device: None)
    def forbidden(*args, **kwargs):
        pytest.fail('Test data opened before all fits were frozen')
    monkeypatch.setattr(e, 'load_cache_split', forbidden)
    with pytest.raises(ValueError, match='All nine'):
        e.evaluate_region(('NSW1', e.configuration(['cpu'] * 3), tmp_path,
                           str(tmp_path), 'fixture-run', 'cpu'))


def test_run_freezes_all_selections_before_dispatching_test(tmp_path, monkeypatch):
    cfg = e.configuration(['cpu'] * 3)
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_text('frozen')
    completed = {key: dict(epochs_completed=20,
        checkpoints={str(checkpoint): e.c.base.digest(checkpoint)}) for key in e.expected_keys(cfg)}
    results = {f'{m}/{s}/{r}': {} for m in e.ALL_OUTPUTS for s in cfg['seeds'] for r in cfg['regions']}
    monkeypatch.setattr(e, 'prepare', lambda *a: dict(run_id='test'))
    monkeypatch.setattr(e, 'verify_manifest', lambda *a: None)
    calls = []
    def pool(worker, specs):
        if worker is e.train_region:
            assert not (tmp_path / 'selection_complete.json').exists()
            calls.append('train')
            return completed
        assert calls == ['train']
        assert json.loads((tmp_path / 'selection_complete.json').read_text()) == completed
        calls.append('test')
        return results
    monkeypatch.setattr(e.c, 'run_pool', pool)
    e.run(cfg, tmp_path, tmp_path)
    assert calls == ['train', 'test']
    assert json.loads((tmp_path / 'progress.json').read_text())['completed_artifacts'] == 54


def test_latest_run_lock_prevents_duplicate_writer(tmp_path):
    import fcntl
    output = tmp_path / 'run'
    with (tmp_path / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='active runner'):
            e.main(['--output', str(output), '--devices', 'cpu', 'cpu', 'cpu'])
    assert not output.exists()
