"""Full-run safety: immutable provenance, caches, and resumable job ownership."""
import copy
import fcntl
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from forecasting.aligned_experiment import (
    SPLITS, archive_incomplete, atomic_json, completed, freeze_sources,
    held_worker, load_cache, save_cache, sha, validate_config, verify_snapshot,
)
from forecasting.aligned_inputs import AlignedSplit, INPUT_KEYS
from forecasting.aligned_scoring import make_prediction_artifact, save_prediction_artifact, QUANTILE_LEVELS

ROOT = Path(__file__).resolve().parents[1]


def test_formal_config_is_complete_and_cannot_be_smoke_or_missing_model():
    config = yaml.safe_load((ROOT / 'configs/aemo_forecast_aligned_20261012.yaml').read_text())
    validate_config(config)
    for key, value in [('max_origins', 64), ('models', config['experiment']['models'][:-1]), ('seeds', [2026])]:
        bad = copy.deepcopy(config)
        bad['experiment'][key] = value
        with pytest.raises(ValueError):
            validate_config(bad)


def test_snapshot_includes_backbone_and_detects_mutation(tmp_path):
    project = tmp_path / 'project'
    for folder in ('forecasting', 'src'):
        (project / folder).mkdir(parents=True)
        (project / folder / 'model.py').write_text('value=1\n')
    files, digest = freeze_sources(project, tmp_path / 'source')
    (tmp_path / 'config.yaml').write_text('version: 1\n')
    manifest = {'source_files': files, 'source_hash': digest, 'config_hash': sha(tmp_path / 'config.yaml')}
    verify_snapshot(tmp_path, manifest)
    (tmp_path / 'source/src/model.py').write_text('value=2\n')
    with pytest.raises(ValueError, match='source changed'):
        verify_snapshot(tmp_path, manifest)


def test_cache_roundtrip_and_tamper_rejection(tmp_path):
    data = {}
    for i, name in enumerate(SPLITS):
        origins = np.arange(3, dtype='int64') * 3600 + i * 100000
        data[name] = AlignedSplit({key: np.ones((3, 2, 1), dtype='float32') for key in INPUT_KEYS},
                                  np.ones((3, 2)), origins, origins[:, None] + np.arange(2) * 3600,
                                  {'test': name}, np.ones((3, 2)))
    save_cache(tmp_path, data)
    fingerprint = sha(tmp_path / 'hashes.json')
    result = load_cache(tmp_path, fingerprint)
    np.testing.assert_array_equal(result['train'].inputs['history_values'], data['train'].inputs['history_values'])
    assert not result['train'].inputs['history_values'].flags.writeable
    np.save(tmp_path / 'test/actual.npy', np.zeros((3, 2)))
    with pytest.raises(ValueError, match='Cached data changed'):
        load_cache(tmp_path, fingerprint)


def test_partial_attempt_is_archived_without_losing_worker_lock_or_log(tmp_path):
    for name in ('worker.lock', 'worker.log', 'best_model.pt', 'status.json'):
        (tmp_path / name).write_text(name)
    with (tmp_path / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert held_worker(tmp_path)
        archive_incomplete(tmp_path)
        assert held_worker(tmp_path)
    assert not held_worker(tmp_path)
    assert (tmp_path / 'worker.log').read_text() == 'worker.log'
    assert not (tmp_path / 'best_model.pt').exists()
    assert len(list((tmp_path / 'attempts').glob('*/best_model.pt'))) == 1


def test_completed_requires_cell_identity_and_all_saved_hashes(tmp_path):
    job = {'id': 'linear/2026/NSW1', 'model': 'linear', 'seed': 2026, 'region': 'NSW1', 'output': 'cell'}
    path = tmp_path / 'cell'
    path.mkdir()
    artifact = make_prediction_artifact(actual=np.zeros((2, 2)), origin_unix=np.array([0, 3600]),
        target_time_unix=np.array([[0, 3600], [3600, 7200]]), quantiles=np.tile(np.arange(5), (2, 2, 1)),
        quantile_levels=QUANTILE_LEVELS, metadata={'protocol_id':'v1','split':'test','region':'NSW1',
        'seed':2026,'model':'linear','source':'linear:source','input_hash':'i','sample_hash':'s',
        'preprocessing_hash':'p','prediction_space':'raw_rrp','output_type':'quantile'})
    save_prediction_artifact(path/'test.npz', artifact)
    atomic_json(path/'metrics.json', {})
    (path/'best_model.pt').write_bytes(b'checkpoint')
    status={'state':'completed','source_hash':'source','config_hash':'config','job_id':job['id'],
            'artifact_hashes':{name:sha(path/name) for name in ('test.npz','metrics.json','best_model.pt')}}
    atomic_json(path/'status.json',status)
    manifest={'source_hash':'source','config_hash':'config'}
    assert completed(tmp_path,job,manifest)
    with pytest.raises(ValueError,match='identity'):
        completed(tmp_path,dict(job,region='QLD1'),manifest)
    (path/'best_model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='corrupt'):
        completed(tmp_path,job,manifest)
