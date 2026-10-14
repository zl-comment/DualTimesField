"""Frozen matrix and derived-feature integrity checks without launching jobs."""
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from forecasting.aligned_experiment import atomic_json, sha
from forecasting.full_v3.experiment import _features, completed, validate_config, validate_matrix
from forecasting.full_v3.priors import VARIANTS


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return yaml.safe_load((ROOT / 'configs/full_v3/experiment.yaml').read_text())


def test_formal_config_keeps_predeclared_fifteen_cells():
    config = _config()
    validate_config(config)
    jobs = {(variant, region, seed) for variant in config['variants']
            for region in config['regions'] for seed in config['seeds']}
    assert len(jobs) == 15
    assert {job[2] for job in jobs} == {2026}


@pytest.mark.parametrize('key,value', [
    ('variants', list(VARIANTS[:-1])), ('variants', list(VARIANTS) + [VARIANTS[0]]),
    ('regions', ['NSW1', 'QLD1']), ('regions', ['NSW1', 'QLD1', 'NSW1']),
    ('seeds', [2026, 2027]), ('seeds', [2027]),
])
def test_changed_formal_matrix_rejected(key, value):
    config = _config()
    config[key] = value
    with pytest.raises(ValueError, match='five contrasts'):
        validate_config(config)


@pytest.mark.parametrize('key,value', [('n_estimators', 20), ('early_stopping_rounds', 5)])
def test_smoke_budget_cannot_masquerade_as_formal_config(key, value):
    config = _config()
    config['lightgbm'][key] = value
    with pytest.raises(ValueError, match='training budget'):
        validate_config(config)


@pytest.fixture
def derived(tmp_path):
    folder = tmp_path / 'features' / 'NSW1' / 'train'
    folder.mkdir(parents=True)
    arrays = dict(raw_features=np.arange(3 * 1421, dtype=np.float32).reshape(3, 1421),
                  field_features=np.arange(3 * 96, dtype=np.float32).reshape(3, 96),
                  audit_valid=np.ones((3, 49), dtype=bool))
    for index, variant in enumerate(VARIANTS):
        arrays[variant] = np.full((3, 24), float(index), dtype=np.float64)
    hashes = {}
    for name, values in arrays.items():
        path = folder / f'{name}.npy'
        np.save(path, values, allow_pickle=False)
        hashes[path.name] = sha(path)
    atomic_json(folder / 'audit.json', dict(query_labels_used=False))
    hashes['audit.json'] = sha(folder / 'audit.json')
    atomic_json(folder / 'hashes.json', hashes)
    manifest = dict(feature_hashes={'NSW1/train/hashes.json': sha(folder / 'hashes.json')})
    return tmp_path, folder, manifest, arrays


def test_feature_views_preserve_raw_and_conditional_contrast_columns(derived):
    root, _, manifest, arrays = derived
    for index, variant in enumerate(VARIANTS):
        features, prior = _features(root, manifest, 'NSW1', 'train', variant)
        np.testing.assert_array_equal(prior, arrays[variant])
        if index == 0:
            assert features.shape == (3, 0)
        elif index in (1, 2):
            np.testing.assert_array_equal(features, arrays['raw_features'])
        else:
            np.testing.assert_array_equal(features, np.concatenate((arrays['raw_features'], arrays['field_features']), axis=1))


@pytest.mark.parametrize('name', ['raw_features.npy', 'audit_valid.npy', f'{VARIANTS[-1]}.npy', 'audit.json'])
def test_unused_columns_priors_and_audit_arrays_are_integrity_protected(derived, name):
    root, folder, manifest, _ = derived
    with (folder / name).open('ab') as handle:
        handle.write(b'changed')
    # Even the level-only variant must audit unused arrays; otherwise two jobs
    # could silently claim a shared prepared bundle with different contents.
    with pytest.raises(ValueError, match='Derived feature changed'):
        _features(root, manifest, 'NSW1', 'train', VARIANTS[0])


def test_rewritten_per_split_hash_manifest_does_not_override_frozen_root(derived):
    root, folder, manifest, _ = derived
    atomic_json(folder / 'hashes.json', {})
    with pytest.raises(ValueError, match='Derived feature manifest changed'):
        _features(root, manifest, 'NSW1', 'train', VARIANTS[0])


@pytest.mark.parametrize('change', ['seed', 'duplicate', 'missing'])
def test_frozen_manifest_matrix_is_revalidated_before_execution(change):
    config = _config()
    jobs = [dict(variant=v, region=r, seed=s) for v in VARIANTS
            for s in config['seeds'] for r in config['regions']]
    validate_matrix(config, {'jobs': jobs})
    if change == 'seed':
        jobs[0]['seed'] = 2027
    elif change == 'duplicate':
        jobs[-1] = copy.deepcopy(jobs[-2])
    else:
        jobs.pop()
    with pytest.raises(ValueError, match='15-cell matrix'):
        validate_matrix(config, {'jobs': jobs})


@pytest.fixture
def completed_cell(tmp_path, monkeypatch):
    job = dict(variant=VARIANTS[0], region='NSW1', seed=2026)
    manifest = dict(source_hash='source', config_hash='configuration')
    hashes = {}
    for name in ['metrics.json', *[f'{s}_predictions.npz' for s in ('selection', 'calibration', 'test')]]:
        (tmp_path / name).write_text('verified content')
        hashes[name] = sha(tmp_path / name)
    atomic_json(tmp_path / 'output_hashes.json', hashes)
    (tmp_path / 'models').mkdir()
    atomic_json(tmp_path / 'models/manifest.json', {'fixture': 'model verified by model tests'})
    atomic_json(tmp_path / 'state.json', dict(job, **manifest, status='completed',
                 model_manifest_sha256=sha(tmp_path / 'models/manifest.json')))
    calls = []
    def verify_model(path):
        calls.append(path)
        return path, {'model_metadata': {'seed': 2026}, 'train_metadata': {'region': 'NSW1'}}
    monkeypatch.setattr('forecasting.full_v3.model._load_manifest', verify_model)
    return tmp_path, job, manifest, hashes, calls


def test_completed_skip_still_verifies_predictions_and_model(completed_cell):
    directory, job, manifest, _, calls = completed_cell
    assert completed(directory, job, manifest)
    assert calls == [directory / 'models']
    (directory / 'test_predictions.npz').write_text('changed')
    with pytest.raises(ValueError, match='Completed output changed'):
        completed(directory, job, manifest)


def test_completed_skip_rejects_foreign_state_or_missing_output(completed_cell):
    directory, job, manifest, hashes, _ = completed_cell
    foreign = dict(manifest, source_hash='different-run')
    with pytest.raises(ValueError, match='identity changed'):
        completed(directory, job, foreign)
    hashes.pop('test_predictions.npz')
    atomic_json(directory / 'output_hashes.json', hashes)
    with pytest.raises(ValueError, match='Incomplete completed output'):
        completed(directory, job, manifest)


def test_completed_skip_rejects_whole_model_directory_replacement(completed_cell):
    directory, job, manifest, _, _ = completed_cell
    atomic_json(directory / 'models/manifest.json', {'fixture': 'another internally valid model'})
    with pytest.raises(ValueError, match='model identity or hash changed'):
        completed(directory, job, manifest)
