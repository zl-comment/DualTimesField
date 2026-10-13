"""A new source snapshot must not weaken the frozen experiment boundary."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from forecasting.aligned_experiment import (
    BASE_MODELS, EXTENSION_MODELS, REGIONS, SEEDS, SPLITS, atomic_json,
    prepare, save_cache, sha, validate_config,
)
from forecasting.aligned_extension import (
    assert_shared_protocol, audit_parent, build_extension_report, verify_extension,
    write_extension_report,
)
from forecasting.aligned_inputs import AlignedSplit, INPUT_KEYS
from forecasting.aligned_scoring import (
    QUANTILE_LEVELS, make_prediction_artifact, save_prediction_artifact, score_predictions,
)

ROOT = Path(__file__).resolve().parents[1]


def _config(models):
    config = yaml.safe_load((ROOT / 'configs/experiments/aemo_forecast_aligned_20261012.yaml').read_text())
    config['experiment']['models'] = list(models)
    return config


def _freeze(root, config, content):
    root.mkdir(parents=True)
    (root / 'source').mkdir()
    (root / 'source/model.py').write_text(content)
    (root / 'config.yaml').write_text(yaml.safe_dump(config))
    files = {'model.py': sha(root / 'source/model.py')}
    return dict(source_files=files,
                source_hash=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
                config_hash=sha(root / 'config.yaml'))


def _write_cell(root, manifest, job, input_hash='same-inputs'):
    actual = np.zeros((2, 2))
    artifact = make_prediction_artifact(
        actual=actual, origin_unix=np.array([100, 200]),
        target_time_unix=np.array([[100, 110], [200, 210]]),
        quantiles=actual[..., None] + np.arange(5), quantile_levels=QUANTILE_LEVELS,
        metadata=dict(protocol_id='aligned-baselines-20261012-static-v1', split='test',
                      region=job['region'], seed=job['seed'], model=job['model'],
                      source=f"{job['model']}:{manifest['source_hash']}", input_hash=input_hash,
                      sample_hash='same-samples', preprocessing_hash='same-preprocess',
                      prediction_space='raw_rrp', output_type='quantile',
                      config_sha256=manifest['config_hash'], smoke=False))
    folder = root / job['output']
    save_prediction_artifact(folder / 'test.npz', artifact)
    atomic_json(folder / 'metrics.json', score_predictions(artifact))
    atomic_json(folder / 'status.json', dict(
        state='completed', job_id=job['id'], source_hash=manifest['source_hash'],
        config_hash=manifest['config_hash'], artifact_hashes={
            name: sha(folder / name) for name in ('test.npz', 'metrics.json')}))


def _jobs(models):
    return [dict(id=f'{m}/{s}/{r}', model=m, region=r, seed=s,
                 output=f'runs/{m}/seed{s}/{r}', resource='gpu')
            for m in models for r in REGIONS for s in SEEDS]


@pytest.fixture
def runs(tmp_path):
    parent_root, child_root = tmp_path / 'parent', tmp_path / 'child'
    parent = dict(_freeze(parent_root, _config(BASE_MODELS), 'parent'),
                  run_id='parent', models=list(BASE_MODELS), jobs=_jobs(BASE_MODELS),
                  regions=list(REGIONS), seeds=list(SEEDS), data_hashes={'original.csv': 'data'},
                  cache_hashes={}, cache_info={})
    for region in REGIONS:
        cache = parent_root / 'data_cache' / region
        data = {split: AlignedSplit(
            {key: np.zeros((2, 2, 1), dtype='float32') for key in INPUT_KEYS},
            np.zeros((2, 2)), np.array([100, 200]), np.array([[100, 110], [200, 210]]),
            {'region': region, 'split': split}, np.zeros((2, 2))) for split in SPLITS}
        parent['cache_info'][region] = save_cache(cache, data)
        parent['cache_hashes'][region] = sha(cache / 'hashes.json')
    atomic_json(parent_root / 'jobs.json', parent)
    for job in parent['jobs']:
        _write_cell(parent_root, parent, job)
    child = dict(_freeze(child_root, _config(EXTENSION_MODELS), 'child'),
                 run_id='child', models=list(EXTENSION_MODELS), jobs=_jobs(EXTENSION_MODELS),
                 regions=list(REGIONS), seeds=list(SEEDS),
                 **{key: parent[key] for key in ('cache_hashes', 'cache_info', 'data_hashes')})
    (child_root / 'data_cache').symlink_to(parent_root / 'data_cache', target_is_directory=True)
    child['extension_from'] = dict(run_root=str(parent_root), jobs_hash=sha(parent_root / 'jobs.json'),
                                   **{key: parent[key] for key in ('run_id', 'source_hash', 'config_hash', 'cache_hashes')})
    atomic_json(child_root / 'jobs.json', child)
    for job in child['jobs']:
        _write_cell(child_root, child, job)
    return parent_root, child_root, parent, child


def test_original_matrix_stays_strict_and_extension_requires_explicit_mode():
    validate_config(_config(BASE_MODELS))
    validate_config(_config(EXTENSION_MODELS), extension=True)
    for models, extension in ((EXTENSION_MODELS, False), (BASE_MODELS, True),
                              (EXTENSION_MODELS[:1], True), (BASE_MODELS[:-1], False)):
        with pytest.raises(ValueError, match='required models'):
            validate_config(_config(models), extension=extension)


@pytest.mark.parametrize('section,key,value', [
    ('forecast_protocol', 'stride_hours', 24), ('data', 'price_transform', 'asinh'),
    ('aligned_protocol', 'point_definition', 'native_point'), ('training', 'epochs', 101),
    ('training', 'patience', 16), ('future_exogenous', 'enabled', False),
])
def test_extension_cannot_change_sampling_preprocessing_scoring_or_budget(section, key, value):
    parent, child = _config(BASE_MODELS), _config(EXTENSION_MODELS)
    child[section][key] = value
    with pytest.raises(ValueError, match='changes shared'):
        assert_shared_protocol(parent, child)


def test_completed_extension_combines_different_sources_without_relabeling(runs):
    parent_root, child_root, parent, child = runs
    assert audit_parent(parent_root)['source_hash'] == parent['source_hash']
    report = build_extension_report(child_root)
    assert report['comparison_available']
    assert report['progress']['completed_cells'] == 108
    assert len(report['tables']['Q']) == 12
    assert report['tables']['Q']['full']['provenance']['source_hash'] == parent['source_hash']
    assert report['tables']['Q']['timexer']['provenance']['source_hash'] == child['source_hash']
    assert parent['source_hash'] != child['source_hash']
    assert report['cross_run_alignment_validation'] == 'passed_all_108_artifacts'
    assert 'source_hash' not in report
    output = child_root / 'combined_report.json'
    write_extension_report(output, report)
    directory = child_root / 'combined_report_tables'
    for name, count in (('main.csv', 12), ('regional.csv', 36), ('per_seed.csv', 108)):
        with (directory / name).open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == count
        assert {row['source_hash'] for row in rows} == {parent['source_hash'], child['source_hash']}
    write_extension_report(output, {'comparison_available': False, 'state': 'incomplete'})
    assert not list(directory.glob('*.csv'))


@pytest.mark.parametrize('missing_manifest', [False, True])
def test_one_missing_cell_prevents_any_joint_ranking(runs, missing_manifest):
    _, child_root, _, child = runs
    job = child['jobs'][-1]
    if missing_manifest:
        child['jobs'].pop()
        atomic_json(child_root / 'jobs.json', child)
    else:
        (child_root / job['output'] / 'status.json').unlink()
    report = build_extension_report(child_root)
    assert not report['comparison_available']
    assert report['progress']['completed_cells'] == 107
    assert 'tables' not in report


def test_cross_source_input_mismatch_is_rejected_even_if_each_run_is_consistent(runs):
    _, child_root, _, child = runs
    for job in child['jobs']:
        _write_cell(child_root, child, job, input_hash='different-inputs')
    with pytest.raises(ValueError, match='input_hash'):
        build_extension_report(child_root)


def test_parent_cache_mutation_is_rejected(runs):
    parent_root, child_root, _, _ = runs
    np.save(parent_root / 'data_cache/NSW1/test/actual.npy', np.ones((2, 2)))
    with pytest.raises(ValueError, match='Cached data changed'):
        build_extension_report(child_root)


def test_parent_manifest_mutation_is_rejected(runs):
    parent_root, child_root, parent, child = runs
    parent['jobs'].pop()
    atomic_json(parent_root / 'jobs.json', parent)
    with pytest.raises(ValueError, match='Parent jobs manifest changed'):
        verify_extension(child_root, child)
    with pytest.raises(ValueError, match='complete 90-cell'):
        audit_parent(parent_root)


def test_prepare_reuses_verified_parent_cache_and_freezes_only_new_jobs(runs, monkeypatch):
    parent_root, _, parent, _ = runs
    project = parent_root.parent / 'project'
    for name in ('forecasting', 'src'):
        (project / name).mkdir(parents=True)
        (project / name / 'new.py').write_text('new model code')
    monkeypatch.chdir(project)
    monkeypatch.setattr('forecasting.aligned_experiment.subprocess.check_output', lambda *a, **k: 'git-head')
    path = project / 'extension.yaml'
    path.write_text(yaml.safe_dump(_config(EXTENSION_MODELS)))
    target = project / 'new-run'
    result = prepare(path, target, parent_root)
    assert len(result['jobs']) == 18
    assert {job['model'] for job in result['jobs']} == set(EXTENSION_MODELS)
    assert (target / 'data_cache').is_symlink()
    assert result['cache_hashes'] == parent['cache_hashes']
    assert result['source_hash'] != parent['source_hash']
    assert result['extension_from']['jobs_hash'] == sha(parent_root / 'jobs.json')
    verify_extension(target, result, verify_cache=True)
