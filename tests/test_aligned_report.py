import hashlib
import json

import numpy as np
import pytest

from forecasting.aligned_report import REGIONS, SEEDS, build_aligned_report
from forecasting.aligned_scoring import (
    QUANTILE_LEVELS, make_prediction_artifact, save_prediction_artifact, score_predictions,
)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run(root, models=('model_a',), completed=True):
    jobs = []
    for model in models:
        for region in REGIONS:
            for seed in SEEDS:
                jobs.append(dict(id=f'{model}-{region}-{seed}', model=model, region=region,
                                 seed=seed, output=f'runs/{model}/seed{seed}/{region}'))
    manifest = dict(run_id='test-run', source_hash='frozen-source', config_hash='frozen-config',
                    models=list(models), jobs=jobs)
    _json(root / 'jobs.json', manifest)
    if completed:
        for job in jobs:
            write_cell(root, manifest, job)
    return manifest


def write_cell(root, manifest, job, error=None, metadata_override=None, target_offset=0,
               actual_offset=0, point_only=False):
    actual = np.array([[0., 1.], [2., 3.]]) + actual_offset
    if error is None:
        error = {2026: -10., 2027: 10., 2028: 0.}[job['seed']]
    metadata = dict(protocol_id='full-v2', split='test', region=job['region'], seed=job['seed'],
                    source=f"{job['model']}:{manifest['source_hash']}", model=job['model'],
                    input_hash='same-inputs', sample_hash='same-samples', preprocessing_hash='same-preprocess',
                    prediction_space='raw_rrp', output_type='point_only' if point_only else 'quantile',
                    config_sha256=manifest['config_hash'], smoke=False)
    metadata.update(metadata_override or {})
    artifact = make_prediction_artifact(
        actual=actual, origin_unix=np.array([100, 200]),
        target_time_unix=np.array([[100, 110], [200, 210]]) + target_offset,
        metadata=metadata, point=actual + error if point_only else actual,
        quantiles=None if point_only else actual[..., None] + error + np.array([-2., -1., 0., 1., 2.]),
        quantile_levels=None if point_only else QUANTILE_LEVELS,
    )
    directory = root / job['output']
    save_prediction_artifact(directory / 'test.npz', artifact)
    _json(directory / 'metrics.json', score_predictions(artifact))
    status = dict(state='completed', job_id=job['id'], source_hash=manifest['source_hash'],
                  config_hash=manifest['config_hash'], seconds=2.5,
                  artifact_hashes={name: _hash(directory / name) for name in ('test.npz', 'metrics.json')})
    _json(directory / 'status.json', status)
    return artifact


def test_partial_run_only_reports_progress_not_any_scores(tmp_path):
    manifest = make_run(tmp_path, models=('model_a', 'model_b'), completed=False)
    write_cell(tmp_path, manifest, manifest['jobs'][0])
    _json(tmp_path / manifest['jobs'][1]['output'] / 'status.json', dict(state='running', elapsed_seconds=12))
    result = build_aligned_report(tmp_path)
    assert result['state'] == 'incomplete'
    assert not result['comparison_available']
    assert result['progress']['completed_cells'] == 1
    assert result['progress']['expected_cells'] == 18
    assert result['progress']['completion_fraction'] == 1 / 18
    assert 'tables' not in result
    assert 'scores' not in json.dumps(result)
    assert result['jobs'][1]['elapsed_seconds'] == 12


def test_complete_summary_scores_seeds_before_averaging_and_uses_q50(tmp_path):
    make_run(tmp_path, models=('model_a', 'model_b'))
    result = build_aligned_report(tmp_path)
    assert result['state'] == 'complete'
    assert result['comparison_available']
    assert result['tables']['P'] == {}
    row = result['tables']['Q']['model_a']
    assert row['n_cells'] == 9
    assert row['macro_regions']['mae'] == pytest.approx(20 / 3)
    assert row['macro_regions']['rmse_pooled'] == pytest.approx(20 / 3)
    region = row['regions']['NSW1']
    assert region['scores']['per_seed']['2026']['mae'] == 10.
    assert region['scores']['per_seed']['2026']['native_point_diagnostic']['mae'] == 0.
    assert region['seed_evidence']['distinct_scored_prediction_count'] == 3
    assert 'not_computed' in row['statistical_significance']
    json.dumps(result, allow_nan=False)


def test_complete_deterministic_model_does_not_claim_three_independent_runs(tmp_path):
    manifest = make_run(tmp_path, models=('lasso_arx72_cp',), completed=False)
    for job in manifest['jobs']:
        write_cell(tmp_path, manifest, job, error=3)
    result = build_aligned_report(tmp_path)
    evidence = result['tables']['Q']['lasso_arx72_cp']['regions']['NSW1']['seed_evidence']
    assert evidence['known_deterministic_model']
    assert evidence['distinct_scored_prediction_count'] == 1
    assert evidence['duplicate_seed_groups'] == [[2026, 2027, 2028]]
    assert evidence['interpretation'] == 'identical_predictions_are_not_independent_replicates'


def test_missing_ninth_manifest_cell_cannot_enter_summary(tmp_path):
    manifest = make_run(tmp_path)
    manifest['jobs'].pop()
    _json(tmp_path / 'jobs.json', manifest)
    result = build_aligned_report(tmp_path)
    assert result['progress']['expected_cells'] == 9
    assert result['progress']['completed_cells'] == 8
    assert len(result['progress']['missing_manifest_cells']) == 1
    assert 'tables' not in result


@pytest.mark.parametrize('mutation', ['labels', 'times', 'input_hash', 'preprocessing_hash'])
def test_rejects_cross_model_misalignment_even_with_rehashed_artifacts(tmp_path, mutation):
    manifest = make_run(tmp_path, models=('model_a', 'model_b'))
    job = manifest['jobs'][9]
    kwargs = {}
    if mutation == 'labels':
        kwargs['actual_offset'] = 1
    elif mutation == 'times':
        kwargs['target_offset'] = 1
    else:
        kwargs['metadata_override'] = {mutation: 'different'}
    write_cell(tmp_path, manifest, job, **kwargs)
    with pytest.raises(ValueError, match='different'):
        build_aligned_report(tmp_path)


def test_rejects_cross_seed_misalignment(tmp_path):
    manifest = make_run(tmp_path)
    write_cell(tmp_path, manifest, manifest['jobs'][1], metadata_override={'sample_hash': 'different'})
    with pytest.raises(ValueError, match='sample_hash'):
        build_aligned_report(tmp_path)


@pytest.mark.parametrize('mutation', ['prediction_bytes', 'metrics_bytes', 'wrong_source', 'wrong_config',
                                    'wrong_job', 'smoke_artifact', 'old_native_head_score'])
def test_rejects_corruption_and_provenance_mismatch(tmp_path, mutation):
    manifest = make_run(tmp_path)
    job = manifest['jobs'][0]
    directory = tmp_path / job['output']
    status = json.loads((directory / 'status.json').read_text())
    if mutation == 'prediction_bytes':
        with (directory / 'test.npz').open('ab') as stream:
            stream.write(b'corruption')
    elif mutation == 'metrics_bytes':
        (directory / 'metrics.json').write_text('{}')
    elif mutation == 'wrong_source':
        status['source_hash'] = 'other-code'
        _json(directory / 'status.json', status)
    elif mutation == 'wrong_config':
        write_cell(tmp_path, manifest, job, metadata_override={'config_sha256': 'other-config'})
    elif mutation == 'wrong_job':
        status['job_id'] = 'another-job'
        _json(directory / 'status.json', status)
    elif mutation == 'smoke_artifact':
        write_cell(tmp_path, manifest, job, metadata_override={'smoke': True})
    else:
        metrics = json.loads((directory / 'metrics.json').read_text())
        metrics['mae'] = metrics['native_point_diagnostic']['mae']
        _json(directory / 'metrics.json', metrics)
        status['artifact_hashes']['metrics.json'] = _hash(directory / 'metrics.json')
        _json(directory / 'status.json', status)
    with pytest.raises(ValueError):
        build_aligned_report(tmp_path)


def test_point_only_models_are_reported_separately(tmp_path):
    manifest = make_run(tmp_path, models=('probabilistic', 'point_only'), completed=False)
    for job in manifest['jobs']:
        write_cell(tmp_path, manifest, job, point_only=job['model'] == 'point_only')
    result = build_aligned_report(tmp_path)
    assert set(result['tables']['P']) == {'point_only'}
    assert set(result['tables']['Q']) == {'probabilistic'}
    assert 'quantile_score_proxy' not in result['tables']['P']['point_only']['macro_regions']


@pytest.mark.parametrize('mutation', ['duplicate_cell', 'same_directory', 'escape_path', 'wrong_region', 'wrong_seed'])
def test_manifest_invalid_cells_and_paths_are_rejected(tmp_path, mutation):
    manifest = make_run(tmp_path, completed=False)
    if mutation == 'duplicate_cell':
        manifest['jobs'].append(dict(manifest['jobs'][0], id='different-id'))
    elif mutation == 'same_directory':
        manifest['jobs'][1]['output'] = manifest['jobs'][0]['output']
    elif mutation == 'escape_path':
        manifest['jobs'][0]['output'] = '../outside'
    elif mutation == 'wrong_region':
        manifest['jobs'][0]['region'] = 'OTHER'
    else:
        manifest['jobs'][0]['seed'] = 999
    _json(tmp_path / 'jobs.json', manifest)
    with pytest.raises(ValueError):
        build_aligned_report(tmp_path)


def test_declared_model_cannot_disappear_from_complete_matrix(tmp_path):
    manifest = make_run(tmp_path, models=('model_a', 'model_b'))
    manifest['jobs'] = [job for job in manifest['jobs'] if job['model'] == 'model_a']
    _json(tmp_path / 'jobs.json', manifest)
    result = build_aligned_report(tmp_path)
    assert result['progress']['expected_cells'] == 18
    assert result['progress']['completed_cells'] == 9
    assert len(result['progress']['missing_manifest_cells']) == 9
    assert not result['comparison_available']
    assert 'tables' not in result


def test_frozen_source_and_config_must_match_manifest(tmp_path):
    manifest = make_run(tmp_path, completed=False)
    code = tmp_path / 'source' / 'forecasting' / 'model.py'
    code.parent.mkdir(parents=True)
    code.write_text('frozen = True\n')
    config = tmp_path / 'config.yaml'
    config.write_text('stage: full\n')
    manifest['source_files'] = {'forecasting/model.py': _hash(code)}
    manifest['source_hash'] = hashlib.sha256(json.dumps(manifest['source_files'], sort_keys=True).encode()).hexdigest()
    manifest['config_hash'] = _hash(config)
    _json(tmp_path / 'jobs.json', manifest)
    assert build_aligned_report(tmp_path)['frozen_snapshot_validation'] == 'verified'
    code.write_text('frozen = False\n')
    with pytest.raises(ValueError, match='Frozen source hash'):
        build_aligned_report(tmp_path)
    code.write_text('frozen = True\n')
    config.write_text('stage: modified\n')
    with pytest.raises(ValueError, match='Frozen configuration'):
        build_aligned_report(tmp_path)
