"""Real small LightGBM fits validate residual units and checkpoint boundaries."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from forecasting.aligned_baselines import _loss
from forecasting.aligned_inputs import AlignedSplit, INPUT_KEYS
from forecasting.full_v3.model import (
    DEFAULT_PARAMETERS, QUANTILES, fit_residual_model, predict_residual_model, residual_input_provenance,
)

PARAMETERS = dict(n_estimators=7, early_stopping_rounds=3, learning_rate=0.15,
                  num_leaves=5, min_child_samples=3, n_jobs=1)


def _split(start, n=64):
    rng = np.random.default_rng(start)
    shapes = [(4, 2), (4, 2), (2, 2), (1,), (2, 2), (2, 1), (2, 3)]
    inputs = {key: rng.normal(size=(n, *shape)).astype(np.float32) for key, shape in zip(INPUT_KEYS, shapes)}
    origin = (start + np.arange(n)) * 3600
    times = origin[:, None] + np.arange(2) * 3600
    target = inputs['future_exogenous'][..., 0] * 2 + inputs['history_values'][:, -1, :1]
    meta = dict(region='NSW1', protocol_id='canonical-test-v1', preprocessing_hash='train-only',
                input_hash=f'base-input-{start}', sample_hash=f'sample-{start}',
                target_mean=45., target_std=12., price_transform='identity', input_block_order=list(INPUT_KEYS))
    split = AlignedSplit(inputs, target.astype(np.float64) * 12 + 45, origin, times, meta, target)
    features = rng.normal(size=(n, 3)).astype(np.float32)
    prior = inputs['future_exogenous'][..., 0].astype(np.float64) * 12 + 45
    return split, features, prior


@pytest.fixture
def fitted(tmp_path, monkeypatch):
    import forecasting.full_v3.model as module
    train, xt, pt = _split(0)
    selection, xs, ps = _split(100, 24)
    evaluation, xe, pe = _split(200, 11)
    selected_predictions = []
    native = module.fit_horizon
    def spy(*args, **kwargs):
        result = native(*args, **kwargs)
        selected_predictions.append(result[0])
        return result
    monkeypatch.setattr(module, 'fit_horizon', spy)
    directory = tmp_path / 'models'
    metadata = fit_residual_model(train, selection, xt, xs, pt, ps, model_directory=directory,
                                  progress_path=tmp_path / 'progress.json', parameters=PARAMETERS)
    expected = np.sort(ps[..., None] + 12 * np.stack(selected_predictions, axis=1), axis=-1)
    return directory, train, selection, evaluation, xt, xs, xe, pt, ps, pe, metadata, expected


def test_reloaded_prediction_is_bit_exact_and_input_identity_is_distinct(fitted):
    directory, train, selection, _, _, xs, _, _, ps, _, metadata, expected = fitted
    np.testing.assert_array_equal(predict_residual_model(directory, selection, xs, ps, threads=1), expected)
    assert expected.dtype == np.float64 and expected.shape == (24, 2, 5)
    assert np.all(np.diff(expected, axis=-1) >= 0)
    assert metadata['feature_count'] == sum(int(np.prod(v.shape[1:])) for v in train.inputs.values()) + 3
    assert len(list(directory.glob('*.txt'))) == 10
    provenance = metadata['train_input_provenance']
    assert provenance['base_input_hash'] == train.metadata['input_hash']
    assert provenance['effective_input_hash'] != provenance['base_input_hash']
    assert metadata['evaluation_data_used'] is False
    assert metadata['pretrained_or_frozen_tree_used'] is False
    assert metadata['residual_transform']['adds_target_mean'] is False
    assert metadata['training_seconds'] > 0


def test_prediction_does_not_access_any_labels(fitted):
    directory, _, _, evaluation, _, _, xe, _, _, pe, *_ = fitted
    expected = predict_residual_model(directory, evaluation, xe, pe, threads=1)
    class NoLabels:
        inputs, origin_unix, target_time_unix, metadata = (
            evaluation.inputs, evaluation.origin_unix, evaluation.target_time_unix, evaluation.metadata)
        def __len__(self):
            return len(self.origin_unix)
        @property
        def actual(self):
            raise AssertionError('Prediction read actual labels')
        @property
        def target(self):
            raise AssertionError('Prediction read standardized labels')
    np.testing.assert_array_equal(predict_residual_model(directory, NoLabels(), xe, pe, threads=1), expected)
    poisoned = copy.deepcopy(evaluation)
    poisoned.actual[:] = np.nan
    poisoned.target[:] = np.inf
    np.testing.assert_array_equal(predict_residual_model(directory, poisoned, xe, pe, threads=1), expected)


def test_constant_raw_residual_decodes_once_without_training_mean(tmp_path):
    train, _, pt = _split(0)
    selection, _, ps = _split(100, 24)
    evaluate, _, pe = _split(200, 11)
    # With u=2 at every sample, every quantile must predict prior + 24,
    # independently of the nonzero (45) standardizer mean.
    train.actual = pt + 24.
    selection.actual = ps + 24.
    xt, xs, xe = (np.empty((len(s), 0), dtype=np.float32) for s in (train, selection, evaluate))
    directory = tmp_path / 'constant'
    fit_residual_model(train, selection, xt, xs, pt, ps, model_directory=directory, parameters=PARAMETERS)
    actual = predict_residual_model(directory, evaluate, xe, pe, threads=1)
    np.testing.assert_allclose(actual, np.repeat((pe + 24.)[..., None], 5, axis=-1), rtol=0, atol=1e-12)
    shifted = predict_residual_model(directory, evaluate, xe, pe + 100., threads=1)
    np.testing.assert_allclose(shifted, actual + 100., rtol=0, atol=1e-12)
    assert residual_input_provenance(evaluate, xe, pe)['effective_input_hash'] != residual_input_provenance(evaluate, xe, pe + 100)['effective_input_hash']


def test_joint_pinball_selection_is_invariant_to_raw_prior_translation(fitted):
    directory, _, selection, _, _, xs, _, _, ps, _, metadata, _ = fitted
    raw = predict_residual_model(directory, selection, xs, ps, threads=1)
    u = (selection.actual - ps) / 12.
    for h, (count, trace) in enumerate(zip(metadata['selected_rounds'], metadata['selection_trace'])):
        raw_loss = _loss(raw[:, h:h+1], selection.actual[:, h:h+1], QUANTILES, 'pinball')
        residual_loss = _loss((raw[:, h:h+1] - ps[:, h:h+1, None]) / 12., u[:, h:h+1], QUANTILES, 'pinball')
        np.testing.assert_allclose(raw_loss, residual_loss * 12., rtol=1e-12, atol=1e-12)
        best = next(row['score'] for row in trace if row['rounds'] == count)
        # LightGBM internally stores labels as float32; cached fitting scores
        # and reloaded raw reconstruction agree to that numerical precision.
        np.testing.assert_allclose(residual_loss, best, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('damage', ['tree', 'manifest', 'grid', 'parameter', 'path'])
def test_tampering_and_incomplete_checkpoint_are_rejected(fitted, damage):
    directory, _, selection, _, _, xs, _, _, ps, *_ = fitted
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if damage == 'tree':
        path = directory / manifest['files'][0]['path']
        path.write_text(path.read_text() + '\nchanged\n')
    else:
        if damage == 'manifest':
            manifest['model_metadata']['target_std'] = 99.
        elif damage == 'grid':
            manifest['files'].pop()
        elif damage == 'parameter':
            manifest['model_metadata']['requested_params']['num_leaves'] = 17
        else:
            manifest['files'][0]['path'] = '../escape.txt'
        manifest_path.write_text(json.dumps(manifest))
        # Refreshing a checksum alone cannot bypass structural/booster checks.
        if damage != 'manifest':
            (directory / 'manifest.sha256').write_text(hashlib.sha256(manifest_path.read_bytes()).hexdigest() + '\n')
    with pytest.raises(ValueError, match='checksum|Incomplete|parameter|path'):
        predict_residual_model(directory, selection, xs, ps, threads=1)


def test_fit_rejects_overlap_schema_mismatch_and_objective_override(tmp_path):
    train, xt, pt = _split(0)
    overlap, xs, ps = _split(63, 24)
    with pytest.raises(ValueError, match='realized'):
        fit_residual_model(train, overlap, xt, xs, pt, ps, model_directory=tmp_path / 'overlap', parameters=PARAMETERS)
    selection, xs, ps = _split(100, 24)
    with pytest.raises(ValueError, match='schema/scaler'):
        fit_residual_model(train, selection, xt, xs[:, :2], pt, ps, model_directory=tmp_path / 'schema', parameters=PARAMETERS)
    with pytest.raises(ValueError, match='Unsupported'):
        fit_residual_model(train, selection, xt, xs, pt, ps, model_directory=tmp_path / 'objective',
                           parameters=dict(PARAMETERS, objective='regression'))


def test_immutable_checkpoint_and_prediction_contract(fitted):
    directory, train, selection, evaluation, xt, xs, xe, pt, ps, pe, *_ = fitted
    with pytest.raises(FileExistsError, match='immutable'):
        fit_residual_model(train, selection, xt, xs, pt, ps, model_directory=directory, parameters=PARAMETERS)
    for key, value in [('region', 'QLD1'), ('preprocessing_hash', 'other'), ('target_std', 9.)]:
        bad = copy.deepcopy(evaluation)
        bad.metadata[key] = value
        with pytest.raises(ValueError, match='mismatch'):
            predict_residual_model(directory, bad, xe, pe, threads=1)
    with pytest.raises(ValueError, match='schema mismatch'):
        predict_residual_model(directory, evaluation, xe[:, :2], pe, threads=1)
    with pytest.raises(ValueError, match='finite'):
        predict_residual_model(directory, evaluation, xe, np.full_like(pe, np.nan), threads=1)


def test_defaults_preserve_baseline_tree_budget():
    assert DEFAULT_PARAMETERS == dict(n_estimators=2000, early_stopping_rounds=50, selection_interval=1,
        learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, colsample_bytree=0.8, n_jobs=4)
