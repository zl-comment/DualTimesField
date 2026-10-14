"""Train fresh LightGBM quantiles around a raw-price conditional prior.

Every canonical input block is retained unchanged. Optional derived features
are appended in their supplied column order. The supervised quantity is
``(actual - prior) / training_price_std``; decoding adds the raw prior exactly
once and never adds the training price mean. No evaluation split enters fit.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from ..aligned_baselines import DEFAULT_QUANTILES, _RunArtifacts, _scale, _strictly_before
from ..aligned_inputs import INPUT_KEYS, _array_hash, flatten_inputs
from ..lightgbm.model import fit_horizon

QUANTILES = np.asarray(DEFAULT_QUANTILES, dtype=np.float64)
DEFAULT_PARAMETERS = dict(n_estimators=2000, early_stopping_rounds=50, selection_interval=1,
                          learning_rate=0.05, num_leaves=31, min_child_samples=50,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=4)
_TRANSFORM = dict(training='(actual_raw - prior_raw) / target_std',
                  prediction='prior_raw + target_std * residual_quantile',
                  adds_target_mean=False)


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _parameters(parameters):
    requested = dict(DEFAULT_PARAMETERS)
    requested.update(parameters or {})
    p = dict(requested)
    if p.pop('task', 'quantile') != 'quantile':
        raise ValueError('Residual model requires quantile training')
    if p.pop('selection_metric', 'pinball') != 'pinball':
        raise ValueError('Residual model requires common pinball selection')
    if p.pop('selection_scope', 'per_horizon') != 'per_horizon':
        raise ValueError('Residual model requires per_horizon selection')
    budget = dict(max_rounds=_integer(p.pop('n_estimators'), 'n_estimators'),
                  patience=_integer(p.pop('early_stopping_rounds'), 'early_stopping_rounds'),
                  interval=_integer(p.pop('selection_interval'), 'selection_interval'),
                  n_jobs=_integer(p.pop('n_jobs'), 'n_jobs'))
    budget['learning_rate'] = float(p.pop('learning_rate'))
    if not np.isfinite(budget['learning_rate']) or budget['learning_rate'] <= 0:
        raise ValueError('learning_rate must be positive and finite')
    allowed = {'max_depth', 'num_leaves', 'min_child_samples', 'subsample', 'colsample_bytree',
               'reg_alpha', 'reg_lambda', 'max_bin'}
    if set(p) - allowed:
        raise ValueError(f'Unsupported LightGBM parameters: {sorted(set(p) - allowed)}')
    return requested, budget, p


def _inputs(split, extra_features, prior):
    """Validate only prediction-time data; deliberately never access labels."""
    if set(split.inputs) != set(INPUT_KEYS) or not len(split):
        raise ValueError('Residual model requires nonempty canonical seven-block inputs')
    for key in ('protocol_id', 'region', 'preprocessing_hash', 'input_hash', 'sample_hash'):
        if not isinstance(split.metadata.get(key), str) or not split.metadata[key]:
            raise ValueError(f'Missing aligned metadata: {key}')
    if split.metadata.get('input_block_order', list(INPUT_KEYS)) != list(INPUT_KEYS):
        raise ValueError('Canonical input block order mismatch')
    if split.metadata.get('price_transform', 'identity') != 'identity':
        raise ValueError('Residual model requires affine raw-price supervision')
    _scale(split)
    for array in split.inputs.values():
        if np.asarray(array).ndim < 2 or len(array) != len(split) or not np.isfinite(array).all():
            raise ValueError('Canonical inputs must be finite and row-aligned')
    origins, targets = np.asarray(split.origin_unix), np.asarray(split.target_time_unix)
    if (origins.ndim != 1 or len(np.unique(origins)) != len(origins) or not np.isfinite(origins).all()
            or targets.ndim != 2 or targets.shape[0] != len(split) or targets.shape[1] < 1
            or not np.isfinite(targets).all()):
        raise ValueError('Invalid aligned origin/target timestamp schema')
    extra = np.asarray(extra_features)
    base_prior = np.asarray(prior)
    if extra.ndim != 2 or extra.shape[0] != len(split) or not np.isfinite(extra).all():
        raise ValueError('Extra features must be finite [origin, feature], including zero-width')
    if base_prior.shape != targets.shape or not np.isfinite(base_prior).all():
        raise ValueError('Raw prior must be finite [origin, horizon]')
    return extra, base_prior


def residual_input_provenance(split, extra_features, prior):
    """Expose the augmented input identity without reusing the base input hash.

    The prior changes prediction even when there are no appended tree columns;
    it is therefore included in the effective input identity in every variant.
    """
    extra, base_prior = _inputs(split, extra_features, prior)
    evidence = dict(base_input_hash=split.metadata['input_hash'],
                    base_sample_hash=split.metadata['sample_hash'],
                    base_preprocessing_hash=split.metadata['preprocessing_hash'],
                    canonical_array_hash=_array_hash({key: split.inputs[key] for key in INPUT_KEYS}),
                    extra_features_hash=_array_hash({'extra_features': extra}),
                    raw_prior_hash=_array_hash({'prior_raw': base_prior}),
                    origin_target_hash=_array_hash({'origin': split.origin_unix, 'target_time': split.target_time_unix}),
                    extra_feature_count=extra.shape[1],
                    feature_order=[*INPUT_KEYS, 'extra_features_in_supplied_column_order'],
                    input_scope='canonical_seven_blocks_plus_derived_features_and_raw_prior')
    evidence['effective_input_hash'] = _hash_json(evidence)
    return evidence


def _shapes(split):
    return {key: list(split.inputs[key].shape[1:]) for key in INPUT_KEYS}


def _design(split, extra):
    canonical = flatten_inputs(split)
    return np.concatenate((canonical, extra), axis=1) if extra.shape[1] else canonical


def _residual(split, prior):
    actual = np.asarray(split.actual, dtype=np.float64)
    if actual.shape != prior.shape or not np.isfinite(actual).all():
        raise ValueError('Fit/selection raw labels must be finite and match their prior')
    residual = (actual - np.asarray(prior, dtype=np.float64)) / _scale(split)[1]
    if not np.isfinite(residual).all():
        raise FloatingPointError('Non-finite scaled residual labels')
    return residual


def fit_residual_model(train, selection, features_train, features_selection, prior_train, prior_selection,
                       *, model_directory, progress_path=None, seed=2026, parameters=None):
    """Fit fresh quantile trees and select prefixes using this selection only.

    Partial fit directories are refused, rather than silently mixing trees from
    different input priors. A completed directory can be loaded by predict.
    """
    started = time.monotonic()
    seed = _integer(seed, 'seed', minimum=0)
    extras, priors = zip(*[_inputs(s, x, p) for s, x, p in
                          ((train, features_train, prior_train), (selection, features_selection, prior_selection))])
    if (_shapes(train) != _shapes(selection) or _scale(train) != _scale(selection)
            or extras[0].shape[1] != extras[1].shape[1] or priors[0].shape[1] != priors[1].shape[1]):
        raise ValueError('Fit and selection schema/scaler mismatch')
    for key in ('region', 'protocol_id', 'preprocessing_hash'):
        if train.metadata[key] != selection.metadata[key]:
            raise ValueError(f'Fit and selection {key} mismatch')
    _strictly_before(train, selection, 'Residual model selection')
    requested, budget, tree_params = _parameters(parameters)
    directory = Path(model_directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Residual model directory must be empty; completed or partial fits are immutable')
    artifacts = _RunArtifacts(progress_path, directory)
    x_train, x_select = _design(train, extras[0]), _design(selection, extras[1])
    y_train, y_select = _residual(train, priors[0]), _residual(selection, priors[1])
    train_provenance = residual_input_provenance(train, extras[0], priors[0])
    selection_provenance = residual_input_provenance(selection, extras[1], priors[1])
    selected, fitted, traces = [], [], []
    artifacts.progress('prepare_inputs', model='full_v3_lightgbm_residual', feature_count=x_train.shape[1])
    for horizon in range(y_train.shape[1]):
        # fit_horizon needs a prediction matrix; only the selection matrix is
        # passed here. The returned forecasts are discarded, never an evaluation.
        _, count, actual_counts, trace = fit_horizon(
            horizon, x_train, y_train, x_select, y_select, x_select,
            q=QUANTILES, task='quantile', metric='pinball', seed=seed,
            parameters=tree_params, artifacts=artifacts, **budget)
        selected.append(count)
        fitted.append(actual_counts)
        traces.append(trace)
        artifacts.progress('horizon_complete', horizon=horizon, horizon_count=y_train.shape[1],
                           selected_rounds=count, fitted_rounds=actual_counts)
    mean, std = _scale(train)
    metadata = dict(model='full_v3_lightgbm_residual', task='quantile', seed=seed,
                    package_version=version('lightgbm'), requested_params=requested,
                    feature_count=x_train.shape[1], canonical_feature_count=x_train.shape[1] - extras[0].shape[1],
                    extra_feature_count=extras[0].shape[1], input_block_order=list(INPUT_KEYS),
                    input_shapes=_shapes(train), feature_order=train_provenance['feature_order'],
                    input_scope=train_provenance['input_scope'],
                    train_input_provenance=train_provenance, selection_input_provenance=selection_provenance,
                    target_mean=mean, target_std=std, residual_transform=_TRANSFORM,
                    training_residual_hash=_array_hash({'residual_target': y_train}),
                    selection_residual_hash=_array_hash({'residual_target': y_select}),
                    quantile_levels=QUANTILES.tolist(), horizon=y_train.shape[1],
                    selected_rounds=selected, actual_fitted_rounds=fitted, selection_trace=traces,
                    selection_metric='pinball', selection_score_space='raw_price_residual_divided_by_train_std',
                    selection_scope='per_horizon', shared_rounds_across_quantiles=True,
                    raw_pinball_equals_selection_pinball_times_target_std=True,
                    prediction_space='raw_rrp', point_definition='q50_after_common_sort',
                    quantile_crossing_correction='common_sort', training_seconds=time.monotonic() - started,
                    evaluation_data_used=False, pretrained_or_frozen_tree_used=False)
    manifest = dict(format_version=1, model_metadata=metadata, files=artifacts.files,
                    train_metadata=dict(train.metadata), selection_metadata=dict(selection.metadata))
    manifest_path = directory / 'manifest.json'
    artifacts.write_json(manifest_path, manifest)
    digest = _sha(manifest_path)
    checksum = directory / 'manifest.sha256'
    temporary = checksum.with_suffix('.sha256.tmp')
    temporary.write_text(digest + '\n')
    temporary.replace(checksum)
    artifacts.progress('complete', selected_rounds=selected, files_saved=len(artifacts.files))
    return dict(metadata, model_manifest=str(manifest_path), model_manifest_sha256=digest)


def _load_manifest(directory):
    directory = Path(directory).resolve()
    path = directory / 'manifest.json'
    if _sha(path) != (directory / 'manifest.sha256').read_text().strip():
        raise ValueError('Residual checkpoint manifest checksum mismatch')
    manifest = json.loads(path.read_text())
    metadata = manifest['model_metadata']
    if (manifest.get('format_version') != 1 or metadata.get('model') != 'full_v3_lightgbm_residual'
            or metadata.get('task') != 'quantile' or metadata.get('residual_transform') != _TRANSFORM):
        raise ValueError('Unsupported residual checkpoint transform/format')
    if metadata.get('package_version') != version('lightgbm'):
        raise ValueError('Residual checkpoint LightGBM package version mismatch')
    if metadata.get('input_block_order') != list(INPUT_KEYS) or not np.array_equal(metadata.get('quantile_levels'), QUANTILES):
        raise ValueError('Residual checkpoint input order/quantile mismatch')
    horizon = _integer(metadata['horizon'], 'checkpoint horizon')
    _, budget, _ = _parameters(metadata['requested_params'])
    if len(metadata['selected_rounds']) != horizon:
        raise ValueError('Residual checkpoint selected-round grid mismatch')
    for count in metadata['selected_rounds']:
        if _integer(count, 'checkpoint selected_rounds') > budget['max_rounds']:
            raise ValueError('Residual selected rounds exceed the declared training budget')
    expected = sum(int(np.prod(shape)) for shape in metadata['input_shapes'].values())
    if (set(metadata['input_shapes']) != set(INPUT_KEYS) or expected != metadata['canonical_feature_count']
            or metadata['feature_count'] != expected + _integer(metadata['extra_feature_count'], 'extra_feature_count', 0)):
        raise ValueError('Residual checkpoint feature schema/count mismatch')
    seen = set()
    for entry in manifest['files']:
        h = _integer(entry['horizon'], 'model horizon', 0)
        indices = np.flatnonzero(QUANTILES == entry['quantile'])
        if len(indices) != 1 or h >= horizon or entry.get('format') != 'lightgbm':
            raise ValueError('Invalid residual model grid entry')
        cell = (h, int(indices[0]))
        if cell in seen or int(entry['selected_rounds']) != metadata['selected_rounds'][h]:
            raise ValueError('Duplicate or inconsistent residual model grid entry')
        seen.add(cell)
        model_path = (directory / entry['path']).resolve()
        if model_path.parent != directory or _sha(model_path) != entry['sha256']:
            raise ValueError('Residual model path/checksum mismatch')
    if len(seen) != horizon * len(QUANTILES):
        raise ValueError('Incomplete residual model grid')
    return directory, manifest


def predict_residual_model(model_directory, split, extra_features, prior, *, threads=4):
    """Reload verified tree files and decode raw forecasts without reading labels."""
    import lightgbm as lgb
    threads = _integer(threads, 'threads')
    extra, base_prior = _inputs(split, extra_features, prior)
    directory, manifest = _load_manifest(model_directory)
    metadata, train_metadata = manifest['model_metadata'], manifest['train_metadata']
    for key in ('region', 'protocol_id', 'preprocessing_hash'):
        if split.metadata[key] != train_metadata[key]:
            raise ValueError(f'Residual prediction {key} mismatch')
    if (_scale(split) != (metadata['target_mean'], metadata['target_std'])
            or _scale(split) != (train_metadata['target_mean'], train_metadata['target_std'])):
        raise ValueError('Residual prediction target scaler mismatch')
    if (_shapes(split) != metadata['input_shapes'] or extra.shape[1] != metadata['extra_feature_count']
            or base_prior.shape[1] != metadata['horizon']):
        raise ValueError('Residual prediction feature/prior schema mismatch')
    x = _design(split, extra)
    residual = np.empty((len(split), metadata['horizon'], len(QUANTILES)), dtype=np.float64)
    aliases = dict(min_child_samples='min_data_in_leaf', subsample='bagging_fraction',
                   colsample_bytree='feature_fraction', reg_alpha='lambda_l1', reg_lambda='lambda_l2')
    requested, budget, params = _parameters(metadata['requested_params'])
    params['learning_rate'] = budget['learning_rate']
    for entry in manifest['files']:
        booster = lgb.Booster(model_file=str(directory / entry['path']))
        h, q = entry['horizon'], int(np.flatnonzero(QUANTILES == entry['quantile'])[0])
        if (booster.num_feature() != metadata['feature_count'] or booster.params.get('objective') != 'quantile'
                or float(booster.params['alpha']) != QUANTILES[q] or int(booster.params['seed']) != metadata['seed']
                or booster.current_iteration() != entry['fitted_rounds']):
            raise ValueError('Residual booster objective/seed/feature/round mismatch')
        for key, value in params.items():
            if booster.params.get(aliases.get(key, key)) != value:
                raise ValueError(f'Residual booster parameter mismatch: {key}')
        residual[:, h, q] = booster.predict(x, num_iteration=entry['selected_rounds'], num_threads=threads)
    output = np.asarray(base_prior, dtype=np.float64)[..., None] + metadata['target_std'] * residual
    if not np.isfinite(output).all():
        raise FloatingPointError('Non-finite residual raw-price prediction')
    return np.sort(output, axis=-1)
