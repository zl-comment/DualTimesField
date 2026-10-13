"""Validated reader for saved aligned LightGBM baselines; no training or V2 dependency."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from ..aligned_baselines import DEFAULT_QUANTILES, _scale
from ..aligned_inputs import INPUT_KEYS, AlignedSplit, flatten_inputs

_LEVELS = np.asarray(DEFAULT_QUANTILES, dtype=np.float64)
_LEGACY_SHAPES = dict(history_values=[72, 2], history_calendar=[72, 10],
                      future_calendar=[24, 10], origin_context=[1],
                      future_exogenous=[24, 11], ctf_exogenous=[24, 1],
                      quantile_exogenous=[24, 3])


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _shapes(split):
    return {key: list(split.inputs[key].shape[1:]) for key in INPUT_KEYS}


def _validate_inputs(split, shapes=None):
    if not len(split) or set(split.inputs) != set(INPUT_KEYS):
        raise ValueError('Anchor requires nonempty canonical seven-block inputs')
    if shapes is not None and _shapes(split) != shapes:
        raise ValueError('Anchor input shape/schema mismatch')
    if split.metadata.get('input_block_order', list(INPUT_KEYS)) != list(INPUT_KEYS):
        raise ValueError('Anchor input block order mismatch')
    for values in split.inputs.values():
        if len(values) != len(split) or not np.isfinite(values).all():
            raise ValueError('Anchor inputs must be finite and row-aligned')
    _scale(split)


class FrozenLightGBMPredictor:
    """Load a trusted aligned model directory, verifying every file and schema.

    ``directory`` may be the models directory or its parent run directory.
    Models are loaded one at a time to bound RAM across concurrent folds.
    """
    def __init__(self, directory, *, seed=None, region=None, threads=4):
        import lightgbm as lgb
        self.directory = Path(directory).resolve()
        if not (self.directory / 'manifest.json').exists():
            self.directory /= 'models'
        manifest_path = self.directory / 'manifest.json'
        self.manifest = manifest = _read(manifest_path)
        metadata = manifest['model_metadata']
        self.threads = int(threads)
        if self.threads < 1:
            raise ValueError('threads must be positive')
        if metadata.get('model') != 'lightgbm' or metadata.get('task') != 'quantile':
            raise ValueError('Expected a LightGBM quantile anchor')
        if not np.array_equal(metadata.get('quantile_levels'), _LEVELS):
            raise ValueError('Anchor quantile levels mismatch')
        if manifest.get('input_block_order') != list(INPUT_KEYS):
            raise ValueError('Anchor canonical flatten order mismatch')
        if metadata.get('package_version') != version('lightgbm'):
            raise ValueError('Anchor LightGBM package version mismatch')
        train_meta = manifest['train_metadata']
        self.seed, self.region = int(metadata['seed']), train_meta['region']
        if seed is not None and int(seed) != self.seed:
            raise ValueError('Anchor seed mismatch')
        if region is not None and region != self.region:
            raise ValueError('Anchor region mismatch')
        self.target_mean, self.target_std = float(manifest['target_mean']), float(manifest['target_std'])
        if (not np.isfinite([self.target_mean, self.target_std]).all() or self.target_std <= 0
                or (self.target_mean, self.target_std) != (train_meta['target_mean'], train_meta['target_std'])):
            raise ValueError('Anchor target scaler mismatch')
        if train_meta.get('price_transform', 'identity') != 'identity':
            raise ValueError('Anchor requires affine raw-price target space')
        self.input_shapes = manifest.get('input_shapes', _LEGACY_SHAPES)
        self.selected_rounds = [int(v) for v in metadata['selected_rounds']]
        self.horizon = len(self.selected_rounds)
        if self.horizon < 1 or min(self.selected_rounds) < 1:
            raise ValueError('Invalid anchor selected iteration counts')
        expected_features = sum(int(np.prod(v)) for v in self.input_shapes.values())
        if expected_features != metadata['feature_count']:
            raise ValueError('Anchor feature count/schema mismatch')
        self.entries = {}
        aliases = dict(min_child_samples='min_data_in_leaf', subsample='bagging_fraction',
                       colsample_bytree='feature_fraction', reg_alpha='lambda_l1', reg_lambda='lambda_l2')
        inspected = {'learning_rate', 'num_leaves', 'max_depth', 'min_child_samples',
                     'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'max_bin'}
        for entry in manifest['files']:
            h, q = int(entry['horizon']), float(entry['quantile'])
            matches = np.flatnonzero(_LEVELS == q)
            if len(matches) != 1 or not 0 <= h < self.horizon:
                raise ValueError('Invalid anchor horizon/quantile entry')
            index = (h, int(matches[0]))
            if index in self.entries or entry.get('format') != 'lightgbm':
                raise ValueError('Duplicate or invalid anchor model entry')
            path = (self.directory / entry['path']).resolve()
            if path.parent != self.directory or _sha(path) != entry['sha256']:
                raise ValueError('Anchor model path/hash mismatch')
            if int(entry['selected_rounds']) != self.selected_rounds[h]:
                raise ValueError('Anchor selected iteration metadata mismatch')
            booster = lgb.Booster(model_file=str(path))
            if (booster.num_feature() != expected_features or booster.params.get('objective') != 'quantile'
                    or float(booster.params['alpha']) != q or int(booster.params['seed']) != self.seed):
                raise ValueError('Anchor model feature/objective/seed mismatch')
            # LightGBM can terminate before the requested round if no split is possible.
            if booster.current_iteration() != int(entry['fitted_rounds']):
                raise ValueError('Anchor fitted iteration metadata mismatch')
            for key, value in metadata.get('requested_params', {}).items():
                if key in inspected and booster.params.get(aliases.get(key, key)) != value:
                    raise ValueError(f'Anchor model parameter mismatch: {key}')
            self.entries[index] = entry
        if len(self.entries) != self.horizon * len(_LEVELS):
            raise ValueError('Incomplete anchor model grid')
        self.fingerprint = _sha(manifest_path)
        self.provenance = dict(manifest_sha256=self.fingerprint, model_directory=str(self.directory),
                               seed=self.seed, region=self.region, package_version=metadata['package_version'],
                               files_verified=len(self.entries), selected_rounds=self.selected_rounds,
                               target_mean=self.target_mean, target_std=self.target_std,
                               preprocessing_hash=train_meta['preprocessing_hash'])

    def predict(self, split: AlignedSplit) -> np.ndarray:
        import lightgbm as lgb
        _validate_inputs(split, self.input_shapes)
        train_meta = self.manifest['train_metadata']
        for key in ('region', 'preprocessing_hash', 'protocol_id'):
            if split.metadata.get(key) != train_meta.get(key):
                raise ValueError(f'Anchor prediction {key} mismatch')
        if _scale(split) != (self.target_mean, self.target_std):
            raise ValueError('Anchor prediction target scaler mismatch')
        if np.asarray(split.target_time_unix).shape != (len(split), self.horizon):
            raise ValueError('Anchor prediction horizon mismatch')
        x = flatten_inputs(split)
        output = np.empty((len(split), self.horizon, len(_LEVELS)), dtype=np.float64)
        for (h, q), entry in sorted(self.entries.items()):
            path = self.directory / entry['path']
            if _sha(path) != entry['sha256']:
                raise ValueError('Frozen anchor changed after loading')
            booster = lgb.Booster(model_file=str(path))
            output[:, h, q] = booster.predict(x, num_iteration=self.selected_rounds[h], num_threads=self.threads)
        output = output * self.target_std + self.target_mean
        if not np.isfinite(output).all():
            raise FloatingPointError('Non-finite anchor prediction')
        return np.sort(output, axis=-1)

    predict_raw = predict
