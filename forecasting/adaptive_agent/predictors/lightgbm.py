"""Reuse the exact saved baseline schema and frozen in-memory boosters."""
from __future__ import annotations

import numpy as np


class FrozenLightGBMAdapter:
    """Validate upstream manifests, then load each verified booster once.

    There is no fitting method. Loaded boosters are private and never mutated;
    later disk edits cannot alter this process's frozen predictions.
    """
    def __init__(self, directory, *, seed=None, region=None, threads=4,
                 train_end_unix=None, selection_end_unix=None):
        import lightgbm as lgb
        from ...lightgbm.predictor import FrozenLightGBMPredictor, _sha
        self.anchor = FrozenLightGBMPredictor(directory, seed=seed, region=region, threads=threads)
        self._boosters = {}
        for index, entry in sorted(self.anchor.entries.items()):
            path = self.anchor.directory / entry['path']
            # Hash the exact bytes that are parsed, avoiding hash/load races.
            import hashlib
            source = path.read_bytes()
            if hashlib.sha256(source).hexdigest() != entry['sha256']:
                raise ValueError('Frozen checkpoint changed during loading')
            self._boosters[index] = lgb.Booster(model_str=source.decode('utf-8'))
        if _sha(self.anchor.directory / 'manifest.json') != self.anchor.fingerprint:
            raise ValueError('Frozen manifest changed during loading')
        self.identity = dict(self.anchor.provenance,
                             model_sha256=self.anchor.fingerprint,
                             train_end_unix=train_end_unix,
                             selection_end_unix=selection_end_unix,
                             frozen_in_memory=True)

    def predict_matrix(self, matrix):
        """Predict a validated canonical flattened matrix with frozen boosters."""
        matrix = np.asarray(matrix)
        expected = sum(int(np.prod(shape)) for shape in self.anchor.input_shapes.values())
        if matrix.ndim != 2 or matrix.shape[1] != expected or not np.isfinite(matrix).all():
            raise ValueError('Frozen LightGBM matrix violates its fixed finite schema')
        result = np.empty((len(matrix), 24, 5), dtype=np.float64)
        for (h, q), booster in self._boosters.items():
            result[:, h, q] = booster.predict(
                matrix, num_iteration=self.anchor.selected_rounds[h],
                num_threads=self.anchor.threads)
        result = result * self.anchor.target_std + self.anchor.target_mean
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite frozen LightGBM output')
        return np.sort(result, axis=-1)

    def predict(self, context, profile_id='full', window_hours=72):
        if profile_id != 'full' or window_hours != 72:
            raise ValueError('Frozen LightGBM requires original full 72-hour feature schema')
        from ...aligned_inputs import AlignedSplit, flatten_inputs
        from ...aligned_baselines import _scale
        from ...lightgbm.predictor import _validate_inputs
        from ..contracts import plain
        # Inference-only split: no actual values are read by any predictor.
        split = AlignedSplit(
            inputs={key: np.array(value, copy=True)[None] for key, value in context.model_inputs.items()},
            actual=np.empty((1, 0)), target=np.empty((1, 0)),
            origin_unix=np.array([context.origin_unix]),
            target_time_unix=np.asarray(context.snapshot.target_time_unix)[None],
            metadata=plain(context.metadata))
        _validate_inputs(split, self.anchor.input_shapes)
        for key in ('region', 'preprocessing_hash', 'protocol_id'):
            if split.metadata.get(key) != self.anchor.manifest['train_metadata'].get(key):
                raise ValueError(f'Frozen LightGBM {key} mismatch')
        if _scale(split) != (self.anchor.target_mean, self.anchor.target_std):
            raise ValueError('Frozen LightGBM scaler mismatch')
        if split.target_time_unix.shape != (1, self.anchor.horizon):
            raise ValueError('Frozen LightGBM horizon mismatch')
        matrix = flatten_inputs(split)
        return self.predict_matrix(matrix)[0]
