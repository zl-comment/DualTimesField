"""One frozen LightGBM exposed through audited input-operation candidates."""
from __future__ import annotations

from dataclasses import asdict
import copy

import numpy as np

from ..contracts import ForecastArtifact, digest, plain
from ..input_operations import OPERATIONS, apply_operation
from .lightgbm import FrozenLightGBMAdapter


class FrozenLightGBMInputAdapter:
    def __init__(self, directory, *, admitted_operation_ids, seed=None, region=None, threads=4):
        admitted = tuple(dict.fromkeys(admitted_operation_ids))
        if not admitted or admitted[0] != 'identity' or any(key not in OPERATIONS for key in admitted):
            raise ValueError('Admitted operations must start with identity and be registered')
        self.base = FrozenLightGBMAdapter(directory, seed=seed, region=region, threads=threads)
        self.admitted_operation_ids = admitted
        self.identity = dict(
            self.base.identity,
            predictor_family='lightgbm',
            input_operation_contract='r21-training-mean-counterfactual-v1',
            admitted_operations={key: {'spec': asdict(OPERATIONS[key]),
                                       'sha256': OPERATIONS[key].identity()}
                                 for key in admitted},
        )

    def predict(self, context, operation_id='identity'):
        if operation_id not in self.admitted_operation_ids:
            raise ValueError('LightGBM input operation was not admitted: ' + str(operation_id))
        from ...aligned_inputs import AlignedSplit, flatten_inputs
        from ...aligned_baselines import _scale
        from ...lightgbm.predictor import _validate_inputs

        blocks, operation = apply_operation(context.model_inputs, operation_id)
        split = AlignedSplit(
            inputs={key: value[None] for key, value in blocks.items()},
            actual=np.empty((1, 0)), target=np.empty((1, 0)),
            origin_unix=np.array([context.origin_unix]),
            target_time_unix=np.asarray(context.snapshot.target_time_unix)[None],
            metadata=plain(context.metadata),
        )
        _validate_inputs(split, self.base.anchor.input_shapes)
        for key in ('region', 'preprocessing_hash', 'protocol_id'):
            if split.metadata.get(key) != self.base.anchor.manifest['train_metadata'].get(key):
                raise ValueError(f'Frozen LightGBM {key} mismatch')
        if _scale(split) != (self.base.anchor.target_mean, self.base.anchor.target_std):
            raise ValueError('Frozen LightGBM scaler mismatch')
        quantiles = self.base.predict_matrix(flatten_inputs(split))[0]
        input_hash = digest({
            'model': self.identity,
            'context': context.identity(),
            'operation_id': operation_id,
            'transformed_input_sha256': operation['transformed_input_sha256'],
        })
        return ForecastArtifact(
            candidate_id='lightgbm-input:' + operation_id,
            context_id=context.identity(), quantiles=quantiles,
            target_time_unix=context.snapshot.target_time_unix,
            model_id='lightgbm', input_hash=input_hash,
            provenance={'predictor_family': 'lightgbm',
                        'training_performed': False,
                        'model_identity': copy.deepcopy(self.identity),
                        'input_operation': operation,
                        'cutoff': int(context.cutoff)},
        )
