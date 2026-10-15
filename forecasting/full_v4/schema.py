"""Stable, physical-unit interfaces. Labels never enter the current encoder."""
import numpy as np
from forecasting.aligned_inputs import INPUT_KEYS, AlignedSplit, _array_hash

SHAPES = ((72, 2), (72, 10), (24, 10), (1,), (24, 11), (24, 1), (24, 3))
OUTPUT_KEYS = ('P', 'P0', 'C0', 'D0', 'delta_C', 'delta_D', 'gC', 'gD')
VARIANTS = ('v4_lgb_oof_rows', 'v4_current_residual', 'v4_full',
            'v4_forced_history', 'v4_direct_tree')


def validate_inputs(inputs):
    if set(inputs) != set(INPUT_KEYS):
        raise ValueError('Expected exactly seven aligned blocks')
    for key, shape in zip(INPUT_KEYS, SHAPES):
        if tuple(inputs[key].shape[1:]) != shape:
            raise ValueError(f'Unexpected {key} shape: {inputs[key].shape}')


def subset(split, rows, *, scale=None):
    inputs = {k: np.asarray(v[rows]) for k, v in split.inputs.items()}
    actual, origins, times = split.actual[rows], split.origin_unix[rows], split.target_time_unix[rows]
    metadata = dict(split.metadata, input_hash=_array_hash(inputs),
                    sample_hash=_array_hash(dict(origin=origins, target_time=times, actual=actual)))
    if scale is not None:
        metadata.update(target_mean=0., target_std=float(scale))
    target = (actual - metadata['target_mean']) / metadata['target_std']
    return AlignedSplit(inputs, actual, origins, times, metadata, target)


def tree_features(outputs, level):
    """No raw memory, embeddings or ungated quality values bypass rejection."""
    result = np.concatenate([*(outputs[k] for k in OUTPUT_KEYS), level[:, None]], axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite physical tree features')
    return result.astype(np.float32)
