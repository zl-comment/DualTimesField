"""Label-free prediction contract shared by the second-batch foundation models.

The first-batch worker handed the full test ``AlignedSplit`` — labels included
— to its adapters. This module is the second batch's correction: a
``PredictionView`` exposes the seven shared input blocks, the origin/target
timestamps and the frozen transform metadata, and nothing else. There is no
``actual`` or ``target`` attribute to read by accident; the scheduler loads
labels itself only after the adapter has returned its predictions.

Adapters receive indices into the view and must treat every origin as an
independent sample: no cross-origin context, grouping or retrieval is allowed,
because one origin's history contains another origin's future.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..aligned_inputs import INPUT_KEYS  # noqa: F401  (re-exported for adapters)


def view_slice_hash(view: "PredictionView", indices: np.ndarray) -> str:
    """SHA-256 over the exact input rows an adapter is asked to predict."""
    digest = hashlib.sha256()
    index = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    digest.update(json.dumps(["indices", index.dtype.str, index.shape]).encode())
    digest.update(index.tobytes())
    for key in INPUT_KEYS:
        array = np.ascontiguousarray(view.inputs[key][index])
        digest.update(json.dumps([key, array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes())
    digest.update(np.ascontiguousarray(view.origin_unix[index]).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PredictionView:
    """The complete, label-free evaluation surface for one region/split.

    ``inputs`` holds exactly the seven shared blocks; ``metadata`` is the
    shared input-builder metadata (hashes, train-fitted target mean/std,
    protocol id). Actual values and transformed targets are not fields of this
    object and are never given to adapters.
    """

    inputs: Mapping[str, np.ndarray]
    origin_unix: np.ndarray
    target_time_unix: np.ndarray
    metadata: dict

    def __len__(self) -> int:
        return int(len(self.origin_unix))

    def slice(self, start: int, stop: int) -> "PredictionView":
        return PredictionView(
            {key: value[start:stop] for key, value in self.inputs.items()},
            self.origin_unix[start:stop], self.target_time_unix[start:stop],
            dict(self.metadata))

    def input_slice_hash(self, indices) -> str:
        return view_slice_hash(self, np.asarray(indices))
