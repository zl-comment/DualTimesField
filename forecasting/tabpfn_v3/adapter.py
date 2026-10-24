"""TabPFN-3 adapter: seven public blocks -> per-horizon in-context regression.

Mapping (all seven blocks, INPUT_KEYS order, flattened to the shared 1465
columns; identical to the tree-baseline flattening):

    history_values[72,2] + history_calendar[72,10] + future_calendar[24,10]
    + origin_context[1] + future_exogenous[24,11] + ctf_exogenous[24,1]
    + quantile_exogenous[24,3]  ->  X [n, 1465]

Stage contract (declared in the foundation registry and recorded in every
artifact):

- ``fit`` registers the train split's labeled rows as in-context support
  examples (train labels only; 24 horizon tasks). No weight is ever updated:
  ``weight_training=False``. This label-context fit is not "never touching
  task labels" and is not hidden anywhere.
- selection/calibration labels are never requested (``selection_use='none'``,
  ``calibration_use=False``).
- the test side is a label-free PredictionView; TabPFN queries are test input
  rows only. Test rows never enter any scaler or support set.
- seed controls only the ensemble-preprocessing RNG (``random_state``); the
  same seed reproduces bitwise-identical output, different seeds give
  genuinely different ensemble draws of the same frozen weights.

Target space: the regressors are fitted on RAW RRP labels (train standardized
targets inverted once with the frozen train mean/std from shared metadata),
so predictions need no inverse transform; output quantiles are raw RRP.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from ..aligned_inputs import INPUT_KEYS, flatten_inputs  # noqa: F401  (shared flattening)
from ..common.foundation_api import PredictionView


class Adapter:
    output_type = "quantile"
    deterministic = True  # same inputs + same random_state -> bitwise identical

    def __init__(self, context):
        self.context = context
        self._regressors: list = []
        self._identity: dict[str, Any] | None = None
        self._canonical_cache = None
        params = context.config["foundation_models"]["tabpfn_v3"]
        expected = {"batch_queries", "weight_file"}
        if set(params) != expected:
            raise ValueError(f"Unexpected tabpfn_v3 parameters: {sorted(params)}")
        if params["weight_file"] != "tabpfn-v3-regressor-v3_default.ckpt":
            raise ValueError("tabpfn_v3 fixes the V3 default regressor checkpoint")
        self.batch_queries = int(params["batch_queries"])
        if self.batch_queries <= 0:
            raise ValueError("batch_queries must be positive")
        from . import model as native

        self.hf_home = os.environ.get(
            "HF_HOME", str(Path(context.config["project_root"]) / "weights" / "hf_home"))
        self.checkpoint = native.resolve_checkpoint_path(self.hf_home)

    # -- stage boundary ----------------------------------------------------
    def fit(self) -> None:
        """Register train-only label context for the 24 horizon tasks.

        Registration is deterministic given weights/config/seed (verified
        bitwise), so a resumed worker simply re-registers it and must get the
        same state; ``models/fit_state.json`` pins the identity and is checked
        against any stored state to refuse mixing snapshots. The registration
        itself is never skipped: it is the in-memory context the frozen
        checkpoint attends over, and rebuilding it is the only correct way to
        resume.
        """
        state_path = self.context.model_directory / "fit_state.json"
        identity = self.identity()
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("identity_hash") != _hash(identity):
                raise ValueError("Stored TabPFN label-context state does not match "
                                 "current weights/config; refusing to mix states")
        train = self.context.train
        if train is None:
            raise ValueError("tabpfn_v3 fit requires the train split (declared "
                             "fit_requires_train_labels=True)")
        from . import model as native

        started = time.monotonic()
        x = np.asarray(flatten_inputs(train), dtype=np.float32)
        mean, std = float(train.metadata["target_mean"]), float(train.metadata["target_std"])
        y_raw = np.asarray(train.target, dtype=np.float64) * std + mean  # one inversion, train labels only
        if x.shape[0] != y_raw.shape[0] or y_raw.shape[1] != 24:
            raise ValueError("Train context shape mismatch")
        self._regressors = []
        self._canonical_cache = None
        for horizon in range(24):
            regressor = native.create_regressor(
                self.checkpoint, device=self.context.device,
                random_state=self.context.seed + horizon)
            regressor.fit(x, y_raw[:, horizon])
            self._regressors.append(regressor)
        if not state_path.exists():
            self.context.model_directory.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps({
                "identity_hash": _hash(identity), "created_at": time.time(),
                "train_rows": int(x.shape[0]), "features": int(x.shape[1]),
                "horizons": 24, "fit_seconds": round(time.monotonic() - started, 3),
                "label_space": "raw_rrp_via_frozen_train_mean_std",
                "context_note": "train labels only; in-context support, no weight updates; "
                                "deterministic registration is rebuilt on resume",
            }, indent=2) + "\n")

    def identity(self) -> dict[str, Any]:
        if self._identity is None:
            from . import model as native

            params = self.context.config["foundation_models"]["tabpfn_v3"]
            self._identity = {
                "implementation": "official_tabpfn_package_local_v3_checkpoint",
                "weights": native.weight_identity(self.hf_home),
                "packages": native.package_versions(),
                "config": {"n_estimators": "auto", "random_state_per_horizon":
                           [self.context.seed + h for h in range(24)],
                           "ignore_pretraining_limits": False,
                           "batch_queries": self.batch_queries,
                           "query_batch_policy": "canonical_view_chunks_v1",
                           "weight_file": params["weight_file"]},
                "input_mapping": "seven_blocks_flattened_1465",
                "output": {"type": "quantile", "levels": [0.05, 0.1, 0.5, 0.9, 0.95],
                           "source": "native_predicted_distribution_quantiles",
                           "point_for_scoring": "rearranged_q50"},
                "seed_scope": "ensemble_preprocessing_rng_only_weights_frozen",
            }
        return self._identity

    # -- prediction --------------------------------------------------------
    def predict_chunk(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        """Replay the same canonical query blocks used by the original full run.

        Subset calls must not change the numerical batch shape. Context labels
        remain train-only. A single-block cache avoids repeated inference when
        inspecting the same block; no predictions are shared between adapters.
        Full ordered 4096-row production calls execute the original path.
        """
        from . import model as native
        if not self._regressors:
            self.fit()
        view = self.context.test_view
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError('Query indices must be a one-dimensional integer array')
        if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(view)):
            raise ValueError('Query indices must be nonempty and inside the view')
        result = np.empty((len(indices), len(self._regressors), 5), dtype=np.float64)
        for block in np.unique(indices // self.batch_queries):
            start = int(block) * self.batch_queries
            stop = min(start + self.batch_queries, len(view))
            key = (id(view), start, stop)
            if self._canonical_cache is None or self._canonical_cache[0] != key:
                x = np.asarray(flatten_inputs(_ViewSplit(view, np.arange(start, stop))), dtype=np.float32)
                quantiles = np.stack([native.predict_quantiles(regressor, x)
                                      for regressor in self._regressors], axis=1)
                self._canonical_cache = (key, quantiles)
            mask = indices // self.batch_queries == block
            result[mask] = self._canonical_cache[1][indices[mask] - start]
        if result.shape != (len(indices), 24, 5):
            raise ValueError(f'TabPFN chunk shape {result.shape}')
        return {'quantiles': result}


def _hash(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


class _ViewSplit:
    """flatten_inputs-compatible facade over a label-free view slice."""

    def __init__(self, view: PredictionView, indices: np.ndarray):
        self.inputs = {key: view.inputs[key][indices] for key in INPUT_KEYS}
        self._n = len(indices)

    def __len__(self) -> int:
        return self._n
