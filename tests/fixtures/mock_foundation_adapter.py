"""Deterministic mock adapter for foundation-runner tests (no model packages).

Emits quantiles/points as pure functions of the input blocks so tests can
check runner mechanics: chunk persistence, resume, tamper detection, seed
reuse and label isolation. Records every predict/fit call so tests can prove
no re-prediction or refit happened on resume. Mirrors the real adapter fit
contract: an existing fit_state.json with matching identity is honoured, not
refitted.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from forecasting.common.foundation_api import INPUT_KEYS


class Adapter:
    output_type = "quantile"

    def __init__(self, context):
        self.context = context
        if hasattr(context.test_view, "actual") or hasattr(context.test_view, "target"):
            raise AssertionError("test view leaked labels into the adapter")
        self.call_log = Path(context.model_directory) / "adapter_calls.jsonl"

    def _log(self, record: dict) -> None:
        self.call_log.parent.mkdir(parents=True, exist_ok=True)
        with self.call_log.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def identity(self):
        return {"implementation": "mock", "weights": {"sha256": "0" * 64},
                "packages": {}, "config": {}, "seed_free": True}

    def fit(self):
        marker = Path(self.context.model_directory) / "fit_state.json"
        if marker.exists():
            state = json.loads(marker.read_text())
            if state.get("identity") == self.identity():
                self._log({"call": "fit", "skipped": "state_verified"})
                return
            raise AssertionError("stored fit state does not match adapter identity")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"identity": self.identity(),
                                      "fitted_at": time.time()}) + "\n")
        self._log({"call": "fit", "skipped": False})

    def predict_chunk(self, indices):
        view = self.context.test_view
        # Per-row pure function: no cross-row aggregation, so a row's output is
        # bitwise identical no matter which batch it travels in.
        rows = [int(np.asarray(view.inputs[key][index]).astype(np.float64).sum())
                for index in indices for key in INPUT_KEYS[:1]]
        base = np.array([50.0 + row * 1e-6 + (index % 7) for row, index in zip(rows, indices)],
                        dtype=np.float64)
        quantiles = np.stack([base * (1 + delta) for delta in
                              (-0.3, -0.15, 0.0, 0.15, 0.3)], axis=-1)[:, None, :]
        quantiles = np.repeat(quantiles, 24, axis=1)
        self._log({"call": "predict", "indices": [int(i) for i in indices]})
        return {"quantiles": quantiles}


class PointAdapter(Adapter):
    output_type = "point_only"

    def predict_chunk(self, indices):
        payload = super().predict_chunk(indices)
        return {"point": payload["quantiles"][:, :, 2]}
