"""Moirai-2 zero-shot adapter: seven public blocks -> official covariate fields.

Per-origin mapping (each origin an independent sample; the official predict
interface receives exactly this origin's history and its as-of covariates):

    history_values[:, :, 0]  price (std) --invert once via frozen train
                                          mean/std--> RAW RRP `past_target` [72,1]
    history_values[:, :, 1]  demand(std) --> `past_feat_dynamic_real` [72,1]
                                             (past-only covariate field)
    history_calendar[72,10] + future_calendar[24,10]
                                          --> `feat_dynamic_real` [96,10]
                                             (fields must span past+future;
                                             calendar is known in the future)
    origin_context, future_exogenous, ctf_exogenous, quantile_exogenous:
        NOT MAPPED (no static field in this model version; future-only
        covariates would need a past span; inventing history is forbidden).
        Declared via input_scope=restricted_... in the registry/artifacts.

Output table: P. The checkpoint's native quantile set is exactly [0.1..0.9];
there is no q05/q95 head and no sampling/distribution path in Moirai-2, so no
five-quantile Q-table entry is fabricated: point = native q50, and native
q10/q90 ride along as auxiliary diagnostics inside the cell directory
(auxiliary_quantiles.npz), never inside the artifact's quantile contract.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ..common.foundation_api import PredictionView


class Adapter:
    output_type = "point_only"
    deterministic = True  # frozen weights, dropout forced to zero, no sampling

    def __init__(self, context):
        self.context = context
        params = context.config["foundation_models"]["moirai2_zeroshot"]
        expected = {"batch_origins"}
        if set(params) != expected:
            raise ValueError(f"Unexpected moirai2_zeroshot parameters: {sorted(params)}")
        self.batch_origins = int(params["batch_origins"])
        from . import model as native

        self.hf_home = os.environ.get(
            "HF_HOME", str(Path(context.config["project_root"]) / "weights" / "hf_home"))
        self.forecast = native.load_forecast(
            prediction_length=24, target_dim=1, feat_dynamic_real_dim=10,
            past_feat_dynamic_real_dim=1, context_length=72, device=context.device)
        self.predictor = native.create_predictor(
            self.forecast, batch_size=self.batch_origins, device=context.device)
        self._identity = None

    def fit(self) -> None:  # zero-shot: nothing to fit
        return None

    def identity(self) -> dict[str, Any]:
        if self._identity is None:
            from . import model as native

            self._identity = {
                "implementation": "official_uni2ts_git_commit_Moirai2Forecast",
                "weights": native.weight_identity(self.hf_home),
                "packages": native.package_versions(),
                "config": {"prediction_length": 24, "context_length": 72,
                           "batch_origins": self.batch_origins, "target_dim": 1,
                           "feat_dynamic_real_dim": 10, "past_feat_dynamic_real_dim": 1},
                "input_mapping": {
                    "history_values.price": "past_target_raw_rrp_72x1",
                    "history_values.demand": "past_feat_dynamic_real_72x1",
                    "history_calendar+future_calendar": "feat_dynamic_real_96x10",
                    "unmapped": ["origin_context", "future_exogenous",
                                 "ctf_exogenous", "quantile_exogenous"],
                    "unmapped_reason": "no static field in Moirai-2 wrapper; "
                                       "future-only covariates lack the past span"},
                "output": {"type": "point_only", "point": "native_q50",
                           "native_quantile_levels": list(native.NATIVE_QUANTILE_LEVELS),
                           "q05_q95": "not available natively; no extrapolation or "
                                      "endpoint copying; therefore P table, not Q5",
                           "auxiliary": "native q10/q90/q20/.../q80 exported to "
                                        "auxiliary_quantiles.npz, excluded from the "
                                        "artifact's quantile contract"},
                "seed_scope": "no_randomness_frozen_weights_zero_shot",
                "zero_shot": True,
            }
        return self._identity

    def predict_chunk(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        from . import model as native

        view: PredictionView = self.context.test_view
        meta = view.metadata
        mean, std = float(meta["target_mean"]), float(meta["target_std"])
        price_raw = view.inputs["history_values"][indices][:, :, 0].astype(np.float64) * std + mean
        demand = view.inputs["history_values"][indices][:, :, 1].astype(np.float64)
        calendar = np.concatenate([view.inputs["history_calendar"][indices],
                                   view.inputs["future_calendar"][indices]], axis=1)
        import pandas as pd

        points, auxiliaries = [], []
        for start in range(0, len(indices), self.batch_origins):
            stop = min(start + self.batch_origins, len(indices))
            entries = [{
                "start": pd.Timestamp(int(origin), unit="s", tz="Australia/Brisbane"),
                "target": price_raw[row],
                "feat_dynamic_real": calendar[row].T,        # [10, 96]
                "past_feat_dynamic_real": demand[row][None, :],  # [1, 72]
            } for row, origin in zip(range(start, stop),
                                     np.asarray(view.origin_unix[indices][start:stop]))]
            batch = native.predict_batch(self.predictor, entries)
            # [b, 9, 24] with levels [0.1..0.9]; index 4 is q50.
            points.append(batch[:, 4, :])
            auxiliaries.append(batch[:, [0, 4, 8], :])  # q10, q50, q90 diagnostic
        point = np.concatenate(points, axis=0)
        self._last_auxiliary = np.concatenate(auxiliaries, axis=0)
        if point.shape != (len(indices), 24):
            raise ValueError(f"Moirai-2 point chunk shape {point.shape}")
        return {"point": point}

    def auxiliary_quantiles(self) -> np.ndarray:
        """Native q10/q50/q90 from the most recent chunk prediction [n, 3, 24]."""
        if getattr(self, "_last_auxiliary", None) is None:
            raise ValueError("No chunk prediction has run yet")
        return self._last_auxiliary
