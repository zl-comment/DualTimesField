"""Chronos-2 zero-shot adapter: seven public blocks -> official covariate API.

Per-origin mapping (each origin is an independent forecast group; no
cross-origin context ever exists because cross_learning=False and batches are
just loops):

    history_values[:, :, 0]  price (std)  --invert once via frozen train
                                            mean/std--> RAW RRP target [72]
    history_values[:, :, 1]  demand (std) --> past-only covariate `demand_std` [72]
    history_calendar [72,10]               --> past covariates cal_0..cal_9 [72]
    future_calendar  [24,10]               --> future covariates cal_0..cal_9 [24]
                                            (future keys must be a subset of
                                            past keys — satisfied exactly)
    origin_context, future_exogenous, ctf_exogenous, quantile_exogenous:
        NOT MAPPED. The official API has no static-covariate slot, and
        future-only covariates (pdpasa/predispatch/interconnector forecasts,
        scarcity shortfalls, gas context) are rejected because their API
        requires values over the history span as well; inventing history is
        forbidden. This restriction is declared in the registry
        (input_scope=restricted_...) and in every artifact, and it is the
        reason the model is not an equal-input comparison to models that
        consume all 1465 columns.

The demand covariate stays in shared standardized units (no inverse constants
for demand exist in the shared metadata); the price target is un-standardized
so model outputs are raw RRP with a single inverse transform at the input.
Zero-shot: no fit step, weights frozen, no test-label access of any kind.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ..common.foundation_api import PredictionView

CALENDAR_NAMES = tuple(f"cal_{index}" for index in range(10))


class Adapter:
    output_type = "quantile"
    deterministic = True  # frozen weights, no sampling; float32 kernels on one device

    def __init__(self, context):
        self.context = context
        params = context.config["foundation_models"]["chronos2_zeroshot"]
        expected = {"batch_origins"}
        if set(params) != expected:
            raise ValueError(f"Unexpected chronos2_zeroshot parameters: {sorted(params)}")
        self.batch_origins = int(params["batch_origins"])
        from . import model as native

        self.hf_home = os.environ.get(
            "HF_HOME", str(Path(context.config["project_root"]) / "weights" / "hf_home"))
        self.pipeline = native.load_pipeline(context.device)
        self._identity = None

    def fit(self) -> None:  # zero-shot: nothing to fit, nothing to persist
        return None

    def identity(self) -> dict[str, Any]:
        if self._identity is None:
            from . import model as native

            self._identity = {
                "implementation": "official_chronos_package_Chronos2Pipeline",
                "weights": native.weight_identity(self.hf_home),
                "packages": native.package_versions(),
                "config": {"prediction_length": 24, "context_length": 72,
                           "cross_learning": False,
                           "batch_origins": self.batch_origins,
                           "batch_size": self.batch_origins},
                "input_mapping": {
                    "history_values.price": "target_raw_rrp_72",
                    "history_values.demand": "past_covariate_demand_std_72",
                    "history_calendar": "past_covariates_cal_0_9_72",
                    "future_calendar": "future_covariates_cal_0_9_24",
                    "unmapped": ["origin_context", "future_exogenous",
                                 "ctf_exogenous", "quantile_exogenous"],
                    "unmapped_reason": "no static slot; future-only covariates "
                                       "lack the required history span"},
                "output": {"type": "quantile", "levels": [0.05, 0.1, 0.5, 0.9, 0.95],
                           "source": "native trained quantile head levels (all five are trained levels)",
                           "point_for_scoring": "rearranged_q50"},
                "seed_scope": "no_randomness_frozen_weights_zero_shot",
                "zero_shot": True,
            }
        return self._identity

    def predict_chunk(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        from . import model as native

        view: PredictionView = self.context.test_view
        inputs = self._build_inputs(view, indices)
        quantiles = native.predict_common_quantiles(
            self.pipeline, inputs, batch_size=self.batch_origins, context_length=72)
        return {"quantiles": quantiles}

    def _build_inputs(self, view: PredictionView, indices: np.ndarray) -> list[dict]:
        meta = view.metadata
        mean, std = float(meta["target_mean"]), float(meta["target_std"])
        price_raw = view.inputs["history_values"][indices][:, :, 0].astype(np.float64) * std + mean
        demand = view.inputs["history_values"][indices][:, :, 1].astype(np.float64)
        calendar_past = view.inputs["history_calendar"][indices].astype(np.float64)
        calendar_future = view.inputs["future_calendar"][indices].astype(np.float64)
        rows = []
        for row in range(len(indices)):
            past_covariates = {"demand_std": demand[row]}
            future_covariates = {}
            for column, name in enumerate(CALENDAR_NAMES):
                past_covariates[name] = calendar_past[row][:, column]
                future_covariates[name] = calendar_future[row][:, column]
            rows.append({"target": price_raw[row],  # [72] raw RRP
                         "past_covariates": past_covariates,
                         "future_covariates": future_covariates})
        return rows
