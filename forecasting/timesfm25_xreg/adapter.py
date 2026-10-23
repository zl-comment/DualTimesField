"""TimesFM-2.5 + official XReg adapter: seven public blocks -> covariate API.

Per-origin mapping (each origin forecasts independently from exactly its own
72h legal history and as-of covariates):

    history_values[:, :, 0]  price (std) --invert once via frozen train
                                          mean/std--> RAW RRP context [72]
    history_calendar[72,10] + future_calendar[24,10]
                                          --> dynamic_numerical_covariates
                                          cal_0..cal_9, each [96] (the XReg
                                          linear fit needs the covariate over
                                          the fitting span AND the horizon;
                                          calendar is deterministic and known
                                          for both)
    origin_context [1]                    --> static_numerical_covariates
                                          gas_7d_log_std (per-origin scalar)
    history_values[:, :, 1]  demand(std): NOT usable — a dynamic covariate
        needs 24 legal future values; the cache holds only forecast-vintage
        demand under a different standardizer, and splicing actual history
        with differently-standardized forecast values would silently mix
        statistical variables (recorded, rejected).
    future_exogenous / ctf_exogenous / quantile_exogenous: future-only, no
        history span, not inventable -> not mapped.

Stage contract: zero-shot backbone; the ONLY fit is the official in-context
XReg least-squares on each origin's own 72h price history (a label-fitting
step on legally available history, explicitly not "no fitting"). No train,
selection or calibration labels are touched (fit_requires_train_labels=False);
weights are frozen (weight_training=False).

Output: P table, point = native median channel + XReg effect (the library's
point return). Native [mean,q10..q90] channels (+XReg on every channel,
equally shifted) are exported as auxiliary diagnostics, never as a
five-quantile artifact.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ..common.foundation_api import PredictionView

CALENDAR_NAMES = tuple(f"cal_{index}" for index in range(10))


class Adapter:
    output_type = "point_only"
    deterministic = True  # frozen weights; deterministic kernels + LS fit given hardware

    def __init__(self, context):
        self.context = context
        params = context.config["foundation_models"]["timesfm25_xreg"]
        expected = {"batch_origins", "xreg_mode", "ridge"}
        if set(params) != expected:
            raise ValueError(f"Unexpected timesfm25_xreg parameters: {sorted(params)}")
        if params["xreg_mode"] != "xreg + timesfm" or float(params["ridge"]) != 0.0:
            raise ValueError("timesfm25_xreg fixes the official defaults "
                             "xreg_mode='xreg + timesfm' and ridge=0.0")
        self.batch_origins = int(params["batch_origins"])
        from . import model as native

        self.hf_home = os.environ.get(
            "HF_HOME", str(Path(context.config["project_root"]) / "weights" / "hf_home"))
        self.model = native.load_model(context.device)
        self.config = native.compile_model(self.model)
        self._identity = None

    def fit(self) -> None:  # the XReg fit is per-origin inside predict (in-context)
        return None

    def identity(self) -> dict[str, Any]:
        if self._identity is None:
            from . import model as native

            self._identity = {
                "implementation": "official_timesfm_2p5_pytorch_package_with_xreg",
                "weights": native.weight_identity(self.hf_home),
                "packages": native.package_versions(),
                "config": {"forecast_config": native.FORECAST_CONFIG,
                           "batch_origins": self.batch_origins,
                           "xreg_mode": "xreg + timesfm", "ridge": 0.0,
                           "normalize_xreg_target_per_input": True},
                "input_mapping": {
                    "history_values.price": "raw_rrp_context_72",
                    "history_calendar+future_calendar": "dynamic_numerical_covariates_cal_0_9_96",
                    "origin_context": "static_numerical_covariate_gas_7d_log_std",
                    "unmapped": ["history_values.demand", "future_exogenous",
                                 "ctf_exogenous", "quantile_exogenous"],
                    "unmapped_reason": "dynamic covariates need legal future spans; "
                                       "future-only cache blocks and differently-standardized "
                                       "forecast vintages are not spliced"},
                "output": {"type": "point_only", "point": "native_median_channel_plus_xreg",
                           "native_quantile_channels": list(native.QUANTILE_CHANNEL_LEVELS),
                           "q05_q95": "not native; P table, no fabricated five-quantile artifact",
                           "auxiliary": "native 10 channels exported to "
                                        "auxiliary_quantiles.npz per cell"},
                "xreg_fit": {"fits_on": "each origin's own legal 72h price history "
                                        "(in-context labels, not train split)",
                             "parameters": "per-origin least squares with per-input "
                                           "target normalization; ridge=0",
                             "order": "fit covariate regression -> forecast horizon "
                                      "effect -> TimesFM forecasts residual -> add back",
                             "quantile_effect": "added equally to every channel (shift)"},
                "infer_is_positive_note": "disabled (official flag): NEM RRP has "
                                          "negative prices; enabling would clip native "
                                          "outputs on nonneg-history origins only",
                "seed_scope": "no_randomness_frozen_weights_deterministic_ls_fit",
                "zero_shot_backbone": True,
            }
        return self._identity

    def predict_chunk(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        from . import model as native

        view: PredictionView = self.context.test_view
        meta = view.metadata
        mean, std = float(meta["target_mean"]), float(meta["target_std"])
        price_raw = view.inputs["history_values"][indices][:, :, 0].astype(np.float64) * std + mean
        calendar = np.concatenate([view.inputs["history_calendar"][indices],
                                   view.inputs["future_calendar"][indices]], axis=1)
        gas = view.inputs["origin_context"][indices][:, 0].astype(np.float64)
        points, auxiliaries = [], []
        for start in range(0, len(indices), self.batch_origins):
            stop = min(start + self.batch_origins, len(indices))
            inputs = [price_raw[row] for row in range(start, stop)]
            dynamic = {name: [calendar[row][:, column] for row in range(start, stop)]
                       for column, name in enumerate(CALENDAR_NAMES)}
            static = {"gas_7d_log_std": [float(gas[row]) for row in range(start, stop)]}
            chunk_points, chunk_quantiles = native.predict_with_xreg(
                self.model, inputs=inputs, dynamic_numerical_covariates=dynamic,
                static_numerical_covariates=static)
            for row_index, point in enumerate(chunk_points):
                array = np.asarray(point, dtype=np.float64)
                if array.shape != (24,):
                    raise ValueError(f"TimesFM point output shape {array.shape}")
                points.append(array)
            for row_index, quantile in enumerate(chunk_quantiles):
                array = np.asarray(quantile, dtype=np.float64)
                if array.shape != (24, 10):
                    raise ValueError(f"TimesFM quantile output shape {array.shape}")
                auxiliaries.append(array)
        point = np.stack(points, axis=0)
        self._last_auxiliary = np.stack(auxiliaries, axis=0)
        if not np.isfinite(point).all():
            raise FloatingPointError("TimesFM produced non-finite points")
        return {"point": point}

    def auxiliary_quantiles(self) -> np.ndarray:
        """Native [mean,q10..q90] channels + XReg shift from the last chunk [n,24,10]."""
        if getattr(self, "_last_auxiliary", None) is None:
            raise ValueError("No chunk prediction has run yet")
        return self._last_auxiliary
