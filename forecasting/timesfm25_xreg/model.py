"""TimesFM 2.5 (200M, pytorch) native loading with the official XReg path.

Method paper (TimesFM base):
    Das et al. "A decoder-only foundation model for time-series forecasting."
    ICML 2024 / arXiv:2310.10688. https://arxiv.org/abs/2310.10688
TimesFM 2.5 checkpoint/model card (generation actually used here):
    https://huggingface.co/google/timesfm-2.5-200m-pytorch (Apache-2.0)
    Implementation: `timesfm` PyPI 2.0.2 — the package that contains
    TimesFM_2p5_200M_torch and timesfm/utils/xreg_lib.py (the official 2.5
    code; PyPI 3.0.x tracks TimesFM 3.0 and is deliberately not used).
Full BibTeX: CITATIONS.bib in this directory.

Facts verified in the installed 2.0.2 source:
- native quantile channels are [mean, q10, q20, ..., q90] — exactly the
  configured `quantiles` [0.1..0.9] plus mean. There is NO q05/q95 channel,
  so this model enters the P table (point = median channel index 5, the
  library's own point return); native quantiles ride along as auxiliary
  diagnostics and never fabricate a five-quantile artifact.
- `forecast_with_covariates` is the official exogenous path: it fits
  BatchedInContextXRegLinear on each origin's own context span using
  dynamic/static covariates, forecasts the covariate effect, and adds it
  back to every quantile channel equally (`xreg[..., None]`). This is a
  label-fitting regression step (on the origin's legal 72h price history),
  NOT "no fitting", and is recorded as such.
- `infer_is_positive` would clamp outputs to >=0 whenever the input window is
  nonnegative. NEM RRP takes negative values; we disable the flag (official
  config option) so no origin's native distribution is asymmetrically clipped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

WEIGHT_REPO = "google/timesfm-2.5-200m-pytorch"
FORECAST_CONFIG = dict(
    max_context=72, max_horizon=24, normalize_inputs=True,
    use_continuous_quantile_head=True, force_flip_invariance=True,
    infer_is_positive=False, fix_quantile_crossing=True, return_backcast=True,
)
QUANTILE_CHANNEL_LEVELS = ("mean", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MEDIAN_CHANNEL = 5  # index of q50 in the 10-channel output


def load_model(device: str):
    """Frozen TimesFM-2.5-200M pytorch checkpoint via the official 2.5 API.

    torch_compile stays off (official flag): eager kernels keep per-chunk
    inference deterministic across resume boundaries and avoid a JIT dependency
    inside offline workers.
    """
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(WEIGHT_REPO, torch_compile=False)
    model.model.to(device)
    return model


def compile_model(model):
    import timesfm

    config = timesfm.ForecastConfig(per_core_batch_size=32, **FORECAST_CONFIG)
    model.compile(config)
    return config


def package_versions() -> dict[str, str]:
    import importlib.metadata as md

    return {"timesfm": md.version("timesfm"), "torch": md.version("torch"),
            "numpy": md.version("numpy")}


def predict_with_xreg(model, *, inputs: list[np.ndarray],
                      dynamic_numerical_covariates: dict[str, list[np.ndarray]],
                      static_numerical_covariates: dict[str, list[float]],
                      ridge: float = 0.0) -> tuple[list, list]:
    """Official in-context XReg forecast; returns (points, native_quantiles).

    xreg_mode="xreg + timesfm": fit the covariate regression on each origin's
    own 72h context, forecast the horizon covariate effect, forecast the
    residual with TimesFM, add the XReg effect back. Default ridge=0 preserves
    the original least-squares run; ridge=1 is the bounded development candidate.
    Per-input target normalization remains enabled.
    """
    points, quantiles = model.forecast_with_covariates(
        inputs=inputs,
        dynamic_numerical_covariates=dynamic_numerical_covariates,
        static_numerical_covariates=static_numerical_covariates,
        xreg_mode="xreg + timesfm",
        normalize_xreg_target_per_input=True,
        ridge=ridge,
        max_rows_per_col=0,
        force_on_cpu=False,
    )
    return points, quantiles


def weight_identity(hf_home: str | Path) -> dict[str, Any]:
    import hashlib
    import json

    root = Path(hf_home) / "hub" / "models--google--timesfm-2.5-200m-pytorch"
    revision = (root / "refs" / "main").read_text().strip()
    snapshot = root / "snapshots" / revision
    weights = snapshot / "model.safetensors"
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    config = json.loads((snapshot / "config.json").read_text())
    return {"repo": WEIGHT_REPO, "revision": revision, "file": "model.safetensors",
            "sha256": digest.hexdigest(), "size_bytes": weights.stat().st_size,
            "config_keys": sorted(config)}
