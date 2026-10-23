"""Moirai-2 zero-shot native loading (uni2ts, pinned GitHub commit).

Technical report (Moirai-2):
    "Unified Training of Universal Time Series Forecasting Transformers"
    (Moirai-2). Salesforce AI Research, 2025. https://arxiv.org/abs/2511.11698
    Weights/model card: https://huggingface.co/Salesforce/moirai-2.0-R-small
    (cc-by-nc-4.0, research use)
    Official implementation: https://github.com/SalesforceAIResearch/uni2ts
    at pinned commit cfd46d4510ed8896f263116f32928eede05b0a75 (vendored
    provenance in source.json; the PyPI `uni2ts` 2.0.0 wheel predates the
    Moirai-2 classes and must not be used as the implementation).
Full BibTeX: CITATIONS.bib in this directory.

Capability facts verified on this checkpoint (config.json, revision 30f43ff0):
`quantile_levels` are exactly [0.1..0.9] — the model has NO q05/q95 head and
its forecast generator is quantile-only (no distribution/sampling output).
Per the protocol, tails are not extrapolated, endpoint-copied or Gauss-padded:
the native q50 becomes the point forecast (P table) and the native q10/q90
are exported as clearly labelled auxiliary diagnostics only.

`Moirai2Forecast` maps any frequency; we use hourly with context_length=72
and prediction_length=24. The constructor forces attention/hidden dropout to
zero (eval determinism). No experiment/stage logic lives here; see adapter.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

WEIGHT_REPO = "Salesforce/moirai-2.0-R-small"
SOURCE_COMMIT = "cfd46d4510ed8896f263116f32928eede05b0a75"
NATIVE_QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def load_forecast(*, prediction_length: int, target_dim: int,
                  feat_dynamic_real_dim: int, past_feat_dynamic_real_dim: int,
                  context_length: int, device: str):
    """Frozen Moirai-2.0-R-small wrapped in the official forecast wrapper."""
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    module = Moirai2Module.from_pretrained(WEIGHT_REPO)
    forecast = Moirai2Forecast(
        prediction_length=prediction_length, target_dim=target_dim,
        feat_dynamic_real_dim=feat_dynamic_real_dim,
        past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        context_length=context_length, module=module)
    return forecast.to(device=device).eval()


def create_predictor(forecast, *, batch_size: int, device: str):
    """Official gluonts predictor; the splitter owns past/future covariate spans.

    This is the supported entry from the official example notebooks: inputs are
    gluonts entries whose ``feat_dynamic_real`` spans the history AND horizon
    while ``past_feat_dynamic_real`` covers history only; TFTInstanceSplitter
    slices each origin's window. The raw ``forecast.predict`` path pads every
    array to context_length and must not be used with future-span covariates.
    """
    return forecast.create_predictor(batch_size=batch_size, device=device)


def predict_batch(predictor, entries: list[dict]) -> np.ndarray:
    """Run the predictor over per-origin gluonts entries.

    Each entry: {"start": pd.Timestamp (tz-aware), "target": [72] raw RRP,
                 "feat_dynamic_real": [10, 96], "past_feat_dynamic_real": [1, 72]}.
    Returns [batch, 9, 24] in raw RRP, levels ordered as NATIVE_QUANTILE_LEVELS.
    """
    import pandas as pd
    from gluonts.dataset.common import ListDataset

    dataset = ListDataset(entries, freq="1h")
    outputs = []
    for forecast in predictor.predict(dataset):
        block = np.stack([np.asarray(forecast.quantile(level), dtype=np.float64)
                          for level in NATIVE_QUANTILE_LEVELS], axis=0)  # [9, 24]
        outputs.append(block)
    result = np.stack(outputs, axis=0)
    if result.shape[1:] != (len(NATIVE_QUANTILE_LEVELS), 24):
        raise ValueError(f"Moirai-2 output has unexpected shape {result.shape}")
    if not np.isfinite(result).all():
        raise FloatingPointError("Moirai-2 produced non-finite outputs")
    return result


def package_versions() -> dict[str, str]:
    import importlib.metadata as md

    return {"uni2ts": f"{md.version('uni2ts')}+git:{SOURCE_COMMIT[:12]}",
            "torch": md.version("torch"), "numpy": md.version("numpy"),
            "gluonts": md.version("gluonts")}


def weight_identity(hf_home: str | Path) -> dict[str, Any]:
    import hashlib
    import json

    root = Path(hf_home) / "hub" / "models--Salesforce--moirai-2.0-R-small"
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
            "config": {key: config[key] for key in
                       ("quantile_levels", "patch_size", "max_seq_len", "d_model", "num_layers")}}
