"""Chronos-2 native pipeline loading for zero-shot forecasting.

Technical report (Chronos-2):
    "Chronos-2: The Time Series Forecasting Model for the Agentic Era"
    Amazon, 2025. https://arxiv.org/abs/2510.15821
    Weights/model card: https://huggingface.co/amazon/chronos-2 (Apache-2.0)
    Official implementation: https://github.com/amazon-science/chronos-forecasting
Full BibTeX: CITATIONS.bib in this directory.

Facts verified against amazon/chronos-2 config.json (revision 29ec3766):
native quantile head trained on 21 levels [.01,.05,...,.99] — every one of
the five common levels is a trained head output, so the common five-quantile
score uses exact native quantiles (no interpolation, no endpoint copying;
the package only interpolates for levels outside its trained set, which we
never request). `use_arcsinh=true` is the checkpoint's own native transform.
This module contains no experiment/stage logic; see adapter.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

WEIGHT_REPO = "amazon/chronos-2"
NATIVE_QUANTILES = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)
COMMON_LEVELS = (0.05, 0.1, 0.5, 0.9, 0.95)


def load_pipeline(device: str):
    """Frozen Chronos-2 pipeline on the pinned public checkpoint."""
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(WEIGHT_REPO, device_map=device)


def snapshot_files(hf_home: str | Path) -> list[str]:
    root = Path(hf_home) / "hub" / "models--amazon--chronos-2" / "snapshots"
    if not root.exists():
        raise FileNotFoundError("amazon/chronos-2 not downloaded; see ENVIRONMENT.md")
    latest = sorted(root.iterdir())[-1]
    return [str(p.relative_to(latest)) for p in sorted(latest.rglob("*")) if p.is_file()]


def package_versions() -> dict[str, str]:
    import importlib.metadata as md

    return {"chronos-forecasting": md.version("chronos-forecasting"),
            "torch": md.version("torch"), "numpy": md.version("numpy"),
            "pandas": md.version("pandas")}


def predict_common_quantiles(pipeline, inputs: list[dict], *, batch_size: int,
                             context_length: int = 72) -> np.ndarray:
    """Zero-shot prediction for a list of per-origin input dicts.

    Returns [n, 24, 5] in the target space of the passed series (raw RRP).
    cross_learning stays False: every origin is an independent group and no
    cross-series attention is built (isolation requirement). All five common
    levels are native trained levels; the pipeline selects them directly.
    """
    quantiles, _mean = pipeline.predict_quantiles(
        inputs, prediction_length=24, quantile_levels=list(COMMON_LEVELS),
        batch_size=batch_size, context_length=context_length, cross_learning=False)
    arrays = []
    for level_block in quantiles:
        array = np.asarray(level_block, dtype=np.float64)  # [n_variates=1, 24, 5]
        if array.ndim != 3 or array.shape[0] != 1 or array.shape[1:] != (24, 5):
            raise ValueError(f"Chronos-2 quantile block has unexpected shape {array.shape}")
        arrays.append(array[0])
    result = np.stack(arrays, axis=0)
    if not np.isfinite(result).all():
        raise FloatingPointError("Chronos-2 produced non-finite quantiles")
    return result


def weight_identity(hf_home: str | Path) -> dict[str, Any]:
    import hashlib
    import json

    root = Path(hf_home) / "hub" / "models--amazon--chronos-2"
    revision = (root / "refs" / "main").read_text().strip()
    snapshot = root / "snapshots" / revision
    weights = snapshot / "model.safetensors"
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    config = json.loads((snapshot / "config.json").read_text())["chronos_config"]
    return {"repo": WEIGHT_REPO, "revision": revision,
            "file": "model.safetensors", "sha256": digest.hexdigest(),
            "size_bytes": weights.stat().st_size,
            "native_quantile_levels": config["quantiles"],
            "use_arcsinh": config["use_arcsinh"],
            "context_length_limit": config["context_length"]}
