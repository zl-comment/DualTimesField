"""TabPFN-3 native loading and per-horizon in-context regression.

Paper (technical report, third generation):
    "TabPFN 3: Machine Learning with Zero Hyperparameter Tuning"
    Prior Labs, 2026. https://arxiv.org/abs/2605.13986
    Weights/model card: https://huggingface.co/Prior-Labs/tabpfn_3
    Official implementation: https://github.com/PriorLabs/TabPFN
    Lineage (v2, Nature paper — cited as ancestry, NOT as this checkpoint):
    Hollmann, Müller, Eggensperger, Hutter. "TabPFN: One Model to Rule Them
    All?" IEEE ICML 2023 / Nature 645 (2025). https://doi.org/10.1038/s41586-024-08328-6
Full BibTeX: CITATIONS.bib in this directory.

Library note: the `tabpfn` package version (9.2.0) is not the model
generation. The default constructor resolves to the latest weights (v3.5 at
audit time); this adapter pins the explicit V3 regressor checkpoint
`tabpfn-v3-regressor-v3_default.ckpt` from `Prior-Labs/tabpfn_3` and passes
its local path directly, so no version drift and no interactive download gate
is involved at run time.

Nothing in this module touches experiment splits, stages or labels; the
adapter in `adapter.py` owns the seven-block mapping and stage boundaries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

WEIGHT_REPO = "Prior-Labs/tabpfn_3"
WEIGHT_FILE = "tabpfn-v3-regressor-v3_default.ckpt"


def resolve_checkpoint_path(hf_home: str | Path) -> Path:
    """Locate the pinned V3 regressor checkpoint inside the shared HF cache."""
    pattern = Path(hf_home) / "hub" / "models--Prior-Labs--tabpfn_3" / "snapshots"
    matches = sorted(pattern.rglob(WEIGHT_FILE))
    if not matches:
        raise FileNotFoundError(
            f"TabPFN V3 checkpoint not downloaded under {pattern}. "
            "Run the download step in ENVIRONMENT.md once; workers run offline.")
    return matches[0].resolve()


def create_regressor(checkpoint: Path, *, device: str, random_state: int):
    """One frozen-weight V3 regressor with the official default settings."""
    from tabpfn import TabPFNRegressor

    return TabPFNRegressor(
        model_path=str(checkpoint), device=device,
        random_state=random_state, n_estimators="auto",
        ignore_pretraining_limits=False, show_progress_bar=False)


def package_versions() -> dict[str, str]:
    import importlib.metadata as md

    return {"tabpfn": md.version("tabpfn"), "torch": md.version("torch"),
            "numpy": md.version("numpy")}


def predict_quantiles(regressor, x: np.ndarray,
                      quantiles: tuple[float, ...] = (0.05, 0.1, 0.5, 0.9, 0.95)) -> np.ndarray:
    """Native distribution quantiles for one horizon task.

    Returns [n, len(quantiles)] in the target space the regressor was fitted
    on (raw RRP here). The five common levels come from the predicted
    distribution's CDF; no sampling, extrapolation or endpoint copying.
    """
    result = np.asarray(regressor.predict(
        x, output_type="quantiles", quantiles=list(quantiles)))
    # tabpfn returns [len(quantiles), n]; transpose to [n, len(quantiles)].
    if result.ndim != 2:
        raise ValueError(f"TabPFN quantile output has unexpected rank {result.ndim}")
    if result.shape[0] == len(quantiles) and result.shape[1] == len(x):
        result = result.T
    elif result.shape != (len(x), len(quantiles)):
        raise ValueError(f"TabPFN quantile output has unexpected shape {result.shape}")
    if not np.isfinite(result).all():
        raise FloatingPointError("TabPFN produced non-finite quantiles")
    return result.astype(np.float64, copy=False)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weight_identity(hf_home: str | Path) -> dict[str, Any]:
    checkpoint = resolve_checkpoint_path(hf_home)
    return {"repo": WEIGHT_REPO, "file": WEIGHT_FILE,
            "path": str(checkpoint), "sha256": sha256_file(checkpoint),
            "size_bytes": checkpoint.stat().st_size}
