"""Registered expansion baselines for the 2026-10-22 fair-baseline batch.

This is the explicit registration mechanism the expansion scope needs: each
first-batch model declares its adapter module, output table, stage and
resource requirements here once. The tuple is fixed source, not runtime
configuration; the historical BASE_MODELS/EXTENSION_MODELS semantics and the
frozen 90/18-cell runs are untouched.

Registry entries carry no behaviour. The adapter in each model directory owns
its native fitting; ``run_expansion_model`` lazily imports it so a missing
optional dependency of one model never blocks importing or running another.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

EXPANSION_SCHEMA = "aligned-expansion-20261022-v1"


@dataclass(frozen=True)
class ExpansionSpec:
    model_id: str
    module: str                 # adapter package under forecasting/
    entry: str                  # adapter callable name, ``entry(context)``
    output_table: str           # 'Q' (five-quantile) or 'P' (point only)
    resource: str               # 'cpu' or 'gpu'
    requires_calibration: bool  # adapter consumes the calibration split labels
    native_training: bool       # adapter owns its training loop (no shared pinball runner)
    base_model: str | None      # frozen parent model this adapter reuses, if any
    dependencies: tuple[str, ...] = ()  # importable modules required at fit time


EXPANSION_MODELS: tuple[ExpansionSpec, ...] = (
    ExpansionSpec("lightgbm_cp", "forecasting.lightgbm_cp.model", "run_expansion",
                  "Q", "cpu", True, False, "lightgbm"),
    ExpansionSpec("lightgbm_qra", "forecasting.lightgbm_qra.model", "run_expansion",
                  "Q", "cpu", True, False, "lightgbm"),
    ExpansionSpec("xgboost_cp", "forecasting.xgboost_cp.model", "run_expansion",
                  "Q", "cpu", True, False, "xgboost"),
    ExpansionSpec("ddnn_jsu", "forecasting.ddnn_jsu.model", "run_expansion",
                  "Q", "gpu", False, True, None, ("torch",)),
    ExpansionSpec("nhits_qra", "forecasting.nhits_qra.model", "run_expansion",
                  "Q", "gpu", True, True, None, ("torch",)),
)

REGISTRY: dict[str, ExpansionSpec] = {spec.model_id: spec for spec in EXPANSION_MODELS}


def expansion_model_ids() -> tuple[str, ...]:
    return tuple(REGISTRY)


def expansion_availability() -> dict[str, dict[str, Any]]:
    """Implemented/available status for every registered expansion model."""
    result = {}
    for spec in EXPANSION_MODELS:
        missing = [module for module in spec.dependencies if find_spec(module) is None]
        result[spec.model_id] = {
            "implemented": True, "registered_entry": f"{spec.module}::{spec.entry}",
            "output_table": spec.output_table, "resource": spec.resource,
            "requires_calibration": spec.requires_calibration,
            "native_training": spec.native_training,
            "frozen_base_model": spec.base_model,
            "missing_dependencies": missing,
            "status": "available" if not missing else "missing_dependency",
        }
    return result


def run_expansion_model(model_id: str, context: Any):
    """Dispatch to the registered adapter; returns a BaselinePrediction."""
    from ..aligned_baselines import BaselinePrediction  # noqa: F401  (type contract)

    if model_id not in REGISTRY:
        raise ValueError(f"Unknown expansion baseline: {model_id}")
    spec = REGISTRY[model_id]
    missing = [module for module in spec.dependencies if find_spec(module) is None]
    if missing:
        raise ImportError(f"{model_id} is blocked: missing dependencies {missing}")
    import importlib

    module = importlib.import_module(spec.module)
    entry = getattr(module, spec.entry, None)
    if entry is None:
        raise NotImplementedError(f"{spec.module} does not provide {spec.entry}")
    result = entry(context)
    if not isinstance(result, BaselinePrediction):
        raise TypeError(f"{model_id} adapter must return a BaselinePrediction")
    return result
