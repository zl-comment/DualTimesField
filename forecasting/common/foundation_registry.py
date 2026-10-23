"""Registered foundation-model baselines for the 2026-10-23 second batch.

Second-batch registry, deliberately separate from the frozen first-batch
``EXPANSION_MODELS``: the 2026-10-22 matrix, its schema and its frozen runs
keep their exact semantics. Each foundation entry declares the stage and
information scope the adapter is allowed to use, because the first batch's
``native_training`` boolean cannot express "frozen pretrained weights whose
only label contact is a train-only context fit".

Registry entries carry no behaviour. The adapter in each model directory owns
its native loading and prediction; ``foundation_availability`` and
``run_foundation_adapter`` import model packages lazily so a missing optional
dependency of one model never blocks another model's import or run.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

FOUNDATION_SCHEMA = "foundation-expansion-20261023-v1"
FOUNDATION_BATCH = "second_20261023"


@dataclass(frozen=True)
class FoundationSpec:
    model_id: str
    module: str                    # adapter package under forecasting/
    entry: str                     # adapter class name, constructed with the context
    output_table: str              # 'Q' (five-quantile) or 'P' (point only)
    resource: str                  # 'cpu' or 'gpu'
    interpreter: str               # model venv python, relative to the project root
    weight_id: str                 # pinned checkpoint identity (repo[/revision])
    # --- declared stage and information scope (frozen before formal runs) ---
    fit_requires_train_labels: bool  # adapter consumes train labels (context/aux fit only)
    selection_use: str               # 'none' or a declared pre-listed use
    calibration_use: bool            # adapter consumes calibration-split labels
    weight_training: bool            # any gradient/boosting update of pretrained weights
    input_scope: str                 # 'full_public' or a restricted, documented scope
    seed_scope: str                  # what the seed can and cannot change
    pretraining_scope: str           # pretrained-data identity statement
    deterministic_inference: bool    # same inputs+seed give bitwise-identical output
    dependencies: tuple[str, ...] = ()  # importable modules required at predict time
    # Same-seed replay tolerance for the seed-reuse verification: 0.0 demands
    # bitwise identity; small nonzero values document GPU kernel jitter that
    # carries no information (recorded in every reuse evidence block).
    reuse_tolerance: float = 0.0

    def scope_declaration(self) -> dict[str, Any]:
        return {
            "fit_requires_train_labels": self.fit_requires_train_labels,
            "selection_use": self.selection_use,
            "calibration_use": self.calibration_use,
            "weight_training": self.weight_training,
            "output_type": "quantile" if self.output_table == "Q" else "point_only",
            "input_scope": self.input_scope,
            "seed_scope": self.seed_scope,
            "pretraining_scope": self.pretraining_scope,
        }


FOUNDATION_MODELS: tuple[FoundationSpec, ...] = (
    FoundationSpec(
        "tabpfn_v3", "forecasting.tabpfn_v3.adapter", "Adapter", "Q", "gpu",
        ".venvs/tabpfn_v3/bin/python", "Prior-Labs/tabpfn_3",
        fit_requires_train_labels=True, selection_use="none", calibration_use=False,
        weight_training=False, input_scope="full_public",
        seed_scope="ensemble_preprocessing_rng_all_common_levels_native_distribution_quantiles",
        pretraining_scope="prior_labs_v3_pretraining_mix_nem_overlap_unaudited",
        deterministic_inference=True, dependencies=("tabpfn",),
    ),
    FoundationSpec(
        "chronos2_zeroshot", "forecasting.chronos2_zeroshot.adapter", "Adapter", "Q", "gpu",
        ".venvs/chronos2/bin/python", "amazon/chronos-2",
        fit_requires_train_labels=False, selection_use="none", calibration_use=False,
        weight_training=False,
        input_scope="restricted_price_target_demand_past_covariate_calendar_past_future",
        seed_scope="no_sampling_native_quantile_head_frozen_weights_deterministic",
        pretraining_scope="chronos2_public_and_synthetic_mix_nem_overlap_unaudited",
        deterministic_inference=True, dependencies=("chronos",), reuse_tolerance=1e-4,
    ),
    FoundationSpec(
        "moirai2_zeroshot", "forecasting.moirai2_zeroshot.adapter", "Adapter", "P", "gpu",
        ".venvs/moirai2/bin/python", "Salesforce/moirai-2.0-R-small",
        fit_requires_train_labels=False, selection_use="none", calibration_use=False,
        weight_training=False,
        input_scope="restricted_price_target_demand_past_covariate_calendar_past_future",
        seed_scope="no_sampling_quantile_head_q10_q90_only_frozen_weights_deterministic",
        pretraining_scope="moirai2_gift_eval_chronos_kernelsynth_mix_nem_overlap_unaudited",
        deterministic_inference=True, dependencies=("uni2ts",), reuse_tolerance=1e-4,
    ),
    FoundationSpec(
        "timesfm25_xreg", "forecasting.timesfm25_xreg.adapter", "Adapter", "P", "gpu",
        ".venvs/timesfm25/bin/python", "google/timesfm-2.5-200m-pytorch",
        fit_requires_train_labels=False, selection_use="none", calibration_use=False,
        weight_training=False,
        input_scope="restricted_price_target_calendar_dynamic_gas_static_incontext_xreg",
        seed_scope="incontext_xreg_least_squares_deterministic_frozen_backbone",
        pretraining_scope="timesfm25_synthetic_plus_real_monolithic_nem_overlap_unaudited",
        deterministic_inference=True, dependencies=("timesfm",), reuse_tolerance=1e-4,
    ),
)

FOUNDATION_REGISTRY: dict[str, FoundationSpec] = {spec.model_id: spec for spec in FOUNDATION_MODELS}


def foundation_model_ids() -> tuple[str, ...]:
    return tuple(FOUNDATION_REGISTRY)


def foundation_availability() -> dict[str, dict[str, Any]]:
    """Implemented/available status for every registered foundation model."""
    result = {}
    for spec in FOUNDATION_MODELS:
        missing = [module for module in spec.dependencies if find_spec(module) is None]
        result[spec.model_id] = {
            "registered_entry": f"{spec.module}::{spec.entry}",
            "output_table": spec.output_table, "resource": spec.resource,
            "interpreter": spec.interpreter, "weight_id": spec.weight_id,
            "missing_dependencies": missing,
            "status": "available" if not missing else "missing_dependency",
            **spec.scope_declaration(),
        }
    return result


def load_foundation_adapter(model_id: str):
    """Import and return the registered adapter class; never imports early."""
    if model_id not in FOUNDATION_REGISTRY:
        raise ValueError(f"Unknown foundation baseline: {model_id}")
    spec = FOUNDATION_REGISTRY[model_id]
    missing = [module for module in spec.dependencies if find_spec(module) is None]
    if missing:
        raise ImportError(f"{model_id} is blocked in this interpreter: missing {missing}")
    import importlib

    module = importlib.import_module(spec.module)
    adapter = getattr(module, spec.entry, None)
    if adapter is None:
        raise NotImplementedError(f"{spec.module} does not provide {spec.entry}")
    return adapter
