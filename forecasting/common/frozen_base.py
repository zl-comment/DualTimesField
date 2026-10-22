"""Read-only access to a frozen tree baseline of the parent aligned run.

Loads the boosters and per-horizon selected boosting rounds that the parent
90-cell run persisted, regenerates out-of-sample predictions for any shared
split, and enforces the reproduction gate: regenerated test predictions must
match the parent's frozen test.npz quantiles exactly (tolerance 0) before any
calibration consumer may use them. The parent run is never written to.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..aligned_inputs import flatten_inputs
from ..aligned_scoring import load_prediction_artifact

SUPPORTED_BASES = ("lightgbm", "xgboost")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenTreeBase:
    """One frozen (model, region, seed) cell of the parent run, read-only."""

    def __init__(self, parent_run: Path, model: str, region: str, seed: int):
        if model not in SUPPORTED_BASES:
            raise ValueError(f"Unsupported frozen base model: {model}")
        self.parent_run = Path(parent_run).resolve()
        self.model, self.region, self.seed = model, region, seed
        self.cell = self.parent_run / "runs" / model / f"seed{seed}" / region
        status_path = self.cell / "status.json"
        if not status_path.exists():
            raise FileNotFoundError(f"Missing frozen base cell: {self.cell}")
        status = json.loads(status_path.read_text())
        if status.get("state") != "completed":
            raise ValueError(f"Frozen base cell is not completed: {self.cell}")
        manifest_path = self.cell / "models" / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        metadata = self.manifest["model_metadata"]
        if metadata.get("model") != model or metadata.get("seed") != seed:
            raise ValueError("Frozen base manifest identity mismatch")
        if metadata.get("prediction_space") != "raw_rrp":
            raise ValueError("Frozen base is not in the shared raw price space")
        self.selected_rounds = [int(r) for r in metadata["selected_rounds"]]
        self.horizons = len(self.selected_rounds)
        self.artifact = load_prediction_artifact(self.cell / "test.npz")
        levels = self.artifact.metadata["model_metadata"].get("quantile_levels") \
            if "model_metadata" in self.artifact.metadata else None
        self.quantile_levels = [float(v) for v in (levels or (0.05, 0.1, 0.5, 0.9, 0.95))]
        if len(self.quantile_levels) != self.artifact.quantiles.shape[-1]:
            raise ValueError("Frozen base quantile level count does not match its artifact")
        self.package_version = metadata.get("package_version")
        self.model_manifest_sha256 = _file_sha256(manifest_path)
        self.booster_sha256: dict[str, str] = {}
        for horizon in range(self.horizons):
            for name in self._booster_names(horizon):
                path = self.cell / "models" / name
                if not path.is_file():
                    raise FileNotFoundError(f"Missing frozen booster: {path}")
                self.booster_sha256[name] = _file_sha256(path)

    def _booster_names(self, horizon: int) -> tuple[str, ...]:
        if self.model == "lightgbm":
            return tuple(f"h{horizon:02d}_q{index:02d}.txt"
                         for index in range(len(self.quantile_levels)))
        return (f"h{horizon:02d}.ubj",)

    def _predict_horizon(self, horizon: int, x: np.ndarray) -> np.ndarray:
        if self.model == "lightgbm":
            import lightgbm as lgb

            columns = []
            for name in self._booster_names(horizon):
                booster = lgb.Booster(model_file=str(self.cell / "models" / name))
                columns.append(booster.predict(x, num_iteration=self.selected_rounds[horizon]))
            return np.column_stack(columns)
        import xgboost as xgb

        booster = xgb.Booster(model_file=str(self.cell / "models" / self._booster_names(horizon)[0]))
        values = booster.predict(xgb.DMatrix(x), iteration_range=(0, self.selected_rounds[horizon]))
        # One multi-quantile booster: predict already returns [N, Q]. Keep the
        # native dtype: the frozen adapters standardized in float32 (NEP 50),
        # so any float64 cast here would break bitwise reproduction.
        return np.asarray(values).reshape(len(x), len(self.quantile_levels))

    def predict_split(self, split: Any) -> np.ndarray:
        """Regenerate this frozen model's raw-price quantile predictions [N, H, Q].

        Unsorted, exactly as the frozen adapter produced them; the shared
        scoring layer performs the common crossing correction.
        """
        x = flatten_inputs(split)
        standardized = np.stack([self._predict_horizon(h, x) for h in range(self.horizons)], axis=1)
        mean, std = split.metadata["target_mean"], split.metadata["target_std"]
        return standardized * std + mean

    def verify_test_reproduction(self, test_split: Any, *, atol: float = 0.0) -> float:
        """Hard gate: regenerated test predictions must equal the frozen artifact."""
        regenerated = self.predict_split(test_split)
        reference = self.artifact.quantiles
        if regenerated.shape != reference.shape:
            raise ValueError("Frozen base reproduction shape mismatch")
        deviation = float(np.abs(regenerated - reference).max())
        if not np.isfinite(deviation) or deviation > atol:
            raise ValueError(
                f"Frozen {self.model}/{self.region}/seed{self.seed} test reproduction "
                f"deviates by {deviation}; refusing to trust regenerated calibration predictions")
        return deviation

    def provenance(self) -> dict[str, Any]:
        return {
            "parent_run": str(self.parent_run),
            "base_model": self.model, "region": self.region, "seed": self.seed,
            "selected_rounds": self.selected_rounds,
            "package_version": self.package_version,
            "model_manifest_sha256": self.model_manifest_sha256,
            "booster_count": len(self.booster_sha256),
            "booster_sha256": dict(sorted(self.booster_sha256.items())),
            "test_npz_origin_count": int(len(self.artifact.origin_unix)),
            "reproduction_gate": "regenerated_test_equals_frozen_npz_atol_0",
        }


def _expansion_artifacts(model_id: str, context: Any) -> "_ExpansionArtifacts":
    return _ExpansionArtifacts(context.model_directory, context.progress_path, model_id)


class _ExpansionArtifacts:
    """Minimal progress/model-file writer mirroring the aligned run conventions."""

    def __init__(self, directory: Path, progress_path: Path, model_id: str):
        self.directory, self.progress_path, self.model_id = Path(directory), Path(progress_path), model_id
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "manifest.json").exists():
            raise FileExistsError("Completed model manifest already exists")
        self.files: list[dict[str, Any]] = []

    def progress(self, stage: str, **details: Any) -> None:
        payload = {"model": self.model_id, "stage": stage, **details}
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.progress_path.with_suffix(self.progress_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, allow_nan=False) + "\n")
        temporary.replace(self.progress_path)

    def save_npz(self, name: str, arrays: dict[str, Any], **details: Any) -> None:
        path = self.directory / name
        if path.exists():
            raise FileExistsError(f"Model artifact already exists: {path}")
        np.savez_compressed(path, **arrays)
        self.files.append(dict(path=name, format="numpy",
                               sha256=_file_sha256(path), **details))

    def finish(self, metadata: dict[str, Any], split_metadata: dict[str, Any]) -> None:
        manifest = dict(model_metadata=metadata, files=self.files, split_metadata=split_metadata,
                        load_warning="Only trust artifacts generated by the audited expansion run.")
        temporary = self.directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.directory / "manifest.json")


def _load_frozen_base_and_verify(context: Any, base_model: str) -> "FrozenTreeBase":
    base = FrozenTreeBase(context.parent_run, base_model, context.region, context.seed)
    deviation = base.verify_test_reproduction(context.splits["test"])
    if deviation != 0.0:
        raise ValueError(f"Unexpected nonzero reproduction deviation: {deviation}")
    return base


def conformal_expansion(context: Any, *, base_model: str, expected_params: dict[str, Any]) -> Any:
    """Shared driver for lightgbm_cp / xgboost_cp: symmetric CP on a frozen tree.

    The centre is the frozen base's rearranged q50 and remains exactly equal to
    it; only interval widths are calibrated per horizon on the 2022H2
    out-of-sample absolute residuals. This scores uncertainty calibration, not
    a centre correction.
    """
    from ..aligned_baselines import BaselinePrediction
    from ..aligned_scoring import QUANTILE_LEVELS
    from .calibration import absolute_residual_cp

    if set(context.splits) != {"calibration", "test"}:
        raise ValueError("CP expansion requires exactly the calibration and test stages")
    params = dict(context.config["experiment"]["baseline_parameters"][context.job["model"]])
    if params != expected_params:
        raise ValueError(f"Unexpected {context.job['model']} parameters: {params}")
    artifacts = _expansion_artifacts(context.job["model"], context)
    artifacts.progress("load_frozen_base", base_model=base_model)
    base = _load_frozen_base_and_verify(context, base_model)
    calibration = context.splits["calibration"]
    artifacts.progress("predict_calibration", origins=len(calibration.origin_unix))
    cal_quantiles = np.sort(base.predict_split(calibration), axis=-1)
    test_quantiles = np.sort(base.predict_split(context.splits["test"]), axis=-1)
    cal_q50, test_q50 = cal_quantiles[..., 2], test_quantiles[..., 2]
    residual = np.abs(calibration.actual - cal_q50)
    quantiles = absolute_residual_cp(test_q50, cal_q50, calibration.actual, tuple(QUANTILE_LEVELS))
    point = test_q50.copy()
    if not np.isclose(quantiles[..., 2], point, rtol=0.0, atol=0.0).all():
        raise ValueError("CP centre must equal the frozen base q50 exactly")
    artifacts.save_npz("cp_calibration.npz",
                       dict(absolute_residual=residual,
                            origin_unix=calibration.origin_unix,
                            target_time_unix=calibration.target_time_unix,
                            widths=np.quantile(residual, np.abs(2 * np.asarray(QUANTILE_LEVELS) - 1),
                                               axis=0, method="linear").T),
                       content="calibration residuals and per-horizon type-7 widths")
    metadata = {
        "model": context.job["model"], "task": "quantile", "seed": context.seed,
        "base_model": base_model, "base_provenance": base.provenance(),
        "test_reproduction_max_abs_deviation": 0.0,
        "calibration": "independent_oos_absolute_residual_type7_per_horizon",
        "calibration_origins": int(len(calibration.origin_unix)),
        "calibration_first_origin": int(calibration.origin_unix.min()),
        "calibration_last_target": int(calibration.target_time_unix.max()),
        "postforecasts_source_commit": "589f9407d375e2272fb731873443f697c0060d85",
        "centre_definition": "frozen_base_rearranged_q50_unchanged",
        "prediction_space": "raw_rrp", "point_definition": "cp_centre_equals_base_q50",
        "quantile_levels": list(QUANTILE_LEVELS),
        "package_versions": {"numpy": np.__version__, base_model: base.package_version},
        "requested_params": params,
        "centre_correction": "none_this_is_uncertainty_calibration_only",
    }
    artifacts.finish(metadata, dict(calibration=calibration.metadata, test=context.splits["test"].metadata))
    return BaselinePrediction(point, quantiles, metadata)


def qra_expansion(context: Any, *, base_model: str, expected_params: dict[str, Any]) -> Any:
    """Shared driver for lightgbm_qra: QRA on a frozen tree's five quantiles.

    First-version local adaptation: unregularized exact-LP linear quantile
    regression per horizon and level, features are the frozen base's sorted
    five quantile forecasts plus intercept, fitted on 2022H2 labels only.
    """
    from ..aligned_baselines import BaselinePrediction
    from ..aligned_scoring import QUANTILE_LEVELS
    from .calibration import apply_qra, fit_qra

    if set(context.splits) != {"calibration", "test"}:
        raise ValueError("QRA expansion requires exactly the calibration and test stages")
    params = dict(context.config["experiment"]["baseline_parameters"][context.job["model"]])
    if params != expected_params:
        raise ValueError(f"Unexpected {context.job['model']} parameters: {params}")
    artifacts = _expansion_artifacts(context.job["model"], context)
    artifacts.progress("load_frozen_base", base_model=base_model)
    base = _load_frozen_base_and_verify(context, base_model)
    calibration = context.splits["calibration"]
    artifacts.progress("fit_qra", origins=len(calibration.origin_unix))
    cal_features = np.sort(base.predict_split(calibration), axis=-1)
    test_features = np.sort(base.predict_split(context.splits["test"]), axis=-1)
    fit = fit_qra(cal_features, calibration.actual, tuple(QUANTILE_LEVELS),
                  alpha=params["qra_alpha"], solver=params["qra_solver"])
    quantiles = apply_qra(fit, test_features)
    point = np.sort(quantiles, axis=-1)[..., 2].copy()
    artifacts.save_npz("qra_fit.npz",
                       dict(coefficients=fit["coefficients"],
                            feature_order=np.array(fit["feature_order"]),
                            quantile_levels=np.array(fit["quantile_levels"]),
                            calibration_origin_unix=calibration.origin_unix,
                            calibration_target_time_unix=calibration.target_time_unix),
                       content="per-horizon per-level QRA coefficients and fit scope",
                       alpha=fit["alpha"], solver=fit["solver"], calibration_rows=fit["calibration_rows"])
    metadata = {
        "model": context.job["model"], "task": "quantile", "seed": context.seed,
        "base_model": base_model, "base_provenance": base.provenance(),
        "test_reproduction_max_abs_deviation": 0.0,
        "calibration": "linear_quantile_regression_averaging_on_base_five_quantiles",
        "qra_alpha": fit["alpha"], "qra_solver": fit["solver"],
        "qra_feature_order": fit["feature_order"], "qra_calibration_rows": fit["calibration_rows"],
        "calibration_origins": int(len(calibration.origin_unix)),
        "calibration_first_origin": int(calibration.origin_unix.min()),
        "calibration_last_target": int(calibration.target_time_unix.max()),
        "prediction_space": "raw_rrp", "point_definition": "rearranged_q50_of_qra_output",
        "quantile_levels": list(QUANTILE_LEVELS),
        "package_versions": {"numpy": np.__version__, base_model: base.package_version},
        "requested_params": params,
        "paper_claim": "local QRA adaptation, not a full reproduction of any paper's QRA design",
    }
    artifacts.finish(metadata, dict(calibration=calibration.metadata, test=context.splits["test"].metadata))
    return BaselinePrediction(point, quantiles, metadata)
