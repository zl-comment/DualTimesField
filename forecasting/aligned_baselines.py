"""Trainable baselines using the common, preprocessed input contract.

These are T1 adapters, not reproductions of the paper's day-ahead LEAR recipe.
No fit or selection operation uses evaluation labels. Foundation models are
registered as pending until official weights and their input adapters are tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
from importlib.util import find_spec
from importlib.metadata import version
from typing import Any

import numpy as np

from .lasso.model import absolute_residual_cp

DEFAULT_QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)


@dataclass
class BaselinePrediction:
    point: np.ndarray
    quantiles: np.ndarray | None
    metadata: dict[str, Any]


FOUNDATION_BASELINES = {
    "tabpfn2": {"module": "tabpfn", "source": "https://github.com/PriorLabs/TabPFN",
                "source_commit": "8eea89ee142a3bdb868ae150baefaaa9d9065b96",
                "blocker": "Version-specific official weights and shared-input adapter are not validated."},
    "tabpfn3": {"module": "tabpfn", "source": "https://github.com/PriorLabs/TabPFN",
                "source_commit": "8eea89ee142a3bdb868ae150baefaaa9d9065b96",
                "blocker": "Version-specific official weights and shared-input adapter are not validated."},
    "chronos2": {"module": "chronos", "source": "https://github.com/amazon-science/chronos-forecasting",
                "source_commit": "1224fafa91e1ecd6674f5204066b159e40b5e4a1",
                "blocker": "Official weights and complete shared-covariate adapter are not validated."},
    "moirai2": {"module": "uni2ts", "source": "https://github.com/SalesforceAIResearch/uni2ts",
                "source_commit": "cfd46d4510ed8896f263116f32928eede05b0a75",
                "blocker": "Official weights/input adapter not validated; default quantile head lacks q05/q95. No tail extrapolation is permitted."},
}


def baseline_availability() -> dict[str, dict[str, Any]]:
    result = {}
    for name, module in (("xgboost", "xgboost"), ("lightgbm", "lightgbm"),
                         ("lasso_arx72_cp", "sklearn")):
        installed = find_spec(module) is not None
        result[name] = {"implemented": True, "dependency_installed": installed,
                        "status": "available" if installed else "missing_dependency"}
    for name, info in FOUNDATION_BASELINES.items():
        result[name] = dict(info, implemented=False, dependency_installed=find_spec(info["module"]) is not None,
                            status="pending_adapter_and_weight_validation")
    return result


def _levels(values: Any) -> np.ndarray:
    q = np.asarray(values, dtype=float)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or not np.all(np.diff(q) > 0) or q[0] <= 0 or q[-1] >= 1:
        raise ValueError("Quantile levels must be strictly increasing within (0, 1)")
    if np.count_nonzero(q == 0.5) != 1:
        raise ValueError("A q50 level is required for the common point prediction")
    return q


def _schema(split: Any) -> tuple:
    return tuple((key, np.asarray(split.inputs[key]).shape[1:]) for key in sorted(split.inputs))


def _strictly_before(left: Any, right: Any, label: str) -> None:
    targets, origins = np.asarray(left.target_time_unix), np.asarray(right.origin_unix)
    if not targets.size or not origins.size or not np.isfinite(targets).all() or not np.isfinite(origins).all():
        raise ValueError(f"{label}: missing or invalid times")
    if np.max(targets) + 3600 > np.min(origins):
        raise ValueError(f"{label}: all earlier targets must be realized before the next origin; purge overlapping windows")


def _scale(split: Any) -> tuple[float, float]:
    mean, std = float(split.metadata["target_mean"]), float(split.metadata["target_std"])
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid shared target scaler")
    return mean, std


def _validate_splits(train: Any, selection: Any, evaluation: Any, calibration: Any | None) -> None:
    pieces = [train, selection] + ([calibration] if calibration is not None else []) + [evaluation]
    for split in pieces:
        for key in ("protocol_id", "region", "preprocessing_hash", "input_hash", "sample_hash"):
            value = split.metadata.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing required aligned metadata: {key}")
        for key in ("protocol_id", "region", "preprocessing_hash"):
            if split.metadata[key] != train.metadata[key]:
                raise ValueError(f"Splits must share common {key} metadata")
        if _schema(split) != _schema(train) or _scale(split) != _scale(train):
            raise ValueError("Splits must share input schema and the training-only target scaler")
        n = len(split.origin_unix)
        if not n or np.asarray(split.target_time_unix).shape != (n, train.target.shape[1]):
            raise ValueError("Split target timestamps must match [origin, horizon]")
        if any(np.asarray(x).shape[0] != n or not np.isfinite(x).all() for x in split.inputs.values()):
            raise ValueError("Input blocks must contain finite aligned rows")
    # Evaluation target/actual values are intentionally never accessed here.
    for split in (train, selection):
        if split.target.shape != split.target_time_unix.shape or not np.isfinite(split.target).all():
            raise ValueError("Fit and selection targets must be finite and aligned")
    for index in range(len(pieces) - 1):
        _strictly_before(pieces[index], pieces[index + 1], "split chronology")


def _loss(prediction: np.ndarray, target: np.ndarray, q: np.ndarray, metric: str) -> float:
    if prediction.ndim == 2:
        return float(np.mean(np.abs(prediction - target)))
    # Sorting is the common quantile-crossing correction, inside the adapter.
    prediction = np.sort(prediction, axis=-1)
    if metric == "mae":
        return float(np.mean(np.abs(prediction[..., np.flatnonzero(q == 0.5)[0]] - target)))
    residual = target[..., None] - prediction
    return float(np.maximum(q * residual, (q - 1) * residual).mean())


class _RunArtifacts:
    """Atomic progress and explicit model files; no evaluation labels are saved."""
    def __init__(self, progress_path=None, model_directory=None):
        self.progress_path = Path(progress_path) if progress_path is not None else None
        self.directory = Path(model_directory) if model_directory is not None else None
        self.started = time.monotonic()
        self.files = []
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            if (self.directory / "manifest.json").exists():
                raise FileExistsError("Completed model manifest already exists")

    @staticmethod
    def write_json(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def progress(self, stage, **details):
        if self.progress_path is not None:
            self.write_json(self.progress_path, dict(stage=stage, elapsed_seconds=time.monotonic() - self.started,
                                                     **details))

    def save(self, name, model, kind, **details):
        if self.directory is None:
            return
        path = self.directory / name
        if path.exists():
            raise FileExistsError(f"Model artifact already exists: {path}")
        if kind == "joblib":
            import joblib
            joblib.dump(model, path)
        elif kind == "numpy":
            np.savez_compressed(path, **model)
        else:
            model.save_model(path)
        self.files.append(dict(path=name, format=kind, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **details))

    def finish(self, metadata, train, selection, calibration):
        if self.directory is not None:
            manifest = dict(model_metadata=metadata, files=self.files,
                            input_block_order=list(train.metadata.get("input_block_order", train.inputs.keys())),
                            target_mean=train.metadata["target_mean"], target_std=train.metadata["target_std"],
                            train_metadata=train.metadata, selection_metadata=selection.metadata,
                            calibration_metadata=calibration.metadata if calibration is not None else None,
                            load_warning="Only load joblib files generated by a trusted experiment.")
            # Use the actual canonical flatten order even when the caller built a dict in another order.
            from .aligned_inputs import INPUT_KEYS
            manifest["input_block_order"] = list(INPUT_KEYS)
            self.write_json(self.directory / "manifest.json", manifest)
            metadata["model_manifest"] = str(self.directory / "manifest.json")
        self.progress("complete", selected_rounds=metadata.get("selected_rounds"),
                      selected_alpha=metadata.get("selected_alpha"), files_saved=len(self.files))


def _fit_native_trees(kind, x_train, y_train, x_select, y_select, x_eval, *,
                      q, task, metric, seed, max_rounds, interval, patience,
                      learning_rate, n_jobs, parameters, artifacts):
    """Use native incremental validation caches, avoiding repeated prefix prediction."""
    from .xgboost.model import fit_horizon as fit_xgboost
    from .lightgbm.model import fit_horizon as fit_lightgbm

    fit = fit_xgboost if kind == "xgboost" else fit_lightgbm
    predictions, chosen, fitted, traces = [], [], [], []
    horizon_count = y_train.shape[1]
    for h in range(horizon_count):
        artifacts.progress("fit_horizon", horizon=h, horizon_count=horizon_count)
        values, best_round, actual_rounds, trace = fit(
            h, x_train, y_train, x_select, y_select, x_eval,
            q=q, task=task, metric=metric, seed=seed, max_rounds=max_rounds,
            interval=interval, patience=patience, learning_rate=learning_rate,
            n_jobs=n_jobs, parameters=parameters, artifacts=artifacts)
        predictions.append(values)
        chosen.append(best_round)
        fitted.append(actual_rounds)
        traces.append(trace)
        artifacts.progress("horizon_complete", horizon=h, horizon_count=horizon_count,
                           selected_rounds=best_round, fitted_rounds=actual_rounds)
    return np.stack(predictions, axis=1), chosen, fitted, traces


def fit_predict(kind: str, train: Any, selection: Any, evaluation: Any, *,
                calibration: Any | None = None, quantile_levels: Any = DEFAULT_QUANTILES,
                seed: int = 2026, params: dict[str, Any] | None = None,
                progress_path: str | Path | None = None, model_directory: str | Path | None = None) -> BaselinePrediction:
    """Fit on train, choose on selection, then predict evaluation exactly once.

    Trees select boosting prefixes independently per horizon by default.
    Set selection_scope="shared" for the explicit shared-prefix variant.
    Native per-horizon early stopping consumes cached validation predictions.
    The optional shared-prefix variant fits all trees before prefix selection.
    Lasso chooses one shared alpha and uses a separate later calibration split.
    """
    if kind in FOUNDATION_BASELINES:
        raise NotImplementedError(FOUNDATION_BASELINES[kind]["blocker"])
    if kind not in {"xgboost", "lightgbm", "lasso_arx72_cp"}:
        raise ValueError(f"Unknown aligned baseline: {kind}")
    from .aligned_inputs import flatten_inputs

    p = dict(params or {})
    q = _levels(quantile_levels)
    task = p.pop("task", "quantile")
    if task not in {"point", "quantile"}:
        raise ValueError("task must be point or quantile")
    metric = p.pop("selection_metric", "mae" if task == "point" or kind == "lasso_arx72_cp" else "pinball")
    if metric not in {"mae", "pinball"} or task == "point" and metric != "mae":
        raise ValueError("selection_metric must be mae, or pinball for quantile trees")
    if kind == "lasso_arx72_cp" and metric != "mae":
        raise ValueError("Lasso alpha selection uses point MAE before independent CP calibration")
    if kind == "lasso_arx72_cp" and calibration is None:
        raise ValueError("Lasso+CP requires an independent chronological calibration split")
    if kind != "lasso_arx72_cp" and calibration is not None:
        raise ValueError("Calibration is only consumed by lasso_arx72_cp")
    _validate_splits(train, selection, evaluation, calibration)
    artifacts = _RunArtifacts(progress_path, model_directory)
    artifacts.progress("prepare_inputs", model=kind)
    x_train, x_select, x_eval = [flatten_inputs(split) for split in (train, selection, evaluation)]
    mean, std = _scale(train)
    metadata = {"model": kind, "task": task, "seed": seed, "input_blocks": sorted(train.inputs),
                "feature_count": x_train.shape[1], "selection_metric": metric,
                "selection_score_space": "shared_standardized_price",
                "prediction_space": "raw_rrp", "target_training_space": "shared_standardized_price",
                "quantile_levels": q.tolist() if task == "quantile" else None,
                "point_definition": "q50_after_common_sort" if task == "quantile" else "l1_point",
                "quantile_crossing_correction": "scoring_layer_sort" if task == "quantile" else None,
                "package_version": version("scikit-learn" if kind == "lasso_arx72_cp" else kind),
                "requested_params": dict(params or {})}
    if kind == "lasso_arx72_cp":
        from .lasso.model import fit_lasso
        return fit_lasso(train, selection, evaluation, calibration,
            x_train=x_train, x_select=x_select, x_eval=x_eval, q=q, task=task,
            metric=metric, p=p, mean=mean, std=std, metadata=metadata, artifacts=artifacts)

    scope = p.pop("selection_scope", "per_horizon")
    if scope not in {"per_horizon", "shared"}:
        raise ValueError("selection_scope must be per_horizon or shared")
    max_rounds = int(p.pop("n_estimators", 200))
    interval = int(p.pop("selection_interval", 1))
    patience = int(p.pop("early_stopping_rounds", 50))
    learning_rate, n_jobs = float(p.pop("learning_rate", 0.05)), int(p.pop("n_jobs", 1))
    if min(max_rounds, interval, patience) < 1 or learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError("Round budgets, learning rate, interval and patience must be positive")
    allowed = ({"max_depth", "min_child_weight", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "device", "max_bin"}
               if kind == "xgboost" else
               {"max_depth", "num_leaves", "min_child_samples", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "max_bin"})
    if set(p) - allowed:
        raise ValueError(f"Unsupported {kind} parameters: {sorted(set(p) - allowed)}")
    if scope == "per_horizon":
        predicted, selected, fitted, trace = _fit_native_trees(
            kind, x_train, train.target, x_select, selection.target, x_eval,
            q=q, task=task, metric=metric, seed=seed, max_rounds=max_rounds,
            interval=interval, patience=patience, learning_rate=learning_rate,
            n_jobs=n_jobs, parameters=p, artifacts=artifacts)
        raw = predicted * std + mean
        quantiles = raw if task == "quantile" else None
        point = np.sort(quantiles, axis=-1)[..., np.flatnonzero(q == 0.5)[0]] if quantiles is not None else raw
        metadata.update(selected_rounds=selected, selection_trace=trace, max_fitted_rounds=max_rounds,
                        actual_fitted_rounds=fitted, selection_scope=scope, shared_rounds_across_horizons=False,
                        selection_interval=interval, early_stopping_rounds=patience,
                        early_stopping_saves_fit_compute=True, validation_cache="native_incremental")
        artifacts.finish(metadata, train, selection, calibration)
        return BaselinePrediction(point, quantiles, metadata)
    common = dict(n_estimators=max_rounds, learning_rate=learning_rate, n_jobs=n_jobs, random_state=seed)
    from .xgboost.model import fit_shared as fit_xgboost
    from .lightgbm.model import fit_shared as fit_lightgbm
    fit = fit_xgboost if kind == "xgboost" else fit_lightgbm
    models, predict = fit(x_train, train, common=common, p=p, q=q, task=task, n_jobs=n_jobs)
    horizon_count = train.target.shape[1]
    count = 1 if scope == "shared" else horizon_count
    best, selected = np.full(count, np.inf), np.ones(count, dtype=int)
    active, trace = np.ones(count, dtype=bool), []
    rounds_to_try = sorted(set([1, max_rounds, *range(interval, max_rounds + 1, interval)]))
    for rounds in rounds_to_try:
        prediction = predict(x_select, rounds)
        scores = np.asarray([_loss(prediction, selection.target, q, metric)] if scope == "shared" else
                            [_loss(prediction[:, h:h + 1], selection.target[:, h:h + 1], q, metric)
                             for h in range(horizon_count)])
        trace.append({"rounds": rounds, "scores": [float(score) if enabled else None
                                                   for score, enabled in zip(scores, active)]})
        improved = active & (scores < best)
        best[improved], selected[improved] = scores[improved], rounds
        active &= (rounds - selected < patience)
        if not active.any():
            break
    selected = int(selected[0]) if scope == "shared" else selected.tolist()
    raw = predict(x_eval, selected) * std + mean
    quantiles = raw if task == "quantile" else None
    point = np.sort(quantiles, axis=-1)[..., np.flatnonzero(q == 0.5)[0]] if quantiles is not None else raw
    metadata.update(selected_rounds=selected, selection_trace=trace, max_fitted_rounds=max_rounds,
                    selection_scope=scope, shared_rounds_across_horizons=scope == "shared", selection_interval=interval,
                    early_stopping_rounds=patience, early_stopping_saves_fit_compute=False)
    for h, model in enumerate(models):
        if kind == "xgboost":
            artifacts.save(f"h{h:02d}.ubj", model, "xgboost", horizon=h, selected_rounds=selected)
        else:
            for index, quantile_model in enumerate(model):
                artifacts.save(f"h{h:02d}_q{index:02d}.txt", quantile_model.booster_, "lightgbm",
                               horizon=h, selected_rounds=selected, quantile=float(q[index]) if task == "quantile" else None)
    artifacts.finish(metadata, train, selection, calibration)
    return BaselinePrediction(point, quantiles, metadata)
