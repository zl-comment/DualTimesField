"""Strict raw-price prediction artifacts and shared baseline scoring.

Quantile models are scored at the rearranged q50 for point metrics. Native point
heads are diagnostic only. Point-only models have a separate P table; they never
receive fabricated probability scores. The five-level probability score is a
pinball proxy, not an exact or densely integrated CRPS.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

QUANTILE_LEVELS = np.array([0.05, 0.1, 0.5, 0.9, 0.95], dtype=np.float64)
SCHEMA_VERSION = 1
_ALIGNMENT_KEYS = ("protocol_id", "split", "region", "input_hash", "sample_hash", "preprocessing_hash")
_REQUIRED_KEYS = (*_ALIGNMENT_KEYS, "seed", "source", "output_type", "prediction_space")


@dataclass(frozen=True)
class PredictionArtifact:
    actual: np.ndarray
    origin_unix: np.ndarray
    target_time_unix: np.ndarray
    metadata: dict[str, Any]
    quantiles: np.ndarray | None = None  # Original outputs, before rearrangement.
    native_point: np.ndarray | None = None

    @property
    def table(self) -> str:
        return "P" if self.metadata["output_type"] == "point_only" else "Q"


def _finite_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return np.array(array, dtype=np.float64, copy=True)


def _timestamps(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu" or (array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max)):
        raise ValueError(f"{name} must contain int64 Unix seconds")
    return np.array(array, dtype=np.int64, copy=True)


def make_prediction_artifact(
    *, actual: Any, origin_unix: Any, target_time_unix: Any,
    metadata: Mapping[str, Any], quantiles: Any = None, point: Any = None,
    quantile_levels: Any = None,
) -> PredictionArtifact:
    """Validate one region/seed/split, preserving native predictions for audit.

    Required metadata: protocol_id, split, region, seed, source, input_hash,
    sample_hash, preprocessing_hash, prediction_space='raw_rrp', output_type.
    output_type is 'quantile', 'point_only', or explicitly 'degenerate_quantile'.
    The hashes must come from the shared input builder, not model-specific data.
    """
    meta = dict(metadata)
    missing = set(_REQUIRED_KEYS) - meta.keys()
    if missing:
        raise ValueError(f"Missing metadata: {sorted(missing)}")
    for key in (*_ALIGNMENT_KEYS, "source"):
        if not isinstance(meta[key], str) or not meta[key].strip():
            raise ValueError(f"metadata.{key} must be a nonempty string")
    if isinstance(meta["seed"], bool) or not isinstance(meta["seed"], (int, np.integer)):
        raise ValueError("metadata.seed must be an integer")
    meta["seed"] = int(meta["seed"])
    if meta["prediction_space"] != "raw_rrp":
        raise ValueError("Predictions and actual must be supplied in raw_rrp space")
    if meta["output_type"] not in ("quantile", "point_only", "degenerate_quantile"):
        raise ValueError("Unknown output_type")
    # JSON-only metadata makes persistence independent of pickle/code execution.
    try:
        meta = json.loads(json.dumps(meta, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be finite JSON data") from exc
    y = _finite_array(actual, "actual")
    origins = _timestamps(origin_unix, "origin_unix")
    targets = _timestamps(target_time_unix, "target_time_unix")
    if y.ndim != 2 or 0 in y.shape:
        raise ValueError("actual must have nonempty shape [N,H]")
    if origins.shape != (y.shape[0],) or targets.shape != y.shape:
        raise ValueError("origin_unix [N] and target_time_unix [N,H] must match actual")
    if np.any(np.diff(origins) <= 0):
        raise ValueError("origin_unix must be strictly increasing and unique")
    if np.any(np.diff(targets, axis=1) <= 0) or np.any(targets[:, 0] < origins):
        raise ValueError("target_time_unix must increase within each horizon and not precede origin")
    native = None if point is None else _finite_array(point, "point")
    if native is not None and native.shape != y.shape:
        raise ValueError("point must have shape [N,H]")
    q = None
    if meta["output_type"] == "point_only":
        if native is None or quantiles is not None or quantile_levels is not None:
            raise ValueError("point_only requires point and forbids quantiles/quantile_levels")
    else:
        if quantiles is None or quantile_levels is None:
            raise ValueError("Quantile models require quantiles and explicit quantile_levels")
        levels = _finite_array(quantile_levels, "quantile_levels")
        if not np.array_equal(levels, QUANTILE_LEVELS):
            raise ValueError("quantile_levels must equal [0.05,0.1,0.5,0.9,0.95] in order")
        q = _finite_array(quantiles, "quantiles")
        if q.shape != (*y.shape, len(QUANTILE_LEVELS)):
            raise ValueError("quantiles must have shape [N,H,5]")
        repeated = bool(np.all(q == q[..., :1]))
        if repeated and meta["output_type"] != "degenerate_quantile":
            raise ValueError("Repeated point values require an explicitly declared degenerate_quantile distribution")
        if meta["output_type"] == "degenerate_quantile" and not repeated:
            raise ValueError("degenerate_quantile must be constant across quantile levels")
    return PredictionArtifact(y, origins, targets, meta, q, native)


def _validated(artifact: PredictionArtifact) -> PredictionArtifact:
    return make_prediction_artifact(
        actual=artifact.actual, origin_unix=artifact.origin_unix,
        target_time_unix=artifact.target_time_unix, metadata=artifact.metadata,
        quantiles=artifact.quantiles, point=artifact.native_point,
        quantile_levels=QUANTILE_LEVELS if artifact.quantiles is not None else None,
    )


def save_prediction_artifact(path: str | Path, artifact: PredictionArtifact) -> None:
    """Persist validated arrays and the complete provenance contract in one NPZ."""
    artifact = _validated(artifact)
    arrays = dict(actual=artifact.actual, origin_unix=artifact.origin_unix,
                  target_time_unix=artifact.target_time_unix,
                  metadata_json=np.asarray(json.dumps(artifact.metadata, sort_keys=True, allow_nan=False)),
                  schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64))
    if artifact.quantiles is not None:
        arrays.update(quantiles=artifact.quantiles, quantile_levels=QUANTILE_LEVELS)
    if artifact.native_point is not None:
        arrays["native_point"] = artifact.native_point
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def load_prediction_artifact(path: str | Path) -> PredictionArtifact:
    with np.load(path, allow_pickle=False) as archive:
        required = {"schema_version", "metadata_json", "actual", "origin_unix", "target_time_unix"}
        if not required.issubset(archive.files):
            raise ValueError("Archive lacks the aligned prediction schema")
        if archive["schema_version"].shape != () or archive["schema_version"].item() != SCHEMA_VERSION:
            raise ValueError("Unsupported prediction schema version")
        try:
            metadata = json.loads(archive["metadata_json"].item())
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid metadata_json") from exc
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must be an object")
        return make_prediction_artifact(
            actual=archive["actual"], origin_unix=archive["origin_unix"],
            target_time_unix=archive["target_time_unix"], metadata=metadata,
            quantiles=archive["quantiles"] if "quantiles" in archive else None,
            point=archive["native_point"] if "native_point" in archive else None,
            quantile_levels=archive["quantile_levels"] if "quantile_levels" in archive else None,
        )


def assert_comparable(left: PredictionArtifact, right: PredictionArtifact, *, allow_different_seed: bool = True) -> None:
    """Reject mismatched inputs, samples, labels, scoring tracks or experiment cells."""
    left, right = _validated(left), _validated(right)
    for key in (*_ALIGNMENT_KEYS, "prediction_space") + (() if allow_different_seed else ("seed",)):
        if left.metadata[key] != right.metadata[key]:
            raise ValueError(f"Not comparable: metadata.{key} differs")
    if left.table != right.table:
        raise ValueError("Not comparable: point_only and quantile models belong to separate tables")
    for key in ("origin_unix", "target_time_unix", "actual"):
        if not np.array_equal(getattr(left, key), getattr(right, key)):
            raise ValueError(f"Not comparable: {key} differs")


def _point_metrics(point: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    error = point - actual
    return {"mae": float(np.abs(error).mean()), "rmse_pooled": float(np.sqrt(np.square(error).mean()))}


def score_predictions(artifact: PredictionArtifact) -> dict[str, Any]:
    """Equal weight for all origin/horizon cells; no clipping or tail reweighting."""
    artifact = _validated(artifact)
    result: dict[str, Any] = {"table": artifact.table, "n_origins": len(artifact.actual),
                              "horizon": artifact.actual.shape[1], "n_cells": artifact.actual.size}
    if artifact.quantiles is None:
        result.update(_point_metrics(artifact.native_point, artifact.actual))
        result["point_definition"] = "native_point"
        return result
    raw = artifact.quantiles
    quantiles = np.sort(raw, axis=-1)
    result["crossing_rate"] = float(np.any(np.diff(raw, axis=-1) < 0, axis=-1).mean())
    result["point_definition"] = "rearranged_q50"
    result.update(_point_metrics(quantiles[..., 2], artifact.actual))
    error = artifact.actual[..., None] - quantiles
    pinball = np.maximum(QUANTILE_LEVELS * error, (QUANTILE_LEVELS - 1) * error)
    result["pinball_by_quantile"] = {str(float(level)): float(pinball[..., i].mean())
                                      for i, level in enumerate(QUANTILE_LEVELS)}
    result["quantile_score_proxy"] = float(2 * pinball.mean())
    for coverage, low, high in ((80, 1, 3), (90, 0, 4)):
        result[f"coverage_{coverage}"] = float(((artifact.actual >= quantiles[..., low]) &
                                               (artifact.actual <= quantiles[..., high])).mean())
        result[f"interval_width_{coverage}"] = float((quantiles[..., high] - quantiles[..., low]).mean())
    if artifact.native_point is not None:
        result["native_point_diagnostic"] = _point_metrics(artifact.native_point, artifact.actual)
    return result


def aggregate_seed_scores(artifacts: Sequence[PredictionArtifact]) -> dict[str, Any]:
    """Average independently scored runs, never average forecasts before scoring."""
    if not artifacts:
        raise ValueError("At least one seed artifact is required")
    reference = artifacts[0]
    seeds = [artifact.metadata["seed"] for artifact in artifacts]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed runs must have distinct seed identifiers")
    for artifact in artifacts:
        assert_comparable(reference, artifact, allow_different_seed=True)
        if artifact.metadata["source"] != reference.metadata["source"]:
            raise ValueError("Seed aggregation requires the same model source")
    scores = [score_predictions(artifact) for artifact in artifacts]

    def aggregate(values: list[Any]) -> Any:
        if isinstance(values[0], dict):
            common = set.intersection(*(set(value) for value in values))
            return {key: aggregate([value[key] for value in values]) for key in sorted(common)}
        if isinstance(values[0], (int, float)):
            return {"mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "n": len(values)}
        if any(value != values[0] for value in values):
            raise ValueError("Seed score definitions differ")
        return values[0]

    return {"per_seed": {str(seed): score for seed, score in zip(seeds, scores)},
            "seed_summary": aggregate(scores), "aggregation": "mean_of_seed_metrics"}
