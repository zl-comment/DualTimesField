"""Verify complete aligned experiment artifacts before publishing comparisons.

A partial matrix produces progress only. Scores are recomputed from the stored
raw-price artifacts; metrics.json is a consistency check, never the authority.
Seed summaries average independently scored runs, not predictions. Identical
seed predictions are reported explicitly, without claiming independent evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .aligned_scoring import (
    PredictionArtifact, aggregate_seed_scores, assert_comparable,
    load_prediction_artifact, score_predictions,
)

REGIONS = ("NSW1", "QLD1", "TAS1")
SEEDS = (2026, 2027, 2028)
_ALIGNMENT_KEYS = ("protocol_id", "region", "split", "input_hash", "sample_hash", "preprocessing_hash")
_MAIN_METRICS = ("mae", "rmse_pooled", "quantile_score_proxy", "crossing_rate",
                 "coverage_80", "coverage_90", "interval_width_80", "interval_width_90")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read valid JSON: {path}") from exc


def _within(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Manifest and artifact paths must be nonempty relative paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes its run directory: {relative}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"Missing completed-job artifact: {path}") from exc
    return digest.hexdigest()


def _verify_snapshot(root: Path, manifest: dict) -> str:
    files = manifest.get("source_files")
    if files is None:
        return "not_declared"
    if not isinstance(files, dict) or not files:
        raise ValueError("source_files must declare the frozen snapshot")
    source_root = _within(root, "source")
    for name, expected in files.items():
        if not isinstance(expected, str) or _sha256(_within(source_root, name)) != expected:
            raise ValueError(f"Frozen source hash mismatch: {name}")
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if digest != manifest.get("source_hash"):
        raise ValueError("Frozen source manifest fingerprint mismatch")
    if _sha256(root / "config.yaml") != manifest.get("config_hash"):
        raise ValueError("Frozen configuration hash mismatch")
    return "verified"


def _same_score(expected: Any, saved: Any, path: str = "metrics") -> None:
    if isinstance(expected, dict):
        if not isinstance(saved, dict) or set(expected) != set(saved):
            raise ValueError(f"{path} keys differ from the shared scorer")
        for key in expected:
            _same_score(expected[key], saved[key], f"{path}.{key}")
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if isinstance(saved, bool) or not isinstance(saved, (float, int)) or not np.isfinite(saved):
            raise ValueError(f"{path} must be a finite score")
        if not np.isclose(expected, saved, rtol=1e-12, atol=1e-12):
            raise ValueError(f"{path} disagrees with the shared scorer")
    elif expected != saved:
        raise ValueError(f"{path} disagrees with the shared scorer")


def _assert_same_samples(left: PredictionArtifact, right: PredictionArtifact) -> None:
    # Data equality applies even to the deliberately separate P and Q tables.
    for key in _ALIGNMENT_KEYS:
        if left.metadata[key] != right.metadata[key]:
            raise ValueError(f"Completed artifacts have different metadata.{key}")
    for key in ("actual", "origin_unix", "target_time_unix"):
        if not np.array_equal(getattr(left, key), getattr(right, key)):
            raise ValueError(f"Completed artifacts have different {key}")
    if left.table == right.table:
        assert_comparable(left, right, allow_different_seed=True)


def _load_completed(root: Path, job: dict, status: dict, manifest: dict) -> tuple[PredictionArtifact, dict]:
    directory = _within(root, job["output"])
    for key in ("source_hash", "config_hash"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"Manifest must declare {key}")
        if status.get(key) != manifest[key]:
            raise ValueError(f"{job['id']}: status {key} differs from manifest")
    if status.get("job_id") != job["id"]:
        raise ValueError(f"{job['id']}: status job_id differs from manifest")
    hashes = status.get("artifact_hashes")
    if not isinstance(hashes, dict) or "test.npz" not in hashes:
        raise ValueError(f"{job['id']}: completed status must hash test.npz")
    for name, expected in hashes.items():
        if not isinstance(expected, str) or _sha256(_within(directory, name)) != expected:
            raise ValueError(f"{job['id']}: artifact hash mismatch for {name}")
    artifact = load_prediction_artifact(directory / "test.npz")
    metadata = artifact.metadata
    for key in ("model", "region", "seed"):
        if metadata.get(key) != job[key]:
            raise ValueError(f"{job['id']}: artifact {key} differs from manifest")
    if metadata["split"] != "test" or metadata.get("smoke", False):
        raise ValueError(f"{job['id']}: only non-smoke test artifacts can enter this report")
    if metadata["source"] != f"{job['model']}:{manifest['source_hash']}":
        raise ValueError(f"{job['id']}: artifact source differs from manifest")
    config_hash = metadata.get("config_sha256", metadata.get("config_hash"))
    if config_hash != manifest["config_hash"]:
        raise ValueError(f"{job['id']}: artifact configuration differs from manifest")
    scores = score_predictions(artifact)
    stored = _read_json(directory / "metrics.json")
    stored = stored["scores"] if isinstance(stored, dict) and "scores" in stored else stored
    _same_score(scores, stored)
    return artifact, scores


def _seed_evidence(artifacts: list[PredictionArtifact], model: str) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for artifact in artifacts:
        # Same scored quantile distribution, including differently crossed heads,
        # supplies no extra evidence about the reported metrics.
        values = (np.sort(artifact.quantiles, axis=-1) if artifact.quantiles is not None
                  else artifact.native_point)
        digest = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        groups[digest].append(artifact.metadata["seed"])
    return {"seed_count": len(artifacts), "distinct_scored_prediction_count": len(groups),
            "duplicate_seed_groups": [sorted(seeds) for seeds in groups.values() if len(seeds) > 1],
            "known_deterministic_model": model == "lasso_arx72_cp",
            "interpretation": ("identical_predictions_are_not_independent_replicates" if len(groups) < len(artifacts)
                               else "distinct_predictions_do_not_establish_statistical_independence")}


def build_aligned_report(run_root: str | Path) -> dict:
    """Read jobs.json and validate every completed cell before making summaries.

    jobs.json: {run_id, source_hash, config_hash, jobs:[{id,model,region,seed,output}]}.
    Each output directory contains status.json, test.npz and metrics.json. There
    must be nine cells per model. Until the entire requested matrix completes,
    the return value contains progress and audit status only, never a ranking.
    """
    root = Path(run_root).resolve()
    manifest = _read_json(root / "jobs.json")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("jobs"), list) or not manifest["jobs"]:
        raise ValueError("jobs.json must contain a nonempty jobs list and run provenance")
    snapshot_audit = _verify_snapshot(root, manifest)
    jobs = manifest["jobs"]
    cells, identifiers, directories = set(), set(), set()
    for job in jobs:
        if not isinstance(job, dict) or not {"id", "model", "region", "seed", "output"} <= job.keys():
            raise ValueError("Every manifest job requires id/model/region/seed/output")
        if not all(isinstance(job[key], str) and job[key] for key in ("id", "model", "region")):
            raise ValueError("Job identifiers must be nonempty strings")
        if isinstance(job["seed"], bool) or not isinstance(job["seed"], int) or job["seed"] not in SEEDS or job["region"] not in REGIONS:
            raise ValueError("Jobs must use the fixed three-region, three-seed grid")
        cell = (job["model"], job["region"], job["seed"])
        directory = _within(root, job["output"])
        if cell in cells or job["id"] in identifiers or directory in directories:
            raise ValueError("Duplicate manifest cell, job identifier or output directory")
        cells.add(cell)
        identifiers.add(job["id"])
        directories.add(directory)
    listed_models = {job["model"] for job in jobs}
    declared_models = manifest.get("models", sorted(listed_models))
    if (not isinstance(declared_models, list) or not declared_models
            or not all(isinstance(model, str) and model for model in declared_models)
            or len(set(declared_models)) != len(declared_models)
            or not listed_models <= set(declared_models)):
        raise ValueError("Manifest models must uniquely declare every scheduled model")
    models = sorted(declared_models)
    for key, required in (("regions", REGIONS), ("seeds", SEEDS)):
        if key in manifest and (not isinstance(manifest[key], list)
                                or len(manifest[key]) != len(required)
                                or set(manifest[key]) != set(required)):
            raise ValueError(f"Manifest {key} differs from the fixed evaluation grid")
    expected = {(model, region, seed) for model in models for region in REGIONS for seed in SEEDS}
    missing = sorted(expected - cells)
    statuses, artifacts, cell_reference = [], {}, {}
    counts: Counter = Counter()
    for job in jobs:
        status_path = _within(root, job["output"]) / "status.json"
        status = _read_json(status_path) if status_path.exists() else {"state": "pending"}
        if not isinstance(status, dict) or not isinstance(status.get("state"), str):
            raise ValueError(f"{job['id']}: status must declare a state")
        state = status["state"]
        counts[state] += 1
        row = {key: job[key] for key in ("id", "model", "region", "seed")}
        row["state"] = state
        for key in ("seconds", "elapsed_seconds", "started_at", "completed_at", "ended_at", "error"):
            if key in status:
                row[key] = status[key]
        statuses.append(row)
        if state != "completed":
            continue
        artifact, _ = _load_completed(root, job, status, manifest)
        reference_key = job["region"]
        if reference_key in cell_reference:
            _assert_same_samples(cell_reference[reference_key], artifact)
        else:
            cell_reference[reference_key] = artifact
        artifacts[(job["model"], job["region"], job["seed"])] = artifact
    complete = not missing and len(artifacts) == len(expected)
    report = {"run_id": manifest.get("run_id"), "state": "complete" if complete else "incomplete",
              "stage": "fixed_configuration_comparison_not_tuned_sota",
              "progress": {"expected_cells": len(expected), "listed_cells": len(jobs),
                           "completed_cells": len(artifacts), "completion_fraction": len(artifacts) / len(expected),
                           "status_counts": dict(counts),
                           "missing_manifest_cells": [{"model": m, "region": r, "seed": s} for m, r, s in missing]},
              "jobs": statuses, "completed_artifact_validation": "passed",
              "frozen_snapshot_validation": snapshot_audit,
              "comparison_available": complete}
    if not complete:
        return report
    report["tables"] = {"P": {}, "Q": {}}
    for model in models:
        regions = {}
        for region in REGIONS:
            runs = [artifacts[(model, region, seed)] for seed in SEEDS]
            summary = aggregate_seed_scores(runs)
            regions[region] = {"scores": summary, "seed_evidence": _seed_evidence(runs, model)}
        tables = {artifacts[(model, region, seed)].table for region in REGIONS for seed in SEEDS}
        if len(tables) != 1:
            raise ValueError(f"{model}: output table changes across experiment cells")
        # Region means give each market equal weight even if its sample count differs.
        macro = {key: float(np.mean([regions[r]["scores"]["seed_summary"][key]["mean"] for r in REGIONS]))
                 for key in _MAIN_METRICS if all(key in regions[r]["scores"]["seed_summary"] for r in REGIONS)}
        if "pinball_by_quantile" in regions[REGIONS[0]]["scores"]["seed_summary"]:
            keys = regions[REGIONS[0]]["scores"]["seed_summary"]["pinball_by_quantile"]
            macro["pinball_by_quantile"] = {key: float(np.mean([
                regions[r]["scores"]["seed_summary"]["pinball_by_quantile"][key]["mean"] for r in REGIONS])) for key in keys}
        report["tables"][next(iter(tables))][model] = {
            "n_cells": len(REGIONS) * len(SEEDS), "regions": regions, "macro_regions": macro,
            "aggregation": "equal_weight_region_mean_of_seed_metrics",
            "statistical_significance": "not_computed_seed_repeats_are_not_independent_time_series"}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_aligned_report(args.run_root)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
