"""V5 stage machine: prepare -> dev -> freeze -> final-test -> report.

Identity is frozen at prepare (sources, config, schema, tests, the full
import closure of reused execution dependencies, prepared baseline
predictions with their origin/target grids, baseline and V4 artefacts,
caches, external data, and the trusted manifests carrying the expected
test-side hashes).  Every later stage re-verifies it and refuses to continue
after any change; done markers additionally bind the persisted development
models and predictions they allow resume to load.  The 2023-2024 test split
can only be opened by final-test, after freeze.json (whose frozen
test-source hashes were recorded from the trusted manifests without reading
any test bytes) has been verified; the final evaluation runs as an explicit
sequential matured-queue pass that predicts and persists each origin block
before any later-maturing label is absorbed, under an OS file lock held for
the process lifetime, with identity-safe resume.  Development may run
region-sharded (dev-region per region + dev-finalize) or serially (dev);
both call the same per-region implementation and finalizer and never change
any model/data/solver/tree parameter, seed, feature, order or gate.
Nothing commits, pushes, retrains the frozen baselines or writes into
pre-existing runs.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from . import controls, data, diagnostics, history, report, schema
from .calibrator import CalibratorModel, StackerModel

REGION_ORDER = ("NSW1", "QLD1", "TAS1")
DEV_EVAL_SEGMENTS = ("v1", "v2")
DEV_SEGMENT_LABELS = ("dfit_fit", "v1", "v2")

# Round-4 H2: the V5 test files and the reused read-only sources are part of
# the frozen identity; any later change makes every stage refuse.
V5_TEST_FILES = (
    "tests/test_full_v5_history.py",
    "tests/test_full_v5_calibrator.py",
    "tests/test_full_v5_protocol.py",
    "tests/test_full_v5_sharded_dev.py",
)
# Round-6 H2: bind the complete import closure of the execution
# dependencies (verified as the set of repo modules actually imported by a
# dev/freeze/final-test process, including the lazily imported frozen
# LightGBM predictor with its aligned_baselines/lasso chain, and the
# src/dualfield torch modules imported by the forecasting package init).
# Any mutation of these files makes every stage of an existing run refuse.
REUSED_SOURCE_FILES = (
    "forecasting/__init__.py",
    "forecasting/aligned_baselines.py",
    "forecasting/aligned_inputs.py",
    "forecasting/aligned_scoring.py",
    "forecasting/datasets.py",
    "forecasting/full/__init__.py",
    "forecasting/full/loss.py",
    "forecasting/full/model.py",
    "forecasting/lasso/__init__.py",
    "forecasting/lasso/model.py",
    "forecasting/lightgbm/__init__.py",
    "forecasting/lightgbm/predictor.py",
    "src/dualfield/__init__.py",
    "src/dualfield/core.py",
    "src/dualfield/losses.py",
    "src/dualfield/metrics.py",
    "src/dualfield/trainer.py",
)
# Sequential final evaluation: origins per predict/persist block.
FINAL_BLOCK_ORIGINS = 24
# Round-4 L1: direction checks use an explicit numerical tolerance (score
# units, AUD/MWh); raw differences are always retained.
DIRECTION_TOLERANCE = 1e-9
# Round-6 L3: the V4 force_null diagnostic remains unrun (non-blocking
# declaration from R1); new runs carry it forward so their reports list it.
UNFINISHED_CARRIED_OVER = [
    "V4 force_null frontend/full-pipeline diagnostic (spec 7.8) not run: lowest priority tier; "
    "requires loading the V4 torch frontend and frozen residual tree for a dev-segment-only "
    "intervention. All controls (C_*/V5/T_*/A_stack) ran in full.",
]


class BudgetExceeded(RuntimeError):
    """Raised when a stage has consumed its pre-declared CPU core-hours."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_config(path) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    schema.validate_config(config)
    return config


def _exp(config) -> dict:
    return config["v5_calibrated_history"]


def _source_files(project_root: Path) -> list[Path]:
    files = sorted((project_root / "forecasting" / "full_v5").glob("*.py"))
    files.append(project_root / "configs" / "full_v5" / "experiment.yaml")
    files += sorted((project_root / "docs").glob("full_v5_*.md"))
    return files


def _source_identity(project_root: Path) -> dict:
    return {str(p.relative_to(project_root)): data.sha256_file(p) for p in _source_files(project_root)}


def _file_identities(project_root: Path, relpaths) -> dict:
    out = {}
    for rel in relpaths:
        path = project_root / rel
        if not path.exists():
            raise FileNotFoundError(f"Required identity file missing: {rel}")
        out[rel] = data.sha256_file(path)
    return out


def _grid_sha256(array) -> str:
    return data.sha256_bytes(np.ascontiguousarray(np.asarray(array)).tobytes())


def _run_paths(run_root):
    run_root = Path(run_root)
    return dict(
        root=run_root,
        config=run_root / "config.yaml",
        identity=run_root / "identity" / "identity.json",
        baseline=run_root / "baseline_predictions",
        audit=run_root / "history_audit",
        development=run_root / "development",
        models=run_root / "final_models",
        freeze=run_root / "freeze.json",
        final=run_root / "final_test",
        reports=run_root / "reports",
        state=run_root / "state.json",
        budget=run_root / "budget.json",
        commands=run_root / "commands.jsonl",
        unfinished=run_root / "unfinished.json",
    )


def _read_state(paths) -> dict:
    if not paths["state"].exists():
        return dict(stage="new", status="none")
    return data.read_json(paths["state"])


def _write_state(paths, **kwargs) -> None:
    state = _read_state(paths)
    state.update(kwargs, updated_at=_now())
    data.atomic_json(paths["state"], state)


# --- budget accounting (Round-6 M4) ------------------------------------------
# Usage = persisted ledger + live keys describing the current process.  The
# live keys are folded into the ledger when a later attempt starts, so an
# interrupted or killed process still counts its last checkpointed usage.
_USAGE: dict = {}


def _process_cpu() -> float:
    return time.process_time()


def _region_dir(paths, region: str) -> Path:
    return paths["development"] / "regions" / region


def _region_budget_path(paths, region: str) -> Path:
    return _region_dir(paths, region) / "budget.json"


def _read_budget(path: Path) -> dict:
    return data.read_json(path) if path.exists() else {}


def _pool_cpu(entry: dict) -> float:
    """Ledger + live CPU seconds recorded for one pool entry."""
    return (float(entry.get("cpu_seconds", 0.0) or 0.0)
            + float(entry.get("live_cpu_seconds", 0.0) or 0.0))


def _dev_pool_cpu_seconds(paths, exp: dict) -> float:
    """Total dev-pool CPU: run-root ledger plus every region shard record.

    Region shards keep their own budget files so concurrent workers never
    race on the shared budget.json; the dev ceiling covers their sum.
    """
    total = _pool_cpu(_read_budget(paths["budget"]).get("dev", {}))
    for region in exp["regions"]:
        total += _pool_cpu(_read_budget(_region_budget_path(paths, region)).get("dev", {}))
    return total


def _usage_begin(paths, pool: str, region: str | None = None) -> None:
    """Start stage usage tracking for this process.

    Any stale live usage a previous crashed attempt left in this pool is
    folded into the ledger first, so interruptions are never forgotten.
    """
    path = _region_budget_path(paths, region) if (pool == "dev" and region) else paths["budget"]
    path.parent.mkdir(parents=True, exist_ok=True)
    budget = _read_budget(path)
    entry = budget.setdefault(pool, dict(cpu_seconds=0.0, wall_seconds=0.0))
    stale_cpu = float(entry.pop("live_cpu_seconds", 0.0) or 0.0)
    stale_wall = float(entry.pop("live_wall_seconds", 0.0) or 0.0)
    if stale_cpu or stale_wall:
        entry["cpu_seconds"] = float(entry.get("cpu_seconds", 0.0)) + stale_cpu
        entry["wall_seconds"] = float(entry.get("wall_seconds", 0.0)) + stale_wall
    entry.update(live_pid=os.getpid(), live_cpu_seconds=0.0, live_wall_seconds=0.0,
                 live_started_at=_now())
    budget["updated_at"] = _now()
    data.atomic_json(path, budget)
    _USAGE.clear()
    _USAGE.update(pool=pool, region=region, cpu0=_process_cpu(), wall0=time.time(), path=str(path))


def _usage_checkpoint(paths) -> None:
    """Persist this process's consumption so far (visible to other shards)."""
    if not _USAGE:
        return
    cpu = _process_cpu() - _USAGE["cpu0"]
    wall = time.time() - _USAGE["wall0"]
    path = Path(_USAGE["path"])
    budget = _read_budget(path)
    entry = budget.setdefault(_USAGE["pool"], dict(cpu_seconds=0.0, wall_seconds=0.0))
    if entry.get("live_pid") not in (None, os.getpid()):
        # someone else's stale live usage: fold it before recording ours
        entry["cpu_seconds"] = float(entry.get("cpu_seconds", 0.0)) + float(entry.pop("live_cpu_seconds", 0.0) or 0.0)
        entry["wall_seconds"] = float(entry.get("wall_seconds", 0.0)) + float(entry.pop("live_wall_seconds", 0.0) or 0.0)
        entry.pop("live_pid", None)
        entry.pop("live_started_at", None)
    entry.update(live_pid=os.getpid(), live_cpu_seconds=cpu, live_wall_seconds=wall)
    budget["updated_at"] = _now()
    data.atomic_json(path, budget)


def _usage_finish(paths) -> tuple[float, float]:
    """Fold this process's measured usage into the ledger; clear live keys."""
    if not _USAGE:
        return 0.0, 0.0
    cpu = _process_cpu() - _USAGE["cpu0"]
    wall = time.time() - _USAGE["wall0"]
    path = Path(_USAGE["path"])
    budget = _read_budget(path)
    entry = budget.setdefault(_USAGE["pool"], dict(cpu_seconds=0.0, wall_seconds=0.0))
    for key in ("live_pid", "live_cpu_seconds", "live_wall_seconds", "live_started_at"):
        entry.pop(key, None)
    entry["cpu_seconds"] = float(entry.get("cpu_seconds", 0.0)) + cpu
    entry["wall_seconds"] = float(entry.get("wall_seconds", 0.0)) + wall
    budget["updated_at"] = _now()
    data.atomic_json(path, budget)
    _USAGE.clear()
    return cpu, wall


def _check_budget(paths, exp: dict, pool: str, region: str | None = None) -> None:
    """Enforce the pre-declared CPU core-hour ceiling for one pool.

    The check totals the persisted ledger(s) plus the live records of the
    running process (checkpointed first so concurrent shards can see each
    other's consumption); for the dev pool this spans every region shard, so
    parallelism may shorten wall time but never expands total CPU budget.
    """
    limit_hours = float(exp["budget"][f"{pool}_core_hours"])
    _usage_checkpoint(paths)
    if pool == "dev":
        used = _dev_pool_cpu_seconds(paths, exp)
    else:
        used = _pool_cpu(_read_budget(paths["budget"]).get(pool, {}))
    if used >= limit_hours * 3600.0:
        raise BudgetExceeded(
            f"{pool} stage budget exhausted: {used / 3600.0:.3f} of {limit_hours:g} CPU core-hours used; "
            "saving progress and stopping (spec section 11)")


# --- OS run locks (Round-6 High) ---------------------------------------------
class _Flock:
    """An exclusive lock held by the process lifetime via the OS (flock).

    Ownership lives in the kernel and is released automatically when the
    holding process exits, so there is no publication window to race on and
    a lock is NEVER removed merely because owner metadata is absent, stale
    or unreadable: a leftover lock file without a live holder is simply
    re-acquirable.  The owner record inside the file is informational only.
    """

    def __init__(self, path, what: str):
        self.path = Path(path)
        self.what = what
        self._fd: int | None = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            holder = {}
            try:
                holder = json.loads(self.path.read_text() or "{}")
            except (ValueError, OSError):
                pass
            pid = holder.get("pid")
            who = f"live pid {pid}" if isinstance(pid, int) and _pid_alive(pid) else \
                f"pid {pid!r}" if pid is not None else "an unknown holder"
            raise PermissionError(
                f"{self.what} lock is held by {who}; refusing concurrent execution")
        try:
            os.ftruncate(fd, 0)
            os.write(fd, (json.dumps(dict(pid=os.getpid(), created_at=_now())) + "\n").encode())
        except OSError:
            pass  # informational only; the flock itself is the mutex
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


def _identity_digest(identity: dict) -> str:
    return data.sha256_bytes(data.canonical_json(identity).encode())


def _verify_identity(paths, project_root: Path, config: dict) -> dict:
    recorded = data.read_json(paths["identity"])
    if recorded["config_sha256"] != data.sha256_file(paths["config"]):
        raise ValueError("Run config.yaml changed after prepare; refusing")
    current = _source_identity(project_root)
    if current != recorded["sources"]:
        changed = sorted(set(recorded["sources"]) ^ set(current)) + [
            k for k in set(recorded["sources"]) & set(current)
            if recorded["sources"][k] != current[k]]
        raise ValueError(f"V5 source/schema files changed after prepare; refusing. Changed: {changed}")
    if schema.SCHEMA_HASH != recorded["schema_hash"]:
        raise ValueError("Schema hash changed after prepare; refusing")
    for section, relpaths in (("tests", V5_TEST_FILES), ("reused_sources", REUSED_SOURCE_FILES)):
        if section not in recorded:
            raise ValueError(f"identity.json predates {section} binding; refusing (rerun prepare)")
        current_files = _file_identities(project_root, relpaths)
        if current_files != recorded[section]:
            changed = sorted(set(recorded[section]) ^ set(current_files)) + [
                k for k in set(recorded[section]) & set(current_files)
                if recorded[section][k] != current_files[k]]
            raise ValueError(f"{section} changed after prepare; refusing. Changed: {changed}")
    if not recorded.get("baseline_predictions"):
        raise ValueError("identity.json predates prepared-baseline binding; refusing (rerun prepare)")
    for region, entry in recorded["baseline_predictions"].items():
        path = paths["baseline"] / f"{region}_calibration.npz"
        if not path.exists() or data.sha256_file(path) != entry["file_sha256"]:
            raise ValueError(f"Prepared baseline predictions changed or missing for {region}; refusing")
        with np.load(path, allow_pickle=False) as archive:
            grids = dict(
                origins_sha256=_grid_sha256(archive["origin_unix"]),
                targets_sha256=_grid_sha256(archive["target_time_unix"]),
                quantiles_sha256=_grid_sha256(archive["quantiles"]),
                n_origins=int(len(archive["origin_unix"])))
        mismatched = [key for key, value in grids.items() if value != entry[key]]
        if mismatched:
            raise ValueError(
                f"Prepared baseline grid/quantiles changed for {region} ({mismatched}); refusing")
    return recorded


def _verify_freeze(paths, project_root: Path, config: dict) -> dict:
    freeze = data.read_json(paths["freeze"])
    identity = _verify_identity(paths, project_root, config)
    if freeze["identity_digest"] != data.sha256_bytes(data.canonical_json(identity).encode()):
        raise ValueError("freeze.json identity digest mismatch")
    mismatches = []
    for relpath, digest in freeze["model_files"].items():
        path = paths["root"] / relpath
        if not path.exists() or data.sha256_file(path) != digest:
            mismatches.append(relpath)
    if mismatches:
        raise ValueError(f"Frozen model files changed or missing: {mismatches}")
    if freeze["schema_hash"] != schema.SCHEMA_HASH:
        raise ValueError("freeze.json schema mismatch")
    return freeze


def _verify_test_sources(paths, project_root: Path, config: dict, exp: dict, freeze: dict) -> None:
    """Verify the test-side hashes frozen at freeze time before any test load."""
    recorded = freeze.get("test_sources")
    if not recorded or not recorded.get("cache"):
        raise ValueError("freeze.json lacks frozen test-source hashes; refusing")
    baseline = data.BaselineSource(config, project_root)
    v4 = data.V4Source(config, project_root)
    problems = []
    for region in exp["regions"]:
        cache_root = project_root / exp["cache"]["root"] / region
        for name, digest in sorted(recorded["cache"].get(region, {}).items()):
            path = cache_root / name
            if not path.exists() or data.sha256_file(path) != digest:
                problems.append(f"cache {region}/{name}")
        path = baseline.test_artifact_path(region)
        if not path.exists() or data.sha256_file(path) != recorded["baseline_test"][region]:
            problems.append(f"baseline test predictions {region}")
        path = v4.artifact_path(region, "test")
        if not path.exists() or data.sha256_file(path) != recorded["v4_test"][region]:
            problems.append(f"V4 test predictions {region}")
    if problems:
        raise ValueError(f"Test-side source files changed since freeze: {problems}")


def _verify_done_marker(path: Path, identity: dict) -> bool:
    """True when the marker exists and every bound output still matches.

    A marker written under the current identity whose outputs no longer match
    is tamper evidence and raises instead of silently recomputing.
    """
    if not path.exists():
        return False
    marker = data.read_json(path)
    if marker.get("identity_digest") != _identity_digest(identity):
        return False  # marker predates the current identity: not usable
    problems = []
    for relpath, digest in marker.get("outputs", {}).items():
        target = Path(relpath) if Path(relpath).is_absolute() else path.parent.parent / relpath
        if not target.exists() or data.sha256_file(target) != digest:
            problems.append(relpath)
    if problems:
        raise ValueError(f"Done marker {path.name} outputs changed or missing: {problems}; refusing")
    return True


def _write_done_marker(path: Path, identity: dict, outputs: dict) -> None:
    digest = _identity_digest(identity)
    rel = {}
    for target in outputs:
        target = Path(target)
        rel[str(target.resolve().relative_to(path.parent.parent.resolve()))
            if target.resolve().is_relative_to(path.parent.parent.resolve()) else str(target)] = data.sha256_file(target)
    data.atomic_json(path, dict(identity_digest=digest, outputs=rel, created_at=_now()))


# --- feature/library construction shared by dev, freeze and final-test ------
def _region_cache(config, project_root, region, split, expected_hash=None):
    cache_root = project_root / _exp(config)["cache"]["root"]
    return data.load_cache_split(cache_root, region, split, expected_hash)


def _region_features_and_library(config, project_root, region, arrays, baseline_q, audit_name, paths):
    z = data.build_z(baseline_q, arrays)
    library = history.HistoryLibrary.from_segment(arrays["origin_unix"], arrays["actual"], baseline_q, z)
    engine = history.FeatureEngine(library)
    features, result = engine.features_matrix(arrays["origin_unix"], z, baseline_q)
    features_recent = engine.features_matrix_recent(z, result)
    manifest = dict(
        region=region, records=len(library),
        origin_range=[int(library.origins[0]), int(library.origins[-1])] if len(library) else None,
        baseline_q_sha=data.sha256_bytes(np.ascontiguousarray(baseline_q).tobytes()),
        residuals_sha=data.sha256_bytes(np.ascontiguousarray(library.residuals).tobytes()),
        audit=engine.audit.summary(),
    )
    data.atomic_json(paths["audit"] / f"{region}_{audit_name}.json", manifest)
    return features, features_recent, library, manifest


def _fit_rows(origins: np.ndarray, segment: str) -> tuple[np.ndarray, dict]:
    start, end = schema.segment_bounds()[segment]
    in_segment = data.window_in_segment(origins, start, end)
    cutoff = schema.fit_cutoffs()[segment]
    available = data.fit_available(origins, cutoff)
    fit = in_segment & available
    info = dict(segment=segment, in_segment=int(in_segment.sum()), fit_rows=int(fit.sum()),
                dropped_by_availability=int((in_segment & ~available).sum()))
    return fit, info


def _predict_all_models(config, project_root, region, arrays, baseline_q, features, features_recent,
                        v4_q, models: dict) -> dict:
    """Predictions of every model over ALL rows; callers slice the segments.

    Works unchanged on full arrays (dev/freeze scoring) and on block slices
    (sequential final evaluation); every model predicts rows independently.
    Tree matrices are built and predicted in origin chunks to bound memory.
    """
    out = {schema.BASELINE: baseline_q}
    for model_id in schema.CALIBRATOR_CONTROLS:
        features_matrix = controls.calibrator_features_for(model_id, {
            "match": features, "recent": features_recent})
        prediction = models[model_id].predict(features_matrix, baseline_q)
        out[model_id] = prediction["quantiles"]
        if model_id == "V5_history":
            out["_V5_parts"] = prediction
    out[schema.STACKER] = models[schema.STACKER].predict(baseline_q, v4_q)
    flat = data.flatten_common(arrays)
    n = len(baseline_q)
    for model_id, with_v in (("T_roll", False), ("T_history", True)):
        boosters = models[model_id][region]
        quantiles = np.empty((n, schema.HORIZONS, 5), dtype=np.float64)
        block = features if with_v else features[:, :, :schema.NZ + schema.NM]
        chunk = 2048
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            matrix = controls.tree_feature_matrix(flat[start:stop], block[start:stop], with_v)
            quantiles[start:stop] = controls.predict_tree_control(boosters, matrix, stop - start)
        out[model_id] = quantiles
    return out


def _score_segments(predictions: dict, arrays: dict, metadata: dict, rows_by_segment: dict) -> dict:
    scored = {}
    for segment, rows in rows_by_segment.items():
        idx = np.flatnonzero(rows)
        if not len(idx):
            raise ValueError(f"Empty segment {segment}")
        subset = {key: arrays[key][idx] for key in ("actual", "origin_unix", "target_time_unix")}
        segment_scores = {}
        for model_id, quantiles in predictions.items():
            if model_id.startswith("_"):
                continue
            wrapped = diagnostics.build_artifact(model_id, quantiles[idx], subset, metadata, segment)
            entry = dict(scores=wrapped["scores"])
            q50 = np.sort(quantiles[idx], axis=-1)[..., 2]
            entry["bands"] = diagnostics.band_decomposition(q50, subset["actual"])
            entry["coverage"] = diagnostics.coverage_diagnostics(quantiles[idx], subset["actual"])
            entry["groups"] = diagnostics.horizon_group_metrics(quantiles[idx], subset["actual"])
            entry["spike_alarm"] = diagnostics.spike_alarm_diagnostics(quantiles[idx], subset["actual"])
            entry["rmse_region"] = float(np.sqrt(np.square(q50 - subset["actual"]).mean()))
            segment_scores[model_id] = entry
        scored[segment] = segment_scores
    return scored


def _feature_distributions_by_group(features: np.ndarray, rows_by_segment: dict) -> dict:
    """Round-4 M5: per feature group (z, m, v) x time segment summaries."""
    groups = {
        "z": features[:, :, :schema.NZ],
        "m": features[:, :, schema.NZ:schema.NZ + schema.NM],
        "v": features[:, :, schema.F_V:],
    }
    return {name: {segment: diagnostics.feature_distribution_summary(matrix, rows, name)
                   for segment, rows in rows_by_segment.items()}
            for name, matrix in groups.items()}


# --- gates -------------------------------------------------------------------
def _macro(scored_by_region: dict, segment: str, model_id: str, key) -> float:
    values = []
    for region in scored_by_region:
        entry = scored_by_region[region][segment][model_id]
        if key == "mae":
            values.append(entry["scores"]["mae"])
        elif key == "qs":
            values.append(entry["scores"]["quantile_score_proxy"])
        elif key == "rmse_macro":
            values.append(entry["rmse_region"])
        elif key.startswith("band_"):
            band = entry["bands"][key[len("band_"):]]["conditional_mae"]
            values.append(band if band is not None else float("nan"))
        else:
            raise KeyError(key)
    return float(np.mean(values))


def _regional(scored_by_region: dict, segment: str, model_id: str, key) -> dict:
    out = {}
    for region, segments in scored_by_region.items():
        entry = segments[segment][model_id]
        if key == "mae":
            out[region] = entry["scores"]["mae"]
        elif key == "qs":
            out[region] = entry["scores"]["quantile_score_proxy"]
        elif key.startswith("band_"):
            band = entry["bands"][key[len("band_"):]]["conditional_mae"]
            out[region] = band if band is not None else float("nan")
        else:
            raise KeyError(key)
    return out


def evaluate_gates(scored_by_region: dict, g0: dict, thresholds: dict) -> dict:
    """Mechanical G1/G2 evaluation on the merged V1+V2 development window."""
    gates = dict(G0=dict(passed=bool(g0.get("identity_verified") and g0.get("tests_passed")),
                         components=g0))
    merged = "v1+v2"
    # Merge V1 and V2 by averaging their per-region cell-level losses is not
    # equivalent to scoring the union; the union is scored directly by callers
    # under the key "v1+v2" when both segments share the grid.
    if merged not in next(iter(scored_by_region.values())):
        raise ValueError("Merged v1+v2 scoring missing for gate evaluation")
    v5, baseline = "V5_history", schema.BASELINE

    def macro_improvement(model_ref, key):
        ref = _macro(scored_by_region, merged, model_ref, key)
        cand = _macro(scored_by_region, merged, v5, key)
        return dict(reference=ref, candidate=cand, improvement=(ref - cand) / ref if ref else float("nan"))

    g1_details = dict(mae=macro_improvement(baseline, "mae"), qs=macro_improvement(baseline, "qs"),
                      per_region={}, per_region_band={})
    for region in scored_by_region:
        ref = _regional(scored_by_region, merged, baseline, "mae")[region]
        cand = _regional(scored_by_region, merged, v5, "mae")[region]
        g1_details["per_region"][region] = dict(reference=ref, candidate=cand,
                                                improvement=(ref - cand) / ref if ref else float("nan"))
        ref_b = _regional(scored_by_region, merged, baseline, "band_range_0_300")[region]
        cand_b = _regional(scored_by_region, merged, v5, "band_range_0_300")[region]
        g1_details["per_region_band"][region] = dict(reference=ref_b, candidate=cand_b,
                                                     improvement=((ref_b - cand_b) / ref_b
                                                                  if ref_b else float("nan")))
    g1_pass = (g1_details["mae"]["improvement"] >= thresholds["g1_macro_improvement"]
               and g1_details["qs"]["improvement"] >= thresholds["g1_macro_improvement"]
               and all(v["improvement"] >= -thresholds["g1_region_mae_max_worsening"]
                       for v in g1_details["per_region"].values())
               and all(v["improvement"] >= -thresholds["g1_region_band0300_max_worsening"]
                       for v in g1_details["per_region_band"].values()))
    gates["G1"] = dict(passed=bool(g1_pass), details=g1_details,
                       thresholds={k: thresholds[k] for k in
                                   ("g1_macro_improvement", "g1_region_mae_max_worsening",
                                    "g1_region_band0300_max_worsening")})

    g2_details = {}
    g2_pass = True
    for control in ("C_roll", "C_recent"):
        entry = dict(mae=macro_improvement(control, "mae"), qs=macro_improvement(control, "qs"),
                     per_segment_direction={})
        for segment in DEV_EVAL_SEGMENTS:
            seg = dict()
            for key in ("mae", "qs"):
                ref = _macro(scored_by_region, segment, control, key)
                cand = _macro(scored_by_region, segment, v5, key)
                # Raw difference retained; the direction verdict tolerates
                # numerical noise (float-precision equivalent predictions must
                # not be read as a reversal).
                seg[key] = dict(reference=ref, candidate=cand, diff_ref_minus_cand=ref - cand,
                                direction_tolerance=DIRECTION_TOLERANCE,
                                non_reversed=bool(ref - cand >= -DIRECTION_TOLERANCE))
            entry["per_segment_direction"][segment] = seg
        macro_ok = (entry["mae"]["improvement"] >= thresholds["g2_macro_improvement"]
                    and entry["qs"]["improvement"] >= thresholds["g2_macro_improvement"])
        direction_ok = all(
            seg[key]["non_reversed"]
            for seg in entry["per_segment_direction"].values() for key in ("mae", "qs"))
        entry["macro_ok"] = bool(macro_ok)
        entry["direction_ok_all_segments"] = bool(direction_ok)
        entry["passed"] = bool(macro_ok and direction_ok)
        g2_pass = g2_pass and entry["passed"]
        g2_details[control] = entry
    gates["G2"] = dict(passed=bool(g2_pass), details=g2_details,
                       thresholds=dict(g2_macro_improvement=thresholds["g2_macro_improvement"],
                                       direction_tolerance=DIRECTION_TOLERANCE))
    gates["passed"] = bool(gates["G0"]["passed"] and g1_pass and g2_pass)
    gates["evaluated_at"] = _now()
    return gates


# --- stages ------------------------------------------------------------------
def prepare(config_path, run_root) -> int:
    project_root = Path.cwd().resolve()
    config = _load_config(config_path)
    paths = _run_paths(run_root)
    if paths["root"].exists():
        raise FileExistsError("Run root already exists; refusing to overwrite (spec section 10)")
    paths["root"].mkdir(parents=True)
    shutil.copyfile(config_path, paths["config"])
    design = project_root / "docs" / "full_v5_design_20261015.md"
    if design.exists():
        shutil.copyfile(design, paths["root"] / "design.md")
    source_dir = paths["root"] / "source"
    for path in _source_files(project_root):
        target = source_dir / path.relative_to(project_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    data.set_stage("prepare")
    exp = _exp(config)
    baseline = data.BaselineSource(config, project_root)
    v4 = data.V4Source(config, project_root)
    if baseline.mode == "frozen_lgbm":
        jobs = data.read_json(project_root / exp["baseline"]["run_root"] / "jobs.json")
        cache_hashes = {region: data.cache_hashes_from_jobs(jobs, region) for region in exp["regions"]}
    else:
        jobs = None
        cache_root = project_root / exp["cache"]["root"]
        cache_hashes = {region: data.sha256_file(cache_root / region / "hashes.json")
                        for region in exp["regions"]}
    identity = dict(
        created_at=_now(),
        git=data.git_identity(project_root),
        config_sha256=data.sha256_file(paths["config"]),
        schema_hash=schema.SCHEMA_HASH,
        sources=_source_identity(project_root),
        tests=_file_identities(project_root, V5_TEST_FILES),
        reused_sources=_file_identities(project_root, REUSED_SOURCE_FILES),
        baseline_run=dict(
            run_root=exp["baseline"]["run_root"], mode=baseline.mode,
            jobs_json_sha256=(data.sha256_file(project_root / exp["baseline"]["run_root"] / "jobs.json")
                              if jobs is not None else None),
            source_hash=jobs.get("source_hash") if jobs else None,
            original_config_hash=jobs.get("config_hash") if jobs else None,
            environment=jobs.get("environment") if jobs else None,
            git_head=jobs.get("git_head") if jobs else None,
            data_hashes_verified=data.verify_data_hashes(jobs, project_root) if jobs else {},
            availability_invariant={},
            test_prediction_hashes_from_jobs={},
        ),
        cache_hashes=cache_hashes,
        v4=dict(run_root=exp["v4"].get("run_root"), mode=v4.mode, files={}),
        baseline_provenance={},
        baseline_predictions={},
    )
    if jobs is not None:
        for path in sorted(jobs.get("data_hashes", {})):
            target = Path(path) if Path(path).is_absolute() else project_root / path
            if not target.exists():
                continue
            with target.open("rb") as handle:
                header = handle.readline()
            if b"delivery_start_aest" in header and b"available_at_aest" in header:
                identity["baseline_run"]["availability_invariant"][str(target)] = \
                    data.check_availability_invariant(target)
        if not identity["baseline_run"]["availability_invariant"]:
            raise ValueError("No region hourly CSV found among frozen data hashes")
    # Round-6 M1: bind the expected test-side hashes from the trusted
    # manifests of the frozen runs (status.json artifact hashes / V4
    # output_hashes.json; a precomputed sidecar manifest in tests).  No test
    # bytes are read here: freeze records these values verbatim and only the
    # authorized final-test compares them against the actual files.
    identity["baseline_run"]["test_hash_manifests"] = {}
    identity["v4"]["test_prediction_hashes"] = {}
    identity["v4"]["test_hash_manifests"] = {}
    for region in exp["regions"]:
        identity["baseline_run"]["test_prediction_hashes_from_jobs"][region] = baseline.trusted_test_hash(region)
        identity["baseline_run"]["test_hash_manifests"][region] = baseline.trusted_test_hash_manifest(region)
        identity["v4"]["test_prediction_hashes"][region] = v4.trusted_test_hash(region)
        identity["v4"]["test_hash_manifests"][region] = v4.trusted_test_hash_manifest(region)

    for region in exp["regions"]:
        arrays, metadata = _region_cache(config, project_root, region, "calibration", cache_hashes[region])
        quantiles = baseline.calibration_quantiles(region, arrays, metadata)
        path = paths["baseline"] / f"{region}_calibration.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, quantiles=quantiles, origin_unix=arrays["origin_unix"],
                            target_time_unix=arrays["target_time_unix"])
        identity["baseline_provenance"][region] = baseline.provenance(region)
        identity["v4"]["files"][f"{region}_calibration"] = v4.file_digest(region, "calibration")
        identity["baseline_predictions"][region] = dict(
            file_sha256=data.sha256_file(path),
            origins_sha256=_grid_sha256(arrays["origin_unix"]),
            targets_sha256=_grid_sha256(arrays["target_time_unix"]),
            quantiles_sha256=_grid_sha256(quantiles),
            actual_sha256=_grid_sha256(arrays["actual"]),
            n_origins=int(len(arrays["origin_unix"])))
        if exp.get("crosscheck_event_diagnosis"):
            alt = project_root / exp["crosscheck_event_diagnosis"] / "predictions" / "seed2026" / region / "calibration.npz"
            if alt.exists():
                with np.load(alt, allow_pickle=False) as archive:
                    other = archive["quantiles"]
                if other.shape == quantiles.shape:
                    identity.setdefault("crosschecks", {})[f"{region}_calibration_vs_event_diagnosis"] = dict(
                        max_abs_diff=float(np.max(np.abs(other - quantiles))),
                        identical=bool(np.array_equal(other, quantiles)))
        del arrays
    if v4.mode == "frozen_artifact":
        identity["v4"]["run_config_sha256"] = data.sha256_file(project_root / exp["v4"]["run_root"] / "config.yaml")
    data.atomic_json(paths["identity"], identity)
    manifest = dict(protocol=exp["protocol_id"], created_at=_now(), regions=list(exp["regions"]),
                    schema_hash=schema.SCHEMA_HASH, config_sha256=identity["config_sha256"],
                    baseline_origins={})
    for region in exp["regions"]:
        with np.load(paths["baseline"] / f"{region}_calibration.npz") as archive:
            manifest["baseline_origins"][region] = int(len(archive["origin_unix"]))
    data.atomic_json(paths["root"] / "manifest.json", manifest)
    data.atomic_json(paths["unfinished"], list(UNFINISHED_CARRIED_OVER))
    _write_state(paths, stage="prepared", status="ok")
    print(f"prepare complete: {paths['root']}")
    return 0


def _dev_model_files(paths, region) -> list[Path]:
    files = sorted(paths["development"].joinpath("models").glob(f"*_{region}.json"))
    for model_id in schema.TREE_CONTROLS:
        files += sorted((paths["development"] / "models" / model_id / region).glob("*.txt"))
    return files


def _all_dev_markers_valid(paths, exp: dict, identity: dict) -> bool:
    for region in exp["regions"]:
        if not _verify_done_marker(paths["development"] / f"done_{region}.json", identity):
            return False
    return True


def _dev_common_checks(paths, project_root, config) -> dict:
    """Identity, state and recorded-test checks shared by region workers,
    the serial dev driver and the finalizer."""
    data.set_stage("dev")
    identity = _verify_identity(paths, project_root, config)
    state = _read_state(paths)
    if state.get("stage") == "dev_failed":
        raise PermissionError("Development already failed its gates; freeze is not permitted (spec section 10)")
    tests_path = paths["development"] / "tests_passed.json"
    if not tests_path.exists():
        raise FileNotFoundError(
            "development/tests_passed.json missing: record the V5 pytest result before judging G0")
    tests = data.read_json(tests_path)
    if int(tests.get("n_failed", 1)) != 0 or int(tests.get("exit_code", 1)) != 0:
        raise ValueError("Recorded V5 test run has failures; G0 cannot pass")
    return identity


def dev(run_root) -> int:
    """Serial development: every region body in canonical order, then finalize.

    Calls exactly the same region implementation and finalizer as the sharded
    dev-region/dev-finalize interface, so serial and sharded development
    produce equivalent artefacts and gates by construction.  Each region books
    its usage under its own region lock exactly like a sharded worker.
    """
    paths = _run_paths(run_root)
    config = _load_config(paths["config"])
    exp = _exp(config)
    try:
        for region in exp["regions"]:
            dev_region(run_root, region)
        return dev_finalize(run_root)
    except BaseException as exc:
        # Serial dev holds no umbrella lock (regions/finalize lock themselves);
        # state has no competing writer in this failure mode and the write is
        # atomic, so mark the interruption here for the stage machine.
        try:
            if paths["state"].exists():
                _write_state(paths, status="interrupted", last_error=f"{type(exc).__name__}: {exc}")
        except OSError:
            pass
        raise


def dev_region(run_root, region) -> int:
    """Execute exactly the existing per-region development body for one region.

    Runs under an atomic per-region OS lock (a duplicate same-region worker
    refuses; different regions may proceed concurrently) and writes only
    region-namespaced artefacts: models, predictions, audit, the per-region
    score/extras/rows bundle, the region budget/command records and the
    hash-complete done_<REGION>.json marker.  It never touches the shared
    metrics.json/gates.json/report/state/budget/commands of the run root.
    """
    project_root = Path.cwd().resolve()
    paths = _run_paths(run_root)
    config = _load_config(paths["config"])
    exp = _exp(config)
    if region not in exp["regions"]:
        raise ValueError(f"Unknown development region {region!r}; config lists {list(exp['regions'])}")
    identity = _dev_common_checks(paths, project_root, config)
    state = _read_state(paths)
    if state.get("stage") == "dev_passed":
        # Completed development is immutable: this region's marker must still
        # verify (tampered outputs raise; a missing marker refuses).
        if not _verify_done_marker(paths["development"] / f"done_{region}.json", identity):
            raise ValueError(
                f"Development recorded as passed but region {region} marker is missing or unusable; refusing")
        if paths["development"].joinpath("gates.json").exists():
            print(f"Development already completed with this identity; region {region} marker verified")
            return 0
    region_dir = _region_dir(paths, region)
    region_dir.mkdir(parents=True, exist_ok=True)
    lock = _Flock(region_dir / "region.lock", f"dev-region {region}").acquire()
    try:
        # The region lock owns the whole budget lifecycle: begin after acquiring,
        # checkpoint through the body, finish plus status/command records before
        # release.  A duplicate worker refused by the lock above can therefore
        # never fold, replace or clear the active owner's live/ledger records.
        entry = dict(stage="dev-region", region=region, started_at=_now())
        try:
            _usage_begin(paths, "dev", region=region)
            try:
                _check_budget(paths, exp, "dev", region=region)
                _dev_region_body(paths, project_root, config, exp, identity, region)
                entry["exit_code"] = 0
            finally:
                cpu, wall = _usage_finish(paths)
                entry.update(cpu_seconds=cpu, wall_seconds=wall)
        except BaseException as exc:
            entry.update(finished_at=_now(), exit_code=1, error=f"{type(exc).__name__}: {exc}")
            _write_region_status(paths, region, "interrupted", error=f"{type(exc).__name__}: {exc}")
            _append_command(paths, entry, region=region)
            raise
        entry.update(finished_at=_now())
        _write_region_status(paths, region, "ok")
        _append_command(paths, entry, region=region)
        return 0
    finally:
        lock.release()


def _dev_region_body(paths, project_root, config, exp, identity, region) -> None:
    """The per-region development body (identical for serial and sharded dev)."""
    marker = paths["development"] / f"done_{region}.json"
    arrays, metadata = _region_cache(config, project_root, region, "calibration", identity["cache_hashes"][region])
    origins = np.asarray(arrays["origin_unix"], dtype=np.int64)
    baseline_q = data.load_prepared_baseline(paths["root"], region, "calibration")
    with np.load(paths["baseline"] / f"{region}_calibration.npz", allow_pickle=False) as archive:
        if not (np.array_equal(archive["origin_unix"], origins)
                and np.array_equal(archive["target_time_unix"],
                                   np.asarray(arrays["target_time_unix"], dtype=np.int64))):
            raise ValueError(f"Prepared baseline grids misaligned with cache for {region}")
    features, features_recent, library, library_manifest = _region_features_and_library(
        config, project_root, region, arrays, baseline_q, "development", paths)
    fit_dfit, info_dfit = _fit_rows(origins, "dfit")
    rows = {"dfit_fit": fit_dfit}
    for segment in DEV_EVAL_SEGMENTS:
        start, end = schema.segment_bounds()[segment]
        rows[segment] = data.window_in_segment(origins, start, end)
    # merged v1+v2 union scoring needs a contiguous selection
    rows["v1+v2"] = rows["v1"] | rows["v2"]
    rows_info = dict(dfit=info_dfit, v1=int(rows["v1"].sum()), v2=int(rows["v2"].sum()))
    v4 = data.V4Source(config, project_root)
    v4_q = v4.quantiles(region, "calibration", origins)
    if v4.file_digest(region, "calibration") != identity["v4"]["files"][f"{region}_calibration"]:
        raise ValueError(f"V4 calibration artefact changed for {region}")

    models = {}
    if _verify_done_marker(marker, identity):
        models = _load_dev_models(paths, region)
    else:
        models.update(controls.fit_all_calibrators(
            {"match": features, "recent": features_recent}, baseline_q, arrays["actual"], fit_dfit))
        models[schema.STACKER] = controls.fit_stacker_control(baseline_q, v4_q, arrays["actual"], fit_dfit)
        flat = data.flatten_common(arrays)
        for model_id, with_v in (("T_roll", False), ("T_history", True)):
            tree_marker = paths["development"] / f"done_trees_{model_id}_{region}.json"
            if _verify_done_marker(tree_marker, identity):
                models[model_id] = {region: controls.load_tree_control(
                    paths["development"] / "models" / model_id / region, exp["trees"])}
                continue
            fit_features = features[fit_dfit] if with_v else features[fit_dfit][:, :, :schema.NZ + schema.NM]
            matrix = controls.tree_feature_matrix(flat[fit_dfit], fit_features, with_v)
            boosters = controls.fit_tree_control(model_id, matrix, controls.tree_targets(arrays["actual"][fit_dfit]), exp["trees"])
            directory = paths["development"] / "models" / model_id / region
            controls.save_tree_control(boosters, directory)
            models[model_id] = {region: boosters}
            _write_done_marker(tree_marker, identity, sorted(directory.glob("*.txt")))
        for model_id in schema.CALIBRATOR_CONTROLS:
            data.atomic_json(paths["development"] / "models" / f"{model_id}_{region}.json",
                             models[model_id].to_json())
        data.atomic_json(paths["development"] / "models" / f"{schema.STACKER}_{region}.json",
                         models[schema.STACKER].to_json())
    predictions = _predict_all_models(config, project_root, region, arrays, baseline_q,
                                      features, features_recent, v4_q, models)
    scored = _score_segments(predictions, arrays, metadata, rows)
    eval_rows = rows["v1"] | rows["v2"]
    equivalence = {}
    for control in ("C_roll", "C_recent"):
        equivalence[control] = float(np.max(np.abs(
            predictions["V5_history"][eval_rows] - predictions[control][eval_rows])))
    extras = dict(
        library=library_manifest,
        feature_distributions=_feature_distributions_by_group(
            features, {segment: rows[segment] for segment in DEV_SEGMENT_LABELS}),
        v5_vs_controls_max_abs_diff=equivalence,
        v5_parts=diagnostics.calibrator_diagnostics(predictions["_V5_parts"], eval_rows),
        oracle_v1v2=diagnostics.oracle_stacker_upper_bound(
            baseline_q[eval_rows], v4_q[eval_rows],
            np.asarray(arrays["actual"])[eval_rows]),
    )
    pred_dir = paths["development"] / "predictions"
    for model_id, quantiles in predictions.items():
        if model_id.startswith("_"):
            continue
        for segment, row_mask in rows.items():
            idx = np.flatnonzero(row_mask)
            out = pred_dir / model_id / f"{region}_{segment}.npz"
            data.atomic_npz(out, quantiles=quantiles[idx], origin_unix=origins[idx],
                            actual=np.asarray(arrays["actual"])[idx])
    # Per-region score/extras/rows bundle, written atomically BEFORE the done
    # marker so the marker binds its exact hash (finalize verifies it).
    bundle_path = _region_dir(paths, region) / "bundle.json"
    data.atomic_json(bundle_path, dict(
        identity_digest=_identity_digest(identity), region=region,
        scored=scored, extras=extras, rows=rows_info, created_at=_now()))
    _write_done_marker(marker, identity,
                       sorted(pred_dir.glob(f"*/{region}_*.npz"))
                       + _dev_model_files(paths, region)
                       + [paths["audit"] / f"{region}_development.json", bundle_path])
    del arrays
    _check_budget(paths, exp, "dev", region=region)


def dev_finalize(run_root) -> int:
    """Merge the completed region shards, evaluate G0/G1/G2, render the
    report and atomically write the final development state.

    Verifies the run identity plus every region's hash-complete done marker
    and score/extras/rows bundle, loads them in canonical config order,
    merges the per-region budget/command records deterministically, and only
    then writes the shared metrics.json/gates.json/report/state.
    """
    project_root = Path.cwd().resolve()
    paths = _run_paths(run_root)
    config = _load_config(paths["config"])
    exp = _exp(config)
    identity = _dev_common_checks(paths, project_root, config)
    state = _read_state(paths)
    if state.get("stage") == "dev_passed" and paths["development"].joinpath("gates.json").exists():
        # Resume of a completed dev: every region marker (models + predictions
        # + audit) must still verify; tampered artefacts are rejected.
        if not _all_dev_markers_valid(paths, exp, identity):
            raise ValueError("Development recorded as passed but region markers are incomplete; refusing")
        print("Development already completed with this identity; see development/gates.json")
        return 0
    # The finalize lock serializes every writer of the shared run-root records
    # (metrics/gates/state/budget) and owns the finalize budget lifecycle.
    lock = _Flock(paths["development"] / "finalize.lock", "dev-finalize").acquire()
    try:
        entry = dict(stage="dev-finalize", started_at=_now())
        try:
            _usage_begin(paths, "dev")
            try:
                _check_budget(paths, exp, "dev")
                _finalize_body(paths, project_root, config, exp, identity)
                entry["exit_code"] = 0
            finally:
                cpu, wall = _usage_finish(paths)
                entry.update(cpu_seconds=cpu, wall_seconds=wall)
        except BaseException as exc:
            entry.update(finished_at=_now(), exit_code=1, error=f"{type(exc).__name__}: {exc}")
            _append_command(paths, entry)
            try:
                if paths["state"].exists():
                    _write_state(paths, status="interrupted", last_error=f"{type(exc).__name__}: {exc}")
            except OSError:
                pass
            raise
        entry.update(finished_at=_now())
        _append_command(paths, entry)
        return 0
    finally:
        lock.release()


def _finalize_body(paths, project_root, config, exp, identity) -> None:
    digest = _identity_digest(identity)
    scored_by_region, extras_by_region, rows_info = {}, {}, {}
    for region in exp["regions"]:  # canonical config order
        if not _verify_done_marker(paths["development"] / f"done_{region}.json", identity):
            raise ValueError(
                f"Development for region {region} is incomplete (missing or unusable done marker); "
                "refusing to finalize; run dev-region for this region first")
        bundle_path = _region_dir(paths, region) / "bundle.json"
        if not bundle_path.exists():
            raise ValueError(f"Region {region} score/extras/rows bundle missing; refusing to finalize")
        bundle = data.read_json(bundle_path)
        if bundle.get("identity_digest") != digest or bundle.get("region") != region:
            raise ValueError(f"Region {region} bundle identity mismatch; refusing to finalize")
        scored_by_region[region] = bundle["scored"]
        extras_by_region[region] = bundle["extras"]
        rows_info[region] = bundle["rows"]
    data.atomic_json(paths["development"] / "metrics.json",
                     dict(scored=scored_by_region, extras=extras_by_region, rows=rows_info))
    tests = data.read_json(paths["development"] / "tests_passed.json")
    g0 = dict(identity_verified=True, tests_passed=True, tests=tests)
    gates = evaluate_gates(scored_by_region, g0, exp["gates"])
    data.atomic_json(paths["development"] / "gates.json", gates)
    report.render_development_report(paths, scored_by_region, extras_by_region, gates, rows_info)
    _merge_region_records(paths, exp)
    _write_state(paths, stage="dev_passed" if gates["passed"] else "dev_failed",
                 status="gates_passed" if gates["passed"] else "gates_failed")
    print(f"dev gates passed: {gates['passed']}")
    if not gates["passed"]:
        print("Development gates failed; freeze/final-test are blocked. See development/gates.json")


def _merge_region_records(paths, exp) -> None:
    """Deterministic merge of the per-region budget/command records into
    run-root summaries, in canonical config order.

    The region budget files stay authoritative for enforcement; the root
    budget gains a point-in-time snapshot (dev_regions, dev_total) and the
    merged command log is written to development/commands_by_region.json.
    """
    budgets, commands = {}, {}
    for region in exp["regions"]:
        entry = _read_budget(_region_budget_path(paths, region)).get("dev", {})
        budgets[region] = dict(cpu_seconds=float(entry.get("cpu_seconds", 0.0) or 0.0),
                               wall_seconds=float(entry.get("wall_seconds", 0.0) or 0.0))
        cpath = _region_dir(paths, region) / "commands.jsonl"
        commands[region] = ([json.loads(line) for line in cpath.read_text().splitlines() if line.strip()]
                            if cpath.exists() else [])
    budget = _read_budget(paths["budget"])
    budget["dev_regions"] = budgets
    budget["dev_total_cpu_seconds"] = _dev_pool_cpu_seconds(paths, exp)
    budget["updated_at"] = _now()
    data.atomic_json(paths["budget"], budget)
    data.atomic_json(paths["development"] / "commands_by_region.json", commands)


def _write_region_status(paths, region, status, error=None) -> None:
    data.atomic_json(_region_dir(paths, region) / "status.json",
                     dict(status=status, last_error=error, updated_at=_now()))


def _load_dev_models(paths, region) -> dict:
    models = {}
    for model_id in schema.CALIBRATOR_CONTROLS:
        payload = data.read_json(paths["development"] / "models" / f"{model_id}_{region}.json")
        models[model_id] = CalibratorModel.from_json(payload, schema.CONTROL_SPECS)
    payload = data.read_json(paths["development"] / "models" / f"{schema.STACKER}_{region}.json")
    models[schema.STACKER] = StackerModel.from_json(payload)
    exp_trees = dict(schema.TREE_PARAMS)
    for model_id in schema.TREE_CONTROLS:
        models[model_id] = {region: controls.load_tree_control(
            paths["development"] / "models" / model_id / region, exp_trees)}
    return models


def freeze(run_root) -> int:
    project_root = Path.cwd().resolve()
    paths = _run_paths(run_root)
    config = _load_config(paths["config"])
    data.set_stage("freeze")
    identity = _verify_identity(paths, project_root, config)
    state = _read_state(paths)
    if state.get("stage") != "dev_passed":
        raise PermissionError("freeze requires development gates to have passed "
                              f"(current stage: {state.get('stage')!r})")
    exp = _exp(config)
    # The final lock guards the whole final-pool budget lifecycle for freeze
    # exactly as for final-test: a refused duplicate final worker can never
    # touch the ledger, and finish/state records happen before release.
    lock = _acquire_final_lock(paths)
    try:
        entry = dict(stage="freeze", started_at=_now())
        try:
            _usage_begin(paths, "final")
            try:
                _check_budget(paths, exp, "final")
                _freeze_body(paths, project_root, config, exp, identity)
                entry["exit_code"] = 0
            finally:
                cpu, wall = _usage_finish(paths)
                entry.update(cpu_seconds=cpu, wall_seconds=wall)
        except BaseException as exc:
            entry.update(finished_at=_now(), exit_code=1, error=f"{type(exc).__name__}: {exc}")
            _append_command(paths, entry)
            try:
                if paths["state"].exists():
                    _write_state(paths, status="interrupted", last_error=f"{type(exc).__name__}: {exc}")
            except OSError:
                pass
            raise
        entry.update(finished_at=_now())
        _append_command(paths, entry)
        return 0
    finally:
        lock.release()


def _freeze_body(paths, project_root, config, exp, identity) -> None:
    baseline = data.BaselineSource(config, project_root)
    v4 = data.V4Source(config, project_root)
    # Round-6 M1, checked BEFORE any refit work: freeze records the EXPECTED
    # test-side hashes from the trusted manifests that prepare bound into the
    # identity (the frozen cache hashes.json values, the baseline run's
    # status.json artifact hashes and the V4 output_hashes.json) — no test
    # bytes are read here; the manifests themselves must be unchanged since
    # prepare, and the actual test files are hash- and grid-verified only by
    # the authorized final-test.
    for source, section in ((baseline, "baseline_run"), (v4, "v4")):
        for region in exp["regions"]:
            if source.trusted_test_hash_manifest(region) != identity[section]["test_hash_manifests"][region]:
                raise ValueError(
                    f"Trusted test-hash manifest for {region} changed after prepare; refusing")
    for region in exp["regions"]:
        if not identity["baseline_run"]["test_prediction_hashes_from_jobs"].get(region):
            raise ValueError(f"No trusted pre-bound baseline test hash for {region}; refusing (rerun prepare)")
        if not identity["v4"]["test_prediction_hashes"].get(region):
            raise ValueError(f"No trusted pre-bound V4 test hash for {region}; refusing (rerun prepare)")
    model_files = {}
    for region in exp["regions"]:
        arrays, metadata = _region_cache(config, project_root, region, "calibration", identity["cache_hashes"][region])
        origins = np.asarray(arrays["origin_unix"], dtype=np.int64)
        baseline_q = data.load_prepared_baseline(paths["root"], region, "calibration")
        features, features_recent, _, _ = _region_features_and_library(
            config, project_root, region, arrays, baseline_q, "h2_refit", paths)
        fit_h2, info = _fit_rows(origins, "h2")
        v4_q = v4.quantiles(region, "calibration", origins)
        calibrators = controls.fit_all_calibrators(
            {"match": features, "recent": features_recent}, baseline_q, arrays["actual"], fit_h2)
        for model_id, model in calibrators.items():
            data.atomic_json(paths["models"] / model_id / f"{region}.json", model.to_json())
        stacker = controls.fit_stacker_control(baseline_q, v4_q, arrays["actual"], fit_h2)
        data.atomic_json(paths["models"] / schema.STACKER / f"{region}.json", stacker.to_json())
        flat = data.flatten_common(arrays)
        for model_id, with_v in (("T_roll", False), ("T_history", True)):
            fit_features = features[fit_h2] if with_v else features[fit_h2][:, :, :schema.NZ + schema.NM]
            matrix = controls.tree_feature_matrix(flat[fit_h2], fit_features, with_v)
            boosters = controls.fit_tree_control(model_id, matrix, controls.tree_targets(arrays["actual"][fit_h2]), exp["trees"])
            controls.save_tree_control(boosters, paths["models"] / model_id / region)
        del arrays, flat
        _check_budget(paths, exp, "final")
    for path in sorted(paths["models"].rglob("*")):
        if path.is_file():
            model_files[str(path.relative_to(paths["root"]))] = data.sha256_file(path)
    test_sources = dict(cache={}, baseline_test={}, v4_test={})
    for region in exp["regions"]:
        cache_root = project_root / exp["cache"]["root"] / region
        entries = {name: digest for name, digest in
                   data.read_json(cache_root / "hashes.json").items() if name.startswith("test/")}
        if not entries:
            raise ValueError(f"No test cache entries in the manifest for {region}")
        test_sources["cache"][region] = dict(sorted(entries.items()))
        baseline_hash = identity["baseline_run"]["test_prediction_hashes_from_jobs"].get(region)
        if not baseline_hash:
            raise ValueError(f"No trusted pre-bound baseline test hash for {region}; refusing (rerun prepare)")
        test_sources["baseline_test"][region] = baseline_hash
        v4_hash = identity["v4"]["test_prediction_hashes"].get(region)
        if not v4_hash:
            raise ValueError(f"No trusted pre-bound V4 test hash for {region}; refusing (rerun prepare)")
        test_sources["v4_test"][region] = v4_hash
    freeze_payload = dict(
        created_at=_now(),
        identity_digest=data.sha256_bytes(data.canonical_json(identity).encode()),
        schema_hash=schema.SCHEMA_HASH,
        config_sha256=identity["config_sha256"],
        model_files=model_files,
        test_sources=test_sources,
        rules=dict(feature_sets={k: {kk: list(vv) if isinstance(vv, tuple) else vv
                                     for kk, vv in v.items()} for k, v in schema.CONTROL_SPECS.items()},
                   tree_params=dict(schema.TREE_PARAMS), horizon_groups=[list(g) for g in schema.HORIZON_GROUPS],
                   gates=exp["gates"], segments={k: list(v) for k, v in schema.SEGMENTS.items()},
                   final_block_origins=FINAL_BLOCK_ORIGINS),
        dev_gates=dict(passed=data.read_json(paths["development"] / "gates.json")["passed"]),
    )
    data.atomic_json(paths["freeze"], freeze_payload)
    _write_state(paths, stage="frozen", status="ok")
    print(f"freeze complete: {len(model_files)} model files frozen")


# --- sequential final evaluation --------------------------------------------
def _pid_alive(pid: int) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _acquire_final_lock(paths) -> "_Flock":
    """Atomic run lock: only one final-test process may touch the run root.

    Round-6 High: the mutex is an OS flock held for the acquiring process's
    lifetime.  There is no create/publish window to race on, and the lock
    file is never deleted because owner metadata is absent or stale — a dead
    holder is released by the kernel, making a leftover file re-acquirable.
    """
    return _Flock(paths["final"] / "run.lock", "final-test").acquire()


def _expected_blocks(n: int, block_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + block_size, n)) for start in range(0, n, block_size)]


def _load_final_progress(paths, region: str, digest: str, blocks: list[tuple[int, int]],
                         origins: np.ndarray, block_size: int) -> tuple[int, dict]:
    """Verify persisted blocks and return the resume point.

    Blocks recorded under the current identity whose files no longer match
    their recorded hashes are tamper evidence (raise); a missing file is an
    interrupted write and the run resumes from that block.
    """
    manifest_path = paths["final"] / "blocks" / region / "manifest.json"
    if not manifest_path.exists():
        return 0, dict(identity_digest=digest, block_size=block_size, blocks={})
    manifest = data.read_json(manifest_path)
    if manifest.get("identity_digest") != digest:
        raise ValueError(
            f"Persisted final blocks for {region} were produced under a different identity; refusing")
    entries = manifest.get("blocks", {})
    valid = {}
    for index, (start, end) in enumerate(blocks):
        entry = entries.get(str(index))
        if entry is None:
            break
        if (entry.get("start_origin") != int(origins[start])
                or entry.get("end_origin") != int(origins[end - 1]) or entry.get("n") != end - start):
            raise ValueError(f"Final block {index} for {region} records unexpected origins; refusing")
        missing = False
        for model_id, digest_file in entry.get("files", {}).items():
            path = paths["final"] / "blocks" / region / model_id / f"block_{index:05d}.npz"
            if not path.exists():
                missing = True
                break
            if data.sha256_file(path) != digest_file:
                raise ValueError(
                    f"Final block file {model_id}/block_{index:05d}.npz for {region} changed after "
                    "it was persisted; refusing")
        if missing:
            break
        valid[str(index)] = entry
    manifest["blocks"] = valid
    return len(valid), manifest


def _persist_final_block(paths, region, index, predictions, block_origins, digest,
                         block_size, library_len, released_through) -> None:
    """Persist one block's predictions and update the resume manifest."""
    block_dir = paths["final"] / "blocks" / region
    files = {}
    for model_id, quantiles in predictions.items():
        if model_id.startswith("_"):
            continue
        out = block_dir / model_id / f"block_{index:05d}.npz"
        data.atomic_npz(out, quantiles=quantiles, origin_unix=block_origins)
        files[model_id] = data.sha256_file(out)
    manifest_path = block_dir / "manifest.json"
    manifest = data.read_json(manifest_path) if manifest_path.exists() else \
        dict(identity_digest=digest, block_size=block_size, blocks={})
    manifest["blocks"][str(index)] = dict(
        start_origin=int(block_origins[0]), end_origin=int(block_origins[-1]), n=len(block_origins),
        files=files, library_len=int(library_len), released_through=int(released_through),
        created_at=_now())
    data.atomic_json(manifest_path, manifest)


def _validate_final_grids(region: str, artifact, arrays: dict, what: str) -> None:
    """Elementwise origin/target-time/actual identity vs the frozen cache."""
    origins = np.asarray(arrays["origin_unix"], dtype=np.int64)
    if not np.array_equal(np.asarray(artifact.origin_unix, dtype=np.int64), origins):
        raise ValueError(f"{what} origin grid misaligned with the test cache for {region}")
    if not np.array_equal(np.asarray(artifact.target_time_unix, dtype=np.int64),
                          np.asarray(arrays["target_time_unix"], dtype=np.int64)):
        raise ValueError(f"{what} target-time grid misaligned with the test cache for {region}")
    if artifact.actual is None or not np.array_equal(
            np.asarray(artifact.actual, dtype=np.float64),
            np.asarray(arrays["actual"], dtype=np.float64)):
        raise ValueError(f"{what} actual labels differ from the frozen test cache for {region}")


def final_test(run_root) -> int:
    project_root = Path.cwd().resolve()
    paths = _run_paths(run_root)
    config = _load_config(paths["config"])
    state = _read_state(paths)
    if state.get("stage") != "frozen":
        raise PermissionError(f"final-test requires frozen state (current: {state.get('stage')!r})")
    if (paths["final"] / "complete.json").exists():
        raise PermissionError("Final evaluation already completed once; re-running is not permitted "
                              "(spec section 11: one frozen final matrix)")
    lock = _acquire_final_lock(paths)
    try:
        # The final lock owns the full final-pool budget lifecycle (Round-8 High
        # fix): begin after acquiring, finish/state/commands before release.
        entry = dict(stage="final-test", started_at=_now())
        try:
            _usage_begin(paths, "final")
            try:
                _final_test_body(paths, project_root, config)
                entry["exit_code"] = 0
            finally:
                cpu, wall = _usage_finish(paths)
                entry.update(cpu_seconds=cpu, wall_seconds=wall)
        except BaseException as exc:
            entry.update(finished_at=_now(), exit_code=1, error=f"{type(exc).__name__}: {exc}")
            _append_command(paths, entry)
            try:
                if paths["state"].exists():
                    _write_state(paths, status="interrupted", last_error=f"{type(exc).__name__}: {exc}")
            except OSError:
                pass
            raise
        entry.update(finished_at=_now())
        _append_command(paths, entry)
        return 0
    finally:
        lock.release()


def _final_test_body(paths, project_root, config) -> None:
    data.set_stage("final-test")
    freeze_payload = _verify_freeze(paths, project_root, config)
    exp = _exp(config)
    _verify_test_sources(paths, project_root, config, exp, freeze_payload)
    _check_budget(paths, exp, "final")
    block_size = int(freeze_payload.get("rules", {}).get("final_block_origins", FINAL_BLOCK_ORIGINS))
    baseline = data.BaselineSource(config, project_root)
    v4 = data.V4Source(config, project_root)
    scored_by_region, extras, origins_by_region = {}, {}, {}
    resumed_from = {}
    for region in exp["regions"]:
        arrays, metadata = _region_cache(config, project_root, region, "test", identity_hash(paths, region))
        origins = np.asarray(arrays["origin_unix"], dtype=np.int64)
        # Frozen baseline test predictions: hash-frozen at freeze time and
        # grid/label-identical to the test cache elementwise.
        artifact = baseline.test_artifact(region)
        _validate_final_grids(region, artifact, arrays, "Baseline test predictions")
        baseline_q = data.sorted_quantiles(artifact.quantiles)
        v4_artifact = v4.artifact(region, "test")
        _validate_final_grids(region, v4_artifact, arrays, "V4 test predictions")
        v4_q = data.sorted_quantiles(v4_artifact.quantiles)
        z_test = data.build_z(baseline_q, arrays)
        cal_arrays, _ = _region_cache(config, project_root, region, "calibration", identity_hash(paths, region))
        baseline_cal = data.load_prepared_baseline(paths["root"], region, "calibration")
        library_h2 = history.HistoryLibrary.from_segment(
            np.asarray(cal_arrays["origin_unix"], dtype=np.int64), cal_arrays["actual"], baseline_cal,
            data.build_z(baseline_cal, cal_arrays))
        if len(library_h2) and library_h2.origins[-1] >= origins[0]:
            raise ValueError(f"Calibration library overlaps the test grid for {region}")
        models = {}
        for model_id in schema.CALIBRATOR_CONTROLS:
            payload = data.read_json(paths["models"] / model_id / f"{region}.json")
            models[model_id] = CalibratorModel.from_json(payload, schema.CONTROL_SPECS)
        models[schema.STACKER] = StackerModel.from_json(
            data.read_json(paths["models"] / schema.STACKER / f"{region}.json"))
        for model_id in schema.TREE_CONTROLS:
            models[model_id] = {region: controls.load_tree_control(paths["models"] / model_id / region, exp["trees"])}

        # Explicit sequential contract: release matured labels block by
        # block, predict and persist the current block first.
        queue = history.MaturedQueue(origins, arrays["actual"], baseline_q, z_test)
        blocks = _expected_blocks(len(origins), block_size)
        start_index, _manifest = _load_final_progress(
            paths, region, freeze_payload["identity_digest"], blocks, origins, block_size)
        resumed_from[region] = start_index
        region_audit = history.HistoryAudit()
        features_blocks, delta_blocks, s_blocks = [], [], []
        for index in range(start_index, len(blocks)):
            s, e = blocks[index]
            query_end = int(origins[e - 1])
            queue.release_through(query_end)
            library_now = library_h2.concat(queue.released())
            engine = history.FeatureEngine(library_now, audit_sample_every=96)
            arrays_blk = {key: value[s:e] for key, value in arrays.items()}
            z_blk = z_test[s:e]
            bq_blk = baseline_q[s:e]
            features_blk, _result = engine.features_matrix(origins[s:e], z_blk, bq_blk)
            features_recent_blk = engine.features_matrix_recent(z_blk, _result)
            region_audit.merge(engine.audit)
            preds_blk = _predict_all_models(config, project_root, region, arrays_blk, bq_blk,
                                            features_blk, features_recent_blk, v4_q[s:e], models)
            _persist_final_block(paths, region, index, preds_blk, origins[s:e],
                                 freeze_payload["identity_digest"], block_size,
                                 library_len=len(library_now), released_through=query_end)
            features_blocks.append(features_blk)
            delta_blocks.append(preds_blk["_V5_parts"]["delta"])
            s_blocks.append(preds_blk["_V5_parts"]["s"])
            del arrays_blk
            _check_budget(paths, exp, "final")
        if len(features_blocks) < len(blocks):
            # Resume: deterministically replay the already-persisted
            # prefix so diagnostics cover the whole test segment (identical
            # values: same frozen models, same maturity frontier per block).
            missing = len(blocks) - len(features_blocks)
            replay_queue = history.MaturedQueue(origins, arrays["actual"], baseline_q, z_test)
            prefix_features, prefix_delta, prefix_s = [], [], []
            for index in range(missing):
                s, e = blocks[index]
                replay_queue.release_through(int(origins[e - 1]))
                engine = history.FeatureEngine(
                    library_h2.concat(replay_queue.released()), audit_sample_every=96)
                features_blk, _r = engine.features_matrix(origins[s:e], z_test[s:e], baseline_q[s:e])
                parts = models["V5_history"].predict(features_blk, baseline_q[s:e])
                prefix_features.append(features_blk)
                prefix_delta.append(parts["delta"])
                prefix_s.append(parts["s"])
                region_audit.merge(engine.audit)
            features_blocks = prefix_features + features_blocks
            delta_blocks = prefix_delta + delta_blocks
            s_blocks = prefix_s + s_blocks
        # Assemble the single frozen matrix from the persisted blocks.
        predictions = {}
        for model_id in schema.ALL_MODELS:
            parts = []
            for index in range(len(blocks)):
                path = paths["final"] / "blocks" / region / model_id / f"block_{index:05d}.npz"
                with np.load(path, allow_pickle=False) as archive:
                    parts.append(archive["quantiles"])
            predictions[model_id] = np.concatenate(parts, axis=0)
        block_origins = np.concatenate([
            np.load(paths["final"] / "blocks" / region / schema.BASELINE / f"block_{index:05d}.npz",
                    allow_pickle=False)["origin_unix"] for index in range(len(blocks))])
        if not np.array_equal(np.asarray(block_origins, dtype=np.int64), origins):
            raise ValueError(f"Assembled final block origins misaligned for {region}")
        features_full = np.concatenate(features_blocks, axis=0) if features_blocks else \
            np.empty((0, schema.HORIZONS, schema.NF))
        data.atomic_json(paths["audit"] / f"{region}_final_test.json", dict(
            region=region, records=len(library_h2) + len(origins),
            sequential=dict(block_size=block_size, blocks=len(blocks),
                            resumed_from=start_index,
                            queue_released_total=queue.released_count),
            audit=region_audit.summary()))
        rows = {"test": np.ones(len(origins), dtype=bool)}
        scored = _score_segments(predictions, arrays, metadata, rows)
        scored_by_region[region] = scored
        origins_by_region[region] = origins
        extras[region] = dict(
            v5_parts=diagnostics.calibrator_diagnostics(
                dict(delta=np.concatenate(delta_blocks, axis=0),
                     s=np.concatenate(s_blocks, axis=0)), rows["test"]),
            feature_distributions=_feature_distributions_by_group(features_full, rows),
        )
        pred_dir = paths["final"] / "predictions"
        for model_id, quantiles in predictions.items():
            if model_id.startswith("_"):
                continue
            out = pred_dir / model_id / f"{region}_test.npz"
            data.atomic_npz(out, quantiles=quantiles, origin_unix=origins,
                            actual=np.asarray(arrays["actual"]))
        del arrays, cal_arrays
    # Pre-declared primary comparisons with joint 168h paired bootstrap.
    bootstrap = {}
    sensitivity = {}
    for candidate, reference in schema.PRIMARY_COMPARISONS:
        losses_cand, losses_ref = {}, {}
        for region in exp["regions"]:
            quantiles_c = np.load(paths["final"] / "predictions" / candidate / f"{region}_test.npz")["quantiles"]
            quantiles_r = np.load(paths["final"] / "predictions" / reference / f"{region}_test.npz")["quantiles"]
            actual = np.load(paths["final"] / "predictions" / candidate / f"{region}_test.npz")["actual"]
            losses_cand[region] = diagnostics.per_origin_losses(quantiles_c, actual)
            losses_ref[region] = diagnostics.per_origin_losses(quantiles_r, actual)
        bootstrap[f"{candidate}-{reference}"] = diagnostics.paired_block_bootstrap(
            origins_by_region, losses_ref, losses_cand, block_hours=exp["bootstrap"]["block_hours"],
            n_boot=exp["bootstrap"]["n"], seed=exp["bootstrap"]["seed"])
        sensitivity[f"{candidate}-{reference}"] = diagnostics.paired_block_bootstrap(
            origins_by_region, losses_ref, losses_cand, block_hours=exp["bootstrap"]["sensitivity_block_hours"],
            n_boot=exp["bootstrap"]["n"], seed=exp["bootstrap"]["seed"])
    holm = diagnostics.holm_adjust([bootstrap[f"{a}-{b}"]["p_value"][m]
                                    for a, b in schema.PRIMARY_COMPARISONS
                                    for m in schema.PRIMARY_METRICS])
    data.atomic_json(paths["final"] / "metrics.json",
                     dict(scored=scored_by_region, extras=extras,
                          bootstrap=bootstrap, bootstrap_sensitivity_24h=sensitivity,
                          holm_adjusted_p=dict(
                              labels=[f"{a}-{b}:{m}" for a, b in schema.PRIMARY_COMPARISONS
                                      for m in schema.PRIMARY_METRICS], values=holm)))
    data.atomic_json(paths["final"] / "complete.json", dict(
        completed_at=_now(), identity_digest=freeze_payload["identity_digest"],
        model_files_verified=len(freeze_payload["model_files"]),
        sequential=dict(block_size=block_size, resumed_from=resumed_from)))
    _write_state(paths, stage="final_test_complete", status="ok")
    print("final-test complete (single frozen matrix, sequential matured-queue)")


def identity_hash(paths, region):
    return data.read_json(paths["identity"])["cache_hashes"][region]


def report_stage(run_root) -> int:
    project_root = Path.cwd().resolve()
    paths = _run_paths(run_root)
    config = _load_config(paths["config"])
    data.set_stage("report")
    _verify_identity(paths, project_root, config)
    state = _read_state(paths)
    if state.get("stage") not in ("dev_passed", "dev_failed", "final_test_complete"):
        raise PermissionError(f"report requires a completed dev or final-test stage (current: {state.get('stage')!r})")
    report.render_final_report(paths)
    print(f"report written: {paths['reports']}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="V5 Calibrated History experiment")
    parser.add_argument("stage",
                        choices=("prepare", "dev", "dev-region", "dev-finalize", "freeze", "final-test", "report"))
    parser.add_argument("--config")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--region", choices=REGION_ORDER,
                        help="region shard for dev-region (one of the config regions)")
    args = parser.parse_args(argv)
    paths = _run_paths(args.run_root)
    # Round-8 High fix: the execution locks own the whole budget lifecycle.
    # dev-region/dev-finalize/freeze/final-test begin AFTER acquiring their
    # lock and finish/status/commands BEFORE releasing it, so main must not
    # touch the budget around them (a refused duplicate worker used to fold
    # the live owner's records before the stage function even ran).  main only
    # records the unlocked overhead stages (prepare/report) here.
    if args.stage in ("prepare", "report"):
        cpu_started, wall_started = _process_cpu(), time.time()
        entry = dict(stage=args.stage, argv=list(map(str, vars(args).values())), started_at=_now())
        try:
            if args.stage == "prepare":
                if not args.config:
                    parser.error("prepare requires --config")
                code = prepare(args.config, args.run_root)
            else:
                code = report_stage(args.run_root)
        except BaseException as exc:
            entry.update(finished_at=_now(), exit_code=1, error=f"{type(exc).__name__}: {exc}",
                         cpu_seconds=_process_cpu() - cpu_started, wall_seconds=time.time() - wall_started)
            try:
                _append_command(paths, entry)
            except OSError:
                pass
            raise
        entry.update(finished_at=_now(), exit_code=code,
                     cpu_seconds=_process_cpu() - cpu_started, wall_seconds=time.time() - wall_started)
        _append_command(paths, entry)
        return code
    if args.stage == "dev":
        return dev(args.run_root)
    if args.stage == "dev-region":
        if not args.region:
            parser.error("dev-region requires --region")
        return dev_region(args.run_root, args.region)
    if args.stage == "dev-finalize":
        return dev_finalize(args.run_root)
    if args.stage == "freeze":
        return freeze(args.run_root)
    if args.stage == "final-test":
        return final_test(args.run_root)
    return report_stage(args.run_root)


def _append_command(paths, entry: dict, region: str | None = None) -> None:
    path = _region_dir(paths, region) / "commands.jsonl" if region else paths["commands"]
    if not path.parent.exists():
        return
    with path.open("a") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
