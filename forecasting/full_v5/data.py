"""Read-only frozen-artefact access, strict time subsets, and the test guard.

Nothing in this module trains anything or writes to pre-existing run
directories.  The 2023-2024 test split can only be opened from the final-test
stage; every other stage raises before any file handle is created.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..aligned_inputs import INPUT_KEYS, AlignedSplit
from ..aligned_scoring import load_prediction_artifact
from . import schema

# --- small local helpers (no aligned_experiment import: it pulls torch workers)
def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def atomic_npz(path, **arrays) -> None:
    """Write an .npz via temp file + atomic replace (crash-safe artefacts)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # the temp name must end in .npz or numpy appends the suffix itself
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temp, **arrays)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def read_json(path):
    return json.loads(Path(path).read_text())


_STAGE = {"current": "prepare"}


def set_stage(name: str) -> None:
    _STAGE["current"] = name


def current_stage() -> str:
    return _STAGE["current"]


def require_stage(*allowed: str, what: str) -> None:
    if _STAGE["current"] not in allowed:
        raise PermissionError(
            f"{what} is only permitted in stage(s) {sorted(allowed)}; current stage is {_STAGE['current']!r}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


# --- cache -----------------------------------------------------------------
def load_cache_split(cache_root, region: str, split: str, expected_hashes_hash: str | None = None):
    """Verify and load one split directory of the frozen shared data cache.

    Only the requested split's files are hashed, so development stages never
    touch test files even for integrity verification.  The test split itself
    can only be opened from the final-test stage.
    """
    if split == "test":
        require_stage("final-test", what=f"loading the {region} test split from the shared cache")
    root = Path(cache_root) / region
    manifest = read_json(root / "hashes.json")
    if expected_hashes_hash is not None and sha256_file(root / "hashes.json") != expected_hashes_hash:
        raise ValueError(f"Cache manifest hash changed for {region}")
    prefix = f"{split}/"
    entries = {name: digest for name, digest in manifest.items() if name.startswith(prefix)}
    if not entries:
        raise ValueError(f"No cache entries for split {split}")
    folder = root / split
    for name, digest in entries.items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Cached data changed: {region}/{name}")
    arrays = {k.removeprefix("input__"): np.load(folder / f"input__{k}.npy")
              for k in INPUT_KEYS}
    arrays.update(
        actual=np.load(folder / "actual.npy"),
        target=np.load(folder / "target.npy"),
        origin_unix=np.load(folder / "origin_unix.npy"),
        target_time_unix=np.load(folder / "target_time_unix.npy"),
    )
    metadata = read_json(folder / "metadata.json")
    if metadata.get("split") != split or metadata.get("region") != region:
        raise ValueError("Cache metadata mismatch")
    return arrays, metadata


def cache_hashes_from_jobs(jobs: dict, region: str) -> str:
    value = jobs["cache_hashes"][region]
    return value if isinstance(value, str) else value["hashes_json_sha256"]


def verify_data_hashes(jobs: dict, project_root) -> dict:
    """Recompute every external data file hash recorded by the frozen run."""
    verified = {}
    for path, digest in sorted(jobs.get("data_hashes", {}).items()):
        target = Path(path) if Path(path).is_absolute() else Path(project_root) / path
        if not target.exists():
            target = (Path(project_root) / path).resolve()
        if not target.exists():
            raise FileNotFoundError(f"External data file recorded in jobs.json is missing: {path}")
        actual = sha256_file(target)
        if actual != digest:
            raise ValueError(f"External data file changed: {path}")
        verified[str(target)] = actual
    if not verified:
        raise ValueError("jobs.json contains no data hashes to verify")
    return verified


def check_availability_invariant(csv_path) -> dict:
    """The publication-time proxy: available_at == delivery_start + 1h."""
    frame = pd.read_csv(csv_path, usecols=lambda c: c in ("delivery_start_aest", "available_at_aest"))
    delivery = pd.to_datetime(frame["delivery_start_aest"], utc=True)
    available = pd.to_datetime(frame["available_at_aest"], utc=True)
    if not (available == delivery + pd.Timedelta(hours=schema.AVAILABILITY_LAG_HOURS)).all():
        raise ValueError("Availability proxy invariant violated in region csv")
    return dict(rows=int(len(frame)), lag_hours=schema.AVAILABILITY_LAG_HOURS)


# --- segment logic ---------------------------------------------------------
def window_in_segment(origins: np.ndarray, start: int, end: int) -> np.ndarray:
    """Complete 24h label window inside [start, end): origin >= start and
    origin + 23h < end (hourly grid -> origin <= end - 24h)."""
    origins = np.asarray(origins, dtype=np.int64)
    return (origins >= start) & (origins <= end - schema.HORIZONS * 3600)


def fit_available(origins: np.ndarray, cutoff: int) -> np.ndarray:
    """Labels used for fitting must be available strictly before the cutoff:
    a_j = origin + 24h < cutoff."""
    origins = np.asarray(origins, dtype=np.int64)
    return origins + schema.HORIZONS * 3600 < cutoff


def segment_rows(origins: np.ndarray, segment: str) -> np.ndarray:
    start, end = schema.segment_bounds()[segment]
    return window_in_segment(origins, start, end)


def assign_segments(origins: np.ndarray) -> dict[str, np.ndarray]:
    return {name: segment_rows(origins, name) for name in ("dfit", "v1", "v2")}


def aest_hour(unix) -> np.ndarray:
    return ((np.asarray(unix, dtype=np.int64) // 3600 + 10) % 24).astype(np.int64)


def aest_dow(unix) -> np.ndarray:
    days = (np.asarray(unix, dtype=np.int64) + 10 * 3600) // 86400
    return ((days + 3) % 7).astype(np.int64)


# --- feature construction --------------------------------------------------
def sorted_quantiles(raw: np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(raw, dtype=np.float64), axis=-1)


def build_z(baseline_q_sorted: np.ndarray, arrays: dict) -> np.ndarray:
    """Spec 4.1: the fixed 16-column current state, [N, 24, 16]."""
    bq = np.asarray(baseline_q_sorted, dtype=np.float64)
    n = len(bq)
    fe = np.asarray(arrays["future_exogenous"], dtype=np.float64)
    ctf = np.asarray(arrays["ctf_exogenous"], dtype=np.float64)
    qe = np.asarray(arrays["quantile_exogenous"], dtype=np.float64)
    targets = np.asarray(arrays["target_time_unix"], dtype=np.int64)

    def exo(block, column):
        raw = np.clip(block[..., column], -schema.EXOGENOUS_CLIP, schema.EXOGENOUS_CLIP) / schema.EXOGENOUS_CLIP
        return raw

    z = np.empty((n, schema.HORIZONS, schema.NZ), dtype=np.float64)
    z[:, :, 0] = 1.0
    z[:, :, 1] = np.clip(bq[..., 2] / schema.PRICE_SCALE, -1.0, 1.0)
    z[:, :, 2] = np.clip((bq[..., 3] - bq[..., 1]) / schema.PRICE_SCALE, 0.0, 1.0)
    # future_exogenous column order (frozen, audited): 0 max_spare, 1 predispatch_rrp,
    # 2 predispatch_total_demand, 3 predispatch_available_generation, 4 predispatch_net_interchange,
    # 5 predispatch_covered, 6 ic_net_import, 7 ic_import_room, 8 ic_export_room,
    # 9 ic_main_import_room, 10 ic_covered.
    z[:, :, 3] = exo(fe, 1)
    z[:, :, 4] = exo(fe, 2)
    z[:, :, 5] = exo(fe, 3)
    z[:, :, 6] = exo(fe, 0)
    z[:, :, 7] = exo(ctf, 0)
    z[:, :, 8] = exo(fe, 7)
    z[:, :, 9] = exo(qe, 1)  # max_spare shortfall below the train q0.05
    z[:, :, 10] = exo(fe, 5)
    z[:, :, 11] = exo(fe, 10)
    hour = aest_hour(targets)
    dow = aest_dow(targets)
    z[:, :, 12] = np.sin(2 * np.pi * hour / 24.0)
    z[:, :, 13] = np.cos(2 * np.pi * hour / 24.0)
    z[:, :, 14] = (dow >= 5).astype(np.float64)
    z[:, :, 15] = np.sin(2 * np.pi * dow / 7.0)
    if not np.isfinite(z).all():
        raise ValueError("Non-finite z features")
    return z


def flatten_common(arrays: dict) -> np.ndarray:
    """The canonical shared columns [N, d], cache block order."""
    n = len(arrays["origin_unix"])
    blocks = [np.asarray(arrays[k], dtype=np.float32).reshape(n, -1) for k in INPUT_KEYS]
    return np.concatenate(blocks, axis=1)


# --- baseline predictions --------------------------------------------------
class BaselineSource:
    """seed-2026 frozen LightGBM, or a precomputed substitute for tests."""

    def __init__(self, config, project_root):
        exp = config["v5_calibrated_history"]
        self.mode = exp["baseline"]["mode"]
        self.run_root = Path(exp["baseline"].get("run_root", "."))
        self.seed = int(exp["baseline"].get("seed", 2026))
        self.threads = int(exp["baseline"].get("threads", 4))
        self.project_root = Path(project_root)
        self._predictors = {}

    def calibration_quantiles(self, region: str, arrays: dict, metadata: dict) -> np.ndarray:
        if self.mode == "precomputed":
            require_stage("prepare", "dev", "freeze", "final-test", "report",
                          what="precomputed baseline calibration read")
            return sorted_quantiles(self._load_precomputed(region, "calibration"))
        from ..lightgbm.predictor import FrozenLightGBMPredictor
        predictor = self._predictors.get(region)
        if predictor is None:
            models = self.run_root / "runs" / "lightgbm" / f"seed{self.seed}" / region / "models"
            predictor = FrozenLightGBMPredictor(models, seed=self.seed, region=region, threads=self.threads)
            self._predictors[region] = predictor
        inputs = {k: arrays[k] for k in INPUT_KEYS}
        split = AlignedSplit(inputs, arrays["actual"], arrays["origin_unix"], arrays["target_time_unix"],
                             metadata, arrays["target"])
        return predictor.predict(split)

    def test_quantiles(self, region: str) -> np.ndarray:
        require_stage("final-test", what="reading frozen baseline test predictions")
        return sorted_quantiles(self.test_artifact(region).quantiles)

    def test_artifact(self, region: str):
        """The frozen baseline test predictions plus their origin/target/actual
        grids (final-test only).  Precomputed substitutes must carry the same
        grids so the final identity check can compare them elementwise."""
        require_stage("final-test", what="reading frozen baseline test predictions")
        path = self.test_artifact_path(region)
        if self.mode == "precomputed":
            from types import SimpleNamespace
            with np.load(path, allow_pickle=False) as archive:
                missing = {"quantiles", "origin_unix", "target_time_unix", "actual"} - set(archive.files)
                if missing:
                    raise ValueError(
                        f"Precomputed baseline test file lacks grid arrays {sorted(missing)}")
                return SimpleNamespace(quantiles=archive["quantiles"],
                                       origin_unix=archive["origin_unix"],
                                       target_time_unix=archive["target_time_unix"],
                                       actual=archive["actual"])
        return load_prediction_artifact(path)

    def test_artifact_path(self, region: str) -> Path:
        if self.mode == "precomputed":
            return self._precomputed_path(region, "test")
        return self.run_root / "runs" / "lightgbm" / f"seed{self.seed}" / region / "test.npz"

    def provenance(self, region: str) -> dict:
        if self.mode == "precomputed":
            path = self._precomputed_path(region, "calibration")
            return dict(mode="precomputed", region=region, files={str(path): sha256_file(path)})
        predictor = self._predictors.get(region)
        if predictor is None:
            raise ValueError(f"Predictor for {region} not loaded yet")
        provenance = dict(predictor.provenance)
        provenance["mode"] = "frozen_lgbm"
        return provenance

    def _precomputed_path(self, region: str, split: str) -> Path:
        return self.run_root / f"{region}_{split}.npz"

    def trusted_test_hash(self, region: str) -> str:
        """Expected test-prediction hash from the trusted sidecar manifest.

        Round-6 M1: freeze records this pre-bound value instead of reading
        the test bytes; only the authorized final-test compares the actual
        file against it.  Production reads the frozen run's own status
        manifest; the precomputed test mode reads a harness-written
        test_source_hashes.json manifest.
        """
        if self.mode == "precomputed":
            return _trusted_manifest_hash(self.run_root / "test_source_hashes.json", region)
        status = self.run_root / "runs" / "lightgbm" / f"seed{self.seed}" / region / "status.json"
        digest = read_json(status).get("artifact_hashes", {}).get("test.npz")
        if not digest:
            raise ValueError(f"Trusted baseline status manifest lacks a test.npz hash for {region}: {status}")
        return digest

    def trusted_test_hash_manifest(self, region: str) -> dict:
        """Identity (path + sha256) of the trusted manifest itself."""
        if self.mode == "precomputed":
            path = self.run_root / "test_source_hashes.json"
        else:
            path = self.run_root / "runs" / "lightgbm" / f"seed{self.seed}" / region / "status.json"
        return dict(path=str(path), sha256=sha256_file(path))

    def _load_precomputed(self, region: str, split: str) -> np.ndarray:
        # The stage guard fires BEFORE any file handle is created: the test
        # split is final-test only (Round-6 M1).
        if split == "test":
            require_stage("final-test", what="precomputed baseline test read")
        path = self._precomputed_path(region, split)
        with np.load(path, allow_pickle=False) as archive:
            return archive["quantiles"]


class V4Source:
    """Read-only v4_full prediction artefacts for the A_stack control."""

    def __init__(self, config, project_root):
        exp = config["v5_calibrated_history"]["v4"]
        self.mode = exp["mode"]
        self.run_root = Path(exp.get("run_root", "."))
        self.variant = exp.get("variant", "v4_full")
        self.seed = int(exp.get("seed", 2026))

    def artifact_path(self, region: str, split: str) -> Path:
        if self.mode == "precomputed":
            return self.run_root / f"{region}_{split}.npz"
        return (self.run_root / "runs" / self.variant / f"seed{self.seed}" / region
                / f"{split}_predictions.npz")

    def artifact(self, region: str, split: str):
        """Load one V4 prediction artifact.  The test split is final-test only
        and the guard fires before any file handle is created."""
        if split == "test":
            require_stage("final-test", what=f"V4 {split} artefact read")
        else:
            require_stage("prepare", "dev", "freeze", "final-test", "report",
                          what=f"V4 {split} artefact read")
        return load_prediction_artifact(self.artifact_path(region, split))

    def quantiles(self, region: str, split: str, expect_origins: np.ndarray) -> np.ndarray:
        artifact = self.artifact(region, split)
        if not np.array_equal(artifact.origin_unix, np.asarray(expect_origins, dtype=np.int64)):
            raise ValueError(f"V4 {split} origins do not align with the cache grid for {region}")
        return sorted_quantiles(artifact.quantiles)

    def file_digest(self, region: str, split: str) -> str:
        # Hashing a file reads it: the same stage guard as a full load applies.
        if split == "test":
            require_stage("final-test", what=f"V4 {split} artefact digest")
        return sha256_file(self.artifact_path(region, split))

    def trusted_test_hash(self, region: str) -> str:
        """Expected test-prediction hash from the V4 run's output_hashes.json
        (Round-6 M1: freeze records this value; final-test verifies the file)."""
        if self.mode == "precomputed":
            return _trusted_manifest_hash(self.run_root / "test_source_hashes.json", region)
        path = self.run_root / "runs" / self.variant / f"seed{self.seed}" / region / "output_hashes.json"
        digest = read_json(path).get("test_predictions.npz")
        if not digest:
            raise ValueError(f"V4 output_hashes.json lacks a test_predictions.npz hash for {region}: {path}")
        return digest

    def trusted_test_hash_manifest(self, region: str) -> dict:
        """Identity (path + sha256) of the trusted manifest itself."""
        if self.mode == "precomputed":
            path = self.run_root / "test_source_hashes.json"
        else:
            path = self.run_root / "runs" / self.variant / f"seed{self.seed}" / region / "output_hashes.json"
        return dict(path=str(path), sha256=sha256_file(path))


def _trusted_manifest_hash(manifest_path, region: str) -> str:
    """Expected artifact hash recorded in a trusted sidecar manifest."""
    payload = read_json(manifest_path)
    digest = payload.get(region)
    if not digest:
        raise ValueError(f"{Path(manifest_path).name} lacks a trusted hash for {region}")
    return digest


def load_prepared_baseline(run_root, region: str, split: str) -> np.ndarray:
    """Load the prepare-stage H2 baseline predictions from the V5 run root."""
    if split == "test":
        require_stage("final-test", what="prepared baseline test read")
    path = Path(run_root) / "baseline_predictions" / f"{region}_{split}.npz"
    with np.load(path, allow_pickle=False) as archive:
        return sorted_quantiles(archive["quantiles"])


def git_identity(project_root) -> dict:
    import subprocess
    def run(*args):
        return subprocess.run(["git", *args], cwd=project_root, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return dict(head=run("rev-parse", "HEAD"), branch=run("branch", "--show-current"),
                    dirty=bool(run("status", "--porcelain")))
    except (subprocess.CalledProcessError, OSError) as exc:
        return dict(head=None, branch=None, dirty=None, note=f"not a git worktree: {exc.__class__.__name__}")
