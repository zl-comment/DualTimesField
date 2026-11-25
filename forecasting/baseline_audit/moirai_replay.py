"""Real-weight old/new Moirai comparison on untouched production test caches.

Use the original Moirai interpreter, HF_HOME and GPU. This diagnostic never
trains, tunes, republishes the main table, or writes to the original run.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from forecasting.aligned_scoring import load_prediction_artifact, score_predictions
from forecasting.common.cache import load_split
from forecasting.common.foundation_api import PredictionView
from forecasting.common.foundation_weights import verified_checkpoint
from forecasting.moirai2_zeroshot import model as native
from forecasting.moirai2_zeroshot.adapter import Adapter
from forecasting.baseline_audit.recompute import digest, fingerprint, git, write

ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "e16f9b396f9143f88f42da2031792495170e3be6"
REGIONS = ("NSW1", "QLD1", "TAS1")
MODEL = "moirai2_zeroshot"
ARCHIVE_RTOL = 1e-3  # Existing foundation_audit Moirai sampled-replay tolerance.


def comparison(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Prediction shape mismatch or nonfinite values")
    delta = np.abs(left - right)
    scale = max(1., float(np.abs(right).max()))
    return {"array_equal": bool(np.array_equal(left, right)),
            "max_abs_diff": float(delta.max()), "max_relative_diff": float(delta.max()) / scale,
            "mean_abs_diff": float(delta.mean()), "n_values": int(delta.size)}


class RecordingPredictor:
    """Observe real GluonTS forecasts without changing model inputs/outputs."""
    def __init__(self, predictor):
        self.predictor = predictor
        self.reset()

    def reset(self):
        self.starts, self.quantiles = [], []

    def predict(self, dataset):
        for forecast in self.predictor.predict(dataset):
            self.starts.append(forecast.start_date.ordinal)
            self.quantiles.append(np.stack([forecast.quantile(q) for q in native.NATIVE_QUANTILE_LEVELS]))
            yield forecast


def expected_start_ordinals(origins):
    local = pd.to_datetime(origins, unit="s", utc=True).tz_convert("Australia/Brisbane").tz_localize(None)
    return local.to_period("h").asi8


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if git("status", "--porcelain"):
        raise RuntimeError("Commit diagnostic code before real inference")
    root, output = args.run_root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root / "jobs.json").read_text())
    config = yaml.safe_load((root / "config.yaml").read_text())
    config["project_root"] = manifest["project_root"]
    old_path = root / "source/forecasting/moirai2_zeroshot/adapter.py"
    if digest(old_path) != manifest["source_files"]["forecasting/moirai2_zeroshot/adapter.py"]:
        raise ValueError("Archived adapter source changed")
    if digest(root / "config.yaml") != manifest["config_hash"]:
        raise ValueError("Archived configuration changed")
    lock = manifest["model_env_locks"][MODEL]
    lock_path = Path(manifest["project_root"]) / lock["lock_file"]
    if digest(lock_path) != lock["sha256"]:
        raise ValueError("Archived environment lock changed")
    snapshot, weights = verified_checkpoint(native.__file__, os.environ["HF_HOME"])
    source_paths = [ROOT / p for p in git("ls-files", "forecasting").splitlines()
                    if Path(p).suffix in (".py", ".json", ".yaml")]
    input_paths = [root / "jobs.json", root / "config.yaml", old_path, lock_path,
                   snapshot / weights["file"], snapshot / "config.json"]
    splits, archives = {}, {}
    for region in REGIONS:
        splits[region] = load_split(root, manifest, region, "test")
        cache = root / "data_cache" / region
        input_paths += [cache / "hashes.json", *sorted((cache / "test").glob("*"))]
        for seed in (2026, 2027, 2028):
            cell = root / "runs" / MODEL / f"seed{seed}" / region
            status = json.loads((cell / "status.json").read_text())
            for name in ("test.npz", "metrics.json"):
                if digest(cell / name) != status["artifact_hashes"][name]:
                    raise ValueError(f"Archived artifact changed: {cell / name}")
            artifact = load_prediction_artifact(cell / "test.npz")
            if artifact.metadata["model_metadata"]["adapter_identity"]["weights"]["sha256"] != weights["sha256"]:
                raise ValueError("Checkpoint differs from original production run")
            for field, values in (("origin_unix", splits[region].origin_unix),
                                  ("target_time_unix", splits[region].target_time_unix),
                                  ("actual", splits[region].actual)):
                if not np.array_equal(getattr(artifact, field), values):
                    raise ValueError(f"Original predictions/cache mismatch: {region}/{seed}/{field}")
            archives[region, seed] = artifact
            input_paths += [cell / name for name in ("test.npz", "metrics.json", "status.json")]
    import torch
    import uni2ts
    import gluonts
    import huggingface_hub.hub_mixin as hub_mixin
    package_paths = [Path(hub_mixin.__file__)]
    for package in (uni2ts, gluonts):
        package_paths += sorted(Path(package.__file__).parent.rglob("*.py"))
    source_hashes, input_hashes = fingerprint(source_paths), fingerprint(input_paths)
    package_hashes = fingerprint(package_paths)
    run_manifest = {
        "baseline_commit": BASE_COMMIT, "execution_commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"), "source_sha256": source_hashes,
        "source_fingerprint": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
        "input_sha256": input_hashes, "installed_package_source_sha256": package_hashes,
        "weights": weights, "packages": native.package_versions(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0),
        "plan": "First production batch smoke check, then ALL remaining test origins in each region; original batch/chunk=512; old and fixed adapters share the same frozen network; compare all nine native quantiles and forecast start times; compare q50 against all nine archived seed/region cells. No tuning.",
        "acceptance": {"old_new": "exact array equality", "archive_relative_tolerance": ARCHIVE_RTOL,
                       "timestamp": "fixed native start equals actual origin; old native start is 72 hours later"},
    }
    write(output / "manifest.json", run_manifest)
    # Execute the SHA-verified archived adapter, but share the same verified
    # network/predictor to isolate the start-date and output-layout changes.
    scope = {"__name__": "forecasting.moirai2_zeroshot._archived_replay",
             "__package__": "forecasting.moirai2_zeroshot"}
    exec(compile(old_path.read_text(), str(old_path), "exec"), scope)
    legacy_class = scope["Adapter"]
    torch.set_num_threads(config["experiment"]["cpu_threads_per_job"])
    started, results, adapter = time.monotonic(), {}, None
    try:
        for region in REGIONS:
            split = splits[region]
            view = PredictionView(split.inputs, split.origin_unix, split.target_time_unix, dict(split.metadata))
            context = SimpleNamespace(config=config, test_view=view, device="cuda:0")
            if adapter is None:
                adapter = Adapter(context)
                recorder = RecordingPredictor(adapter.predictor)
                adapter.predictor = recorder
            adapter.context = context
            legacy = legacy_class.__new__(legacy_class)
            legacy.context, legacy.batch_origins, legacy.predictor = context, adapter.batch_origins, recorder
            batch_size = adapter.batch_origins
            if batch_size != 512 or config["experiment"]["chunk_size"][MODEL] != batch_size:
                raise ValueError("Production batching contract changed")
            old_points, fixed_points, old_aux, fixed_aux, old_starts, fixed_starts = [], [], [], [], [], []
            native_max, native_count = 0., 0
            for start in range(0, len(view), batch_size):
                indices = np.arange(start, min(start + batch_size, len(view)))
                recorder.reset()
                old = legacy.predict_chunk(indices)["point"]
                oq, ot = np.asarray(recorder.quantiles).copy(), np.asarray(recorder.starts).copy()
                oa = legacy.auxiliary_quantiles().transpose(0, 2, 1).copy()
                recorder.reset()
                fixed = adapter.predict_chunk(indices)["point"]
                fq, ft = np.asarray(recorder.quantiles), np.asarray(recorder.starts)
                fa = adapter.auxiliary_quantiles().copy()
                checked = comparison(fq, oq)
                if not checked["array_equal"] or not np.array_equal(old, fixed) or not np.array_equal(oa, fa):
                    raise ValueError(f"Old/new numeric divergence at {region}/{start}: {checked}")
                expected = expected_start_ordinals(view.origin_unix[indices])
                if not np.array_equal(ft, expected) or not np.array_equal(ot, expected + 72):
                    raise ValueError(f"Unexpected native forecast dates at {region}/{start}")
                archived = comparison(fixed, archives[region, 2026].native_point[indices])
                if start == 0 and archived["max_relative_diff"] > ARCHIVE_RTOL:
                    raise ValueError(f"First-batch archive replay failed: {region}: {archived}")
                native_max = max(native_max, checked["max_abs_diff"])
                native_count += checked["n_values"]
                for destination, value in ((old_points, old), (fixed_points, fixed), (old_aux, oa),
                                           (fixed_aux, fa), (old_starts, ot), (fixed_starts, ft)):
                    destination.append(value.copy())
                if start == 0 or start // batch_size % 8 == 0:
                    print(f"{region}: {indices[-1]+1}/{len(view)} origins; old/new native max diff={native_max}; archive batch max diff={archived['max_abs_diff']}", flush=True)
            fixed = np.concatenate(fixed_points)
            archived_checks, score_checks = {}, {}
            for seed in (2026, 2027, 2028):
                artifact = archives[region, seed]
                archived_checks[str(seed)] = comparison(fixed, artifact.native_point)
                score_checks[str(seed)] = {"archive": score_predictions(artifact),
                                           "fixed": score_predictions(replace(artifact, native_point=fixed))}
            results[region] = {
                "n_origins": len(view), "n_horizons": 24, "batch_size": batch_size,
                "old_new_point": comparison(fixed, np.concatenate(old_points)),
                "old_new_auxiliary": comparison(np.concatenate(fixed_aux), np.concatenate(old_aux)),
                "old_new_all_nine_native_quantiles": {"array_equal": True, "max_abs_diff": native_max, "n_values": native_count},
                "all_fixed_native_forecast_starts_correct": True, "all_old_native_starts_late_hours": 72,
                "archived_seed_comparisons": archived_checks, "scores": score_checks,
                "archive_replay_pass": all(v["max_relative_diff"] <= ARCHIVE_RTOL for v in archived_checks.values()),
            }
            np.savez_compressed(output / f"{region}_comparison.npz", origin_unix=view.origin_unix,
                target_time_unix=view.target_time_unix, actual=split.actual,
                old_point=np.concatenate(old_points), fixed_point=fixed,
                old_auxiliary=np.concatenate(old_aux), fixed_auxiliary=np.concatenate(fixed_aux),
                auxiliary_levels=np.array([.1, .5, .9]),
                old_native_start_hour=np.concatenate(old_starts), fixed_native_start_hour=np.concatenate(fixed_starts))
            write(output / "progress.json", results)
        if (fingerprint(source_paths) != source_hashes or fingerprint(input_paths) != input_hashes
                or fingerprint(package_paths) != package_hashes):
            raise ValueError("Sources, installed implementation, weights or frozen inputs changed")
        result = {"regions": results, "seconds": time.monotonic()-started,
                  "full_test_replay": True, "originals_and_source_unchanged": True,
                  "passed": all(r["archive_replay_pass"] for r in results.values()),
                  "main_table_republished": False}
        write(output / "results.json", result)
        write(output / "completed.json", {"output_sha256": fingerprint(sorted(output.glob("*")))})
        print(f"Completed full replay: passed={result['passed']}, seconds={result['seconds']:.1f}", flush=True)
    except BaseException as exc:
        write(output / "failed.json", {"error": repr(exc), "completed_regions": results})
        raise


if __name__ == "__main__":
    main()
