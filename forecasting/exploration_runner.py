"""Isolated, resumable workers for the forecasting exploration manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash():
    digest = hashlib.sha256()
    for path in sorted([*ROOT.joinpath("forecasting").rglob("*.py"),
                        *ROOT.joinpath("src").rglob("*.py")]):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path must be inside project: {value}")
    return path


def job_identity(job, expected_source):
    value = {"job": job, "source_hash": expected_source,
             "config_hash": file_hash(project_path(job["config"]))}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def completed(job, expected_source):
    output = project_path(job["output"])
    marker = output / "run_metadata.json"
    if not marker.exists():
        return False
    try:
        metadata = json.loads(marker.read_text())
        if metadata.get("state") != "completed" or metadata.get("identity") != job_identity(job, expected_source):
            return False
        required = {"validation.npz", "test.npz", "metrics.json"}
        if not required.issubset(metadata.get("artifact_hashes", {})):
            return False
        if "checkpoint_directory" in metadata:
            checkpoint = Path(metadata["checkpoint_directory"]) / "best_model.pt"
            if not checkpoint.is_file() or file_hash(checkpoint) != metadata.get("checkpoint_hash"):
                return False
        return all((output / name).is_file() and file_hash(output / name) == digest
                   for name, digest in metadata["artifact_hashes"].items())
    except (ValueError, OSError, KeyError):
        return False


def training_output(job):
    from .datasets import load_forecast_config
    config = load_forecast_config(project_path(job["config"]))
    if job["seed"] == 2026 and config["training"].get("seed") != 2026:
        raise ValueError("The default seed 2026 job requires training.seed=2026")
    suffix = "" if job["seed"] == 2026 else f"_seed{job['seed']}"
    return project_path(config["training"]["output_directory"] + suffix) / job["region"]


def export_dual(job, checkpoint_dir, output):
    import numpy as np
    import torch
    from .datasets import build_region_datasets, load_forecast_config
    from .evaluate_paper_metrics import collect_predictions
    from .train import build_model, resolve_device
    config_path = project_path(job["config"])
    config = load_forecast_config(config_path)
    device = resolve_device(config["training"]["device"])
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.set_epoch(checkpoint.get("model_epoch", checkpoint["best_epoch"]))
    datasets = build_region_datasets(config_path, job["region"])
    event_scores = {}
    for split in ("validation", "test"):
        dataset = datasets[split]
        prediction = collect_predictions(model, dataset, device, 1024, include_diagnostics=True)
        arrays = {name: prediction[name] for name in ("point", "quantile", "actual")}
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise FloatingPointError(f"Non-finite {split} forecasts")
        arrays.update(origin_unix=dataset.delivery_unix_seconds[np.asarray(dataset.origin_indices)],
                      quantile_levels=np.asarray(model.quantiles), seed=job["seed"], region=job["region"])
        for key in ("dgf_fusion_weight", "dgf_quantile_fusion_weight"):
            if key in prediction:
                arrays[key] = prediction[key]
        if model.event_field is not None:
            from .event_metrics import _scores
            event_scores[split] = {}
            for name in ("spike", "trough"):
                probability = prediction[f"event_probability_{name}"].astype(np.float64)
                if not np.isfinite(probability).all():
                    raise FloatingPointError(f"Non-finite event probabilities: {split}/{name}")
                arrays[f"prob_{name}"] = probability
                raw = arrays["actual"]
                labels = raw > config["loss"].get("spike_threshold", 300.) if name == "spike" else raw < config["loss"].get("trough_threshold", 0.)
                scores = _scores(probability, labels)
                event_scores[split][name] = {k: v if np.isfinite(v) else None for k, v in scores.items()}
        temporary = output / f"{split}.npz.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, output / f"{split}.npz")
    if event_scores:
        atomic_json(output / "events.json", {"seed": job["seed"], "region": job["region"],
                    "scores": event_scores})
    if checkpoint_dir != output:
        import shutil
        shutil.copy2(checkpoint_dir / "metrics.json", output / "metrics.json")


def validate_predictions(output, job):
    import numpy as np
    for split in ("validation", "test"):
        with np.load(output / f"{split}.npz", allow_pickle=False) as data:
            event_control = job.get("baseline_kind") == "event_controls"
            predictions = ("logistic_prob_spike", "logistic_prob_trough",
                           "persistence_prob_spike", "persistence_prob_trough") if event_control else ("point", "quantile")
            for name in (*predictions, "actual", "origin_unix", "quantile_levels"):
                if not np.isfinite(data[name]).all():
                    raise FloatingPointError(f"Non-finite {split}/{name}")
            shape = data["actual"].shape
            if event_control:
                if any(data[name].shape != shape or np.any((data[name] < 0) | (data[name] > 1))
                       for name in predictions):
                    raise ValueError(f"Invalid event control probabilities: {split}")
            elif data["point"].shape != shape or data["quantile"].shape != (*shape, len(data["quantile_levels"])):
                raise ValueError(f"Invalid prediction shapes: {split}")
            if len(data["origin_unix"]) != shape[0] or not np.all(np.diff(data["origin_unix"]) > 0):
                raise ValueError(f"Invalid forecast origins: {split}")
            if int(data["seed"]) != job["seed"] or str(data["region"]) != job["region"]:
                raise ValueError(f"Wrong seed/region in {split}")


def run_worker(job, expected_source):
    os.chdir(ROOT)
    if source_hash() != expected_source:
        raise RuntimeError("Source changed before worker startup")
    if completed(job, expected_source):
        print(json.dumps({"job": job["id"], "state": "already_completed"}), flush=True)
        return
    output = project_path(job["output"])
    identity = job_identity(job, expected_source)
    marker = output / "run_metadata.json"
    prior = json.loads(marker.read_text()) if marker.exists() else None
    if output.exists() and any(output.iterdir()) and (prior is None or prior.get("identity") != identity):
        raise RuntimeError(f"Refusing to overwrite unowned experiment output: {output}")
    if job["kind"] != "baseline":
        train_dir = training_output(job)
        if train_dir.exists() and any(train_dir.iterdir()) and prior is None:
            raise RuntimeError(f"Refusing to overwrite existing training output: {train_dir}")
    output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    metadata = {"job": job, "identity": identity, "source_hash": expected_source,
                "state": "started", "started": start, "command": sys.argv,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    atomic_json(marker, metadata)
    try:
        config = project_path(job["config"])
        seed_arg = None if job["seed"] == 2026 else job["seed"]
        if job["kind"] == "baseline":
            from .exploration_baselines import run
            run(job["baseline_kind"], config, job["region"], job["seed"], output)
        elif job["kind"] in {"dual", "frozen_event"}:
            if job["kind"] == "dual":
                from .train import run_training as run
            else:
                from .train_event_field import run
            checkpoint_dir = Path(run(config, job["region"], seed_arg)).resolve()
            metadata["checkpoint_directory"] = str(checkpoint_dir)
            metadata["checkpoint_hash"] = file_hash(checkpoint_dir / "best_model.pt")
            export_dual(job, checkpoint_dir, output)
        else:
            raise ValueError(f"Unknown job kind: {job['kind']}")
        validate_predictions(output, job)
        if source_hash() != expected_source or job_identity(job, expected_source) != identity:
            raise RuntimeError("Source/config changed during experiment; result is not accepted")
        metadata.update(state="completed", completed=time.time(), seconds=time.time()-start,
                        artifact_hashes={name: file_hash(output / name)
                            for name in ("validation.npz", "test.npz", "metrics.json")})
        if (output / "events.json").exists():
            metadata["artifact_hashes"]["events.json"] = file_hash(output / "events.json")
        atomic_json(marker, metadata)
        print(json.dumps({"job": job["id"], "state": "completed", "seconds": metadata["seconds"]}), flush=True)
    except BaseException as exc:
        metadata.update(state="failed", failed=time.time(), seconds=time.time()-start,
                        error=f"{type(exc).__name__}: {exc}")
        atomic_json(marker, metadata)
        raise


def validate_manifest(manifest):
    jobs = manifest.get("jobs", [])
    if not jobs:
        raise ValueError("Manifest must have jobs")
    by_id = {}
    outputs = set()
    for job in jobs:
        for key in ("id", "kind", "config", "region", "seed", "output"):
            if key not in job:
                raise ValueError(f"Job missing {key}")
        if job["id"] in by_id:
            raise ValueError(f"Duplicate job ID: {job['id']}")
        if job["kind"] not in {"dual", "baseline", "frozen_event"}:
            raise ValueError(f"Unknown job kind: {job['kind']}")
        if job["kind"] == "baseline" and not job.get("baseline_kind"):
            raise ValueError("Baseline job requires baseline_kind")
        if job["region"] not in {"NSW1", "QLD1", "TAS1"} or job["seed"] not in {2026, 2027, 2028}:
            raise ValueError("Jobs require the declared three regions and seeds")
        output = project_path(job["output"])
        if output in outputs:
            raise ValueError(f"Duplicate job output: {output}")
        if not project_path(job["config"]).is_file():
            raise ValueError(f"Missing configuration: {job['config']}")
        outputs.add(output)
        by_id[job["id"]] = job
    resolved = set()
    pending = set(by_id)
    while pending:
        ready = {key for key in pending if set(by_id[key].get("dependencies", [])).issubset(resolved)}
        if not ready:
            raise ValueError("Dependencies contain a cycle or unknown job ID")
        resolved.update(ready)
        pending.difference_update(ready)
    return by_id


def verify_manifest_hashes(manifest, include_data=True):
    for name, expected in manifest.get("config_hashes", {}).items():
        if file_hash(project_path(name)) != expected:
            raise ValueError(f"Manifest configuration hash mismatch: {name}")
    if "config_hashes" in manifest:
        missing = {job["config"] for job in manifest["jobs"]} - set(manifest["config_hashes"])
        if missing:
            raise ValueError(f"Manifest lacks configuration hashes: {sorted(missing)}")
    if include_data:
        for name, expected in manifest.get("data_hashes", {}).items():
            if file_hash(ROOT / name) != expected:
                raise ValueError(f"Manifest data hash mismatch: {name}")


def run_scheduler(manifest_path, manifest, gpus, workers_per_gpu):
    import fcntl
    import signal
    import subprocess
    if not gpus or workers_per_gpu < 1:
        raise ValueError("At least one GPU and one worker per GPU are required")
    by_id = validate_manifest(manifest)
    verify_manifest_hashes(manifest)
    status_path = manifest_path.with_suffix(".status.json")
    lock_path = manifest_path.with_suffix(".lock")
    log_root = manifest_path.parent / (manifest_path.stem + "_logs")
    log_root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another scheduler owns this manifest") from exc
        expected_source = source_hash()
        expected_manifest = file_hash(manifest_path)
        identities = {key: job_identity(job, expected_source) for key, job in by_id.items()}
        prior = json.loads(status_path.read_text()) if status_path.exists() else None
        if prior and (prior.get("source_hash") != expected_source or prior.get("manifest_hash") != expected_manifest):
            raise RuntimeError("Existing manifest run uses different source/manifest; use a new output root")
        status = {"manifest": str(manifest_path), "manifest_hash": expected_manifest,
                  "source_hash": expected_source, "state": "running", "started": time.time(),
                  "jobs": {key: {"state": "completed" if completed(job, expected_source) else "pending"}
                           for key, job in by_id.items()}}
        if prior:
            for key, item in status["jobs"].items():
                if item["state"] == "completed":
                    item.update(prior["jobs"].get(key, {}))
                    item["state"] = "completed"
        atomic_json(status_path, status)
        slots = [(str(gpu), worker) for gpu in gpus for worker in range(workers_per_gpu)]
        running = {}
        static_announced = False
        def unchanged():
            return (source_hash() == expected_source and file_hash(manifest_path) == expected_manifest
                    and all(job_identity(job, expected_source) == identities[key] for key, job in by_id.items()))
        def persist():
            status["updated"] = time.time()
            atomic_json(status_path, status)
        try:
            while True:
                if not unchanged():
                    raise RuntimeError("Source, manifest, or configuration changed during scheduled experiments")
                for slot, (process, stream, key, start) in list(running.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    stream.close()
                    good = code == 0 and completed(by_id[key], expected_source)
                    status["jobs"][key].update(state="completed" if good else "failed", returncode=code,
                                               completed=time.time(), seconds=time.monotonic()-start)
                    if not good:
                        status["jobs"][key]["error"] = "Worker failed or complete artifacts are missing"
                    print(json.dumps({"job": key, **status["jobs"][key]}), flush=True)
                    del running[slot]
                    persist()
                for key, job in by_id.items():
                    if status["jobs"][key]["state"] != "pending":
                        continue
                    deps = [status["jobs"][dep]["state"] for dep in job.get("dependencies", [])]
                    if any(state in {"failed", "blocked"} for state in deps):
                        status["jobs"][key].update(state="blocked", error="Dependency failed")
                ready = [key for key, job in by_id.items()
                         if status["jobs"][key]["state"] == "pending"
                         and all(status["jobs"][dep]["state"] == "completed" for dep in job.get("dependencies", []))]
                for slot in slots:
                    if slot in running or not ready:
                        continue
                    key = ready.pop(0)
                    safe_id = hashlib.sha256(key.encode()).hexdigest()[:12]
                    logfile = log_root / f"{safe_id}-{time.time_ns()}.log"
                    command = [sys.executable, "-u", "-m", "forecasting.exploration_runner",
                               "--manifest", str(manifest_path), "--worker", key,
                               "--source-hash", expected_source]
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=slot[0], OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
                    stream = logfile.open("w")
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    running[slot] = (process, stream, key, time.monotonic())
                    status["jobs"][key].update(state="running", started=time.time(), command=command,
                                               log=str(logfile), gpu=slot[0], pid=process.pid)
                    print(json.dumps({"job": key, "state": "started", "gpu": slot[0], "log": str(logfile)}), flush=True)
                    persist()
                static = [key for key, job in by_id.items() if job.get("protocol", "static") == "static"]
                if not static_announced and all(status["jobs"][key]["state"] == "completed" for key in static):
                    static_announced = True
                    print(json.dumps({"event": "all_static_jobs_completed", "count": len(static)}), flush=True)
                if not running and not any(v["state"] == "pending" for v in status["jobs"].values()):
                    break
                time.sleep(2)
            good = all(v["state"] == "completed" for v in status["jobs"].values())
            status.update(state="reporting" if good else "failed")
            persist()
            if not good:
                raise RuntimeError("Experiment matrix has failed or blocked jobs; inspect status and logs")
            report_log = log_root / f"report-{time.time_ns()}.log"
            report_command = [sys.executable, "-u", "-m", "forecasting.exploration_report",
                              "--manifest", str(manifest_path)]
            with report_log.open("w") as stream:
                report = subprocess.run(report_command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            status["report"] = {"command": report_command, "log": str(report_log), "returncode": report.returncode}
            if report.returncode:
                status["state"] = "report_failed"
                raise RuntimeError("All experiments completed, but report generation failed")
            status.update(state="completed", completed=time.time())
            persist()
        except BaseException as exc:
            for process, stream, key, start in running.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                stream.close()
                status["jobs"][key].update(state="interrupted", seconds=time.monotonic()-start)
            status.update(state="report_failed" if status["state"] == "report_failed" else "failed",
                          error=f"{type(exc).__name__}: {exc}")
            persist()
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--worker", help="Run a single job ID")
    parser.add_argument("--source-hash")
    parser.add_argument("--gpus", nargs="+", default=["6", "7"])
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if args.worker:
        verify_manifest_hashes(manifest, include_data=False)
        matching = [job for job in manifest["jobs"] if job["id"] == args.worker]
        if len(matching) != 1:
            raise ValueError("Worker job ID must occur exactly once")
        run_worker(matching[0], args.source_hash or source_hash())
    else:
        run_scheduler(manifest_path, manifest, args.gpus, args.workers_per_gpu)


if __name__ == "__main__":
    main()
