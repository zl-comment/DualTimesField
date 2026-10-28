#!/usr/bin/env python3
"""Run the single G0 CPU smoke only after the protocol and source are committed."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_worker_report(path):
    if not Path(path).exists():
        return {}, None
    try:
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError("worker report is not an object")
        return value, None
    except (OSError, ValueError) as error:
        return {}, type(error).__name__


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def reserve_smoke(path: Path, config: dict) -> dict:
    """Reserve before spawning; failed and interrupted attempts consume budget."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        records = [json.loads(line) for line in stream if line.strip()]
        if any(row["kind"] == "basic_smoke" for row in records):
            raise RuntimeError("basic smoke budget exhausted; two event calls remain reserved")
        if len(records) >= config["resources"]["max_model_calls"]:
            raise RuntimeError("G0 model call budget exhausted")
        record = {"attempt": len(records) + 1, "kind": "basic_smoke", "reserved_at_utc": utc_now()}
        stream.seek(0, 2)
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        return record


def worker(config_path: Path, output: Path):
    config = read_json(config_path)
    # Linux parent-death signal prevents an orphaned worker if its monitor dies.
    import ctypes
    expected_parent = int(os.environ.get("G0_MONITOR_PID", "-1"))
    if expected_parent <= 0 or os.getppid() != expected_parent:
        raise RuntimeError("worker requires its live budget monitor")
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise RuntimeError("cannot install parent-death protection")
    if os.getppid() != expected_parent:
        raise RuntimeError("budget monitor died during worker startup")
    signal.alarm(config["resources"]["per_call_wall_seconds"])
    if output != (ROOT / config["smoke_output"]).resolve():
        raise RuntimeError("unregistered worker output")
    manifest = read_json(output / "run_manifest.json")
    reservations = [json.loads(line) for line in (ROOT / config["attempt_ledger"]).read_text().splitlines() if line]
    if manifest["attempt"] not in reservations or manifest["attempt"]["kind"] != "basic_smoke":
        raise RuntimeError("worker has no persisted call reservation")
    # Exclusive marker prevents using the private entry point for an uncounted replay.
    write_json_new(output / "worker_started.json", {"started_at_utc": utc_now(), "pid": os.getpid()})
    started = time.monotonic()
    guard = {"blocked_network_calls": 0, "blocked_label_reads": 0}

    def audit(event, args):
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.sendto"}:
            guard["blocked_network_calls"] += 1
            raise RuntimeError("network forbidden in G0 prediction process")
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            name = Path(os.fsdecode(args[0])).name
            if name in {"actual.npy", "target.npy", "test.npz", "train.npz", "selection.npz"}:
                guard["blocked_label_reads"] += 1
                raise RuntimeError("label-containing file forbidden in G0 prediction process")

    sys.addaudithook(audit)
    report = {"started_at_utc": utc_now(), "status": "failed", "guard": guard}
    try:
        from forecasting.agent_forecasting.frozen_chronos import (
            load_development_input, predict_frozen, sha256_file,
        )
        from forecasting.agent_forecasting.trace import AppendOnlyTrace
        request, provenance = load_development_input(config)
        trace = AppendOnlyTrace(output / "tool_trace.jsonl")
        trace.append("basic_smoke", "request", request.input_hash())
        write_json_new(output / "input.json", request.to_dict())
        write_json_new(output / "input_provenance.json", provenance)
        quantiles, identity = predict_frozen(request, config)
        # Preserve the numerical tool result, without labels or a scoring pass.
        forecast = {"origin_unix": request.origin_unix,
                    "target_time_unix": list(request.target_time_unix),
                    "quantile_levels": list(identity["quantile_levels"]),
                    "quantiles": quantiles.tolist(), "input_sha256": request.input_hash()}
        write_json_new(output / "forecast.json", forecast)
        if any(guard.values()):
            raise RuntimeError("forbidden access was attempted")
        report.update(status="passed", model_identity=identity,
                      input_sha256=request.input_hash(), forecast_sha256=sha256_file(output / "forecast.json"),
                      shape=list(quantiles.shape), future_labels_read=False, scores_computed=False)
        trace.append("basic_smoke", "publish", request.input_hash(),
                     output_hash=report["forecast_sha256"])
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(wall_seconds=time.monotonic() - started, user_cpu_seconds=usage.ru_utime,
                      system_cpu_seconds=usage.ru_stime, peak_rss_bytes=usage.ru_maxrss * 1024,
                      finished_at_utc=utc_now())
        write_json_new(output / "worker_report.json", report)
    return 0 if report["status"] == "passed" else 1


def kill_group_and_wait(process):
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/agent_forecasting/g0_20261028.json")
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.config.resolve(), args.worker.resolve())
    if not args.run_smoke:
        parser.error("use --run-smoke only after committing protocol, configuration and tests")
    config = read_json(args.config)
    if config["status"] != "frozen_g0" or config["execution_scope"] != "g0_only":
        raise RuntimeError("configuration is not frozen for G0")
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()
    if git("branch", "--show-current") != config["branch"]:
        raise RuntimeError("wrong experiment branch")
    source_files = config["source_files"]
    git("ls-files", "--error-unmatch", *source_files)
    if git("diff", "HEAD", "--", *source_files):
        raise RuntimeError("experiment source/protocol/config/tests must be committed")
    from forecasting.agent_forecasting.frozen_chronos import sha256_file
    manifest = {"code_commit": git("rev-parse", "HEAD"), "base_commit": config["base_commit"],
                "branch": config["branch"], "source_sha256": {p: sha256_file(ROOT / p) for p in source_files},
                "started_at_utc": utc_now(), "python": config["interpreter"]}
    output = ROOT / config["smoke_output"]
    output.mkdir(parents=True, exist_ok=False)
    attempt = reserve_smoke(ROOT / config["attempt_ledger"], config)
    write_json_new(output / "run_manifest.json", dict(manifest, attempt=attempt))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", WANDB_DISABLED="true", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="2",
               G0_MONITOR_PID=str(os.getpid()))
    allowed_cpus = sorted(os.sched_getaffinity(0))[:config["resources"]["cpu_threads"]]
    if not allowed_cpus:
        raise RuntimeError("no CPU affinity available")
    command = [config["interpreter"], str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
               "--worker", str(output)]
    import psutil
    start = time.monotonic()
    peak_rss = 0
    limit_reason = None
    process = None
    returncode = None
    monitor_error = None
    previous_term = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("budget monitor terminated")
    signal.signal(signal.SIGTERM, interrupted)
    with (output / "stdout.txt").open("x") as stdout, (output / "stderr.txt").open("x") as stderr:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                       start_new_session=True, preexec_fn=lambda: os.sched_setaffinity(0, allowed_cpus))
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    rss = sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)] if p.is_running())
                    peak_rss = max(peak_rss, rss)
                except psutil.NoSuchProcess:
                    pass
                if time.monotonic() - start >= config["resources"]["per_call_wall_seconds"]:
                    limit_reason = "timeout"
                if peak_rss > config["resources"]["rss_bytes"]:
                    limit_reason = "memory_limit"
                if limit_reason:
                    break
                time.sleep(0.1)
        except BaseException as error:
            monitor_error = type(error).__name__
            limit_reason = "monitor_interrupted_or_failed"
        finally:
            returncode = kill_group_and_wait(process)
            signal.signal(signal.SIGTERM, previous_term)
    worker_path = output / "worker_report.json"
    child, report_parse_error = read_worker_report(worker_path)
    status = "passed" if returncode == 0 and child.get("status") == "passed" and not limit_reason else "failed"
    report = {"basic_tool_status": status, "event_path_status": "not_run_requires_g0a_same_origin_evidence",
              "overall_g0_status": "not_passed", "model_calls_reserved": 1, "event_calls_reserved_remaining": 2,
              "returncode": returncode, "limit_reason": limit_reason,
              "monitor_error": monitor_error,
              "worker_report_parse_error": report_parse_error,
              "wall_seconds_including_spawn": time.monotonic() - start,
              "monitor_peak_rss_bytes": peak_rss, "cpu_affinity": allowed_cpus,
              "local_gpu_usage": 0, "new_training": 0, "model_downloads": 0,
              "external_llm_calls": 0, "scores_computed": False,
              "worker_report_present": bool(child), "finished_at_utc": utc_now()}
    write_json_new(output / "resource_report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
