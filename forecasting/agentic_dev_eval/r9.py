"""R9 frozen three-seed run: label-free decisions, shared execution primitives.

Stages are explicit so preparation, CPU forecasts, decisions and scoring run in
separate processes. No result-dependent reruns or parameter selection.
"""
from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import tempfile
import time

import numpy as np

from . import arms, data, llm_transport, metrics, protocol
from forecasting.agent_forecasting.execution import call_with_deadline, publish_directory

CONFIG = Path("configs/agent_forecasting/r9_development_20261030.json")


def _read(path):
    return json.loads(Path(path).read_text())


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _sha(path):
    return data.sha256_file(path)


def _npz(path, **values):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **values)


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _publish(final: Path, write, seconds=30):
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(f"refusing completed output overwrite: {final}")
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    deadline = time.monotonic() + seconds
    try:
        def prepare_commit():
            write(staging)
            inventory = {str(path.relative_to(staging)): {
                "sha256": _sha(path), "bytes": path.stat().st_size}
                for path in sorted(staging.rglob("*")) if path.is_file()}
            _json(staging / "COMMIT.json", {"files": inventory})
        call_with_deadline(prepare_commit, deadline - time.monotonic())
        committed = publish_directory(staging, final, deadline=deadline)
        print(json.dumps({"committed": str(committed.path), "durable": committed.durable}), flush=True)
        return committed
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_directory(path):
    path = Path(path)
    expected = _read(path / "COMMIT.json")["files"]
    actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
    if actual != set(expected) | {"COMMIT.json"}:
        raise ValueError(f"incomplete committed directory: {path}")
    for name, record in expected.items():
        if _sha(path / name) != record["sha256"] or (path / name).stat().st_size != record["bytes"]:
            raise ValueError(f"artifact changed: {path / name}")


def _configuration():
    cfg = _read(CONFIG)
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    if branch != cfg["branch"]:
        raise ValueError("wrong experiment branch")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for name in cfg["source_files"]:
        if Path(name).read_bytes() != subprocess.check_output(["git", "show", f"{head}:{name}"]):
            raise ValueError(f"uncommitted execution source: {name}")
    return cfg, head


def _execution(cfg, root):
    cfg, head = _configuration()
    frozen = _read(root / "inputs" / "manifest.json")
    if frozen["execution_commit"] != head:
        raise ValueError("execution commit differs from prepared inputs")
    for name, digest in frozen["sources"].items():
        if _sha(name) != digest:
            raise ValueError(f"execution fingerprint changed: {name}")
    verify_directory(root / "inputs")
    return frozen


def prepare(cfg, head):
    root = Path(cfg["output_root"])
    if root.exists():
        raise FileExistsError(f"new experiment root required: {root}")
    root.mkdir(parents=True)
    region_records, label_payloads, grid = {}, {}, {}
    for region in cfg["regions"]:
        csv = Path(cfg["data_root"]) / f"{region.lower()}_hourly.csv"
        if _sha(csv) != cfg["data_sha256"][region]:
            raise ValueError(f"source changed since r8: {region}")
        records, info = data.build_origin_grid(region, cfg["data_root"])
        if len(records) != cfg["origins_per_region"]:
            raise ValueError(f"unexpected origin count: {region}: {len(records)}")
        frame, _ = data.load_region_frame(region, cfg["data_root"])
        safe = []
        for record in records:
            visible = frame.loc[frame["available_at_aest"] <= record.origin_local].tail(14 * 24)
            safe.append({
                "region": region, "origin_unix": record.origin_unix,
                "origin_local": record.origin_local, "origin_dow": record.origin_dow,
                "origin_month": record.origin_month, "half": record.half,
                "target_time_unix": record.target_time_unix.tolist(),
                "history_prices": record.history_prices.tolist(),
                "history_demand": record.history_demand.tolist(),
                "history_calendar": record.history_calendar.tolist(),
                "future_calendar": record.future_calendar.tolist(),
                "tool_history": {
                    "available_at_aest": [t.isoformat() for t in visible["available_at_aest"]],
                    "_local": [t.isoformat() for t in visible["_local"]],
                    "rrp_aud_per_mwh": visible["rrp_aud_per_mwh"].tolist(),
                    "total_demand_mw": visible["total_demand_mw"].tolist(),
                },
            })
        region_records[region] = safe
        label_payloads[region] = {
            "origin_unix": np.asarray([r.origin_unix for r in records]),
            "target_time_unix": np.stack([r.target_time_unix for r in records]),
            "actual": np.stack([r.actual for r in records]),
        }
        grid[region] = info

    def inputs(stage):
        for region, records in region_records.items():
            _json(stage / f"{region}.json", records)
        _json(stage / "manifest.json", {
            "execution_commit": head, "branch": cfg["branch"],
            "seeds": cfg["seeds"], "grid": grid,
            "sources": {name: _sha(name) for name in cfg["source_files"]},
            "data_sha256": cfg["data_sha256"],
            "availability_scope": "archived available_at proxy; not verified historical public receipt",
            "final_test": "not scored or passed to prediction; shared source CSV includes final-test rows",
        })

    _publish(root / "inputs", inputs)
    _publish(root / "labels", lambda stage: [
        _npz(stage / f"{region}.npz", **payload) for region, payload in label_payloads.items()])


def _identity(records):
    return {"origin_unix": np.asarray([r["origin_unix"] for r in records]),
            "target_time_unix": np.asarray([r["target_time_unix"] for r in records]),
            "half": np.asarray([r["half"] for r in records])}


def _usage(start):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"wall_seconds": time.monotonic() - start, "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime, "peak_rss_bytes": usage.ru_maxrss * 1024,
            "cpu_measurement": "process total including imports; wall starts at stage entry"}


def base(cfg, region, seed):
    start = time.monotonic()
    root = Path(cfg["output_root"])
    frozen = _execution(cfg, root)
    records = _read(root / "inputs" / f"{region}.json")
    final = root / str(seed) / region / "A"
    if final.exists():
        raise FileExistsError(final)
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:cfg["cpu_threads"]])
    random.seed(seed)
    np.random.seed(seed)
    import torch
    from chronos import Chronos2Pipeline
    torch.manual_seed(seed)
    torch.set_num_threads(cfg["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    checkpoint = Path(cfg["checkpoint"]["directory"])
    for name, key in (("config.json", "config_sha256"), ("model.safetensors", "weights_sha256")):
        if _sha(checkpoint / name) != cfg["checkpoint"][key]:
            raise ValueError(f"checkpoint changed: {name}")

    def predict():
        pipeline = Chronos2Pipeline.from_pretrained(str(checkpoint), device_map="cpu",
                                                    torch_dtype=torch.float32, local_files_only=True)
        pipeline.model.eval()
        for parameter in pipeline.model.parameters():
            parameter.requires_grad_(False)
        batches = []
        for record in records:
            past = np.asarray(record["history_calendar"])
            future = np.asarray(record["future_calendar"])
            batches.append({"target": np.asarray(record["history_prices"]),
                "past_covariates": {"total_demand_mw": np.asarray(record["history_demand"]),
                                    **{name: past[:, i] for i, name in enumerate(data.CALENDAR_FEATURES)}},
                "future_covariates": {name: future[:, i] for i, name in enumerate(data.CALENDAR_FEATURES)}})
        with torch.inference_mode():
            q, _ = pipeline.predict_quantiles(
                batches, prediction_length=24, quantile_levels=list(protocol.QUANTILE_LEVELS),
                batch_size=len(batches), context_length=72, cross_learning=False)
        return np.asarray(q, dtype=np.float64)[:, 0]

    # The parent launcher also bounds the process; this protects Python-side calls.
    q = call_with_deadline(predict, cfg["base_deadline_seconds"] - (time.monotonic() - start))
    if q.shape != (len(records), 24, 5) or not np.isfinite(q).all():
        raise ValueError("invalid base forecast")
    for name, key in (("config.json", "config_sha256"), ("model.safetensors", "weights_sha256")):
        if _sha(checkpoint / name) != cfg["checkpoint"][key]:
            raise ValueError("checkpoint changed during inference")
    report = {"region": region, "seed": seed, "arm": "A", "execution_commit": frozen["execution_commit"],
              "input_sha256": _sha(root / "inputs" / f"{region}.json"),
              "checkpoint": cfg["checkpoint"], "predict_calls": 1, "training_calls": 0,
              "device": "cpu", "affinity": sorted(os.sched_getaffinity(0)),
              "native_crossings": int(np.sum(np.diff(q, axis=-1) < 0)), **_usage(start)}

    def write(stage):
        _npz(stage / "predictions.npz", quantiles=q, **_identity(records))
        _json(stage / "run.json", report)
    _publish(final, write, cfg["publication_deadline_seconds"])


def decision_result(record, base_quantiles, arm, frame, seed, transport):
    if arm == "B":
        result = arms.decide_b(record, base_quantiles)
    elif arm == "C":
        result = arms.decide_c(record, base_quantiles, transport=transport, seed=seed)
    else:
        result = arms.decide_d(record, base_quantiles, frame, transport=transport, seed=seed)
    # Failure retains exact A values and remains a failed decision, not a success.
    q = (protocol.apply_action(base_quantiles, np.asarray(record["history_prices"]), result["action"])
         if result["exit_code"] == 0 else np.array(base_quantiles, copy=True))
    return result, q


def decisions(cfg, region, seed, arm):
    import pandas as pd
    start = time.monotonic()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:cfg["cpu_threads"]])
    root = Path(cfg["output_root"])
    frozen = _execution(cfg, root)
    records = _read(root / "inputs" / f"{region}.json")
    base_dir = root / str(seed) / region / "A"
    verify_directory(base_dir)
    baseline = _load_npz(base_dir / "predictions.npz")
    if not np.array_equal(baseline["origin_unix"], _identity(records)["origin_unix"]):
        raise ValueError("base identity mismatch")
    final = root / str(seed) / region / arm
    attempts = root / "attempts" / f"{seed}_{region}_{arm}.jsonl"
    attempts.parent.mkdir(exist_ok=True)
    # Exclusive journal creation records attempted execution and prevents silent replay.
    with attempts.open("x") as journal:
        rows, values, wire = [], [], []
        for i, original in enumerate(records):
            record = {key: value for key, value in original.items() if key != "tool_history"}
            frame = pd.DataFrame(original["tool_history"])
            frame["available_at_aest"] = pd.to_datetime(frame["available_at_aest"], utc=True)
            frame["_local"] = pd.to_datetime(frame["_local"], utc=True).dt.tz_convert(protocol.AEST)
            traces = []
            def capture(entry):
                traced = {"seed": seed, "region": region, "arm": arm, "origin_unix": record["origin_unix"], **entry}
                traces.append(traced)
                journal.write(json.dumps({"event": "llm", **traced}, ensure_ascii=False) + "\n")
                journal.flush()
            journal.write(json.dumps({"event": "origin_started", "origin_unix": record["origin_unix"], "seed": seed}) + "\n")
            journal.flush()
            decision_start = time.monotonic()
            try:
                result, q = call_with_deadline(
                    lambda: decision_result(record, baseline["quantiles"][i], arm, frame, seed,
                                            partial(llm_transport.chat, trace=capture)),
                    cfg["decision_deadline_seconds"])
            except Exception as exc:
                result = {"action": "", "exit_code": 1, "failure_reason": type(exc).__name__,
                          "detail": str(exc)[:200], "latency_ms": 1000 * (time.monotonic() - decision_start),
                          "llm_calls": len(traces), "tool_calls": 0, "usage_incomplete": True,
                          "tokens": {"prompt": 0, "completion": 0}}
                q = baseline["quantiles"][i].copy()
            row = {"origin_unix": record["origin_unix"], "region": region, "seed": seed, "arm": arm,
                   "fallback_to_A": result["exit_code"] != 0, **result}
            rows.append(row)
            values.append(q)
            wire.extend(traces)
            journal.write(json.dumps({"event": "origin_completed", **row}, ensure_ascii=False) + "\n")
            journal.flush()
            print(json.dumps({k: row[k] for k in ("region", "seed", "arm", "origin_unix", "action", "exit_code", "failure_reason")}), flush=True)
        os.fsync(journal.fileno())
    report = {"region": region, "seed": seed, "arm": arm, "execution_commit": frozen["execution_commit"],
              "input_sha256": _sha(root / "inputs" / f"{region}.json"),
              "base_sha256": _sha(base_dir / "predictions.npz"), "n_origins": len(records),
              "n_decision_failures": sum(r["exit_code"] != 0 for r in rows),
              "affinity": sorted(os.sched_getaffinity(0)),
              "attempt_journal_sha256": _sha(attempts), **_usage(start)}

    def write(stage):
        _npz(stage / "predictions.npz", quantiles=np.asarray(values),
             exit_codes=np.asarray([r["exit_code"] for r in rows]), **_identity(records))
        (stage / "decisions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        (stage / "wire.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in wire))
        _json(stage / "run.json", report)
    _publish(final, write, cfg["publication_deadline_seconds"])


def score(cfg):
    root = Path(cfg["output_root"])
    frozen = _execution(cfg, root)
    verify_directory(root / "labels")
    results = []
    for seed in cfg["seeds"]:
        for region in cfg["regions"]:
            labels = _load_npz(root / "labels" / f"{region}.npz")
            payloads = {}
            for arm in ("A", "B", "C", "D"):
                path = root / str(seed) / region / arm
                verify_directory(path)
                payload = _load_npz(path / "predictions.npz")
                for key in ("origin_unix", "target_time_unix"):
                    if not np.array_equal(payload[key], labels[key]):
                        raise ValueError(f"score identity mismatch: {seed}/{region}/{arm}/{key}")
                if not np.isfinite(payload["quantiles"]).all():
                    raise ValueError("missing forecast: cannot report full-grid scores")
                payloads[arm] = payload
            arm_scores = {}
            for arm, payload in payloads.items():
                failures = int(np.count_nonzero(payload.get("exit_codes", np.zeros(len(labels["origin_unix"])))))
                if failures:
                    failed = payload["exit_codes"] != 0
                    if not np.array_equal(payload["quantiles"][failed], payloads["A"]["quantiles"][failed]):
                        raise ValueError("failed decision did not preserve A")
                arm_scores[arm] = {**metrics.score_arrays(payload["quantiles"], labels["actual"]),
                                   "n_origins": len(labels["origin_unix"]), "decision_failures": failures}
            results.append({"seed": seed, "region": region, "arms": arm_scores,
                "increments": {f"D-{other}": {key: arm_scores["D"][key] - arm_scores[other][key]
                    for key in ("mae", "quantile_score")} for other in ("A", "B", "C")}})
    keys = ("mae", "quantile_score", "coverage_80", "coverage_90", "width_80", "width_90")
    macro_by_seed = {str(seed): {arm: {key: float(np.mean([r["arms"][arm][key]
        for r in results if r["seed"] == seed])) for key in keys} for arm in ("A", "B", "C", "D")}
        for seed in cfg["seeds"]}
    across_seeds = {arm: {key: {"mean": float(np.mean([macro_by_seed[str(s)][arm][key] for s in cfg["seeds"]])),
                               "sample_std": float(np.std([macro_by_seed[str(s)][arm][key] for s in cfg["seeds"]], ddof=1))}
        for key in keys} for arm in ("A", "B", "C", "D")}
    summary = {"execution_commit": frozen["execution_commit"], "results": results,
               "macro_by_seed": macro_by_seed, "across_seeds": across_seeds,
               "metric_population": "all 53 planned origins per region; failed decisions use unchanged A",
               "seed_limit": "temperature=0; repeats are not independent datasets or trained models",
               "final_test_evaluated": False}
    _publish(root / "results", lambda stage: _json(stage / "summary.json", summary))
    print(json.dumps(across_seeds, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "base", "decide", "score"))
    parser.add_argument("--region", choices=protocol.REGIONS)
    parser.add_argument("--seed", type=int, choices=(2026, 2027, 2028))
    parser.add_argument("--arm", choices=("B", "C", "D"))
    args = parser.parse_args()
    cfg, head = _configuration()
    if args.stage == "prepare":
        prepare(cfg, head)
    elif args.stage == "score":
        score(cfg)
    elif args.stage == "base":
        base(cfg, args.region, args.seed)
    else:
        decisions(cfg, args.region, args.seed, args.arm)


if __name__ == "__main__":
    main()
