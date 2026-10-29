"""M0 candidate-2 deterministic-response check: frozen Chronos-2 CPU tool.

Runs base x2 and candidate x2 in fresh child processes (predict_frozen is
one-call-per-process), with the candidate input carrying the typed
predispatchis_totaldemand_mw future tensor built from the committed synthetic
PREDISPATCHIS fixture.  PASS requires different input hashes, bytewise-level
determinism within atol=rtol=1e-6, a response difference above the frozen
tolerance, unchanged model/checkpoint identity, and zero training calls.
Also records the two chronos-2 preprocessing probes that fix the tool's
covariate rules (future-only name rejected; NaN-past honest absence accepted).
No labels, no scores, no network.
Companion protocol: docs/research/agent_forecasting/m0_predispatch_td_abs_protocol_20261029.txt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "fixtures" / "agent_forecasting"))

import numpy as np  # noqa: E402

from forecasting.agent_forecasting.contracts import FrozenForecastInput  # noqa: E402
from forecasting.agent_forecasting.frozen_chronos import (  # noqa: E402
    load_development_input, sha256_file,
)
from forecasting.agent_forecasting.predispatch_td_abs import (  # noqa: E402
    attach_future_forecast, build_future_series,
)
from predispatchis_td_v2_fixture import (  # noqa: E402
    FIXTURE_FILENAME, FIXTURE_ORIGIN_UNIX, FIXTURE_RECEIPT_UNIX,
)
from forecasting.agent_forecasting.predispatch_td_abs import (  # noqa: E402
    synthetic_receipt_from_raw_body,
)


def run_child(workdir: Path, request_path: Path, per_call_seconds: int):
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", str(workdir),
         str(request_path)],
        cwd=str(ROOT), capture_output=True, text=True,
        timeout=per_call_seconds + 60)
    wall = time.monotonic() - started
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout + completed.stderr)
        raise RuntimeError(f"child model call failed with exit {completed.returncode}")
    output = np.load(workdir / "output.npy", allow_pickle=False)
    trace = json.loads((workdir / "trace.json").read_text())
    return output, trace, wall


def child_call(workdir: Path, request_path: Path) -> int:
    from forecasting.agent_forecasting.frozen_chronos import predict_frozen

    request = FrozenForecastInput.from_dict(json.loads(request_path.read_text()))
    config = json.loads((workdir / "config.json").read_text())
    output, trace = predict_frozen(request, config)
    np.save(workdir / "output.npy", output, allow_pickle=False)
    (workdir / "trace.json").write_text(json.dumps(trace, sort_keys=True) + "\n")
    (workdir / "child_input_hash.txt").write_text(request.input_hash() + "\n")
    return 0


def tool_probes() -> dict:
    """Fix the chronos-2 covariate rules empirically inside the frozen venv."""
    from chronos.chronos2.preprocess import from_list_of_dicts

    target = np.linspace(-40.0, 90.0, 72)
    probes = {}
    try:
        from_list_of_dicts(
            [{"target": target, "future_covariates": {"probe_only": np.ones(24)}}],
            prediction_length=24)
        probes["future_only_name"] = "ACCEPTED"
    except ValueError as exc:
        probes["future_only_name"] = f"REJECTED: {str(exc)[:100]}"
    prepared = from_list_of_dicts(
        [{"target": target,
          "past_covariates": {"probe_nan": np.full(72, np.nan)},
          "future_covariates": {"probe_nan": np.linspace(6000, 9000, 24)}}],
        prediction_length=24)
    entry = prepared[0]
    probes["nan_past_real_future"] = {
        "accepted": True,
        "past_all_nan": bool(entry["context"][-1].isnan().all().item()),
        "future_all_finite": bool(entry["future_covariates"][-1].isfinite().all().item()),
        "n_future_covariates": int(entry["n_future_covariates"]),
    }
    return probes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",
                        default="configs/agent_forecasting/m0_predispatch_td_abs_c2_20261029.json")
    parser.add_argument("--child", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        workdir, request_path = Path(args.child[0]), Path(args.child[1])
        return child_call(workdir, request_path)

    started_monotonic = time.monotonic()
    # CPU-only offline guards must hold in every child process (predict_frozen
    # refuses to run otherwise); children inherit this environment.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    config = json.loads((ROOT / args.config).read_text())
    spec = config["response_check"]
    tolerance = config["numerical_tolerance"]
    evidence_dir = ROOT / spec["evidence_dir"]
    evidence_dir.mkdir(parents=True, exist_ok=True)

    fixture_zip = ROOT / spec["fixture_zip"]
    fixture_sha = sha256_file(fixture_zip)
    if fixture_sha != spec["fixture_zip_sha256"]:
        raise SystemExit("fixture zip hash does not match the frozen config")
    receipt = synthetic_receipt_from_raw_body(
        fixture_zip.read_bytes(), FIXTURE_RECEIPT_UNIX, FIXTURE_FILENAME)
    series = build_future_series(FIXTURE_ORIGIN_UNIX, [receipt])
    base_request, base_provenance = load_development_input(config)
    if series.origin_unix != base_request.origin_unix:
        raise SystemExit("fixture origin does not match the frozen development input")
    candidate_request = attach_future_forecast(base_request, series)
    base_hash = base_request.input_hash()
    candidate_hash = candidate_request.input_hash()
    if base_hash == candidate_hash:
        raise SystemExit("candidate input hash equals the base input hash")

    runs = []
    for label, request in (("base", base_request), ("candidate", candidate_request)):
        for repeat in range(spec["runs_per_input"]):
            with tempfile.TemporaryDirectory() as tmp:
                workdir = Path(tmp) / "w"
                workdir.mkdir()
                (workdir / "config.json").write_text(json.dumps(config))
                request_path = Path(tmp) / "request.json"
                request_path.write_text(json.dumps(request.to_dict()))
                output, trace, wall = run_child(
                    workdir, request_path,
                    config["resources"]["per_call_wall_seconds"])
            runs.append({"label": label, "repeat": repeat, "output": output,
                         "trace": trace, "wall_seconds": wall})
            if time.monotonic() - started_monotonic > config["resources"]["total_wall_seconds"]:
                raise SystemExit("total wall budget exceeded")

    base_runs = [r for r in runs if r["label"] == "base"]
    candidate_runs = [r for r in runs if r["label"] == "candidate"]
    atol = rtol = tolerance["atol"]

    def same(a, b):
        return np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False)

    checks = {
        "input_hashes_differ": base_hash != candidate_hash,
        "base_deterministic": same(base_runs[0]["output"], base_runs[1]["output"]),
        "candidate_deterministic": same(candidate_runs[0]["output"], candidate_runs[1]["output"]),
        "model_state_unchanged": all(
            r["trace"]["state_sha256_before"] == r["trace"]["state_sha256_after"]
            for r in runs),
        "checkpoint_identity_stable": len({
            json.dumps(r["trace"]["checkpoint"], sort_keys=True) for r in runs}) == 1,
        "training_calls_zero": all(r["trace"]["training_calls"] == 0 for r in runs),
        "typed_encoding_recorded": all(
            r["trace"].get("typed_future_forecasts") == {}
            for r in base_runs) and all(
            r["trace"]["typed_future_forecasts"]["predispatchis_totaldemand_mw"]["past_encoding"]
            == "all_nan_honest_absence" for r in candidate_runs),
    }
    delta = np.abs(candidate_runs[0]["output"] - base_runs[0]["output"])
    checks["response_diff_above_tolerance"] = bool(
        delta.max() > spec["min_output_delta_abs"])

    report = {
        "candidate_id": "predispatchis_absolute_totaldemand_future_v1",
        "base_input_hash": base_hash,
        "candidate_input_hash": candidate_hash,
        "fixture_zip_sha256": fixture_sha,
        "series_hash": series.series_hash,
        "series_publication_basis": series.publication_basis,
        "series_strict_asof": series.strict_asof,
        "max_abs_delta": float(delta.max()),
        "min_output_delta_abs": spec["min_output_delta_abs"],
        "tolerance": tolerance,
        "checks": checks,
        "walls": {f"{r['label']}_{r['repeat']}": r["wall_seconds"] for r in runs},
        "state_hashes": sorted({r["trace"]["state_sha256_before"] for r in runs}),
        "tool_probes": tool_probes(),
        "base_provenance": base_provenance,
        "model_calls": len(runs),
        "elapsed_seconds": time.monotonic() - started_monotonic,
    }
    (evidence_dir / "response_check.json").write_text(json.dumps(report, indent=2) + "\n")
    for run in runs:
        np.save(evidence_dir / f"output__{run['label']}_{run['repeat']}.npy",
                run["output"], allow_pickle=False)
    ok = all(checks.values())
    print(json.dumps({"checks": checks, "max_abs_delta": float(delta.max())}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
