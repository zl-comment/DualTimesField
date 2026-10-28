"""Minimal G0-A audit runner (protocol §7 replay commands).

record-tests: turn the true pytest exit code captured in preflight_tests.exit
              into preflight_tests.json (no pipe exit codes anywhere).
audit:        run the six frozen checks on the frozen NPZ for the selected
              origins, under the budget guard, network-disabled, and the
              path allowlist; write g0a_verdict.json and evidence files.
archive:      mirror the outputs run dir into the committed logs dir and
              write artifact_manifest.json (sha256 of every file).

This script never reads labels, never touches the network, never modifies
the NPZ, and cannot emit a strict as-of PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from forecasting.agent_forecasting import g0a_predispatch as g0a  # noqa: E402

AEST = timezone(timedelta(hours=10))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def aest(unix: int) -> str:
    return datetime.fromtimestamp(int(unix), AEST).strftime("%Y-%m-%d %H:%M AEST")


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def load_config(path: str) -> dict:
    config = json.loads(Path(path).read_text())
    run_dir = REPO_ROOT / config["output_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    return config


def enforce_frozen_binding(config: dict) -> Path:
    """Live NPZ and builder must still match the hashes frozen in the config."""
    npz = REPO_ROOT / config["npz_path"]
    if sha256_file(npz) != config["npz_sha256_at_freeze"]:
        raise SystemExit("FROZEN_BINDING_VIOLATION: live NPZ hash differs from config")
    builder = REPO_ROOT / config["builder_path"]
    if sha256_file(builder) != config["builder_sha256_at_freeze"]:
        raise SystemExit("FROZEN_BINDING_VIOLATION: live builder hash differs from config")
    return npz


def source_hashes(config: dict) -> dict:
    return {
        "harness_module": sha256_file(REPO_ROOT / config["harness_module"]),
        "audit_script": sha256_file(REPO_ROOT / config["audit_script"]),
        "config": sha256_file(REPO_ROOT / "configs/agent_forecasting/g0a_predispatch_rrp_20261028.json"),
        "protocol": sha256_file(REPO_ROOT / config["protocol"]),
        "builder": sha256_file(REPO_ROOT / config["builder_path"]),
        "npz": sha256_file(REPO_ROOT / config["npz_path"]),
        "tests": sha256_file(REPO_ROOT / config["tests"]),
    }


def cmd_record_tests(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    run_dir = REPO_ROOT / config["output_dir"]
    exit_file, log_file = run_dir / "preflight_tests.exit", run_dir / "preflight_tests.txt"
    if not exit_file.exists() or not log_file.exists():
        raise SystemExit("preflight_tests.exit/.txt missing: run the pytest replay command first")
    record = {
        "recorded_at_utc": utc_now(),
        "returncode": int(exit_file.read_text().strip()),
        "log_path": "preflight_tests.txt",
        "log_sha256": sha256_file(log_file),
        "command": (
            ".venv/bin/python -m pytest tests/test_g0a_predispatch_rrp.py -q "
            "> <run_dir>/preflight_tests.txt 2>&1; echo $? > <run_dir>/preflight_tests.exit"
        ),
        "note": "returncode is pytest's own exit status read from the .exit file; no pipe status",
    }
    (run_dir / "preflight_tests.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


def cmd_audit(args: argparse.Namespace) -> None:
    started_utc = utc_now()
    started = time.monotonic()
    config = load_config(args.config)
    run_dir = REPO_ROOT / config["output_dir"]
    npz = enforce_frozen_binding(config)
    budget = g0a.BudgetGuard(float(config["budget"]["wall_seconds"]))
    allow_roots = [REPO_ROOT / root for root in config["allow_roots"]]
    guard_violations = []
    for source_path, label in (
        (config["harness_module"], "harness_module"),
        (config["audit_script"], "audit_script"),
    ):
        hits = g0a.scan_source_for_forbidden((REPO_ROOT / source_path).read_text())
        guard_violations.extend(f"static scan {label}: {hit}" for hit in hits)

    with g0a.network_disabled():
        opened = g0a.assert_allowed_path(npz, allow_roots)
        arrays = dict(np.load(opened))
        budget.check()
        origins = np.asarray(arrays["forecast_origin_unix"])
        indices = g0a.select_origin_indices(len(origins))
        selected = {key: np.asarray(arrays[key])[indices] for key in g0a.REQUIRED_KEYS}
        budget.check()
        checks = [
            g0a.check_schema(arrays),
            g0a.check_units(selected["predispatch_rrp"], tuple(config["unit_envelope_aud_per_mwh"][k] for k in ("min", "max"))),
            g0a.check_target_time(selected["forecast_origin_unix"], selected["predispatch_rrp"].shape[1]),
            g0a.check_vintage(
                selected["forecast_origin_unix"],
                selected["source_run_unix"],
                selected["source_run_seqno"],
                stale_seconds=int(config["stale_seconds"]),
            ),
            g0a.check_coverage(
                selected["forecast_origin_unix"],
                selected["predispatch_covered"],
                selected["source_run_unix"],
                stale_seconds=int(config["stale_seconds"]),
            ),
            g0a.check_publication_basis(
                selected["source_run_unix"], selected["source_last_changed_unix"]
            ),
        ]
        budget.check()

        vintage_rows = {row["origin_unix"]: row for row in checks[3]["rows"]}
        coverage_rows = {row["origin_unix"]: row for row in checks[4]["rows"]}
        rrp = np.asarray(selected["predispatch_rrp"], dtype=np.float64)
        selected_rows = []
        for position, origin in enumerate(selected["forecast_origin_unix"]):
            origin = int(origin)
            values = rrp[position]
            selected_rows.append(
                {
                    "selection_index": int(indices[position]),
                    "origin_unix": origin,
                    "origin_aest": aest(origin),
                    "origin_hour_aest": int(datetime.fromtimestamp(origin, AEST).hour),
                    "target_window_aest": [aest(origin + 3600), aest(origin + 24 * 3600)],
                    "rrp_min": float(values.min()),
                    "rrp_max": float(values.max()),
                    "rrp_mean": round(float(values.mean()), 6),
                    **vintage_rows[origin],
                    "covered_hours": coverage_rows[origin]["covered_hours"],
                    "uncovered_hours": coverage_rows[origin]["uncovered_hours"],
                    "fill_kind": coverage_rows[origin]["fill_kind"],
                    "publication_basis": "archive_proxy",
                }
            )
        budget.check()

    preflight_path = run_dir / "preflight_tests.json"
    preflight = json.loads(preflight_path.read_text()) if preflight_path.exists() else None
    verdict = g0a.assemble_verdict(
        checks=checks, budget=budget.report(), preflight=preflight, guard_violations=guard_violations
    )
    verdict["selected_origin_count"] = len(selected_rows)
    verdict["config"] = "configs/agent_forecasting/g0a_predispatch_rrp_20261028.json"

    budget.check()
    (run_dir / "selected_origins.json").write_text(json.dumps(selected_rows, indent=2) + "\n")
    (run_dir / "g0a_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    (run_dir / "budget_report.json").write_text(
        json.dumps({**budget.report(), "phases": {"total_monotonic_s": round(time.monotonic() - started, 6)}}, indent=2) + "\n"
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "started_at_utc": started_utc,
                "finished_at_utc": utc_now(),
                "wall_seconds": round(time.monotonic() - started, 6),
                "interpreter": sys.executable,
                "python_version": platform.python_version(),
                "command": sys.argv,
                "git_head": git_head(),
                "branch": config["branch"],
                "base_commit": config["base_commit"],
                "network_disabled": True,
                "npz_path": config["npz_path"],
                "selection_rule": config["origin_selection"],
                "source_sha256": source_hashes(config),
            },
            indent=2,
        )
        + "\n"
    )
    summary = {
        "engineering": verdict["engineering"],
        "strict_asof": verdict["strict_asof"],
        "quick_veto": verdict["quick_veto"]["status"],
        "checks": verdict["checks_status"],
        "selected_origins": len(selected_rows),
    }
    print(json.dumps(summary, indent=2))


def cmd_archive(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    source = REPO_ROOT / config["output_dir"]
    destination = REPO_ROOT / config["archive_dir"]
    if not (source / "g0a_verdict.json").exists():
        raise SystemExit("refusing to archive: g0a_verdict.json missing")
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        files[str(relative)] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "created_at_utc": utc_now(),
        "algorithm": "sha256",
        "scope": "All files under this archive except artifact_manifest.json itself",
        "source_output_dir": config["output_dir"],
        "git_head_at_archive": git_head(),
        "file_count": len(files),
        "files": files,
    }
    (destination / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"archived_files": len(files), "destination": str(destination)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/agent_forecasting/g0a_predispatch_rrp_20261028.json")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("record-tests", cmd_record_tests), ("audit", cmd_audit), ("archive", cmd_archive)):
        sub = subparsers.add_parser(name)
        sub.set_defaults(handler=handler)
        sub.add_argument("--config", default="configs/agent_forecasting/g0a_predispatch_rrp_20261028.json")
    args = parser.parse_args()
    if not Path(args.config).is_absolute():
        args.config = str(REPO_ROOT / args.config)
    args.handler(args)


if __name__ == "__main__":
    main()
