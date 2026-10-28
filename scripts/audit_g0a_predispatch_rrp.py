"""Minimal G0-A audit runner (protocol §7 replay commands).

record-tests: turn the true pytest exit code captured in preflight_tests.exit
              into preflight_tests.json (no pipe exit codes anywhere).
audit:        run the six frozen checks on the frozen NPZ for the selected
              origins, under the budget guard, network-disabled, and the
              path allowlist; write g0a_verdict.json and evidence files.
archive:      mirror the outputs run dir into the committed logs dir and
              write artifact_manifest.json (sha256 of every file).

Gate-reliability contract (2026-10-28 repair round 2):
- The budget clock is anchored at process start (best effort via
  /proc/self/stat, else the first line of this module, always before the
  numpy/harness imports) and covers module imports, CLI parsing, config
  load, all checks, the final source hashes/HEAD, and evidence finalization;
  budget elapsed is never smaller than the recorded full-run wall time.
- Success evidence is staged in a temporary directory and published with a
  single atomic rename only AFTER all final checks passed; partial success
  sets cannot appear in the run dir.
- Any failure (including BudgetExceeded in the final phases or a publication
  fault) exits non-zero and atomically writes an explicit ENGINEERING_FAIL
  record next to the run dir; already-published evidence is never modified
  or overwritten by a failed or rerun audit.

This script never reads labels, never touches the network, never modifies
the NPZ, and cannot emit a strict as-of PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROCESS_IMPORT_STARTED = time.monotonic()  # captured before the numpy/harness imports


def _process_start_monotonic() -> float:
    """Best-effort interpreter start on the monotonic clock: /proc/self/stat
    field 22 (starttime, clock ticks since boot) against CLOCK_BOOTTIME.
    Falls back to the first line of this module when /proc is unavailable."""
    try:
        stat_text = Path("/proc/self/stat").read_text()
        fields = stat_text.rsplit(")", 1)[1].split()
        start_ticks = float(fields[19])  # field 22 overall; here fields start at 3
        ticks_per_second = os.sysconf("SC_CLK_TCK")
        boottime_elapsed = time.clock_gettime(time.CLOCK_BOOTTIME)
        candidate = boottime_elapsed - start_ticks / ticks_per_second
        return min(candidate, PROCESS_IMPORT_STARTED)
    except Exception:
        return PROCESS_IMPORT_STARTED


AUDIT_STARTED_MONOTONIC = _process_start_monotonic()

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from forecasting.agent_forecasting import g0a_predispatch as g0a  # noqa: E402

AEST = timezone(timedelta(hours=10))

EVIDENCE_FILENAMES = (
    "g0a_verdict.json",
    "budget_report.json",
    "run_manifest.json",
    "selected_origins.json",
)
PREFLIGHT_FILENAMES = (
    "preflight_tests.json",
    "preflight_tests.txt",
    "preflight_tests.exit",
)


class AuditPublishError(RuntimeError):
    """Publication refused: the run dir already holds evidence or unknown files."""


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


def git_head(repo_root: Path = REPO_ROOT) -> str:
    """Single-process HEAD reader (budget: processes=1): resolves .git/HEAD to
    a commit id by reading the ref file or packed-refs directly. Never spawns
    a git subprocess."""
    git_dir = repo_root / ".git"
    if git_dir.is_file():  # worktree/submodule gitdir pointer
        pointer = git_dir.read_text().strip()
        git_dir = Path(pointer.split(":", 1)[1].strip())
        if not git_dir.is_absolute():
            git_dir = (repo_root / git_dir).resolve()
    head = (git_dir / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head  # detached HEAD already carries the commit id
    ref = head[5:]
    ref_file = git_dir.joinpath(*ref.split("/"))
    if ref_file.exists():
        return ref_file.read_text().strip()
    for line in (git_dir / "packed-refs").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == ref:
            return parts[0]
    raise RuntimeError(f"HEAD ref {ref} not found under {git_dir}")


def display_path(path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def load_config(path: str) -> dict:
    config = json.loads(Path(path).read_text())
    run_dir = REPO_ROOT / config["output_dir"]
    if run_dir.resolve() in (REPO_ROOT.resolve(), REPO_ROOT.parent.resolve(), Path("/")):
        raise SystemExit("FROZEN_BINDING_VIOLATION: output_dir must be a dedicated run dir")
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


def source_hashes(config: dict, config_path=None) -> dict:
    config_path = Path(config_path) if config_path is not None else (
        REPO_ROOT / "configs/agent_forecasting/g0a_predispatch_rrp_20261028.json"
    )
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    return {
        "harness_module": sha256_file(REPO_ROOT / config["harness_module"]),
        "audit_script": sha256_file(REPO_ROOT / config["audit_script"]),
        "config": sha256_file(config_path),
        "protocol": sha256_file(REPO_ROOT / config["protocol"]),
        "builder": sha256_file(REPO_ROOT / config["builder_path"]),
        "npz": sha256_file(REPO_ROOT / config["npz_path"]),
        "tests": sha256_file(REPO_ROOT / config["tests"]),
    }


def disarm_wall_deadline() -> None:
    try:
        signal.setitimer(signal.ITIMER_REAL, 0)
    except (AttributeError, OSError, ValueError):
        pass


def _fsync_tree(directory: Path) -> None:
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_failure_record(*, run_dir, phase, exc, budget, staging, config_path) -> Path:
    """Atomically write an explicit ENGINEERING_FAIL record NEXT TO the run dir
    (never inside it, so existing evidence can never be touched). The record
    itself is staged then os.replace'd into place."""
    disarm_wall_deadline()
    if budget is not None:
        budget_state = budget.report()
    else:
        budget_state = {
            "wall_seconds_budget": g0a.WALL_SECONDS_BUDGET,
            "wall_seconds_elapsed": round(time.monotonic() - AUDIT_STARTED_MONOTONIC, 6),
            "within_budget": False,
        }
    if run_dir is not None:
        failures_dir = run_dir.parent / (run_dir.name + "_failures")
    else:
        failures_dir = REPO_ROOT / "outputs/forecasting/g0a_audit_failures"
    record = {
        "audit": "g0a_predispatch_rrp_asof_preaudit",
        "record_kind": "g0a_failure",
        "engineering": "ENGINEERING_FAIL",
        "strict_asof": g0a.STRICT_ASOF_STATUS,
        "quick_veto": "not_evaluated_or_incomplete",
        "failed_phase": phase,
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback_tail": traceback.format_exception(type(exc), exc, exc.__traceback__)[-8:],
        "budget": budget_state,
        "evidence_published": False,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "staging_dir": str(staging) if staging is not None else None,
        "config": str(config_path),
        "exit_code": 3,
        "written_atomically": True,
        "recorded_at_utc": utc_now(),
        "pid": os.getpid(),
        "command": sys.argv,
        "note": (
            "The audit failed before atomic publication; no success evidence set "
            "exists in the run dir. If staging_dir is set, it holds the complete "
            "unpublished evidence for manual recovery."
        ),
    }
    failures_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f") + "Z"
    final_path = failures_dir / f"g0a_failure_{stamp}_{os.getpid()}.json"
    tmp_path = failures_dir / f".tmp-g0a_failure_{os.getpid()}.json"
    tmp_path.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp_path, final_path)
    return final_path


def _publish_run_dir(run_dir: Path, staging: Path) -> None:
    """One-shot atomic publication: the staged directory becomes the run dir
    via a single rename. At this point the run dir may only hold preflight
    input files; existing evidence or unknown occupants abort publication
    before anything is removed."""
    if run_dir.resolve() in (REPO_ROOT.resolve(), REPO_ROOT.parent.resolve(), Path("/")):
        raise AuditPublishError(f"unsafe run dir refused: {run_dir}")
    if run_dir.exists():
        occupants = {path.name for path in run_dir.iterdir()}
        clash = occupants.intersection(EVIDENCE_FILENAMES)
        if clash:
            raise AuditPublishError(
                f"refusing to overwrite published evidence: {sorted(clash)} already in {run_dir}"
            )
        unknown = occupants.difference(PREFLIGHT_FILENAMES)
        if unknown:
            raise AuditPublishError(
                f"refusing publication: unexpected occupants in run dir: {sorted(unknown)}"
            )
        for name in sorted(occupants):
            (run_dir / name).unlink()  # preflight inputs only; copies are staged
        run_dir.rmdir()
    # The rename below is the single atomic commit of the whole evidence set;
    # the wall deadline is disarmed immediately before it so the commit cannot
    # be interrupted after all checks passed.
    disarm_wall_deadline()
    os.rename(staging, run_dir)


def cmd_record_tests(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    run_dir = REPO_ROOT / config["output_dir"]
    if (run_dir / "g0a_verdict.json").exists():
        raise SystemExit(
            "refusing record-tests: published evidence exists in the run dir; "
            "rerunning would mutate an evidence directory"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
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
    audit_entry = time.monotonic()
    pre_audit_s = audit_entry - AUDIT_STARTED_MONOTONIC
    started_utc = utc_now()
    phase = "budget_anchor"
    run_dir = None
    budget = None
    staging = None
    try:
        # Resource guard first (protocol §5): the budget is anchored at
        # process start and covers config load, the frozen-binding hashes,
        # all checks, evidence staging, the final source hashes and HEAD;
        # the SIGALRM deadline interrupts immediately if it expires.
        config = load_config(args.config)
        phase = "config_binding"
        if float(config["budget"]["wall_seconds"]) != g0a.WALL_SECONDS_BUDGET:
            raise SystemExit(
                "FROZEN_BINDING_VIOLATION: config wall_seconds disagrees with "
                f"protocol budget {g0a.WALL_SECONDS_BUDGET}s"
            )
        run_dir = REPO_ROOT / config["output_dir"]
        remaining = g0a.WALL_SECONDS_BUDGET - (time.monotonic() - AUDIT_STARTED_MONOTONIC)
        if remaining <= 0:
            raise g0a.BudgetExceeded(
                f"wall budget {g0a.WALL_SECONDS_BUDGET:.3f}s already exhausted at "
                f"process start +{time.monotonic() - AUDIT_STARTED_MONOTONIC:.3f}s"
            )
        budget = g0a.BudgetGuard(
            g0a.WALL_SECONDS_BUDGET, started_monotonic=AUDIT_STARTED_MONOTONIC
        )
        npz = enforce_frozen_binding(config)
        allow_roots = [REPO_ROOT / root for root in config["allow_roots"]]
        guard_violations = []
        for source_path, label in (
            (config["harness_module"], "harness_module"),
            (config["audit_script"], "audit_script"),
        ):
            hits = g0a.scan_source_for_forbidden((REPO_ROOT / source_path).read_text())
            guard_violations.extend(f"static scan {label}: {hit}" for hit in hits)

        phase = "checks"
        with g0a.wall_deadline(remaining):
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
                        selected["predispatch_rrp"],
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
                    coverage = coverage_rows[origin]
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
                            "covered_hours": coverage["covered_hours"],
                            "uncovered_hours": coverage["uncovered_hours"],
                            "fill_kind": coverage["fill_kind"],
                            "tail_repeat_verified": coverage.get("tail_repeat_verified"),
                            **{
                                k: coverage[k]
                                for k in ("last_covered_hour", "last_covered_value", "tail_hours_after_last_covered")
                                if k in coverage
                            },
                            "publication_basis": "archive_proxy",
                        }
                    )
                budget.check()
            after_checks = time.monotonic()

            # Final hashes and HEAD stay inside the guarded region (single
            # process: HEAD is read from .git directly, no subprocess).
            phase = "final_hashes"
            preflight_path = run_dir / "preflight_tests.json"
            preflight = json.loads(preflight_path.read_text()) if preflight_path.exists() else None
            verdict = g0a.assemble_verdict(
                checks=checks, budget=budget.report(), preflight=preflight, guard_violations=guard_violations
            )
            verdict["selected_origin_count"] = len(selected_rows)
            verdict["config"] = display_path(args.config)
            source_sha256 = source_hashes(config, args.config)
            head = git_head()
            budget.check()
            after_hashes = time.monotonic()

            # Evidence finalization: stage the COMPLETE success set in a
            # temporary sibling directory; nothing touches the run dir yet.
            phase = "evidence_staging"
            staging = run_dir.parent / f"{run_dir.name}.publish-{os.getpid()}-{int(time.time() * 1000)}"
            staging.mkdir(parents=True)
            (staging / "selected_origins.json").write_text(json.dumps(selected_rows, indent=2) + "\n")
            (staging / "g0a_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
            manifest_wall = round(time.monotonic() - AUDIT_STARTED_MONOTONIC, 6)
            (staging / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "started_at_utc": started_utc,
                        "finished_at_utc": utc_now(),
                        "wall_seconds": manifest_wall,
                        "wall_clock_anchor": "process_start",
                        "phases": {"pre_audit_process_import_cli_s": round(pre_audit_s, 6)},
                        "evidence_publication": {
                            "mode": "staging_dir_single_atomic_rename",
                            "partial_success_sets": "impossible",
                        },
                        "interpreter": sys.executable,
                        "python_version": platform.python_version(),
                        "command": sys.argv,
                        "git_head": head,
                        "git_head_source": "in-process .git read (no subprocess)",
                        "sigalrm_deadline_seconds": g0a.WALL_SECONDS_BUDGET,
                        "branch": config["branch"],
                        "base_commit": config["base_commit"],
                        "network_disabled": True,
                        "npz_path": config["npz_path"],
                        "selection_rule": config["origin_selection"],
                        "source_sha256": source_sha256,
                    },
                    indent=2,
                )
                + "\n"
            )
            for name in PREFLIGHT_FILENAMES:
                if (run_dir / name).exists():
                    shutil.copyfile(run_dir / name, staging / name)
            _fsync_tree(staging)
            budget_report = {
                **budget.report(),
                "wall_clock_anchor": "process_start",
                "phases": {
                    "pre_audit_process_import_cli_s": round(pre_audit_s, 6),
                    "checks_s": round(after_checks - audit_entry, 6),
                    "final_hashes_head_s": round(after_hashes - after_checks, 6),
                    "evidence_staging_s": round(time.monotonic() - after_hashes, 6),
                    "total_monotonic_s": budget.report()["wall_seconds_elapsed"],
                },
            }
            if budget_report["wall_seconds_elapsed"] < manifest_wall:
                raise AuditPublishError(
                    "budget accounting invariant violated: "
                    f"{budget_report['wall_seconds_elapsed']}s < run wall {manifest_wall}s"
                )
            (staging / "budget_report.json").write_text(json.dumps(budget_report, indent=2) + "\n")
            _fsync_tree(staging)
            budget.check()

            phase = "publication"
            _publish_run_dir(run_dir, staging)
            staging = None  # published: nothing left to recover
    except BaseException as exc:  # every failure becomes an explicit record
        try:
            write_failure_record(
                run_dir=run_dir, phase=phase, exc=exc, budget=budget,
                staging=staging, config_path=args.config,
            )
        except Exception as record_exc:  # noqa: BLE001
            print(f"FAILED TO WRITE FAILURE RECORD: {record_exc!r}", file=sys.stderr)
            raise SystemExit(4) from exc
        raise SystemExit(3) from exc

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
    if getattr(args, "archive_subdir", None):
        # Repair runs archive into a sibling subdir; committed earlier runs
        # stay read-only (protocol §6 归档只读).
        destination = destination / args.archive_subdir
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
    subparsers.choices["archive"].add_argument(
        "--archive-subdir",
        default=None,
        help="archive this run under a subdir instead of the canonical archive root",
    )
    args = parser.parse_args()
    if not Path(args.config).is_absolute():
        args.config = str(REPO_ROOT / args.config)
    args.handler(args)


if __name__ == "__main__":
    main()
