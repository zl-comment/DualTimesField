#!/usr/bin/env python3
"""Verify M1 evidence closure from OUTSIDE the evidence root.

Checks (audit-2): every commands.jsonl record has a unique id AND a globally
unique log path; every log exists, its header names the record id, its git_head
provenance line equals the ledger record, and its exact final line is the real
EXIT_CODE; the artifact manifest closes bidirectionally (every listed file
exists with its sha256/size and every file on disk is listed); every published
run directory under runs/ passes bidirectional manifest/COMMIT closure; no
orphan staging or lock remains; and — unless suppressed for unit tests — the
reducer recomputes to the archived report and gate.json equals the frozen
verdict mapping over it.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.agent_forecasting import m1_contract  # noqa: E402
from forecasting.agent_forecasting.m1_publication import verify_run_closure  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_ledger(root: Path):
    """Ledger/log bijection, header provenance and exact exit-code tails."""
    problems: list[str] = []
    records: list[dict] = []
    ledger = root / "commands.jsonl"
    if not ledger.exists():
        return ["commands.jsonl missing"], records
    records = [json.loads(line) for line in ledger.read_text().splitlines()
               if line.strip()]
    ids = [record["id"] for record in records]
    paths = [record["log_path"] for record in records]
    if len(ids) != len(set(ids)):
        problems.append("command ledger ids are not injective")
    if len(paths) != len(set(paths)):
        problems.append("command ledger log paths are not globally injective")
    for record in records:
        log = root / record["log_path"]
        if not log.exists():
            problems.append(f"log path missing: {record['log_path']}")
            continue
        lines = log.read_text().splitlines()
        if not lines or lines[0] != f"=== run_logged_m1 id={record['id']} ===":
            problems.append(
                f"log {record['log_path']} header does not name {record['id']}")
        head_line = next((line for line in lines
                          if line.startswith("git_head: ")), None)
        if head_line != f"git_head: {record['git_head']}":
            problems.append(
                f"log {record['log_path']} git_head provenance mismatch")
        if not lines or lines[-1] != f"EXIT_CODE={record['exit_code']}":
            problems.append(
                f"log {record['log_path']} final line is not the exact exit code")
    return problems, records


def check_manifest(root: Path) -> list[str]:
    """Bidirectional manifest closure over every file in the evidence root."""
    problems: list[str] = []
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.exists():
        return ["artifact_manifest.json missing"]
    manifest = json.loads(manifest_path.read_text())
    on_disk = {str(path.relative_to(root)): path
               for path in sorted(root.rglob("*"))
               if path.is_file() and path != manifest_path}
    listed = set(manifest.get("files", {}))
    for name in sorted(listed - set(on_disk)):
        problems.append(f"manifest lists a missing file: {name}")
    for name in sorted(set(on_disk) - listed):
        problems.append(f"file on disk is not in the manifest: {name}")
    for name, record in sorted(manifest.get("files", {}).items()):
        path = on_disk.get(name)
        if path is None:
            continue
        if (path.stat().st_size != record["size_bytes"]
                or sha256_file(path) != record["sha256"]):
            problems.append(f"manifest hash/size drift: {name}")
    return problems


def check_runs(root: Path) -> list[str]:
    problems: list[str] = []
    runs = root / "runs"
    if runs.exists():
        for final in sorted(runs.iterdir()):
            if not final.is_dir():
                problems.append(f"non-directory inside runs/: {final.name}")
                continue
            try:
                verify_run_closure(final)
            except Exception as exc:  # noqa: BLE001 - report, then fail closed
                problems.append(f"run closure failed for {final.name}: {exc}")
    staging = root / "staging"
    if staging.exists() and any(staging.iterdir()):
        problems.append("orphan staging directories remain")
    locks = root / "locks"
    if locks.exists() and any(locks.iterdir()):
        problems.append("orphan lock files remain")
    return problems


def _resolve_suite_log(root: Path, suite_str: str) -> Path | None:
    for candidate in (Path(suite_str), root / suite_str,
                      root / Path(suite_str).name, root.parent / suite_str):
        if candidate.exists():
            return candidate
    return None


def check_gate_and_reducer(root: Path, *, recompute: bool) -> list[str]:
    problems: list[str] = []
    report_path = root / "reducibility" / "reducer_report.json"
    gate_path = root / "gate.json"
    if not report_path.exists():
        return problems  # a pre-reducer evidence root is incomplete but checkable
    report = json.loads(report_path.read_text())
    if recompute:
        from forecasting.agent_forecasting import m1_reducer
        fresh = m1_reducer.run_reducer()
        for key in ("verdict", "scenarios_total", "d_traces_enumerated",
                    "llm_failures_checked", "internal_consistency_ok"):
            if fresh[key] != report.get(key):
                problems.append(
                    f"reducer recompute mismatch on {key}: "
                    f"{fresh[key]} != {report.get(key)}")
        if len(fresh["witnesses"]) != len(report.get("witnesses", [])):
            problems.append("reducer recompute witness count mismatch")
    if not gate_path.exists():
        problems.append("gate.json missing while a reducer report exists")
        return problems
    gate = json.loads(gate_path.read_text())
    engineering = gate.get("engineering", {})
    suite_log = _resolve_suite_log(root, engineering.get("suite_log", ""))
    if suite_log is None:
        problems.append(f"gate suite log missing: {engineering.get('suite_log')}")
    else:
        lines = suite_log.read_text().splitlines()
        if not lines or lines[-1] != f"EXIT_CODE={engineering.get('suite_exit_code')}":
            problems.append("gate suite exit code does not match the log tail")
        passed = engineering.get("tests_passed")
        if passed is not None and not any(
                f"{passed} passed" in line for line in lines):
            problems.append("gate tests_passed count not found in the suite log")
    expected = m1_contract.map_gate({
        **report,
        "engineering_green": engineering.get("suite_exit_code") == 0,
    })
    for key in ("scientific_decision", "stop_token", "metrics_status",
                "advance_m1", "reducibility_verdict", "mission_goal_achieved"):
        if gate.get(key) != expected.get(key):
            problems.append(
                f"gate field {key} does not match the frozen verdict mapping")
    if gate.get("metrics_status") != "NOT_EVALUATED":
        problems.append("gate must keep metrics NOT_EVALUATED")
    return problems


def check_run_entry(root: Path) -> list[str]:
    """Audit-3/H1: the closure verifier itself enters through verify_run_entry.

    The evidence tree must close at a resolvable git HEAD over a tracked-clean
    tree whose frozen config file equals its committed bytes, and every ledger
    record's git_head must be a commit reachable from the current HEAD (a log
    claiming a head outside this branch's history is not provenance).
    """
    problems: list[str] = []
    try:
        report = m1_contract.verify_run_entry(ROOT)
    except m1_contract.M1IdentityDrift as drift:
        return [f"run entry identity drift: {drift}"]
    head = report["git_head"]
    ledger = root / "commands.jsonl"
    if not ledger.exists():
        return problems
    import subprocess
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        recorded = record.get("git_head")
        if recorded == head:
            continue
        reachable = subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
             str(recorded), head], capture_output=True, check=False)
        if reachable.returncode != 0:
            problems.append(
                f"ledger record {record.get('id')} git_head {recorded} is not "
                "reachable from the current HEAD")
    return problems


def verify_evidence_root(root: Path, *, recompute: bool = True) -> list[str]:
    """All closure problems of one evidence root; an empty list means closed."""
    problems, _records = check_ledger(root)
    problems += check_manifest(root)
    problems += check_runs(root)
    problems += check_gate_and_reducer(root, recompute=recompute)
    return problems


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_m1_evidence.py <evidence_root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    problems = verify_evidence_root(root)
    problems += check_run_entry(root)
    ledger = root / "commands.jsonl"
    count = 0
    if ledger.exists():
        count = len([line for line in ledger.read_text().splitlines()
                     if line.strip()])
    if problems:
        print("CLOSURE FAILED")
        for problem in problems:
            print(f" - {problem}")
        return 1
    runs = root / "runs"
    print(f"CLOSURE OK: commands={count} "
          f"runs={len(list(runs.iterdir())) if runs.exists() else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
