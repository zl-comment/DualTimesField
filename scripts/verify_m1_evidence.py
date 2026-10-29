#!/usr/bin/env python3
"""Verify M1 evidence closure from OUTSIDE the evidence root.

Checks: every commands.jsonl record has a unique id and an existing log whose tail
carries the same real EXIT_CODE; every published run directory under runs/ passes
bidirectional manifest/COMMIT closure; no orphan staging remains; no success
artifact was amended after COMMIT.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from forecasting.agent_forecasting.m1_publication import verify_run_closure


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_m1_evidence.py <evidence_root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    problems: list[str] = []
    ledger = root / "commands.jsonl"
    if not ledger.exists():
        problems.append("commands.jsonl missing")
    else:
        records = [json.loads(line) for line in ledger.read_text().splitlines()
                   if line.strip()]
        ids = [record["id"] for record in records]
        if len(ids) != len(set(ids)):
            problems.append("command ledger ids are not injective")
        for record in records:
            log = root / record["log_path"]
            if not log.exists():
                problems.append(f"log path missing: {record['log_path']}")
                continue
            text = log.read_text()
            tail = text.rstrip().rsplit("EXIT_CODE=", 1)
            if len(tail) != 2 or tail[1].strip() != str(record["exit_code"]):
                problems.append(
                    f"log {record['log_path']} tail exit code does not match ledger")
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
    if problems:
        print("CLOSURE FAILED")
        for problem in problems:
            print(f" - {problem}")
        return 1
    print(f"CLOSURE OK: commands={len(ids) if ledger.exists() else 0} "
          f"runs={len(list(runs.iterdir())) if runs.exists() else 0}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
