#!/usr/bin/env python3
"""Build the M1 gate file from the suite log and the reducer verdict JSON."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.agent_forecasting import m1_contract  # noqa: E402


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: build_m1_gate.py <suite_log> <reducer_json> <gate_out>",
              file=sys.stderr)
        return 2
    suite_log, reducer_path, gate_out = (Path(sys.argv[1]), Path(sys.argv[2]),
                                         Path(sys.argv[3]))
    log_text = suite_log.read_text()
    match = re.search(r"^EXIT_CODE=(\d+)$", log_text, flags=re.M)
    if match is None:
        print(f"suite log carries no real EXIT_CODE line: {suite_log}",
              file=sys.stderr)
        return 3
    exit_code = int(match.group(1))
    passed = re.search(r"^(\d+) passed", log_text, flags=re.M)
    reducer = json.loads(reducer_path.read_text())
    gate = m1_contract.map_gate({
        "verdict": reducer["verdict"],
        "engineering_green": exit_code == 0,
    })
    gate.update({
        "engineering": {"suite_exit_code": exit_code, "suite_log":
                        str(suite_log), "tests_passed": int(passed.group(1))
                        if passed else None},
        "reducibility": {key: reducer[key] for key in (
            "verdict", "scenarios_total", "d_traces_enumerated",
            "llm_failures_checked", "internal_consistency_ok",
            "non_vacuity_control", "uses_labels_or_scores")},
        "metrics_status": "NOT_EVALUATED",
    })
    gate_out.parent.mkdir(parents=True, exist_ok=True)
    gate_out.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
