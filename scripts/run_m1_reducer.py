#!/usr/bin/env python3
"""Execute the M1 label-free reducibility gate and write the verdict JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "fixtures" / "agent_forecasting"))

from forecasting.agent_forecasting import m1_reducer  # noqa: E402


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    report = m1_reducer.run_reducer()
    summary = {key: report[key] for key in (
        "verdict", "scenarios_total", "d_traces_enumerated",
        "llm_failures_checked", "internal_consistency_ok", "uses_labels_or_scores")}
    summary["non_vacuity_control"] = report["non_vacuity_control"]
    summary["witness_count"] = len(report["witnesses"])
    print(json.dumps(summary, indent=2, sort_keys=True))
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
