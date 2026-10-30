"""r8 scorer driver.

Reuses the frozen r6 scoring implementation verbatim: every numeric field is
produced by forecasting.agentic_dev_eval.run_score.score_region / macro_average
(byte-identical to the r6 versions since r8 changed only arms.py,
llm_transport.py and tests). Differences from run_score.run are limited to the
round identity fields (round_id/branch) and the report writer; the built-in
arm-A self-check from run_score.run is kept unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from forecasting.agentic_dev_eval import metrics, protocol, run_score

ROUND_ID = "agentic-m2m3-decision-only-c2-20261030-r8"
BRANCH = "research/agentic-m2m3-decision-only-c2-20261030-r8"


def build_summary(output_dir: str, regions: tuple[str, ...]) -> dict:
    region_results = [run_score.score_region(output_dir, region) for region in regions]
    return {
        "round_id": ROUND_ID,
        "branch": BRANCH,
        "protocol": {
            "quantile_levels": list(protocol.QUANTILE_LEVELS),
            "actions": list(protocol.ACTIONS), "seed": protocol.SEED,
            "history_hours": protocol.HISTORY_HOURS, "horizon_hours": protocol.HORIZON_HOURS,
            "grid_first_date": protocol.GRID_FIRST_DATE, "grid_step_days": protocol.GRID_STEP_DAYS,
            "frozen_constants": {
                "shift_fraction": protocol.SHIFT_FRACTION,
                "spread_factor": protocol.SPREAD_FACTOR,
                "rule_momentum_threshold": protocol.RULE_MOMENTUM_THRESHOLD,
                "rule_vol_high": protocol.RULE_VOL_HIGH, "rule_vol_low": protocol.RULE_VOL_LOW,
                "d_max_turns": protocol.D_MAX_TURNS, "d_max_tool_calls": protocol.D_MAX_TOOL_CALLS,
                "llm_model": protocol.LLM_MODEL, "llm_temperature": protocol.LLM_TEMPERATURE,
                "d_max_tokens": protocol.D_MAX_TOKENS,
            },
        },
        "regions": region_results,
        "macro": run_score.macro_average(region_results),
    }


def run(output_dir: str, regions: tuple[str, ...]) -> int:
    summary = build_summary(output_dir, regions)
    out = Path(output_dir) / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    # Independent self-check carried over unchanged from run_score.run:
    # re-derive every region's FULL-split A-arm MAE from the npz.
    for region_result in summary["regions"]:
        payload = run_score._load_arm_npz(output_dir, region_result["region"], "A")
        recomputed = metrics.score_arrays(payload["quantiles"], payload["actual"])
        if abs(recomputed["mae"] - region_result["per_arm"]["FULL"]["A"]["mae"]) > 1e-12:
            raise RuntimeError(f"self-check failed for {region_result['region']} arm A")
    print(json.dumps({"status": "ok", "results": str(out / "summary.json"),
                      "macro_full": summary["macro"]["per_arm"]["FULL"]}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--regions", default=",".join(protocol.REGIONS))
    args = parser.parse_args(argv)
    regions = tuple(r for r in args.regions.split(",") if r)
    return run(args.output_dir, regions)


if __name__ == "__main__":
    sys.exit(main())
