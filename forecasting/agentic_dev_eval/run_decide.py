"""Arm B/C/D decision runner over the saved arm-A base forecasts (repo venv)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import arms, data, protocol


def load_base(output_dir: str, region: str) -> dict:
    path = Path(output_dir) / "base" / f"{region}_base.npz"
    if not path.exists():
        raise FileNotFoundError(f"base forecast missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def run(region: str, arm: str, output_dir: str, data_root: str, limit: int | None) -> int:
    base = load_base(output_dir, region)
    records, grid_info = data.build_origin_grid(region, data_root)
    base_origins = [int(v) for v in base["origin_unix"]]
    grid_origins = [r.origin_unix for r in records]
    if limit is not None:
        records = records[:limit]
        grid_origins = grid_origins[:limit]
    if grid_origins != base_origins:
        raise RuntimeError(f"origin identity mismatch between grid and base npz for {region}")
    if not np.array_equal(np.stack([r.target_time_unix for r in records]), base["target_time_unix"]):
        raise RuntimeError(f"target identity mismatch for {region}")
    if not np.array_equal(np.stack([r.actual for r in records]), base["actual"]):
        raise RuntimeError(f"label identity mismatch for {region}")

    frame, _ = data.load_region_frame(region, data_root)
    decider = {"B": arms.decide_b, "C": arms.decide_c, "D": arms.decide_d}[arm]

    quantiles = np.full((len(records), protocol.HORIZON_HOURS, len(protocol.QUANTILE_LEVELS)),
                        np.nan, dtype=np.float64)
    actions, exit_codes, failures = [], [], []
    decision_rows = []
    for i, record in enumerate(records):
        if arm == "D":
            decision = decider(record, base["quantiles"][i], frame)
        elif arm == "C":
            decision = decider(record, base["quantiles"][i])
        else:
            decision = decider(record, base["quantiles"][i])
        actions.append(decision["action"])
        exit_codes.append(decision["exit_code"])
        if decision["exit_code"] == 0:
            quantiles[i] = protocol.apply_action(base["quantiles"][i], record.history_prices,
                                                 decision["action"])
        else:
            failures.append({"origin_local": record.origin_local,
                             "reason": decision["failure_reason"],
                             "detail": decision["detail"]})
        decision_rows.append({"origin_local": record.origin_local,
                              "origin_unix": record.origin_unix, "half": record.half,
                              **decision})
        print(json.dumps({"region": region, "arm": arm, "i": i,
                          "origin": record.origin_local,
                          "action": decision["action"] or "-",
                          "exit_code": decision["exit_code"],
                          "failure": decision["failure_reason"],
                          "latency_ms": decision["latency_ms"],
                          "llm_calls": decision["llm_calls"],
                          "tool_calls": decision["tool_calls"]}), flush=True)

    out = Path(output_dir)
    (out / "decisions").mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "decisions" / f"{region}_{arm}.jsonl"
    with jsonl_path.open("w") as stream:
        for row in decision_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    npz_path = out / "predictions" / f"{region}_{arm}.npz"
    with npz_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            origin_unix=np.asarray(base_origins, dtype=np.int64),
            origin_local=base["origin_local"], half=base["half"],
            target_time_unix=base["target_time_unix"], actual=base["actual"],
            quantiles=quantiles,
            actions=np.asarray(actions),
            exit_codes=np.asarray(exit_codes, dtype=np.int64),
        )
    n_fail = len(failures)
    print(json.dumps({"status": "done", "region": region, "arm": arm,
                      "n_origins": len(records), "n_success": len(records) - n_fail,
                      "n_fail": n_fail,
                      "failures": failures[:10],
                      "total_prompt_tokens": sum(r["tokens"]["prompt"] for r in decision_rows),
                      "total_completion_tokens": sum(r["tokens"]["completion"] for r in decision_rows),
                      "decisions": str(jsonl_path), "predictions": str(npz_path)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=protocol.REGIONS)
    parser.add_argument("--arm", required=True, choices=("B", "C", "D"))
    parser.add_argument("--data-root", default=data.DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir",
                        default="outputs/forecasting/agentic_m2m3_dev_c2_20261030_r6")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    return run(args.region, args.arm, args.output_dir, args.data_root, args.limit)


if __name__ == "__main__":
    sys.exit(main())
