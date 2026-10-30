"""Independent r8 audit: recompute every reported number from raw artifacts
with self-contained numpy code (no import of the package scoring modules),
then diff against the machine summary.json and the r6 archive.

Reads:
  r8 output root:  base/, predictions/, decisions/, results/summary.json
  r6 output root:  frozen A/B/C artifacts + r6 D predictions/decisions
Writes:
  <r8_root>/results/audit_independent.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

Q_LEVELS = (0.05, 0.10, 0.50, 0.90, 0.95)


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name] for name in z.files}


def score(q: np.ndarray, y: np.ndarray) -> dict:
    q = np.sort(np.asarray(q, dtype=np.float64), axis=-1)
    y = np.asarray(y, dtype=np.float64)
    mae = float(np.abs(q[:, :, 2] - y).mean())
    error = y[..., None] - q
    levels = np.asarray(Q_LEVELS)
    pinball = np.maximum(levels * error, (levels - 1) * error)
    return {
        "n_cells": int(y.size),
        "mae": mae,
        "quantile_score": float(2 * pinball.mean()),
        "coverage_80": float(((y >= q[:, :, 1]) & (y <= q[:, :, 3])).mean()),
        "coverage_90": float(((y >= q[:, :, 0]) & (y <= q[:, :, 4])).mean()),
    }


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def usage_from_rows(rows: list[dict]) -> dict:
    lat = [r["latency_ms"] for r in rows] or [0.0]
    fails = [r for r in rows if r["exit_code"] != 0]
    reasons: dict[str, int] = {}
    for r in fails:
        reasons[r["failure_reason"]] = reasons.get(r["failure_reason"], 0) + 1
    return {
        "n": len(rows),
        "n_success": len(rows) - len(fails),
        "failure_reasons": reasons,
        "latency_ms_mean": float(np.mean(lat)),
        "latency_ms_max": float(np.max(lat)),
        "llm_calls_total": int(sum(r["llm_calls"] for r in rows)),
        "tool_calls_total": int(sum(r["tool_calls"] for r in rows)),
        "prompt_tokens_total": int(sum(r["tokens"]["prompt"] for r in rows)),
        "completion_tokens_total": int(sum(r["tokens"]["completion"] for r in rows)),
    }


def phase_breakdown(rows: list[dict]) -> dict:
    """detail fields look like 'phase=<reason> ...'; count final-turn entry reasons."""
    phases: dict[str, int] = {}
    for r in rows:
        detail = str(r.get("detail") or "")
        if "phase=" in detail:
            tag = detail.split("phase=", 1)[1].split()[0].split("->")[0]
            phases[tag] = phases.get(tag, 0) + 1
    return phases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r8-root", required=True)
    parser.add_argument("--r6-root", required=True)
    args = parser.parse_args()
    r8 = Path(args.r8_root)
    r6 = Path(args.r6_root)
    regions = ("NSW1", "QLD1", "TAS1")

    audit: dict = {"regions": {}, "comparisons": {"max_abs_diff_vs_summary": {}}}

    for region in regions:
        base = load_npz(r8 / "base" / f"{region}_base.npz")
        arms = {"A": base}
        for arm in ("B", "C", "D"):
            arms[arm] = load_npz(r8 / "predictions" / f"{region}_{arm}.npz")

        # identity: origins/targets/labels identical across arms and vs base
        ident = {}
        for arm in ("B", "C", "D"):
            ident[arm] = bool(
                np.array_equal(arms[arm]["origin_unix"], base["origin_unix"])
                and np.array_equal(arms[arm]["target_time_unix"], base["target_time_unix"])
                and np.array_equal(arms[arm]["actual"], base["actual"]))

        # frozen A/B/C artifacts byte-identical to r6
        frozen_same = {}
        for sub, names in (("base", [f"{region}_base.npz"]),
                           ("predictions", [f"{region}_{arm}.npz" for arm in ("B", "C")]),
                           ("decisions", [f"{region}_{arm}.jsonl" for arm in ("B", "C")])):
            for name in names:
                a = (r8 / sub / name).read_bytes()
                b = (r6 / sub / name).read_bytes()
                frozen_same[f"{sub}/{name}"] = a == b

        per_arm = {}
        for arm in ("A", "B", "C", "D"):
            payload = arms[arm]
            exit_codes = (np.zeros(len(base["origin_unix"]), dtype=np.int64)
                          if arm == "A" else np.asarray(payload["exit_codes"]))
            success = exit_codes == 0
            entry = {"n_origins": int(len(exit_codes)), "n_success": int(success.sum())}
            if success.any():
                entry.update(score(payload["quantiles"][success], base["actual"][success]))
            per_arm[arm] = entry

        increments = {}
        for left, right in (("D", "C"), ("D", "B")):
            shared = (np.asarray(arms[left]["exit_codes"]) == 0) & \
                     (np.asarray(arms[right]["exit_codes"]) == 0)
            if shared.any():
                ls = score(arms[left]["quantiles"][shared], base["actual"][shared])
                rs = score(arms[right]["quantiles"][shared], base["actual"][shared])
                increments[f"{left}-{right}"] = {
                    "n_shared_origins": int(shared.sum()),
                    "mae_delta": ls["mae"] - rs["mae"],
                    "quantile_score_delta": ls["quantile_score"] - rs["quantile_score"],
                    "coverage_80_delta": ls["coverage_80"] - rs["coverage_80"],
                    "coverage_90_delta": ls["coverage_90"] - rs["coverage_90"],
                }
            else:
                increments[f"{left}-{right}"] = {"n_shared_origins": 0}

        usage = {arm: usage_from_rows(load_rows(r8 / "decisions" / f"{region}_{arm}.jsonl"))
                 for arm in ("B", "C", "D")}
        d_rows = load_rows(r8 / "decisions" / f"{region}_D.jsonl")
        audit["regions"][region] = {
            "identity_all_arms_match_base": ident,
            "frozen_abc_identical_to_r6": frozen_same,
            "per_arm": per_arm,
            "increments": increments,
            "usage": usage,
            "D_final_turn_entry_phases": phase_breakdown(d_rows),
            "D_actions": {a: int(sum(1 for r in d_rows if r["action"] == a))
                          for a in sorted({r["action"] for r in d_rows})},
        }

    # macro (mean over regions with shared>0; n_shared summed) mirroring run_score
    macro_inc = {}
    for pair in ("D-C", "D-B"):
        vals = [audit["regions"][r]["increments"][pair] for r in regions
                if audit["regions"][r]["increments"][pair]["n_shared_origins"] > 0]
        macro_inc[pair] = {
            "n_shared_origins": int(sum(v["n_shared_origins"] for v in vals)),
            "mae_delta_macro": float(np.mean([v["mae_delta"] for v in vals])),
            "quantile_score_delta_macro": float(np.mean([v["quantile_score_delta"] for v in vals])),
        }
    audit["macro_increments"] = macro_inc

    # diff audit numbers vs machine summary.json
    summary = json.loads((r8 / "results" / "summary.json").read_text())
    max_diff = 0.0
    for region_result in summary["regions"]:
        region = region_result["region"]
        for arm in ("A", "B", "C", "D"):
            for key in ("mae", "quantile_score", "coverage_80", "coverage_90"):
                mval = region_result["per_arm"]["FULL"][arm].get(key)
                aval = audit["regions"][region]["per_arm"][arm].get(key)
                if mval is not None and aval is not None:
                    max_diff = max(max_diff, abs(mval - aval))
            a_entry = audit["regions"][region]["per_arm"][arm]
            m_succ = region_result["per_arm"]["FULL"][arm].get("n_success")
            a_succ = a_entry.get("n_success")
            if m_succ is not None and a_succ is not None:
                max_diff = max(max_diff, abs(m_succ - a_succ))
                m_fail = region_result["per_arm"]["FULL"][arm].get("n_fail")
                a_fail = a_entry["n_origins"] - a_succ
                if m_fail is not None:
                    max_diff = max(max_diff, abs(m_fail - a_fail))
        for pair in ("D-C", "D-B"):
            mval = region_result["increments"]["FULL"][pair]
            aval = audit["regions"][region]["increments"][pair]
            if mval.get("n_shared_origins", 0) > 0:
                max_diff = max(max_diff, abs(mval["mae_delta"] - aval["mae_delta"]),
                               abs(mval["quantile_score_delta"] - aval["quantile_score_delta"]))
    for pair in ("D-C", "D-B"):
        mval = summary["macro"]["increments"]["FULL"][pair]
        aval = macro_inc[pair]
        max_diff = max(max_diff, abs(mval["mae_delta_macro"] - aval["mae_delta_macro"]),
                       abs(mval["quantile_score_delta_macro"] - aval["quantile_score_delta_macro"]))
    audit["comparisons"]["max_abs_diff_vs_summary"]["FULL_metrics_and_increments"] = max_diff

    # r6 vs r8 D comparison (failure composition, success sets, per-origin action flips)
    r6_vs_r8 = {"regions": {}}
    for region in regions:
        r6_d = load_npz(r6 / "predictions" / f"{region}_D.npz")
        r8_d = load_npz(r8 / "predictions" / f"{region}_D.npz")
        r6_rows = load_rows(r6 / "decisions" / f"{region}_D.jsonl")
        r8_rows = load_rows(r8 / "decisions" / f"{region}_D.jsonl")
        r6_ok = np.asarray(r6_d["exit_codes"]) == 0
        r8_ok = np.asarray(r8_d["exit_codes"]) == 0
        r6_reasons: dict[str, int] = {}
        for row in r6_rows:
            if row["exit_code"] != 0:
                r6_reasons[row["failure_reason"]] = r6_reasons.get(row["failure_reason"], 0) + 1
        both = int((r6_ok & r8_ok).sum())
        gained = int((~r6_ok & r8_ok).sum())
        lost = int((r6_ok & ~r8_ok).sum())
        r6_actions = {i: r6_rows[i]["action"] for i in range(len(r6_rows))}
        flips = int(sum(1 for i in range(len(r8_rows))
                        if r6_ok[i] and r8_ok[i]
                        and r6_actions[i] != r8_rows[i]["action"]))
        r6_vs_r8["regions"][region] = {
            "r6_success": int(r6_ok.sum()), "r8_success": int(r8_ok.sum()),
            "both_success": both, "gained_success": gained, "lost_success": lost,
            "action_flips_on_both_success": flips,
            "r6_failure_reasons": r6_reasons,
            "r8_failure_reasons": audit["regions"][region]["usage"]["D"]["failure_reasons"],
        }
    totals = {
        key: sum(v[key] for v in r6_vs_r8["regions"].values())
        for key in ("r6_success", "r8_success", "both_success", "gained_success",
                    "lost_success", "action_flips_on_both_success")
    }
    r6_reason_total: dict[str, int] = {}
    r8_reason_total: dict[str, int] = {}
    for v in r6_vs_r8["regions"].values():
        for k, c in v["r6_failure_reasons"].items():
            r6_reason_total[k] = r6_reason_total.get(k, 0) + c
        for k, c in v["r8_failure_reasons"].items():
            r8_reason_total[k] = r8_reason_total.get(k, 0) + c
    r6_vs_r8["totals"] = {**totals, "r6_failure_reasons": r6_reason_total,
                          "r8_failure_reasons": r8_reason_total}
    audit["r6_vs_r8_D"] = r6_vs_r8

    out = r8 / "results" / "audit_independent.json"
    out.write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"audit": str(out), "max_abs_diff": max_diff,
                      "r6_vs_r8_totals": r6_vs_r8["totals"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
