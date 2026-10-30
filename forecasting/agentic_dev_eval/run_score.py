"""Scorer and report writer: per-arm metrics, increments on shared cells, identity checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import data, metrics, protocol

ARMS = ("A", "B", "C", "D")
SPLITS = ("H1", "H2", "FULL")
INCREMENT_PAIRS = (("D", "C"), ("D", "B"), ("C", "B"))


def _load_arm_npz(output_dir: str, region: str, arm: str) -> dict:
    sub = "base" if arm == "A" else "predictions"
    stem = f"{region}_base" if arm == "A" else f"{region}_{arm}"
    path = Path(output_dir) / sub / f"{stem}.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["_sha256"] = data.sha256_file(path)
    payload["_path"] = str(path)
    return payload


def _split_mask(half: np.ndarray, split: str) -> np.ndarray:
    if split == "FULL":
        return np.ones(len(half), dtype=bool)
    return np.asarray(half) == split


def score_region(output_dir: str, region: str) -> dict:
    base = _load_arm_npz(output_dir, region, "A")
    arms_data = {"A": {**base, "exit_codes": np.zeros(len(base["origin_unix"]), dtype=np.int64)}}
    for arm in ("B", "C", "D"):
        payload = _load_arm_npz(output_dir, region, arm)
        for key, reference in (("origin_unix", base), ("target_time_unix", base),
                               ("actual", base)):
            if not np.array_equal(payload[key], reference[key]):
                raise RuntimeError(f"identity mismatch {region} {arm}: {key}")
        arms_data[arm] = payload

    region_result = {"region": region, "identity": {
        "csv_sha256": json.loads(str(base["metadata_json"]))["grid"]["csv_sha256"],
        "origin_unix_first": int(base["origin_unix"][0]),
        "origin_unix_last": int(base["origin_unix"][-1]),
        "n_origins": int(len(base["origin_unix"])),
        "npz_sha256": {arm: arms_data[arm]["_sha256"] for arm in ARMS},
    }}

    per_arm, increments = {}, {}
    for split in SPLITS:
        mask = _split_mask(base["half"], split)
        split_scores = {}
        for arm in ARMS:
            payload = arms_data[arm]
            split_scores[arm] = metrics.arm_metrics_with_failures(
                payload["quantiles"], payload["actual"], payload["exit_codes"], mask)
        per_arm[split] = split_scores
        pair_scores = {}
        for left, right in INCREMENT_PAIRS:
            shared = mask & (arms_data[left]["exit_codes"] == 0) & (arms_data[right]["exit_codes"] == 0)
            if shared.any():
                left_score = metrics.score_arrays(arms_data[left]["quantiles"][shared],
                                                  base["actual"][shared])
                right_score = metrics.score_arrays(arms_data[right]["quantiles"][shared],
                                                   base["actual"][shared])
                pair_scores[f"{left}-{right}"] = {
                    "n_shared_origins": int(shared.sum()),
                    "mae_delta": left_score["mae"] - right_score["mae"],
                    "quantile_score_delta": left_score["quantile_score"] - right_score["quantile_score"],
                    "coverage_80_delta": left_score["coverage_80"] - right_score["coverage_80"],
                    "coverage_90_delta": left_score["coverage_90"] - right_score["coverage_90"],
                }
            else:
                pair_scores[f"{left}-{right}"] = {"n_shared_origins": 0}
        increments[split] = pair_scores
    region_result["per_arm"] = per_arm
    region_result["increments"] = increments

    # Failure reasons, latency and token usage from decision records.
    usage = {}
    for arm in ("B", "C", "D"):
        jsonl = Path(output_dir) / "decisions" / f"{region}_{arm}.jsonl"
        rows = [json.loads(line) for line in jsonl.read_text().splitlines()] if jsonl.exists() else []
        reasons: dict[str, int] = {}
        for row in rows:
            if row["exit_code"] != 0:
                reasons[row["failure_reason"]] = reasons.get(row["failure_reason"], 0) + 1
        latencies = [row["latency_ms"] for row in rows] or [0.0]
        usage[arm] = {
            "n": len(rows),
            "failure_reasons": reasons,
            "latency_ms_mean": float(np.mean(latencies)),
            "latency_ms_max": float(np.max(latencies)),
            "llm_calls_total": int(sum(r["llm_calls"] for r in rows)),
            "tool_calls_total": int(sum(r["tool_calls"] for r in rows)),
            "prompt_tokens_total": int(sum(r["tokens"]["prompt"] for r in rows)),
            "completion_tokens_total": int(sum(r["tokens"]["completion"] for r in rows)),
            "cost_aud": 0.0,
        }
    region_result["usage"] = usage
    return region_result


def macro_average(region_results: list[dict]) -> dict:
    macro = {"per_arm": {}, "increments": {}}
    for split in SPLITS:
        arm_view = {}
        for arm in ARMS:
            cells = [r["per_arm"][split][arm] for r in region_results]
            view = {"n_success": int(sum(c["n_success"] for c in cells)),
                    "n_fail": int(sum(c["n_fail"] for c in cells))}
            scored = [c for c in cells if c.get("n_cells")]
            if scored:
                for key in ("mae", "quantile_score", "coverage_80", "coverage_90",
                            "width_80", "width_90"):
                    view[key] = float(np.mean([c[key] for c in scored]))
            arm_view[arm] = view
        macro["per_arm"][split] = arm_view
        inc_view = {}
        for left, right in INCREMENT_PAIRS:
            key = f"{left}-{right}"
            values = [r["increments"][split][key] for r in region_results
                      if r["increments"][split].get(key, {}).get("n_shared_origins", 0) > 0]
            if values:
                inc_view[key] = {
                    "n_shared_origins": int(sum(v["n_shared_origins"] for v in values)),
                    "mae_delta_macro": float(np.mean([v["mae_delta"] for v in values])),
                    "quantile_score_delta_macro": float(np.mean([v["quantile_score_delta"] for v in values])),
                }
            else:
                inc_view[key] = {"n_shared_origins": 0}
        macro["increments"][split] = inc_view
    return macro


MACRO_METRIC_KEYS = ("mae", "quantile_score", "coverage_80", "coverage_90")


def write_report(summary: dict, path: Path) -> None:
    lines = [
        "# r6 真实 2022 三地区开发集评估（A/B/C/D）", "",
        f"- 轮次：`{summary['round_id']}`；分支：`{summary['branch']}`",
        f"- 地区：NSW1 / QLD1 / TAS1；起报网格：2022 年每 7 天 00:00 AEST（UTC+10，与数据时间轴一致），共 "
        f"{summary['regions'][0]['identity']['n_origins']} 个起报/地区；72h 历史 → 24h 预测；"
        "分位数 0.05/0.10/0.50/0.90/0.95（q50 为点预测）；seed 2026。",
        "- A=冻结数值流（Chronos-2 zero-shot，CPU，离线，权重指纹固定）；B=固定规则；"
        "C=固定单次 LLM（qwen3-30b3a，temperature 0）；D=白名单有界 Agent（≤3 轮、≤4 次工具调用）。",
        f"- 成功/失败：{json.dumps({arm: [r['usage'][arm]['n'] - sum(r['usage'][arm]['failure_reasons'].values()) for r in summary['regions']] for arm in ('B','C','D')}, ensure_ascii=False)}（每地区）",
        "", "## 全年（FULL）宏观指标", "",
        "| Arm | MAE | QS | 覆盖80% | 覆盖90% |", "|---|---|---|---|---|",
    ]
    full = summary["macro"]["per_arm"]["FULL"]
    for arm in ARMS:
        row = full[arm]
        if "mae" in row:
            lines.append(f"| {arm} | {row['mae']:.2f} | {row['quantile_score']:.2f} | "
                         f"{row['coverage_80']*100:.1f}% | {row['coverage_90']*100:.1f}% |")
        else:
            lines.append(f"| {arm} | 无成功单元 | - | - | - |")
    lines += ["", "## 增量（负值=左侧 Arm 更优）", "",
              "| 对比 | MAE Δ(宏观) | QS Δ(宏观) |", "|---|---|---|"]
    for key, value in summary["macro"]["increments"]["FULL"].items():
        if value.get("n_shared_origins", 0) > 0:
            lines.append(f"| {key} | {value['mae_delta_macro']:+.3f} | "
                         f"{value['quantile_score_delta_macro']:+.3f} |")
        else:
            lines.append(f"| {key} | 无共享成功单元 | - |")
    lines += ["", "## 分地区全年 MAE / QS", "",
              "| 地区 | A | B | C | D |", "|---|---|---|---|---|"]
    for region in summary["regions"]:
        cells = []
        for arm in ARMS:
            row = region["per_arm"]["FULL"][arm]
            cells.append(f"{row['mae']:.2f}/{row['quantile_score']:.2f}" if "mae" in row else "失败")
        lines.append(f"| {region['region']} | " + " | ".join(cells) + " |")
    lines += ["", "## 失败、延迟与用量", ""]
    for region in summary["regions"]:
        for arm in ("B", "C", "D"):
            u = region["usage"][arm]
            lines.append(
                f"- {region['region']}/{arm}: 失败原因 {json.dumps(u['failure_reasons'], ensure_ascii=False)}；"
                f"平均延迟 {u['latency_ms_mean']:.0f}ms；LLM 调用 {u['llm_calls_total']}；"
                f"工具调用 {u['tool_calls_total']}；tokens {u['prompt_tokens_total']}+{u['completion_tokens_total']}；"
                f"成本 {u['cost_aud']} AUD（本地端点）")
    lines += ["", "## H1 / H2", ""]
    for split in ("H1", "H2"):
        rows = []
        for region in summary["regions"]:
            for arm in ARMS:
                row = region["per_arm"][split][arm]
                if "mae" in row:
                    rows.append(f"{region['region']}/{arm}={row['mae']:.2f}")
        lines.append(f"- {split}: " + ", ".join(rows))
    lines += [
        "", "## 身份与守卫", "",
        f"- 标签 CSV SHA-256（NSW1）：`{summary['regions'][0]['identity']['csv_sha256']}`；"
        "每个 npz 的 SHA-256 见 results/summary.json。",
        "- 严格 as-of：历史/工具仅访问 available_at ≤ origin 的行；标签窗口 24 行必须全部属于 2022 "
        "validation split（2023–2024 test 行被硬性排除）。",
        "- 结果身份：四个 Arm 的 origin/target/label 数组逐元素一致；增量仅在双方成功的共享单元上计算。",
        "- 真实退出码：每次 LLM/工具调用失败按原因记录且不得计为成功；进程退出码真实透传。",
        "- 本轮无任何训练、拟合、调参、模型下载或 GPU 使用。",
        "",
    ]
    path.write_text("\n".join(lines))


def run(output_dir: str, regions: tuple[str, ...] = protocol.REGIONS) -> int:
    region_results = [score_region(output_dir, region) for region in regions]
    summary = {
        "round_id": "agentic-m2m3-dev-eval-c2-20261030-r6",
        "branch": "research/agentic-m2m3-dev-eval-c2-20261030-r6",
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
            },
        },
        "regions": region_results,
        "macro": macro_average(region_results),
    }
    out = Path(output_dir) / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_report(summary, out / "summary_cn.md")

    # Independent self-check: re-derive every region's FULL-split A-arm MAE from npz.
    for region_result in region_results:
        payload = _load_arm_npz(output_dir, region_result["region"], "A")
        recomputed = metrics.score_arrays(payload["quantiles"], payload["actual"])
        if abs(recomputed["mae"] - region_result["per_arm"]["FULL"]["A"]["mae"]) > 1e-12:
            raise RuntimeError(f"self-check failed for {region_result['region']} arm A")
    print(json.dumps({"status": "ok", "results": str(out / "summary.json"),
                      "macro_full": summary["macro"]["per_arm"]["FULL"]}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",
                        default="outputs/forecasting/agentic_m2m3_dev_c2_20261030_r6")
    parser.add_argument("--regions", default=",".join(protocol.REGIONS),
                        help="comma-separated subset, e.g. NSW1 for smoke")
    args = parser.parse_args(argv)
    regions = tuple(r for r in args.regions.split(",") if r)
    return run(args.output_dir, regions)


if __name__ == "__main__":
    sys.exit(main())
