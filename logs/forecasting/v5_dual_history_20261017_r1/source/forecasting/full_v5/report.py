"""Report rendering from persisted artefacts; never writes model configs."""
from __future__ import annotations

import numpy as np

from . import data, schema

# Rendered only when a final evaluation actually ran this round.
POLLUTION_DECLARATION_FINAL_RAN = (
    "2023-2024 已在此前研究中反复查看。本次在开发规则和模型冻结后仅进行一次新增最终评价，"
    "但该评价仍为已污染时期的非盲回顾性评价。因果边界检查依赖现有档案时间代理，"
    "未独立认证真实首次发布时间。显著性检验不恢复盲测资格。"
)
# Rendered when no final evaluation ran (e.g. development gates failed and the
# protocol stopped before freeze): the report must never imply one was run.
POLLUTION_DECLARATION_FINAL_NOT_RUN = (
    "2023-2024 已在此前研究中反复查看。本次未运行任何新增最终评价（开发门禁未全部通过，"
    "按协议在 freeze 之前停止），本报告不含任何新的 2023-2024 回顾性评价结果。"
    "既有污染仍然存在；显著性检验不恢复盲测资格。"
)

# Round-6 L1: equivalence claims are conditional on an explicit numeric
# threshold (score units, AUD/MWh); raw max abs differences are always shown.
EQUIVALENCE_TOLERANCE = 1e-9

MODEL_ORDER = (schema.BASELINE,) + schema.CALIBRATOR_CONTROLS + schema.TREE_CONTROLS + (schema.STACKER,)


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value != 0.0 and abs(value) < 1e-4:
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value)


def _model_table(scored_by_region: dict, segment: str, metric_key: str, metric_name: str) -> str:
    regions = sorted(scored_by_region)
    lines = ["| model | " + " | ".join(regions) + " | macro |", "|---|" + "---:|" * (len(regions) + 1)]
    macro_values = {}
    for model_id in MODEL_ORDER:
        if model_id not in scored_by_region[regions[0]][segment]:
            continue
        cells = []
        for region in regions:
            entry = scored_by_region[region][segment][model_id]
            value = (entry["scores"]["mae"] if metric_key == "mae"
                     else entry["scores"]["quantile_score_proxy"] if metric_key == "qs"
                     else entry["rmse_region"])
            cells.append(_fmt(value))
            macro_values.setdefault(model_id, []).append(value)
        macro = float(np.mean(macro_values[model_id]))
        lines.append(f"| {model_id} | " + " | ".join(cells) + f" | **{_fmt(macro)}** |")
    return "\n".join(lines)


def _equivalence_lines(extras: dict) -> list[str]:
    """Conditional equivalence between V5 and its history controls.

    Round-6 L1: the equivalence wording is conditional on an explicit numeric
    threshold — only a max abs difference within EQUIVALENCE_TOLERANCE may be
    called numerically equivalent (never bitwise identical); anything larger
    is reported as a substantive difference.
    """
    lines = []
    for region, extra in sorted(extras.items()):
        diffs = extra.get("v5_vs_controls_max_abs_diff")
        if not diffs:
            continue
        parts = []
        for control in sorted(diffs):
            value = float(diffs[control])
            verdict = ("数值精度内等价" if value <= EQUIVALENCE_TOLERANCE
                       else "超过等价阈值，不声明等价（存在实质差异）")
            parts.append(f"{control} max|Δ| {_fmt(value)} [{verdict}]")
        lines.append(f"- {region}: V5_history vs controls — " + "; ".join(parts)
                     + f"；等价判定阈值 max|Δ| ≤ {EQUIVALENCE_TOLERANCE:g}（数值精度内等价，非逐位相同声明）")
    return lines


def render_development_report(paths, scored_by_region, extras, gates, rows_info) -> None:
    lines = ["# V5 Calibrated History — development report", ""]
    lines.append(f"Generated: {data.read_json(paths['state']).get('updated_at', '')}")
    lines.append("")
    lines.append("## Gate decision")
    lines.append("")
    g1, g2 = gates["G1"], gates["G2"]
    lines.append(f"- G0 identity/tests: **{'PASS' if gates['G0']['passed'] else 'FAIL'}**")
    lines.append(f"- G1 dev utility vs B: **{'PASS' if g1['passed'] else 'FAIL'}** "
                 f"(macro MAE improvement {_fmt(g1['details']['mae']['improvement']*100, 3)}%, "
                 f"macro QS improvement {_fmt(g1['details']['qs']['improvement']*100, 3)}%; "
                 f"thresholds {g1['thresholds']})")
    for region, value in g1["details"]["per_region"].items():
        lines.append(f"  - {region} MAE improvement {_fmt(value['improvement']*100, 3)}% "
                     f"(ref {_fmt(value['reference'])}, V5 {_fmt(value['candidate'])})")
    for region, value in g1["details"]["per_region_band"].items():
        lines.append(f"  - {region} 0-300 band MAE improvement {_fmt(value['improvement']*100, 3)}%")
    for control, entry in g2["details"].items():
        lines.append(f"- G2 vs {control}: **{'PASS' if entry['passed'] else 'FAIL'}** "
                     f"(macro MAE {_fmt(entry['mae']['improvement']*100, 3)}%, "
                     f"macro QS {_fmt(entry['qs']['improvement']*100, 3)}%, "
                     f"V1/V2 direction stable: {entry['direction_ok_all_segments']} "
                     f"(direction tolerance {g2['thresholds'].get('direction_tolerance', 'n/a')})")
        for segment, seg in entry["per_segment_direction"].items():
            lines.append(f"  - {segment}: MAE diff(ref-V5) {_fmt(seg['mae']['diff_ref_minus_cand'])}, "
                         f"QS diff {_fmt(seg['qs']['diff_ref_minus_cand'])}")
    lines.append(f"- overall: **{'PASS' if gates['passed'] else 'FAIL'}**")
    lines.append("")
    equivalence = _equivalence_lines(extras)
    if equivalence:
        lines.append("## V5 vs history controls — prediction equivalence")
        lines.append("")
        lines.extend(equivalence)
        lines.append("")
    for segment in ("dfit_fit", "v1", "v2", "v1+v2"):
        if segment not in next(iter(scored_by_region.values())):
            continue
        lines.append(f"## {segment} — MAE (AUD/MWh)")
        lines.append("")
        lines.append(_model_table(scored_by_region, segment, "mae", "MAE"))
        lines.append("")
        lines.append(f"## {segment} — QS (2x mean five-quantile pinball)")
        lines.append("")
        lines.append(_model_table(scored_by_region, segment, "qs", "QS"))
        lines.append("")
    lines.append("## Diagnostics (V1+V2)")
    lines.append("")
    for region, extra in extras.items():
        audit = extra["library"]["audit"]
        lines.append(f"### {region}")
        lines.append(f"- library records: {extra['library']['records']}; "
                     f"history-branch open fraction {_fmt(audit['v_open_fraction'], 3)}; "
                     f"closed by candidates {_fmt(audit['v_closed_candidate_fraction'], 3)}; "
                     f"closed by 28d stats {_fmt(audit['v_closed_stats_fraction'], 3)}")
        lines.append(f"- m-window fallback (7d/28d): {_fmt(audit['m_short_fallback_fraction'], 3)} / "
                     f"{_fmt(audit['m_long_fallback_fraction'], 3)}")
        parts = extra.get("v5_parts", {})
        lines.append(f"- V5 delta mean {_fmt(parts.get('delta_mean'))} (|delta| max "
                     f"{_fmt(parts.get('delta_abs_max'))}, at-bound {_fmt(parts.get('delta_at_bound_fraction'), 3)}); "
                     f"s mean {_fmt(parts.get('s_mean'))} in [{_fmt(parts.get('s_min'))}, "
                     f"{_fmt(parts.get('s_max'))}]")
        distributions = extra.get("feature_distributions")
        if distributions and "z" in distributions and isinstance(distributions["z"], dict):
            segments_list = sorted(distributions["z"])
            lines.append(f"- feature-distribution summaries saved per group (z/m/v) x segment "
                         f"({', '.join(segments_list)}) in development/metrics.json")
        oracle = extra.get("oracle_v1v2")
        if oracle:
            lines.append(f"- in-segment stacker oracle (descriptive upper bound only): "
                         f"MAE {_fmt(oracle['in_segment_mae'])}, weights {oracle['weights']}")
        lines.append("")
    unfinished = data.read_json(paths["unfinished"]) if paths["unfinished"].exists() else []
    if unfinished:
        lines.append("## Unfinished items")
        lines.append("")
        for item in unfinished:
            lines.append(f"- {item}")
        lines.append("")
    paths["reports"].mkdir(parents=True, exist_ok=True)
    (paths["reports"] / "dev_report.md").write_text("\n".join(lines) + "\n")


def render_final_report(paths) -> None:
    development = data.read_json(paths["development"] / "metrics.json")
    gates = data.read_json(paths["development"] / "gates.json")
    scored = development["scored"]
    extras = development.get("extras", {})
    lines = ["# V5 Calibrated History — final report", ""]
    lines.append(f"Generated from persisted artefacts; gates evaluated at dev time.")
    lines.append("")
    final_metrics = paths["final"] / "metrics.json"
    final_ran = final_metrics.exists()
    lines.append("## Retrospective-evaluation pollution declaration")
    lines.append("")
    declaration = POLLUTION_DECLARATION_FINAL_RAN if final_ran else POLLUTION_DECLARATION_FINAL_NOT_RUN
    lines.append(f"> {declaration}")
    lines.append("")
    lines.append("## Development gate outcome")
    lines.append("")
    lines.append(f"- G0/G1/G2 at dev time: **{'PASS' if gates['passed'] else 'FAIL'}** "
                 "(see development/gates.json for every number)")
    lines.append("")
    equivalence = _equivalence_lines(extras)
    if equivalence:
        lines.append("## V5 vs history controls — prediction equivalence (development)")
        lines.append("")
        lines.extend(equivalence)
        lines.append("")
    if final_ran:
        final = data.read_json(final_metrics)
        lines.append("## Final frozen-matrix evaluation (2023-2024, single run)")
        lines.append("")
        lines.append("### MAE (AUD/MWh)")
        lines.append("")
        lines.append(_model_table(final["scored"], "test", "mae", "MAE"))
        lines.append("")
        lines.append("### QS")
        lines.append("")
        lines.append(_model_table(final["scored"], "test", "qs", "QS"))
        lines.append("")
        lines.append("### Regional RMSE and pooled RMSE (named separately)")
        lines.append("")
        lines.append(_model_table(final["scored"], "test", "rmse_region", "RMSE"))
        pooled = _pooled_table(final)
        lines.extend(pooled)
        lines.append("")
        lines.append("### Pre-declared primary comparisons (168h joint paired bootstrap, 9999 draws)")
        lines.append("")
        labels = final["holm_adjusted_p"]["labels"]
        values = final["holm_adjusted_p"]["values"]
        for label, adjusted in zip(labels, values):
            name = label.rsplit(":", 1)
            comparison, metric = name[0], name[1]
            boot = final["bootstrap"][comparison]
            diff = boot["point_difference"][metric]
            ci = boot["ci95"][metric]
            raw_p = boot["p_value"][metric]
            lines.append(f"- {comparison} — {metric.upper()}: diff(ref-V5) {_fmt(diff)} AUD/MWh "
                         f"(positive = V5 lower loss), "
                         f"95% CI [{_fmt(ci[0])}, {_fmt(ci[1])}], raw p {_fmt(raw_p, 4)}, "
                         f"Holm-adjusted p {_fmt(adjusted, 4)} (24h-block sensitivity in metrics.json)")
        lines.append("")
        lines.append("### Interpretation rules (pre-declared)")
        lines.append("")
        lines.append("- C_roll reaching the same effect: no evidence of a state-matched history increment.")
        lines.append("- C_recent reaching the same effect: no evidence that matching beats recency/calendar history.")
        lines.append("- T_history gaining similarly: supports the new history information, not the specialised structure.")
        lines.append("- A_stack gaining: supports artefact complementarity, not the history mechanism.")
        lines.append("- A single region improving cannot replace the three-region judgement.")
        lines.append("")
    else:
        lines.append("## Final evaluation not run")
        lines.append("")
        state = data.read_json(paths["state"])
        lines.append(f"State at report time: {state.get('stage')} — dev gates "
                     f"{'passed' if gates.get('passed') else 'failed'}; "
                     "final-test is only permitted after passing gates and freeze. "
                     "本次没有进行 2023-2024 最终评价，也未生成任何最终评价结果。")
        lines.append("")
    unfinished = data.read_json(paths["unfinished"]) if paths["unfinished"].exists() else []
    if unfinished:
        lines.append("## Unfinished items")
        lines.append("")
        for item in unfinished:
            lines.append(f"- {item}")
        lines.append("")
    if paths["budget"].exists():
        budget = data.read_json(paths["budget"])
        lines.append("## Budget usage")
        lines.append("")
        limits = {"dev": "dev_core_hours", "final": "final_core_hours"}
        for pool, entry in budget.items():
            if isinstance(entry, dict) and "cpu_seconds" in entry:
                limit = limits.get(pool)
                note = f" of {limit}" if limit else ""
                lines.append(f"- {pool}: {entry['cpu_seconds']/3600:.3f} CPU core-hours{note}, "
                             f"{entry['wall_seconds']/3600:.3f} wall-hours")
        regions = budget.get("dev_regions") or {}
        if regions:
            parts = [f"{region} {float(entry['cpu_seconds'])/3600:.3f}"
                     for region, entry in sorted(regions.items())]
            lines.append(f"- dev region shards: " + ", ".join(parts) + " CPU core-hours (per-region records)")
        total = budget.get("dev_total_cpu_seconds")
        if total is not None:
            lines.append(f"- dev total (root + region shards): {float(total)/3600:.3f} CPU core-hours "
                         "of dev_core_hours (parallelism shortens wall time, never total CPU budget)")
        lines.append("")
    paths["reports"].mkdir(parents=True, exist_ok=True)
    (paths["reports"] / "final_report.md").write_text("\n".join(lines) + "\n")
    summary = dict(gates_passed=bool(gates.get("passed")), has_final=bool(final_ran))
    data.atomic_json(paths["reports"] / "summary.json", summary)


def _pooled_table(final: dict) -> list[str]:
    regions = sorted(final["scored"])
    lines = ["| model | pooled RMSE | macro RMSE |", "|---|---:|---:|"]
    for model_id in MODEL_ORDER:
        if model_id not in final["scored"][regions[0]]["test"]:
            continue
        sq, n, region_rmses = 0.0, 0, []
        for region in regions:
            entry = final["scored"][region]["test"][model_id]
            region_rmses.append(entry["rmse_region"])
            n_cells = entry["scores"]["n_cells"]
            sq += (entry["rmse_region"] ** 2) * n_cells
            n += n_cells
        pooled = (sq / n) ** 0.5
        lines.append(f"| {model_id} | {_fmt(pooled)} | {_fmt(float(np.mean(region_rmses)))} |")
    return lines
