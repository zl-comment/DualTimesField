"""Read-only decomposition of the Full-minus-XGBoost gap in the frozen exploration run.

Inputs (never modified):
  * logs/forecasting/exploration_20261011/reports/report.json.gz  (static protocol)
  * outputs/forecasting/<run>/<region>/training_history.csv       (earlier tracked runs)

The per-origin prediction files of the frozen run (``test.npz``) live only on the
machine that trained them and are not in this repository, so every number here is
recomputed from the committed report summaries. No numpy/pandas/torch is required
(standard library only). Outputs go to logs/forecasting/full_diagnosis_20261012/.

Usage (from the repository root):
  python3 -m forecasting.analysis.full_vs_xgboost_gap
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import json
import statistics
from pathlib import Path

REGIONS = ("NSW1", "QLD1", "TAS1")
METRICS = ("mae", "crps_quantile_approx")
BANDS = ("negative", "0_to_100", "100_to_300", "spike")
DEFAULT_REPORT = "logs/forecasting/exploration_20261011/reports/report.json.gz"
DEFAULT_OUT = "logs/forecasting/full_diagnosis_20261012"
HISTORY_RUNS = ("pdpasa_dgf", "pdpasa_dgf_seed2027", "pdpasa_dgf_seed2028",
                "gas_price_ctf", "gas_price_ctf_seed2027", "gas_price_ctf_seed2028",
                "residual_skip_path", "residual_skip_path_seed2027", "residual_skip_path_seed2028")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean(x):
    return sum(x) / len(x)


def _m(cell):
    return cell["mean"] if isinstance(cell, dict) else cell


def slice_table(full, xgb, key, names, metric):
    """Contribution of each slice to (Full - XGBoost) in one region, weighted by observation share."""
    total_n = _m(full["overall"]["n_observations"])
    rows = {}
    for name in names:
        f, x = full[key][name], xgb[key][name]
        w = _m(f["n_observations"]) / total_n
        rows[name] = {"weight": w, "full": _m(f[metric]), "xgboost": _m(x[metric]),
                      "slice_diff": _m(f[metric]) - _m(x[metric]),
                      "contribution": w * (_m(f[metric]) - _m(x[metric]))}
    return rows


def analyse(report):
    static = report["protocols"]["static"]["summaries"]
    full = static["full"]["region_seed_summary"]
    xgb = static["xgboost"]["region_seed_summary"]
    out = {"regions": {}, "macro": {}, "checks": {}}
    for metric in METRICS:
        diffs = {r: _m(full[r]["overall"][metric]) - _m(xgb[r]["overall"][metric]) for r in REGIONS}
        macro = mean(list(diffs.values()))
        out["macro"][metric] = {
            "full": mean([_m(full[r]["overall"][metric]) for r in REGIONS]),
            "xgboost": mean([_m(xgb[r]["overall"][metric]) for r in REGIONS]),
            "gap": macro,
            "region_gap": diffs,
            "region_share_of_macro_gap": {r: diffs[r] / 3 / macro for r in REGIONS},
        }
    # price-band / horizon / month contributions
    months = list(full["NSW1"]["months_aest"])
    horizons = [str(h) for h in range(1, 25)]
    for r in REGIONS:
        out["regions"][r] = {}
        for metric in METRICS:
            reg_gap = _m(full[r]["overall"][metric]) - _m(xgb[r]["overall"][metric])
            bands = slice_table(full[r], xgb[r], "price_bands", BANDS, metric)
            hz = slice_table(full[r], xgb[r], "horizons", horizons, metric)
            mo = slice_table(full[r], xgb[r], "months_aest", months, metric)
            for label, rows in (("price_bands", bands), ("horizons", hz), ("months", mo)):
                total = sum(v["contribution"] for v in rows.values())
                # price bands partition observations: contributions must sum to the regional gap
                out["checks"][f"{r}/{metric}/{label}_sum_minus_gap"] = total - reg_gap
            out["regions"][r][metric] = {"gap": reg_gap, "price_bands": bands, "horizons": hz, "months": mo}
    # tail/calibration descriptors (seed means)
    desc = ("coverage_90", "interval_width_90", "ais_90", "crps_negative", "crps_spike",
            "pinball_q05_negative_hours", "pinball_q95_spike_hours", "rmse_window_mean")
    out["calibration_tails"] = {r: {d: {"full": _m(full[r]["overall"][d]), "xgboost": _m(xgb[r]["overall"][d])}
                                    for d in desc} for r in REGIONS}
    # seed dispersion
    seeds = {}
    for name in ("full", "xgboost"):
        per = static[name]["macro_region_per_seed"]
        seeds[name] = {s: {m: per[s][m] for m in METRICS} for s in per}
    out["macro_per_seed"] = seeds
    out["region_per_seed_mae"] = {
        name: {r: {s: static[name]["per_seed"][r][s]["overall"]["mae"] for s in static[name]["per_seed"][r]}
               for r in REGIONS} for name in ("full", "xgboost")}
    # validation vs test (unclipped candidate = first entry of the floor search)
    fc = report["floor_choices"]
    val = {}
    for r in REGIONS:
        row = {}
        for metric, key in (("mae", "validation_mae"), ("crps_quantile_approx", "validation_crps")):
            f = [fc[f"static/full/{s}/{r}"][key][0] for s in (2026, 2027, 2028)]
            x = [fc[f"static/xgboost/{s}/{r}"][key][0] for s in (2026, 2027, 2028)]
            row[metric] = {"full_seed_mean": mean(f), "xgboost_seed_mean": mean(x), "gap_validation": mean(f) - mean(x),
                           "gap_test": out["regions"][r][metric]["gap"], "full_per_seed": f, "xgboost_per_seed": x}
        val[r] = row
    out["validation_vs_test"] = val
    # ranked slices
    ranked = []
    for r in REGIONS:
        for metric in METRICS:
            for label in ("price_bands", "months", "horizons"):
                for name, v in out["regions"][r][metric][label].items():
                    ranked.append({"metric": metric, "region": r, "family": label, "slice": name,
                                   "macro_contribution": v["contribution"] / 3,
                                   "share_of_macro_gap": v["contribution"] / 3 / out["macro"][metric]["gap"]})
    out["ranked_slices"] = ranked
    # macro contribution by band
    out["macro_band_contribution"] = {
        m: {b: mean([out["regions"][r][m]["price_bands"][b]["contribution"] for r in REGIONS]) for b in BANDS}
        for m in METRICS}
    return out


def history_analysis(root: Path):
    rows = []
    for run in HISTORY_RUNS:
        for region in REGIONS:
            path = root / "outputs/forecasting" / run / region / "training_history.csv"
            if not path.exists():
                continue
            h = list(csv.DictReader(open(path)))
            f = lambda c: [float(r[c]) for r in h]
            vmae, vtot = f("validation_mae_aud_per_mwh"), f("validation_total")
            tail = vmae[3:]
            last = h[-1]
            comp = {c: float(last[f"train_{c}"]) for c in ("point", "quantile", "decomposition")}
            comp["smoothness_w"] = 0.001 * float(last["train_smoothness"])
            comp["sparsity_w"] = 0.001 * float(last["train_sparsity"])
            s = sum(comp.values())
            rows.append({
                "run": run, "region": region, "epochs": len(h),
                "best_epoch_val_mae": vmae.index(min(vmae)) + 1, "best_epoch_val_total": vtot.index(min(vtot)) + 1,
                "val_mae_min": min(vmae), "val_mae_max": max(vmae), "val_mae_last": vmae[-1],
                "val_mae_sd_epochs4plus": statistics.pstdev(tail), "val_mae_first": vmae[0],
                "val_mae_mean_abs_epoch_to_epoch_change": mean([abs(a - b) for a, b in zip(tail[1:], tail[:-1])]),
                "train_point_first_to_last": [float(h[0]["train_point"]), float(last["train_point"])],
                "final_train_loss_share": {k: v / s for k, v in comp.items()},
                "val_total_best_over_val_mae_best": min(vtot) / vtot[vmae.index(min(vmae))],
            })
    return rows


def extras(report, result):
    """Protocol gaps, variant-vs-XGBoost band tables, month breadth, calibration by band, RMSE."""
    ex = {}
    # (a) macro gap per protocol
    ex["protocol_gaps"] = {}
    for proto, block in report["protocols"].items():
        s = block["summaries"]
        row = {}
        for m in METRICS + ("rmse_window_mean",):
            f = mean([_m(s["full"]["region_seed_summary"][r]["overall"][m]) for r in REGIONS])
            x = mean([_m(s["xgboost"]["region_seed_summary"][r]["overall"][m]) for r in REGIONS])
            row[m] = {"full": f, "xgboost": x, "gap": f - x}
        ex["protocol_gaps"][proto] = row
    # (b) each static variant minus XGBoost, macro band contributions
    s = report["protocols"]["static"]["summaries"]
    ex["variant_band_gap"] = {}
    for name in ("full", "no_tail", "raw_history", "no_fields", "no_pd", "linear", "dlinear", "gatets_adapter", "pd_calibrator"):
        entry = {"macro": {}, "bands": {}}
        for m in METRICS:
            entry["macro"][m] = mean([_m(s[name]["region_seed_summary"][r]["overall"][m]) -
                                      _m(s["xgboost"]["region_seed_summary"][r]["overall"][m]) for r in REGIONS])
            entry["bands"][m] = {b: mean([slice_table(s[name]["region_seed_summary"][r], s["xgboost"]["region_seed_summary"][r],
                                                       "price_bands", (b,), m)[b]["contribution"] for r in REGIONS]) for b in BANDS}
        ex["variant_band_gap"][name] = entry
    # (c) breadth: in how many region-months / region-horizons is Full worse?
    ex["breadth"] = {}
    for m in METRICS:
        for fam in ("months", "horizons"):
            vals = [v["slice_diff"] for r in REGIONS for v in result["regions"][r][m][fam].values()]
            ex["breadth"][f"{m}/{fam}"] = {"n": len(vals), "n_full_worse": sum(1 for v in vals if v > 0)}
    # (d) calibration/sharpness inside each realised-price band (descriptive; conditioned on the outcome)
    ex["band_calibration"] = {}
    for r in REGIONS:
        ex["band_calibration"][r] = {}
        for b in BANDS:
            f = s["full"]["region_seed_summary"][r]["price_bands"][b]
            x = s["xgboost"]["region_seed_summary"][r]["price_bands"][b]
            ex["band_calibration"][r][b] = {k: {"full": _m(f[k]), "xgboost": _m(x[k])}
                                            for k in ("coverage_90", "interval_width_90", "rmse_pooled")}
    # (e) RMSE (squared-error) comparison, region level
    ex["rmse"] = {r: {"full_rmse_window_mean": _m(s["full"]["region_seed_summary"][r]["overall"]["rmse_window_mean"]),
                      "xgboost_rmse_window_mean": _m(s["xgboost"]["region_seed_summary"][r]["overall"]["rmse_window_mean"]),
                      "full_rmse_pooled": _m(s["full"]["region_seed_summary"][r]["overall"]["rmse_pooled"]),
                      "xgboost_rmse_pooled": _m(s["xgboost"]["region_seed_summary"][r]["overall"]["rmse_pooled"])} for r in REGIONS}
    # (f) fraction of the linear -> XGBoost MAE gap recovered by Full
    lin = mean([_m(s["linear"]["region_seed_summary"][r]["overall"]["mae"]) for r in REGIONS])
    fu = mean([_m(s["full"]["region_seed_summary"][r]["overall"]["mae"]) for r in REGIONS])
    xg = mean([_m(s["xgboost"]["region_seed_summary"][r]["overall"]["mae"]) for r in REGIONS])
    ex["linear_to_xgboost"] = {"linear": lin, "full": fu, "xgboost": xg, "recovered_fraction": (lin - fu) / (lin - xg)}
    ex["crossing_rate_before_rearrangement"] = {
        r: {n: _m(s[n]["region_seed_summary"][r]["quantile_crossing_rate_before_rearrangement"]) for n in ("full", "xgboost")}
        for r in REGIONS}
    # (h) gap by calendar year and half-year of the test period (month rows are observation-weighted contributions)
    ex["by_year_half"] = {}
    for m in METRICS:
        acc = {}
        for r in REGIONS:
            for month, v in result["regions"][r][m]["months"].items():
                year, mon = month.split("-")
                key = f"{year}H{1 if int(mon) <= 6 else 2}"
                acc[key] = acc.get(key, 0.0) + v["contribution"] / 3
        ex["by_year_half"][m] = acc
    # (i) seed spread of Full per region, and how many Full seeds beat the (near-constant) XGBoost mean
    st = report["protocols"]["static"]["summaries"]
    spread = {}
    for r in REGIONS:
        fm = [st["full"]["per_seed"][r][k]["overall"]["mae"] for k in sorted(st["full"]["per_seed"][r])]
        xm = [st["xgboost"]["per_seed"][r][k]["overall"]["mae"] for k in sorted(st["xgboost"]["per_seed"][r])]
        spread[r] = {"full_min": min(fm), "full_max": max(fm), "xgb_min": min(xm), "xgb_max": max(xm),
                     "n_full_seeds_better_than_xgb_mean": sum(1 for v in fm if v < mean(xm))}
    ex["mae_seed_spread"] = spread
    # (g) paired bootstrap already in the report: copy macro rows
    ex["bootstrap_full_vs_xgboost"] = {}
    for blk in ("168", "24"):
        c = report["protocols"]["static"]["comparisons"][blk]["xgboost"]["metrics"]
        ex["bootstrap_full_vs_xgboost"][blk] = {m: {r: {k: c[m][r][k] for k in ("difference", "ci_low", "ci_high", "p_value_holm")}
                                                    for r in (*REGIONS, "macro_regions")} for m in METRICS}
    return ex


def history_summary(rows):
    early = [r for r in rows if r["best_epoch_val_mae"] <= 5]
    return {"n_runs": len(rows), "n_best_val_mae_epoch_le5": len(early),
            "median_best_val_mae_epoch": statistics.median(r["best_epoch_val_mae"] for r in rows),
            "n_best_epoch_differs_mae_vs_total": sum(1 for r in rows if r["best_epoch_val_mae"] != r["best_epoch_val_total"]),
            "mean_val_mae_last_minus_min": mean([r["val_mae_last"] - r["val_mae_min"] for r in rows]),
            "mean_val_mae_range": mean([r["val_mae_max"] - r["val_mae_min"] for r in rows])}


def selection_rule_sensitivity(root: Path):
    """Earlier tracked runs saved test MAE for the epoch chosen by validation MAE and by validation total loss."""
    diffs, differing, better = [], 0, 0
    for path in sorted(glob.glob(str(root / "outputs/forecasting/*/*/metrics.json"))):
        sr = json.load(open(path)).get("selection_results")
        if not sr or "mae" not in sr or "total" not in sr:
            continue
        a = sr["mae"]["test"]["metrics"]["mae_aud_per_mwh"]
        b = sr["total"]["test"]["metrics"]["mae_aud_per_mwh"]
        diffs.append(abs(a - b))
        differing += sr["mae"]["best_epoch"] != sr["total"]["best_epoch"]
        better += a < b
    return {"n_runs": len(diffs), "mean_abs_test_mae_difference": mean(diffs) if diffs else None,
            "max_abs_test_mae_difference": max(diffs) if diffs else None,
            "n_different_best_epoch": differing, "n_mae_rule_better_on_test": better}


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", default=DEFAULT_REPORT)
    p.add_argument("--root", default=".")
    p.add_argument("--out", default=DEFAULT_OUT)
    a = p.parse_args()
    root = Path(a.root).resolve()
    report_path = root / a.report
    report = json.load(gzip.open(report_path, "rt"))
    result = analyse(report)
    result["history_runs"] = history_analysis(root)
    result["extras"] = extras(report, result)
    result["history_summary"] = history_summary(result["history_runs"])
    result["selection_rule_sensitivity"] = selection_rule_sensitivity(root)
    result["input"] = {"report": a.report, "sha256": sha256(report_path)}
    out = root / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "gap_decomposition.json").write_text(json.dumps(result, indent=1))

    m = result["macro"]
    lines = ["# Full minus XGBoost: decomposition (static protocol, seed means)", "",
             f"input: {a.report}", f"sha256: {result['input']['sha256']}", ""]
    lines += ["## Macro and regional gaps (Full - XGBoost; positive = Full worse)", ""]
    rows = []
    for metric in METRICS:
        rows.append([metric, f"{m[metric]['full']:.3f}", f"{m[metric]['xgboost']:.3f}", f"{m[metric]['gap']:+.3f}"] +
                    [f"{m[metric]['region_gap'][r]:+.3f} ({m[metric]['region_share_of_macro_gap'][r]*100:.0f}%)" for r in REGIONS])
    lines.append(md_table(["metric", "Full", "XGBoost", "macro gap", *[f"{r} (share of macro)" for r in REGIONS]], rows))
    lines += ["", "## Price-band contributions to the regional gap (weight x slice gap)", ""]
    rows = []
    for metric in METRICS:
        for r in REGIONS:
            for b in BANDS:
                v = result["regions"][r][metric]["price_bands"][b]
                rows.append([metric, r, b, f"{v['weight']*100:.1f}%", f"{v['full']:.2f}", f"{v['xgboost']:.2f}",
                             f"{v['contribution']:+.3f}", f"{v['contribution']/3/m[metric]['gap']*100:+.0f}%"])
    lines.append(md_table(["metric", "region", "actual band", "obs share", "Full", "XGBoost", "contribution", "share of macro gap"], rows))
    lines += ["", "## Macro band contribution (mean over regions)", ""]
    lines.append(md_table(["metric", *BANDS, "sum"], [[mt, *[f"{result['macro_band_contribution'][mt][b]:+.3f}" for b in BANDS],
                                                       f"{sum(result['macro_band_contribution'][mt].values()):+.3f}"] for mt in METRICS]))
    lines += ["", "## Horizon blocks (macro contribution over regions, hours 1-6/7-12/13-18/19-24)", ""]
    rows = []
    for metric in METRICS:
        blocks = []
        for lo in (1, 7, 13, 19):
            blocks.append(mean([sum(result["regions"][r][metric]["horizons"][str(h)]["contribution"] for h in range(lo, lo + 6)) for r in REGIONS]))
        rows.append([metric, *[f"{b:+.3f}" for b in blocks]])
    lines.append(md_table(["metric", "1-6", "7-12", "13-18", "19-24"], rows))
    lines += ["", "## Per-horizon slice gap (Full-XGBoost, macro over regions)", ""]
    rows = []
    for h in range(1, 25):
        rows.append([h] + [f"{mean([result['regions'][r][mt]['horizons'][str(h)]['slice_diff'] for r in REGIONS]):+.3f}" for mt in METRICS])
    lines.append(md_table(["horizon", "MAE gap", "CRPS~ gap"], rows))
    lines += ["", "## Top slices by macro contribution (positive = Full worse)", ""]
    for metric in METRICS:
        for family in ("months", "price_bands"):
            sl = [s for s in result["ranked_slices"] if s["metric"] == metric and s["family"] == family]
            sl.sort(key=lambda s: -s["macro_contribution"])
            top = sl[:8]
            share_top = sum(s["share_of_macro_gap"] for s in top)
            lines += [f"### {metric} / {family}: top 8 positive slices cover {share_top*100:.0f}% of the macro gap; "
                      f"all positive slices sum to {sum(s['share_of_macro_gap'] for s in sl if s['macro_contribution']>0)*100:.0f}%", ""]
            lines.append(md_table(["region", "slice", "macro contribution", "share"],
                                  [[s["region"], s["slice"], f"{s['macro_contribution']:+.4f}", f"{s['share_of_macro_gap']*100:+.1f}%"] for s in top]))
            lines.append("")
    lines += ["## Validation (2022) vs test gap (Full-XGBoost, unclipped, seed means)", ""]
    rows = []
    for r in REGIONS:
        for metric in METRICS:
            v = result["validation_vs_test"][r][metric]
            rows.append([r, metric, f"{v['full_seed_mean']:.2f}", f"{v['xgboost_seed_mean']:.2f}", f"{v['gap_validation']:+.2f}", f"{v['gap_test']:+.2f}"])
    lines.append(md_table(["region", "metric", "Full val", "XGB val", "val gap", "test gap"], rows))
    lines += ["", "## Calibration / tail descriptors (seed means)", ""]
    rows = []
    for r in REGIONS:
        for d, v in result["calibration_tails"][r].items():
            rows.append([r, d, f"{v['full']:.3f}", f"{v['xgboost']:.3f}"])
    lines.append(md_table(["region", "metric", "Full", "XGBoost"], rows))
    lines += ["", "## Seed dispersion (macro MAE / CRPS~ per seed)", ""]
    rows = []
    for name in ("full", "xgboost"):
        for s, v in result["macro_per_seed"][name].items():
            rows.append([name, s, f"{v['mae']:.3f}", f"{v['crps_quantile_approx']:.3f}"])
    lines.append(md_table(["model", "seed", "MAE", "CRPS~"], rows))
    lines += ["", "## Per-region per-seed MAE", ""]
    rows = [[n, r, *[f"{v:.3f}" for v in result["region_per_seed_mae"][n][r].values()]] for n in ("full", "xgboost") for r in REGIONS]
    lines.append(md_table(["model", "region", "seed 2026", "seed 2027", "seed 2028"], rows))
    lines += ["", "## Earlier tracked runs of the same architecture family (not the frozen run): training curves", "",
              "Validation MAE per epoch is not monotone although training loss is; the epoch minimising validation MAE usually differs from the one minimising total loss.", ""]
    rows = []
    for h in result["history_runs"]:
        sh = h["final_train_loss_share"]
        rows.append([h["run"], h["region"], h["best_epoch_val_mae"], h["best_epoch_val_total"], f"{h['val_mae_min']:.1f}", f"{h['val_mae_max']:.1f}",
                     f"{h['val_mae_sd_epochs4plus']:.2f}", f"{h['val_mae_mean_abs_epoch_to_epoch_change']:.2f}",
                     f"{sh['point']*100:.0f}/{sh['quantile']*100:.0f}/{sh['decomposition']*100:.0f}"])
    lines.append(md_table(["run", "region", "best ep (val MAE)", "best ep (val total)", "val MAE min", "max", "SD ep>=4", "mean |d epoch|", "final loss share point/quantile/decomp %"], rows))
    ex = result["extras"]
    lines += ["", "## Macro Full-XGBoost gap by protocol (positive = Full worse)", ""]
    lines.append(md_table(["protocol", "MAE gap", "CRPS~ gap", "window-RMSE gap"],
                          [[p, *[f"{v[m]['gap']:+.3f}" for m in (*METRICS, 'rmse_window_mean')]] for p, v in ex["protocol_gaps"].items()]))
    lines += ["", "## Static variants minus XGBoost: macro gap and band contributions", ""]
    rows = []
    for name, e in ex["variant_band_gap"].items():
        for m in METRICS:
            rows.append([name, m, f"{e['macro'][m]:+.3f}", *[f"{e['bands'][m][b]:+.3f}" for b in BANDS]])
    lines.append(md_table(["variant", "metric", "macro gap", *BANDS], rows))
    lines += ["", "## Breadth: region-slices where Full is worse than XGBoost", ""]
    lines.append(md_table(["metric/family", "slices", "Full worse"], [[k, v["n"], v["n_full_worse"]] for k, v in ex["breadth"].items()]))
    lines += ["", "## Squared-error comparison (Full better on RMSE in NSW1 and QLD1)", ""]
    lines.append(md_table(["region", "Full window RMSE", "XGB window RMSE", "Full pooled RMSE", "XGB pooled RMSE"],
                          [[r, *[f"{v:.2f}" for v in e.values()]] for r, e in ex["rmse"].items()]))
    lines += ["", "## Coverage and width of the 90% interval within realised-price bands (descriptive only)", ""]
    rows = []
    for r in REGIONS:
        for b in BANDS:
            c = ex["band_calibration"][r][b]
            rows.append([r, b, f"{c['coverage_90']['full']:.3f}", f"{c['coverage_90']['xgboost']:.3f}",
                         f"{c['interval_width_90']['full']:.1f}", f"{c['interval_width_90']['xgboost']:.1f}"])
    lines.append(md_table(["region", "band", "cov90 Full", "cov90 XGB", "width90 Full", "width90 XGB"], rows))
    l2x = ex["linear_to_xgboost"]
    lines += ["", "## Gap by half-year of the test period (macro contribution; positive = Full worse)", ""]
    keys = sorted(ex["by_year_half"]["mae"])
    lines.append(md_table(["metric", *keys], [[mt, *[f"{ex['by_year_half'][mt][k]:+.3f}" for k in keys]] for mt in METRICS]))
    lines += ["", "## Absolute macro scores by protocol", ""]
    lines.append(md_table(["protocol", "Full MAE", "XGB MAE", "Full CRPS~", "XGB CRPS~"],
                          [[p, f"{v['mae']['full']:.3f}", f"{v['mae']['xgboost']:.3f}", f"{v['crps_quantile_approx']['full']:.3f}",
                            f"{v['crps_quantile_approx']['xgboost']:.3f}"] for p, v in ex["protocol_gaps"].items()]))
    lines += ["", "## MAE seed spread per region (static)", ""]
    lines.append(md_table(["region", "Full min", "Full max", "XGB min", "XGB max", "Full seeds (of 3) below XGB mean"],
                          [[r, *[f"{v:.3f}" for v in (e['full_min'], e['full_max'], e['xgb_min'], e['xgb_max'])], e['n_full_seeds_better_than_xgb_mean']]
                           for r, e in ex["mae_seed_spread"].items()]))
    lines += ["", f"Linear -> XGBoost macro MAE gap recovered by Full: linear {l2x['linear']:.3f}, Full {l2x['full']:.3f}, "
              f"XGBoost {l2x['xgboost']:.3f}, recovered {l2x['recovered_fraction']*100:.0f}%", ""]
    lines += ["## Paired block bootstrap in the report (candidate XGBoost minus Full; negative favours XGBoost)", ""]
    rows = []
    for blk, mm in ex["bootstrap_full_vs_xgboost"].items():
        for m, regs in mm.items():
            for r, v in regs.items():
                rows.append([blk, m, r, f"{v['difference']:+.3f}", f"[{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]", f"{v['p_value_holm']:.4f}"])
    lines.append(md_table(["block h", "metric", "region", "diff", "95% CI", "Holm p"], rows))
    hs, sr = result["history_summary"], result["selection_rule_sensitivity"]
    lines += ["", "## Earlier tracked Full-family runs: epoch behaviour and selection-rule sensitivity", "",
              f"- 9 runs x 3 regions = {hs['n_runs']} curves; validation-MAE-optimal epoch <= 5 in {hs['n_best_val_mae_epoch_le5']} of {hs['n_runs']} (median epoch {hs['median_best_val_mae_epoch']}); "
              f"best epoch differs between val-MAE and val-total in {hs['n_best_epoch_differs_mae_vs_total']}; mean (last - min) val MAE = {hs['mean_val_mae_last_minus_min']:.2f}; mean val-MAE range over epochs = {hs['mean_val_mae_range']:.2f}.",
              f"- {sr['n_runs']} stored runs: test MAE of the val-MAE-selected checkpoint vs the val-total-selected checkpoint differ by {sr['mean_abs_test_mae_difference']:.3f} on average "
              f"(max {sr['max_abs_test_mae_difference']:.3f}); different epoch in {sr['n_different_best_epoch']}; MAE rule better on test in {sr['n_mae_rule_better_on_test']}.", ""]
    lines += ["", "## Internal consistency checks (should be ~0)", ""]
    worst = max(abs(v) for v in result["checks"].values())
    lines.append(f"max |sum of slice contributions - regional gap| over {len(result['checks'])} checks = {worst:.2e}")
    (out / "gap_decomposition.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
