"""Read-only temporal/regime decomposition of the V5 joint-history final test.

Compares the frozen 2023-2024 final-test predictions of the V5 joint model
(``v5_joint_history_20261020_final_seed2026_r1``) against the archived
baseline_lightgbm seed-2026 cell predictions from ``aligned_baselines_20261012``.

Both runs share identical origins/targets/actuals (verified at load time). The
point metric is the repository convention: rearranged q50, equal weight per
origin-hour cell. QS proxy = 2 x mean pinball over cells and five levels.

Decomposition reported for every slice:
  gap_total   = MAE(Full_joint)    - MAE(baseline_lightgbm)
  gap_anchor  = MAE(Anchor_joint)  - MAE(baseline_lightgbm)   (joint correction effect)
  gap_history = MAE(Full_joint)    - MAE(Anchor_joint)        (history branch effect)

Slices: overall, year, month-of-year (pooled), year-month, realized 24h mean
price band, realized 24h max price band, horizon hour, target hour-of-day
(AEST). Signed mean error (bias) is also tabulated for drift diagnosis.

Outputs to ``logs/forecasting/v5_joint_gap_decomposition_20261020/``. Inputs are
never modified.

Usage (repository root):
  .venv/bin/python -m forecasting.analysis.v5_joint_vs_lightgbm_gap
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REGIONS = ("NSW1", "QLD1", "TAS1")
AEST = timezone(timedelta(hours=10))  # repo convention: all dates AEST (UTC+10)
LGB_DIR = Path("outputs/forecasting/aligned_baselines_20261012/runs/lightgbm/seed2026")
JOINT_DIR = Path("outputs/forecasting/v5_joint_history_20261020_final_seed2026_r1")
OUT_DIR = Path("logs/forecasting/v5_joint_gap_decomposition_20261020")
JOINT_MODELS = ("B_retrained", "Anchor_joint", "Full_joint",
                "event_off_same_head", "time_shift_off_same_head")

# Fixed, interpretable AUD/MWh bands (window statistics of the realized 24h).
MEAN_BANDS = ((0.0, 50.0), (50.0, 75.0), (75.0, 100.0), (100.0, 150.0),
              (150.0, 300.0), (300.0, float("inf")))
MAX_BANDS = ((0.0, 100.0), (100.0, 300.0), (300.0, 1000.0), (1000.0, float("inf")))


def load_region(region: str) -> dict[str, dict[str, np.ndarray]]:
    """Load baseline_lightgbm plus the joint-run artifacts; verify alignment."""
    lgb = np.load(LGB_DIR / region / "test.npz")
    out = {"baseline_lightgbm": {
        "point": np.sort(lgb["quantiles"], axis=-1)[..., 2],
        "quantiles": np.sort(lgb["quantiles"], axis=-1),
        "actual": lgb["actual"],
        "origin_unix": lgb["origin_unix"],
        "target_unix": lgb["target_time_unix"],
    }}
    for name in JOINT_MODELS:
        d = np.load(JOINT_DIR / region / f"{name}.npz")
        for key in ("origin_unix", "target_time_unix", "actual"):
            if not np.array_equal(lgb[key], d[key]):
                raise ValueError(f"{region}/{name}: {key} differs from baseline")
        out[name] = {
            "point": np.sort(d["quantiles"], axis=-1)[..., 2],
            "quantiles": np.sort(d["quantiles"], axis=-1),
        }
    return out


def qs_proxy(quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Per-cell 2 x mean pinball so any slice can average it afterwards."""
    levels = np.array([0.05, 0.1, 0.5, 0.9, 0.95])
    err = actual[..., None] - quantiles
    pin = np.maximum(levels * err, (levels - 1.0) * err)
    return 2.0 * pin.mean(axis=-1)


def slice_stats(models: dict, mask: np.ndarray) -> dict[str, dict[str, float]]:
    """MAE / RMSE / bias / QS averaged over the masked cells for every model."""
    actual = models["baseline_lightgbm"]["actual"]
    stats = {}
    for name, art in models.items():
        err = art["point"][mask] - actual[mask]
        s = {"mae": float(np.abs(err).mean()),
             "bias": float(err.mean()),
             "rmse": float(np.sqrt((err ** 2).mean()))}
        if "quantiles" in art:
            s["qs"] = float(qs_proxy(art["quantiles"], actual)[mask].mean())
        stats[name] = s
    return stats


def emit_rows(rows: list[dict], dim: str) -> None:
    with open(OUT_DIR / f"{dim}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_lines: list[str] = ["# V5 joint vs LightGBM: temporal/regime gap decomposition",
                               "",
                               "Frozen 2023-2024 final test, seed 2026, rearranged-q50 MAE,",
                               "equal weight per origin-hour cell. All slices share the same cells",
                               "across models. gap_total = Full_joint - LightGBM;",
                               "gap_anchor = Anchor_joint - LightGBM; gap_history = Full_joint - Anchor_joint.",
                               ""]
    csv_rows: dict[str, list[dict]] = {}

    region_data = {r: load_region(r) for r in REGIONS}
    for region, models in region_data.items():
        actual = models["baseline_lightgbm"]["actual"]
        origin_dt = np.array([datetime.fromtimestamp(int(t), tz=AEST)
                              for t in models["baseline_lightgbm"]["origin_unix"]])
        target_dt = np.array([datetime.fromtimestamp(int(t), tz=AEST)
                              for t in models["baseline_lightgbm"]["target_unix"][:, 0]])
        n = len(origin_dt)
        w_mean = actual.mean(axis=1)
        w_max = actual.max(axis=1)

        def run_dim(dim_name: str, labels: list[str], group_of) -> None:
            rows = []
            header = (f"### {region} — {dim_name}\n\n| group | n | LGB MAE | Anchor MAE | "
                      f"Full MAE | gap_anchor | gap_history | gap_total | LGB bias | Full bias |\n"
                      f"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            tbl = [header]
            for lab in labels:
                mask = group_of(lab)
                if mask.sum() == 0:
                    continue
                st = slice_stats(models, mask)
                row = {"dim": dim_name, "region": region, "group": str(lab), "n_cells": int(mask.sum())}
                for mn, s in st.items():
                    for k, v in s.items():
                        row[f"{mn}_{k}"] = v
                row["gap_anchor"] = st["Anchor_joint"]["mae"] - st["baseline_lightgbm"]["mae"]
                row["gap_history"] = st["Full_joint"]["mae"] - st["Anchor_joint"]["mae"]
                row["gap_total"] = st["Full_joint"]["mae"] - st["baseline_lightgbm"]["mae"]
                rows.append(row)
                tbl.append(f"| {lab} | {mask.sum()} | {fmt(st['baseline_lightgbm']['mae'])} | "
                           f"{fmt(st['Anchor_joint']['mae'])} | {fmt(st['Full_joint']['mae'])} | "
                           f"{fmt(row['gap_anchor'])} | {fmt(row['gap_history'])} | "
                           f"{fmt(row['gap_total'])} | {fmt(st['baseline_lightgbm']['bias'])} | "
                           f"{fmt(st['Full_joint']['bias'])} |")
            csv_rows.setdefault(dim_name, []).extend(rows)
            report_lines.extend(tbl + [""])

        years = sorted({d.year for d in origin_dt})
        run_dim("by_year", years, lambda y: np.array([d.year == y for d in origin_dt]))
        months = list(range(1, 13))
        run_dim("by_month", [f"{m:02d}" for m in months],
                lambda m: np.array([f"{d.month:02d}" == m for d in origin_dt]))
        yms = sorted({(d.year, d.month) for d in origin_dt})
        run_dim("by_year_month", [f"{y}-{m:02d}" for y, m in yms],
                lambda s: np.array([f"{d.year}-{d.month:02d}" == s for d in origin_dt]))
        run_dim("by_window_mean_price",
                [f"[{lo:.0f},{hi:.0f})" if np.isfinite(hi) else f">={lo:.0f}" for lo, hi in MEAN_BANDS],
                lambda s: _band_mask(w_mean, s, MEAN_BANDS))
        run_dim("by_window_max_price",
                [f"[{lo:.0f},{hi:.0f})" if np.isfinite(hi) else f">={lo:.0f}" for lo, hi in MAX_BANDS],
                lambda s: _band_mask(w_max, s, MAX_BANDS))

        # Horizon hour: per-cell mask over the horizon axis (broadcast over origins).
        h_rows, h_tbl = [], [f"### {region} — by horizon hour\n\n| h | n | LGB MAE | Anchor MAE | "
                             f"Full MAE | gap_anchor | gap_history | gap_total |\n|---|---:|---:|---:|---:|---:|---:|---:|"]
        for h in range(24):
            mask = np.zeros((n, 24), dtype=bool)
            mask[:, h] = True
            st = slice_stats(models, mask)
            gap_a = st["Anchor_joint"]["mae"] - st["baseline_lightgbm"]["mae"]
            gap_h = st["Full_joint"]["mae"] - st["Anchor_joint"]["mae"]
            h_rows.append({"dim": "by_horizon", "region": region, "group": h + 1,
                           "n_cells": n,
                           **{f"{mn}_{k}": v for mn, s in st.items() for k, v in s.items()},
                           "gap_anchor": gap_a, "gap_history": gap_h,
                           "gap_total": gap_a + gap_h})
            h_tbl.append(f"| {h + 1} | {n} | {fmt(st['baseline_lightgbm']['mae'])} | "
                         f"{fmt(st['Anchor_joint']['mae'])} | {fmt(st['Full_joint']['mae'])} | "
                         f"{fmt(gap_a)} | {fmt(gap_h)} | {fmt(gap_a + gap_h)} |")
        csv_rows.setdefault("by_horizon", []).extend(h_rows)
        report_lines.extend(h_tbl + [""])

        run_dim("by_target_hour_aest", [f"{h:02d}" for h in range(24)],
                lambda s: np.array([f"{d.hour:02d}" == s for d in target_dt]))

    # Macro summary (equal weight over the three regions), overall and per year.
    macro_tbl = ["## Macro summary (mean over NSW1/QLD1/TAS1)", "",
                 "| slice | LGB | Anchor | Full | event_off | time_shift_off | gap_anchor | gap_history | gap_total |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    overall = {}
    for region, models in region_data.items():
        n = models["baseline_lightgbm"]["actual"].shape
        mask = np.ones(n, dtype=bool)
        overall[region] = slice_stats(models, mask)
    for key in ("overall", "y2023", "y2024"):
        vals = {}
        for region, models in region_data.items():
            origin_dt = np.array([datetime.fromtimestamp(int(t), tz=AEST)
                                  for t in models["baseline_lightgbm"]["origin_unix"]])
            mask = (np.ones(models["baseline_lightgbm"]["actual"].shape, dtype=bool)
                    if key == "overall"
                    else np.repeat(np.array([d.year == int(key[1:]) for d in origin_dt])[:, None], 24, axis=1))
            st = slice_stats(models, mask)
            for mn in ("baseline_lightgbm", "Anchor_joint", "Full_joint",
                       "event_off_same_head", "time_shift_off_same_head"):
                vals.setdefault(mn, []).append(st[mn]["mae"])
        row = {mn: float(np.mean(v)) for mn, v in vals.items()}
        macro_tbl.append(f"| {key} | {fmt(row['baseline_lightgbm'])} | {fmt(row['Anchor_joint'])} | "
                         f"{fmt(row['Full_joint'])} | {fmt(row['event_off_same_head'])} | "
                         f"{fmt(row['time_shift_off_same_head'])} | "
                         f"{fmt(row['Anchor_joint'] - row['baseline_lightgbm'])} | "
                         f"{fmt(row['Full_joint'] - row['Anchor_joint'])} | "
                         f"{fmt(row['Full_joint'] - row['baseline_lightgbm'])} |")
    report_lines.extend([""] + macro_tbl + [""])

    # Counterfactual: align each model's in-group mean bias to LightGBM's and
    # rescore. Remaining gap is shape/dispersion, not systematic level shift.
    cf_tbl = ["## Bias-alignment counterfactual", "",
              "point' = point - (bias_model - bias_LGB) within each group (diagnostic only;",
              "uses group labels unavailable online). remaining gap = MAE(debiased) - MAE(LGB).", "",
              "| region | group | n | LGB MAE | Anchor debiased | Full debiased | gap_anchor_left | gap_total_left | gap_anchor_orig |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for region, models in region_data.items():
        actual = models["baseline_lightgbm"]["actual"]
        lgb_pt = models["baseline_lightgbm"]["point"]
        origin_dt = np.array([datetime.fromtimestamp(int(t), tz=AEST)
                              for t in models["baseline_lightgbm"]["origin_unix"]])
        groups = [("overall", np.ones(len(origin_dt), dtype=bool))]
        for m in range(1, 13):
            groups.append((f"m{m:02d}", np.array([d.month == m for d in origin_dt])))
        for gname, omask in groups:
            mask = np.repeat(omask[:, None], 24, axis=1)
            row = [f"| {region} | {gname} | {int(omask.sum() * 24)} | "]
            lgb_mae = float(np.abs(lgb_pt[mask] - actual[mask]).mean())
            lgb_bias = float((lgb_pt[mask] - actual[mask]).mean())
            vals = {}
            for mn in ("Anchor_joint", "Full_joint"):
                err = models[mn]["point"][mask] - actual[mask]
                bias = float(err.mean())
                deb = err - (bias - lgb_bias)
                vals[mn] = (float(np.abs(err).mean()), float(np.abs(deb).mean()))
            row.append(f"{fmt(lgb_mae)} | {fmt(vals['Anchor_joint'][1])} | {fmt(vals['Full_joint'][1])} | "
                       f"{fmt(vals['Anchor_joint'][1] - lgb_mae)} | {fmt(vals['Full_joint'][1] - lgb_mae)} | "
                       f"{fmt(vals['Anchor_joint'][0] - lgb_mae)} |")
            cf_tbl.append("".join(row))
    report_lines.extend([""] + cf_tbl + [""])

    for dim, rows in csv_rows.items():
        emit_rows(rows, dim)
    with open(OUT_DIR / "report.md", "w") as f:
        f.write("\n".join(report_lines))
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({r: {m: {k: v for k, v in st["baseline_lightgbm"].items()}
                       for m, st in {"baseline_lightgbm": overall[r]}.items()} for r in REGIONS}, f, indent=2)
    print(f"written: {OUT_DIR}/report.md and {len(csv_rows)} csv files")
    print("\n".join(macro_tbl))


def _band_mask(stat: np.ndarray, label: str, bands) -> np.ndarray:
    for lo, hi in bands:
        lab = f"[{lo:.0f},{hi:.0f})" if np.isfinite(hi) else f">={lo:.0f}"
        if lab == label:
            return (stat >= lo) & (stat < hi)
    raise KeyError(label)


if __name__ == "__main__":
    main()
