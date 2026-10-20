"""Read-only spike-window ranking of the V5 joint final test against all baselines.

Ranks every archived aligned-baseline model (seed 2026) together with the five
outputs of the frozen joint run (``v5_joint_history_20261020_final_seed2026_r1``)
on the shared 2023-2024 test cells, restricted to realized spike windows of the
24h forecast target:

  * sustained extreme : window mean actual >= 300 AUD/MWh
  * extreme spikes    : window max  actual >= 1000
  * moderate spikes   : window max  actual in [300, 1000)
  * calm complement   : window max  actual <  300   (for contrast)

Point metric is the repository convention (rearranged q50, equal weight per
origin-hour cell); signed mean error (bias) is tabulated alongside MAE.
All models are verified to share identical origins/targets/actuals per region.

This is a retrospective, single-seed, no-significance-test comparison of fixed
checkpoints; extreme bands contain few origins, so sub-percent MAE differences
between close models are not separable. It describes where the joint model's
advantage lives, not a SOTA claim.

Outputs to ``logs/forecasting/v5_joint_spike_ranking_20261020/``. Inputs are
never modified.

Usage (repository root):
  .venv/bin/python -m forecasting.analysis.v5_joint_spike_window_ranking
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

REGIONS = ("NSW1", "QLD1", "TAS1")
BASELINE_RUNS = (Path("outputs/forecasting/aligned_baselines_20261012/runs"),
                 Path("outputs/forecasting/aligned_baselines_20261013_extension/runs"))
JOINT_DIR = Path("outputs/forecasting/v5_joint_history_20261020_final_seed2026_r1")
JOINT_MODELS = ("B_retrained", "Anchor_joint", "Full_joint",
                "event_off_same_head", "time_shift_off_same_head")
OUT_DIR = Path("logs/forecasting/v5_joint_spike_ranking_20261020")

BANDS = {
    "sustained_mean_ge300": lambda wm, wmax: wm >= 300.0,
    "extreme_max_ge1000": lambda wm, wmax: wmax >= 1000.0,
    "moderate_max_300_1000": lambda wm, wmax: (wmax >= 300.0) & (wmax < 1000.0),
    "calm_max_lt300": lambda wm, wmax: wmax < 300.0,
}
BAND_ORDER = ("moderate_max_300_1000", "extreme_max_ge1000",
              "sustained_mean_ge300", "calm_max_lt300")
BAND_TITLES = {
    "moderate_max_300_1000": "moderate spikes: window max in [300, 1000)",
    "extreme_max_ge1000": "extreme spikes: window max >= 1000",
    "sustained_mean_ge300": "sustained extreme: window mean >= 300",
    "calm_max_lt300": "calm complement: window max < 300 (contrast)",
}
FOCUS = {"Full_joint", "lightgbm", "xgboost"}


def load_region(region: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    ref_path = BASELINE_RUNS[0] / "lightgbm" / "seed2026" / region / "test.npz"
    ref = np.load(ref_path)
    actual = ref["actual"]
    models: dict[str, np.ndarray] = {}
    for run_root in BASELINE_RUNS:
        if not run_root.exists():
            continue
        for mdir in sorted(p for p in run_root.iterdir() if p.is_dir()):
            f = mdir / "seed2026" / region / "test.npz"
            if not f.exists():
                continue
            z = np.load(f)
            for key in ("origin_unix", "target_time_unix", "actual"):
                if not np.array_equal(ref[key], z[key]):
                    raise ValueError(f"{mdir.name}/{region}: {key} differs from reference")
            models[mdir.name] = np.sort(z["quantiles"], axis=-1)[..., 2]
    for name in JOINT_MODELS:
        z = np.load(JOINT_DIR / region / f"{name}.npz")
        for key in ("origin_unix", "target_time_unix", "actual"):
            if not np.array_equal(ref[key], z[key]):
                raise ValueError(f"{name}/{region}: {key} differs from reference")
        models[name] = np.sort(z["quantiles"], axis=-1)[..., 2]
    if len(models) != 10 + 2 + len(JOINT_MODELS):
        raise ValueError(f"{region}: expected {12 + len(JOINT_MODELS)} models, got {len(models)}")
    return models, actual


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# V5 joint final test: spike-window ranking vs all aligned baselines",
             "",
             "Frozen 2023-2024 final test, seed 2026, rearranged-q50 MAE (AUD/MWh),",
             "equal weight per origin-hour cell, identical cells across all models.",
             "Bias = signed mean error (negative = under-forecast).",
             "Retrospective, single seed, no significance tests; extreme bands hold",
             "few origins, so sub-percent differences between close models are noise-level.",
             ""]
    csv_rows: list[dict] = []
    for band in BAND_ORDER:
        lines += [f"## {BAND_TITLES[band]}", ""]
        for region in REGIONS:
            models, actual = load_region(region)
            wm, wmax = actual.mean(axis=1), actual.max(axis=1)
            omask = BANDS[band](wm, wmax)
            mask = np.repeat(omask[:, None], actual.shape[1], axis=1)
            stats = {}
            for name, point in models.items():
                err = point[mask] - actual[mask]
                stats[name] = (float(np.abs(err).mean()), float(err.mean()))
            ranked = sorted(stats.items(), key=lambda kv: kv[1][0])
            lines.append(f"### {region} — n={int(omask.sum())} origins / {int(omask.sum()) * actual.shape[1]} cells")
            lines.append("")
            lines.append("| rank | model | MAE | bias |")
            lines.append("|---:|---|---:|---:|")
            for i, (name, (mae, bias)) in enumerate(ranked):
                mark = " **" if name in FOCUS else ""
                lines.append(f"| {i + 1} | {name}{mark} | {mae:.3f} | {bias:.1f} |")
                csv_rows.append({"band": band, "region": region, "rank": i + 1,
                                 "model": name, "n_origins": int(omask.sum()),
                                 "mae": mae, "bias": bias})
            lines.append("")
    with open(OUT_DIR / "spike_ranking.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    with open(OUT_DIR / "report.md", "w") as f:
        f.write("\n".join(lines))
    print(f"written: {OUT_DIR}/report.md, {OUT_DIR}/spike_ranking.csv ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
