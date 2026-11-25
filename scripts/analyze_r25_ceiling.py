"""R25 selection-ceiling analysis (plan v4.1, 2026-11-19).

Pre-registered initial S2 design, fixed before any S2 evaluation numbers were
seen: routing feature = mean as-of predispatch total-demand forecast over the
24h horizon; buckets = selection-period quartiles per fold x region
(boundaries estimated and frozen on the selection period only).

Evaluation periods never feed any rule: bucket boundaries, per-bucket
admission, S2 choices, S1 rolling windows and ensemble members are all
estimated from selection-period scores only. Outputs the fold/region summary,
the G1 gate evaluation (section 5.2), the section 5.4 trigger values and the
section 7 ablation ladders.

Usage: .venv/bin/python scripts/analyze_r25_ceiling.py \
    --run-dir outputs/forecasting/r25_ceiling_20261119 [--label preliminary]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forecasting.aligned_scoring import per_origin_scores as aligned_per_origin_scores
from forecasting.common.statistics import block_bootstrap_and_jackknife
FOLDS = ("A", "B", "C")
REGIONS = ("NSW1", "QLD1", "TAS1")
Q = np.array([0.05, 0.10, 0.50, 0.90, 0.95])
SEEDS = (2026, 2027, 2028)
R0_KEY = "lightgbm:2026"
ADMISSION_BAND = 1.25
S1_WINDOW_DAYS = 28
S1_MIN_MATURED = 50
G1_ABS, G1_REL = 0.25, 0.006
QS_HARM, MAE_HARM, QS_FLOOR, MAE_FLOOR = 0.05, 0.15, 2.0, 2.0
JACKKNIFE_MAX_SHARE = 0.25
NET_HEADROOM_TRIGGER = 1.00
BOOTSTRAP_REPS = 2000
DISTINCT_MODELS = ("lightgbm:2026", "persistence", "snaive", "chronos2", "xgboost")
CANDIDATE_ORDER = ("lightgbm:2026", "lightgbm:2027", "lightgbm:2028",
                   "persistence", "snaive", "chronos2", "xgboost")


def per_origin_scores(actual: np.ndarray, quantiles: np.ndarray):
    scores = aligned_per_origin_scores(actual, quantiles)
    return scores["quantile_score_proxy"], scores["mae"]


def load_scores(run_dir: Path) -> pd.DataFrame:
    rows = []
    references = {}
    for fold_dir in sorted(p for p in run_dir.iterdir() if p.name in FOLDS):
        for region_dir in sorted(fold_dir.iterdir()):
            for cell in sorted(region_dir.iterdir()):
                for period_file in sorted(cell.glob("*.npz")):
                    with np.load(period_file) as z:
                        qs, mae = per_origin_scores(z["actual"], z["quantiles"])
                        origins = z["origin_unix"]
                        if len(origins) != len(qs) or np.any(np.diff(origins) <= 0):
                            raise ValueError(f"Unordered or misaligned origins: {period_file}")
                        reference_key = (fold_dir.name, region_dir.name, period_file.stem)
                        current = (origins.copy(), z["actual"].copy())
                        if reference_key in references:
                            if not all(np.array_equal(a, b) for a, b in
                                       zip(current, references[reference_key])):
                                raise ValueError(f"Candidate sample/label mismatch: {period_file}")
                        else:
                            references[reference_key] = current
                        cell_name = cell.name
                        if "_seed" in cell_name:
                            model, seed = cell_name.rsplit("_seed", 1)
                            key = f"{model}:{seed}"
                        else:
                            key = cell_name
                        for o, a, b in zip(origins, qs, mae):
                            rows.append({"fold": fold_dir.name, "region": region_dir.name,
                                         "key": key, "period": period_file.stem,
                                         "origin": int(o), "qs": float(a), "mae": float(b)})
    return pd.DataFrame(rows)


def bucket_feature(region: str, origins: np.ndarray) -> np.ndarray:
    with np.load(ROOT / "data/aemo_exogenous" / f"{region.lower()}_predispatch.npz") as z:
        index = {int(t): i for i, t in enumerate(z["forecast_origin_unix"])}
        demand = z["predispatch_total_demand_mw"]
    return demand[[index[int(t)] for t in origins]].mean(axis=1).astype(np.float64)


def s2_policy(sel_f: pd.DataFrame, features: pd.DataFrame):
    """Selection-only: quartile bounds, per-bucket admission and S2 choice."""
    policy, bounds = {}, {}
    for region in REGIONS:
        sel_r = sel_f[sel_f.region == region].copy()
        feat = features[features.region == region].set_index("origin")["feature"]
        sel_r["feature"] = sel_r.origin.map(feat)
        bounds[region] = np.quantile(sel_r.feature.to_numpy(), [0.25, 0.5, 0.75])
        sel_r["bucket"] = np.searchsorted(bounds[region], sel_r.feature.to_numpy(), side="right")
        r0 = sel_r[sel_r.key == R0_KEY].set_index("origin")["qs"]
        for bucket in range(4):
            members = sel_r[sel_r.bucket == bucket]
            default = (R0_KEY, [], [])
            if members.empty:
                policy[(region, bucket)] = default
                continue
            table = members.groupby("key")["qs"].mean()
            admitted = sorted(table[table <= ADMISSION_BAND * table[R0_KEY]].index)
            best, best_se = R0_KEY, np.inf
            for key in admitted:
                diffs = np.array([row.qs - r0[row.origin] for row in members.itertuples()
                                  if row.key == key], dtype=np.float64)
                se = diffs.std(ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else np.inf
                margin = table[R0_KEY] - table[key]
                if margin >= se and table[key] < best_se:
                    best, best_se = key, table[key]
            policy[(region, bucket)] = (best, admitted, float(best_se))
    return bounds, policy


def s1_rolling(ev_r: pd.DataFrame) -> np.ndarray:
    wide = ev_r.pivot_table(index="origin", columns="key", values="qs").sort_index()
    origins = wide.index.to_numpy()
    values = wide.to_numpy()
    columns = list(wide.columns)
    out = np.empty(len(origins))
    for i, t in enumerate(origins):
        mask = (origins >= t - S1_WINDOW_DAYS * 86400) & (origins + 25 * 3600 <= t)
        out[i] = values[i, columns.index(R0_KEY)] if mask.sum() < S1_MIN_MATURED \
            else values[i, int(np.argmin(values[mask].mean(axis=0)))]
    return out


def region_equal(series: dict) -> float:
    return float(np.mean(list(series.values())))


def bootstrap_and_jackknife(per_region_paired: dict[str, pd.DataFrame]):
    """Synchronous 7-day blocks, preserving resampled multiplicities.

    The historical ``bootstrap_p_leq0`` name denotes an empirical bootstrap
    tail fraction, not a null-centred hypothesis-test p-value.
    """
    days = {region: pd.to_datetime(frame.origin, unit="s", utc=True).dt.tz_convert(
                "Australia/Brisbane").dt.date.to_numpy()
            for region, frame in per_region_paired.items()}
    values = {region: frame.diff_qs.to_numpy() for region, frame in per_region_paired.items()}
    return block_bootstrap_and_jackknife(values, days, block_days=7,
                                        repetitions=BOOTSTRAP_REPS, seed=2026,
                                        alternative="greater")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "outputs/forecasting/r25_ceiling_20261119")
    parser.add_argument("--label", default="preliminary")
    parser.add_argument("--output", type=Path, help="New corrected report; never overwrite a frozen report")
    args = parser.parse_args()
    out = args.output or args.run_dir / f"analysis_{args.label}.json"
    if out.exists():
        raise FileExistsError(f"Preserve the existing analysis; choose a new output: {out}")

    scores = load_scores(args.run_dir)
    expected_candidates = {*(f"lightgbm:{s}" for s in SEEDS), "xgboost:2026",
                           "chronos2", "persistence", "snaive"}
    expected_cells = {(f, r, k, p) for f in FOLDS for r in REGIONS
                      for k in expected_candidates for p in ("selection", "evaluation")}
    observed_cells = set(scores[["fold", "region", "key", "period"]].itertuples(index=False, name=None))
    if observed_cells != expected_cells:
        raise ValueError("R25 requires the complete frozen three-fold, three-region, seven-candidate grid")
    evaluation, selection = scores[scores.period == "evaluation"], scores[scores.period == "selection"]
    present = sorted(evaluation.key.unique())
    quantile_cache = {}

    def quantiles_of(fold, region, key):
        if (fold, region, key) not in quantile_cache:
            quantile_cache[(fold, region, key)] = load_quantile_arrays(args.run_dir, fold, region, key)
        return quantile_cache[(fold, region, key)]

    folds_out = {}
    for fold in FOLDS:
        ev_f, sel_f = evaluation[evaluation.fold == fold], selection[selection.fold == fold]
        if ev_f.empty:
            continue
        bounds, policy = s2_policy(sel_f, features_for(ev_f, sel_f))
        regions_out, per_region_paired, per_region_paired_s3 = {}, {}, {}
        for region in REGIONS:
            ev_r = ev_f[ev_f.region == region]
            qs_t = ev_r.pivot_table(index="origin", columns="key", values="qs").sort_index()
            mae_t = ev_r.pivot_table(index="origin", columns="key", values="mae").sort_index()
            origins = qs_t.index.to_numpy()
            feat = bucket_feature(region, origins)
            buckets = np.searchsorted(bounds[region], feat, side="right")
            choices = np.array([policy[(region, b)][0] for b in buckets])
            col = [qs_t.columns.get_loc(k) for k in choices]
            s2_qs = qs_t.to_numpy()[np.arange(len(origins)), col]
            s2_mae = mae_t.to_numpy()[np.arange(len(origins)), col]
            r0_qs, r0_mae = qs_t[R0_KEY].to_numpy(), mae_t[R0_KEY].to_numpy()
            seeds = np.stack([qs_t[f"lightgbm:{s}"].to_numpy() for s in SEEDS])
            month = pd.to_datetime(origins, unit="s", utc=True).tz_convert(
                "Australia/Brisbane").strftime("%Y-%m")
            # group oracles: choose the best candidate per group, take its per-origin scores
            best_by_month = qs_t.groupby(month).mean().idxmin(axis=1)
            c2_qs = float(pd.concat([qs_t[month == m][k] for m, k in best_by_month.items()]).mean())
            best_by_bucket = pd.DataFrame(qs_t.to_numpy()).groupby(buckets).mean()
            best_by_bucket.columns = qs_t.columns
            r3_qs = float(pd.concat([qs_t[buckets == b][k]
                                     for b, k in best_by_bucket.idxmin(axis=1).items()]).mean())
            # S3: per-bucket admitted members, per-quantile median
            s3_qs = np.empty(len(origins))
            s3_mae = np.empty(len(origins))
            for b in range(4):
                rows_b = buckets == b
                members = policy[(region, b)][1] or [R0_KEY]
                stack = np.stack([quantiles_of(fold, region, k)[0][rows_b] for k in members])
                actual_b = quantiles_of(fold, region, members[0])[1][rows_b]
                s3_qs[rows_b], s3_mae[rows_b] = per_origin_scores(
                    actual_b, np.median(stack, axis=0))
            s1_arr = s1_rolling(ev_r)
            per_region_paired[region] = pd.DataFrame({
                "origin": origins, "diff_qs": r0_qs - s2_qs, "diff_mae": r0_mae - s2_mae})
            per_region_paired_s3[region] = pd.DataFrame({
                "origin": origins, "diff_qs": r0_qs - s3_qs, "diff_mae": r0_mae - s3_mae})
            seed_qs_agg = [float(qs_t[f"lightgbm:{s}"].mean()) for s in SEEDS]
            seed_mae_agg = [float(mae_t[f"lightgbm:{s}"].mean()) for s in SEEDS]
            regions_out[region] = {
                "r0_qs": float(r0_qs.mean()), "r0_mae": float(r0_mae.mean()),
                "s2_qs": float(s2_qs.mean()), "s2_mae": float(s2_mae.mean()),
                "s1_qs": float(s1_arr.mean()),
                "s3_qs": float(s3_qs.mean()), "s3_mae": float(s3_mae.mean()),
                "c1_qs": float(qs_t.min(axis=1).mean()),
                "c1_null_qs": float(seeds.min(axis=0).mean()),
                "c2_qs": c2_qs,
                "r1_impl_qs": float(qs_t[sel_best_key(sel_f, region)].mean()),
                "r1_hindsight_qs": float(qs_t[qs_t.mean().idxmin()].mean()),
                "r3_hindsight_qs": r3_qs,
                "seed_qs_sd": float(np.std(seed_qs_agg, ddof=1)),
                "seed_mae_sd": float(np.std(seed_mae_agg, ddof=1)),
                "s2_choices": pd.Series(choices).value_counts().to_dict(),
                "selected_by_bucket": {str(b): policy[(region, b)][0] for b in range(4)},
                "admitted_by_bucket": {str(b): policy[(region, b)][1] for b in range(4)},
                "bucket_bounds": bounds[region].tolist(),
            }
        robust = bootstrap_and_jackknife(per_region_paired)
        robust_s3 = bootstrap_and_jackknife(per_region_paired_s3)
        folds_out[fold] = {"regions": regions_out, "robust": robust, "robust_s3": robust_s3}

    # ---- fold-level gate evaluation (S2 main estimator; S3 branch wording) ----
    gate = {}
    for fold, data in folds_out.items():
        regions = data["regions"]
        re = lambda k: region_equal({r: regions[r][k] for r in REGIONS})
        delta_qs = re("r0_qs") - re("s2_qs")
        delta_s3 = re("r0_qs") - re("s3_qs")
        harm = lambda arm_qs, arm_mae: (
            all(regions[r][arm_qs] - regions[r]["r0_qs"]
                <= max(QS_HARM, QS_FLOOR * regions[r]["seed_qs_sd"]) for r in REGIONS)
            and all(regions[r][arm_mae] - regions[r]["r0_mae"]
                    <= max(MAE_HARM, MAE_FLOOR * regions[r]["seed_mae_sd"]) for r in REGIONS))
        gate[fold] = {
            "r0": re("r0_qs"), "s2": re("s2_qs"), "s1": re("s1_qs"), "s3": re("s3_qs"),
            "c1": re("c1_qs"), "c1_null": re("c1_null_qs"), "c2": re("c2_qs"),
            "delta_s2": delta_qs, "delta_rel": delta_qs / re("r0_qs"),
            "delta_s3": delta_s3, "delta_s3_rel": delta_s3 / re("r0_qs"),
            "net_headroom": re("c1_null_qs") - re("c1_qs"),
            "a_pass": bool(delta_qs >= G1_ABS and delta_qs / re("r0_qs") >= G1_REL),
            "b_pass": harm("s2_qs", "s2_mae"),
            "a_s3_pass": bool(delta_s3 >= G1_ABS and delta_s3 / re("r0_qs") >= G1_REL),
            "b_s3_pass": harm("s3_qs", "s3_mae"),
            "robust": data["robust"], "robust_s3": data["robust_s3"],
        }
    def c_pass(key):
        return (sum(g[key]["bootstrap_p_leq0"] < 0.05 for g in gate.values()) >= 2
                and all(g[key]["jackknife_max_day_share"] is not None
                        and g[key]["jackknife_max_day_share"] < JACKKNIFE_MAX_SHARE
                        for g in gate.values()))
    verdict = {
        "G1_a": all(g["a_pass"] for g in gate.values()),
        "G1_b": all(g["b_pass"] for g in gate.values()),
        "G1_c": c_pass("robust"),
        "S3_branch_a": all(g["a_s3_pass"] for g in gate.values()),
        "S3_branch_b": all(g["b_s3_pass"] for g in gate.values()),
        "S3_branch_c": c_pass("robust_s3"),
    }
    verdict["G1_pass"] = verdict["G1_a"] and verdict["G1_b"] and verdict["G1_c"]
    verdict["S3_branch_pass"] = (verdict["S3_branch_a"] and verdict["S3_branch_b"]
                                 and verdict["S3_branch_c"])
    verdict["outcome"] = (
        "G1_pass" if verdict["G1_pass"] else
        "close_selection_routing_combination_default" if verdict["S3_branch_pass"]
        else "direction_closed")

    # ---- section 5.4 trigger values ----
    trigger = {
        "net_headroom_ge_1_all_folds": all(g["net_headroom"] >= NET_HEADROOM_TRIGGER
                                           for g in gate.values()),
        "s2_only_failed_on_a": bool((not verdict["G1_pass"]) and (not verdict["G1_a"])
                                    and verdict["G1_b"] and verdict["G1_c"]),
        "positive_impl_all_folds": all(
            g["delta_s2"] > 0 or g["s1"] < g["r0"] or g["s3"] < g["r0"] for g in gate.values()),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"candidates": present, "gate": gate, "verdict": verdict,
                               "trigger_5_4": trigger, "folds": folds_out,
                               "correction": {"scoring": "shared_sorted_five_quantile_score",
                                   "selection_recomputed": True,
                                   "statistics": "synchronous_nonoverlapping_7_calendar_day_blocks",
                                   "bootstrap_note": "Empirical tail fractions are not null-centred p-values; historical gate thresholds unchanged."}},
                              indent=2, default=str, allow_nan=False))
    print(f"wrote {out}")
    print("candidates:", present)
    for fold, g in gate.items():
        jackknife = g['robust_s3']['jackknife_max_day_share']
        jackknife_text = "undefined" if jackknife is None else f"{jackknife:.1%}"
        print(f"fold {fold}: R0={g['r0']:.2f} S2={g['s2']:.2f} Δ={g['delta_s2']:+.3f} "
              f"({g['delta_rel']:+.2%}) S3={g['s3']:.2f} Δ3={g['delta_s3']:+.3f} "
              f"({g['delta_s3_rel']:+.2%}) net={g['net_headroom']:+.2f} "
              f"p2={g['robust']['bootstrap_p_leq0']:.3f} p3={g['robust_s3']['bootstrap_p_leq0']:.3f} "
              f"jk3={jackknife_text}")
    print("verdict:", verdict, "| §5.4 trigger:", trigger)


def load_quantile_arrays(run_dir: Path, fold: str, region: str, key: str):
    cell = key.replace(":", "_seed")
    with np.load(run_dir / fold / region / cell / "evaluation.npz") as z:
        return z["quantiles"], z["actual"]


def features_for(ev_f: pd.DataFrame, sel_f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        ev_origins = ev_f[(ev_f.region == region) & (ev_f.key == R0_KEY)].origin.to_numpy()
        sel_origins = sel_f[(sel_f.region == region) & (sel_f.key == R0_KEY)].origin.to_numpy()
        origins = np.unique(np.concatenate([ev_origins, sel_origins]))
        values = bucket_feature(region, origins)
        rows.extend({"region": region, "origin": int(t), "feature": v}
                    for t, v in zip(origins, values))
    return pd.DataFrame(rows)


def sel_best_key(sel_f: pd.DataFrame, region: str) -> str:
    table = sel_f[sel_f.region == region].groupby("key")["qs"].mean()
    return table.idxmin()


if __name__ == "__main__":
    main()
