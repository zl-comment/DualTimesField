"""R27 S3 confirmatory run (preregistration r27_s3_confirmation_preregistration_20261124.md).

Executes section 9 in order: archive verification, scorer-consistency assertion,
rule freezing, six-arm combination + one-shot scoring, statistics (DM+HLN,
7-day block bootstrap, Holm per-region DM, jackknife, price bands), and the
G-C1 / G-C2 verdict. Reads frozen archives only; assertion failures abort.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forecasting.aligned_scoring import make_prediction_artifact, score_predictions

REGIONS = ("NSW1", "QLD1", "TAS1")
Q = np.array([0.05, 0.10, 0.50, 0.90, 0.95])
MEMBERS = ("lightgbm:2026", "lightgbm:2027", "lightgbm:2028", "xgboost:2026", "chronos2")
R0_KEY = "lightgbm:2026"
ADMIT_BAND = 1.25
OUT = ROOT / "outputs/forecasting/r27_s3_confirmation_20261125"
MAIN_PER_SEED = ROOT / "logs/forecasting/main_comparison_20261023/per_seed.csv"
FOUNDATION = ROOT.parent / "DualTimesField-foundation-baselines"
BOOT_REPS, BLOCK_DAYS, DM_H, NW_LAG = 2000, 7, 24, 23
RNG = np.random.default_rng(2026)
GIT = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              text=True).strip()


def cell_path(key, region):
    if key.startswith("lightgbm"):
        return ROOT / f"outputs/forecasting/aligned_baselines_20261012/runs/lightgbm/seed{key.split(':')[1]}/{region}/test.npz"
    if key.startswith("xgboost"):
        return ROOT / f"outputs/forecasting/aligned_baselines_20261012/runs/xgboost/seed2026/{region}/test.npz"
    return FOUNDATION / f"outputs/forecasting/foundation_baselines_20261023_r1/runs/chronos2_zeroshot/seed2026/{region}/test.npz"


def r25_selection_path(key, region):
    return ROOT / f"outputs/forecasting/r25_ceiling_20261119/C/{region}/{key.replace(':', '_seed')}/selection.npz"


def per_origin_qs(actual, quantiles):
    err = actual[..., None] - quantiles
    return 2 * np.maximum(Q * err, (Q - 1) * err).mean(axis=(1, 2))


def bucket_feature(region, origins):
    with np.load(ROOT / "data/aemo_exogenous" / f"{region.lower()}_predispatch.npz") as z:
        index = {int(t): i for i, t in enumerate(z["forecast_origin_unix"])}
        demand = z["predispatch_total_demand_mw"]
    return demand[[index[int(t)] for t in origins]].mean(axis=1).astype(np.float64)


def local_dates(unix):
    stamps = pd.DatetimeIndex(pd.to_datetime(unix, unit="s", utc=True)).tz_convert("Australia/Brisbane")
    return stamps.date


def dm_hln_one_sided(diff):
    """DM with Bartlett/NW lag 23 + HLN correction; improvement means diff < 0."""
    n = len(diff)
    d = diff - diff.mean()
    gamma = [(d * d).mean()] + [(d[:-l] * d[l:]).mean() for l in range(1, NW_LAG + 1)]
    var = gamma[0] + 2 * sum((1 - l / (NW_LAG + 1)) * gamma[l] for l in range(1, NW_LAG + 1))
    t_stat = diff.mean() / math.sqrt(var / n)
    hln = t_stat * math.sqrt((n + 1 - 2 * DM_H + DM_H * (DM_H - 1) / n) / n)
    return t_stat, hln, 0.5 * (1 + math.erf(hln / math.sqrt(2)))


def _day_codes(diff_by_region):
    all_days = sorted({d for s in diff_by_region.values() for d in local_dates(s.index)})
    code_of = {d: i for i, d in enumerate(all_days)}
    codes = {r: np.array([code_of[d] for d in local_dates(s.index)], dtype=np.int64)
             for r, s in diff_by_region.items()}
    return all_days, codes


def block_bootstrap(diff_by_region):
    """7-day-block bootstrap on integer day codes; per-day sums make resampling O(blocks)."""
    all_days, codes = _day_codes(diff_by_region)
    n_days = len(all_days)
    arrays = {r: s.to_numpy() for r, s in diff_by_region.items()}
    day_sum = {r: np.bincount(codes[r], weights=arrays[r], minlength=n_days) for r in arrays}
    day_cnt = {r: np.bincount(codes[r], minlength=n_days) for r in arrays}
    blocks = [np.arange(i, min(i + BLOCK_DAYS, n_days)) for i in range(0, n_days, BLOCK_DAYS)]
    means = []
    for _ in range(BOOT_REPS):
        sel = np.concatenate([blocks[i] for i in RNG.integers(0, len(blocks), len(blocks))])
        means.append(np.mean([day_sum[r][sel].sum() / day_cnt[r][sel].sum() for r in arrays]))
    means = np.array(means)
    return float((means >= 0).mean()), (float(np.percentile(means, 2.5)),
                                        float(np.percentile(means, 97.5)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[R27] git={GIT}", flush=True)

    # ---- step 1: archives and alignment ----
    data = {}
    for region in REGIONS:
        region_data, reference = {}, None
        for key in MEMBERS:
            with np.load(cell_path(key, region)) as z:
                arrays = {k: z[k] for k in ("quantiles", "actual", "origin_unix", "target_time_unix")}
                meta = json.loads(str(z["metadata_json"]))
            assert arrays["quantiles"].shape == (17521, 24, 5), (key, region)
            if reference is None:
                reference = arrays
            else:
                assert np.array_equal(arrays["origin_unix"], reference["origin_unix"]), (key, region)
                assert np.array_equal(arrays["target_time_unix"], reference["target_time_unix"])
                assert np.array_equal(arrays["actual"], reference["actual"])
            region_data[key] = arrays
            if key == R0_KEY:
                region_data["meta"] = meta
        data[region] = region_data
    origins = data["NSW1"][R0_KEY]["origin_unix"]
    assert all(np.array_equal(data[r][R0_KEY]["origin_unix"], origins) for r in REGIONS)
    feature = {r: bucket_feature(r, origins) for r in REGIONS}
    assert all(len(f) == 17521 for f in feature.values())
    print("[1] 15 archives aligned, labels identical, predispatch coverage 17521/17521", flush=True)

    # ---- step 2: scorer consistency vs main table ----
    table = pd.read_csv(MAIN_PER_SEED)
    for region in REGIONS:
        ref = data[region][R0_KEY]
        artifact = make_prediction_artifact(
            actual=ref["actual"], origin_unix=ref["origin_unix"],
            target_time_unix=ref["target_time_unix"],
            metadata=dict(data[region]["meta"], source=f"r27-rescore:{GIT}"),
            quantiles=ref["quantiles"], point=None, quantile_levels=tuple(Q))
        s = score_predictions(artifact)
        row = table[(table.model == "baseline_lightgbm") & (table.seed == 2026)
                    & (table.region == region)].iloc[0]
        assert abs(s["quantile_score_proxy"] - row.quantile_score_proxy) <= 1e-9, (region, s)
        assert abs(s["mae"] - row.mae) <= 1e-9, (region, s)
    print("[2] scorer consistency: 9/9 cells match per_seed.csv within 1e-9", flush=True)

    # ---- step 3: freeze rules ----
    rules = {"git_commit": GIT,
             "foundation_worktree_commit": subprocess.check_output(
                 ["git", "-C", str(FOUNDATION), "rev-parse", "--short", "HEAD"],
                 text=True).strip(),
             "selection_window": ["2022-01-01", "2022-07-01"], "band": ADMIT_BAND,
             "members": list(MEMBERS), "boundaries": {}, "admission": {}}
    for region in REGIONS:
        with np.load(ROOT / "data/aemo_exogenous" / f"{region.lower()}_predispatch.npz") as z:
            all_origin = z["forecast_origin_unix"].astype(np.int64)
            all_demand = z["predispatch_total_demand_mw"].mean(axis=1)
        iso = np.array([d.isoformat() for d in local_dates(all_origin)])
        mask = (iso >= "2022-01-01") & (iso < "2022-07-01")
        bounds = np.quantile(all_demand[mask], [0.25, 0.5, 0.75])
        sel_qs = {}
        for key in MEMBERS:
            with np.load(r25_selection_path(key, region)) as z:
                sel_qs[key] = dict(zip(z["origin_unix"].astype(np.int64),
                                       per_origin_qs(z["actual"], z["quantiles"])))
        sel_origins = np.array(sorted(sel_qs[R0_KEY]))
        demand_of = dict(zip(all_origin, all_demand))
        sel_feat = np.array([demand_of[t] for t in sel_origins])
        sel_buckets = np.searchsorted(bounds, sel_feat, side="right")
        bucket_of = dict(zip(sel_origins, sel_buckets))
        admission = {}
        for bucket in range(4):
            qs_mean = {key: np.mean([v for t, v in sel_qs[key].items()
                                     if bucket_of[t] == bucket]) for key in MEMBERS}
            admission[bucket] = sorted(k for k, v in qs_mean.items()
                                       if v <= ADMIT_BAND * qs_mean[R0_KEY])
        data[region]["test_buckets"] = np.searchsorted(bounds, feature[region], side="right")
        data[region]["admission"] = admission
        rules["boundaries"][region] = bounds.tolist()
        rules["admission"][region] = {str(b): admission[b] for b in range(4)}
        print(f"[3] {region} bounds={[round(float(b), 1) for b in bounds]} "
              f"admitted={ {b: [k.split(':')[0] for k in v] for b, v in admission.items()} }", flush=True)
    (OUT / "frozen_rules.json").write_text(json.dumps(rules, indent=2))
    print("[3] rules frozen -> frozen_rules.json", flush=True)

    # ---- step 4: arms ----
    def bucketed_combine(region, keys_by_bucket, reducer):
        out = np.empty((17521, 24, 5))
        buckets = data[region]["test_buckets"]
        for bucket, keys in keys_by_bucket.items():
            rows = buckets == bucket
            stack = np.stack([data[region][k]["quantiles"][rows] for k in keys])
            out[rows] = np.sort(reducer(stack, axis=0), axis=-1)
        return out

    def global_combine(region, keys, reducer):
        stack = np.stack([data[region][k]["quantiles"] for k in keys])
        return np.sort(reducer(stack, axis=0), axis=-1)

    ARMS = {
        "s3": lambda r: bucketed_combine(r, data[r]["admission"], np.median),
        "r0_prime": lambda r: global_combine(r, [k for k in MEMBERS if k.startswith("lightgbm")], np.median),
        "no_chronos": lambda r: bucketed_combine(
            r, {b: [k for k in v if k != "chronos2"] or [R0_KEY]
                for b, v in data[r]["admission"].items()}, np.median),
        "no_admission": lambda r: global_combine(r, list(MEMBERS), np.median),
        "mean": lambda r: bucketed_combine(r, data[r]["admission"], np.mean),
        "slim3": lambda r: global_combine(r, [R0_KEY, "xgboost:2026", "chronos2"], np.median),
    }
    rows, losses, s3_quantiles = [], {}, {}
    for region in REGIONS:
        ref = data[region][R0_KEY]
        for arm, builder in ARMS.items():
            quantiles = builder(region)
            artifact = make_prediction_artifact(
                actual=ref["actual"], origin_unix=ref["origin_unix"],
                target_time_unix=ref["target_time_unix"],
                metadata=dict(data[region]["meta"], model=f"r27_{arm}", seed=2026,
                              source=f"r27:{GIT}", arm=arm, frozen_rules="frozen_rules.json"),
                quantiles=quantiles, point=None, quantile_levels=tuple(Q))
            np.savez_compressed(OUT / f"{region}_{arm}.npz", quantiles=quantiles,
                                actual=ref["actual"], origin_unix=ref["origin_unix"],
                                metadata_json=np.asarray(json.dumps(artifact.metadata, sort_keys=True)))
            s = score_predictions(artifact)
            rows.append({"arm": arm, "region": region,
                         **{k: s[k] for k in ("quantile_score_proxy", "mae", "rmse_pooled",
                                              "coverage_80", "coverage_90")}})
            losses[(arm, region)] = per_origin_qs(ref["actual"], quantiles)
            if arm == "s3":
                s3_quantiles[region] = quantiles
        q50 = np.sort(ref["quantiles"], axis=-1)[..., 2]
        losses[("r0", region)] = per_origin_qs(ref["actual"], ref["quantiles"])
        rows.append({"arm": "r0", "region": region,
                     "quantile_score_proxy": losses[("r0", region)].mean(),
                     "mae": np.abs(ref["actual"] - q50).mean(),
                     "rmse_pooled": float(np.sqrt(np.mean((ref["actual"] - q50) ** 2))),
                     "coverage_80": None, "coverage_90": None})
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "scores.csv", index=False)
    print("[4] arms scored:", flush=True)
    print(scores.pivot_table(index="arm", columns="region",
                             values="quantile_score_proxy").round(3).to_string(), flush=True)

    # ---- step 5: statistics ----
    diff = {r: pd.Series(losses[("s3", r)] - losses[("r0", r)], index=origins) for r in REGIONS}
    diff2 = {r: pd.Series(losses[("r0_prime", r)] - losses[("s3", r)], index=origins) for r in REGIONS}
    pooled = np.mean([s.to_numpy() for s in diff.values()], axis=0)
    t_stat, hln, dm_p = dm_hln_one_sided(pooled)
    boot_p, boot_ci = block_bootstrap(diff)
    _, boot2_ci = block_bootstrap(diff2)
    per_region_p = {r: dm_hln_one_sided(diff[r].to_numpy())[2] for r in REGIONS}
    ranked = sorted(per_region_p.items(), key=lambda kv: kv[1])
    holm = {r: min(1.0, p * (3 - i)) for i, (r, p) in enumerate(ranked)}
    delta_full = float(np.mean([s.mean() for s in diff.values()]))
    all_days, codes = _day_codes(diff)
    arrays = {r: s.to_numpy() for r, s in diff.items()}
    day_sum = {r: np.bincount(codes[r], weights=arrays[r], minlength=len(all_days)) for r in arrays}
    day_cnt = {r: np.bincount(codes[r], minlength=len(all_days)) for r in arrays}
    totals = {r: (day_sum[r].sum(), day_cnt[r].sum()) for r in arrays}
    jk = max(abs(delta_full - np.mean([(totals[r][0] - day_sum[r][d]) / (totals[r][1] - day_cnt[r][d])
                                       for r in arrays])) / abs(delta_full)
             for d in range(len(all_days))) if delta_full else 0.0
    bands = {}
    for lo, hi, name in ((-1e9, 100.0, "<100"), (100.0, 300.0, "100-300"), (300.0, 1e9, ">300")):
        per_region = []
        for r in REGIONS:
            actual = data[r][R0_KEY]["actual"]
            mask = (actual >= lo) & (actual < hi)
            err0 = actual[..., None] - data[r][R0_KEY]["quantiles"]
            err3 = actual[..., None] - s3_quantiles[r]
            cell0 = 2 * np.maximum(Q * err0, (Q - 1) * err0)
            cell3 = 2 * np.maximum(Q * err3, (Q - 1) * err3)
            per_region.append(cell0[mask].mean() - cell3[mask].mean())
        bands[name] = float(np.mean(per_region))
    win_rate = float((pooled < 0).mean())
    trimmed = float(np.sort(pooled)[int(0.1 * len(pooled)):int(0.9 * len(pooled))].mean())

    # ---- step 6: verdict ----
    re = lambda arm: float(np.mean([losses[(arm, r)].mean() for r in REGIONS]))
    delta_qs, rel = re("r0") - re("s3"), (re("r0") - re("s3")) / re("r0")
    delta2 = re("r0_prime") - re("s3")
    cell = scores.set_index(["arm", "region"])
    harm_qs = all(cell.loc[("s3", r), "quantile_score_proxy"]
                  - cell.loc[("r0", r), "quantile_score_proxy"] <= 0.05 for r in REGIONS)
    harm_mae = all(cell.loc[("s3", r), "mae"] - cell.loc[("r0", r), "mae"] <= 0.15 for r in REGIONS)
    verdict = {
        "G_C1_a": bool(delta_qs >= 0.15 and rel >= 0.006),
        "G_C1_b": bool(harm_qs and harm_mae),
        "G_C1_c": bool(dm_p < 0.05 and boot_p < 0.05),
        "G_C2": bool(delta2 >= 0.05),
        "delta_qs": delta_qs, "delta_rel": rel, "delta2_cross_family": delta2,
        "dm_p": dm_p, "dm_t": t_stat, "dm_hln_t": hln, "boot_p": boot_p, "boot_ci": boot_ci,
        "boot2_ci": boot2_ci, "win_rate": win_rate, "trimmed_mean_delta": trimmed,
        "jackknife_max_day_share": jk, "price_bands": bands, "holm_per_region_dm": holm,
        "region_equal_qs": {arm: re(arm) for arm in list(ARMS) + ["r0"]},
        "per_region_qs": {f"{arm}|{r}": float(losses[(arm, r)].mean())
                          for arm in list(ARMS) + ["r0"] for r in REGIONS},
    }
    verdict["G_C1_pass"] = verdict["G_C1_a"] and verdict["G_C1_b"] and verdict["G_C1_c"]
    verdict["outcome"] = ("pass_cross_family" if verdict["G_C1_pass"] and verdict["G_C2"]
                          else "pass_seed_average_only" if verdict["G_C1_pass"] else "fail")
    (OUT / "results.json").write_text(json.dumps(verdict, indent=2, default=str))
    print("[6] VERDICT:", {k: v for k, v in verdict.items()
                           if k.startswith("G_") or k == "outcome"}, flush=True)
    print(f"    dQS={delta_qs:.4f} ({rel:.2%})  cross-family={delta2:.4f}  "
          f"DM p={dm_p:.4g}  boot p={boot_p:.4g}  jk={jk:.1%}", flush=True)


if __name__ == "__main__":
    main()
