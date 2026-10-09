"""Are price spikes predictable from slowly building precursors?

For every origin and forecast hour, the label is whether the actual price
exceeds a threshold. Gradient-boosted classifiers see nested feature sets, all
point-in-time at the origin:

* ``cal``: calendar and lead only;
* ``P``: calendar plus recent price history (levels, recent spikes, same hour
  yesterday and last week);
* ``F``: calendar plus the fundamentals snapshot of the target hour (PD PASA
  spare capacity, demand and available capacity, derived reserve margins, gas);
* ``S``: ``F`` plus the predispatch snapshot (price, demand, available
  generation, net interchange);
* revisions ``R``: how the PD PASA and predispatch values for the same target
  hour changed over the preceding 3, 6 and 12 hours of runs;
* process ``T``: slow trajectories of the drivers (realized demand, the
  margins and predispatch prices of the hours just passed, gas trend) and the
  shape of the forecast margin across the next 24 hours.

If the process features (``R``, ``T``) add precision-recall skill over the
snapshot ``S`` on the 2023-2024 test split, spikes have a learnable slow
precursor. Skill gains are tested with a week-block bootstrap, and episode
recall is reported at a fixed false-alarm rate by lead time.

Usage::

    python -m forecasting.spike_predictability --output logs/forecasting/spike_predictability
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PRICE_DIR = ROOT.parent / "electricity-price-forecasting-research/data/multi_market_energy_reserve/aemo_nem"
EXO = ROOT / "data/aemo_exogenous"
REGIONS = ("NSW1", "QLD1", "TAS1")
THRESHOLDS = (300.0, 1000.0)
HORIZON = 24
LEAD_BUCKETS = {"0-5": range(0, 6), "6-11": range(6, 12), "12-17": range(12, 18), "18-23": range(18, 24)}
TRAIN_END, VAL_END = "2022-10-01", "2023-01-01"
REVISION_LAGS = (3, 6, 12)
EPISODE_GAP_HOURS = 6
ALARM_RATE = 0.01
MIN_HISTORY = 168


def _lag(array: np.ndarray, k: int) -> np.ndarray:
    """Value published k origins earlier for the same target hour (NaN beyond its horizon)."""
    out = np.full_like(array, np.nan)
    out[k:, : HORIZON - k] = array[:-k, k:]
    return out


def _trail(series: np.ndarray, window: int, how: str) -> np.ndarray:
    """Statistic of the window ending at the previous row (the latest completed hour)."""
    rolled = pd.Series(series).shift(1).rolling(window, min_periods=1)
    return getattr(rolled, how)().to_numpy()


def load_region(region: str) -> dict:
    frame = pd.read_csv(PRICE_DIR / f"{region.lower()}_hourly.csv")
    start = pd.to_datetime(frame["delivery_start_aest"], utc=True)
    unix = ((start - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).to_numpy()
    rows = pd.Series(np.arange(len(frame)), index=unix)
    count = len(frame)

    def per_origin(name: str, fields: tuple[str, ...]) -> dict:
        archive = np.load(EXO / f"{region.lower()}_{name}.npz")
        position = rows.reindex(archive["forecast_origin_unix"]).to_numpy()
        keep = ~np.isnan(position)
        position = position[keep].astype(int)
        out = {}
        for field in fields:
            dense = np.full((count, HORIZON), np.nan, dtype=np.float32)
            dense[position] = archive[field][keep]
            out[field] = dense
        return out

    pasa = per_origin("pdpasa", ("max_spare_capacity_mw", "demand10_mw", "demand50_mw", "demand90_mw",
                                 "uigf_mw", "available_capacity_mw"))
    pd_fields = per_origin("predispatch", ("predispatch_rrp", "predispatch_total_demand_mw",
                                           "predispatch_available_generation_mw",
                                           "predispatch_net_interchange_mw", "predispatch_covered"))
    gas = pd.read_csv(EXO / "raw_gas/dwgm_prices.csv")
    daily = gas.groupby("Gas_Date")["Price"].mean()
    daily.index = pd.to_datetime(daily.index)
    local = (start.dt.tz_convert("Australia/Brisbane").dt.tz_localize(None))
    gas_day = (local - pd.Timedelta(hours=6)).dt.normalize()
    gas_log = np.log(daily.rolling(7).mean().shift(1).reindex(gas_day).to_numpy())
    return {
        "frame": frame, "local": local, "price": frame["rrp_aud_per_mwh"].to_numpy(np.float64),
        "demand": frame["total_demand_mw"].to_numpy(np.float64), "gas": gas_log, **pasa, **pd_fields,
    }


def build_features(data: dict) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Return feature arrays of shape (rows, 24) and the feature names of each group."""
    count = len(data["price"])
    price, demand, gas = data["price"], data["demand"], data["gas"]
    local = data["local"]
    lead = np.broadcast_to(np.arange(HORIZON, dtype=np.float32), (count, HORIZON))
    target_time = local.to_numpy()[:, None] + pd.to_timedelta(np.arange(HORIZON), unit="h").to_numpy()[None, :]
    target = pd.DatetimeIndex(target_time.reshape(-1))
    doy = target.dayofyear.to_numpy().reshape(count, HORIZON)
    wide = lambda vector: np.broadcast_to(np.asarray(vector, dtype=np.float32)[:, None], (count, HORIZON))

    spare, d10, d50, d90 = (data[key] for key in ("max_spare_capacity_mw", "demand10_mw", "demand50_mw", "demand90_mw"))
    uigf, avail = data["uigf_mw"], data["available_capacity_mw"]
    netload, margin50, margin90 = d50 - uigf, avail - d50, avail - d90
    pd_rrp = data["predispatch_rrp"]
    pd_demand = data["predispatch_total_demand_mw"]
    pd_gen = data["predispatch_available_generation_mw"]
    pd_net = data["predispatch_net_interchange_mw"]

    features: dict[str, np.ndarray] = {}
    groups: dict[str, list[str]] = {}

    def add(group: str, name: str, values: np.ndarray) -> None:
        features[name] = np.asarray(values, dtype=np.float32)
        groups.setdefault(group, []).append(name)

    add("cal", "lead", lead)
    add("cal", "hour", target.hour.to_numpy().reshape(count, HORIZON))
    add("cal", "dow", target.dayofweek.to_numpy().reshape(count, HORIZON))
    add("cal", "month", target.month.to_numpy().reshape(count, HORIZON))
    add("cal", "doy_sin", np.sin(2 * np.pi * doy / 365.25))
    add("cal", "doy_cos", np.cos(2 * np.pi * doy / 365.25))
    add("gas", "gas", wide(gas))

    for name, values in (("spare", spare), ("d10", d10), ("d50", d50), ("d90", d90), ("uigf", uigf),
                         ("avail", avail), ("netload", netload), ("margin50", margin50), ("margin90", margin90)):
        add("snapF", name, values)
    for name, values in (("pd_rrp", pd_rrp), ("pd_demand", pd_demand), ("pd_gen", pd_gen),
                         ("pd_net", pd_net), ("pd_covered", data["predispatch_covered"])):
        add("snapPD", name, values)

    index = np.arange(count)[:, None] + np.arange(HORIZON)[None, :]
    def at(series: np.ndarray, shift: int) -> np.ndarray:
        position = index - shift
        out = np.full((count, HORIZON), np.nan)
        valid = (position >= 0) & (position < count)
        out[valid] = series[position[valid]]
        return out

    spike_recent = (price > 300).astype(float)
    add("P", "price_last", wide(pd.Series(price).shift(1).to_numpy()))
    add("P", "price_max24", wide(_trail(price, 24, "max")))
    add("P", "price_max72", wide(_trail(price, 72, "max")))
    add("P", "price_mean24", wide(_trail(price, 24, "mean")))
    add("P", "spikes24", wide(_trail(spike_recent, 24, "sum")))
    add("P", "spikes72", wide(_trail(spike_recent, 72, "sum")))
    add("P", "price_day_before", at(price, 24))
    add("P", "price_week_before", at(price, 168))

    for k in REVISION_LAGS:
        add("RF", f"d_spare_{k}", spare - _lag(spare, k))
        add("RF", f"d_netload_{k}", netload - _lag(netload, k))
        add("RF", f"d_margin50_{k}", margin50 - _lag(margin50, k))
        add("RPD", f"d_pd_rrp_{k}", pd_rrp - _lag(pd_rrp, k))
        add("RPD", f"d_pd_demand_{k}", pd_demand - _lag(pd_demand, k))
        add("RPD", f"d_pd_gen_{k}", pd_gen - _lag(pd_gen, k))

    margin0, spare0, pd0 = margin50[:, 0].astype(np.float64), spare[:, 0].astype(np.float64), pd_rrp[:, 0].astype(np.float64)
    day = lambda s, how: _trail(s, 24, how)
    add("TF", "demand_last", wide(pd.Series(demand).shift(1).to_numpy()))
    add("TF", "demand_mean24", wide(day(demand, "mean")))
    add("TF", "demand_max72", wide(_trail(demand, 72, "max")))
    add("TF", "demand_trend", wide(_trail(demand, 6, "mean") - pd.Series(_trail(demand, 6, "mean")).shift(24).to_numpy()))
    add("TF", "demand_peak_change", wide(day(demand, "max") - pd.Series(day(demand, "max")).shift(24).to_numpy()))
    add("TF", "margin_min24", wide(day(margin0, "min")))
    add("TF", "margin_mean24", wide(day(margin0, "mean")))
    add("TF", "margin_trend", wide(day(margin0, "mean") - pd.Series(day(margin0, "mean")).shift(24).to_numpy()))
    add("TF", "spare_min72", wide(_trail(spare0, 72, "min")))
    add("TF", "gas_trend", wide(gas - pd.Series(gas).shift(168).to_numpy()))
    with np.errstate(all="ignore"):
        add("TF", "margin_ahead_min", wide(np.nanmin(margin50, axis=1)))
        add("TF", "margin_ahead_argmin", wide(np.nan_to_num(np.nanargmin(np.where(np.isnan(margin50), np.inf, margin50), axis=1))))
        add("TF", "margin_ahead_mean", wide(np.nanmean(margin50, axis=1)))
        add("TF", "margin_vs_ahead_min", margin50 - features["margin_ahead_min"])
        add("TF", "netload_ahead_max", wide(np.nanmax(netload, axis=1)))
        add("TF", "spare_ahead_min", wide(np.nanmin(spare, axis=1)))
    add("TPD", "pd_last_max24", wide(day(pd0, "max")))
    add("TPD", "pd_last_mean72", wide(_trail(pd0, 72, "mean")))
    add("TPD", "pd_last_gap24", wide(day(pd0 - price, "mean")))
    with np.errstate(all="ignore"):
        add("TPD", "pd_ahead_max", wide(np.nanmax(pd_rrp, axis=1)))
        add("TPD", "pd_ahead_mean", wide(np.nanmean(pd_rrp, axis=1)))
        add("TPD", "pd_vs_ahead_max", pd_rrp - features["pd_ahead_max"])
    return features, groups


FEATURE_SETS = {
    "cal": ("cal",),
    "P": ("cal", "gas", "P"),
    "F": ("cal", "gas", "snapF"),
    "F+R": ("cal", "gas", "snapF", "RF"),
    "F+T": ("cal", "gas", "snapF", "TF"),
    "F+R+T": ("cal", "gas", "snapF", "RF", "TF"),
    "S": ("cal", "gas", "snapF", "snapPD"),
    "S+P": ("cal", "gas", "snapF", "snapPD", "P"),
    "S+R": ("cal", "gas", "snapF", "snapPD", "RF", "RPD"),
    "S+T": ("cal", "gas", "snapF", "snapPD", "TF", "TPD"),
    "S+R+T": ("cal", "gas", "snapF", "snapPD", "RF", "RPD", "TF", "TPD"),
    "S+P+R+T": ("cal", "gas", "snapF", "snapPD", "P", "RF", "RPD", "TF", "TPD"),
}


def split_masks(data: dict) -> dict[str, np.ndarray]:
    count = len(data["price"])
    local = data["local"]
    origin = np.arange(count)
    complete = (origin >= MIN_HISTORY) & (origin + HORIZON <= count)
    valid = complete & ~np.isnan(data["max_spare_capacity_mw"][:, 0]) & ~np.isnan(data["predispatch_rrp"][:, 0])
    # a one-day gap keeps the 24 target hours of a training origin out of the next split
    ahead = local + pd.Timedelta(hours=HORIZON - 1)
    train = valid & (ahead < pd.Timestamp(TRAIN_END)).to_numpy()
    val = valid & (local >= pd.Timestamp(TRAIN_END)).to_numpy() & (ahead < pd.Timestamp(VAL_END)).to_numpy()
    test = valid & (local >= pd.Timestamp(VAL_END)).to_numpy()
    return {"train": train, "val": val, "test": test}


def labels(price: np.ndarray, threshold: float) -> np.ndarray:
    count = len(price)
    index = np.arange(count)[:, None] + np.arange(HORIZON)[None, :]
    out = np.zeros((count, HORIZON), dtype=np.float32)
    inside = index < count
    out[inside] = price[index[inside]] > threshold
    return out


def fit_predict(names: list[str], features: dict, label: np.ndarray, masks: dict, seed: int, device: str) -> np.ndarray:
    def stack(mask: np.ndarray) -> np.ndarray:
        return np.stack([features[name][mask] for name in names], axis=-1).reshape(-1, len(names))

    x_train, y_train = stack(masks["train"]), label[masks["train"]].reshape(-1)
    x_val, y_val = stack(masks["val"]), label[masks["val"]].reshape(-1)
    model = xgboost.XGBClassifier(
        tree_method="hist", device=device, n_estimators=1000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, eval_metric="aucpr",
        early_stopping_rounds=50, random_state=seed,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    out = np.full(label.shape, np.nan, dtype=np.float32)
    for split in ("val", "test"):
        out[masks[split]] = model.predict_proba(stack(masks[split]))[:, 1].reshape(-1, HORIZON)
    return out


def per_target_alarm_score(probability: np.ndarray, leads: range, mask: np.ndarray) -> np.ndarray:
    """Highest probability issued for each target hour by origins at the given leads."""
    count = probability.shape[0]
    score = np.full(count + HORIZON, -np.inf)
    for h in leads:
        origins = np.flatnonzero(mask)
        np.maximum.at(score, origins + h, probability[origins, h])
    return score[:count]


def episodes(spike_hours: np.ndarray) -> list[np.ndarray]:
    if len(spike_hours) == 0:
        return []
    cuts = np.flatnonzero(np.diff(spike_hours) > EPISODE_GAP_HOURS) + 1
    return np.split(spike_hours, cuts)


def episode_recall(probability: np.ndarray, label: np.ndarray, masks: dict, price: np.ndarray, threshold: float) -> dict:
    count = len(price)
    actual = price > threshold
    out = {}
    for bucket, leads in LEAD_BUCKETS.items():
        score_val = per_target_alarm_score(probability, leads, masks["val"])
        score_test = per_target_alarm_score(probability, leads, masks["test"])
        in_val = np.isfinite(score_val) & ~actual
        cut = np.quantile(score_val[in_val], 1 - ALARM_RATE) if in_val.any() else np.inf
        in_test = np.isfinite(score_test)
        spikes = np.flatnonzero(actual & in_test)
        found = [bool((score_test[ep] >= cut).any()) for ep in episodes(spikes)]
        quiet = in_test & ~actual
        out[bucket] = {
            "episodes": len(found), "detected": int(sum(found)),
            "recall": float(np.mean(found)) if found else None,
            "alarm_rate_test": float((score_test[quiet] >= cut).mean()),
        }
    return out


def block_bootstrap(prob_a: np.ndarray, prob_b: np.ndarray, label: np.ndarray, mask: np.ndarray,
                    local: pd.Series, draws: int, seed: int) -> dict:
    origins = np.flatnonzero(mask)
    week = ((local.iloc[origins] - pd.Timestamp(VAL_END)).dt.days // 7).to_numpy()
    week_ids = np.unique(week)
    index = np.searchsorted(week_ids, week)
    y = label[origins].reshape(-1)
    a, b = prob_a[origins].reshape(-1), prob_b[origins].reshape(-1)
    block = np.repeat(index, HORIZON)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        weights = np.bincount(rng.integers(0, len(week_ids), len(week_ids)), minlength=len(week_ids))[block]
        keep = weights > 0
        differences.append(
            average_precision_score(y[keep], b[keep], sample_weight=weights[keep])
            - average_precision_score(y[keep], a[keep], sample_weight=weights[keep])
        )
    differences = np.asarray(differences)
    return {"mean": float(differences.mean()), "lo": float(np.quantile(differences, 0.025)),
            "hi": float(np.quantile(differences, 0.975))}


def evaluate(probability: np.ndarray, label: np.ndarray, mask: np.ndarray) -> dict:
    y, p = label[mask].reshape(-1), probability[mask].reshape(-1)
    out = {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
           "brier": float(np.mean((p - y) ** 2)), "base_rate": float(y.mean())}
    lead_y, lead_p = label[mask], probability[mask]
    out["pr_auc_by_lead"] = {
        bucket: float(average_precision_score(lead_y[:, list(leads)].reshape(-1), lead_p[:, list(leads)].reshape(-1)))
        for bucket, leads in LEAD_BUCKETS.items()
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=ROOT / "logs/forecasting/spike_predictability")
    parser.add_argument("--regions", nargs="+", default=list(REGIONS))
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(THRESHOLDS))
    parser.add_argument("--sets", nargs="+", default=list(FEATURE_SETS))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    for region in args.regions:
        data = load_region(region)
        features, groups = build_features(data)
        masks = split_masks(data)
        print(region, {k: int(v.sum()) for k, v in masks.items()}, flush=True)
        for threshold in args.thresholds:
            label = labels(data["price"], threshold)
            key = f"{region}|{int(threshold)}"
            results[key] = {"positives": {s: float(label[m].sum()) for s, m in masks.items()}, "sets": {}}
            predictions = {}
            for set_name in args.sets:
                names = [n for group in FEATURE_SETS[set_name] for n in groups[group]]
                probability = fit_predict(names, features, label, masks, args.seed, args.device)
                predictions[set_name] = probability
                entry = evaluate(probability, label, masks["test"])
                entry["episodes"] = episode_recall(probability, label, masks, data["price"], threshold)
                entry["features"] = len(names)
                results[key]["sets"][set_name] = entry
                print(key, set_name, round(entry["pr_auc"], 4), round(entry["roc_auc"], 4), flush=True)
            reference = "S" if "S" in predictions else next(iter(predictions))
            results[key]["bootstrap_vs_" + reference] = {
                s: block_bootstrap(predictions[reference], p, label, masks["test"], data["local"], args.bootstrap, args.seed)
                for s, p in predictions.items() if s != reference
            }
            np.savez_compressed(args.output / f"predictions_{region}_{int(threshold)}.npz", **predictions)
            (args.output / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
