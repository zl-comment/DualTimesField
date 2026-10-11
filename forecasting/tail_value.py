"""Plan 3: tail scores and decision value of the stored forecasts, with block-bootstrap intervals.

* Tail scores for the forecasters that give quantiles (levels 0.05, 0.10, 0.50, 0.90, 0.95): pinball loss of q90 and q95
  over all hours, over spike hours (actual > 300) and over negative-price hours, and the coverage of the 5-95% interval.
* Decision value: a one-battery arbitrage schedule. At each midnight (AEST) the 24 forecast hourly prices of the origin
  define a linear program (power 1 MW, energy 1, 2 or 4 MWh, round-trip efficiency 0.9, starts at half energy and must
  end at or above it); the schedule is settled at the realized prices. Reported as revenue and as the share of the
  perfect-foresight revenue, pooled over the days of each region and averaged over regions.
* Paired differences between forecasters with week-block bootstrap intervals (blocks of 7 consecutive days for revenue,
  of 168 consecutive origins for the tail scores).

    python -m forecasting.tail_value --period dev --output logs/forecasting/tail_value
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

REGIONS = ("NSW1", "QLD1", "TAS1")
LEVELS = [0.05, 0.10, 0.50, 0.90, 0.95]
SEEDS = (2026, 2027, 2028)
MEMBERS = Path("outputs/forecasting/members")
HOLD = Path("outputs/forecasting/members_hold")
XGB_BASELINE = Path("outputs/forecasting/baselines_rolling_pd/raw/xgboost")
SIGNIFICANCE = Path("outputs/forecasting/significance_inputs")
EFFICIENCY = 0.9 ** 0.5


def _stack(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts)


def load_dev(region: str) -> dict:
    """Point and (where available) quantile forecasts of every forecaster on 2023-2024, seed-averaged."""
    out = {}
    with np.load(XGB_BASELINE / "seed2026" / f"{region}.npz") as a:
        actual, origin = a["actual"], a["origin_unix"]
    xs = [np.load(XGB_BASELINE / f"seed{s}" / f"{region}.npz") for s in SEEDS]
    out["xgb_default"] = {"point": np.mean([x["point"] for x in xs], 0), "quantile": np.mean([x["quantile"] for x in xs], 0)}
    for name in ("nn_c_mae_v1", "nn_dual_b"):
        with np.load(MEMBERS / name / f"{region}.npz") as a:
            out[name] = {"point": a["point"].mean(0), "quantile": a["quantile"].mean(0)}
    for name in ("xgb_profile_l1", "lgbm_profile_reg"):
        parts = [np.load(MEMBERS / name / f"q{k}" / f"{region}.npz") for k in range(8)]
        out[name] = {"point": _stack([p["test_point"] for p in parts]), "quantile": None}
    for label, folder in (("xgb_base_tails", "rolling_xgb_base_tails"), ("xgb_base_raw_tails", "rolling_xgb_base_raw_tails")):
        with np.load(SIGNIFICANCE / folder / f"{region}.npz") as a:
            out[label] = {"point": a["point"].mean(0), "quantile": a["quantile"].mean(0)}
    return {"forecasts": out, "actual": actual, "origin": origin}


def load_hold(region: str) -> dict:
    out = {}
    parts = [np.load(HOLD / "xgb_default" / f"q{k}" / f"{region}.npz") for k in range(3)]
    actual = _stack([p["test_actual"] for p in parts]); origin = _stack([p["test_origin_unix"] for p in parts])
    out["xgb_default"] = {"point": _stack([p["test_point"] for p in parts]), "quantile": _stack([p["test_quantile"] for p in parts])}
    for name in ("nn_c_mae_v1", "nn_dual_b"):
        with np.load(HOLD / name / f"{region}.npz") as a:
            out[name] = {"point": a["point"].mean(0), "quantile": a["quantile"].mean(0)}
    for name in ("xgb_profile_l1", "lgbm_profile_reg"):
        parts = [np.load(HOLD / name / f"q{k}" / f"{region}.npz") for k in range(3)]
        out[name] = {"point": _stack([p["test_point"] for p in parts]), "quantile": None}
    return {"forecasts": out, "actual": actual, "origin": origin}


def add_combinations(forecasts: dict, hold: bool) -> None:
    point = lambda names: np.mean([forecasts[n]["point"] for n in names], 0)
    for name, members in (("ens_xgb_lgbm", ["xgb_profile_l1", "lgbm_profile_reg"]),
                          ("ens_xgb_lgbm_C", ["xgb_profile_l1", "lgbm_profile_reg", "nn_c_mae_v1"]),
                          ("ens_xgb_lgbm_B", ["xgb_profile_l1", "lgbm_profile_reg", "nn_dual_b"])):
        forecasts[name] = {"point": point(members), "quantile": None}
    for name, members in (("qens_xgb_C", ["xgb_default", "nn_c_mae_v1"]), ("qens_xgb_B", ["xgb_default", "nn_dual_b"])):
        forecasts[name] = {"point": point(members), "quantile": np.mean([forecasts[n]["quantile"] for n in members], 0)}


def pinball(quantile: np.ndarray, actual: np.ndarray, level: float) -> np.ndarray:
    """Per-origin mean pinball loss of one level, [N] (mean over the 24 hours)."""
    error = actual - quantile[..., LEVELS.index(level)]
    return np.maximum(level * error, (level - 1) * error)


def tail_scores(forecast: dict, actual: np.ndarray) -> dict:
    q = forecast["quantile"]
    scores = {}
    spike, negative = actual > 300, actual < 0
    for level in (0.90, 0.95):
        loss = pinball(q, actual, level)
        scores[f"q{int(level * 100)}_all"] = loss.mean(1)
        for tag, mask in (("spike", spike), ("negative", negative)):
            # per-origin mean over the event hours of the horizon; origins without events contribute nothing
            count = mask.sum(1)
            scores[f"q{int(level * 100)}_{tag}"] = np.where(count > 0, (loss * mask).sum(1) / np.maximum(count, 1), np.nan)
    scores["covered_90"] = ((actual >= q[..., LEVELS.index(0.05)]) & (actual <= q[..., LEVELS.index(0.95)])).mean(1)
    return scores


def battery(forecast_price: np.ndarray, real_price: np.ndarray, energy: float, power: float = 1.0) -> float:
    """Revenue at realized prices of the LP schedule that is optimal for the forecast prices."""
    n = 24
    # variables: charge[24], discharge[24]; soc_t = e0 + sum_{s<=t}(eta*c_s - d_s/eta)
    cumulative = np.tril(np.ones((n, n)))
    soc_matrix = np.hstack([EFFICIENCY * cumulative, -cumulative / EFFICIENCY])
    e0 = energy / 2
    a_ub = np.vstack([soc_matrix, -soc_matrix])
    b_ub = np.concatenate([np.full(n, energy - e0), np.full(n, e0)])
    # end at or above the starting energy
    a_ub = np.vstack([a_ub, -soc_matrix[-1:]])
    b_ub = np.concatenate([b_ub, [0.0]])
    cost = np.concatenate([forecast_price, -forecast_price])
    result = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=[(0, power)] * (2 * n), method="highs")
    x = result.x
    return float(np.dot(real_price, x[n:] - x[:n]))


def daily_revenue(forecast: np.ndarray, actual: np.ndarray, origin: np.ndarray, energies=(1.0, 2.0, 4.0)) -> dict:
    """Revenue per midnight-origin day for each battery size: forecast or perfect foresight."""
    hour = pd.to_datetime(origin + 36000, unit="s").hour.to_numpy()
    days = np.flatnonzero(hour == 0)
    out = {}
    for e in energies:
        out[e] = np.array([battery(forecast[i], actual[i], e) for i in days])
    return out, days


def block_bootstrap(difference: np.ndarray, block: int, draws: int = 4000, seed: int = 0) -> list[float]:
    d = difference[np.isfinite(difference)]
    blocks = len(d) // block
    sums = d[: blocks * block].reshape(blocks, block).mean(1)
    rng = np.random.default_rng(seed)
    boot = [sums[rng.integers(0, blocks, blocks)].mean() for _ in range(draws)]
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def run(period: str) -> dict:
    loader = load_dev if period == "dev" else load_hold
    data = {r: loader(r) for r in REGIONS}
    for r in REGIONS:
        add_combinations(data[r]["forecasts"], period == "hold")
    names = list(data[REGIONS[0]]["forecasts"])
    result = {"tails": {}, "battery": {}}
    # tails
    quantile_names = [n for n in names if data[REGIONS[0]]["forecasts"][n]["quantile"] is not None]
    tails = {n: {r: tail_scores(data[r]["forecasts"][n], data[r]["actual"]) for r in REGIONS} for n in quantile_names}
    for n in quantile_names:
        result["tails"][n] = {k: float(np.nanmean([np.nanmean(tails[n][r][k]) for r in REGIONS])) for k in tails[n][REGIONS[0]]}
    def paired_tail(a, b, key):
        d = np.concatenate([tails[a][r][key] - tails[b][r][key] for r in REGIONS])
        return {"difference": float(np.nanmean(d)), "ci95": block_bootstrap(d, 168)}
    result["tail_pairs"] = {f"{a}-{b}": {k: paired_tail(a, b, k) for k in ("q90_all", "q95_all", "q90_spike", "q95_spike", "q90_negative", "q95_negative")}
                            for a, b in (("nn_dual_b", "nn_c_mae_v1"), ("qens_xgb_B", "qens_xgb_C"), ("xgb_base_tails", "xgb_base_raw_tails"),
                                         ("qens_xgb_C", "xgb_default"), ("nn_c_mae_v1", "xgb_default"), ("xgb_base_tails", "xgb_default"))
                            if a in tails and b in tails}
    # battery
    revenue = {}
    for r in REGIONS:
        actual, origin = data[r]["actual"], data[r]["origin"]
        hour = pd.to_datetime(origin + 36000, unit="s").hour.to_numpy()
        day_index = np.flatnonzero(hour == 0)
        forecasts = {n: data[r]["forecasts"][n]["point"] for n in names}
        forecasts["perfect"] = actual
        # the 24 hours of the previous day, as realized (a naive benchmark); the first day has none and is skipped by all
        naive = np.zeros_like(actual)
        naive[day_index[1:]] = actual[day_index[:-1]]
        forecasts["yesterday"] = naive
        for n, f in forecasts.items():
            rev, _ = daily_revenue(f, actual, origin)
            revenue.setdefault(n, {})[r] = {e: v[1:] for e, v in rev.items()}
    perfect = {r: revenue["perfect"][r] for r in REGIONS}
    for n in revenue:
        result["battery"][n] = {}
        for e in (1.0, 2.0, 4.0):
            share = np.mean([revenue[n][r][e].sum() / perfect[r][e].sum() for r in REGIONS])
            result["battery"][n][f"{int(e)}MWh_share_of_perfect"] = float(share)
            result["battery"][n][f"{int(e)}MWh_mean_daily_revenue"] = float(np.mean([revenue[n][r][e].mean() for r in REGIONS]))
    pairs = (("ens_xgb_lgbm_C", "lgbm_profile_reg"), ("ens_xgb_lgbm_B", "ens_xgb_lgbm_C"), ("nn_dual_b", "nn_c_mae_v1"),
             ("lgbm_profile_reg", "xgb_default"), ("ens_xgb_lgbm_C", "xgb_default"), ("xgb_default", "yesterday"))
    result["battery_pairs"] = {}
    for a, b in pairs:
        result["battery_pairs"][f"{a}-{b}"] = {}
        for e in (1.0, 2.0, 4.0):
            d = np.concatenate([revenue[a][r][e] - revenue[b][r][e] for r in REGIONS])
            result["battery_pairs"][f"{a}-{b}"][f"{int(e)}MWh"] = {"mean_daily_difference": float(d.mean()), "ci95": block_bootstrap(d, 7),
                                                                    "relative_to_b": float(d.mean() / np.mean(np.concatenate([revenue[b][r][e] for r in REGIONS])))}
    result["days_per_region"] = int(len(revenue["perfect"][REGIONS[0]][1.0]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--period", choices=("dev", "hold"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.period)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"tail_value_{args.period}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"days": result["days_per_region"], "battery_share_2MWh": {n: round(v["2MWh_share_of_perfect"], 3) for n, v in result["battery"].items()}}, indent=1))


if __name__ == "__main__":
    main()
