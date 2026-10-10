"""Ensembles of members refit quarterly, with configurations and weights chosen without the test period.

Members live in ``outputs/forecasting/members/<name>`` as either stitched files ``<region>.npz`` (neural members written
by ``forecasting.price_floor``: ``point[S, N, 24]``, ``val_point[S, Nv, 24]``, ``val_refit``) or quarterly files
``q<k>/<region>.npz`` (``forecasting.xgb_member``: ``val_point``, ``test_point`` per refit). Seeds are averaged.

* Configuration choice: of several variants of one algorithm, the one with the lowest mean validation MAE over the
  validation quarters of all refits and the three regions.
* Equal weights, and per-refit weights fitted on that refit's own validation quarter on a coarse simplex grid and shrunk
  halfway toward equal weights. The test quarter is never used to pick anything.
* Test MAE per region and mean, Diebold-Mariano against the best single member (Newey-West, 48 lags, three-region mean
  loss per origin) and a week-block bootstrap of the MAE difference.

    python -m forecasting.ensemble --members xgb_profile xgb_reg ... --pools xgb=xgb_profile,xgb_reg --output DIR
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .significance import diebold_mariano

ROOT = Path("outputs/forecasting/members")
REGIONS = ("NSW1", "QLD1", "TAS1")
QUARTERS = 8


def load_member(name: str, region: str, quarters: int = QUARTERS) -> dict:
    stitched = ROOT / name / f"{region}.npz"
    if stitched.exists():
        with np.load(stitched) as a:
            return {"val_point": a["val_point"].mean(0), "val_actual": a["val_actual"], "val_refit": a["val_refit"],
                    "val_origin": a["val_origin_unix"], "test_point": a["point"].mean(0), "test_actual": a["actual"],
                    "test_origin": a["origin_unix"]}
    parts = [np.load(ROOT / name / f"q{k}" / f"{region}.npz") for k in range(quarters)]
    return {"val_point": np.concatenate([p["val_point"] for p in parts]),
            "val_actual": np.concatenate([p["val_actual"] for p in parts]),
            "val_refit": np.concatenate([np.full(len(p["val_point"]), k) for k, p in enumerate(parts)]),
            "val_origin": np.concatenate([p["val_origin_unix"] for p in parts]),
            "test_point": np.concatenate([p["test_point"] for p in parts]),
            "test_actual": np.concatenate([p["test_actual"] for p in parts]),
            "test_origin": np.concatenate([p["test_origin_unix"] for p in parts])}


def simplex_grid(count: int, step: float) -> list[np.ndarray]:
    steps = int(round(1 / step))
    return [np.array(c) / steps for c in itertools.product(range(steps + 1), repeat=count) if sum(c) == steps]


def fit_weights(val_stack: np.ndarray, actual: np.ndarray, step: float) -> np.ndarray:
    """Grid-search the simplex for the lowest validation MAE, then shrink halfway toward equal weights."""
    count = val_stack.shape[0]
    best = min(simplex_grid(count, step), key=lambda w: np.abs(np.tensordot(w, val_stack, axes=1) - actual).mean())
    return 0.5 * best + 0.5 / count


def evaluate(members: list[str], data: dict, step: float = 0.1, refits: int = QUARTERS) -> dict:
    """Test forecasts of the equal-weight and validation-weighted ensembles, per region."""
    out = {"equal": {}, "weighted": {}, "weights": {}}
    for region in REGIONS:
        loaded = [data[m][region] for m in members]
        for other in loaded[1:]:
            assert np.array_equal(loaded[0]["test_origin"], other["test_origin"]), "test origins differ"
        out["equal"][region] = np.mean([m["test_point"] for m in loaded], axis=0)
        weighted = np.zeros_like(out["equal"][region])
        weights = []
        test_refit = np.concatenate([np.full(n, k) for k, n in enumerate(_refit_sizes(loaded[0], refits))])
        for k in range(refits):
            vmask = loaded[0]["val_refit"] == k
            stack = np.stack([m["val_point"][vmask] for m in loaded])
            w = fit_weights(stack, loaded[0]["val_actual"][vmask], step)
            weights.append(w.round(3).tolist())
            tmask = test_refit == k
            weighted[tmask] = np.tensordot(w, np.stack([m["test_point"][tmask] for m in loaded]), axes=1)
        out["weighted"][region] = weighted
        out["weights"][region] = weights
    return out


def _refit_sizes(member: dict, refits: int) -> list[int]:
    """Test origins per refit from the test quarter boundaries (origin times)."""
    import pandas as pd
    stamps = pd.to_datetime(member["test_origin"] + 36000, unit="s")
    quarter = (stamps.year.to_numpy() * 4 + (stamps.month.to_numpy() - 1) // 3)
    codes = np.unique(quarter)
    assert len(codes) == refits, f"{len(codes)} test quarters, expected {refits}"
    return [int((quarter == c).sum()) for c in codes]


def test_mae(forecasts: dict, data: dict, member: str) -> dict:
    mae = {r: float(np.abs(forecasts[r] - data[member][r]["test_actual"]).mean()) for r in REGIONS}
    mae["mean"] = float(np.mean([mae[r] for r in REGIONS]))
    return mae


def compare(a: dict, b: dict, data: dict, member: str, draws: int = 2000, seed: int = 0) -> dict:
    """Three-region mean per-origin MAE difference a - b: Diebold-Mariano and a week-block bootstrap."""
    loss = lambda f: np.mean([np.abs(f[r] - data[member][r]["test_actual"]).mean(axis=1) for r in REGIONS], axis=0)
    d = loss(a) - loss(b)
    result = diebold_mariano(d, 48)
    week = np.arange(len(d)) // (24 * 7)
    sums, counts = np.bincount(week, d), np.bincount(week)
    rng = np.random.default_rng(seed)
    boot = [sums[i].sum() / counts[i].sum() for i in (rng.integers(0, len(sums), len(sums)) for _ in range(draws))]
    return {"difference": float(d.mean()), "p_value": float(result["p_value"]),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]}


def validation_mae(name: str, data: dict) -> float:
    return float(np.mean([np.abs(data[name][r]["val_point"] - data[name][r]["val_actual"]).mean() for r in REGIONS]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--members", nargs="+", required=True, help="all member directories to load")
    parser.add_argument("--pools", nargs="*", default=[], help="family=variant1,variant2: pick the best variant by validation")
    parser.add_argument("--singles", nargs="*", default=[], help="members that stand alone in the ensembles")
    parser.add_argument("--ensembles", nargs="+", required=True, help="name=member1+member2 (family names allowed)")
    parser.add_argument("--quarters", type=int, default=QUARTERS)
    parser.add_argument("--root", type=Path, default=ROOT, help="directory of the member outputs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    globals()["ROOT"] = args.root
    data = {m: {r: load_member(m, r, args.quarters) for r in REGIONS} for m in args.members}
    chosen, report = {}, {"validation_mae": {m: validation_mae(m, data) for m in args.members}, "chosen": {}}
    for pool in args.pools:
        family, variants = pool.split("=")
        best = min(variants.split(","), key=lambda v: report["validation_mae"][v])
        chosen[family] = best
        report["chosen"][family] = best
    resolve = lambda m: chosen.get(m, m)
    reference = next(iter(data))
    singles = {m: {r: data[m][r]["test_point"] for r in REGIONS} for m in args.members}
    report["single_test_mae"] = {m: test_mae(singles[m], data, m) for m in args.members}
    report["ensembles"] = {}
    best_single = min(args.members, key=lambda m: report["validation_mae"][m])
    for spec in args.ensembles:
        name, parts = spec.split("=")
        members = [resolve(p) for p in parts.split("+")]
        result = evaluate(members, data, step=0.1 if len(members) > 2 else 0.05, refits=args.quarters)
        entry = {"members": members, "weights_per_refit": result["weights"]}
        for kind in ("equal", "weighted"):
            entry[kind] = test_mae(result[kind], data, members[0])
            entry[kind + "_vs_best_member"] = {m: compare(result[kind], singles[m], data, members[0])
                                               for m in members}
        report["ensembles"][name] = entry
        print(name, members, "equal %.3f weighted %.3f" % (entry["equal"]["mean"], entry["weighted"]["mean"]), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ensemble_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
