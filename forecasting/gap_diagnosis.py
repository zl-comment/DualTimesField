"""Where does one forecaster lose to another? Error-gap diagnosis by condition.

Both forecasters are seed ensembles (``point`` averaged over seeds; patterns as
in ``significance``). For every test (origin, hour) the absolute errors of the
two are compared, and the total gap ``sum(|e_first|) - sum(|e_second|)`` is
split over bins of one condition at a time: actual-price band, hour of day of
the delivered hour, forecast horizon, quarter, the predispatch price for that
hour, predispatch coverage, PD PASA net load and spare capacity. Each bin also
reports both forecasters' mean signed error (bias).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = (2026, 2027, 2028)
AEST = pd.Timedelta(hours=10)


def _ensemble(pattern: str, region: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = [pattern.format(region=region, seed=s) for s in SEEDS] if "{seed}" in pattern else [pattern.format(region=region)]
    points, actual, origins = [], None, None
    for path in paths:
        with np.load(path) as archive:
            point = archive["point"]
            points.extend(point if point.ndim == 3 else [point])
            actual, origins = archive["actual"], archive["origin_unix"]
    return np.mean(points, axis=0), actual, origins


def _by_origin(path: Path, names: list[str], origins: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        available = archive["forecast_origin_unix"]
        position = np.searchsorted(available, origins)
        if not np.array_equal(available[position], origins):
            raise ValueError(f"{path} lacks some test origins")
        return {name: archive[name][position] for name in names}


def diagnose(first: str, second: str, region: str, data_dir: Path) -> dict:
    a, actual, origins = _ensemble(first, region)
    b, actual_b, origins_b = _ensemble(second, region)
    if not (np.array_equal(origins, origins_b) and np.allclose(actual, actual_b)):
        raise ValueError("forecasters are not aligned")
    horizon = actual.shape[1]
    delivered = pd.to_datetime((origins[:, None] + 3600 * np.arange(horizon)[None, :]).ravel(), unit="s") + AEST
    predispatch = _by_origin(data_dir / f"{region.lower()}_predispatch.npz", ["predispatch_rrp", "predispatch_covered"], origins)
    pdpasa = _by_origin(data_dir / f"{region.lower()}_pdpasa.npz", ["max_spare_capacity_mw", "demand50_mw", "uigf_mw"], origins)
    frame = pd.DataFrame({
        "actual": actual.ravel(),
        "error_first": (a - actual).ravel(),
        "error_second": (b - actual).ravel(),
        "hour_of_day": delivered.hour,
        "horizon": np.tile(np.arange(1, horizon + 1), len(origins)),
        "quarter": delivered.to_period("Q").astype(str),
        "predispatch_rrp": predispatch["predispatch_rrp"].ravel(),
        "covered": predispatch["predispatch_covered"].ravel().astype(int),
        "net_load_mw": (pdpasa["demand50_mw"] - pdpasa["uigf_mw"]).ravel(),
        "spare_mw": pdpasa["max_spare_capacity_mw"].ravel(),
    })
    frame["gap"] = frame.error_first.abs() - frame.error_second.abs()
    total = float(frame.gap.sum())
    bins = {
        "actual_band": pd.cut(frame.actual, [-np.inf, 0, 50, 100, 200, 300, np.inf]),
        "hour_of_day": pd.cut(frame.hour_of_day, [-1, 5, 9, 15, 19, 23], labels=["0-5", "6-9", "10-15", "16-19", "20-23"]),
        "horizon": pd.cut(frame.horizon, [0, 6, 12, 18, 24], labels=["1-6", "7-12", "13-18", "19-24"]),
        "quarter": frame.quarter,
        "predispatch_rrp": pd.cut(frame.predispatch_rrp, [-np.inf, 0, 50, 100, 200, 300, 1000, np.inf]),
        "covered": frame.covered,
        "net_load_quintile": pd.qcut(frame.net_load_mw, 5, labels=["q1 low", "q2", "q3", "q4", "q5 high"]),
        "spare_quintile": pd.qcut(frame.spare_mw, 5, labels=["q1 scarce", "q2", "q3", "q4", "q5 ample"]),
    }
    result = {"region": region, "mae_first": float(frame.error_first.abs().mean()),
              "mae_second": float(frame.error_second.abs().mean()), "total_gap": total, "conditions": {}}
    for name, key in bins.items():
        grouped = frame.groupby(key, observed=True)
        table = pd.DataFrame({
            "share_of_hours": grouped.size() / len(frame),
            "mae_first": grouped.error_first.apply(lambda e: e.abs().mean()),
            "mae_second": grouped.error_second.apply(lambda e: e.abs().mean()),
            "bias_first": grouped.error_first.mean(),
            "bias_second": grouped.error_second.mean(),
            "share_of_gap": grouped.gap.sum() / total,
        })
        result["conditions"][name] = {str(index): {k: float(v) for k, v in row.items()} for index, row in table.iterrows()}
    return result


def markdown(result: dict, first_name: str, second_name: str) -> str:
    lines = [f"{result['region']}: MAE {first_name} {result['mae_first']:.2f}, {second_name} {result['mae_second']:.2f}", ""]
    for name, table in result["conditions"].items():
        lines += [f"| {name} | hours | MAE {first_name} | MAE {second_name} | bias {first_name} | bias {second_name} | share of gap |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for index, row in table.items():
            lines.append(f"| {index} | {100 * row['share_of_hours']:.1f}% | {row['mae_first']:.2f} | {row['mae_second']:.2f} | "
                         f"{row['bias_first']:+.2f} | {row['bias_second']:+.2f} | {100 * row['share_of_gap']:+.0f}% |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True, help="name=pattern")
    parser.add_argument("--second", required=True, help="name=pattern")
    parser.add_argument("--region", default="QLD1")
    parser.add_argument("--data-dir", type=Path, default=Path("data/aemo_exogenous"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    first_name, first = args.first.split("=", 1)
    second_name, second = args.second.split("=", 1)
    result = diagnose(first, second, args.region, args.data_dir)
    text = markdown(result, first_name, second_name)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".json").write_text(json.dumps(result, indent=2))
        args.output.with_suffix(".md").write_text(text)


if __name__ == "__main__":
    main()
