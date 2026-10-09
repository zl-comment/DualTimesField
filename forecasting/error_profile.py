"""Where does the absolute error accumulate? Per-target-hour error of stitched forecasts.

Every target hour is forecast from 24 origins; its error is the mean of those
24 absolute errors (and of the seeds). Daily means show the peaks over the
2023-2024 test period, hour-of-day means show the daily shape, and the
concentration table shows how much of the MAE sits in the worst hours.

    python -m forecasting.error_profile --model name=path-pattern ... --output DIR
Patterns contain ``{region}`` and ``{seed}`` is optional (as ``forecasting.significance``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REGIONS = ("NSW1", "QLD1", "TAS1")
SEEDS = (2026, 2027, 2028)
HORIZON = 24


def load_errors(pattern: str, region: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-origin absolute errors (N, 24) averaged over seeds, the actuals and the origin times."""
    errors = []
    if "{seed}" in pattern:
        files = [pattern.format(region=region, seed=s) for s in SEEDS]
    else:
        files = [pattern.format(region=region)]
    for file in files:
        with np.load(file) as archive:
            point, actual, origin = archive["point"], archive["actual"], archive["origin_unix"]
        for p in (point if point.ndim == 3 else point[None]):
            errors.append(np.abs(p - actual))
    return np.mean(errors, axis=0), actual, origin


def per_target_hour(error: np.ndarray, actual: np.ndarray, origin: np.ndarray):
    """Mean over leads of the error of each target hour, with the actual price and its time."""
    count = error.shape[0]
    total = np.zeros(count + HORIZON - 1)
    seen = np.zeros(count + HORIZON - 1)
    price = np.zeros(count + HORIZON - 1)
    for h in range(HORIZON):
        total[h:h + count] += error[:, h]
        seen[h:h + count] += 1
        price[h:h + count] = actual[:, h]
    time = origin[0] + 3600 * np.arange(count + HORIZON - 1)
    keep = seen == HORIZON  # hours forecast from all 24 leads
    return time[keep], (total / np.maximum(seen, 1))[keep], price[keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", action="append", required=True, help="name=pattern")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = dict(item.split("=", 1) for item in args.model)
    summary: dict = {}
    series: dict = {}
    for region in REGIONS:
        series[region] = {}
        for name, pattern in models.items():
            error, actual, origin = load_errors(pattern, region)
            time, hourly, price = per_target_hour(error, actual, origin)
            series[region][name] = (time, hourly, price)
            order = np.argsort(hourly)[::-1]
            cumulative = np.cumsum(hourly[order]) / hourly.sum()
            bands = {}
            for label, low, high in (("<=0", -np.inf, 0), ("0-100", 0, 100), ("100-300", 100, 300), (">300", 300, np.inf)):
                inside = (price > low) & (price <= high) if low > -np.inf else price <= high
                bands[label] = {"share_hours": float(inside.mean()), "share_of_mae": float(hourly[inside].sum() / hourly.sum()),
                                "mae": float(hourly[inside].mean())}
            summary.setdefault(region, {})[name] = {
                "mae_per_target_hour": float(hourly.mean()),
                "top_1pct_hours_share": float(cumulative[int(0.01 * len(hourly)) - 1]),
                "top_5pct_hours_share": float(cumulative[int(0.05 * len(hourly)) - 1]),
                "median_hour_error": float(np.median(hourly)),
                "by_price_band": bands,
            }
    (args.output / "error_profile.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(args.output / "error_profile_series.npz", **{
        f"{r}|{m}|{k}": v for r in series for m in series[r] for k, v in zip(("time", "error", "price"), series[r][m])})
    print(json.dumps({r: {m: {k: v for k, v in s.items() if k != "by_price_band"} for m, s in d.items()} for r, d in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
