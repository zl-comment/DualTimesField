"""Diebold-Mariano tests between forecasting methods on the test split.

For every forecast origin the loss is the mean absolute error over the 24
horizons. A method with several seeds contributes the mean of its per-seed
losses, so the comparison is between methods rather than single runs. The
loss differential ``d_t = L_A(t) - L_B(t)`` is tested for zero mean with a
Newey-West (Bartlett) long-run variance, because rolling hourly origins
overlap and are strongly autocorrelated. Tests are run per region and on the
three-region mean differential at each origin time.

With ``--loss crps`` the per-origin loss is CRPS~ (twice the mean pinball loss
over the five quantiles of ``quantile``, averaged over the 24 horizons); a
method with several seeds then uses its seed-averaged quantiles.

Tail losses restrict the per-origin loss to event hours within the horizon:
``crps_negative`` and ``q05_negative`` (CRPS~ and the 0.05-quantile pinball
loss on hours with a negative price), ``crps_spike`` and ``q95_spike`` (on hours
above 300 AUD/MWh). Origins without such hours are dropped; they are the same
for every method, since the actual prices are shared.

Sources are ``name=pattern`` where ``pattern`` is an ``.npz`` path containing
``{region}`` and optionally ``{seed}``; each file holds ``point`` (``[N, 24]``
or ``[S, N, 24]``), ``actual``, and ``origin_unix``. The name ``naive`` with a
file that stores ``naive`` uses that array as the forecast.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REGIONS = ("NSW1", "QLD1", "TAS1")
SEEDS = (2026, 2027, 2028)
LEVELS = np.array([0.05, 0.10, 0.50, 0.90, 0.95])


TAIL_LOSSES = {
    # name: (event mask on actual prices, quantile level index or None for CRPS~)
    "crps_negative": (lambda actual: actual < 0, None),
    "q05_negative": (lambda actual: actual < 0, 0),
    "crps_spike": (lambda actual: actual > 300, None),
    "q95_spike": (lambda actual: actual > 300, 4),
}


def _hourly_pinball(quantile: np.ndarray, actual: np.ndarray) -> np.ndarray:
    error = actual[..., None] - np.sort(quantile, axis=-1)
    return np.maximum(LEVELS * error, (LEVELS - 1) * error)


def _crps(quantile: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return 2 * _hourly_pinball(quantile, actual).mean(axis=-1).mean(axis=1)


def _tail(quantile: np.ndarray, actual: np.ndarray, loss: str) -> np.ndarray:
    select, level = TAIL_LOSSES[loss]
    pinball = _hourly_pinball(quantile, actual)
    hourly = 2 * pinball.mean(axis=-1) if level is None else pinball[..., level]
    mask = select(actual)
    with np.errstate(invalid="ignore"):
        return np.where(mask.any(axis=1), (hourly * mask).sum(axis=1) / mask.sum(axis=1), np.nan)


def _load(pattern: str, region: str, use_naive: bool = False, loss: str = "mae"):
    paths = (
        [pattern.format(region=region, seed=seed) for seed in SEEDS]
        if "{seed}" in pattern
        else [pattern.format(region=region)]
    )
    points, actual, origins = [], None, None
    probabilistic = loss == "crps" or loss in TAIL_LOSSES
    key = "quantile" if probabilistic else ("naive" if use_naive else "point")
    for path in paths:
        with np.load(path) as archive:
            point = archive[key]
            points.extend(point if point.ndim == (4 if probabilistic else 3) else [point])
            if actual is None:
                actual, origins = archive["actual"], archive["origin_unix"]
            elif not (np.array_equal(archive["origin_unix"], origins) and np.allclose(archive["actual"], actual)):
                raise ValueError(f"{path} is not aligned with the other seeds")
    if loss == "crps":
        return _crps(np.mean(points, axis=0), actual), actual, origins, len(points)
    if loss in TAIL_LOSSES:
        return _tail(np.mean(points, axis=0), actual, loss), actual, origins, len(points)
    losses = np.mean([np.abs(p - actual).mean(axis=1) for p in points], axis=0)
    return losses, actual, origins, len(points)


def diebold_mariano(differential: np.ndarray, lag: int) -> dict:
    # Tail losses are undefined at origins without event hours; drop them.
    differential = differential[np.isfinite(differential)]
    d = differential - differential.mean()
    n = d.size
    variance = d @ d / n
    for k in range(1, lag + 1):
        variance += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    statistic = differential.mean() / math.sqrt(variance / n)
    p_value = math.erfc(abs(statistic) / math.sqrt(2))
    return {"mean_difference": float(differential.mean()), "statistic": float(statistic), "p_value": float(p_value)}


def run(reference: str, sources: dict[str, str], lag: int, output: Path, loss: str = "mae") -> dict:
    loaded = {}
    for name, pattern in sources.items():
        loaded[name] = {}
        for region in REGIONS:
            loaded[name][region] = _load(pattern, region, use_naive=name == "naive", loss=loss)
    description = {"crps": "per-origin CRPS~ over 24 hours, seed-averaged quantiles",
                   "mae": "per-origin mean absolute error over 24 hours"}.get(
        loss, f"per-origin {loss} over the event hours of the horizon, seed-averaged quantiles")
    result = {"reference": reference, "lag": lag, "loss": description,
              "seeds": {name: loaded[name][REGIONS[0]][3] for name in sources}, "comparisons": {}}
    for name in sources:
        if name == reference:
            continue
        comparison = {}
        pooled = []
        for region in REGIONS:
            ref_loss, ref_actual, ref_origins, _ = loaded[reference][region]
            loss, actual, origins, _ = loaded[name][region]
            if not (np.array_equal(origins, ref_origins) and np.allclose(actual, ref_actual)):
                raise ValueError(f"{name} and {reference} are not aligned for {region}")
            differential = ref_loss - loss
            comparison[region] = diebold_mariano(differential, lag) | {
                "reference_mae": float(ref_loss.mean()), "other_mae": float(loss.mean()),
            }
            pooled.append(differential)
        with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            comparison["three_region_mean"] = diebold_mariano(np.nanmean(pooled, axis=0), lag)
        result["comparisons"][name] = comparison
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--source", action="append", required=True, help="name=pattern")
    parser.add_argument("--lag", type=int, default=48)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--loss", choices=("mae", "crps", *TAIL_LOSSES), default="mae")
    args = parser.parse_args()
    sources = dict(item.split("=", 1) for item in args.source)
    result = run(args.reference, sources, args.lag, args.output, args.loss)
    for name, comparison in result["comparisons"].items():
        pooled = comparison["three_region_mean"]
        regions = " ".join(
            f"{region}:{comparison[region]['mean_difference']:+.2f}(p={comparison[region]['p_value']:.3f})"
            for region in REGIONS
        )
        print(f"{args.reference} - {name}: pooled {pooled['mean_difference']:+.3f} "
              f"(DM {pooled['statistic']:+.2f}, p={pooled['p_value']:.4f}) | {regions}")


if __name__ == "__main__":
    main()
