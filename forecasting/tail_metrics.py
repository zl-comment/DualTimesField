"""Probabilistic scores in the price tails.

For each source (``name=pattern`` of ``.npz`` files with ``quantile`` and
``actual``, as written by ``protocol_alignment collect`` or
``forecasting.baselines``), the quantiles of seeds 2026-2028 are averaged
(quantile averaging) and scored on the test split:

* CRPS~ (twice the mean pinball loss over the five quantiles) overall and by
  actual-price band;
* the pinball loss of the 0.95 quantile on spike hours (actual above 300) and
  of the 0.05 quantile on negative-price hours;
* spike hit rate (share of spike hours whose 0.95 quantile exceeds 300) and
  false-alarm rate (share of other hours whose 0.95 quantile exceeds 300), and
  the same for negative prices with the 0.05 quantile below 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REGIONS = ("NSW1", "QLD1", "TAS1")
SEEDS = (2026, 2027, 2028)
LEVELS = np.array([0.05, 0.10, 0.50, 0.90, 0.95])
BANDS = {
    "<0": lambda p: p < 0,
    "0-100": lambda p: (p >= 0) & (p <= 100),
    "100-300": lambda p: (p > 100) & (p <= 300),
    ">300": lambda p: p > 300,
}
SPIKE, TROUGH = 300.0, 0.0


def _quantiles(pattern: str, region: str) -> tuple[np.ndarray, np.ndarray]:
    paths = [pattern.format(region=region, seed=s) for s in SEEDS] if "{seed}" in pattern else [pattern.format(region=region)]
    quantiles, actual = [], None
    for path in paths:
        with np.load(path) as archive:
            quantile = archive["quantile"]
            quantiles.extend(quantile if quantile.ndim == 4 else [quantile])
            actual = archive["actual"]
    return np.sort(np.mean(quantiles, axis=0), axis=-1), actual


def _pinball(quantile: np.ndarray, actual: np.ndarray, levels: np.ndarray) -> np.ndarray:
    error = actual[..., None] - quantile
    return np.maximum(levels * error, (levels - 1) * error)


def score(quantile: np.ndarray, actual: np.ndarray) -> dict:
    loss = _pinball(quantile, actual, LEVELS)
    crps = 2 * loss.mean(axis=-1)
    q05, q95 = quantile[..., 0], quantile[..., -1]
    spike, trough = actual > SPIKE, actual < TROUGH
    return {
        "crps_quantile_approx": float(crps.mean()),
        "crps_by_band": {band: float(crps[select(actual)].mean()) for band, select in BANDS.items()},
        "pinball_q95_spike_hours": float(loss[..., -1][spike].mean()),
        "pinball_q05_negative_hours": float(loss[..., 0][trough].mean()),
        "spike_hit_rate_q95": float((q95[spike] > SPIKE).mean()),
        "spike_false_alarm_rate_q95": float((q95[~spike] > SPIKE).mean()),
        "negative_hit_rate_q05": float((q05[trough] < TROUGH).mean()),
        "negative_false_alarm_rate_q05": float((q05[~trough] < TROUGH).mean()),
        "coverage_90": float(((actual >= q05) & (actual <= q95)).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, help="name=pattern")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {}
    for item in args.source:
        name, pattern = item.split("=", 1)
        result[name] = {}
        for region in REGIONS:
            quantile, actual = _quantiles(pattern, region)
            result[name][region] = score(quantile, actual)
        result[name]["mean"] = {
            key: float(np.mean([result[name][r][key] for r in REGIONS]))
            for key, value in result[name][REGIONS[0]].items() if not isinstance(value, dict)
        }
        result[name]["mean"]["crps_by_band"] = {
            band: float(np.mean([result[name][r]["crps_by_band"][band] for r in REGIONS])) for band in BANDS
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    keys = ("crps_quantile_approx", "pinball_q95_spike_hours", "pinball_q05_negative_hours", "spike_hit_rate_q95",
            "spike_false_alarm_rate_q95", "negative_hit_rate_q05", "negative_false_alarm_rate_q05", "coverage_90")
    print("| Model | CRPS~ | " + " | ".join(f"CRPS~ {b}" for b in BANDS) + " | q95 pinball, spikes | q05 pinball, negative | "
          "spike hit | spike false alarm | negative hit | negative false alarm | 90% coverage |")
    print("|---|" + "---:|" * (1 + len(BANDS) + 7))
    for name, data in result.items():
        mean = data["mean"]
        cells = [f"{mean['crps_quantile_approx']:.2f}", *(f"{mean['crps_by_band'][b]:.2f}" for b in BANDS),
                 f"{mean['pinball_q95_spike_hours']:.1f}", f"{mean['pinball_q05_negative_hours']:.2f}",
                 *(f"{100 * mean[k]:.1f}%" for k in keys[3:])]
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
