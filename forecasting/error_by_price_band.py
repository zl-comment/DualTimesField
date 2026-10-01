"""Split the absolute-error gap between two forecasters by actual-price band.

Sources are ``.npz`` patterns as in ``significance``; a pattern with ``{seed}``
is averaged over seeds 2026-2028 (a file with ``[S, N, 24]`` points is averaged
over its first axis), so both forecasters are seed-ensembles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REGIONS = ("NSW1", "QLD1", "TAS1")
SEEDS = (2026, 2027, 2028)
# Actual-price bands in AUD/MWh: below 0, 0 to 100, above 100 to 300, above 300.
BANDS = {
    "<0": lambda p: p < 0,
    "0-100": lambda p: (p >= 0) & (p <= 100),
    "100-300": lambda p: (p > 100) & (p <= 300),
    ">300": lambda p: p > 300,
}


def _point(pattern: str, region: str) -> tuple[np.ndarray, np.ndarray]:
    paths = [pattern.format(region=region, seed=s) for s in SEEDS] if "{seed}" in pattern else [pattern.format(region=region)]
    points, actual = [], None
    for path in paths:
        with np.load(path) as archive:
            point = archive["point"]
            points.extend(point if point.ndim == 3 else [point])
            actual = archive["actual"]
    return np.mean(points, axis=0), actual


def compare(first: str, second: str) -> dict:
    result, gap_by_band, total_gap = {}, {band: 0.0 for band in BANDS}, 0.0
    for region in REGIONS:
        a, actual = _point(first, region)
        b, _ = _point(second, region)
        error_a, error_b = np.abs(a - actual), np.abs(b - actual)
        total_gap += float(error_a.sum() - error_b.sum())
        result[region] = {}
        for band, select in BANDS.items():
            mask = select(actual)
            result[region][band] = {
                "share": float(mask.mean()),
                "first_mae": float(error_a[mask].mean()),
                "second_mae": float(error_b[mask].mean()),
            }
            gap_by_band[band] += float(error_a[mask].sum() - error_b[mask].sum())
    result["share_of_total_gap"] = {band: gap / total_gap for band, gap in gap_by_band.items()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.first, args.second)
    for region in REGIONS:
        for band, row in result[region].items():
            print(f"{region} {band:8s} share {row['share'] * 100:5.2f}%  first {row['first_mae']:8.2f}  second {row['second_mae']:8.2f}")
    print("share of the total absolute-error gap:",
          {band: round(share, 3) for band, share in result["share_of_total_gap"].items()})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
