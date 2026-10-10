"""Join the 2015-2024 per-origin exogenous arrays with the 2025-2026 extension.

``data/aemo_exogenous/<region>_<name>.npz`` (existing) and ``data/aemo_exogenous_holdout/<region>_<name>.npz`` (rebuilt
from 2024 or 2024-11 so that the boundary origins have their earlier runs) are concatenated, keeping the existing
origins up to the existing last origin and the rebuilt ones after it. The overlap is checked for equality first, so a
change in the pipeline cannot slip in. Output: ``data/aemo_exogenous_holdout/merged/``.

    python -m forecasting.merge_holdout_exogenous --names pdpasa predispatch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REGIONS = ("nsw1", "qld1", "tas1")


def merge(name: str, region: str, old_dir: Path, new_dir: Path, out_dir: Path) -> dict:
    with np.load(old_dir / f"{region}_{name}.npz") as a, np.load(new_dir / f"{region}_{name}.npz") as b:
        old = {k: a[k] for k in a.files}
        new = {k: b[k] for k in b.files}
    if sorted(old) != sorted(new):
        raise ValueError(f"{name}/{region}: keys differ {sorted(old)} vs {sorted(new)}")
    o_old, o_new = old["forecast_origin_unix"], new["forecast_origin_unix"]
    common = np.intersect1d(o_old, o_new)
    worst = 0.0
    for key in old:
        if key == "forecast_origin_unix" or old[key].dtype.kind not in "fiu":
            continue
        x = old[key][np.searchsorted(o_old, common)].astype(np.float64)
        y = new[key][np.searchsorted(o_new, common)].astype(np.float64)
        both = np.isfinite(x) & np.isfinite(y)
        if not np.array_equal(np.isfinite(x), np.isfinite(y)) or (both.any() and np.abs(x[both] - y[both]).max() > 0):
            worst = max(worst, float(np.abs(x[both] - y[both]).max()) if both.any() else np.inf)
    if worst > 0:
        raise ValueError(f"{name}/{region}: the rebuilt overlap differs from the stored arrays (max {worst})")
    keep = o_new > o_old.max()
    merged = {k: np.concatenate([old[k], new[k][keep]]) for k in old}
    if np.any(np.diff(merged["forecast_origin_unix"]) <= 0):
        raise ValueError(f"{name}/{region}: origins not increasing after the merge")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{region}_{name}.npz", **merged)
    return {"overlap": int(len(common)), "added": int(keep.sum()), "origins": int(len(merged["forecast_origin_unix"]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--names", nargs="+", default=["pdpasa", "predispatch"])
    parser.add_argument("--old-dir", type=Path, default=Path("data/aemo_exogenous"))
    parser.add_argument("--new-dir", type=Path, default=Path("data/aemo_exogenous_holdout"))
    args = parser.parse_args()
    for name in args.names:
        for region in REGIONS:
            print(name, region, merge(name, region, args.old_dir, args.new_dir, args.new_dir / "merged"), flush=True)


if __name__ == "__main__":
    main()
