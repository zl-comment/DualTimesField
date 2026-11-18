"""Verify the trailing-gas origin-context feature respects the origin boundary.

Read-only audit: replicates the exact indexing of datasets.py
(_trailing_gas_price over frame rows, origin_context_values[origin row]) and
checks, for every hourly row of every region, that the latest gas day entering
the 7-day trailing mean ended at or before that row's delivery hour (the
origin instant under the aligned protocol).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REGIONS = ("nsw1", "qld1", "tas1")


def verify(gas_csv: Path, region_dir: Path, window_days: int = 7) -> list[dict]:
    gas = pd.read_csv(gas_csv)
    daily = gas.groupby("Gas_Date")["Price"].mean()
    daily.index = pd.to_datetime(daily.index)
    expected = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    missing = expected.difference(pd.DatetimeIndex(daily.index))
    if len(missing):
        raise ValueError(f"Gas archive has {len(missing)} missing days, first {missing[0].date()}")

    rows = []
    for region in REGIONS:
        frame = pd.read_csv(region_dir / f"{region}_hourly.csv")
        naive = pd.to_datetime(frame["delivery_start_aest"]).dt.tz_localize(None)
        # datasets.py:275-283, verbatim logic
        current_gas_day = (naive - pd.Timedelta(hours=6)).dt.normalize()
        trailing = daily.rolling(window_days).mean().shift(1)
        values = trailing.reindex(current_gas_day).to_numpy(dtype=np.float64)
        # The last gas day in the window is current_gas_day - 1; a gas day D
        # ends 06:00 on D+1, so it ends 06:00 on current_gas_day's date.
        end_margin_h = (naive - (current_gas_day + pd.Timedelta(hours=6))).dt.total_seconds() / 3600
        # Its final DWGM schedule stamps 22:00 on the day it starts.
        schedule_margin_h = (
            naive - (current_gas_day - pd.DateOffset(days=1) + pd.Timedelta(hours=22))
        ).dt.total_seconds() / 3600
        if (end_margin_h < 0).any() or np.isnan(values).any() or (values <= 0).any():
            raise ValueError(f"{region}: as-of boundary or gas value violation")
        rows.append({
            "region": region.upper(), "rows": len(frame),
            "first": str(naive.min()), "last": str(naive.max()),
            "min_end_margin_hours": float(end_margin_h.min()),
            "max_end_margin_hours": float(end_margin_h.max()),
            "min_final_schedule_margin_hours": float(schedule_margin_h.min()),
            "boundary_violations": int((end_margin_h < 0).sum()),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gas-csv", type=Path,
                        default=ROOT / "data/aemo_exogenous/raw_gas/dwgm_prices.csv")
    parser.add_argument("--region-dir", type=Path,
                        default=ROOT.parent / "electricity-price-forecasting-research/data/multi_market_energy_reserve/aemo_nem")
    parser.add_argument("--window-days", type=int, default=7)
    args = parser.parse_args()
    for row in verify(args.gas_csv, args.region_dir, args.window_days):
        print(row)


if __name__ == "__main__":
    main()
