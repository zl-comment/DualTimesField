"""Extend the hourly price and demand series with AEMO's aggregated price and demand files.

``PRICE_AND_DEMAND_<yyyymm>_<region>.csv`` holds one row per dispatch interval (five minutes since
2021-10-01, 30 minutes before) with ``SETTLEMENTDATE`` the end of the interval in market time (AEST).
An hourly row is the mean of the intervals inside the hour, labelled by the hour start, with
``available_at`` one hour later, as in the existing series. ``verify`` rebuilds months that the existing
series already holds and compares them.

Usage::

    python -m forecasting.build_price_demand verify --months 2024-11 2024-12
    python -m forecasting.build_price_demand extend --start 2025-01 --end 2026-08 --output-dir data/holdout
"""

from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{ym}_{region}.csv"
REGIONS = ("NSW1", "QLD1", "TAS1")
EXISTING = Path(__file__).resolve().parents[2] / "electricity-price-forecasting-research/data/multi_market_energy_reserve/aemo_nem"
CACHE = Path("data/aemo_exogenous/raw/price_demand")


def fetch(ym: str, region: str, proxy: str | None) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"PRICE_AND_DEMAND_{ym}_{region}.csv"
    if not path.exists():
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"https": proxy, "http": proxy} if proxy else {}))
        request = urllib.request.Request(URL.format(ym=ym, region=region), headers={"User-Agent": "Mozilla/5.0"})
        with opener.open(request, timeout=120) as response:
            path.write_bytes(response.read())
    return pd.read_csv(path)


def hourly(frame: pd.DataFrame, region: str) -> pd.DataFrame:
    stamp = pd.to_datetime(frame["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
    interval = (stamp.diff().dropna().mode().iloc[0]).total_seconds() / 60
    start = (stamp - pd.Timedelta(minutes=interval)).dt.floor("h")
    grouped = frame.assign(start=start).groupby("start")
    out = pd.DataFrame({
        "rrp_aud_per_mwh": grouped["RRP"].mean(),
        "total_demand_mw": grouped["TOTALDEMAND"].mean(),
        "source_interval_count": grouped["RRP"].count(),
    })
    out["source_interval_minutes"] = int(interval)
    out["region"] = region
    out = out.reset_index().rename(columns={"start": "delivery_start"})
    out["delivery_start_aest"] = out["delivery_start"].dt.strftime("%Y-%m-%dT%H:%M:%S+10:00")
    out["available_at_aest"] = (out["delivery_start"] + pd.Timedelta(hours=1)).dt.strftime("%Y-%m-%dT%H:%M:%S+10:00")
    return out


def months(start: str, end: str) -> list[str]:
    return [p.strftime("%Y%m") for p in pd.period_range(start, end, freq="M")]


def build(region: str, yms: list[str], proxy: str | None) -> pd.DataFrame:
    return pd.concat([hourly(fetch(ym, region, proxy), region) for ym in yms], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("verify", "extend"))
    parser.add_argument("--months", nargs="*", default=["2024-11", "2024-12"])
    parser.add_argument("--start", default="2025-01")
    parser.add_argument("--end", default="2026-08")
    parser.add_argument("--output-dir", type=Path, default=Path("data/holdout"))
    parser.add_argument("--proxy", default="http://127.0.0.1:7892")
    args = parser.parse_args()
    if args.command == "verify":
        for region in REGIONS:
            new = build(region, [m.replace("-", "") for m in args.months], args.proxy)
            old = pd.read_csv(EXISTING / f"{region.lower()}_hourly.csv")
            merged = new.merge(old, on="delivery_start_aest", suffixes=("_new", "_old"))
            price = (merged["rrp_aud_per_mwh_new"] - merged["rrp_aud_per_mwh_old"]).abs().max()
            demand = (merged["total_demand_mw_new"] - merged["total_demand_mw_old"]).abs().max()
            print(f"{region}: {len(new)} new hours, {len(merged)} matched, max |price diff| {price:.6f}, "
                  f"max |demand diff| {demand:.6f}, interval counts equal {(merged['source_interval_count_new'] == merged['source_interval_count_old']).mean():.3f}")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for region in REGIONS:
            old = pd.read_csv(EXISTING / f"{region.lower()}_hourly.csv")
            new = build(region, months(args.start, args.end), args.proxy)
            new["split"] = "test"
            columns = list(old.columns)
            combined = pd.concat([old, new[columns]], ignore_index=True)
            combined.to_csv(args.output_dir / f"{region.lower()}_hourly.csv", index=False)
            print(region, "old", len(old), "new", len(new), "combined", len(combined))


if __name__ == "__main__":
    main()
