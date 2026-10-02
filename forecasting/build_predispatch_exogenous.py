"""Build point-in-time AEMO predispatch forecasts for each hourly origin.

AEMO's predispatch re-runs the market dispatch every 30 minutes with the
current offers, constraints, and demand forecasts, and publishes regional
prices and quantities for every half hour up to the end of the next trading
day. The full run history is in the MMSDM archive,
``.../MMSDM_<year>_<month>/MMSDM_Historical_Data_SQLLoader/PREDISP_ALL_DATA/``
(tables PREDISPATCHPRICE and PREDISPATCHREGIONSUM).

For every hourly origin, the latest run whose rows were all last changed at or
before the origin supplies the 24 forecast hours (the mean of the two half
hours of each hour). Runs published between about 04:00 and 12:30 end at
04:00 the next day, before the 24th hour; those hours repeat the last covered
hour and are flagged by ``predispatch_covered`` = 0.

AEMO's archive has no run history for October 2022 (only the final-run
``_D`` tables, which are not point-in-time). Origins whose latest available run
is more than six hours old are marked uncovered for all 24 hours and take the
region's 2015-2021 median of each field, so they carry no predispatch
information and no future information.

Usage::

    python -m forecasting.build_predispatch_exogenous download --start 2015-01 --end 2024-12
    python -m forecasting.build_predispatch_exogenous build --start 2015-01 --end 2024-12
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE_ROOT = "https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
REGIONS = ("NSW1", "QLD1", "TAS1")
TABLES = ("PREDISPATCHPRICE", "PREDISPATCHREGIONSUM")
# (output array, table, column)
FIELDS = (
    ("predispatch_rrp", "PREDISPATCHPRICE", "RRP"),
    ("predispatch_total_demand_mw", "PREDISPATCHREGIONSUM", "TOTALDEMAND"),
    ("predispatch_available_generation_mw", "PREDISPATCHREGIONSUM", "AVAILABLEGENERATION"),
    ("predispatch_net_interchange_mw", "PREDISPATCHREGIONSUM", "NETINTERCHANGE"),
)
HALF_HOUR = 1800
AEST_OFFSET = pd.Timedelta(hours=10)


def _months(start: str, end: str) -> list[tuple[int, int]]:
    return [(p.year, p.month) for p in pd.period_range(start, end, freq="M")]


def _directory(year: int, month: int) -> str:
    return f"{ARCHIVE_ROOT}/{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/PREDISP_ALL_DATA/"


def _archive_url(year: int, month: int, table: str, proxy: str | None) -> str:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": proxy, "http": proxy} if proxy else {}))
    directory = _directory(year, month)
    with opener.open(directory, timeout=120) as response:
        listing = response.read().decode("utf-8", errors="replace")
    names = [html.unescape(h) for h in re.findall(r'href=["\']([^"\']+\.zip)["\']', listing, flags=re.I)]
    # Before August 2024: PUBLIC_DVD_<TABLE>_<yyyymm>010000.zip; afterwards
    # PUBLIC_ARCHIVE#<TABLE>#ALL#FILE01#<yyyymm>010000.zip.
    pattern = re.compile(rf"(PUBLIC_DVD_{table}_\d{{12}}|PUBLIC_ARCHIVE(%23|#){table}(%23|#)ALL(%23|#)FILE\d+(%23|#)\d{{12}})\.zip$", re.I)
    matches = [n for n in names if pattern.search(urllib.parse.unquote(n.rsplit("/", 1)[-1]))]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {table} archive in {directory}, found {matches}")
    return urllib.parse.urljoin(directory, matches[0])


def _cache_path(cache_dir: Path, table: str, year: int, month: int) -> Path:
    return cache_dir / f"{table}_{year}{month:02d}.zip"


def _download_one(year: int, month: int, table: str, cache_dir: Path, proxy: str | None) -> str:
    """Downloads one archive, resuming a partial file with HTTP range requests."""
    import time

    path = _cache_path(cache_dir, table, year, month)
    if path.exists() and zipfile.is_zipfile(path):
        return f"cached {path.name}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": proxy, "http": proxy} if proxy else {}))
    part = path.with_suffix(".part")
    attempts = 12
    for attempt in range(1, attempts + 1):
        try:
            url = _archive_url(year, month, table, proxy)
            offset = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "DualTimesField/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with opener.open(urllib.request.Request(url, headers=headers), timeout=600) as response:
                if offset and response.status == 206:
                    total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
                    mode = "ab"
                else:
                    total = int(response.headers["Content-Length"])
                    mode = "wb"
                with part.open(mode) as output:
                    shutil.copyfileobj(response, output, length=1 << 20)
            if part.stat().st_size < total:
                raise ConnectionError(f"{part.name}: {part.stat().st_size} of {total} bytes")
            part.replace(path)
            if zipfile.is_zipfile(path):
                return f"downloaded {path.name} {path.stat().st_size / 1e6:.1f} MB"
            path.unlink(missing_ok=True)
        except Exception as error:
            if attempt == attempts:
                return f"FAILED {path.name}: {error!r}"
            # The archive server drops connections under load; back off and resume.
            time.sleep(min(120, 10 * attempt))
    return f"FAILED {path.name}: invalid archive"


def download(start: str, end: str, cache_dir: Path, proxy: str | None, workers: int = 1) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(year, month, table) for year, month in _months(start, end) for table in TABLES]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download_one, y, m, t, cache_dir, proxy) for y, m, t in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


def _read_table(path: Path, table: str, columns: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise RuntimeError(f"Expected one CSV in {path}, found {names}")
        text = archive.read(names[0]).decode("utf-8-sig")
    lines = text.splitlines()
    header = next(line for line in lines if line.startswith("I,"))
    body = "\n".join(line for line in lines if line.startswith("D,"))
    names = [name.upper() for name in header.split(",")]
    missing = set(columns) - set(names)
    if missing:
        raise RuntimeError(f"Missing columns {sorted(missing)} in {path}")
    frame = pd.read_csv(io.StringIO(body), header=None, names=names, usecols=columns, low_memory=False)
    return frame


def _parse_aest(values: pd.Series) -> np.ndarray:
    # AEMO market time is AEST (UTC+10, no daylight saving).
    stamps = pd.to_datetime(values, format="%Y/%m/%d %H:%M:%S")
    return ((stamps - pd.Timestamp("1970-01-01")) - AEST_OFFSET).dt.total_seconds().astype(np.int64).to_numpy()


def _read_month(cache_dir: Path, year: int, month: int) -> pd.DataFrame:
    keys = ["PREDISPATCHSEQNO", "REGIONID", "PERIODID", "INTERVENTION", "LASTCHANGED", "DATETIME"]
    merged = None
    for table in TABLES:
        fields = [column for _, t, column in FIELDS if t == table]
        frame = _read_table(_cache_path(cache_dir, table, year, month), table, keys + fields)
        frame = frame[frame["REGIONID"].isin(REGIONS) & (frame["INTERVENTION"] == 0)]
        frame = frame.sort_values("LASTCHANGED").drop_duplicates(["PREDISPATCHSEQNO", "REGIONID", "PERIODID"], keep="last")
        frame = frame.rename(columns={"LASTCHANGED": f"LASTCHANGED_{table}"}).drop(columns="INTERVENTION")
        merged = frame if merged is None else merged.merge(frame, on=["PREDISPATCHSEQNO", "REGIONID", "PERIODID", "DATETIME"], how="inner")
    merged["period_end"] = _parse_aest(merged["DATETIME"])
    changed = np.maximum(*(_parse_aest(merged[f"LASTCHANGED_{t}"]) for t in TABLES))
    merged["changed"] = changed
    return merged[["PREDISPATCHSEQNO", "REGIONID", "period_end", "changed", *[c for _, _, c in FIELDS]]]


STALE_SECONDS = 6 * 3600
TRAIN_END = pd.Timestamp("2022-01-01")


def build(start: str, end: str, cache_dir: Path, output_dir: Path) -> dict:
    per_region = {region: [] for region in REGIONS}
    for year, month in _months(start, end):
        if not all(_cache_path(cache_dir, t, year, month).exists() for t in TABLES):
            print(f"no archive for {year}-{month:02d}; its origins are filled as uncovered", flush=True)
            continue
        frame = _read_month(cache_dir, year, month)
        for region in REGIONS:
            per_region[region].append(frame[frame["REGIONID"] == region].drop(columns="REGIONID"))
        print(f"read {year}-{month:02d}: {len(frame)} rows", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for region in REGIONS:
        rows = pd.concat(per_region[region], ignore_index=True)
        # Some monthly archives repeat the first days of the next month (e.g.
        # 2016-08 holds the runs of 1-7 September); keep one copy of each row.
        duplicates = int(rows.duplicated(["PREDISPATCHSEQNO", "period_end"]).sum())
        rows = rows.sort_values("changed").drop_duplicates(["PREDISPATCHSEQNO", "period_end"], keep="last")
        # A run is available once every one of its rows has been published.
        availability = rows.groupby("PREDISPATCHSEQNO")["changed"].transform("max")
        rows = rows.assign(available=availability).sort_values(["available", "PREDISPATCHSEQNO", "period_end"])
        values = rows[[c for _, _, c in FIELDS]].to_numpy(np.float64)
        period_end = rows["period_end"].to_numpy(np.int64)
        runs = rows.groupby("PREDISPATCHSEQNO", sort=False)["available"].first()
        run_ids = runs.index.to_numpy()
        run_available = runs.to_numpy(np.int64)
        order = np.argsort(run_available, kind="stable")
        run_ids, run_available = run_ids[order], run_available[order]
        # Exact (run, half hour) lookup: a period missing from a run is uncovered.
        run_position = pd.Series(np.arange(len(run_ids)), index=run_ids)[rows["PREDISPATCHSEQNO"].to_numpy()].to_numpy()
        base = period_end.min()
        slot = (period_end - base) // HALF_HOUR
        keys = run_position.astype(np.int64) * (1 << 24) + slot
        key_order = np.argsort(keys, kind="stable")
        sorted_keys = keys[key_order]
        first_origin = int(np.ceil(run_available[0] / 3600) * 3600)
        last_origin = int(np.floor(run_available[-1] / 3600) * 3600)
        origins = np.arange(first_origin, last_origin + 1, 3600, dtype=np.int64)
        latest = np.searchsorted(run_available, origins, side="right") - 1
        # Half hour j (0..47) of an origin ends at origin + (j + 1) * 30 min.
        ends = origins[:, None] + (np.arange(48)[None, :] + 1) * HALF_HOUR
        query = latest[:, None].astype(np.int64) * (1 << 24) + (ends - base) // HALF_HOUR
        found = np.clip(np.searchsorted(sorted_keys, query), 0, len(sorted_keys) - 1)
        covered = (sorted_keys[found] == query) & (ends >= base)
        row_index = np.where(covered, key_order[found], 0)
        if np.any(period_end[row_index][covered] != ends[covered]):
            raise RuntimeError(f"{region}: predispatch lookup mismatch")
        # Diagnostic: half hours missing inside a run's covered span.
        span_gaps = int(rows.groupby("PREDISPATCHSEQNO")["period_end"].agg(lambda e: (e.max() - e.min()) // HALF_HOUR + 1 - len(e)).sum())
        half = np.where(covered[..., None], values[row_index], np.nan)
        hourly = np.nanmean(half.reshape(len(origins), 24, 2, len(FIELDS)), axis=2)
        hour_covered = covered.reshape(len(origins), 24, 2).all(axis=2)
        hourly[~hour_covered] = np.nan
        stale = origins - run_available[latest] > STALE_SECONDS
        hour_covered[stale] = False
        hourly[stale] = np.nan
        # Repeat the last covered hour over the uncovered tail.
        filled = np.stack(
            [pd.DataFrame(hourly[..., f]).ffill(axis=1).to_numpy() for f in range(len(FIELDS))], axis=-1
        )
        train = (origins < (TRAIN_END - AEST_OFFSET - pd.Timestamp("1970-01-01")).total_seconds()) & ~stale
        medians = np.nanmedian(filled[train].reshape(-1, len(FIELDS)), axis=0)
        filled[stale] = medians
        complete = np.isfinite(filled).all(axis=(1, 2))
        output = output_dir / f"{region.lower()}_predispatch.npz"
        np.savez_compressed(
            output,
            forecast_origin_unix=origins[complete],
            **{name: filled[complete][..., i].astype(np.float32) for i, (name, _, _) in enumerate(FIELDS)},
            predispatch_covered=hour_covered[complete].astype(np.float32),
            source_run_unix=run_available[latest][complete],
            source_last_changed_unix=run_available[latest][complete],
            source_run_seqno=run_ids[latest][complete].astype(np.int64),
        )
        summary[region] = {
            "origins": int(complete.sum()),
            "dropped_origins": int((~complete).sum()),
            "stale_origins_filled": int((stale & complete).sum()),
            "duplicate_rows_dropped": duplicates,
            "half_hours_missing_inside_runs": span_gaps,
            "covered_hours_share": float(hour_covered[complete].mean()),
            "first_origin": str(pd.to_datetime(origins[complete][0], unit="s") + AEST_OFFSET),
            "last_origin": str(pd.to_datetime(origins[complete][-1], unit="s") + AEST_OFFSET),
        }
        print(region, summary[region], flush=True)
    (output_dir / "predispatch_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("download", "build"))
    parser.add_argument("--start", default="2015-01")
    parser.add_argument("--end", default="2024-12")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/aemo_exogenous/raw/predispatch"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/aemo_exogenous"))
    parser.add_argument("--proxy", default=None, help="e.g. http://127.0.0.1:7892")
    parser.add_argument("--workers", type=int, default=1, help="parallel downloads")
    args = parser.parse_args()
    if args.command == "download":
        download(args.start, args.end, args.cache_dir, args.proxy, args.workers)
    else:
        build(args.start, args.end, args.cache_dir, args.output_dir)


if __name__ == "__main__":
    main()
