"""Build point-in-time AEMO PD PASA maximum-spare-capacity features.

The generated arrays are keyed by the hourly forecast origin.  Values use the
latest available OUTAGE_LRC PD PASA run, with its short end-of-market-day tail
filled from the latest already-published ST PASA run.  The two half-hour values
within an hour are reduced with ``min`` so scarcity is not averaged away.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


AEST = timezone(timedelta(hours=10))
ARCHIVE_ROOT = "https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
REGIONS = ("NSW1", "QLD1", "TAS1")
TIME_FORMAT = "%Y/%m/%d %H:%M:%S"


def _month_directory(year: int, month: int) -> str:
    return (
        f"{ARCHIVE_ROOT}/{year}/MMSDM_{year}_{month:02d}/"
        "MMSDM_Historical_Data_SQLLoader/DATA/"
    )


def _discover_archive(
    year: int,
    month: int,
    table: str = "PDPASA_REGIONSOLUTION",
) -> str:
    directory = _month_directory(year, month)
    listing = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(directory, timeout=60) as response:
                listing = response.read().decode("utf-8", errors="replace")
            break
        except Exception:
            if attempt == 3:
                raise
    if listing is None:
        raise RuntimeError(f"Could not read archive directory {directory}")
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', listing, flags=re.I)
    matches = [
        html.unescape(href)
        for href in hrefs
        if table in href.upper()
        and href.lower().endswith(".zip")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {table} archive in {directory}, "
            f"found {matches}"
        )
    return urllib.parse.urljoin(directory, matches[0])


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DualTimesField/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(destination)


def _ensure_archive(url: str, destination: Path) -> None:
    if destination.exists() and zipfile.is_zipfile(destination):
        return
    destination.unlink(missing_ok=True)
    for attempt in range(1, 4):
        try:
            _download(url, destination)
            if zipfile.is_zipfile(destination):
                return
        except Exception:
            if attempt == 3:
                raise
        destination.unlink(missing_ok=True)
        destination.with_suffix(destination.suffix + ".part").unlink(missing_ok=True)
    raise RuntimeError(f"Downloaded archive is invalid after three attempts: {url}")


def _parse_time(value: str) -> datetime:
    return datetime.strptime(value, TIME_FORMAT).replace(tzinfo=AEST)


def _unix_seconds(value: datetime) -> int:
    return int(value.timestamp())


def _read_month(
    archive_path: Path,
    runs: dict[str, dict[int, dict[str, object]]],
    start: datetime,
    end: datetime,
) -> tuple[int, int]:
    accepted = 0
    rejected_late = 0
    with zipfile.ZipFile(archive_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError(f"Expected one CSV in {archive_path}, found {csv_names}")
        with archive.open(csv_names[0]) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            columns = None
            for row in reader:
                if len(row) < 5:
                    continue
                if row[0] == "I" and row[1:3] == ["PDPASA", "REGIONSOLUTION"]:
                    columns = {name.upper(): index for index, name in enumerate(row)}
                    required = {
                        "RUN_DATETIME",
                        "INTERVAL_DATETIME",
                        "REGIONID",
                        "MAXSPARECAPACITY",
                        "LASTCHANGED",
                        "RUNTYPE",
                    }
                    missing = required - set(columns)
                    if missing:
                        raise RuntimeError(f"Missing columns {sorted(missing)} in {archive_path}")
                    continue
                if row[0] != "D" or columns is None:
                    continue
                region = row[columns["REGIONID"]]
                if region not in runs or row[columns["RUNTYPE"]] != "OUTAGE_LRC":
                    continue
                run_time = _parse_time(row[columns["RUN_DATETIME"]])
                if not (start <= run_time < end) or run_time.minute != 0:
                    continue
                interval_time = _parse_time(row[columns["INTERVAL_DATETIME"]])
                half_hour_index = int((interval_time - run_time).total_seconds() // 1800) - 1
                if not 0 <= half_hour_index < 48:
                    continue
                last_changed = _parse_time(row[columns["LASTCHANGED"]])
                if last_changed > run_time:
                    rejected_late += 1
                    continue
                value_text = row[columns["MAXSPARECAPACITY"]]
                if not value_text:
                    continue
                run_unix = _unix_seconds(run_time)
                record = runs[region].setdefault(
                    run_unix,
                    {
                        "half_hour": np.full(48, np.nan, dtype=np.float32),
                        "last_changed": 0,
                    },
                )
                record["half_hour"][half_hour_index] = float(value_text)
                record["last_changed"] = max(
                    int(record["last_changed"]), _unix_seconds(last_changed)
                )
                accepted += 1
    return accepted, rejected_late


def _read_stpasa_month(
    archive_path: Path,
    runs: dict[str, dict[int, dict[str, object]]],
    start: datetime,
    end: datetime,
) -> int:
    accepted = 0
    with zipfile.ZipFile(archive_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError(f"Expected one CSV in {archive_path}, found {csv_names}")
        with archive.open(csv_names[0]) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            columns = None
            for row in reader:
                if len(row) < 5:
                    continue
                if row[0] == "I" and row[1:3] == ["STPASA", "REGIONSOLUTION"]:
                    columns = {name.upper(): index for index, name in enumerate(row)}
                    required = {
                        "RUN_DATETIME",
                        "INTERVAL_DATETIME",
                        "REGIONID",
                        "MAXSPARECAPACITY",
                        "LASTCHANGED",
                        "RUNTYPE",
                    }
                    missing = required - set(columns)
                    if missing:
                        raise RuntimeError(f"Missing columns {sorted(missing)} in {archive_path}")
                    continue
                if row[0] != "D" or columns is None:
                    continue
                region = row[columns["REGIONID"]]
                if region not in runs or row[columns["RUNTYPE"]] != "OUTAGE_LRC":
                    continue
                run_time = _parse_time(row[columns["RUN_DATETIME"]])
                if not (start <= run_time < end):
                    continue
                interval_time = _parse_time(row[columns["INTERVAL_DATETIME"]])
                if not timedelta(0) < interval_time - run_time <= timedelta(hours=48):
                    continue
                value_text = row[columns["MAXSPARECAPACITY"]]
                if not value_text:
                    continue
                last_changed = _parse_time(row[columns["LASTCHANGED"]])
                run_unix = _unix_seconds(run_time)
                record = runs[region].setdefault(
                    run_unix,
                    {"values": {}, "last_changed": 0},
                )
                record["values"][_unix_seconds(interval_time)] = float(value_text)
                record["last_changed"] = max(
                    int(record["last_changed"]), _unix_seconds(last_changed)
                )
                accepted += 1
    return accepted


def _write_region(
    region: str,
    region_runs: dict[int, dict[str, object]],
    stpasa_runs: dict[int, dict[str, object]],
    output: Path,
) -> dict:
    origins = []
    values = []
    source_pdpasa_run = []
    source_last_changed = []
    source_stpasa_run = []
    incomplete = 0
    stpasa_filled_values = 0
    available_stpasa = sorted(
        (
            int(record["last_changed"]),
            run_time,
            record,
        )
        for run_time, record in stpasa_runs.items()
    )
    stpasa_availability = [item[0] for item in available_stpasa]
    available_pdpasa = sorted(
        (
            int(record["last_changed"]),
            run_time,
            record,
        )
        for run_time, record in region_runs.items()
    )
    pdpasa_availability = [item[0] for item in available_pdpasa]
    first_origin = min(region_runs)
    last_origin = max(region_runs)
    for origin in range(first_origin, last_origin + 1, 3600):
        pdpasa_index = bisect_right(pdpasa_availability, origin) - 1
        if pdpasa_index < 0:
            incomplete += 1
            continue
        _, pdpasa_run, record = available_pdpasa[pdpasa_index]
        half_hour = np.full(48, np.nan, dtype=np.float32)
        used_pdpasa_run = 0
        used_pdpasa_last_changed = 0
        for index in range(48):
            interval = origin + (index + 1) * 1800
            for candidate in range(pdpasa_index, -1, -1):
                candidate_changed, candidate_run, candidate_record = (
                    available_pdpasa[candidate]
                )
                if candidate_changed < origin - 24 * 3600:
                    break
                source_index = int((interval - candidate_run) // 1800) - 1
                if (
                    0 <= source_index < 48
                    and np.isfinite(candidate_record["half_hour"][source_index])
                ):
                    half_hour[index] = candidate_record["half_hour"][source_index]
                    used_pdpasa_run = max(used_pdpasa_run, candidate_run)
                    used_pdpasa_last_changed = max(
                        used_pdpasa_last_changed, candidate_changed
                    )
                    break
        stpasa_index = bisect_right(stpasa_availability, origin) - 1
        stpasa_record = None
        stpasa_run = 0
        used_stpasa_last_changed = 0
        used_stpasa = False
        if stpasa_index >= 0:
            for index in np.flatnonzero(np.isnan(half_hour)):
                interval = origin + (int(index) + 1) * 1800
                for candidate in range(stpasa_index, -1, -1):
                    candidate_changed, candidate_run, candidate_record = (
                        available_stpasa[candidate]
                    )
                    if candidate_changed < origin - 24 * 3600:
                        break
                    if interval in candidate_record["values"]:
                        half_hour[index] = candidate_record["values"][interval]
                        stpasa_run = max(stpasa_run, candidate_run)
                        stpasa_record = candidate_record
                        used_stpasa_last_changed = max(
                            used_stpasa_last_changed, candidate_changed
                        )
                        stpasa_filled_values += 1
                        used_stpasa = True
                        break
        if np.isnan(half_hour).any():
            incomplete += 1
            continue
        origins.append(origin)
        values.append(np.minimum(half_hour[0::2], half_hour[1::2]))
        source_pdpasa_run.append(used_pdpasa_run)
        source_last_changed.append(
            max(
                used_pdpasa_last_changed,
                used_stpasa_last_changed,
            )
        )
        source_stpasa_run.append(stpasa_run if used_stpasa else 0)
    if not origins:
        raise RuntimeError(f"No complete hourly origins were produced for {region}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        forecast_origin_unix=np.asarray(origins, dtype=np.int64),
        max_spare_capacity_mw=np.asarray(values, dtype=np.float32),
        source_run_unix=np.asarray(source_pdpasa_run, dtype=np.int64),
        source_stpasa_run_unix=np.asarray(source_stpasa_run, dtype=np.int64),
        source_last_changed_unix=np.asarray(source_last_changed, dtype=np.int64),
    )
    return {
        "region": region,
        "complete_origins": len(origins),
        "incomplete_origins": incomplete,
        "stpasa_filled_half_hours": stpasa_filled_values,
        "first_origin_unix": origins[0],
        "last_origin_unix": origins[-1],
    }


def build_dataset(start_year: int, end_year: int, output_dir: Path, cache_dir: Path) -> None:
    start = datetime(start_year, 1, 1, tzinfo=AEST)
    # AEMO stores the midnight run at a year boundary in the preceding
    # December archive.  Retain that one extra origin so independently built
    # yearly shards join without a gap.
    end = datetime(end_year + 1, 1, 1, tzinfo=AEST) + timedelta(hours=1)
    runs: dict[str, dict[int, dict[str, object]]] = {region: {} for region in REGIONS}
    stpasa_runs: dict[str, dict[int, dict[str, object]]] = {
        region: {} for region in REGIONS
    }
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            pd_url = _discover_archive(year, month, "PDPASA_REGIONSOLUTION")
            pd_archive = cache_dir / (
                f"{year}-{month:02d}-{Path(urllib.parse.urlparse(pd_url).path).name}"
            )
            if not pd_archive.exists():
                print(f"download source=PDPASA year={year} month={month:02d} url={pd_url}", flush=True)
            _ensure_archive(pd_url, pd_archive)
            accepted, rejected_late = _read_month(pd_archive, runs, start, end)
            st_url = _discover_archive(year, month, "STPASA_REGIONSOLUTION")
            st_archive = cache_dir / (
                f"{year}-{month:02d}-{Path(urllib.parse.urlparse(st_url).path).name}"
            )
            if not st_archive.exists():
                print(f"download source=STPASA year={year} month={month:02d} url={st_url}", flush=True)
            _ensure_archive(st_url, st_archive)
            st_accepted = _read_stpasa_month(
                st_archive, stpasa_runs, start, end
            )
            print(
                f"processed year={year} month={month:02d} "
                f"pdpasa_rows={accepted} stpasa_rows={st_accepted} "
                f"rejected_late_rows={rejected_late}",
                flush=True,
            )
    for region in REGIONS:
        summary = _write_region(
            region,
            runs[region],
            stpasa_runs[region],
            output_dir / f"{region.lower()}_max_spare.npz",
        )
        print(f"summary={summary}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--output-dir", type=Path, default=Path("data/aemo_exogenous"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/aemo_exogenous/raw"))
    args = parser.parse_args()
    build_dataset(args.start_year, args.end_year, args.output_dir, args.cache_dir)


if __name__ == "__main__":
    main()
