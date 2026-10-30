"""2022 development data loading with strict as-of visibility and origin-grid construction.

Self-contained on purpose (stdlib + numpy + pandas) so the chronos-2 venv and the
repo venv can both import it without repo-internal dependencies.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import protocol

DEFAULT_DATA_ROOT = "/root/workspace/Code/DualTimesField/data/holdout"

CALENDAR_FEATURES = (
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "month_sin", "month_cos", "is_weekend",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class OriginRecord:
    region: str
    origin_unix: int          # seconds UTC
    origin_local: str         # local wall-clock string for prompts and reports
    origin_dow: str
    origin_month: int
    half: str                 # 'H1' if local month <= 6 else 'H2'
    history_prices: np.ndarray  # (72,) raw AUD/MWh, available_at <= origin
    history_demand: np.ndarray  # (72,) raw MW, same rows
    history_calendar: np.ndarray   # (72, 9) closed-form calendar features
    future_calendar: np.ndarray    # (24, 9) features of the label delivery hours
    target_time_unix: np.ndarray   # (24,) label delivery instants
    actual: np.ndarray            # (24,) raw AUD/MWh labels
    visible_rows: int             # total rows with available_at <= origin

    def __getitem__(self, key: str):
        return getattr(self, key)


def calendar_matrix(local_timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Nine closed-form calendar features; deterministic, no fitted statistics."""
    hour = local_timestamps.hour.to_numpy()
    dow = local_timestamps.dayofweek.to_numpy()
    doy = local_timestamps.dayofyear.to_numpy()
    diy = np.where(np.asarray(local_timestamps.is_leap_year, dtype=bool), 366.0, 365.0)
    month = local_timestamps.month.to_numpy()
    cols = {
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        "doy_sin": np.sin(2 * np.pi * (doy - 1) / diy),
        "doy_cos": np.cos(2 * np.pi * (doy - 1) / diy),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12.0),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12.0),
        "is_weekend": (dow >= 5).astype(np.float64),
    }
    return np.column_stack([cols[name] for name in CALENDAR_FEATURES]).astype(np.float64)


def _utc_ns(series: pd.Series) -> np.ndarray:
    """UTC nanosecond int64 regardless of the pandas default time unit (ns/us/ms/s)."""
    parsed = pd.to_datetime(series, utc=True)
    dtype = str(parsed.dtype)
    unit = dtype[dtype.index("[") + 1:dtype.index(",")]
    scale = {"ns": 1, "us": 10**3, "ms": 10**6, "s": 10**9}[unit]
    return parsed.astype("int64").to_numpy() * scale


def load_region_frame(region: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> tuple[pd.DataFrame, str]:
    path = Path(data_root) / f"{region.lower()}_hourly.csv"
    frame = pd.read_csv(path)
    frame["delivery_start_aest"] = pd.to_datetime(frame["delivery_start_aest"], utc=True)
    frame["available_at_aest"] = pd.to_datetime(frame["available_at_aest"], utc=True)
    frame = frame.sort_values("delivery_start_aest").reset_index(drop=True)
    # 2023-2024 (and all 'test' rows) are final-test and forbidden: label selection
    # below requires split == 'validation' AND AEST calendar year == 2022.
    local = frame["delivery_start_aest"].dt.tz_convert(protocol.AEST)
    frame["_local"] = local
    frame["_year"] = local.dt.year
    return frame, str(path)

def build_origin_grid(region: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> tuple[list[OriginRecord], dict]:
    """Fixed 7-day AEST-midnight origin grid over the 2022 validation period.

    A grid date is scored only if (a) 72 rows are visible at the origin
    (available_at <= origin), and (b) the following 24 delivery rows stay inside
    the 2022 validation split. Dropped candidates are reported with reasons;
    nothing is shifted or substituted.
    """
    frame, csv_path = load_region_frame(region, data_root)
    delivery_ns = _utc_ns(frame["delivery_start_aest"])
    available_ns = _utc_ns(frame["available_at_aest"])
    rrp = frame["rrp_aud_per_mwh"].to_numpy(dtype=np.float64)
    demand = frame["total_demand_mw"].to_numpy(dtype=np.float64)
    split = frame["split"].to_numpy()
    year = frame["_year"].to_numpy()

    first = pd.Timestamp(protocol.GRID_FIRST_DATE).tz_localize(protocol.AEST)
    drop_log = []
    records: list[OriginRecord] = []
    candidate = first
    while candidate.year == protocol.DEV_YEAR:
        origin_ts = candidate.tz_convert("UTC")
        origin_ns = origin_ts.value  # UTC nanoseconds
        visible_end = int(np.searchsorted(available_ns, origin_ns, side="right"))
        if visible_end < protocol.HISTORY_HOURS:
            drop_log.append({"origin_local": str(candidate), "reason": "insufficient_history"})
            candidate += pd.Timedelta(days=protocol.GRID_STEP_DAYS)
            continue
        hist_idx = np.arange(visible_end - protocol.HISTORY_HOURS, visible_end)
        label_idx = hist_idx[-1] + 1 + np.arange(protocol.HORIZON_HOURS)
        label_idx = label_idx[label_idx < len(frame)]
        ok = label_idx.size == protocol.HORIZON_HOURS
        if ok:
            # The origin hour itself is invisible at the origin (available = origin+1h);
            # the label window is the first 24 deliveries with delivery >= origin.
            ok = bool(np.all(delivery_ns[label_idx] >= origin_ns))
        if ok:
            span_hours = (int(delivery_ns[label_idx[-1]]) - origin_ns) / 3.6e12
            ok = 0 < span_hours <= protocol.LABEL_WINDOW_MAX_HOURS
        if ok:
            ok = bool(np.all(split[label_idx] == protocol.DEV_SPLIT))
        if ok:
            ok = bool(np.all(year[label_idx] == protocol.DEV_YEAR))
        if not ok:
            drop_log.append({"origin_local": str(candidate), "reason": "label_window_not_in_2022_validation"})
            candidate += pd.Timedelta(days=protocol.GRID_STEP_DAYS)
            continue
        hist_local = frame["_local"].iloc[hist_idx]
        label_local = frame["_local"].iloc[label_idx]
        records.append(OriginRecord(
            region=region,
            origin_unix=int(origin_ts.value // 10**9),
            origin_local=str(candidate),
            origin_dow=candidate.day_name(),
            origin_month=int(candidate.month),
            half="H1" if candidate.month <= 6 else "H2",
            history_prices=rrp[hist_idx].copy(),
            history_demand=demand[hist_idx].copy(),
            history_calendar=calendar_matrix(pd.DatetimeIndex(hist_local)),
            future_calendar=calendar_matrix(pd.DatetimeIndex(label_local)),
            target_time_unix=(delivery_ns[label_idx] // 10**9).copy(),
            actual=rrp[label_idx].copy(),
            visible_rows=int(visible_end),
        ))
        candidate += pd.Timedelta(days=protocol.GRID_STEP_DAYS)

    info = {
        "region": region,
        "csv_path": csv_path,
        "csv_sha256": sha256_file(csv_path),
        "grid_first_date": protocol.GRID_FIRST_DATE,
        "grid_step_days": protocol.GRID_STEP_DAYS,
        "n_scored_origins": len(records),
        "dropped": drop_log,
    }
    return records, info


def visible_slice(frame: pd.DataFrame, origin_local: str, rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Last `rows` rows with available_at <= origin: (prices, demands, local_hours)."""
    origin_ts = pd.Timestamp(origin_local)
    mask = frame["available_at_aest"] <= origin_ts
    sub = frame.loc[mask].tail(rows)
    return (sub["rrp_aud_per_mwh"].to_numpy(dtype=np.float64),
            sub["total_demand_mw"].to_numpy(dtype=np.float64),
            sub["_local"].dt.hour.to_numpy())


