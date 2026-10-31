"""Structural TOTALDEMAND coverage from monthly archives; no price scoring.

The sequence-derived clock is a schedule label, not a publication/receipt time.
Coverage is conditional on that label and the archived LASTCHANGED values being
at or before the cutoff. It can only be reported as ``archive_proxy``.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
from typing import Iterable, Sequence
import zipfile

import numpy as np
import pandas as pd

AEST = timezone(timedelta(hours=10))
CHUNK_ROWS = 100_000
REQUIRED_COLUMNS = (
    "PREDISPATCHSEQNO", "RUNNO", "REGIONID", "PERIODID", "INTERVENTION",
    "TOTALDEMAND", "LASTCHANGED", "DATETIME",
)
COLUMNS = (
    "seqno", "runno", "region", "periodid", "intervention", "totaldemand",
    "lastchanged_unix", "period_end_unix", "schedule_label_unix",
)


def vintage_label_unix(seqno: str | int) -> int:
    """YYYYMMDDPP -> AEST date 04:00 + PP * 30 min (01 labels 04:30)."""
    value = str(seqno)
    if len(value) != 10 or not value.isdigit() or not 1 <= int(value[8:]) <= 48:
        raise ValueError(f"invalid PREDISPATCHSEQNO: {seqno}")
    day = datetime.strptime(value[:8], "%Y%m%d").replace(tzinfo=AEST)
    return int((day + timedelta(hours=4, minutes=30 * int(value[8:]))).timestamp())


def _aest_seconds(values: pd.Series) -> np.ndarray:
    stamps = pd.to_datetime(values, format="%Y/%m/%d %H:%M:%S")
    delta = stamps - pd.Timestamp("1970-01-01") - pd.Timedelta(hours=10)
    return delta.dt.total_seconds().astype(np.int64).to_numpy()


def _convert_chunk(records: list[list[str]]) -> pd.DataFrame:
    part = pd.DataFrame(records, columns=REQUIRED_COLUMNS)
    seqno = pd.to_numeric(part["PREDISPATCHSEQNO"]).astype(np.int64)
    labels = {int(seq): vintage_label_unix(int(seq)) for seq in seqno.unique()}
    return pd.DataFrame({
        "seqno": seqno.to_numpy(),
        "runno": pd.to_numeric(part["RUNNO"]).astype(np.int64).to_numpy(),
        "region": part["REGIONID"].to_numpy(),
        "periodid": pd.to_numeric(part["PERIODID"]).astype(np.int64).to_numpy(),
        "intervention": np.zeros(len(part), dtype=np.int64),
        "totaldemand": pd.to_numeric(part["TOTALDEMAND"], errors="coerce").to_numpy(dtype=np.float64),
        "lastchanged_unix": _aest_seconds(part["LASTCHANGED"]),
        "period_end_unix": _aest_seconds(part["DATETIME"]),
        "schedule_label_unix": seqno.map(labels).to_numpy(dtype=np.int64),
    })


def read_month(path: str | Path, regions: Sequence[str]) -> pd.DataFrame:
    """Read only REGION_SOLUTION demand columns, in chunks, retaining RUNNO.

    Returns normalized ``COLUMNS``; filters to requested regions/intervention 0.
    Missing or nonnumeric demand is retained as NaN for coverage diagnostics.
    No duplicate removal, value filling, PRICE join or availability claim occurs.
    """
    pieces = []
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("monthly archive must contain one CSV")
        with archive.open(members[0]) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            header = next((row for row in reader if row and row[0] == "I"), None)
            if header is None or header[1:3] != ["PREDISPATCH", "REGION_SOLUTION"]:
                raise ValueError("archive is not PREDISPATCH REGION_SOLUTION")
            missing = set(REQUIRED_COLUMNS) - set(header)
            if missing:
                raise ValueError(f"monthly archive missing columns: {sorted(missing)}")
            positions = [header.index(name) for name in REQUIRED_COLUMNS]
            region_position, intervention_position = header.index("REGIONID"), header.index("INTERVENTION")
            pending = []
            for row in reader:
                if not row or row[0] != "D" or row[1:3] != ["PREDISPATCH", "REGION_SOLUTION"]:
                    continue
                if len(row) != len(header):
                    raise ValueError("REGION_SOLUTION row width differs from its header")
                if row[region_position] not in regions or row[intervention_position] != "0":
                    continue
                pending.append([row[position] for position in positions])
                if len(pending) == CHUNK_ROWS:
                    pieces.append(_convert_chunk(pending))
                    pending = []
            if pending:
                pieces.append(_convert_chunk(pending))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=COLUMNS)


def _window_status(indices, ends, changed, values, periods, required, cutoff):
    """Inspect only required slots in the visible prefix of one exact vintage."""
    take = indices[(changed[indices] <= cutoff) & np.isin(ends[indices], required)]
    take = take[np.argsort(ends[take], kind="stable")]
    nonfinite = bool(np.any(~np.isfinite(values[take])))
    duplicate = ends[take[1:]] == ends[take[:-1]]
    same_value = ((values[take[1:]] == values[take[:-1]])
                  | (np.isnan(values[take[1:]]) & np.isnan(values[take[:-1]])))
    conflict = bool(np.any(duplicate & (~same_value | (periods[take[1:]] != periods[take[:-1]]))))
    complete = len(take) - int(duplicate.sum()) == len(required)
    return {
        "covered": bool(complete and not conflict and not nonfinite),
        "conflict": conflict, "nonfinite": nonfinite,
        "lastchanged_max": int(changed[take].max()) if len(take) else None,
    }


def coverage(rows: pd.DataFrame, origins: Iterable[int], regions: Sequence[str]) -> list[dict]:
    """Per-region daily-O coverage, newest complete (seqno, RUNNO), no filling.

    25-step ends are O-1800 through O+86400; target-only ends are O+1800
    through O+86400. ``target24_only`` means target coverage without bridge
    coverage. Consistent duplicate rows are harmless; conflicting demand or
    PERIODID at a required endpoint invalidates that candidate. Later rows are
    excluded before conflicts are examined. Run age is O minus schedule label.
    Candidate counts cover schedule-visible vintages overlapping the 25h window.
    """
    frame = rows.reset_index(drop=True)
    ends = frame["period_end_unix"].to_numpy(dtype=np.int64)
    changed = frame["lastchanged_unix"].to_numpy(dtype=np.int64)
    values = frame["totaldemand"].to_numpy(dtype=np.float64)
    periods = frame["periodid"].to_numpy(dtype=np.int64)
    by_region = {region: [] for region in regions}
    for (region, seqno, runno), indices in frame.groupby(["region", "seqno", "runno"], sort=False).indices.items():
        if region in by_region:
            by_region[region].append({
                "seqno": int(seqno), "runno": int(runno), "indices": indices,
                "label": vintage_label_unix(int(seqno)),
                "first": int(ends[indices].min()), "last": int(ends[indices].max()),
            })
    for candidates in by_region.values():
        candidates.sort(key=lambda item: (item["seqno"], item["runno"]), reverse=True)

    result = []
    for origin_value in origins:
        origin = int(origin_value)
        if (origin + 10 * 3600) % 86400:
            raise ValueError("origins must be AEST 00:00")
        cutoff = origin - 600
        bridge = origin - 1800 + np.arange(50, dtype=np.int64) * 1800
        target = bridge[2:]
        for region in regions:
            candidates = [item for item in by_region[region]
                          if item["label"] <= cutoff and item["last"] >= bridge[0]
                          and item["first"] <= bridge[-1]]
            selected = target_selected = None
            conflicts = nonfinite = newer_incomplete = 0
            for item in candidates:
                full = _window_status(item["indices"], ends, changed, values, periods, bridge, cutoff)
                short = _window_status(item["indices"], ends, changed, values, periods, target, cutoff)
                conflicts += int(full["conflict"])
                nonfinite += int(full["nonfinite"])
                if selected is None:
                    if full["covered"]:
                        selected = (item, full)
                    else:
                        newer_incomplete += 1
                if target_selected is None and short["covered"]:
                    target_selected = (item, short)
            row = {
                "origin_unix": origin, "region": region, "cutoff_unix": cutoff,
                "publication_basis": "archive_proxy", "strict_asof_verified": False,
                "covered": selected is not None,
                "target24_covered": target_selected is not None,
                "target24_only": selected is None and target_selected is not None,
                "candidate_count": len(candidates), "candidate_conflict_count": conflicts,
                "candidate_nonfinite_count": nonfinite,
                "newer_incomplete_candidates": newer_incomplete if selected is not None else None,
                "selected_seqno": None, "selected_runno": None,
                "selected_schedule_label_unix": None, "run_age_seconds": None,
                "lastchanged_max": None,
                "target24_selected_seqno": None, "target24_selected_runno": None,
            }
            if selected is not None:
                item, state = selected
                row.update(selected_seqno=item["seqno"], selected_runno=item["runno"],
                           selected_schedule_label_unix=item["label"],
                           run_age_seconds=origin - item["label"],
                           lastchanged_max=state["lastchanged_max"])
            if target_selected is not None:
                item, _state = target_selected
                row.update(target24_selected_seqno=item["seqno"], target24_selected_runno=item["runno"])
            result.append(row)
    return result
