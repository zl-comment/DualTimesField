"""R12's explicitly approved historical demand proxy and frozen model adapter.

Using the same numerical channel does not establish equality of the historical
and predispatch demand definitions. Availability remains an archive proxy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.agentic_dev_eval.data import CALENDAR_FEATURES
from .totaldemand_information import vintage_label_unix


def selected_proxy(rows: pd.DataFrame, coverage_record: dict) -> dict | None:
    """Extract only the audited vintage; never repair it by selecting another.

    A source record marked uncovered returns None. A covered record must still
    match the source's visible 50-slot prefix and audit timestamps. Later changes
    are excluded before conflicts are checked; equal-value/equal-period copies
    are harmless. No missing value, period, vintage or hour is filled.
    """
    if not coverage_record["covered"]:
        return None
    origin = int(coverage_record["origin_unix"])
    cutoff = int(coverage_record["cutoff_unix"])
    region = coverage_record["region"]
    seqno, runno = int(coverage_record["selected_seqno"]), int(coverage_record["selected_runno"])
    schedule = vintage_label_unix(seqno)
    if (cutoff != origin - 600 or (origin + 10 * 3600) % 86400
            or schedule > cutoff or schedule != int(coverage_record["selected_schedule_label_unix"])):
        raise ValueError("selected audit timing is inconsistent")
    required = origin - 1800 + np.arange(50, dtype=np.int64) * 1800
    take = rows.loc[
        (rows["region"] == region) & (rows["seqno"] == seqno) & (rows["runno"] == runno)
        & (rows["lastchanged_unix"] <= cutoff) & rows["period_end_unix"].isin(required)
    ].copy()
    if (not np.isfinite(take["totaldemand"].to_numpy(dtype=np.float64)).all()
            or not np.isfinite(take["periodid"].to_numpy(dtype=np.float64)).all()
            or not (take["intervention"] == 0).all()
            or not (take["schedule_label_unix"] == schedule).all()):
        raise ValueError("selected source contains invalid values or identity")
    grouped = take.groupby("period_end_unix", sort=True)
    if (grouped[["totaldemand", "periodid"]].nunique(dropna=False) > 1).any().any():
        raise ValueError("selected source has a visible demand or period conflict")
    slots = grouped.agg(totaldemand=("totaldemand", "first"), periodid=("periodid", "first"),
                        lastchanged_unix=("lastchanged_unix", "max"))
    if not np.array_equal(slots.index.to_numpy(dtype=np.int64), required):
        raise ValueError("selected source lacks the exact 50 half-hour endpoints")
    changed = slots["lastchanged_unix"].to_numpy(dtype=np.int64)
    if int(changed.max()) != int(coverage_record["lastchanged_max"]):
        raise ValueError("selected source LASTCHANGED does not match the audit")
    values = slots["totaldemand"].to_numpy(dtype=np.float64)
    hourly = values.reshape(25, 2).mean(axis=1)
    return {
        "region": region, "origin_unix": origin, "cutoff_unix": cutoff,
        "seqno": seqno, "runno": runno, "schedule_label_unix": schedule,
        "run_age_seconds": origin - schedule,
        "half_hour_end_unix": required.tolist(), "half_hour_values": values.tolist(),
        "periodid": slots["periodid"].to_numpy(dtype=np.int64).tolist(),
        "half_hour_lastchanged_unix": changed.tolist(), "lastchanged_max": int(changed.max()),
        "hour_start_unix": (origin + np.arange(-1, 24, dtype=np.int64) * 3600).tolist(),
        "hourly_values": hourly.tolist(), "duplicate_rows": int(len(take) - 50),
        "publication_basis": "archive_proxy", "strict_asof_verified": False,
        "mapping": "approved_historical_proxy_mean_of_two_half_hours_same_named_demand_channel",
    }


def model_input(record: dict, future_demand=None) -> dict:
    """Build A, or B with only 25 future values added to total_demand_mw.

    The target is the same 72 past prices in both arms. This function reads no
    labels and does not generate, sort, clip or otherwise modify predictions.
    """
    target = np.array(record["history_prices"], dtype=np.float64, copy=True)
    demand = np.array(record["history_demand"], dtype=np.float64, copy=True)
    past = np.array(record["history_calendar"], dtype=np.float64, copy=True)
    future = np.array(record["future_calendar"], dtype=np.float64, copy=True)
    if (target.shape != (72,) or demand.shape != (72,) or past.shape != (72, 9)
            or future.shape != (25, 9)
            or not all(np.isfinite(x).all() for x in (target, demand, past, future))):
        raise ValueError("expected finite 72-hour history and 25-step calendar inputs")
    result = {
        "target": target,
        "past_covariates": {"total_demand_mw": demand,
                            **{name: past[:, i] for i, name in enumerate(CALENDAR_FEATURES)}},
        "future_covariates": {name: future[:, i] for i, name in enumerate(CALENDAR_FEATURES)},
    }
    if future_demand is not None:
        values = np.array(future_demand, dtype=np.float64, copy=True)
        if values.shape != (25,) or not np.isfinite(values).all():
            raise ValueError("future demand proxy must contain 25 finite hourly values")
        result["future_covariates"]["total_demand_mw"] = values
    return result
