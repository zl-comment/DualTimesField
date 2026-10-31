"""Historical replay input windows and native-quantile scores; no model calls.

The caller supplies the planned AEST dates and a fixed historical data snapshot.
Input availability timestamps remain archive proxies, not receipt evidence.
Candidate exogenous inputs are deliberately outside this module.
"""
from __future__ import annotations

from datetime import timedelta, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from forecasting.agentic_dev_eval.data import calendar_matrix

AEST = timezone(timedelta(hours=10))
QUANTILES = np.array([0.05, 0.10, 0.50, 0.90, 0.95], dtype=np.float64)


def _seconds(values: pd.Series) -> np.ndarray:
    stamps = pd.to_datetime(values, utc=True)
    if stamps.isna().any():
        raise ValueError("source timestamps must be present")
    return ((stamps - pd.Timestamp("1970-01-01", tz="UTC"))
            .dt.total_seconds().to_numpy(dtype=np.int64))


def build_records(frame: pd.DataFrame, region: str, dates: Iterable) -> tuple[list[dict], np.ndarray, list[dict]]:
    """Return input records, aligned [N,24] labels, and planned-origin issues.

    Dates are AEST midnight dates (naive dates are interpreted as fixed UTC+10).
    Each input uses the exact 72 hourly starts [O-73h,O-1h), available by O-10min.
    Its model calendar starts at O-1h for 25 steps; scoring uses the last 24.
    Missing or immature history never gets replaced with an earlier ``tail``.

    An invalid input has no record and is listed in ``missing``. Missing labels
    do not remove otherwise usable inputs: their aligned label cells are NaN
    and the issue is listed separately. Callers must choose a common complete
    label mask when scoring. No label or label-validity flag enters a record.
    """
    source = frame.loc[frame["region"] == region] if "region" in frame else frame
    source = source.reset_index(drop=True)
    starts = _seconds(source["delivery_start_aest"])
    available = _seconds(source["available_at_aest"])
    index = pd.Index(starts)
    if not index.is_unique:
        raise ValueError("hourly source must have unique delivery starts per region")
    prices = source["rrp_aud_per_mwh"].to_numpy(dtype=np.float64)
    demand = source["total_demand_mw"].to_numpy(dtype=np.float64)

    records, labels, missing = [], [], []
    seen = set()
    for date in dates:
        local = pd.Timestamp(date)
        local = local.tz_localize(AEST) if local.tzinfo is None else local.tz_convert(AEST)
        if local != local.normalize():
            raise ValueError("planned origins must be AEST midnight")
        origin = int(local.timestamp())
        if origin in seen:
            raise ValueError("planned origins must be unique")
        seen.add(origin)
        cutoff = origin - 600
        history_times = origin + np.arange(-73, -1, dtype=np.int64) * 3600
        model_times = origin + np.arange(-1, 24, dtype=np.int64) * 3600
        target_times = model_times[1:]
        history_index = index.get_indexer(history_times)
        present = history_index >= 0
        reasons = []
        if not present.all():
            reasons.append("history_missing")
        present_index = history_index[present]
        late_count = int(np.sum(available[present_index] > cutoff))
        if late_count:
            reasons.append("history_not_available")
        nonfinite_count = int(np.sum(~np.isfinite(prices[present_index])
                                    | ~np.isfinite(demand[present_index])))
        if nonfinite_count:
            reasons.append("history_nonfinite")
        identity = {"region": region, "origin_unix": origin, "origin_local": local.isoformat()}
        if reasons:
            missing.append({**identity, "reasons": reasons,
                            "history_missing_hours": int((~present).sum()),
                            "history_unavailable_hours": late_count,
                            "history_nonfinite_hours": nonfinite_count})
            continue

        history_local = pd.to_datetime(history_times, unit="s", utc=True).tz_convert(AEST)
        model_local = pd.to_datetime(model_times, unit="s", utc=True).tz_convert(AEST)
        records.append({
            **identity, "cutoff_unix": cutoff, "model_start_unix": int(model_times[0]),
            "history_time_unix": history_times.tolist(),
            "model_target_time_unix": model_times.tolist(),
            "target_time_unix": target_times.tolist(),
            "history_prices": prices[history_index].tolist(),
            "history_demand": demand[history_index].tolist(),
            "history_calendar": calendar_matrix(history_local).tolist(),
            "future_calendar": calendar_matrix(model_local).tolist(),
            "history_available_max_unix": int(available[history_index].max()),
            "publication_basis": "archive_proxy",
        })

        label_index = index.get_indexer(target_times)
        label_present = label_index >= 0
        actual = np.full(24, np.nan, dtype=np.float64)
        actual[label_present] = prices[label_index[label_present]]
        labels.append(actual)
        label_reasons = []
        if not label_present.all():
            label_reasons.append("label_missing")
        bad_label_count = int(np.sum(label_present & ~np.isfinite(actual)))
        if bad_label_count:
            label_reasons.append("label_nonfinite")
        if label_reasons:
            missing.append({**identity, "reasons": label_reasons,
                            "label_missing_hours": int((~label_present).sum()),
                            "label_nonfinite_hours": bad_label_count})

    return records, np.asarray(labels, dtype=np.float64).reshape(-1, 24), missing


def score_native(q: np.ndarray, y: np.ndarray) -> dict:
    """Score finite native [...,5] quantiles without sorting or clipping.

    Crossings are reported and remain in the score. Missing labels/predictions
    require an explicit common mask upstream, rather than an implicit drop here.
    """
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if q.shape != y.shape + (5,) or not y.size:
        raise ValueError("expected nonempty native quantiles [...,5] and matching actual [...]")
    if not np.isfinite(q).all() or not np.isfinite(y).all():
        raise ValueError("native scoring requires finite values on an explicit common mask")
    error = y[..., None] - q
    pinball = np.maximum(QUANTILES * error, (QUANTILES - 1) * error)
    return {
        "n_cells": int(y.size),
        "mae": float(np.abs(q[..., 2] - y).mean()),
        "quantile_score": float(2 * pinball.mean()),
        "pinball_by_quantile": {str(float(level)): float(pinball[..., i].mean())
                                for i, level in enumerate(QUANTILES)},
        "coverage_80": float(((y >= q[..., 1]) & (y <= q[..., 3])).mean()),
        "coverage_90": float(((y >= q[..., 0]) & (y <= q[..., 4])).mean()),
        "width_80": float((q[..., 3] - q[..., 1]).mean()),
        "width_90": float((q[..., 4] - q[..., 0]).mean()),
        "native_crossings": int(np.sum(np.diff(q, axis=-1) < 0)),
    }
