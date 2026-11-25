"""Paired calendar-block uncertainty summaries for frozen forecast losses.

The bootstrap is a NONOVERLAPPING calendar-block bootstrap, not an iid-day or
moving-block bootstrap. Blocks start at the first observed calendar day; the
last block can be shorter. Resample the original number of blocks with
replacement, retaining repeated days and their observation counts. The same
draw is used for every region; each region's mean is then equally weighted.
The percentile interval and empirical tail fraction are descriptive bootstrap
summaries. The tail fraction is not a null-centred hypothesis-test p-value.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm adjusted p-values, including the required cumulative maximum."""
    values = {key: float(value) for key, value in pvalues.items()}
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values.values()):
        raise ValueError("p-values must be finite and within [0,1]")
    adjusted, running = {}, 0.0
    for rank, (key, p) in enumerate(sorted(values.items(), key=lambda item: item[1])):
        running = max(running, min(1.0, (len(values) - rank) * p))
        adjusted[key] = running
    return {key: adjusted[key] for key in values}


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def block_bootstrap_and_jackknife(
    values_by_region: Mapping[str, Any], days_by_region: Mapping[str, Any], *,
    block_days: int = 7, repetitions: int = 2000, seed: int = 2026,
    alternative: str = "greater", confidence: float = 0.95,
) -> dict[str, Any]:
    """Summarize paired differences; greater means positive is improvement.

    Pass one loss difference and its LOCAL calendar day per forecast origin.
    Convert Unix seconds to the intended timezone before extracting dates;
    this function intentionally does not guess a timezone. Missing days retain
    their calendar positions. Each region must have observations in every
    block, and must remain nonempty when any one day is deleted. These checks
    prevent silently dropping regions or changing macro weights in a draw.

    The jackknife statistic is max |full - leave_one_day_out| / |full|. If
    full=0 and a deletion changes it, the share is undefined (None), not zero.
    This is a concentration diagnostic, not a jackknife confidence interval.
    """
    block_days = _positive_integer(block_days, "block_days")
    repetitions = _positive_integer(repetitions, "repetitions")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if alternative not in ("greater", "less"):
        raise ValueError("alternative must be greater or less")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly within (0,1)")
    if not values_by_region or set(values_by_region) != set(days_by_region):
        raise ValueError("values and days must contain the same nonempty set of regions")
    values, dates = {}, {}
    for region in sorted(values_by_region):
        raw = np.asarray(values_by_region[region])
        if raw.ndim != 1 or not raw.size or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
            raise ValueError(f"{region}: values must be a nonempty finite real vector")
        day = np.asarray(days_by_region[region])
        # Numeric Unix seconds are not calendar days and must not silently cast.
        if day.shape != raw.shape or day.dtype.kind in "iufb":
            raise ValueError(f"{region}: supply one calendar date per value, not numeric timestamps")
        try:
            day = day.astype("datetime64[D]")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{region}: invalid calendar dates") from exc
        if np.isnat(day).any():
            raise ValueError(f"{region}: calendar dates cannot contain NaT")
        values[region], dates[region] = raw.astype(np.float64), day
    first = min(day.min() for day in dates.values())
    last = max(day.max() for day in dates.values())
    n_days = int((last - first) / np.timedelta64(1, "D")) + 1
    n_blocks = (n_days + block_days - 1) // block_days
    if n_blocks < 2:
        raise ValueError("At least two calendar blocks are required")
    day_sums, day_counts = [], []
    for region in values:
        codes = (dates[region] - first).astype(np.int64)
        day_sums.append(np.bincount(codes, weights=values[region], minlength=n_days))
        day_counts.append(np.bincount(codes, minlength=n_days))
    day_sums, day_counts = np.asarray(day_sums), np.asarray(day_counts)
    starts = np.arange(0, n_days, block_days)
    block_sums = np.add.reduceat(day_sums, starts, axis=1)
    block_counts = np.add.reduceat(day_counts, starts, axis=1)
    if np.any(block_counts == 0):
        raise ValueError("Every region must have observations in every calendar block")
    total_sums, total_counts = day_sums.sum(axis=1), day_counts.sum(axis=1)
    remaining_counts = total_counts[:, None] - day_counts
    if np.any(remaining_counts <= 0):
        raise ValueError("Every region must remain nonempty after deleting one day")
    full = float(np.mean(total_sums / total_counts))
    rng = np.random.default_rng(int(seed))
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected = rng.integers(0, n_blocks, size=n_blocks)
        # Indexing, unlike membership masks, preserves replacement multiplicity.
        draws[index] = np.mean(block_sums[:, selected].sum(axis=1)
                               / block_counts[:, selected].sum(axis=1))
    tail = float(np.mean(draws <= 0 if alternative == "greater" else draws >= 0))
    alpha = (1 - confidence) / 2
    low, high = np.quantile(draws, [alpha, 1 - alpha])
    leave_one_out = ((total_sums[:, None] - day_sums) / remaining_counts).mean(axis=0)
    max_change = float(np.max(np.abs(full - leave_one_out)))
    share = max_change / abs(full) if full != 0 else (0.0 if max_change == 0 else None)
    return {
        "delta": full, "ci_low": float(low), "ci_high": float(high),
        "bootstrap_tail_probability": tail,
        "bootstrap_p_leq0" if alternative == "greater" else "bootstrap_p_geq0": tail,
        "jackknife_max_day_share": share,
        "method": "nonoverlapping_calendar_block_bootstrap",
        "block_days": block_days, "n_days": n_days, "n_blocks": n_blocks,
        "last_block_days": n_days - int(starts[-1]), "repetitions": repetitions,
        "seed": int(seed), "confidence": float(confidence), "alternative": alternative,
        "region_aggregation": "equal_region_means_of_resampled_origin_losses",
        "tail_probability_interpretation": "empirical uncentered bootstrap tail fraction; not a strict null-test p-value",
        "compatibility_p_field": "historical field name only; same empirical tail fraction",
    }
