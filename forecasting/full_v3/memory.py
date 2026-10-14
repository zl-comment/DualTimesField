"""Calendar analogs whose complete history and outcome were available as of origin.

The library may include later evaluation rows. Eligibility is checked per query
using raw delivery/availability timestamps; no transform is fitted here and no
outcome value participates in candidate selection. All descriptor features are
the already-frozen features from the candidate's historical forecast origin.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.aligned_inputs import AlignedSplit, INPUT_KEYS


def _integer_array(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError(f"{name} must contain integer Unix seconds")
    return array.astype(np.int64, copy=False)


def _unix_seconds(values, timezone, *, round_release_up=False):
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.isna().any():
        raise ValueError("Missing delivery or availability timestamp")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    nanoseconds = parsed.dt.tz_convert("UTC").to_numpy(dtype="datetime64[ns]").astype(np.int64)
    if not round_release_up and np.any(nanoseconds % 1_000_000_000):
        raise ValueError("Delivery timestamps must be exact whole seconds")
    # A release a fraction of a second after an hourly query is not available
    # at that query. Ceiling preserves that distinction in integer-second maps.
    return (nanoseconds + (999_999_999 if round_release_up else 0)) // 1_000_000_000


def load_availability(path: Path | str, region: str, *,
                      delivery_column="delivery_start_aest", available_column="available_at_aest",
                      region_column="region", timezone="Australia/Brisbane") -> dict[int, int]:
    """Read actual raw release timestamps; never synthesize delivery + one hour.

    Multi-region files are supported by selecting the requested region. Missing
    hours remain absent and will invalidate any candidate needing those hours.
    Duplicate deliveries, missing timestamps and release-before-delivery rows
    are rejected for the selected region.
    """
    frame = pd.read_csv(path, usecols=[delivery_column, available_column, region_column])
    frame = frame.loc[frame[region_column].eq(region)]
    if frame.empty:
        raise ValueError(f"No availability rows for region {region}")
    deliveries = _unix_seconds(frame[delivery_column], timezone)
    available = _unix_seconds(frame[available_column], timezone, round_release_up=True)
    if len(np.unique(deliveries)) != len(deliveries):
        raise ValueError("Duplicate delivery timestamps in availability data")
    if np.any(deliveries % 3600):
        raise ValueError("Delivery timestamps must lie on the hourly grid")
    if np.any(available < deliveries):
        raise ValueError("Actual observations cannot be available before delivery")
    return dict(zip(deliveries.tolist(), available.tolist()))


class HistoricalMemory:
    """Immutable-layout library with deterministic, padded calendar retrieval.

    Defaults provide at most 49 slots: years 1/2/3, each with day offsets -7..7,
    followed by weekly lags 7/14/21/28 days. Duplicates and unavailable candidates
    are removed preserving that order; remaining slots are -1. ``DateOffset``
    maps leap day to Feb 28 in a non-leap year before applying day offsets.

    ``levels`` is the raw median of each candidate's preceding 24 prices.
    ``background_descriptors`` concatenates four normalized historical summaries
    (price median/std/last, load mean), historical-origin future calendar, CTF
    features and origin context. ``event_descriptors`` concatenates future
    exogenous and quantile-exogenous trajectories from that same origin. No
    candidate future realized value enters either descriptor.
    """

    load_availability = staticmethod(load_availability)

    def __init__(self, splits: Mapping[str, AlignedSplit], availability: Mapping[int, int], *, config=None):
        settings = dict(config or {})
        if "memory" in settings:
            settings = dict(settings["memory"])
        self.timezone = settings.get("timezone", "Australia/Brisbane")
        self.year_lags = tuple(settings.get("year_lags", (1, 2, 3)))
        self.day_offsets = tuple(settings.get("day_offsets", range(-7, 8)))
        self.weekly_lags_days = tuple(settings.get("weekly_lags_days", (7, 14, 21, 28)))
        for name, values, positive in (("year_lags", self.year_lags, True),
                                       ("day_offsets", self.day_offsets, False),
                                       ("weekly_lags_days", self.weekly_lags_days, True)):
            if any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
                   or (positive and v <= 0) for v in values):
                raise ValueError(f"Invalid {name}")
        self.max_candidates = len(self.year_lags) * len(self.day_offsets) + len(self.weekly_lags_days)
        if not splits or self.max_candidates == 0:
            raise ValueError("Memory requires nonempty splits and candidate offsets")
        ordered = list(splits.values())
        first = ordered[0]
        self.region = first.metadata.get("region")
        self.preprocessing_hash = first.metadata.get("preprocessing_hash")
        self.target_mean = float(first.metadata["target_mean"])
        self.target_std = float(first.metadata["target_std"])
        if not np.isfinite([self.target_mean, self.target_std]).all() or self.target_std <= 0:
            raise ValueError("Invalid frozen target scale")
        for split in ordered:
            if split.metadata.get("region") != self.region:
                raise ValueError("A historical memory must contain one region")
            if (split.metadata["target_mean"] != self.target_mean
                    or split.metadata["target_std"] != self.target_std
                    or split.metadata.get("preprocessing_hash") != first.metadata.get("preprocessing_hash")
                    or split.metadata.get("price_transform", "identity") != "identity"):
                raise ValueError("All memory splits must use the same frozen affine preprocessing")
            if set(split.inputs) != set(INPUT_KEYS):
                raise ValueError("Memory requires all seven canonical input blocks")
            if split.inputs["history_values"].shape[1:] != (72, 2) or split.actual.shape != (len(split), 24):
                raise ValueError("Memory requires 72-hour two-channel histories and 24-hour targets")
            if split.target_time_unix.shape != split.actual.shape:
                raise ValueError("Target timestamp shape mismatch")
            if any(len(values) != len(split) for values in split.inputs.values()):
                raise ValueError("Memory inputs are not row-aligned")
            _integer_array(split.origin_unix, "origins")
            _integer_array(split.target_time_unix, "target timestamps")
        origins = np.concatenate([s.origin_unix for s in ordered]).astype(np.int64)
        if not len(origins) or np.any(origins % 3600):
            raise ValueError("Memory origins must be nonempty hourly timestamps")
        order = np.argsort(origins, kind="stable")
        origins = origins[order]
        curves = np.concatenate([s.actual for s in ordered])[order]
        targets = np.concatenate([s.target_time_unix for s in ordered])[order]
        inputs = {key: np.concatenate([s.inputs[key] for s in ordered])[order] for key in INPUT_KEYS}
        duplicate = np.flatnonzero(np.diff(origins) == 0)
        for index in duplicate:
            if (not np.array_equal(curves[index], curves[index + 1], equal_nan=True)
                    or not np.array_equal(targets[index], targets[index + 1])
                    or any(not np.array_equal(value[index], value[index + 1], equal_nan=True)
                           for value in inputs.values())):
                raise ValueError("Conflicting duplicate memory origin")
        keep = np.r_[True, np.diff(origins) != 0]
        self.origins = origins[keep]
        self.curves = curves[keep].astype(np.float64)
        self.target_time_unix = targets[keep].astype(np.int64)
        self.inputs = {key: value[keep] for key, value in inputs.items()}
        self.origin_unix = self.origins
        history = self.inputs["history_values"]
        recent_price = history[:, -24:, 0].astype(np.float64)
        normalized_level = np.median(recent_price, axis=1)
        self.levels = np.median(recent_price * self.target_std + self.target_mean, axis=1)
        summaries = np.column_stack((normalized_level, history[:, -24:, 0].std(axis=1),
                                     history[:, -1, 0], history[:, -24:, 1].mean(axis=1)))
        self.background_descriptors = np.concatenate([summaries, *[
            self.inputs[key].reshape(len(self.origins), -1)
            for key in ("future_calendar", "ctf_exogenous", "origin_context")]], axis=1)
        self.event_descriptors = np.concatenate([
            self.inputs[key].reshape(len(self.origins), -1)
            for key in ("future_exogenous", "quantile_exogenous")], axis=1)
        self.descriptors = {"background": self.background_descriptors, "event": self.event_descriptors}

        if availability:
            delivery = _integer_array(list(availability.keys()), "availability delivery keys")
            releases = _integer_array(list(availability.values()), "availability release values")
            if np.any(delivery % 3600) or np.any(releases < delivery):
                raise ValueError("Invalid raw delivery/availability ordering")
            sorted_rows = np.argsort(delivery)
            delivery, releases = delivery[sorted_rows], releases[sorted_rows]
            required = self.origins[:, None] + np.arange(-72, 24)[None, :] * 3600
            positions = np.searchsorted(delivery, required)
            safe_positions = np.minimum(positions, len(delivery) - 1)
            found = (positions < len(delivery)) & (delivery[safe_positions] == required)
            self.complete = found.all(axis=1)
            self.max_availability = np.max(releases[safe_positions], axis=1)
        else:
            self.complete = np.zeros(len(self.origins), dtype=bool)
            self.max_availability = np.full(len(self.origins), np.iinfo(np.int64).max, dtype=np.int64)
        expected_targets = self.origins[:, None] + np.arange(24)[None, :] * 3600
        self.complete &= (self.target_time_unix == expected_targets).all(axis=1)
        self.max_availability[~self.complete] = np.iinfo(np.int64).max

    def _calendar_candidates(self, query_origins):
        dates = pd.to_datetime(query_origins, unit="s", utc=True).tz_convert(self.timezone)
        columns = []
        for year in self.year_lags:
            center = dates - pd.DateOffset(years=int(year))
            for day in self.day_offsets:
                shifted = center + pd.DateOffset(days=int(day))
                columns.append(shifted.tz_convert("UTC").to_numpy(dtype="datetime64[s]").astype(np.int64))
        for day in self.weekly_lags_days:
            shifted = dates - pd.DateOffset(days=int(day))
            columns.append(shifted.tz_convert("UTC").to_numpy(dtype="datetime64[s]").astype(np.int64))
        return np.column_stack(columns)

    def candidate_indices(self, query_origins) -> np.ndarray:
        queries = _integer_array(query_origins, "query origins")
        if queries.ndim != 1 or np.any(queries % 3600):
            raise ValueError("Query origins must be a one-dimensional hourly array")
        if not len(queries):
            return np.full((0, self.max_candidates), -1, dtype=np.int64)
        stamps = self._calendar_candidates(queries)
        positions = np.searchsorted(self.origins, stamps)
        safe = np.minimum(positions, len(self.origins) - 1)
        valid = (positions < len(self.origins)) & (self.origins[safe] == stamps)
        valid &= self.complete[safe] & (self.max_availability[safe] <= queries[:, None])
        # Delivery must also have finished, even if a bad upstream availability
        # timestamp claims release at the beginning of the delivery hour.
        valid &= (self.origins[safe] + 24 * 3600 <= queries[:, None])
        result = np.full(positions.shape, -1, dtype=np.int64)
        counts = np.zeros(len(queries), dtype=np.int64)
        rows = np.arange(len(queries))
        for column in range(positions.shape[1]):
            candidate = positions[:, column]
            unique = ~np.any(result == candidate[:, None], axis=1)
            use = valid[:, column] & unique
            result[rows[use], counts[use]] = candidate[use]
            counts += use
        return result

    @staticmethod
    def candidate_validity(indices) -> np.ndarray:
        return np.asarray(indices) >= 0

    def candidate_max_availability(self, indices) -> np.ndarray:
        indices = _integer_array(indices, "candidate indices")
        if np.any(indices < -1) or np.any(indices >= len(self.origins)):
            raise ValueError("Candidate index outside the memory")
        return np.where(indices >= 0, self.max_availability[np.maximum(indices, 0)], -1)
