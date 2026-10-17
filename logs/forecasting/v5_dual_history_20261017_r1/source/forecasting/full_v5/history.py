"""Matured residual library, rolling statistics, state matching and fallbacks.

No training happens here.  The library is an append-only, origin-sorted array
bundle; queries can only see records whose full 24h label window was already
available at query time (j < t and a_j <= t).  All controls read the same
library with the same permissions.
"""
from __future__ import annotations

import numpy as np

from . import schema

HOUR = 3600
DAY = 86400


class HistoryLibrary:
    """Origin-sorted matured residual records for one region."""

    def __init__(self, origins, available_at, residuals, baseline, state6):
        self.origins = np.asarray(origins, dtype=np.int64)
        self.available_at = np.asarray(available_at, dtype=np.int64)
        self.residuals = np.asarray(residuals, dtype=np.float64)
        self.baseline = np.asarray(baseline, dtype=np.float64)
        self.state6 = np.asarray(state6, dtype=np.float64)
        if len(self.origins) and np.any(np.diff(self.origins) <= 0):
            raise ValueError("Library origins must be strictly increasing")
        shapes = {"residuals": self.residuals.shape[:2], "baseline": self.baseline.shape[:2],
                  "state6": self.state6.shape[:2]}
        if len(self.origins) and any(shape != (len(self.origins), schema.HORIZONS)
                                     for shape in shapes.values()):
            raise ValueError("Library array shape mismatch")
        if not np.isfinite(self.residuals).all():
            raise ValueError("Non-finite residuals in library")

    def __len__(self):
        return len(self.origins)

    @classmethod
    def from_segment(cls, origins, actual, baseline_q_sorted, z):
        """Build the legal out-of-sample library for one calendar segment.

        Residuals use the sorted frozen baseline median: r = Y - B_q50.
        """
        bq = np.asarray(baseline_q_sorted, dtype=np.float64)
        actual = np.asarray(actual, dtype=np.float64)
        residuals = actual - bq[:, :, 2]
        state6 = np.asarray(z, dtype=np.float64)[:, :, list(schema.MATCH_Z_INDICES)]
        available = np.asarray(origins, dtype=np.int64) + schema.HORIZONS * HOUR
        return cls(origins, available, residuals, bq, state6)

    def concat(self, other: "HistoryLibrary") -> "HistoryLibrary":
        if not len(other):
            return self
        if not len(self):
            return other
        if self.origins[-1] >= other.origins[0]:
            raise ValueError("Library segments must be disjoint and ordered")
        return HistoryLibrary(
            np.concatenate([self.origins, other.origins]),
            np.concatenate([self.available_at, other.available_at]),
            np.concatenate([self.residuals, other.residuals]),
            np.concatenate([self.baseline, other.baseline]),
            np.concatenate([self.state6, other.state6]),
        )

    def matured_bound(self, t: int) -> int:
        """Fast path when available_at == origin + 24h for every record."""
        return int(np.searchsorted(self.origins, t - schema.HORIZONS * HOUR, side="right"))

    def matured_mask(self, t: int) -> np.ndarray:
        return (self.available_at <= t) & (self.origins < t)


class HistoryAudit:
    def __init__(self):
        self.queries = 0
        self.m_short_fallback = 0     # 7d window with < 24 mature records
        self.m_long_fallback = 0      # 28d window with < 24 mature records
        self.v_closed_candidates = 0  # < 8 legal candidates
        self.v_closed_stats = 0       # 28d statistics insufficient
        self.v_open = 0
        self.pool_sizes: list[int] = []
        self.neighbor_ages: list[float] = []
        self.neighbor_distances: list[float] = []
        self.selection_samples: list[dict] = []

    def summary(self) -> dict:
        n = max(self.queries, 1)
        return dict(
            queries=self.queries,
            m_short_fallback_fraction=self.m_short_fallback / (n * schema.HORIZONS),
            m_long_fallback_fraction=self.m_long_fallback / (n * schema.HORIZONS),
            v_closed_candidate_fraction=self.v_closed_candidates / n,
            v_closed_stats_fraction=self.v_closed_stats / n,
            v_open_fraction=self.v_open / n,
            pool_size_median=float(np.median(self.pool_sizes)) if self.pool_sizes else None,
            pool_size_min=int(np.min(self.pool_sizes)) if self.pool_sizes else None,
            neighbor_age_days_median=(float(np.median(self.neighbor_ages)) if self.neighbor_ages else None),
            neighbor_distance_median=(float(np.median(self.neighbor_distances)) if self.neighbor_distances else None),
            selection_samples=self.selection_samples,
        )

    def merge(self, other: "HistoryAudit") -> "HistoryAudit":
        """Accumulate another audit (per-block engines) into this one."""
        self.queries += other.queries
        self.m_short_fallback += other.m_short_fallback
        self.m_long_fallback += other.m_long_fallback
        self.v_closed_candidates += other.v_closed_candidates
        self.v_closed_stats += other.v_closed_stats
        self.v_open += other.v_open
        self.pool_sizes.extend(other.pool_sizes)
        self.neighbor_ages.extend(other.neighbor_ages)
        self.neighbor_distances.extend(other.neighbor_distances)
        self.selection_samples.extend(other.selection_samples)
        return self


def _window_medians(values: np.ndarray):
    if len(values) < schema.WINDOW_MIN_RECORDS:
        return None
    return float(np.median(values))


def compute_m(library: HistoryLibrary, t: int, matured_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    """Spec 4.2: 8 rolling statistics per horizon, [24, 8].

    Returns (m, raw) where raw carries the untruncated 28d medians needed by
    the matched-history increments (v) and the per-window record counts.
    """
    m = np.zeros((schema.HORIZONS, schema.NM), dtype=np.float64)
    raw_med28 = np.full(schema.HORIZONS, np.nan)
    raw_abs28 = np.full(schema.HORIZONS, np.nan)
    counts = dict(n7=0, n28=0)
    origins = library.origins
    residuals = library.residuals
    baseline = library.baseline
    lo28 = int(np.searchsorted(origins[matured_idx], t - schema.WINDOW_LONG_DAYS * DAY, side="right"))
    lo7 = int(np.searchsorted(origins[matured_idx], t - schema.WINDOW_SHORT_DAYS * DAY, side="right"))
    w28 = matured_idx[lo28:]
    w7 = matured_idx[lo7:]
    counts["n7"], counts["n28"] = len(w7), len(w28)
    for h in range(schema.HORIZONS):
        r7 = residuals[w7, h]
        med7 = _window_medians(r7)
        abs7 = _window_medians(np.abs(r7))
        r28 = residuals[w28, h]
        med28 = _window_medians(r28)
        abs28 = _window_medians(np.abs(r28))
        if med7 is not None:
            m[h, 0] = np.clip(med7 / schema.PRICE_SCALE, -1.0, 1.0)
        if abs7 is not None:
            m[h, 2] = np.clip(abs7 / schema.PRICE_SCALE, 0.0, 1.0)
        if med28 is not None:
            m[h, 1] = np.clip(med28 / schema.PRICE_SCALE, -1.0, 1.0)
            raw_med28[h] = med28
        if abs28 is not None:
            m[h, 3] = np.clip(abs28 / schema.PRICE_SCALE, 0.0, 1.0)
            raw_abs28[h] = abs28
            y28 = baseline[w28, h, 2] + r28
            m[h, 4] = float(np.mean((y28 >= baseline[w28, h, 1]) & (y28 <= baseline[w28, h, 3]))) - 0.8
            m[h, 5] = float(np.mean((y28 >= baseline[w28, h, 0]) & (y28 <= baseline[w28, h, 4]))) - 0.9
        m[h, 6] = np.clip(len(r7) / schema.COUNT_NORMALISER[schema.WINDOW_SHORT_DAYS], -1.0, 1.0)
        m[h, 7] = np.clip(len(r28) / schema.COUNT_NORMALISER[schema.WINDOW_LONG_DAYS], -1.0, 1.0)
    stats_ok = {h: raw_med28[h] == raw_med28[h] and raw_abs28[h] == raw_abs28[h] for h in range(schema.HORIZONS)}
    return m, dict(raw_med28=raw_med28, raw_abs28=raw_abs28, counts=counts, stats_ok=stats_ok)


def _candidate_pool(library: HistoryLibrary, t: int, matured_idx: np.ndarray, query_hour: int) -> np.ndarray:
    lo = int(np.searchsorted(library.origins[matured_idx], t - schema.LOOKBACK_DAYS * DAY, side="right"))
    pool = matured_idx[lo:]
    hours = ((library.origins[pool] // HOUR + 10) % 24)
    return pool[hours == query_hour]


def _matched_increments(library: HistoryLibrary, pool: np.ndarray, h: int, query_state6: np.ndarray,
                        raw_med28: float, raw_abs28: float) -> tuple[float, float, list[float], list[float]]:
    distances = np.sum((library.state6[pool, h] - query_state6[h]) ** 2, axis=1)
    # pool is origin-ascending; a stable sort keeps the origin tie-break.
    order = np.argsort(distances, kind="stable")
    k = min(schema.MATCH_K, len(pool))
    chosen = pool[order[:k]]
    eta = k / schema.MATCH_K
    med = float(np.median(library.residuals[chosen, h]))
    med_abs = float(np.median(np.abs(library.residuals[chosen, h])))
    v_loc = eta * float(np.clip((med - raw_med28) / schema.PRICE_SCALE, -1.0, 1.0))
    v_abs = eta * float(np.clip((med_abs - raw_abs28) / schema.PRICE_SCALE, -1.0, 1.0))
    return v_loc, v_abs, distances[order[:k]].tolist(), library.origins[chosen].tolist()


def _recent_increments(library: HistoryLibrary, pool: np.ndarray, h: int,
                       raw_med28: float, raw_abs28: float) -> tuple[float, float, list[int]]:
    k = min(schema.MATCH_K, len(pool))
    chosen = pool[-k:]
    eta = k / schema.MATCH_K
    med = float(np.median(library.residuals[chosen, h]))
    med_abs = float(np.median(np.abs(library.residuals[chosen, h])))
    v_loc = eta * float(np.clip((med - raw_med28) / schema.PRICE_SCALE, -1.0, 1.0))
    v_abs = eta * float(np.clip((med_abs - raw_abs28) / schema.PRICE_SCALE, -1.0, 1.0))
    return v_loc, v_abs, library.origins[chosen].tolist()


class FeatureEngine:
    """Sequential per-origin feature computation over a fixed library.

    ``compute`` is stateless: the visible library slice for query t is decided
    purely by the maturity rule, so future labels can never influence earlier
    queries.  This is the structural equivalent of "predict and persist origin
    t, then absorb later labels".
    """

    def __init__(self, library: HistoryLibrary, audit_sample_every: int = 96):
        self.library = library
        self.audit = HistoryAudit()
        self.audit_sample_every = audit_sample_every
        self._fast_path = bool(len(library)) and bool(np.array_equal(
            library.available_at, library.origins + schema.HORIZONS * HOUR))

    def _matured(self, t: int) -> np.ndarray:
        if not len(self.library):
            return np.empty(0, dtype=np.int64)
        if self._fast_path:
            hi = self.library.matured_bound(t)
            return np.arange(hi, dtype=np.int64)
        return np.flatnonzero(self.library.matured_mask(t))

    def compute(self, query_origins, query_z, query_b_sorted) -> dict:
        """Return m [N,24,8], matched v [N,24,2] and recent v [N,24,2]."""
        query_origins = np.asarray(query_origins, dtype=np.int64)
        query_z = np.asarray(query_z, dtype=np.float64)
        query_b = np.asarray(query_b_sorted, dtype=np.float64)
        n = len(query_origins)
        m_out = np.zeros((n, schema.HORIZONS, schema.NM), dtype=np.float64)
        v_out = np.zeros((n, schema.HORIZONS, schema.NV), dtype=np.float64)
        vr_out = np.zeros((n, schema.HORIZONS, schema.NV), dtype=np.float64)
        query_state6 = query_z[:, :, list(schema.MATCH_Z_INDICES)]
        for i in range(n):
            t = int(query_origins[i])
            matured = self._matured(t)
            m, raw = compute_m(self.library, t, matured)
            m_out[i] = m
            if raw["counts"]["n7"] < schema.WINDOW_MIN_RECORDS:
                self.audit.m_short_fallback += schema.HORIZONS
            if raw["counts"]["n28"] < schema.WINDOW_MIN_RECORDS:
                self.audit.m_long_fallback += schema.HORIZONS
            query_hour = int((t // HOUR + 10) % 24)
            pool = _candidate_pool(self.library, t, matured, query_hour)
            self.audit.queries += 1
            self.audit.pool_sizes.append(int(len(pool)))
            stats_ready = all(raw["stats_ok"].values())
            closed = len(pool) < schema.MATCH_MIN_CANDIDATES
            if closed:
                self.audit.v_closed_candidates += 1
            elif not stats_ready:
                self.audit.v_closed_stats += 1
            else:
                self.audit.v_open += 1
                sample = (self.audit_sample_every > 0 and self.audit.queries % self.audit_sample_every == 0)
                for h in range(schema.HORIZONS):
                    v_loc, v_abs, dist, chosen = _matched_increments(
                        self.library, pool, h, query_state6[i], raw["raw_med28"][h], raw["raw_abs28"][h])
                    v_out[i, h] = (v_loc, v_abs)
                    vr_loc, vr_abs, recent_ids = _recent_increments(
                        self.library, pool, h, raw["raw_med28"][h], raw["raw_abs28"][h])
                    vr_out[i, h] = (vr_loc, vr_abs)
                    if sample and h == 0:
                        self.audit.neighbor_ages.extend(
                            float((t - o) / DAY) for o in chosen)
                        self.audit.neighbor_distances.extend(dist)
                        self.audit.selection_samples.append(dict(
                            query_origin=int(t), lead=h, matched_ids=[int(o) for o in chosen],
                            matched_distances=[round(d, 6) for d in dist],
                            recent_ids=[int(o) for o in recent_ids],
                            matched_available_at=[int(o + 24 * HOUR) for o in chosen]))
        return dict(m=m_out, v=v_out, v_recent=vr_out)

    def features_matrix(self, query_origins, query_z, query_b_sorted) -> np.ndarray:
        """Full [N, 24, 26] feature block [z | m | matched v]."""
        result = self.compute(query_origins, query_z, query_b_sorted)
        return np.concatenate([query_z, result["m"], result["v"]], axis=2), result

    def features_matrix_recent(self, query_z, result: dict) -> np.ndarray:
        return np.concatenate([query_z, result["m"], result["v_recent"]], axis=2)


class MaturedQueue:
    """Explicit sequential absorption of test labels for the final evaluation.

    The queue holds one candidate record per test origin (available_at =
    origin + 24h).  ``release_through(t)`` moves the frontier so that only
    records whose complete 24h label window was already available at time t
    enter the returned library; nothing with available_at > t is ever exposed.
    The caller predicts and persists its current block before advancing the
    frontier further, which is the frozen sequential final-evaluation contract.
    """

    def __init__(self, origins, actual, baseline_q_sorted, z):
        origins = np.asarray(origins, dtype=np.int64)
        bq = np.asarray(baseline_q_sorted, dtype=np.float64)
        actual = np.asarray(actual, dtype=np.float64)
        residuals = actual - bq[:, :, 2]
        state6 = np.asarray(z, dtype=np.float64)[:, :, list(schema.MATCH_Z_INDICES)]
        available = origins + schema.HORIZONS * HOUR
        order = np.argsort(available, kind="stable")  # == origin order on the hourly grid
        self._origins = origins[order]
        self._available = available[order]
        self._residuals = residuals[order]
        self._baseline = bq[order]
        self._state6 = state6[order]
        self._released = 0

    def release_through(self, t: int) -> int:
        """Advance the frontier to include every record with available_at <= t.

        Returns the number of newly released records; no later record is
        exposed (asserted), so predictions for queries <= t can never be
        influenced by labels maturing after t.
        """
        target = int(np.searchsorted(self._available, int(t), side="right"))
        if target > self._released:
            assert self._available[self._released:target].max() <= int(t)
        newly = target - self._released
        self._released = target
        return newly

    @property
    def released_count(self) -> int:
        return self._released

    def released(self) -> HistoryLibrary:
        return HistoryLibrary(
            self._origins[:self._released], self._available[:self._released],
            self._residuals[:self._released], self._baseline[:self._released],
            self._state6[:self._released])

    def pending_count(self) -> int:
        return len(self._origins) - self._released
