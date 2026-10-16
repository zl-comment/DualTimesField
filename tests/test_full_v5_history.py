"""History-engine unit tests: maturity, sequential labels, dedup, fallbacks.

Spec section 9 items 1-5, 8, 11 (engine level) and the calendar arithmetic.
"""
import numpy as np
import pandas as pd
import pytest

from forecasting.full_v5 import data, history, schema

HOUR = 3600
DAY = 86400
T0 = schema.unix_timestamp("2022-07-01")


def make_library(origins, residuals, state6=None, baseline=None, avail=None):
    origins = np.asarray(origins, dtype=np.int64)
    n = len(origins)
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.ndim == 1:
        residuals = np.tile(residuals[:, None], (1, schema.HORIZONS))
    if state6 is None:
        state6 = np.zeros((n, schema.HORIZONS, 6))
    if baseline is None:
        baseline = np.tile(np.array([-20.0, -10.0, 0.0, 10.0, 20.0]), (n, schema.HORIZONS, 1))
    if avail is None:
        avail = origins + 24 * HOUR
    return history.HistoryLibrary(origins, avail, residuals, baseline, np.asarray(state6))


def run_engine(library, query_origins, query_z=None, query_b=None):
    n = len(query_origins)
    if query_z is None:
        query_z = np.zeros((n, schema.HORIZONS, schema.NZ))
        query_z[:, :, 0] = 1.0
    if query_b is None:
        query_b = np.tile(np.array([-20.0, -10.0, 0.0, 10.0, 20.0]), (n, schema.HORIZONS, 1))
    engine = history.FeatureEngine(library, audit_sample_every=0)
    return engine.compute(query_origins, query_z, query_b), engine


def test_calendar_arithmetic_matches_pandas():
    rng = np.random.default_rng(7)
    stamps = rng.integers(0, 1_800_000_000, 500) * 3600
    index = pd.to_datetime(stamps, unit="s", utc=True).tz_convert("Australia/Brisbane")
    assert np.array_equal(data.aest_hour(stamps), index.hour.to_numpy())
    assert np.array_equal(data.aest_dow(stamps), index.dayofweek.to_numpy())
    assert np.array_equal((data.aest_dow(stamps) >= 5),
                          index.dayofweek.to_numpy() >= 5)


def test_maturity_boundary_inclusive_at_available_time():
    # A record with a_j == t is visible (inclusive boundary); a record with
    # a_j == t + 1s is explicitly constructed and must NOT be visible.
    origins = np.arange(T0, T0 + 30 * DAY, HOUR)
    avail = origins + 24 * HOUR
    library = make_library(origins, np.full(len(origins), 7.0), avail=avail)
    t = T0 + 10 * DAY
    engine = history.FeatureEngine(library)
    visible = engine._matured(t)
    assert int(visible[-1]) == int(np.searchsorted(origins, t - 24 * HOUR, side="right")) - 1
    assert library.origins[visible[-1]] + 24 * HOUR == t   # last visible matures exactly at t
    # The t+1s record: delay that record's availability by one second.
    delayed = avail.copy()
    delayed[visible[-1]] = t + 1                            # a_j = t + 1s
    one_second_later = history.HistoryLibrary(
        origins, delayed, library.residuals, library.baseline, library.state6)
    visible_later = history.FeatureEngine(one_second_later)._matured(t)
    assert int(visible_later[-1]) == int(visible[-1]) - 1   # exactly that record dropped
    assert one_second_later.available_at[visible_later].max() <= t
    assert one_second_later.origins[visible_later].max() < t
    # General (non fast-path) availability keeps the same boundary semantics.
    shuffled_avail = avail.copy()
    shuffled_avail[:3] -= 3600  # some records mature even earlier
    general = history.HistoryLibrary(origins, shuffled_avail, library.residuals,
                                    library.baseline, library.state6)
    mask_visible = history.FeatureEngine(general)._matured(t)
    assert np.all(general.available_at[mask_visible] <= t)
    assert np.all(general.origins[mask_visible] < t)


def test_matured_queue_releases_only_past_frontier():
    origins = np.arange(T0, T0 + 10 * DAY, HOUR)
    rng = np.random.default_rng(31)
    actual = 100.0 + rng.normal(0, 5, size=(len(origins), schema.HORIZONS))
    bq = np.tile(np.array([-20.0, -10.0, 0.0, 10.0, 20.0]), (len(origins), schema.HORIZONS, 1))
    z = np.zeros((len(origins), schema.HORIZONS, schema.NZ))
    queue = history.MaturedQueue(origins, actual, bq, z)
    t = int(origins[48])
    assert queue.release_through(t) == 25        # origins[0]..origins[24]: a_j <= t
    assert queue.released_count == 25
    released = queue.released()
    assert released.origins[-1] == origins[24]
    assert released.available_at.max() == origins[24] + 24 * HOUR <= t
    assert queue.pending_count() == len(origins) - 25
    assert queue.release_through(int(origins[100])) == 77 - 25
    assert queue.released().origins[-1] == origins[76]
    # The frontier never exposes records maturing after the query time.
    frontier = queue.released()
    assert frontier.available_at.max() == origins[76] + 24 * HOUR <= origins[100]


def test_future_label_changes_never_affect_earlier_queries():
    origins = np.arange(T0, T0 + 40 * DAY, HOUR)
    rng = np.random.default_rng(11)
    residuals = rng.normal(0, 20, size=(len(origins), schema.HORIZONS))
    library = make_library(origins, residuals)
    queries = origins[30 * 24:35 * 24]
    z = np.zeros((len(queries), schema.HORIZONS, schema.NZ)); z[:, :, 0] = 1.0
    b = np.tile(np.array([-15.0, -8.0, 0.0, 8.0, 15.0]), (len(queries), schema.HORIZONS, 1))
    before, _ = run_engine(library, queries, z, b)
    # Poison only labels that are NOT visible at the first query (origin > t0 - 24h).
    poisoned = residuals.copy()
    poisoned[origins > queries[0] - 24 * HOUR] += 500.0
    after, _ = run_engine(make_library(origins, poisoned), queries, z, b)
    assert np.array_equal(before["m"][0], after["m"][0])
    assert np.array_equal(before["v"][0], after["v"][0])
    # A later query sees the poisoned labels and must differ.
    assert not np.array_equal(before["m"][-1], after["m"][-1])


def test_segment_windows_and_fit_availability():
    origins = np.arange(schema.unix_timestamp("2022-09-29"),
                        schema.unix_timestamp("2022-10-02"), HOUR, dtype=np.int64)
    dfit = data.window_in_segment(origins, *schema.segment_bounds()["dfit"])
    v1 = data.window_in_segment(origins, *schema.segment_bounds()["v1"])
    assert dfit.sum() == 25 and v1.sum() == 24  # 09-29T00..09-30T00 in Dfit; 10-01 in V1
    fit = dfit & data.fit_available(origins, schema.fit_cutoffs()["dfit"])
    assert int(fit.sum()) == 24  # origin 09-30T00's label matures exactly at the cutoff
    dropped = dfit & ~data.fit_available(origins, schema.fit_cutoffs()["dfit"])
    assert origins[dropped] == schema.unix_timestamp("2022-09-30")


def test_fixed_lead_delivery_hours_counted_once():
    origins = np.arange(T0, T0 + 40 * DAY, HOUR)
    library = make_library(origins, np.arange(len(origins), dtype=float))
    result, engine = run_engine(library, origins[35 * 24:36 * 24])
    m = result["m"]
    # The 7d window is (t-7d, t) intersected with maturity (origin <= t-24h):
    # at most 168 - 24 = 144 hourly origins, one per delivery hour per lead.
    assert m[:, 0, 6].max() == 144 / 168
    assert m[:, 0, 7].max() == 648 / 672
    # Delivery hour d appears exactly once per lead in the long window.
    t = int(origins[8 * 24 + 5])
    matured = engine._matured(t)
    w28 = matured[int(np.searchsorted(origins[matured], t - 28 * DAY, side="right")):]
    deliveries = origins[w28] + 5 * HOUR
    assert len(np.unique(deliveries)) == len(deliveries)


def test_match_tie_break_prefers_older_origin():
    origins = np.arange(T0, T0 + 12 * DAY, HOUR)
    rng = np.random.default_rng(3)
    state = np.zeros((len(origins), schema.HORIZONS, 6))
    state[:, :, 0] = rng.normal(size=len(origins))[:, None] * 0 + 0.5  # identical states
    residuals = np.linspace(-5, 5, len(origins))[:, None] * np.ones((1, schema.HORIZONS))
    library = make_library(origins, residuals, state6=state)
    pool = np.arange(len(origins) - 20, len(origins))
    v_loc, v_abs, dist, chosen = history._matched_increments(
        library, pool, 0, np.zeros((schema.HORIZONS, 6)), 0.0, 0.0)
    assert list(np.asarray(chosen)) == sorted(chosen)  # ascending origin order under ties


def test_fallbacks_are_exact():
    # Empty library.
    empty = make_library(np.array([], dtype=np.int64), np.zeros((0, schema.HORIZONS)))
    result, engine = run_engine(empty, np.array([T0 + 5 * DAY]))
    assert np.all(result["m"][0, :, :6] == 0.0)
    assert np.all(result["m"][0, :, 6:] == 0.0)
    assert np.all(result["v"] == 0.0) and np.all(result["v_recent"] == 0.0)
    # Fewer than 8 same-hour candidates -> matched branch closed.
    origins = np.arange(T0, T0 + 6 * DAY, HOUR)  # 6 same-hour candidates mature
    library = make_library(origins, np.full(len(origins), 3.0))
    query = np.array([T0 + 6 * DAY + 10 * HOUR])
    result, engine = run_engine(library, query)
    assert engine.audit.v_closed_candidates == 1 and engine.audit.v_open == 0
    assert np.all(result["v"] == 0.0)
    # 28d window with 23 mature records: statistics zero, counts retained.
    origins23 = np.arange(T0, T0 + 23 * HOUR, HOUR)
    library23 = make_library(origins23, np.full(23, 4.0))
    result23, _ = run_engine(library23, np.array([T0 + 2 * DAY]))
    assert np.all(result23["m"][0, :, :6] == 0.0)
    assert np.allclose(result23["m"][0, :, 6], 23 / 168)
    assert np.allclose(result23["m"][0, :, 7], 23 / 672)
    assert np.all(result23["v"] == 0.0)


def test_constant_rolling_bias_gives_no_matched_increment():
    origins = np.arange(T0, T0 + 30 * DAY, HOUR)
    library = make_library(origins, np.full(len(origins), 12.0))
    rng = np.random.default_rng(5)
    state = rng.normal(0, 1, size=(len(origins), schema.HORIZONS, 6))
    library.state6 = state
    result, _ = run_engine(library, origins[28 * 24:29 * 24])
    # median over any neighbourhood equals the 28d median -> v exactly zero.
    assert np.allclose(result["v"][0], 0.0, atol=1e-12)


def test_state_determined_residuals_recover_direction():
    origins = np.arange(T0, T0 + 40 * DAY, HOUR)
    rng = np.random.default_rng(13)
    state = np.zeros((len(origins), schema.HORIZONS, 6))
    state[:, :, 0] = rng.normal(0, 0.4, size=len(origins))[:, None]  # single informative axis
    # Asymmetric three-state residual: middle state is the stable 28d median
    # (a symmetric +-30 residual would make the median itself flip signs).
    residual = (40.0 * (state[:, :, 0] > 0.2) - 10.0 * (state[:, :, 0] < -0.2)
                + rng.normal(0, 2, size=len(origins))[:, None])
    library = make_library(origins, np.asarray(residual, dtype=np.float64), state6=state)
    queries = origins[35 * 24:37 * 24]
    query_z = np.zeros((len(queries), schema.HORIZONS, schema.NZ)); query_z[:, :, 0] = 1.0
    query_z[:, :, list(schema.MATCH_Z_INDICES)[0]] = state[35 * 24:37 * 24, :, 0]
    b = np.zeros((len(queries), schema.HORIZONS, 5))
    result, _ = run_engine(library, queries, query_z, b + np.array([-10, -5, 0, 5, 10]))
    v_loc = result["v"][:, :, 0]
    s0 = state[35 * 24:37 * 24, :, 0]
    assert np.median(v_loc[s0 > 0.35]) > 0.05   # well-separated positive state
    assert np.median(v_loc[s0 < -0.35]) < -0.02  # well-separated negative state


def test_c_recent_uses_recency_not_distance():
    origins = np.arange(T0, T0 + 30 * DAY, HOUR)
    state = np.zeros((len(origins), schema.HORIZONS, 6))
    # Recent records are far in state space; old records match the query state.
    state[:, :, 0] = 0.9
    state[-20:, :, 0] = -0.9
    residuals = np.linspace(100, -100, len(origins))[:, None] * np.ones((1, schema.HORIZONS))
    library = make_library(origins, residuals, state6=state)
    queries = np.array([T0 + 30 * DAY])
    z = np.zeros((1, schema.HORIZONS, schema.NZ)); z[:, :, 0] = 1.0
    z[:, :, list(schema.MATCH_Z_INDICES)[0]] = 0.9
    result, _ = run_engine(library, queries, z)
    assert result["v"][0, 0, 0] != result["v_recent"][0, 0, 0]
    # Recent branch tracks the most recent residuals (negative here).
    assert result["v_recent"][0, 0, 0] < result["v"][0, 0, 0]


def test_library_concat_rejects_overlap():
    origins = np.arange(T0, T0 + 5 * DAY, HOUR)
    a = make_library(origins, np.zeros(len(origins)))
    with pytest.raises(ValueError):
        a.concat(a)
    b = make_library(origins + 10 * DAY, np.zeros(len(origins)))
    merged = a.concat(b)
    assert len(merged) == 2 * len(origins)
