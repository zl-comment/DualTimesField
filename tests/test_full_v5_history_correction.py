"""Causal and optimization contracts for the frozen-base history repair."""
import numpy as np
import pytest

from forecasting.full_v5 import history_correction as hc, schema


def arrays(days=30, start=None):
    start = hc.HISTORY_START if start is None else start
    origins = start + np.arange(days * 24) * hc.HOUR
    q = np.broadcast_to(np.array([-20., -10., 0., 10., 20.]),
                        (len(origins), 24, 5)).copy()
    y = np.full((len(origins), 24), 12.)
    z = np.zeros((len(origins), 24, schema.NZ))
    return origins, y, q, z


@pytest.mark.parametrize("policy", ["state", "fresh28", "recent_consistent"])
def test_future_and_immature_labels_cannot_affect_past_predictions(policy):
    o, y, q, z = arrays()
    cut = 20 * 24
    before = hc.compute_history_corrections(o, y, q, z, policy=policy)
    changed = y.copy()
    # Includes labels of past origins whose full window has not yet matured.
    changed[o + 24 * hc.HOUR > o[cut]] = 1e8
    after = hc.compute_history_corrections(o, changed, q, z, policy=policy)
    for key in ("matched", "recent"):
        np.testing.assert_array_equal(before[key][:cut + 1], after[key][:cut + 1])


def test_pre_fit_library_exclusion_same_hour_and_maturity_boundary():
    o, y, q, z = arrays(40, hc.HISTORY_START - 10 * hc.DAY)
    y[o < hc.HISTORY_START] = 1e8
    # At midnight October 9 there are exactly eight fully mature same-hour
    # OOS origins. Other hours and horizons must not contaminate their median.
    hour = (o // hc.HOUR + 10) % 24
    y[hour != 0] = 1e8
    y[:, 1] = -12.
    out = hc.compute_history_corrections(o, y, q, z)
    i = int(np.flatnonzero(o == hc.HISTORY_START + 8 * hc.DAY)[0])
    assert out["matched"][i, 0] == 6.  # 8/16 shrinkage times residual 12
    assert out["matched"][i, 1] == -6.
    assert out["recent"][i, 0] == 6.
    assert out["matched"][i - 24, 0] == 0.  # only seven candidates
    for sample in out["audit"]["selection_samples"]:
        chosen = np.asarray(sample["selected_origins"])
        assert np.all(chosen >= hc.HISTORY_START)
        assert np.all(chosen + hc.DAY <= sample["query_origin"])
        assert np.all(chosen % hc.DAY == sample["query_origin"] % hc.DAY)
    with pytest.raises(ValueError, match="predate"):
        hc.compute_history_corrections(o, y, q, z, history_start=hc.HISTORY_START - 1)


def test_matching_uses_state_and_preserves_horizon_residuals():
    o, y, q, z = arrays(40)
    # All tied states choose earliest legal origins (stable origin tie-break),
    # whereas recent control chooses newest. Their outputs must differ.
    y[:] = (np.arange(len(o)) // 24)[:, None]
    out = hc.compute_history_corrections(o, y, q, z)
    i = 35 * 24
    assert out["matched"][i, 0] == np.median(np.arange(16))
    assert out["recent"][i, 0] == np.median(np.arange(19, 35))
    # Put current query near the recent records: matching now chooses recent.
    z[:19 * 24, :, schema.MATCH_Z_INDICES[0]] = 1.
    matched = hc.compute_history_corrections(o, y, q, z)
    assert matched["matched"][i, 0] == out["recent"][i, 0]


def test_additive_correction_is_bounded_ordered_and_null_exact():
    _, _, q, _ = arrays(1)
    d = np.linspace(-150, 150, len(q) * 24).reshape(-1, 24)
    np.testing.assert_array_equal(hc.apply_history_correction(q, d, [0, 0, 0, 0]), q)
    result = hc.apply_history_correction(q, d, [1, .5, .25, 0])
    assert np.all(np.diff(result, axis=-1) >= 0)
    np.testing.assert_allclose(np.diff(result, axis=-1), np.diff(q, axis=-1))
    assert np.max(np.abs(result - q)) <= 150
    with pytest.raises(ValueError):
        hc.apply_history_correction(q, d, [1.1, 0, 0, 0])


def test_weight_fit_enforces_strict_cutoff_and_ignores_later_labels():
    o, y, q, _ = arrays(92)
    d = np.full_like(y, 20.)
    before = hc.fit_history_weights(o, y, q, d)
    changed = y.copy()
    changed[o + hc.DAY >= hc.FIT_CUTOFF] = -1e9
    after = hc.fit_history_weights(o, changed, q, d)
    assert before == after
    assert before["diagnostics"]["latest_fit_label_available"] < hc.FIT_CUTOFF
    assert any(a > 0 for a in before["alpha"])
    with pytest.raises(ValueError, match="October-November"):
        hc.fit_history_weights(o, y, q, d, fit_cutoff=hc.FIT_CUTOFF + 1)


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_breakpoint_optimizer_matches_exhaustive_candidates(seed):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=37) * 10
    q = np.sort(rng.normal(size=(37, 5)) * 10, axis=-1)
    d = rng.normal(size=37) * 15
    d[:5] = 0
    a, before, after = hc._fit_scalar(y, q, d)
    active = d != 0
    candidates = np.unique(np.clip(np.r_[0., 1.,
        ((y[active, None] - q[active]) / d[active, None]).ravel()], 0, 1))
    losses = [hc._loss(y, q + b * d[:, None]) for b in candidates]
    assert 0 <= a <= 1
    assert after <= before
    assert after == pytest.approx(min(losses), rel=1e-12, abs=1e-10)


def test_zero_or_harmful_signal_prefers_null():
    y = np.zeros(10)
    q = np.zeros((10, 5))
    assert hc._fit_scalar(y, q, np.zeros(10))[0] == 0
    assert hc._fit_scalar(y, q, np.ones(10))[0] == 0


def test_clipping_and_input_integrity():
    o, y, q, z = arrays(20)
    y[:] = 1e8
    saved = [a.copy() for a in (o, y, q, z)]
    out = hc.compute_history_corrections(o, y, q, z)
    assert np.max(out["matched"]) == 150
    for a, b in zip((o, y, q, z), saved):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="increasing"):
        hc.compute_history_corrections(o[::-1], y, q, z)


@pytest.mark.parametrize("policy", ["fresh28", "recent_consistent"])
def test_freshness_boundary_excludes_day_28_for_both_controls(policy):
    o, y, q, z = arrays(41)
    # At day 40, day 12 is exactly 28 days old and must be excluded. Only
    # seven remaining legal candidates -> both matched/recent must be null.
    indices = np.r_[np.arange(12, 20), 40] * 24
    o, y, q, z = (a[indices] for a in (o, y, q, z))
    legacy = hc.compute_history_corrections(o, y, q, z, policy="state")
    fresh = hc.compute_history_corrections(o, y, q, z, policy=policy)
    for key in ("matched", "recent"):
        assert np.all(legacy[key][-1] == 6.)  # eight records, shrinkage 8/16
        assert np.all(fresh[key][-1] == 0.)
    assert fresh["audit"]["lookback_days"] == 28


def test_fresh_matching_and_recent_share_same_eligible_pool():
    o, y, q, z = arrays(41)
    o, y, q, z = (a[::24] for a in (o, y, q, z))
    y[:] = np.arange(41)[:, None]
    out = hc.compute_history_corrections(o, y, q, z, policy="fresh28")
    # Day 40's pool is day 13..39; equal states select earliest 16, while
    # recent control selects latest 16 from exactly the same eligible pool.
    assert np.all(out["matched"][-1] == np.median(np.arange(13, 29)))
    assert np.all(out["recent"][-1] == np.median(np.arange(24, 40)))
    changed = y.copy()
    changed[:13] = -1e9
    after = hc.compute_history_corrections(o, changed, q, z, policy="fresh28")
    for key in ("matched", "recent"):
        np.testing.assert_array_equal(out[key][-1], after[key][-1])


@pytest.mark.parametrize("proposal,reference,expected", [
    (100., -20., 0.), (100., 20., 20.), (-100., -20., -20.),
    (10., 20., 10.), (100., 0., 0.), (0., 20., 0.),
])
def test_agreement_guard_nulls_conflict_and_caps_same_sign(proposal, reference, expected):
    o, y, q, z = arrays(41)
    o, y, q, z = (a[::24] for a in (o, y, q, z))
    y[:24] = proposal
    y[24:] = reference
    raw = hc.compute_history_corrections(o, y, q, z, policy="fresh28")
    guarded = hc.compute_history_corrections(o, y, q, z, policy="recent_consistent")
    assert np.all(raw["matched"][-1] == proposal)
    assert np.all(guarded["matched"][-1] == expected)
    # The trust policy modifies only matched correction, never its control.
    np.testing.assert_array_equal(raw["recent"], guarded["recent"])
    assert np.all(np.abs(guarded["matched"]) <= np.abs(guarded["recent"]))
    assert np.all(guarded["matched"] * guarded["recent"] >= 0)


def test_default_policy_reproduces_state_policy_and_rejects_unknown():
    o, y, q, z = arrays(10)
    default = hc.compute_history_corrections(o, y, q, z)
    explicit = hc.compute_history_corrections(o, y, q, z, policy="state")
    for key in ("matched", "recent"):
        np.testing.assert_array_equal(default[key], explicit[key])
    assert default["audit"] == explicit["audit"]
    with pytest.raises(ValueError, match="Unknown"):
        hc.compute_history_corrections(o, y, q, z, policy="unregistered")
