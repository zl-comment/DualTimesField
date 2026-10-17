"""Causal release and controlled-comparison tests for dual-field memory."""
import numpy as np
import pytest

from forecasting.full_v5 import dual_history_memory as memory, history_correction as hc, schema


class FakeEncoder:
    def __init__(self, zero_event=False, permute=False):
        self.calls = []
        self.zero_event = zero_event
        self.permute = permute

    def encode_matured(self, origins, actual, z, lag_prices, *, as_of):
        assert np.all(origins >= hc.HISTORY_START)
        assert np.all(origins + hc.DAY <= as_of)
        self.calls.append(dict(origins=origins.copy(), actual=actual.copy(), as_of=as_of))
        trend = actual * .7
        event = np.zeros_like(actual) if self.zero_event else actual * .3
        ids = np.arange(len(origins))
        if self.permute:
            ids = ids[::-1]
        return dict(indices=ids, trend=trend[ids], event=event[ids],
                    reconstruction=(trend + event)[ids])


def arrays(days=40, pre_days=0):
    origins = hc.HISTORY_START + np.arange(-pre_days, days) * hc.DAY
    n = len(origins)
    actual = np.arange(n)[:, None] + np.arange(24)[None, :] * .1 + 12.
    q = np.broadcast_to(np.array([-20., -10., 0., 10., 20.]), (n, 24, 5)).copy()
    z = np.zeros((n, 24, schema.NZ))
    lag = np.ones((n, 24, 3))
    return origins, actual, q, z, lag


def test_future_and_immature_targets_never_reach_encoder_or_earlier_corrections():
    o, y, q, z, lag = arrays()
    cutoff = o[20]
    first = FakeEncoder()
    before = memory.build_corrections(o, y, q, z, lag, first)
    changed = y.copy()
    changed[o + hc.DAY > cutoff] = 1e8
    second = FakeEncoder()
    after = memory.build_corrections(o, changed, q, z, lag, second)
    for name in memory.VARIANTS:
        np.testing.assert_array_equal(before["corrections"][name][:21],
                                      after["corrections"][name][:21])
    for a, b in zip(first.calls, second.calls):
        if a["as_of"] <= cutoff:
            np.testing.assert_array_equal(a["actual"], b["actual"])
    assert before["audit"]["maturity_violations"] == 0
    assert before["audit"]["minimum_release_margin_seconds"] == 0
    audit = before["encoded_curves"]
    assert np.all(audit["available_at"] <= audit["encoded_at"])
    assert audit["origins"][-1] == o[-2]  # last query target is never encoded


def test_exact_24h_release_boundary_and_pre_october_exclusion():
    o, y, q, z, lag = arrays(12, pre_days=10)
    y[o < hc.HISTORY_START] = 1e8
    encoder = FakeEncoder()
    out = memory.build_corrections(o, y, q, z, lag, encoder)
    row = int(np.flatnonzero(o == hc.HISTORY_START + 8 * hc.DAY)[0])
    assert np.all(out["corrections"]["rawmatched"][row - 1] == 0)
    expected = np.median(y[(o >= hc.HISTORY_START) & (o < o[row])], axis=0) * .5
    np.testing.assert_allclose(out["corrections"]["rawmatched"][row], expected)
    assert all(np.all(c["origins"] >= hc.HISTORY_START) for c in encoder.calls)
    # A query one second before maturity must not release even the first row.
    origins = np.array([hc.HISTORY_START, hc.HISTORY_START + hc.DAY - 1,
                        hc.HISTORY_START + hc.DAY])
    _, yy, qq, zz, ll = arrays(3)
    spy = FakeEncoder()
    edge = memory.build_corrections(origins, yy, qq, zz, ll, spy)
    assert len(spy.calls) == 1
    assert spy.calls[0]["as_of"] == origins[-1]
    np.testing.assert_array_equal(edge["encoded_curves"]["indices"], [0])
    with pytest.raises(ValueError, match="predate"):
        memory.build_corrections(o, y, q, z, lag, encoder, history_start=hc.HISTORY_START - 1)


def test_all_matched_representations_share_candidates_and_raw_matches_existing():
    o, y, q, z, lag = arrays()
    out = memory.build_corrections(o, y, q, z, lag, FakeEncoder())
    reference = hc.compute_history_corrections(o, y, q, z)
    np.testing.assert_array_equal(out["corrections"]["rawmatched"], reference["matched"])
    np.testing.assert_array_equal(out["corrections"]["recentraw"], reference["recent"])
    assert len(set(out["audit"]["matched_variant_candidate_sha256"].values())) == 1
    assert set(out["audit"]["matched_variant_candidate_sha256"]) == {
        "rawmatched", "fixedsmooth", "dualtrend", "dualfull"}
    np.testing.assert_array_equal(out["matched_indices"][-1, 0], np.arange(16))
    np.testing.assert_array_equal(out["recent_indices"][-1], np.arange(23, 39))
    for i, origin in enumerate(o):
        ids = out["matched_indices"][i]
        ids = ids[ids >= 0]
        assert np.all(o[ids] + hc.DAY <= origin)
        assert np.all(o[ids] % hc.DAY == origin % hc.DAY)


def test_matching_changes_only_with_state_and_preserves_lead():
    o, y, q, z, lag = arrays()
    y[:, 1] = -y[:, 0]
    z[:23, :, schema.MATCH_Z_INDICES[0]] = 1.
    out = memory.build_corrections(o, y, q, z, lag, FakeEncoder())
    np.testing.assert_array_equal(out["matched_indices"][-1, 0], np.arange(23, 39))
    np.testing.assert_array_equal(out["corrections"]["rawmatched"][-1],
                                  out["corrections"]["recentraw"][-1])
    assert out["corrections"]["rawmatched"][-1, 1] == -out["corrections"]["rawmatched"][-1, 0]


def test_event_removal_and_whole_branch_removal_are_exact_interventions():
    args = arrays()
    out = memory.build_corrections(*args, FakeEncoder())
    zero = memory.build_corrections(*args, FakeEncoder(zero_event=True))
    np.testing.assert_array_equal(zero["corrections"]["dualtrend"], zero["corrections"]["dualfull"])
    assert np.any(out["corrections"]["dualfull"] != out["corrections"]["dualtrend"])
    np.testing.assert_array_equal(out["corrections"]["dualtrend"], zero["corrections"]["dualtrend"])
    q = args[2]
    np.testing.assert_array_equal(hc.apply_history_correction(
        q, out["corrections"]["dualfull"], [0., 0., 0., 0.]), q)


def test_fixed_smoothing_retains_only_dc_and_first_harmonic():
    h = np.arange(24)
    wanted = 12 + 3 * np.cos(2 * np.pi * h / 24) + 2 * np.sin(2 * np.pi * h / 24)
    curve = wanted + 11 * np.cos(4 * np.pi * h / 24)
    np.testing.assert_allclose(memory.fixed_smooth(curve[None])[0], wanted, atol=1e-12)


def test_encoder_local_index_mapping_is_checked_and_permutations_remapped():
    # Sparse queries allow multiple newly matured records in one release.
    o, y, q, z, lag = arrays()
    o = hc.HISTORY_START + np.r_[np.arange(35), [100, 124, 148, 172, 196]] * hc.HOUR
    args = (o, y, q, z, lag)
    normal = memory.build_corrections(*args, FakeEncoder())
    encoder = FakeEncoder(permute=True)
    permuted = memory.build_corrections(*args, encoder)
    assert max(len(call["origins"]) for call in encoder.calls) > 1
    for name in memory.VARIANTS:
        np.testing.assert_array_equal(normal["corrections"][name], permuted["corrections"][name])
    for name in ("indices", "trend", "event", "reconstruction"):
        np.testing.assert_array_equal(normal["encoded_curves"][name], permuted["encoded_curves"][name])

    class BadEncoder(FakeEncoder):
        def encode_matured(self, *args, **kwargs):
            result = super().encode_matured(*args, **kwargs)
            result["indices"] += 1
            return result

    with pytest.raises(ValueError, match="indices"):
        memory.build_corrections(*args, BadEncoder())


def test_bounds_and_no_input_mutation():
    args = arrays()
    args[1][:] = 1e8
    originals = [a.copy() for a in args]
    out = memory.build_corrections(*args, FakeEncoder())
    for name in memory.VARIANTS:
        assert np.max(np.abs(out["corrections"][name])) == 150.
    for before, after in zip(originals, args):
        np.testing.assert_array_equal(before, after)
    with pytest.raises(ValueError, match="lag_prices"):
        memory.build_corrections(*args[:4], args[4][..., 0], FakeEncoder())
