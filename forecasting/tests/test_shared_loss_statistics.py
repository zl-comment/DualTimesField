"""Synthetic regression for common loss semantics and paired calendar blocks."""
from datetime import date, timedelta

import numpy as np
import pytest

from forecasting.aligned_scoring import (
    QUANTILE_LEVELS, make_prediction_artifact, per_origin_scores, score_predictions,
)
from forecasting.common.statistics import block_bootstrap_and_jackknife, holm_adjust


def test_crossed_quantiles_share_the_formal_scoring_definition_without_mutation():
    y = np.array([[0., 3.], [-5., 100.], [40., 20.]])
    q = y[..., None] + np.array([10., -8., 6., -2., 1.])
    before = q.copy()
    artifact = make_prediction_artifact(
        actual=y, quantiles=q, quantile_levels=QUANTILE_LEVELS,
        origin_unix=np.array([100, 200, 300]),
        target_time_unix=np.array([[100, 110], [200, 210], [300, 310]]),
        metadata=dict(protocol_id="synthetic", split="test", region="A", seed=1,
                      source="test", input_hash="x", sample_hash="y", preprocessing_hash="z",
                      output_type="quantile", prediction_space="raw_rrp"))
    formal = score_predictions(artifact)
    origin = per_origin_scores(y, q)
    for key in ("mae", "quantile_score_proxy", "crossing_rate", "coverage_80",
                "coverage_90", "interval_width_80", "interval_width_90"):
        assert origin[key].shape == (3,)
        assert origin[key].mean() == pytest.approx(formal[key], abs=1e-14)
    assert np.sqrt(origin["mse"].mean()) == pytest.approx(formal["rmse_pooled"])
    np.testing.assert_allclose(origin["pinball_by_quantile"].mean(axis=0),
                               list(formal["pinball_by_quantile"].values()))
    np.testing.assert_array_equal(q, before)
    # The unsorted legacy implementation would give a different score.
    error = y[..., None] - q
    old = 2 * np.maximum(QUANTILE_LEVELS * error, (QUANTILE_LEVELS - 1) * error).mean()
    assert old != pytest.approx(formal["quantile_score_proxy"])


@pytest.mark.parametrize("actual,quantiles", [
    (np.zeros((0, 2)), np.zeros((0, 2, 5))),
    (np.zeros((2,)), np.zeros((2, 5))),
    (np.zeros((2, 2)), np.zeros((2, 2, 4))),
    (np.array([[np.nan]]), np.zeros((1, 1, 5))),
    (np.zeros((1, 1)), np.full((1, 1, 5), np.inf)),
    (np.zeros((1, 1)), np.ones((1, 1, 5), dtype=complex)),
])
def test_per_origin_rejects_invalid_shapes_or_nonfinite_values(actual, quantiles):
    with pytest.raises(ValueError):
        per_origin_scores(actual, quantiles)


def _days(n=15):
    return np.array([date(2022, 1, 1) + timedelta(days=i) for i in range(n)])


def test_bootstrap_preserves_repeated_blocks_short_tail_and_region_weights():
    days = _days()
    a = np.array([-2.] * 7 + [3.] * 7 + [100.])
    # Region B has twice the origins, but must retain equal macro weight.
    b = np.repeat(a * .5 + 2, 2)
    seed, reps = 81, 120
    result = block_bootstrap_and_jackknife(
        {"A": a, "B": b}, {"A": days, "B": np.repeat(days, 2)},
        repetitions=reps, seed=seed)
    # Independent, explicit origin-index expansion retains duplicate draws.
    blocks = [np.arange(0, 7), np.arange(7, 14), np.arange(14, 15)]
    rng, expected, buggy = np.random.default_rng(seed), [], []
    for _ in range(reps):
        take = np.concatenate([blocks[i] for i in rng.integers(0, 3, 3)])
        expected.append((a[take].mean() + (a[take] * .5 + 2).mean()) / 2)
        unique = np.unique(take)
        buggy.append((a[unique].mean() + (a[unique] * .5 + 2).mean()) / 2)
    assert not np.allclose(expected, buggy)
    np.testing.assert_allclose([result["ci_low"], result["ci_high"]],
                               np.quantile(expected, [.025, .975]))
    assert result["bootstrap_tail_probability"] == np.mean(np.array(expected) <= 0)
    assert result["delta"] == pytest.approx((a.mean() + b.mean()) / 2)
    full = (a.mean() + b.mean()) / 2
    changes = []
    for d in days:
        loo = (a[days != d].mean() + b[np.repeat(days, 2) != d].mean()) / 2
        changes.append(abs(full - loo) / abs(full))
    assert result["jackknife_max_day_share"] == pytest.approx(max(changes))
    assert result["last_block_days"] == 1


def test_same_block_draws_are_synchronized_across_regions_and_rng_is_local():
    days = _days(21)
    a = np.arange(21, dtype=float)
    values = {"A": a, "B": 2 - a}
    dates = {"A": days, "B": days}
    first = block_bootstrap_and_jackknife(values, dates, repetitions=80, seed=19)
    np.random.seed(423)
    np.random.normal(size=1000)
    second = block_bootstrap_and_jackknife(values, dates, repetitions=80, seed=19)
    assert first == second
    assert [first["delta"], first["ci_low"], first["ci_high"]] == pytest.approx([1., 1., 1.])
    assert first["bootstrap_p_leq0"] == 0
    reverse = block_bootstrap_and_jackknife(values, dates, repetitions=80, seed=19,
                                           alternative="less")
    assert reverse["bootstrap_p_geq0"] == 1


def test_missing_calendar_days_are_not_compressed_and_sparse_region_fails_explicitly():
    days = _days(15)[[0, 6, 7, 13, 14]]
    result = block_bootstrap_and_jackknife({"A": np.ones(5)}, {"A": days}, repetitions=20)
    assert (result["n_days"], result["n_blocks"], result["last_block_days"]) == (15, 3, 1)
    with pytest.raises(ValueError, match="every calendar block"):
        block_bootstrap_and_jackknife({"A": np.ones(2)}, {"A": _days(15)[[0, 14]]})
    with pytest.raises(ValueError, match="two calendar blocks"):
        block_bootstrap_and_jackknife({"A": np.ones(7)}, {"A": _days(7)})


def test_invalid_bootstrap_inputs_and_zero_effect_are_explicit():
    with pytest.raises(ValueError, match="numeric timestamps"):
        block_bootstrap_and_jackknife({"A": np.ones(15)}, {"A": np.arange(15)})
    with pytest.raises(ValueError, match="finite"):
        block_bootstrap_and_jackknife({"A": [np.nan]}, {"A": _days(1)})
    with pytest.raises(ValueError, match="NaT"):
        block_bootstrap_and_jackknife({"A": [1.]}, {"A": ["NaT"]})
    zero = block_bootstrap_and_jackknife({"A": np.zeros(14)}, {"A": _days(14)}, repetitions=5)
    assert zero["jackknife_max_day_share"] == 0
    mixed = block_bootstrap_and_jackknife({"A": np.array([-1.] * 7 + [1.] * 7)},
                                          {"A": _days(14)}, repetitions=5)
    assert mixed["delta"] == 0 and mixed["jackknife_max_day_share"] is None


def test_holm_uses_cumulative_maximum_and_preserves_key_order():
    p = {"c": .04, "a": .03, "b": .031}
    adjusted = holm_adjust(p)
    assert list(adjusted) == list(p)
    assert adjusted == pytest.approx({"a": .09, "b": .09, "c": .09})
    assert holm_adjust({"a": .8, "b": .9}) == {"a": 1., "b": 1.}
    with pytest.raises(ValueError):
        holm_adjust({"a": np.nan})
