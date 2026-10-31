"""Synthetic S1 macro, missingness, block-inference and decision examples."""
import json

import numpy as np
import pytest

from forecasting.agent_forecasting.historical_comparison import compare_s1


def inputs(days=20):
    a = np.full((days, 3, 24, 5), 4.0)
    b = np.full_like(a, 3.0)
    y = np.zeros((days, 3, 24))
    source = np.ones((days, 3), dtype=bool)
    dates = np.datetime64("2022-01-01") + np.arange(days)
    return a, b, y, source, dates


def test_macro_is_equal_region_weight_not_pooled_cells_and_missing_dates_stay():
    a, b, y, source, dates = inputs()
    a[:] = np.array([1.0, 10.0, 100.0])[None, :, None, None]
    b[:] = a
    y[5, 0, 0] = np.nan  # Entire region-day is excluded, not only this hour.
    report = compare_s1(a, b, y, source, dates)
    assert report["macro"]["A"]["quantile_score"] == pytest.approx(37.0)
    pooled = (19 * 1 + 20 * 10 + 20 * 100) / 59
    assert report["macro"]["A"]["quantile_score"] != pytest.approx(pooled)
    assert report["regional"]["NSW1"]["common_days"] == 19
    assert report["regional"]["NSW1"]["common_scoring_rate"] == 0.95
    assert len(report["common_mask"]) == report["bootstrap"]["date_axis_length"] == 20
    assert report["common_mask"][5] == [False, True, True]
    assert report["decision"] == "NO_USEFUL_GAIN"


def test_fallback_counts_in_main_score_while_source_coverage_uses_planned_denominator():
    a, b, y, source, dates = inputs()
    b[0] = a[0]
    source[0] = False
    report = compare_s1(a, b, y, source, dates)
    assert report["macro"]["B"]["quantile_score"] == pytest.approx(3.05)
    assert report["regional"]["NSW1"]["source_coverage"] == 0.95
    assert report["regional"]["NSW1"]["common_scoring_rate"] == 1
    assert report["regional"]["NSW1"]["source_missing_scored_days"] == 1
    assert report["decision"] == "PASS_INFORMATION"
    source[1] = False
    blocked = compare_s1(a, b, y, source, dates)
    assert blocked["decision"] == "DATA_OR_EXECUTION_BLOCKED"
    assert blocked["macro"] == report["macro"]  # Source failures do not filter scores.
    json.dumps(blocked, allow_nan=False)


def test_synchronous_blocks_preserve_constant_relative_change_with_distinct_daily_patterns():
    a, b, y, source, dates = inputs(28)
    daily = np.arange(1, 29, dtype=float)
    a[:] = np.column_stack([daily, daily[::-1] * 3, (daily % 5 + 1) * 10])[:, :, None, None]
    b[:] = 0.9 * a
    report = compare_s1(a, b, y, source, dates)
    assert report["bootstrap"]["valid_replicates"] == 2000
    assert report["bootstrap"]["seed"] == 20261031
    assert report["bootstrap"]["relative_qs_ci95"] == pytest.approx([-0.1, -0.1])
    assert report["decision"] == "PASS_INFORMATION"


def test_one_missing_b_cannot_be_dropped_to_pass_even_above_95_percent_availability():
    a, b, y, source, dates = inputs(28)
    b[0, 0] = np.nan
    report = compare_s1(a, b, y, source, dates)
    assert report["regional"]["NSW1"]["b_prediction_availability"] > 0.95
    assert report["regional"]["NSW1"]["primary_days"] == 28
    assert report["regional"]["NSW1"]["b_unresolved_on_primary_days"] == 1
    assert report["scores_are_partial_diagnostics"]
    assert report["decision"] == "DATA_OR_EXECUTION_BLOCKED"
    assert "NSW1/b_covers_primary" in report["failed_criteria"]
    b[0, 0] = a[0, 0]  # An explicitly published fallback preserves that unit.
    complete = compare_s1(a, b, y, source, dates)
    assert not complete["scores_are_partial_diagnostics"]
    assert complete["decision"] == "PASS_INFORMATION"


def test_short_series_is_inconclusive_and_a_compressed_calendar_is_rejected():
    report = compare_s1(*inputs(6))
    assert report["decision"] == "INCONCLUSIVE"
    assert report["bootstrap"]["relative_qs_ci95"] is None
    a, b, y, source, dates = inputs(20)
    keep = np.arange(20) != 5
    with pytest.raises(ValueError, match="consecutive"):
        compare_s1(a[keep], b[keep], y[keep], source[keep], dates[keep])


def test_point_gain_without_negative_ci_upper_bound_is_inconclusive():
    a, b, y, source, dates = inputs()
    a[:] = 10
    b[:7] = 8
    b[7:14] = 11.5
    b[14:] = 10
    report = compare_s1(a, b, y, source, dates)
    assert report["macro"]["relative_qs_change"] == pytest.approx(-0.0175)
    assert report["bootstrap"]["relative_qs_ci95"][1] > 0
    assert report["decision"] == "INCONCLUSIVE"


@pytest.mark.parametrize("guard", ["region", "mae", "coverage"])
def test_macro_qs_gain_cannot_override_a_failed_protection(guard):
    a, b, y, source, dates = inputs()
    if guard == "region":
        a[:] = 10
        b[:] = 8
        b[:, 0] = 10.3
        failure = "NSW1/qs_protection"
    elif guard == "mae":
        a[:] = [-20, -10, 1, 10, 20]
        b[:] = [-3, -2, 1.02, 2, 3]
        failure = "macro/mae_protection"
    else:
        a[:] = [-10, -5, 4, 5, 10]
        b[:] = 1
        failure = "NSW1/coverage_80_protection"
    report = compare_s1(a, b, y, source, dates)
    assert report["macro"]["relative_qs_change"] < -0.01
    assert report["decision"] == "NO_USEFUL_GAIN"
    assert failure in report["failed_criteria"]


def test_zero_reference_and_empty_region_cannot_manufacture_a_pass():
    a, b, y, source, dates = inputs()
    a[:] = 0
    b[:] = 0
    zero = compare_s1(a, b, y, source, dates)
    assert zero["macro"]["relative_qs_change"] is None
    assert zero["decision"] == "NO_USEFUL_GAIN"
    a[:, 1] = np.nan
    empty = compare_s1(a, b, y, source, dates)
    assert empty["macro"]["A"] is None
    assert empty["decision"] == "DATA_OR_EXECUTION_BLOCKED"
    json.dumps(empty, allow_nan=False)
