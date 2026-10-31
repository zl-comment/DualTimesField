"""Synthetic replay-window and native-score checks; no datasets or inference."""
import json

import numpy as np
import pandas as pd
import pytest

from forecasting.agent_forecasting.historical_replay import AEST, build_records, score_native
from forecasting.agentic_dev_eval.metrics import score_arrays


ORIGIN = pd.Timestamp("2022-03-01", tz=AEST)


def frame_for_origin():
    # Includes an extra earlier hour and the unused bridge hour, to catch tail fills.
    starts = pd.date_range(ORIGIN - pd.Timedelta(hours=74), periods=100, freq="h")
    return pd.DataFrame({
        "region": "NSW1", "delivery_start_aest": starts,
        "available_at_aest": starts + pd.Timedelta(hours=1),
        "rrp_aud_per_mwh": np.arange(100, dtype=float),
        "total_demand_mw": 1000.0 + np.arange(100),
    })


def test_exact_72_25_24_windows_and_fixed_aest_calendar():
    records, labels, missing = build_records(frame_for_origin(), "NSW1", ["2022-03-01"])
    record = records[0]
    origin = int(ORIGIN.timestamp())
    assert missing == []
    assert record["cutoff_unix"] == origin - 600
    assert record["history_time_unix"] == list(range(origin - 73 * 3600, origin - 3600, 3600))
    assert record["history_prices"] == list(range(1, 73))
    assert record["model_target_time_unix"] == list(range(origin - 3600, origin + 86400, 3600))
    assert record["target_time_unix"] == record["model_target_time_unix"][1:]
    assert np.shape(record["history_calendar"]) == (72, 9)
    assert np.shape(record["future_calendar"]) == (25, 9)
    np.testing.assert_allclose(record["future_calendar"][1][:2], [0, 1], atol=1e-14)
    np.testing.assert_array_equal(labels, [np.arange(74, 98)])
    json.dumps(record, allow_nan=False)


def test_missing_history_and_missing_day_never_tail_fill_or_disappear():
    frame = frame_for_origin().drop(index=10)
    records, labels, missing = build_records(frame, "NSW1", ["2022-03-01", "2022-03-20"])
    assert records == [] and labels.shape == (0, 24)
    assert [item["history_missing_hours"] for item in missing] == [1, 72]
    assert [item["reasons"] for item in missing] == [["history_missing"], ["history_missing"]]


def test_history_visibility_cutoff_is_inclusive_and_future_values_do_not_rescue_it():
    frame = frame_for_origin()
    cutoff = ORIGIN - pd.Timedelta(minutes=10)
    frame.loc[72, "available_at_aest"] = cutoff
    records, _, missing = build_records(frame, "NSW1", [ORIGIN])
    assert len(records) == 1 and missing == []
    frame.loc[72, "available_at_aest"] = cutoff + pd.Timedelta(seconds=1)
    records, labels, missing = build_records(frame, "NSW1", [ORIGIN])
    assert records == [] and labels.shape == (0, 24)
    assert missing[0]["reasons"] == ["history_not_available"]
    assert missing[0]["history_unavailable_hours"] == 1


def test_future_label_changes_and_missing_labels_do_not_change_input_record():
    frame = frame_for_origin()
    records, labels, _ = build_records(frame, "NSW1", [ORIGIN])
    frame.loc[74:97, "rrp_aud_per_mwh"] += 10000
    modified_records, modified_labels, _ = build_records(frame, "NSW1", [ORIGIN])
    assert modified_records == records
    np.testing.assert_array_equal(modified_labels - labels, np.full((1, 24), 10000))
    frame = frame.drop(index=75)
    incomplete_records, incomplete_labels, missing = build_records(frame, "NSW1", [ORIGIN])
    assert incomplete_records == records
    assert np.isnan(incomplete_labels[0, 1])
    assert missing[0]["reasons"] == ["label_missing"]
    assert not ({"actual", "labels", "label_valid"} & set(records[0]))


def test_nonfinite_history_and_nonmidnight_origin_are_explicit():
    frame = frame_for_origin()
    frame.loc[12, "total_demand_mw"] = np.nan
    records, _, missing = build_records(frame, "NSW1", [ORIGIN])
    assert records == [] and missing[0]["reasons"] == ["history_nonfinite"]
    with pytest.raises(ValueError, match="midnight"):
        build_records(frame, "NSW1", [ORIGIN + pd.Timedelta(hours=1)])


def test_native_pinball_and_coverage_match_hand_computation():
    result = score_native(np.array([[-2., -1., 0., 1., 2.]]), np.array([0.]))
    assert result["mae"] == 0
    assert result["quantile_score"] == pytest.approx(0.16)
    assert list(result["pinball_by_quantile"].values()) == pytest.approx([0.1, 0.1, 0, 0.1, 0.1])
    assert result["coverage_80"] == result["coverage_90"] == 1
    assert result["width_80"] == 2 and result["width_90"] == 4
    assert result["native_crossings"] == 0
    with pytest.raises(ValueError, match="finite"):
        score_native(np.zeros((1, 5)), np.array([np.nan]))


def test_crossed_native_predictions_are_not_repaired_by_sorting_or_clipping():
    q = np.broadcast_to([4., -2., 3., 1., 0.], (1, 24, 5)).copy()
    original = q.copy()
    y = np.zeros((1, 24))
    native = score_native(q, y)
    legacy = score_arrays(q, y)
    assert native["mae"] == 3 and legacy["mae"] == 1
    assert native["quantile_score"] == pytest.approx(2.24)
    assert native["quantile_score"] != legacy["quantile_score"]
    assert native["width_90"] == -4 and native["native_crossings"] == 72
    np.testing.assert_array_equal(q, original)
