"""Synthetic version selection and A/B input mapping; no real data or models."""
from copy import deepcopy
from datetime import datetime
import json

import numpy as np
import pandas as pd
import pytest

from forecasting.agent_forecasting.totaldemand_information import AEST, coverage, vintage_label_unix
from forecasting.agent_forecasting.totaldemand_proxy import model_input, selected_proxy
from forecasting.agentic_dev_eval.data import CALENDAR_FEATURES


ORIGIN = int(datetime(2022, 1, 2, tzinfo=AEST).timestamp())


def source_rows(seqno=2022010139, runno=1):
    return pd.DataFrame({
        "seqno": seqno, "runno": runno, "region": "NSW1", "intervention": 0,
        "periodid": np.arange(1, 51), "totaldemand": 1000.0 + np.arange(50),
        "lastchanged_unix": ORIGIN - 1200,
        "period_end_unix": ORIGIN - 1800 + np.arange(50) * 1800,
        "schedule_label_unix": vintage_label_unix(seqno),
    })


def audited(rows):
    return coverage(rows, [ORIGIN], ["NSW1"])[0]


def model_record():
    return {
        "history_prices": np.arange(72, dtype=float).tolist(),
        "history_demand": (1000 + np.arange(72, dtype=float)).tolist(),
        "history_calendar": np.zeros((72, 9)).tolist(),
        "future_calendar": np.ones((25, 9)).tolist(),
        "actual": [999999.0] * 24,  # Deliberately supplied; must never be read.
    }


def test_exact_half_hour_pair_means_include_bridge_and_final_hour():
    rows = source_rows().sample(frac=1, random_state=3)
    result = selected_proxy(rows, audited(rows))
    assert result["half_hour_end_unix"][0] == ORIGIN - 1800
    assert result["half_hour_end_unix"][-1] == ORIGIN + 86400
    assert result["hour_start_unix"][0] == ORIGIN - 3600
    assert result["hour_start_unix"][-1] == ORIGIN + 23 * 3600
    assert result["hourly_values"] == (1000.5 + np.arange(25) * 2).tolist()
    assert result["seqno"] == 2022010139 and result["runno"] == 1
    assert result["lastchanged_max"] == ORIGIN - 1200
    json.dumps(result, allow_nan=False)


def test_late_conflicts_are_excluded_before_visible_duplicate_check():
    rows = source_rows()
    record = audited(rows)
    late = rows.iloc[[0]].copy()
    late["lastchanged_unix"] = ORIGIN - 599
    late["totaldemand"] = np.nan
    late["periodid"] = 999
    combined = pd.concat([rows, rows.iloc[[0]], late])
    result = selected_proxy(combined, record)
    assert result["duplicate_rows"] == 1
    assert result["hourly_values"][0] == 1000.5
    late["lastchanged_unix"] = ORIGIN - 600
    with pytest.raises(ValueError, match="invalid values"):
        selected_proxy(pd.concat([rows, late]), record)


@pytest.mark.parametrize("field,value", [("totaldemand", 5000), ("periodid", 999)])
def test_visible_conflict_rejects_demand_and_period_identity(field, value):
    rows = source_rows()
    duplicate = rows.iloc[[2]].copy()
    duplicate[field] = value
    with pytest.raises(ValueError, match="conflict"):
        selected_proxy(pd.concat([rows, duplicate]), audited(rows))


def test_missing_selected_slot_cannot_be_filled_from_another_vintage_or_region():
    rows = source_rows()
    record = audited(rows)
    extras = pd.concat([source_rows(runno=2), source_rows(seqno=2022010138)])
    other_region = rows.copy()
    other_region["region"] = "QLD1"
    broken = pd.concat([rows.iloc[1:], extras, other_region])
    with pytest.raises(ValueError, match="50 half-hour"):
        selected_proxy(broken, record)
    assert selected_proxy(pd.DataFrame(), {"covered": False}) is None


def test_selected_audit_timestamp_mismatch_is_not_silently_reselected():
    rows = source_rows()
    record = audited(rows)
    record["lastchanged_max"] -= 1
    with pytest.raises(ValueError, match="LASTCHANGED"):
        selected_proxy(rows, record)


def test_a_b_only_differ_in_future_demand_without_nan_channel_or_labels():
    record = model_record()
    original = deepcopy(record)
    values = np.arange(25, dtype=float) + 1200
    a, b = model_input(record), model_input(record, values)
    np.testing.assert_array_equal(a["target"], b["target"])
    assert set(a) == set(b) == {"target", "past_covariates", "future_covariates"}
    assert set(a["past_covariates"]) == set(b["past_covariates"]) == {*CALENDAR_FEATURES, "total_demand_mw"}
    assert set(b["future_covariates"]) - set(a["future_covariates"]) == {"total_demand_mw"}
    for name in a["past_covariates"]:
        np.testing.assert_array_equal(a["past_covariates"][name], b["past_covariates"][name])
        assert np.isfinite(b["past_covariates"][name]).all()
    for name in a["future_covariates"]:
        np.testing.assert_array_equal(a["future_covariates"][name], b["future_covariates"][name])
    np.testing.assert_array_equal(b["future_covariates"]["total_demand_mw"], values)
    record["actual"] = None
    np.testing.assert_array_equal(model_input(record)["target"], a["target"])
    a["target"][0] = -1000
    assert record["history_prices"] == original["history_prices"]
    with pytest.raises(ValueError, match="25 finite"):
        model_input(record, values[:-1])
