from types import SimpleNamespace

import numpy as np
import pytest

from forecasting.analysis.lightgbm_event_diagnosis import (
    detection, event_rows, events, oracle_diagnosis, unique_labels,
)


def fixture():
    series = np.zeros(40)
    series[14:16] = [500, 600]
    index = np.arange(17)[:, None] + np.arange(24)
    split = SimpleNamespace(actual=series[index], origin_unix=np.arange(17)*3600,
                            target_time_unix=index*3600)
    quantiles = np.stack([split.actual-20, split.actual-10, split.actual,
                         split.actual+10, split.actual+20], -1)
    return split, quantiles


def test_unique_deliveries_are_counted_once_and_conflicting_labels_rejected():
    split, _ = fixture()
    assert len(unique_labels(split)[0]) == 40
    split.actual[1, 13] += 1
    with pytest.raises(ValueError, match='disagree'):
        unique_labels(split)


def test_gaps_split_events_and_censor_both_sides():
    result = events(np.array([0, 3600, 10800, 14400, 18000]),
                    np.array([0, 500, 500, 500, 0]), 300)
    assert [row['duration'] for row in result] == [1, 2]
    assert not any(row['complete'] for row in result)


def test_one_origin_per_episode_fixed_six_hour_lead():
    split, quantiles = fixture()
    rows = event_rows(split, quantiles, 300)
    assert len(rows) == 1
    row = rows[0]
    assert row['eligible'] and row['origin_unix'] == 8*3600
    assert row['q50_on_time_detected'] and row['q50_duration_error_hours'] == 0
    assert row['q50_local_peak_lag_hours'] == 0


def test_long_censored_event_is_reported_but_excluded_from_shape_matching():
    split, quantiles = fixture()
    split.actual[:] = 500
    rows = event_rows(split, quantiles, 300)
    assert len(rows) == 1 and not rows[0]['eligible']
    assert 'q50_duration_error_hours' not in rows[0]


def test_oracle_shift_and_level_controls_use_same_target_window():
    actual = np.zeros((2, 24))
    actual[:, 8:10] = [500, 600]
    exact = oracle_diagnosis(actual, actual)
    assert exact['all_oracle_lag_gain'] == exact['all_oracle_level_gain'] == 0
    shifted = oracle_diagnosis(actual, np.roll(actual, 3, axis=1))
    assert shifted['all_oracle_lag_gain'] == shifted['all_base_mae']
    offset = oracle_diagnosis(actual, actual+5)
    assert offset['all_oracle_level_gain'] == 5


def test_alarm_precision_does_not_confuse_risk_flag_with_point_prediction():
    actual = np.array([0, 500, 0, 500])
    point = detection(actual, np.zeros(4), 300)
    upper = detection(actual, np.full(4, 600), 300)
    assert point['recall'] == 0 and point['precision'] is None
    assert upper['recall'] == 1 and upper['precision'] == .5
