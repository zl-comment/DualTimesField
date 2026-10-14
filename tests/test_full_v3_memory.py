"""As-of analog eligibility does not use query outcomes or synthetic releases."""
import copy

import numpy as np
import pandas as pd
import pytest

from forecasting.aligned_inputs import AlignedSplit
from forecasting.full_v3.memory import HistoricalMemory, load_availability


TZ = "Australia/Brisbane"


def unix(value):
    return int(pd.Timestamp(value, tz=TZ).timestamp())


def split(origins, *, region="NSW1"):
    origins = np.asarray(origins, dtype=np.int64)
    size = len(origins)
    history = np.broadcast_to(np.arange(144, dtype=np.float32).reshape(1, 72, 2), (size, 72, 2)).copy()
    inputs = dict(history_values=history, history_calendar=np.zeros((size, 72, 10), np.float32),
        future_calendar=np.zeros((size, 24, 10), np.float32), origin_context=np.zeros((size, 1), np.float32),
        future_exogenous=np.zeros((size, 24, 11), np.float32),
        ctf_exogenous=np.zeros((size, 24, 1), np.float32),
        quantile_exogenous=np.zeros((size, 24, 3), np.float32))
    curves = np.arange(size * 24, dtype=np.float32).reshape(size, 24)
    return AlignedSplit(inputs, curves, origins, origins[:, None] + np.arange(24)[None, :] * 3600,
        dict(region=region, target_mean=100., target_std=20., preprocessing_hash="frozen", price_transform="identity"),
        (curves - 100) / 20)


def availability(origins):
    return {int(origin + hour * 3600): int(origin + (hour + 1) * 3600)
            for origin in origins for hour in range(-72, 24)}


def test_calendar_year_windows_and_weekly_lags_preserve_local_hour_and_order():
    query = unix("2024-05-15 12:00")
    dates = [pd.Timestamp("2024-05-15 12:00", tz=TZ) - pd.DateOffset(years=year) + pd.DateOffset(days=day)
             for year in (1, 2, 3) for day in range(-7, 8)]
    dates += [pd.Timestamp("2024-05-15 12:00", tz=TZ) - pd.DateOffset(days=day) for day in (7, 14, 21, 28)]
    expected = np.array([int(date.timestamp()) for date in dates])
    memory = HistoricalMemory({"train": split(expected[::-1])}, availability(expected))
    result = memory.candidate_indices(np.array([query]))
    assert result.shape == (1, 49)
    np.testing.assert_array_equal(memory.origins[result[0]], expected)
    assert (np.diff(memory.origins) > 0).all()
    local = pd.to_datetime(memory.origins[result[0]], unit="s", utc=True).tz_convert(TZ)
    assert (local.hour == 12).all()
    assert memory.candidate_validity(result).all()
    assert (memory.candidate_max_availability(result) <= query).all()


def test_leap_day_maps_to_feb28_before_day_offsets():
    origins = [unix("2023-02-28 08:00"), unix("2023-03-01 08:00"), unix("2022-02-28 08:00")]
    memory = HistoricalMemory({"train": split(origins)}, availability(origins),
                               config=dict(day_offsets=[0, 1]))
    result = memory.candidate_indices(np.array([unix("2024-02-29 08:00")]))[0]
    np.testing.assert_array_equal(memory.origins[result[result >= 0]], origins)


@pytest.mark.parametrize("hour", [-72, -1, 0, 23])
def test_missing_history_or_outcome_hour_invalidates_candidate(hour):
    origin, query = unix("2023-05-01 12:00"), unix("2024-05-01 12:00")
    release = availability([origin])
    del release[origin + hour * 3600]
    memory = HistoricalMemory({"train": split([origin])}, release)
    assert not memory.complete[0]
    assert (memory.candidate_indices(np.array([query])) == -1).all()


@pytest.mark.parametrize("hour", [-20, 15])
def test_actual_delayed_release_blocks_candidate_until_available(hour):
    origin, query = unix("2024-05-01 12:00"), unix("2024-05-08 12:00")
    release = availability([origin])
    release[origin + hour * 3600] = query + 1
    blocked = HistoricalMemory({"train": split([origin])}, release)
    assert (blocked.candidate_indices(np.array([query])) == -1).all()
    release[origin + hour * 3600] = query
    allowed = HistoricalMemory({"train": split([origin])}, release)
    found = allowed.candidate_indices(np.array([query]))
    assert found[0, 0] == 0
    assert allowed.candidate_max_availability(found)[0, 0] == query
    assert (allowed.candidate_max_availability(found)[0, 1:] == -1).all()


def test_same_origin_and_future_rows_are_never_eligible_and_early_rows_survive():
    origin = unix("2024-05-01 12:00")
    memory = HistoricalMemory({"test": split([origin])}, availability([origin]),
        config=dict(year_lags=[], day_offsets=[0], weekly_lags_days=[1, 7]))
    queries = np.array([origin - 3600, origin, origin + 23 * 3600])
    result = memory.candidate_indices(queries)
    assert result.shape == (3, 2)
    assert (result == -1).all()
    assert memory.candidate_indices(np.array([], dtype=np.int64)).shape == (0, 2)


def test_evaluation_library_expands_only_after_observations_realized():
    origins = [unix("2022-05-01 12:00"), unix("2023-05-01 12:00"), unix("2024-05-01 12:00")]
    splits = {name: split([origin]) for name, origin in zip(("selection", "test2023", "test2024"), origins)}
    memory = HistoricalMemory(splits, availability(origins))
    before = memory.candidate_indices(np.array([unix("2023-05-01 12:00")]))[0]
    after = memory.candidate_indices(np.array([unix("2024-05-08 12:00")]))[0]
    assert set(memory.origins[before[before >= 0]]) == {origins[0]}
    assert origins[2] in memory.origins[after[after >= 0]]


def test_future_label_poisoning_does_not_change_candidates_or_available_curves():
    past, query, future = unix("2023-05-01 12:00"), unix("2024-05-01 12:00"), unix("2025-05-01 12:00")
    parts = dict(train=split([past]), test=split([query, future]))
    original = HistoricalMemory(parts, availability([past, query, future]))
    poisoned = copy.deepcopy(parts)
    poisoned["test"].actual[:] = np.nan
    poisoned["test"].target[:] = 1e20
    changed = HistoricalMemory(poisoned, availability([past, query, future]))
    original_indices = original.candidate_indices(np.array([query]))
    changed_indices = changed.candidate_indices(np.array([query]))
    np.testing.assert_array_equal(original_indices, changed_indices)
    rows = original_indices[original_indices >= 0]
    np.testing.assert_array_equal(original.curves[rows], changed.curves[rows])
    np.testing.assert_array_equal(original.background_descriptors, changed.background_descriptors)
    np.testing.assert_array_equal(original.event_descriptors, changed.event_descriptors)


def test_raw_levels_use_shared_history_scale_and_descriptors_use_origin_inputs():
    data = split([unix("2023-05-01 12:00")])
    data.inputs["future_exogenous"][0, :, 0] = 17
    memory = HistoricalMemory({"train": data}, availability(data.origin_unix))
    expected = np.median(data.inputs["history_values"][0, -24:, 0].astype(np.float64) * 20 + 100)
    assert memory.levels[0] == expected
    assert memory.curves.dtype == np.float64
    np.testing.assert_array_equal(memory.inputs["future_exogenous"][0], data.inputs["future_exogenous"][0])
    assert 17 in memory.event_descriptors[0]


def test_deduplication_is_stable_and_conflicting_duplicate_origin_fails():
    origin, query = unix("2023-05-01 12:00"), unix("2024-05-01 12:00")
    data = split([origin])
    memory = HistoricalMemory({"one": data, "two": copy.deepcopy(data)}, availability([origin]),
        config=dict(year_lags=[1, 1], day_offsets=[0, 0], weekly_lags_days=[]))
    np.testing.assert_array_equal(memory.candidate_indices(np.array([query])), [[0, -1, -1, -1]])
    conflicting = copy.deepcopy(data)
    conflicting.actual[0, 0] += 1
    with pytest.raises(ValueError, match="Conflicting"):
        HistoricalMemory({"one": data, "two": conflicting}, availability([origin]))


def test_wrong_target_timestamps_and_incompatible_scalers_rejected():
    origin = unix("2023-05-01 12:00")
    data = split([origin])
    data.target_time_unix[0, 20] += 3600
    memory = HistoricalMemory({"train": data}, availability([origin]))
    assert not memory.complete[0]
    other = split([origin + 7 * 86400])
    other.metadata["target_std"] += 1
    with pytest.raises(ValueError, match="preprocessing"):
        HistoricalMemory({"train": data, "test": other}, availability([origin]))


def test_load_raw_availability_uses_actual_release_and_validates_source(tmp_path):
    path = tmp_path / "raw.csv"
    frame = pd.DataFrame(dict(region=["NSW1", "NSW1", "QLD1"],
        delivery_start_aest=["2024-01-01 00:00:00+10:00", "2024-01-01 01:00:00+10:00", "2024-01-01 00:00:00+10:00"],
        available_at_aest=["2024-01-01 01:00:00+10:00", "2024-01-02 12:00:00+10:00", "2024-01-01 01:00:00+10:00"]))
    frame.to_csv(path, index=False)
    values = load_availability(path, "NSW1")
    assert values[unix("2024-01-01 01:00")] == unix("2024-01-02 12:00")
    with pytest.raises(ValueError, match="No availability"):
        load_availability(path, "TAS1")
    pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_availability(path, "NSW1")
    frame.loc[0, "available_at_aest"] = "2023-12-31 23:00:00+10:00"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="before delivery"):
        load_availability(path, "NSW1")


def test_no_availability_does_not_assume_release_schedule():
    memory = HistoricalMemory({"train": split([unix("2023-05-01 12:00")])}, {})
    assert not memory.complete.any()
    assert (memory.candidate_indices(np.array([unix("2024-05-01 12:00")])) == -1).all()


def test_fractional_release_is_not_rounded_back_into_query_information_set(tmp_path):
    path = tmp_path / "fractional.csv"
    pd.DataFrame(dict(region=["NSW1"], delivery_start_aest=["2024-01-01 00:00:00+10:00"],
                     available_at_aest=["2024-01-01 01:00:00.5+10:00"])).to_csv(path, index=False)
    release = load_availability(path, "NSW1")
    assert release[unix("2024-01-01 00:00")] == unix("2024-01-01 01:00") + 1
