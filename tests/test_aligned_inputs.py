"""Input-boundary tests: information equality and train/development isolation."""
import copy

import numpy as np
import pandas as pd
import pytest

from forecasting.aligned_inputs import (
    INPUT_KEYS, AlignedWindows, build_aligned_splits, flatten_inputs,
)


@pytest.fixture
def config(tmp_path):
    delivery = pd.date_range("2019-12-31", periods=96, freq="h", tz="Australia/Brisbane")
    values = np.arange(96, dtype=float)
    frame = pd.DataFrame({
        "delivery": delivery, "available": delivery + pd.Timedelta(hours=1),
        "region": "NSW1", "split": ["train"] * 48 + ["validation"] * 24 + ["test"] * 24,
        "price": 10 + values ** 1.2, "demand": 100 + values * 3 + np.sin(values),
    })
    frame.to_csv(tmp_path / "prices.csv", index=False)
    origins = delivery.tz_convert("UTC").to_numpy(dtype="datetime64[s]").astype(np.int64)
    forecasts = values[:, None] + np.arange(3)[None, :] * 0.3
    np.savez(tmp_path / "exo.npz", forecast_origin_unix=origins,
             source_run_unix=origins - 60, source_last_changed_unix=origins - 30,
             load=forecasts)
    dates = pd.date_range("2019-12-20", "2020-01-04", freq="D")
    pd.DataFrame({"Gas_Date": dates, "Price": np.arange(len(dates)) + 1}).to_csv(
        tmp_path / "gas.csv", index=False)
    exo = {"enabled": True, "features": ["load"], "region_files": {"NSW1": "exo.npz"}}
    return {
        "project_root": str(tmp_path), "aligned_protocol": {"id": "fixture-v1"},
        "data": {
            "timezone": "Australia/Brisbane", "delivery_column": "delivery",
            "available_column": "available", "region_column": "region", "split_column": "split",
            "history_columns": ["price", "demand"], "target_column": "price",
            "price_transform": "asinh", "quantile_target": "price",
            "region_files": {"NSW1": "prices.csv"},
            "split_names": ["train", "validation", "test"],
            "split_boundaries": {
                "train": {"start": "2019-12-31", "end_exclusive": "2020-01-02"},
                "validation": {"start": "2020-01-02", "end_exclusive": "2020-01-03"},
                "test": {"start": "2020-01-03", "end_exclusive": "2020-01-04"},
            },
        },
        "forecast_protocol": {"input_hours": 4, "output_hours": 3, "stride_hours": 1,
                              "calendar_features": ["hour_sin", "year_standardized"]},
        "origin_context": {"enabled": True, "path": "gas.csv", "window_days": 3},
        "future_exogenous": copy.deepcopy(exo), "ctf_exogenous": copy.deepcopy(exo),
        "quantile_exogenous": copy.deepcopy(exo),
    }


def test_shared_inputs_are_identical_in_tree_and_torch(config):
    split = build_aligned_splits(config, "NSW1", split_names=["validation"], max_origins=4)["validation"]
    assert tuple(split.inputs) == INPUT_KEYS
    sample = AlignedWindows(split)[0]
    from_torch = np.concatenate([sample[key].numpy().reshape(-1) for key in INPUT_KEYS])
    np.testing.assert_array_equal(flatten_inputs(split)[0], from_torch)
    assert all(split.inputs[key].size > 0 for key in INPUT_KEYS)
    assert len(split) == 4
    np.testing.assert_array_equal(sample["target_price"], sample["target_quantile"])
    assert "target_price" not in split.inputs
    assert config["data"]["price_transform"] == "asinh"  # caller config remains immutable


def test_samples_targets_and_affine_price_space(config):
    splits = build_aligned_splits(config, "NSW1", max_origins=5)
    raw = pd.read_csv(config["project_root"] + "/prices.csv")["price"].to_numpy()
    for split in splits.values():
        np.testing.assert_array_equal(split.target_time_unix[:, 0], split.origin_unix)
        np.testing.assert_array_equal(np.diff(split.target_time_unix, axis=1), np.full((5, 2), 3600))
        rows = split.dataset.origin_indices[:, None] + np.arange(3)
        np.testing.assert_array_equal(split.actual, raw[rows].astype(np.float32))
        np.testing.assert_array_equal(
            split.target, split.dataset.history_values[rows, split.dataset.target_index])
        assert split.metadata["target_mean"] == pytest.approx(raw[:48].mean())
        assert split.metadata["target_std"] == pytest.approx(raw[:48].std())
        restored = split.target * split.metadata["target_std"] + split.metadata["target_mean"]
        np.testing.assert_allclose(restored, split.actual, rtol=1e-6)
        history_rows = split.dataset.origin_indices[:, None] + np.arange(-4, 0)
        expected = (raw[history_rows] - raw[:48].mean()) / raw[:48].std()
        np.testing.assert_allclose(split.inputs["history_values"][..., 0], expected, rtol=1e-6)
    assert len({s.metadata["preprocessing_hash"] for s in splits.values()}) == 1


def test_later_test_corruption_cannot_change_development(config):
    before = build_aligned_splits(config, "NSW1", split_names=["train", "validation"], max_origins=3)
    price_path = config["project_root"] + "/prices.csv"
    frame = pd.read_csv(price_path)
    frame.loc[72:, ["price", "demand"]] = np.nan
    frame.to_csv(price_path, index=False)
    exo_path = config["project_root"] + "/exo.npz"
    with np.load(exo_path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["load"][72:] = np.nan
    arrays["source_last_changed_unix"][72:] += 10_000
    np.savez(exo_path, **arrays)
    after = build_aligned_splits(config, "NSW1", split_names=["train", "validation"], max_origins=3)
    for name in before:
        for key in ("preprocessing_hash", "input_hash", "sample_hash"):
            assert before[name].metadata[key] == after[name].metadata[key]
    with pytest.raises(ValueError, match="Missing values"):
        build_aligned_splits(config, "NSW1", max_origins=3)


def test_validation_targets_do_not_fit_scalers(config):
    before = build_aligned_splits(config, "NSW1", split_names=["validation"], max_origins=1)["validation"]
    path = config["project_root"] + "/prices.csv"
    frame = pd.read_csv(path)
    frame.loc[48:, "price"] += 100_000
    frame.to_csv(path, index=False)
    after = build_aligned_splits(config, "NSW1", split_names=["validation"], max_origins=1)["validation"]
    assert before.metadata["target_mean"] == after.metadata["target_mean"]
    assert before.metadata["target_std"] == after.metadata["target_std"]
    assert before.metadata["preprocessing_hash"] == after.metadata["preprocessing_hash"]
    # First validation origin has only training history; altered labels cannot enter inputs.
    assert before.metadata["input_hash"] == after.metadata["input_hash"]
    assert before.metadata["sample_hash"] != after.metadata["sample_hash"]


def test_missing_features_remain_explicit_empty_blocks(config):
    for key in ("origin_context", *INPUT_KEYS[-3:]):
        config[key]["enabled"] = False
    split = build_aligned_splits(config, "NSW1", split_names=["train"], max_origins=2)["train"]
    assert tuple(split.inputs) == INPUT_KEYS
    assert split.inputs["origin_context"].shape == (2, 0)
    assert split.inputs["quantile_exogenous"].shape == (2, 3, 0)
    assert flatten_inputs(split).shape[0] == 2


def test_reject_training_truncation_and_price_cap(config):
    with pytest.raises(ValueError, match="training period"):
        build_aligned_splits(config, "NSW1", end_exclusive="2020-01-01")
    config["data"]["price_cap"] = 650
    with pytest.raises(ValueError, match="price_cap"):
        build_aligned_splits(config, "NSW1")


def test_available_forecast_vintages_enforced(config):
    path = config["project_root"] + "/exo.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["source_run_unix"][10] += 7200
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="Future information leakage"):
        build_aligned_splits(config, "NSW1", split_names=["train"])


def test_same_preprocessing_identity_when_loading_extra_scoring_split(config):
    dev = build_aligned_splits(config, "NSW1", split_names=["validation"], max_origins=2)["validation"]
    all_splits = build_aligned_splits(config, "NSW1", max_origins=2)["validation"]
    for key in ("preprocessing_hash", "sample_hash", "input_hash"):
        assert dev.metadata[key] == all_splits.metadata[key]


def test_nonfinite_values_and_invalid_sample_limit_rejected(config):
    for value in (0, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            build_aligned_splits(config, "NSW1", max_origins=value)
    path = config["project_root"] + "/prices.csv"
    frame = pd.read_csv(path)
    frame.loc[48, "price"] = np.inf
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Non-finite aligned"):
        build_aligned_splits(config, "NSW1", split_names=["validation"], max_origins=1)
