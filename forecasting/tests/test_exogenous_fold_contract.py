"""Fold-local missing-vintage fills and independently recorded horizon times."""
import copy
import hashlib

import numpy as np
import pandas as pd
import pytest

from forecasting import datasets
from forecasting.aligned_inputs import build_aligned_splits, annotate_cached_input_contracts
from forecasting.build_predispatch_exogenous import _align
from forecasting.build_pdpasa_exogenous import _write_region


@pytest.fixture
def config(tmp_path):
    times = pd.date_range("2019-12-31", periods=96, freq="h", tz="Australia/Brisbane")
    x = np.arange(96, dtype=float)
    pd.DataFrame({"delivery": times, "available": times + pd.Timedelta(hours=1),
                  "region": "NSW1", "split": "unused", "price": x + np.sin(x),
                  "demand": 100 + x}).to_csv(tmp_path / "history.csv", index=False)
    origins = times.as_unit("s").asi8
    values = x[:, None] + np.arange(3)[None, :] / 10
    np.savez(tmp_path / "base.npz", forecast_origin_unix=origins, load=values,
             source_run_unix=origins - 60, source_last_changed_unix=origins - 60)
    changed = origins - 60
    changed[50] = origins[50] - 30000
    values[50] = 999999  # Legacy global-fill value: must not survive this fold.
    np.savez(tmp_path / "pred.npz", forecast_origin_unix=origins, predispatch_rrp=values,
             source_run_unix=changed, source_last_changed_unix=changed)
    return {"project_root": str(tmp_path), "data": {
        "timezone": "Australia/Brisbane", "delivery_column": "delivery",
        "available_column": "available", "region_column": "region", "split_column": "split",
        "history_columns": ["price", "demand"], "target_column": "price",
        "region_files": {"NSW1": "history.csv"}, "split_names": ["train", "validation", "test"],
        "split_boundaries": {
            "train": {"start": "2019-12-31", "end_exclusive": "2020-01-02"},
            "validation": {"start": "2020-01-02", "end_exclusive": "2020-01-03"},
            "test": {"start": "2020-01-03", "end_exclusive": "2020-01-04"}}},
        "forecast_protocol": {"input_hours": 4, "output_hours": 3, "stride_hours": 1,
                              "calendar_features": ["hour_sin", "year_standardized"]},
        "future_exogenous": {"enabled": True, "features": ["load"],
                             "region_files": {"NSW1": "base.npz"}},
        "predispatch_exogenous": {"enabled": True, "features": ["predispatch_rrp"],
                                  "region_files": {"NSW1": "pred.npz"}}}


def rewrite(path, change):
    with np.load(path) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    change(arrays)
    np.savez(path, **arrays)


def test_legacy_fill_recomputed_without_later_fold_values(config, tmp_path):
    before = build_aligned_splits(config, "NSW1", split_names=["validation"])["validation"]
    def perturb(arrays):
        arrays["predispatch_rrp"][48:] += 100000
    rewrite(tmp_path / "pred.npz", perturb)
    after = build_aligned_splits(config, "NSW1", split_names=["validation"])["validation"]
    # Row 50 is the third validation origin. Its replacement is train-only.
    np.testing.assert_array_equal(before.inputs["future_exogenous"][2], after.inputs["future_exogenous"][2])
    assert before.metadata["exogenous_delivery_time_contracts"]["predispatch_exogenous"].startswith("unverified_legacy")
    assert before.metadata["exogenous_imputation_contracts"]["predispatch_exogenous"] == "fold_train_raw_median_float32"


def test_new_raw_mask_matches_legacy_reconstruction(config, tmp_path):
    before = build_aligned_splits(config, "NSW1", split_names=["validation"])["validation"]
    def upgrade(arrays):
        origins = arrays["forecast_origin_unix"]
        arrays["stale_origin_mask"] = origins - arrays["source_last_changed_unix"] > 21600
        arrays["predispatch_rrp"][arrays["stale_origin_mask"]] = np.nan
        arrays["target_time_unix"] = origins[:, None] + np.arange(3)[None, :] * 3600
    rewrite(tmp_path / "pred.npz", upgrade)
    after = build_aligned_splits(config, "NSW1", split_names=["validation"])["validation"]
    np.testing.assert_array_equal(before.inputs["future_exogenous"], after.inputs["future_exogenous"])
    assert after.metadata["exogenous_delivery_time_contracts"]["predispatch_exogenous"] == "verified_archive_target_time_unix"


@pytest.mark.parametrize("fault", ["horizon", "mask"])
def test_reject_wrong_recorded_horizons_or_stale_mask(config, tmp_path, fault):
    def corrupt(arrays):
        origins = arrays["forecast_origin_unix"]
        if fault == "horizon":
            arrays["target_time_unix"] = origins[:, None] + (np.arange(3)[None, :] + 1) * 3600
        else:
            arrays["stale_origin_mask"] = np.zeros(len(origins), dtype=bool)
    rewrite(tmp_path / "pred.npz", corrupt)
    with pytest.raises(ValueError, match="delivery timestamps|stale-origin mask"):
        build_aligned_splits(config, "NSW1", split_names=["validation"])


def test_legacy_preservation_requires_exact_bytes_and_fold(tmp_path, monkeypatch):
    path = tmp_path / "archive.npz"
    np.savez(path, value=np.array([1.0]))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(datasets, "_LEGACY_PREDISPATCH_2015_2021_HASHES", {digest})
    cfg = {"data": {"timezone": "Australia/Brisbane", "split_boundaries": {
        "train": {"start": "2015-01-01", "end_exclusive": "2022-01-01"}}}}
    assert datasets._legacy_imputation_matches_fold(path, cfg)
    earlier = copy.deepcopy(cfg)
    earlier["data"]["split_boundaries"]["train"]["end_exclusive"] = "2020-01-01"
    assert not datasets._legacy_imputation_matches_fold(path, earlier)
    np.savez(path, value=np.array([2.0]))
    assert not datasets._legacy_imputation_matches_fold(path, cfg)


def test_builder_keeps_stale_raw_values_and_delivery_times(tmp_path):
    origin = int(pd.Timestamp("2019-01-01", tz="Australia/Brisbane").timestamp())
    rows = pd.DataFrame([{"PREDISPATCHSEQNO": run + 1, "period_end": start + h * 1800,
                          "changed": start, "value": float(run + h)}
                         for run, start in enumerate([origin, origin + 10 * 3600])
                         for h in range(1, 49)])
    path = tmp_path / "new.npz"
    _align("NSW1", rows, ["value"], ["predispatch_rrp"], "predispatch_covered", path)
    with np.load(path) as archive:
        stale = archive["stale_origin_mask"]
        assert stale.any()
        assert np.isnan(archive["predispatch_rrp"][stale]).all()
        assert (archive["predispatch_covered"][stale] == 0).all()
        np.testing.assert_array_equal(archive["target_time_unix"],
            archive["forecast_origin_unix"][:, None] + np.arange(24)[None, :] * 3600)


def test_builders_refuse_to_replace_legacy_archive(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, sentinel=np.array([123]))
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="Preserve existing"):
        _align("NSW1", pd.DataFrame(), [], [], "predispatch_covered", path)
    with pytest.raises(FileExistsError, match="Preserve existing"):
        _write_region("NSW1", {}, {}, path)
    assert path.read_bytes() == before


def test_old_cache_metadata_stays_explicitly_unverified_and_immutable():
    original = {"input_hash": "frozen"}
    annotated = annotate_cached_input_contracts(original)
    assert original == {"input_hash": "frozen"}
    assert annotated["input_hash"] == "frozen"
    assert annotated["exogenous_delivery_time_contracts"]["status"].startswith("unverified_legacy_cache")
    assert annotate_cached_input_contracts(annotated) == annotated
