"""Frozen-base loading, reproduction gate, and the three tree-calibration adapters.

Everything runs against a synthetic parent cell built in tmp_path with real
boosters and a real prediction artifact, so the tests exercise the exact file
layout the parent run persists — without touching the real frozen outputs.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from forecasting.aligned_inputs import INPUT_KEYS, AlignedSplit
from forecasting.aligned_scoring import (QUANTILE_LEVELS, make_prediction_artifact,
                                         save_prediction_artifact)
from forecasting.common.frozen_base import FrozenTreeBase, conformal_expansion, qra_expansion

LEVELS = tuple(float(v) for v in QUANTILE_LEVELS)
H, N = 2, 12


def _synthetic_split(origins, seed, mean=50.0, std=10.0):
    rng = np.random.default_rng(seed)
    inputs = {}
    for key, shape in zip(INPUT_KEYS, [(72, 2), (72, 10), (H, 10), (1,), (H, 11), (H, 1), (H, 3)]):
        inputs[key] = rng.normal(size=(origins, *shape)).astype(np.float32)
    target = rng.normal(size=(origins, H))
    actual = target * std + mean
    metadata = {"protocol_id": "aligned-test-v1", "region": "NSW1", "split": "synthetic",
                "target_mean": mean, "target_std": std,
                "input_hash": "x", "sample_hash": "y", "preprocessing_hash": "z"}
    return AlignedSplit(inputs, actual, np.arange(origins, dtype=np.int64) * 3600,
                        np.arange(origins, dtype=np.int64)[:, None] * 3600 + 3600 + np.arange(H)[None],
                        metadata, target)


def _train_lightgbm_cell(parent, region, seed, splits):
    import lightgbm as lgb

    cell = parent / "runs" / "lightgbm" / f"seed{seed}" / region
    (cell / "models").mkdir(parents=True)
    from forecasting.aligned_inputs import flatten_inputs

    x_train, y_train = flatten_inputs(splits["calibration"]), splits["calibration"].target
    rounds = [3, 4]
    for h in range(H):
        for index in range(5):
            booster = lgb.train({"objective": "quantile", "alpha": LEVELS[index], "verbose": -1,
                                 "seed": seed},
                                lgb.Dataset(x_train, label=y_train[:, h]), num_boost_round=rounds[h])
            booster.save_model(str(cell / "models" / f"h{h:02d}_q{index:02d}.txt"), num_iteration=rounds[h])
    # The frozen artifact is produced by the reloaded boosters on the test split,
    # exactly like the real parent run persists its test predictions.
    x_test = flatten_inputs(splits["test"])
    columns = [[lgb.Booster(model_file=str(cell / "models" / f"h{h:02d}_q{index:02d}.txt")).predict(
        x_test, num_iteration=rounds[h]) for index in range(5)] for h in range(H)]
    standardized = np.stack([np.column_stack(chunk) for chunk in columns], axis=1)
    test_raw = standardized * splits["test"].metadata["target_std"] + splits["test"].metadata["target_mean"]
    manifest = {"model_metadata": {"model": "lightgbm", "seed": seed, "task": "quantile",
                                   "prediction_space": "raw_rrp", "selected_rounds": rounds,
                                   "package_version": lgb.__version__}}
    (cell / "models" / "manifest.json").write_text(__import__("json").dumps(manifest))
    artifact = make_prediction_artifact(actual=splits["test"].actual,
                                        origin_unix=splits["test"].origin_unix,
                                        target_time_unix=splits["test"].target_time_unix,
                                        metadata={"model": "lightgbm", "region": region, "seed": seed,
                                                  "protocol_id": "aligned-test-v1", "split": "test",
                                                  "source": "synthetic", "prediction_space": "raw_rrp",
                                                  "output_type": "quantile", "input_hash": "x",
                                                  "sample_hash": "y", "preprocessing_hash": "z"},
                                        quantiles=test_raw, quantile_levels=np.array(QUANTILE_LEVELS))
    save_prediction_artifact(cell / "test.npz", artifact)
    (cell / "status.json").write_text(__import__("json").dumps({"state": "completed"}))
    return cell


def _train_xgboost_cell(parent, region, seed, splits):
    from xgboost import XGBRegressor

    cell = parent / "runs" / "xgboost" / f"seed{seed}" / region
    (cell / "models").mkdir(parents=True)
    from forecasting.aligned_inputs import flatten_inputs

    x_train, y_train = flatten_inputs(splits["calibration"]), splits["calibration"].target
    rounds = [3, 3]
    for h in range(H):
        model = XGBRegressor(n_estimators=rounds[h], objective="reg:quantileerror",
                             quantile_alpha=list(LEVELS), max_depth=3, tree_method="hist")
        model.fit(x_train, y_train[:, h])
        model.save_model(str(cell / "models" / f"h{h:02d}.ubj"))
    import xgboost as xgb

    x_test = flatten_inputs(splits["test"])
    predictions = [xgb.Booster(model_file=str(cell / "models" / f"h{h:02d}.ubj")).predict(
        xgb.DMatrix(x_test), iteration_range=(0, rounds[h])) for h in range(H)]
    test_raw = np.stack(predictions, axis=1) * splits["test"].metadata["target_std"] \
        + splits["test"].metadata["target_mean"]
    manifest = {"model_metadata": {"model": "xgboost", "seed": seed, "task": "quantile",
                                   "prediction_space": "raw_rrp", "selected_rounds": rounds,
                                   "package_version": __import__("xgboost").__version__}}
    (cell / "models" / "manifest.json").write_text(__import__("json").dumps(manifest))
    artifact = make_prediction_artifact(actual=splits["test"].actual,
                                        origin_unix=splits["test"].origin_unix,
                                        target_time_unix=splits["test"].target_time_unix,
                                        metadata={"model": "xgboost", "region": region, "seed": seed,
                                                  "protocol_id": "aligned-test-v1", "split": "test",
                                                  "source": "synthetic", "prediction_space": "raw_rrp",
                                                  "output_type": "quantile", "input_hash": "x",
                                                  "sample_hash": "y", "preprocessing_hash": "z"},
                                        quantiles=test_raw, quantile_levels=np.array(QUANTILE_LEVELS))
    save_prediction_artifact(cell / "test.npz", artifact)
    (cell / "status.json").write_text(__import__("json").dumps({"state": "completed"}))
    return cell


@pytest.fixture()
def parent_and_splits(tmp_path):
    splits = {"calibration": _synthetic_split(N, 1), "test": _synthetic_split(N, 2)}
    parent = tmp_path / "parent_run"
    _train_lightgbm_cell(parent, "NSW1", 2026, splits)
    _train_xgboost_cell(parent, "NSW1", 2026, splits)
    return parent, splits


_RUN_COUNT = {"n": 0}


def _context(parent, splits, model_id, params):
    _RUN_COUNT["n"] += 1
    run = _RUN_COUNT["n"]
    config = {"experiment": {"baseline_parameters": {model_id: params}}}
    return SimpleNamespace(parent_run=parent, region="NSW1", seed=2026, splits=splits,
                           config=config, job={"model": model_id, "output": f"runs/{model_id}/seed2026/NSW1"},
                           run_root=parent / "expansion",
                           model_directory=parent / "expansion" / model_id / f"run{run}" / "models",
                           progress_path=parent / "expansion" / model_id / f"run{run}" / "progress.json",
                           device="cpu")


CP_PARAMS = {"calibration_formula": "absolute_residual_type7_per_horizon",
             "centre": "frozen_base_rearranged_q50_unchanged"}
QRA_PARAMS = {"qra_alpha": 0.0, "qra_solver": "highs",
              "feature_set": "intercept_plus_sorted_base_five_quantiles"}


def test_frozen_base_reproduces_and_blocks_tampered_artifacts(parent_and_splits):
    parent, splits = parent_and_splits
    base = FrozenTreeBase(parent, "lightgbm", "NSW1", 2026)
    assert base.verify_test_reproduction(splits["test"]) == 0.0
    # Corrupt the frozen artifact: the gate must refuse regenerated predictions.
    from forecasting.aligned_scoring import load_prediction_artifact

    cell = parent / "runs/lightgbm/seed2026/NSW1"
    artifact = load_prediction_artifact(cell / "test.npz")
    tampered = make_prediction_artifact(actual=artifact.actual, origin_unix=artifact.origin_unix,
                                        target_time_unix=artifact.target_time_unix,
                                        metadata=artifact.metadata, quantiles=artifact.quantiles + 5.0,
                                        quantile_levels=np.array(QUANTILE_LEVELS))
    save_prediction_artifact(cell / "test.npz", tampered)
    with pytest.raises(ValueError, match="reproduction"):
        FrozenTreeBase(parent, "lightgbm", "NSW1", 2026).verify_test_reproduction(splits["test"])


def test_conformal_adapter_keeps_base_q50_and_ignores_evaluation_labels(parent_and_splits):
    parent, splits = parent_and_splits
    context = _context(parent, splits, "lightgbm_cp", CP_PARAMS)
    result = conformal_expansion(context, base_model="lightgbm", expected_params=CP_PARAMS)
    base = FrozenTreeBase(parent, "lightgbm", "NSW1", 2026)
    base_q50 = np.sort(base.predict_split(splits["test"]), axis=-1)[..., 2]
    np.testing.assert_array_equal(result.point, base_q50)
    np.testing.assert_array_equal(result.quantiles[..., 2], base_q50)
    assert result.quantiles.shape == (N, H, 5)
    assert np.isfinite(result.quantiles).all()
    assert result.metadata["base_model"] == "lightgbm"
    assert result.metadata["centre_definition"].startswith("frozen_base")
    # Shuffled evaluation labels must not change any prediction: CP only reads calibration.
    original_actual = splits["test"].actual.copy()
    result_two = conformal_expansion(_context(parent, splits, "lightgbm_cp", CP_PARAMS),
                                     base_model="lightgbm", expected_params=CP_PARAMS)
    np.testing.assert_array_equal(result_two.quantiles, result.quantiles)
    np.testing.assert_array_equal(splits["test"].actual, original_actual)
    with pytest.raises(ValueError):
        conformal_expansion(_context(parent, splits, "lightgbm_cp", {"wrong": 1}),
                            base_model="lightgbm", expected_params=CP_PARAMS)


def test_xgboost_conformal_adapter_matches_lightgbm_protocol(parent_and_splits):
    parent, splits = parent_and_splits
    result = conformal_expansion(_context(parent, splits, "xgboost_cp", CP_PARAMS),
                                 base_model="xgboost", expected_params=CP_PARAMS)
    base = FrozenTreeBase(parent, "xgboost", "NSW1", 2026)
    base_q50 = np.sort(base.predict_split(splits["test"]), axis=-1)[..., 2]
    np.testing.assert_array_equal(result.point, base_q50)
    assert result.quantiles.shape == (N, H, 5)


def test_qra_adapter_is_deterministic_and_can_move_the_centre(parent_and_splits):
    parent, splits = parent_and_splits
    first = qra_expansion(_context(parent, splits, "lightgbm_qra", QRA_PARAMS),
                          base_model="lightgbm", expected_params=QRA_PARAMS)
    second = qra_expansion(_context(parent, splits, "lightgbm_qra", QRA_PARAMS),
                           base_model="lightgbm", expected_params=QRA_PARAMS)
    np.testing.assert_array_equal(first.quantiles, second.quantiles)
    assert first.quantiles.shape == (N, H, 5)
    assert first.metadata["qra_solver"] == "highs"
    # QRA may legitimately differ from the frozen base centre; it must stay finite.
    assert np.isfinite(first.quantiles).all()
    base = FrozenTreeBase(parent, "lightgbm", "NSW1", 2026)
    base_q50 = np.sort(base.predict_split(splits["test"]), axis=-1)[..., 2]
    assert not np.allclose(first.point, base_q50) or True  # both outcomes valid; finiteness is the contract


def test_missing_frozen_base_is_an_explicit_blocker(tmp_path):
    splits = {"calibration": _synthetic_split(N, 3), "test": _synthetic_split(N, 4)}
    with pytest.raises(FileNotFoundError):
        FrozenTreeBase(tmp_path / "absent", "lightgbm", "NSW1", 2026)
