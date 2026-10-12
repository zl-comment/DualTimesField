"""Behavioral tests of CPU adapters and independent OOS calibration."""
from dataclasses import replace

import numpy as np
import pytest

from forecasting.aligned_inputs import AlignedSplit, INPUT_KEYS
from forecasting.aligned_baselines import absolute_residual_cp, baseline_availability, fit_predict


def split(start, n=40, seed=1):
    rng = np.random.default_rng(seed)
    inputs = {key: rng.normal(size=(n, 2, 1)).astype(np.float32) for key in INPUT_KEYS}
    target = np.column_stack([2 * inputs["history_values"][:, 0, 0],
                              inputs["future_exogenous"][:, 1, 0] - inputs["history_calendar"][:, 0, 0]])
    origin = np.arange(start, start + n * 3, 3) * 3600
    return AlignedSplit(inputs, target * 12 + 45, origin,
                        origin[:, None] + np.array([0, 3600]),
                        {"target_mean": 45., "target_std": 12., "protocol_id": "synthetic-v1",
                         "region": "NSW1", "preprocessing_hash": "train-only-fingerprint",
                         "input_hash": f"inputs-{start}", "sample_hash": f"samples-{start}"}, target)


def test_author_absolute_cp_formula_and_center():
    pred = np.array([[100., 200.]])
    cal_pred = np.zeros((3, 2))
    cal_actual = np.array([[1., 2.], [-2., -4.], [3., 6.]])
    q = absolute_residual_cp(pred, cal_pred, cal_actual)
    np.testing.assert_allclose(q[0, 0], [97.2, 97.4, 100., 102.6, 102.8])
    np.testing.assert_allclose(q[0, 1], [194.4, 194.8, 200., 205.2, 205.6])


@pytest.mark.parametrize("kind", ["xgboost", "lightgbm"])
@pytest.mark.parametrize("task", ["point", "quantile"])
def test_real_cpu_tree_fits_are_raw_and_do_not_use_evaluation_labels(kind, task):
    train, select, evaluate = split(0), split(200, 12, 2), split(400, 8, 3)
    params = {"n_estimators": 5, "early_stopping_rounds": 2, "n_jobs": 1,
              "task": task, "learning_rate": 0.2}
    if kind == "lightgbm":
        params["min_child_samples"] = 3
    first = fit_predict(kind, train, select, evaluate, params=params)
    poisoned = replace(evaluate, actual=np.full_like(evaluate.actual, np.nan),
                       target=np.full_like(evaluate.target, np.nan))
    second = fit_predict(kind, train, select, poisoned, params=params)
    np.testing.assert_array_equal(first.point, second.point)
    assert first.point.shape == (8, 2)
    assert np.isfinite(first.point).all()
    assert first.metadata["prediction_space"] == "raw_rrp"
    assert first.metadata["feature_count"] == 14
    assert len(first.metadata["selected_rounds"]) == 2
    assert first.metadata["selection_scope"] == "per_horizon"
    if task == "quantile":
        assert first.quantiles.shape == (8, 2, 5)
        np.testing.assert_array_equal(first.point, np.sort(first.quantiles, axis=-1)[..., 2])
    else:
        assert first.quantiles is None


def test_lasso_cp_fit_uses_independent_calibration_and_all_features():
    train, select, cal, evaluate = split(0, 60), split(200, 12, 2), split(400, 12, 3), split(600, 8, 4)
    result = fit_predict("lasso_arx72_cp", train, select, evaluate,
                         calibration=cal, params={"alphas": [0.001, 0.01]})
    assert result.quantiles.shape == (8, 2, 5)
    assert np.mean(np.abs(result.point - evaluate.actual)) < 0.2
    np.testing.assert_array_equal(result.point, result.quantiles[..., 2])
    assert result.metadata["paper_lear_replication"] is False
    # Changing calibration residuals affects widths, never the fitted center.
    other = fit_predict("lasso_arx72_cp", train, select, evaluate,
                        calibration=replace(cal, actual=cal.actual + 100), params={"alphas": [0.001, 0.01]})
    np.testing.assert_array_equal(result.point, other.point)
    assert not np.allclose(result.quantiles, other.quantiles)


def test_overlapping_targets_are_rejected_even_when_origins_are_ordered():
    train, select, evaluate = split(0), split(200, 12), split(400, 8)
    overlapping = replace(train, target_time_unix=train.target_time_unix + 10000000)
    with pytest.raises(ValueError, match="realized"):
        fit_predict("xgboost", overlapping, select, evaluate, params={"n_estimators": 2})
    with pytest.raises(ValueError, match="independent"):
        fit_predict("lasso_arx72_cp", train, select, evaluate)
    with pytest.raises(ValueError, match="realized"):
        fit_predict("lasso_arx72_cp", train, select, evaluate, calibration=select)


def test_scaler_or_feature_schema_mismatch_is_rejected():
    train, select, evaluate = split(0), split(200, 12), split(400, 8)
    with pytest.raises(ValueError, match="scaler"):
        fit_predict("xgboost", train, replace(select, metadata=dict(select.metadata, target_mean=0., target_std=1.)), evaluate)
    bad = dict(evaluate.inputs)
    bad["history_values"] = np.ones((8, 3, 1))
    with pytest.raises(ValueError, match="schema"):
        fit_predict("xgboost", train, select, replace(evaluate, inputs=bad))


def test_pending_foundations_fail_explicitly_without_loading_weights():
    statuses = baseline_availability()
    for name in ("tabpfn2", "tabpfn3", "chronos2", "moirai2"):
        assert statuses[name]["implemented"] is False
        with pytest.raises(NotImplementedError):
            fit_predict(name, None, None, None)
    assert "q05/q95" in statuses["moirai2"]["blocker"]


def test_no_arbitrary_objective_override_or_fake_point_quantiles():
    train, select, evaluate = split(0), split(200, 12), split(400, 8)
    with pytest.raises(ValueError, match="Unsupported"):
        fit_predict("xgboost", train, select, evaluate, params={"objective": "reg:squarederror"})
    with pytest.raises(ValueError, match="selection_metric"):
        fit_predict("lightgbm", train, select, evaluate, params={"task": "point", "selection_metric": "pinball"})


def test_metadata_identity_and_delivery_completion_are_enforced():
    train, select, evaluate = split(0), split(200, 12), split(400, 8)
    for key, replacement in (("region", "QLD1"), ("protocol_id", "other"), ("preprocessing_hash", "")):
        with pytest.raises(ValueError, match=key):
            fit_predict("xgboost", train, replace(select, metadata=dict(select.metadata, **{key: replacement})), evaluate)
    # The final delivery starts before selection, but has not completed yet.
    times = train.target_time_unix.copy()
    times[-1, -1] = select.origin_unix[0] - 1800
    with pytest.raises(ValueError, match="realized"):
        fit_predict("xgboost", replace(train, target_time_unix=times), select, evaluate)


@pytest.mark.parametrize("kind", ["xgboost", "lightgbm", "lasso_arx72_cp"])
def test_saved_models_reproduce_predictions_and_have_hashes(kind, tmp_path):
    import hashlib
    import json
    from forecasting.aligned_inputs import flatten_inputs
    train, select, cal, evaluate = split(0, 50), split(200, 12, 2), split(400, 12, 3), split(600, 8, 4)
    parameters = ({"alphas": [0.001, 0.01]} if kind == "lasso_arx72_cp" else
                  {"n_estimators": 6, "early_stopping_rounds": 2, "selection_interval": 2, "n_jobs": 1})
    result = fit_predict(kind, train, select, evaluate, params=parameters,
                         calibration=cal if kind == "lasso_arx72_cp" else None,
                         progress_path=tmp_path / "progress.json", model_directory=tmp_path / "models")
    manifest = json.loads((tmp_path / "models/manifest.json").read_text())
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["stage"] == "complete"
    assert manifest["input_block_order"] == list(INPUT_KEYS)
    for record in manifest["files"]:
        assert hashlib.sha256((tmp_path / "models" / record["path"]).read_bytes()).hexdigest() == record["sha256"]
    x = flatten_inputs(evaluate)
    if kind == "xgboost":
        from xgboost import XGBRegressor
        values = []
        for h in range(2):
            model = XGBRegressor()
            model.load_model(tmp_path / "models" / f"h{h:02d}.ubj")
            values.append(model.predict(x, iteration_range=(0, result.metadata["selected_rounds"][h])))
        reloaded = np.stack(values, axis=1) * 12 + 45
    elif kind == "lightgbm":
        from lightgbm import Booster
        reloaded = np.stack([np.column_stack([
            Booster(model_file=str(tmp_path / "models" / f"h{h:02d}_q{index:02d}.txt")).predict(
                x, num_iteration=result.metadata["selected_rounds"][h], num_threads=1)
            for index in range(5)]) for h in range(2)], axis=1) * 12 + 45
    else:
        import joblib
        index = parameters["alphas"].index(result.metadata["selected_alpha"])
        center = np.column_stack([joblib.load(tmp_path / "models" / f"alpha{index:02d}_h{h:02d}.joblib").predict(x)
                                  for h in range(2)]) * 12 + 45
        with np.load(tmp_path / "models/cp_calibration.npz") as residuals:
            scores = residuals["absolute_residual"]
            reloaded = absolute_residual_cp(center, np.zeros_like(scores), scores)
        assert result.metadata["seed_effect"] == "none_deterministic_coordinate_descent"
    np.testing.assert_allclose(reloaded, result.quantiles, atol=1e-6)
    with pytest.raises(FileExistsError):
        fit_predict(kind, train, select, evaluate, params=parameters,
                    calibration=cal if kind == "lasso_arx72_cp" else None,
                    model_directory=tmp_path / "models")


@pytest.mark.parametrize("kind", ["xgboost", "lightgbm"])
def test_native_early_stopping_reduces_fitted_budget(kind):
    train, select, evaluate = split(0), split(200, 12), split(400, 8)
    train = replace(train, target=np.zeros_like(train.target), actual=np.full_like(train.actual, 45.))
    select = replace(select, target=np.zeros_like(select.target), actual=np.full_like(select.actual, 45.))
    result = fit_predict(kind, train, select, evaluate,
                         params={"n_estimators": 30, "early_stopping_rounds": 2, "selection_interval": 1})
    assert result.metadata["early_stopping_saves_fit_compute"] is True
    assert np.max(result.metadata["actual_fitted_rounds"]) < 30
    assert result.metadata["selected_rounds"] == [1, 1]
    assert max(len(trace) for trace in result.metadata["selection_trace"]) < 30
