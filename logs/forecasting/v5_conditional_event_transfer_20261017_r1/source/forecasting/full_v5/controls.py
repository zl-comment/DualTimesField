"""Fixed-budget controls: calibrated variants, stacker inputs, tree boosters."""
from __future__ import annotations

import numpy as np

from . import schema
from .calibrator import CalibratorModel, StackerModel, fit_calibrator, fit_stacker


def fit_all_calibrators(features_by_source: dict, baseline_q_sorted: np.ndarray,
                        actual: np.ndarray, fit_rows: np.ndarray) -> dict[str, CalibratorModel]:
    """Fit C_const, C_state, C_roll, C_recent and V5_history on the same rows.

    features_by_source maps "match" and "recent" to the corresponding full
    [N, 24, 26] feature matrices; controls without history share the match
    matrix because their specs never reference the v columns.
    """
    models = {}
    for model_id in schema.CALIBRATOR_CONTROLS:
        spec = schema.CONTROL_SPECS[model_id]
        features = features_by_source[spec["v_source"] or "match"]
        models[model_id] = fit_calibrator(model_id, schema.CONTROL_SPECS, features,
                                          baseline_q_sorted, actual, fit_rows)
    return models


def calibrator_features_for(model_id: str, features_by_source: dict) -> np.ndarray:
    spec = schema.CONTROL_SPECS[model_id]
    return features_by_source[spec["v_source"] or "match"]


def fit_stacker_control(baseline_q_sorted, v4_q_sorted, actual, fit_rows) -> StackerModel:
    return fit_stacker(baseline_q_sorted, v4_q_sorted, actual, fit_rows)


# --- fixed-budget tree controls -------------------------------------------
def tree_feature_matrix(flat_common: np.ndarray, features: np.ndarray, with_v: bool) -> np.ndarray:
    """[origin, h] rows: 1465 shared columns + z + m (+ v) + h/23 + group one-hot.

    flat_common: [N, 1465]; features: [N, 24, 26].  Returns [N*24, d].
    """
    flat_common = np.asarray(flat_common, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    n = len(flat_common)
    n_feat = schema.NZ + schema.NM + (schema.NV if with_v else 0)
    width = flat_common.shape[1] + n_feat + 1 + len(schema.HORIZON_GROUPS)
    # Zero-initialised so every one-hot column is deterministically 0 before the
    # single active group entry is set to 1; np.empty here previously left the
    # three inactive group columns holding arbitrary memory (round-4 H1).
    out = np.zeros((n * schema.HORIZONS, width), dtype=np.float32)
    for h in range(schema.HORIZONS):
        block = features[:, h, :n_feat] if with_v else features[:, h, :schema.NZ + schema.NM]
        start = h * n
        out[start:start + n, :flat_common.shape[1]] = flat_common
        offset = flat_common.shape[1]
        out[start:start + n, offset:offset + block.shape[1]] = block
        offset += block.shape[1]
        out[start:start + n, offset] = h / 23.0
        offset += 1
        out[start:start + n, offset + schema.horizon_group_of(h)] = 1.0
    return out


def tree_targets(actual: np.ndarray) -> np.ndarray:
    """[N, 24] actuals -> [24*N] rows in the same horizon-major order as the
    feature matrix (row = h * N + origin)."""
    return np.asarray(actual, dtype=np.float64).T.reshape(-1)


def fit_tree_control(model_id: str, matrix: np.ndarray, targets: np.ndarray, params: dict):
    """Five pooled-horizon quantile boosters with the frozen budget."""
    if model_id not in schema.TREE_CONTROLS:
        raise ValueError(f"Unknown tree control {model_id}")
    import lightgbm as lgb
    boosters = []
    training = dict(objective="quantile", learning_rate=float(params["learning_rate"]),
                    num_leaves=int(params["num_leaves"]), max_depth=int(params["max_depth"]),
                    min_child_samples=int(params["min_child_samples"]), max_bin=int(params["max_bin"]),
                    feature_fraction=float(params["feature_fraction"]),
                    bagging_fraction=float(params["bagging_fraction"]),
                    seed=int(params["seed"]), num_threads=int(params["threads"]), verbosity=-1)
    for level in schema.QUANTILE_LEVELS:
        booster = lgb.train({**training, "alpha": float(level)}, lgb.Dataset(matrix, label=targets),
                            num_boost_round=int(params["rounds"]))
        boosters.append(booster)
    return boosters


def predict_tree_control(boosters, matrix: np.ndarray, n_origins: int) -> np.ndarray:
    """[24*N, d] -> sorted quantiles [N, 24, 5]; rows are horizon-major."""
    q = np.stack([booster.predict(matrix) for booster in boosters], axis=1)
    q = q.reshape(schema.HORIZONS, n_origins, 5).transpose(1, 0, 2)
    return np.sort(q, axis=-1)


def save_tree_control(boosters, directory) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for level, booster in zip(schema.QUANTILE_LEVELS, boosters):
        name = f"q{int(round(level * 100)):02d}.txt"
        booster.save_model(str(directory / name), num_iteration=booster.num_trees())


def load_tree_control(directory, params: dict):
    import lightgbm as lgb
    boosters = []
    for level in schema.QUANTILE_LEVELS:
        name = f"q{int(round(level * 100)):02d}.txt"
        boosters.append(lgb.Booster(model_file=str(directory / name)))
    return boosters
