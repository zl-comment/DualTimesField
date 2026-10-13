"""本地 Lasso ARX72 + CP 适配；不等同于论文 LEAR。来源见 README.md。"""
from typing import Any
import numpy as np
DEFAULT_QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)


def absolute_residual_cp(point: np.ndarray, calibration_point: np.ndarray,
                         calibration_actual: np.ndarray,
                         quantile_levels: Any = DEFAULT_QUANTILES) -> np.ndarray:
    """Author-style symmetric CP, per horizon, with type-7 interpolation.

    This reproduces the absolute-residual formula, not a finite-sample coverage
    guarantee under overlapping hourly observations. Calibration must be OOS;
    its chronological checks are enforced by ``fit_predict``.
    Reference: lipiecki/PostForecasts.jl src/models/cp.jl at 589f940.
    """
    from ..aligned_baselines import _levels
    q = _levels(quantile_levels)
    point, predicted, actual = map(lambda x: np.asarray(x, dtype=float),
                                   (point, calibration_point, calibration_actual))
    if point.ndim != 2 or predicted.ndim != 2 or predicted.shape != actual.shape or not len(actual) or predicted.shape[1] != point.shape[1]:
        raise ValueError("CP requires matching nonempty [origin, horizon] calibration arrays")
    if not all(np.isfinite(x).all() for x in (point, predicted, actual)):
        raise ValueError("CP inputs must be finite")
    widths = np.quantile(np.abs(actual - predicted), np.abs(2 * q - 1), axis=0, method="linear").T
    return point[..., None] + widths[None, ...] * np.sign(q - 0.5)


def fit_lasso(train, selection, evaluation, calibration, *, x_train, x_select, x_eval, q, task, metric, p, mean, std, metadata, artifacts):
    from ..aligned_baselines import _loss, BaselinePrediction
    from ..aligned_inputs import flatten_inputs
    from sklearn.linear_model import Lasso
    alphas = np.asarray(p.pop("alphas", (0.001, 0.01, 0.1, 1.0)), dtype=float)
    max_iter, tol = int(p.pop("max_iter", 10000)), float(p.pop("tol", 1e-4))
    if p:
        raise ValueError(f"Unsupported Lasso parameters: {sorted(p)}")
    if alphas.ndim != 1 or not len(alphas) or not np.isfinite(alphas).all() or np.any(alphas <= 0):
        raise ValueError("alphas must be a nonempty positive finite sequence")
    best, models, trace = np.inf, None, []
    # Fortran layout avoids an additional layout conversion inside coordinate descent.
    # Keep copy_X=True: sklearn centers X in place and must not mutate shared inputs.
    x_train = np.asfortranarray(x_train)
    for alpha_index, alpha in enumerate(alphas):
        candidates = []
        for h in range(train.target.shape[1]):
            artifacts.progress("fit_lasso", alpha=float(alpha), alpha_index=alpha_index, horizon=h,
                               horizon_count=train.target.shape[1])
            model = Lasso(alpha=float(alpha), max_iter=max_iter, tol=tol, copy_X=True).fit(x_train, train.target[:, h])
            candidates.append(model)
            artifacts.save(f"alpha{alpha_index:02d}_h{h:02d}.joblib", model, "joblib",
                           alpha=float(alpha), horizon=h, iterations=int(model.n_iter_), dual_gap=float(model.dual_gap_))
        score = _loss(np.column_stack([m.predict(x_select) for m in candidates]), selection.target, q, metric)
        trace.append({"alpha": float(alpha), "score": score})
        if score < best:
            best, models, chosen = score, candidates, float(alpha)
    point = np.column_stack([m.predict(x_eval) for m in models]) * std + mean
    x_cal = flatten_inputs(calibration)
    cal_point = np.column_stack([m.predict(x_cal) for m in models]) * std + mean
    quantiles = absolute_residual_cp(point, cal_point, calibration.actual, q) if task == "quantile" else None
    artifacts.save("cp_calibration.npz", dict(absolute_residual=np.abs(calibration.actual - cal_point),
                   origin_unix=calibration.origin_unix, target_time_unix=calibration.target_time_unix), "numpy")
    metadata.update(selected_alpha=chosen, selection_trace=trace, seed_effect="none_deterministic_coordinate_descent",
                    point_definition="lasso_center_equals_cp_q50", paper_lear_replication=False,
                    calibration="independent_oos_absolute_residual_type7_per_horizon",
                    calibration_count=len(cal_point), calibration_last_target=int(np.max(calibration.target_time_unix)),
                    postforecasts_source_commit="589f9407d375e2272fb731873443f691c0060d85")
    artifacts.finish(metadata, train, selection, calibration)
    return BaselinePrediction(point, quantiles, metadata)
