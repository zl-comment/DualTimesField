"""本地 Lasso ARX72 + CP 适配；不等同于论文 LEAR。来源见 README.md。

论文链接与引用 / References
--------------------------
本地 ARX72+CP，使用 sklearn Lasso；不是原论文 LEAR。本地逐 horizon 对称绝对残差校准，使用 type-7 分位插值；Lei 等仅提供分割共形预测的方法背景，具体公式参考 PostForecasts.jl。插值规则与时间序列非交换性意味着这里不主张该文的有限样本覆盖保证。

BibTeX（可直接复制）: 本目录 CITATIONS.bib；逐项用途与核对来源: source.json。

[tibshirani1996lasso] Lasso 原论文
Tibshirani, Robert (1996). Regression Shrinkage and Selection Via the Lasso. Journal of the Royal
Statistical Society: Series B (Methodological), 58(1), 267–288.
https://doi.org/10.1111/j.2517-6161.1996.tb02080.x

[lei2018distributionfree] 分割共形预测的方法背景
Lei, Jing; G'Sell, Max; Rinaldo, Alessandro; Tibshirani, Ryan J.; Wasserman, Larry (2018).
Distribution-Free Predictive Inference for Regression. Journal of the American Statistical
Association, 113(523), 1094–1111.
https://doi.org/10.1080/01621459.2017.1307116

[lipiecki2025postforecasts] 绝对残差 CP 公式的软件来源
Lipiecki, Arkadiusz; Weron, Rafał (2025). PostForecasts.jl: A Julia package for probabilistic
forecasting by postprocessing point predictions. SoftwareX, 31, 102200.
https://doi.org/10.1016/j.softx.2025.102200
"""
import numpy as np
DEFAULT_QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)

# Moved verbatim to forecasting/common/calibration.py; re-exported so every
# existing import path and the numerical behaviour stay unchanged.
from ..common.calibration import absolute_residual_cp  # noqa: F401,E402


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
