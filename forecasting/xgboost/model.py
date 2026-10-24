"""Official xgboost library with local aligned selection/output adapter. See README.md.

论文链接与引用 / References
--------------------------
官方库调用；逐 horizon、五分位输出和公共选择流程为本地适配。

BibTeX（可直接复制）: 本目录 CITATIONS.bib；逐项用途与核对来源: source.json。

[chen2016xgboost] XGBoost 原论文
Chen, Tianqi; Guestrin, Carlos (2016). XGBoost: A Scalable Tree Boosting System. KDD, 785–794.
https://doi.org/10.1145/2939672.2939785
"""
import numpy as np


def fit_horizon(h, x_train, y_train, x_select, y_select, x_eval, *, q, task, metric, seed, max_rounds, interval, patience, learning_rate, n_jobs, parameters, artifacts):
    from ..aligned_baselines import _loss
    horizon_count = y_train.shape[1]
    trace = []
    from xgboost import XGBRegressor
    from xgboost.callback import TrainingCallback

    def aligned_score(actual, forecast):
        shape = (len(actual), 1, len(q)) if task == "quantile" else (len(actual), 1)
        return _loss(np.asarray(forecast).reshape(shape), actual[:, None], q, metric)

    class SelectPrefix(TrainingCallback):
        def __init__(self):
            self.best_score, self.best_round = np.inf, 1

        def after_iteration(self, model, epoch, evals_log):
            rounds = epoch + 1
            if rounds != 1 and rounds != max_rounds and rounds % interval:
                return False
            score = float(evals_log["validation_0"]["aligned_score"][-1])
            if not np.isfinite(score):
                raise FloatingPointError("Non-finite XGBoost validation score")
            trace.append({"rounds": rounds, "score": score})
            if score < self.best_score:
                self.best_score, self.best_round = score, rounds
            artifacts.progress("fit_horizon", horizon=h, horizon_count=horizon_count,
                               rounds=rounds, selected_rounds=self.best_round, selection_score=self.best_score)
            return rounds - self.best_round >= patience

    callback = SelectPrefix()
    common = dict(n_estimators=max_rounds, learning_rate=learning_rate, n_jobs=n_jobs,
                  random_state=seed, max_depth=6, tree_method="hist", disable_default_eval_metric=True,
                  eval_metric=aligned_score, callbacks=[callback])
    common.update(parameters)
    objective = ({"objective": "reg:quantileerror", "quantile_alpha": q}
                 if task == "quantile" else {"objective": "reg:absoluteerror"})
    model = XGBRegressor(**common, **objective).fit(
        x_train, y_train[:, h], eval_set=[(x_select, y_select[:, h])], verbose=False)
    best_round = callback.best_round
    values = model.predict(x_eval, iteration_range=(0, best_round))
    actual_rounds = model.get_booster().num_boosted_rounds()
    artifacts.save(f"h{h:02d}.ubj", model, "xgboost", horizon=h,
                   selected_rounds=best_round, fitted_rounds=actual_rounds)

    return values, best_round, actual_rounds, trace


def fit_shared(x_train, train, *, common, p, q, task, n_jobs):
    from xgboost import XGBRegressor
    common.update(max_depth=6, tree_method="hist")
    common.update(p)
    objective = {"objective": "reg:quantileerror", "quantile_alpha": q} if task == "quantile" else {"objective": "reg:absoluteerror"}
    models = [XGBRegressor(**common, **objective).fit(x_train, train.target[:, h]) for h in range(train.target.shape[1])]
    def predict(x, rounds):
        counts = [rounds] * len(models) if np.isscalar(rounds) else rounds
        values = [m.predict(x, iteration_range=(0, int(count))) for m, count in zip(models, counts)]
        return np.stack(values, axis=1)

    return models, predict
