"""Official lightgbm library with local aligned selection/output adapter. See README.md."""
import numpy as np


def fit_horizon(h, x_train, y_train, x_select, y_select, x_eval, *, q, task, metric, seed, max_rounds, interval, patience, learning_rate, n_jobs, parameters, artifacts):
    from ..aligned_baselines import _loss
    horizon_count = y_train.shape[1]
    trace = []
    import lightgbm as lgb
    # One binned training/validation dataset is shared by this horizon's
    # separate quantile boosters; each retains its native cached scores.
    train_set = lgb.Dataset(x_train, label=np.ascontiguousarray(y_train[:, h]), free_raw_data=False)
    valid_set = lgb.Dataset(x_select, label=np.ascontiguousarray(y_select[:, h]), reference=train_set, free_raw_data=False)
    common = dict(learning_rate=learning_rate, num_threads=n_jobs, seed=seed,
                  verbosity=-1, bagging_freq=1, metric="None")
    common.update(parameters)
    levels = q if task == "quantile" else [None]
    boosters = []
    for level in levels:
        objective = ({"objective": "quantile", "alpha": float(level)}
                     if level is not None else {"objective": "regression_l1"})
        booster = lgb.Booster(params=dict(common, **objective), train_set=train_set)
        booster.add_valid(valid_set, "selection")
        boosters.append(booster)
    best_score, best_round = np.inf, 1
    for rounds in range(1, max_rounds + 1):
        for booster in boosters:
            booster.update()
        if rounds != 1 and rounds != max_rounds and rounds % interval:
            continue
        cached = []
        def capture(predicted, dataset):
            cached.append(np.array(predicted, copy=True))
            return "cache_only", 0., False
        for booster in boosters:
            booster.eval_valid(feval=capture)
        values = np.column_stack(cached)
        shaped = values[:, None, :] if task == "quantile" else values
        score = _loss(shaped, y_select[:, h:h + 1], q, metric)
        if not np.isfinite(score):
            raise FloatingPointError("Non-finite LightGBM validation score")
        trace.append({"rounds": rounds, "score": score})
        if score < best_score:
            best_score, best_round = score, rounds
        artifacts.progress("fit_horizon", horizon=h, horizon_count=horizon_count,
                           rounds=rounds, selected_rounds=best_round, selection_score=best_score)
        if rounds - best_round >= patience:
            break
    values = np.column_stack([b.predict(x_eval, num_iteration=best_round, num_threads=n_jobs) for b in boosters])
    if task == "point":
        values = values[:, 0]
    actual_rounds = [b.current_iteration() for b in boosters]
    for index, booster in enumerate(boosters):
        artifacts.save(f"h{h:02d}_q{index:02d}.txt", booster, "lightgbm", horizon=h,
                       quantile=float(q[index]) if task == "quantile" else None,
                       selected_rounds=best_round, fitted_rounds=actual_rounds[index])

    return values, best_round, actual_rounds, trace


def fit_shared(x_train, train, *, common, p, q, task, n_jobs):
    from lightgbm import LGBMRegressor
    common.update(verbosity=-1, subsample_freq=1, **p)
    fitted_levels = q if task == "quantile" else [None]
    models = [[LGBMRegressor(**common, objective="quantile", alpha=float(level)).fit(x_train, train.target[:, h])
               if level is not None else LGBMRegressor(**common, objective="regression_l1").fit(x_train, train.target[:, h])
               for level in fitted_levels] for h in range(train.target.shape[1])]
    def predict(x, rounds):
        # Booster avoids sklearn's synthetic feature-name warning for ndarray input.
        counts = [rounds] * len(models) if np.isscalar(rounds) else rounds
        out = np.stack([np.column_stack([m.booster_.predict(x, num_iteration=int(count), num_threads=n_jobs) for m in horizon])
                        for horizon, count in zip(models, counts)], axis=1)
        return out if task == "quantile" else out[..., 0]

    return models, predict
