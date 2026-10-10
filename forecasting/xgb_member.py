"""A tunable XGBoost ensemble member that also stores its validation forecasts.

Same model as the ``forecasting.baselines`` XGBoost (one multi-quantile model per forecast hour on the flattened
history, the origin context and that hour's calendar and known inputs; the median is the point forecast), with named
configurations so that hyperparameters and features can be compared with an equal budget:

* ``default``  the stored baseline settings;
* ``profile``  default plus the whole 24-hour profile of every predispatch / PD PASA input at the origin and
               summaries of the recent price (last 24 h maximum, mean and number of hours above 300, 72 h maximum,
               168 h mean), so a forecast hour sees the hours around it;
* ``reg``      default features with stronger regularization (depth 5, minimum child weight 10, lambda 5,
               column sampling 0.5, row sampling 0.7, learning rate 0.03, up to 3000 rounds);
* ``profile_reg`` both.

The validation and test forecasts of one refit are written, so a configuration can be chosen by validation MAE and
ensemble weights learned on the validation quarter, never on the test quarter.

    python -m forecasting.xgb_member --config configs/aemo_forecast_rolling_pd_baseline_raw_q0.yaml \
        --region NSW1 --variant profile --output outputs/forecasting/members/xgb_profile/q0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import xgboost

from .baselines import QUANTILES
from .datasets import build_region_datasets
from .gbdt_baseline import _features

HORIZON = 24
VARIANTS = {
    "default": dict(profile=False, params={}),
    "profile": dict(profile=True, params={}),
    "reg": dict(profile=False, params=dict(max_depth=5, min_child_weight=10, reg_lambda=5.0, colsample_bytree=0.5,
                                           subsample=0.7, learning_rate=0.03, n_estimators=3000)),
    "profile_reg": dict(profile=True, params=dict(max_depth=5, min_child_weight=10, reg_lambda=5.0,
                                                  colsample_bytree=0.5, subsample=0.7, learning_rate=0.03,
                                                  n_estimators=3000)),
}


def profile_features(dataset) -> np.ndarray:
    """Whole forecast-time profile of the known inputs and recent-price summaries, per origin."""
    origins = np.asarray(dataset.origin_indices)
    unix = dataset.delivery_unix_seconds[origins]
    price = np.asarray(dataset.target_values_raw, dtype=np.float64)
    blocks = []
    if dataset.future_exogenous_by_origin:
        blocks.append(np.stack([dataset.future_exogenous_by_origin[int(t)].reshape(-1) for t in unix]))
    summary = np.zeros((len(origins), 5))
    for row, i in enumerate(origins):
        day, three_days, week = price[i - 24:i], price[i - 72:i], price[i - 168:i]
        summary[row] = [day.max(), day.mean(), (day > 300).sum(), three_days.max(), week.mean()]
    blocks.append(summary)
    return np.hstack(blocks)


def run(config: Path, region: str, variant: str, seed: int, device: str) -> dict:
    spec = VARIANTS[variant]
    datasets = build_region_datasets(config, region)
    parts = {}
    for name in ("train", "validation", "test"):
        x, c, y = _features(datasets[name], True)
        if spec["profile"]:
            x = np.hstack([x, profile_features(datasets[name])])
        parts[name] = (x, c, y)
    (x_train, c_train, y_train), (x_val, c_val, y_val), (x_test, c_test, _) = parts["train"], parts["validation"], parts["test"]
    out = {key: np.zeros((x.shape[0], HORIZON, len(QUANTILES))) for key, x in (("val_quantile", x_val), ("test_quantile", x_test))}
    rounds = []
    for h in range(HORIZON):
        arguments = dict(objective="reg:quantileerror", quantile_alpha=np.asarray(QUANTILES), tree_method="hist",
                         device=device, n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, early_stopping_rounds=50, random_state=seed)
        arguments.update(spec["params"])
        model = xgboost.XGBRegressor(**arguments)
        model.fit(np.hstack([x_train, c_train[:, h]]), y_train[:, h],
                  eval_set=[(np.hstack([x_val, c_val[:, h]]), y_val[:, h])], verbose=False)
        out["val_quantile"][:, h] = np.sort(model.predict(np.hstack([x_val, c_val[:, h]])), axis=-1)
        out["test_quantile"][:, h] = np.sort(model.predict(np.hstack([x_test, c_test[:, h]])), axis=-1)
        rounds.append(int(model.best_iteration) + 1)
        print(f"{region} {variant} horizon {h + 1}: {rounds[-1]} rounds", flush=True)
    median = QUANTILES.index(0.5)
    result = {"val_point": out["val_quantile"][..., median], "val_quantile": out["val_quantile"], "val_actual": y_val,
              "test_point": out["test_quantile"][..., median], "test_quantile": out["test_quantile"], "rounds": rounds}
    for name, dataset in (("val", datasets["validation"]), ("test", datasets["test"])):
        origins = np.asarray(dataset.origin_indices)
        result[f"{name}_origin_unix"] = dataset.delivery_unix_seconds[origins]
        if name == "test":
            result["test_actual"] = dataset.target_values_raw[origins[:, None] + np.arange(HORIZON)]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--region", required=True)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    start = time.time()
    result = run(args.config, args.region, args.variant, args.seed, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / f"{args.region}.npz", **{k: v for k, v in result.items() if k != "rounds"})
    (args.output / f"{args.region}.json").write_text(json.dumps({"rounds": result["rounds"], "seconds": time.time() - start}))


if __name__ == "__main__":
    main()
