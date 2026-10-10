"""A LightGBM ensemble member (median forecast) with the variants of ``forecasting.xgb_member``.

One L1-loss LightGBM model per forecast hour on the same features as the XGBoost member; ``default``, ``profile``
(plus the 24-hour profile of the known inputs and recent-price summaries), ``reg`` (num_leaves 31, minimum 50 samples per
leaf, lambda 5, feature fraction 0.5, learning rate 0.03) and ``profile_reg``. Validation and test forecasts of
the refit are written in the format of ``xgb_member`` (point forecasts only).

    python -m forecasting.lgbm_member --config configs/aemo_forecast_rolling_pd_baseline_raw_q0.yaml \
        --region NSW1 --variant profile --output outputs/forecasting/members/lgbm_profile/q0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .datasets import build_region_datasets
from .gbdt_baseline import _features
from .xgb_member import HORIZON, profile_features

VARIANTS = {
    "default": dict(profile=False, params={}),
    "profile": dict(profile=True, params={}),
    "reg": dict(profile=False, params=dict(num_leaves=31, min_child_samples=50, reg_lambda=5.0, colsample_bytree=0.5,
                                           learning_rate=0.03, n_estimators=3000)),
    "profile_reg": dict(profile=True, params=dict(num_leaves=31, min_child_samples=50, reg_lambda=5.0,
                                                  colsample_bytree=0.5, learning_rate=0.03, n_estimators=3000)),
}


def run(config: Path, region: str, variant: str, seed: int, threads: int) -> dict:
    spec = VARIANTS[variant]
    datasets = build_region_datasets(config, region)
    parts = {}
    for name in ("train", "validation", "test"):
        x, c, y = _features(datasets[name], True)
        if spec["profile"]:
            x = np.hstack([x, profile_features(datasets[name])])
        parts[name] = (x, c, y)
    (x_train, c_train, y_train), (x_val, c_val, y_val), (x_test, c_test, _) = parts["train"], parts["validation"], parts["test"]
    val_point = np.zeros((x_val.shape[0], HORIZON))
    test_point = np.zeros((x_test.shape[0], HORIZON))
    rounds = []
    for h in range(HORIZON):
        arguments = dict(objective="l1", n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=threads, verbose=-1)
        arguments.update(spec["params"])
        model = lgb.LGBMRegressor(**arguments)
        model.fit(np.hstack([x_train, c_train[:, h]]), y_train[:, h],
                  eval_set=[(np.hstack([x_val, c_val[:, h]]), y_val[:, h])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        val_point[:, h] = model.predict(np.hstack([x_val, c_val[:, h]]))
        test_point[:, h] = model.predict(np.hstack([x_test, c_test[:, h]]))
        rounds.append(int(model.best_iteration_))
        print(f"{region} {variant} horizon {h + 1}: {rounds[-1]} rounds", flush=True)
    result = {"val_point": val_point, "val_actual": y_val, "test_point": test_point, "rounds": rounds}
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
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    start = time.time()
    result = run(args.config, args.region, args.variant, args.seed, args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / f"{args.region}.npz", **{k: v for k, v in result.items() if k != "rounds"})
    (args.output / f"{args.region}.json").write_text(json.dumps({"rounds": result["rounds"], "seconds": time.time() - start}))


if __name__ == "__main__":
    main()
