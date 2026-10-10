"""Does the history carry exploitable information that the forecast heads miss?

A cheap upper-bound probe for history-attending heads. The quarterly XGBoost-with-predispatch baseline
(``forecasting.baselines``, seed 2026) sees the 72-hour price and demand history, the calendar and every
forward-looking input, but not how wrong the operator's own predispatch has recently been. This script refits
the same model with extra history-conditional features, all known at the origin:

* recent predispatch error: the mean of (realized price - the predispatch price its own run gave for that
  hour, lead 0) over the last 3, 6 and 24 hours, and the 24-hour mean absolute and maximum error;
* per forecast hour, the same error of the matching hour yesterday, two days ago and a week ago (the
  analog a query on the forecast hour would pick from the history).

Everything else (features, hyperparameters, seed, splits, device) is unchanged, so the difference from the
stored baseline is the value of this history information. Usage::

    python -m forecasting.history_probe --config configs/aemo_forecast_rolling_pd_baseline_raw_q0.yaml \
        --region NSW1 --output outputs/forecasting/history_probe/q0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost

from .baselines import QUANTILES
from .datasets import build_region_datasets, load_forecast_config
from .gbdt_baseline import _features

HORIZON = 24
PREDISPATCH = "data/aemo_exogenous/{region}_predispatch.npz"


def predispatch_error(dataset, region: str, root: Path) -> np.ndarray:
    """Realized price minus the lead-0 predispatch price, for every row of the frame (NaN if uncovered)."""
    with np.load(root / PREDISPATCH.format(region=region.lower())) as archive:
        origins = archive["forecast_origin_unix"].astype(np.int64)
        lead0 = archive["predispatch_rrp"][:, 0].astype(np.float64)
        covered = archive["predispatch_covered"][:, 0] > 0
        last_changed = archive["source_last_changed_unix"].astype(np.int64)
    if np.any(last_changed > origins):
        raise ValueError("predispatch run published after its origin")
    lead0 = np.where(covered, lead0, np.nan)
    position = pd.Series(np.arange(len(origins)), index=origins).reindex(dataset.delivery_unix_seconds)
    found = ~position.isna().to_numpy()
    pd0 = np.full(len(dataset.delivery_unix_seconds), np.nan)
    pd0[found] = lead0[position.to_numpy()[found].astype(int)]
    return dataset.target_values_raw.astype(np.float64) - pd0


def extras(dataset, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shared [N, 5] and per-horizon [N, 24, 3] features, from errors strictly before each origin."""
    origins = np.asarray(dataset.origin_indices)
    shared = np.full((len(origins), 5), np.nan)
    per_horizon = np.full((len(origins), HORIZON, 3), np.nan)
    with np.errstate(all="ignore"):
        for row, i in enumerate(origins):
            window = error[i - 24:i]
            shared[row] = [np.nanmean(error[i - 3:i]), np.nanmean(error[i - 6:i]), np.nanmean(window),
                           np.nanmean(np.abs(window)), np.nanmax(np.abs(window))]
            for lag, column in ((24, 0), (48, 1), (168, 2)):
                per_horizon[row, :, column] = error[i + np.arange(HORIZON) - lag]
    return shared, per_horizon


def run(config_path: Path, region: str, seed: int, device: str) -> dict:
    config = load_forecast_config(config_path)
    root = Path(config["project_root"])
    datasets = build_region_datasets(config_path, region)
    error = predispatch_error(datasets["train"], region, root)
    parts = {}
    for name in ("train", "validation", "test"):
        x, c, y = _features(datasets[name], True)
        shared, per_horizon = extras(datasets[name], error)
        parts[name] = (np.hstack([x, shared]), np.concatenate([c, per_horizon], axis=-1), y)
    (x_train, c_train, y_train), (x_val, c_val, y_val), (x_test, c_test, _) = parts["train"], parts["validation"], parts["test"]
    point = np.zeros((x_test.shape[0], HORIZON))
    quantile = np.zeros((x_test.shape[0], HORIZON, len(QUANTILES)))
    rounds = []
    for h in range(HORIZON):
        model = xgboost.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=np.asarray(QUANTILES), tree_method="hist", device=device,
            n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=50, random_state=seed,
        )
        model.fit(np.hstack([x_train, c_train[:, h]]), y_train[:, h],
                  eval_set=[(np.hstack([x_val, c_val[:, h]]), y_val[:, h])], verbose=False)
        predicted = np.sort(model.predict(np.hstack([x_test, c_test[:, h]])), axis=-1)
        quantile[:, h] = predicted
        point[:, h] = predicted[:, QUANTILES.index(0.5)]
        rounds.append(int(model.best_iteration) + 1)
        print(f"{region} horizon {h + 1}: {rounds[-1]} rounds", flush=True)
    test = datasets["test"]
    return {
        "point": point, "quantile": quantile,
        "actual": test.target_values_raw[np.asarray(test.origin_indices)[:, None] + np.arange(HORIZON)],
        "origin_unix": test.delivery_unix_seconds[np.asarray(test.origin_indices)],
        "rounds": rounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--region", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    start = time.time()
    result = run(args.config, args.region, args.seed, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / f"{args.region}.npz", **{k: v for k, v in result.items() if k != "rounds"})
    (args.output / f"{args.region}.json").write_text(json.dumps({"rounds": result["rounds"], "seconds": time.time() - start}))


if __name__ == "__main__":
    main()
