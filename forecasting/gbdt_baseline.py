"""Per-horizon gradient-boosted-tree baseline on the forecasting datasets.

One scikit-learn ``HistGradientBoostingRegressor`` is fitted per forecast
horizon with absolute-error loss.  Features are the 72-hour history in
original units, the origin context when configured, and, for the horizon being
predicted, its calendar and future exogenous values.  When the configuration
has CTF exogenous inputs, the horizon's value and the whole 24-hour profile are
added.  The iteration count is chosen on the validation split.  Trees are
invariant to monotone feature transforms, so standardization, clipping, and
soft saturation of an input do not change the fitted model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .datasets import build_region_datasets


def _features(dataset, use_ctf_exogenous: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origins = np.asarray(dataset.origin_indices)
    horizon = dataset.output_hours
    history = (
        dataset.history_values.astype(np.float64) * dataset.history_standardizer.std
        + dataset.history_standardizer.mean
    )
    shared = [np.stack([history[i - dataset.input_hours:i].reshape(-1) for i in origins])]
    if dataset.origin_context_values is not None:
        shared.append(dataset.origin_context_values[origins][:, None])
    per_horizon = [np.stack([dataset.calendar_values[i:i + horizon] for i in origins])]
    origin_unix = dataset.delivery_unix_seconds[origins]
    if dataset.future_exogenous_by_origin:
        per_horizon.append(
            np.stack([dataset.future_exogenous_by_origin[int(t)] for t in origin_unix])
        )
    if use_ctf_exogenous and dataset.ctf_exogenous_by_origin:
        profile = np.stack([dataset.ctf_exogenous_by_origin[int(t)] for t in origin_unix])
        per_horizon.append(profile)
        shared.append(profile.reshape(len(origins), -1))
    raw = dataset.target_values_raw.astype(np.float64)
    target = np.stack([raw[i:i + horizon] for i in origins])
    return np.hstack(shared), np.concatenate(per_horizon, axis=-1), target


def run(config: Path, region: str, output: Path, use_ctf_exogenous: bool, seed: int) -> dict:
    datasets = build_region_datasets(config, region)
    splits = {name: _features(datasets[name], use_ctf_exogenous) for name in ("train", "validation", "test")}
    (x_train, c_train, y_train) = splits["train"]
    (x_val, c_val, y_val) = splits["validation"]
    (x_test, c_test, y_test) = splits["test"]
    prediction = np.zeros_like(y_test)
    iterations = []
    start = time.time()
    for h in range(y_test.shape[1]):
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=63,
            early_stopping=False,
            random_state=seed,
        )
        model.fit(np.hstack([x_train, c_train[:, h]]), y_train[:, h])
        # Choose the iteration count on the chronological validation split.
        best, best_mae = 0, np.inf
        for k, staged in enumerate(model.staged_predict(np.hstack([x_val, c_val[:, h]]))):
            mae = np.abs(staged - y_val[:, h]).mean()
            if mae < best_mae:
                best, best_mae = k, mae
        for k, staged in enumerate(model.staged_predict(np.hstack([x_test, c_test[:, h]]))):
            if k == best:
                prediction[:, h] = staged
                break
        iterations.append(best + 1)
    test = datasets["test"]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        point=prediction,
        actual=y_test,
        origin_unix=test.delivery_unix_seconds[np.asarray(test.origin_indices)],
    )
    error = prediction - y_test
    summary = {
        "config": str(config),
        "region": region,
        "use_ctf_exogenous": use_ctf_exogenous,
        "seed": seed,
        "test_mae": float(np.abs(error).mean()),
        "test_rmse_window_mean": float(np.sqrt((error ** 2).mean(axis=1)).mean()),
        "iterations_per_horizon": iterations,
        "seconds": round(time.time() - start, 1),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--region", required=True, choices=("NSW1", "QLD1", "TAS1"))
    parser.add_argument("--output", required=True, type=Path, help="Destination .npz")
    parser.add_argument("--no-ctf-exogenous", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.region, args.output, not args.no_ctf_exogenous, args.seed)))


if __name__ == "__main__":
    main()
