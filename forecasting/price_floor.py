"""A validation-chosen price floor for recalibrated dual-field forecasts.

A linear head extrapolates in the asinh target space, and the inverse
transform turns that into negative prices far below any observed (diagnosis in
``EXPERIMENT_RESULTS.md``, "QLD1 gap to XGBoost"). Negative prices have a
market floor in practice (renewable offers priced near minus their certificate
value), so forecasts below it are errors.

For every quarterly refit of ``forecasting.rolling`` and every region, this
script predicts the refit's validation and test quarters, and clips the
forecasts from below at a level chosen on the validation quarter only: none,
or one of several quantiles of that refit's training prices, capped at zero. The point
forecast's floor minimizes the seed-averaged validation MAE, the quantiles'
floor the validation CRPS~. The stitched test forecasts are written like
``forecasting.rolling collect`` output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .datasets import build_region_datasets, load_forecast_config
from .evaluate_paper_metrics import collect_predictions, point_metrics, probabilistic_metrics
from .rolling import QUARTERS, REGIONS, SEEDS, _checkpoint, quarter_config_path
from .train import build_model, resolve_device

CANDIDATES = (None, 0.0005, 0.001, 0.002, 0.005, 0.01)
LEVELS = np.array([0.05, 0.10, 0.50, 0.90, 0.95])


def _crps(quantile: np.ndarray, actual: np.ndarray) -> float:
    error = actual[..., None] - np.sort(quantile, axis=-1)
    return float(2 * np.maximum(LEVELS * error, (LEVELS - 1) * error).mean())


def _floor(values: np.ndarray, level: float | None) -> np.ndarray:
    return values if level is None else np.maximum(values, level)


def run(name: str, region: str, device_name: str, checkpoint_name: str = "best_model.pt") -> dict:
    device = resolve_device(device_name)
    test_parts = {"point": [], "quantile": [], "actual": [], "naive": [], "origin": [], "point_unfloored": [], "quantile_unfloored": []}
    # The floored validation forecasts of every refit, so that ensemble weights can be learned without the test period.
    val_parts = {"point": [], "quantile": [], "actual": [], "origin": [], "refit": []}
    choices = []
    for k in range(len(QUARTERS) - 1):
        config_path = quarter_config_path(name, k)
        config = load_forecast_config(config_path)
        datasets = build_region_datasets(config_path, region)
        train = datasets["train"]
        raw = np.asarray(train.target_values_raw, dtype=np.float64)
        train_prices = np.concatenate([raw[i:i + train.output_hours] for i in train.origin_indices])
        # The floor bounds negative extrapolation, so it is never above zero.
        levels = {q: (None if q is None else min(float(np.quantile(train_prices, q)), 0.0)) for q in CANDIDATES}
        predictions = {"validation": [], "test": []}
        for seed in SEEDS:
            model = build_model(config).to(device)
            checkpoint = torch.load(_checkpoint(config, seed, region).with_name(checkpoint_name), map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            model.set_epoch(checkpoint.get("model_epoch", checkpoint["best_epoch"]))
            for split in predictions:
                predictions[split].append(collect_predictions(model, datasets[split], device, 1024))
        validation_actual = predictions["validation"][0]["actual"]
        point_scores = {q: np.mean([np.abs(_floor(p["point"], f) - validation_actual).mean() for p in predictions["validation"]])
                        for q, f in levels.items()}
        quantile_scores = {q: np.mean([_crps(_floor(p["quantile"], f), validation_actual) for p in predictions["validation"]])
                           for q, f in levels.items()}
        point_choice = min(point_scores, key=point_scores.get)
        quantile_choice = min(quantile_scores, key=quantile_scores.get)
        choices.append({
            "refit": k, "point_quantile": point_choice, "point_floor": levels[point_choice],
            "quantile_quantile": quantile_choice, "quantile_floor": levels[quantile_choice],
            "validation_mae": {str(q): float(v) for q, v in point_scores.items()},
            "validation_crps": {str(q): float(v) for q, v in quantile_scores.items()},
        })
        validation = datasets["validation"]
        val_parts["origin"].append(validation.delivery_unix_seconds[np.asarray(validation.origin_indices)])
        val_parts["actual"].append(validation_actual)
        val_parts["point"].append(np.stack([_floor(p["point"], levels[point_choice]) for p in predictions["validation"]]))
        val_parts["quantile"].append(np.stack([np.sort(_floor(p["quantile"], levels[quantile_choice]), axis=-1)
                                               for p in predictions["validation"]]))
        val_parts["refit"].append(np.full(len(validation_actual), k))
        test = datasets["test"]
        test_parts["origin"].append(test.delivery_unix_seconds[np.asarray(test.origin_indices)])
        test_parts["actual"].append(predictions["test"][0]["actual"])
        test_parts["naive"].append(predictions["test"][0]["naive"])
        test_parts["point_unfloored"].append(np.stack([p["point"] for p in predictions["test"]]))
        test_parts["quantile_unfloored"].append(np.stack([np.sort(p["quantile"], axis=-1) for p in predictions["test"]]))
        test_parts["point"].append(np.stack([_floor(p["point"], levels[point_choice]) for p in predictions["test"]]))
        test_parts["quantile"].append(np.stack([np.sort(_floor(p["quantile"], levels[quantile_choice]), axis=-1)
                                                for p in predictions["test"]]))
        print(f"{name} {region} refit {k}: point floor q={point_choice} ({levels[point_choice]}), "
              f"quantile floor q={quantile_choice} ({levels[quantile_choice]})", flush=True)
    return {
        "point": np.concatenate(test_parts["point"], axis=1),
        "quantile": np.concatenate(test_parts["quantile"], axis=1),
        "point_unfloored": np.concatenate(test_parts["point_unfloored"], axis=1),
        "quantile_unfloored": np.concatenate(test_parts["quantile_unfloored"], axis=1),
        "actual": np.concatenate(test_parts["actual"]),
        "naive": np.concatenate(test_parts["naive"]),
        "origin_unix": np.concatenate(test_parts["origin"]),
        "choices": choices,
        "val_point": np.concatenate(val_parts["point"], axis=1),
        "val_quantile": np.concatenate(val_parts["quantile"], axis=1),
        "val_actual": np.concatenate(val_parts["actual"]),
        "val_origin_unix": np.concatenate(val_parts["origin"]),
        "val_refit": np.concatenate(val_parts["refit"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="rolling configuration name, e.g. pd_calibrator")
    parser.add_argument("--region", required=True, choices=REGIONS)
    parser.add_argument("--static-npz-dir", required=True, type=Path, help="for the origin check")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--metrics-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="best_model.pt",
                        help="checkpoint file of each refit (best_model.pt selects on total validation loss, "
                             "best_mae_model.pt on validation MAE)")
    args = parser.parse_args()
    result = run(args.name, args.region, args.device, args.checkpoint)
    with np.load(args.static_npz_dir / f"{args.region}.npz") as archive:
        if not np.array_equal(archive["origin_unix"], result["origin_unix"]):
            raise ValueError("stitched origins do not match the static test split")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / f"{args.region}.npz", **{k: v for k, v in result.items() if k != "choices"})
    for index, seed in enumerate(SEEDS):
        metrics = point_metrics(result["point"][index], result["actual"]) | probabilistic_metrics(
            result["quantile"][index], result["actual"], list(LEVELS)
        )
        path = args.metrics_dir / f"seed{seed}" / f"{args.region}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        unfloored = point_metrics(result["point_unfloored"][index], result["actual"]) | probabilistic_metrics(
            result["quantile_unfloored"][index], result["actual"], list(LEVELS)
        )
        path.write_text(json.dumps({"region": args.region, "floor": args.name, "checkpoint": args.checkpoint,
                                    "test": {"model": metrics}, "unfloored": {"model": unfloored}}, indent=2))
    (args.metrics_dir / f"choices_{args.region}.json").write_text(json.dumps(result["choices"], indent=2))
    print(f"{args.region}: test MAE {np.mean([np.abs(p - result['actual']).mean() for p in result['point']]):.2f}", flush=True)


if __name__ == "__main__":
    main()
