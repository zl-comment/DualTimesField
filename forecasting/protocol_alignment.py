"""Compare local forecasts with RE-Price under alternative price treatments.

RE-Price (Chen et al., Applied Energy 426 (2026) 128712) reports window-level
RMSE almost equal to SDE and RMSE/MAE ratios of 1.20-1.45 on AEMO prices whose
test period contains spikes above 10,000 AUD/MWh.  On raw prices those ratios
are not reachable, which suggests an undisclosed spike treatment.  This module
re-scores cached forecasts under several treatments so that the comparison can
be reported with its assumptions explicit:

* ``raw``: original prices.
* ``cap<c>``: actual and forecast prices are both clipped at ``c`` AUD/MWh.
* ``drop300``: 24-hour windows with any actual price above 300 are removed.

It also finds, for each region, the cap at which the local model's window
RMSE/MAE ratio matches RE-Price's.

Usage::

    python -m forecasting.protocol_alignment collect \
        --config <config> --checkpoint-root <output_directory> \
        --output-dir <dir>
    python -m forecasting.protocol_alignment report \
        --model <dir> --baseline gbdt=<dir> --output <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REGIONS = ("NSW1", "QLD1", "TAS1")
SEED_SUFFIXES = ("", "_seed2027", "_seed2028")
CAPS = (5000, 1000, 500, 300)
# Table 2 of RE-Price: MAE, RMSE and SDE averaged over horizons.
RE_PRICE_TABLE = {
    "NSW1": {
        "RE-Price": (23.48, 34.15, 34.14),
        "GPT4TS": (28.12, 41.79, 41.72),
        "XGBoost": (38.63, 51.99, 50.61),
    },
    "QLD1": {
        "RE-Price": (25.85, 37.15, 36.89),
        "GPT4TS": (29.80, 43.40, 39.84),
        "XGBoost": (44.70, 64.05, 61.36),
    },
    "TAS1": {
        "RE-Price": (18.96, 22.81, 21.57),
        "GPT4TS": (22.88, 29.49, 27.25),
        "XGBoost": (37.56, 40.33, 38.96),
    },
}


def collect(config: Path, checkpoint_root: str, output_dir: Path, device_name: str) -> None:
    from .datasets import build_region_datasets, load_forecast_config
    from .evaluate_paper_metrics import collect_predictions
    from .train import build_model, resolve_device

    device = resolve_device(device_name)
    model_config = load_forecast_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    for region in REGIONS:
        dataset = build_region_datasets(config, region)["test"]
        points, quantiles = [], []
        for suffix in SEED_SUFFIXES:
            checkpoint_path = Path(f"{checkpoint_root}{suffix}") / region / "best_model.pt"
            model = build_model(model_config).to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            model.set_epoch(checkpoint.get("model_epoch", checkpoint["best_epoch"]))
            predictions = collect_predictions(model, dataset, device, 1024)
            points.append(predictions["point"])
            quantiles.append(predictions["quantile"])
        np.savez_compressed(
            output_dir / f"{region}.npz",
            point=np.stack(points),
            quantile=np.stack(quantiles),
            actual=predictions["actual"],
            naive=predictions["naive"],
            origin_unix=dataset.delivery_unix_seconds[np.asarray(dataset.origin_indices)],
        )
        print(f"collected region={region} seeds={len(points)}", flush=True)


def _scores(point: np.ndarray, actual: np.ndarray) -> dict:
    error = point - actual
    window_rmse = np.sqrt((error ** 2).mean(axis=1))
    window_sde = error.std(axis=1, ddof=1)
    mae = float(np.abs(error).mean())
    return {
        "mae": mae,
        "rmse_window": float(window_rmse.mean()),
        "sde_window": float(window_sde.mean()),
        "ratio": float(window_rmse.mean() / mae),
    }


def _treat(point: np.ndarray, actual: np.ndarray, treatment: str) -> tuple[np.ndarray, np.ndarray]:
    if treatment == "raw":
        return point, actual
    if treatment.startswith("cap"):
        cap = float(treatment[3:])
        return np.minimum(point, cap), np.minimum(actual, cap)
    if treatment == "drop300":
        keep = (actual <= 300).all(axis=1)
        return point[keep], actual[keep]
    raise ValueError(f"Unknown treatment {treatment}")


def _load_points(directory: Path, region: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(directory / f"{region}.npz") as archive:
        point = archive["point"]
        if point.ndim == 2:
            point = point[None]
        return point, archive["actual"], archive["origin_unix"]


def _matched_cap(point: np.ndarray, actual: np.ndarray, target_ratio: float) -> int:
    caps = np.arange(100, 5001, 25)
    ratios = [_scores(np.minimum(point, c), np.minimum(actual, c))["ratio"] for c in caps]
    return int(caps[int(np.argmin(np.abs(np.asarray(ratios) - target_ratio)))])


def report(model_dir: Path, baselines: dict[str, Path], output: Path) -> dict:
    treatments = ["raw", *(f"cap{c}" for c in CAPS), "drop300"]
    result = {"treatments": treatments, "regions": {}}
    for region in REGIONS:
        model_points, actual, origins = _load_points(model_dir, region)
        with np.load(model_dir / f"{region}.npz") as archive:
            naive = archive["naive"]
        forecasts = {"model": model_points, "seasonal_naive": naive[None]}
        for name, directory in baselines.items():
            points, base_actual, base_origins = _load_points(directory, region)
            if not (np.array_equal(base_origins, origins) and np.allclose(base_actual, actual)):
                raise ValueError(f"{name} forecasts for {region} are not aligned with the model")
            forecasts[name] = points
        region_result = {"n_windows": int(actual.shape[0]), "scores": {}}
        for name, points in forecasts.items():
            region_result["scores"][name] = {}
            for treatment in treatments:
                per_seed = [_scores(*_treat(p, actual, treatment)) for p in points]
                region_result["scores"][name][treatment] = {
                    key: float(np.mean([s[key] for s in per_seed])) for key in per_seed[0]
                } | {"mae_std": float(np.std([s["mae"] for s in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0}
        re_price = RE_PRICE_TABLE[region]["RE-Price"]
        cap = _matched_cap(model_points.mean(axis=0), actual, re_price[1] / re_price[0])
        region_result["ratio_matched_cap"] = cap
        region_result["scores_at_matched_cap"] = {
            name: float(np.mean([_scores(*_treat(p, actual, f"cap{cap}"))["mae"] for p in points]))
            for name, points in forecasts.items()
        }
        region_result["share_hours_above_300"] = float((actual > 300).mean())
        region_result["share_windows_kept_drop300"] = float((actual <= 300).all(axis=1).mean())
        region_result["re_price_table2"] = RE_PRICE_TABLE[region]
        result["regions"][region] = region_result
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--config", required=True, type=Path)
    collect_parser.add_argument("--checkpoint-root", required=True)
    collect_parser.add_argument("--output-dir", required=True, type=Path)
    collect_parser.add_argument("--device", default="cuda")
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--model", required=True, type=Path)
    report_parser.add_argument(
        "--baseline", action="append", default=[], help="name=directory of <REGION>.npz"
    )
    report_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.config, args.checkpoint_root, args.output_dir, args.device)
    else:
        baselines = dict(item.split("=", 1) for item in args.baseline)
        result = report(args.model, {k: Path(v) for k, v in baselines.items()}, args.output)
        for region, data in result["regions"].items():
            print(region, "matched cap", data["ratio_matched_cap"], data["scores_at_matched_cap"])


if __name__ == "__main__":
    main()
