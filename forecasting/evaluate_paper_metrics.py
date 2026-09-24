import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import build_region_datasets, load_forecast_config
from .train import build_model, move_inputs, resolve_device


REGIONS = ("NSW1", "QLD1", "TAS1")
INTERVALS = {"80": (0.10, 0.90), "90": (0.05, 0.95)}


def collect_predictions(model, dataset, device, batch_size: int) -> Dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    points, quantiles, actuals = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = move_inputs(batch, device)
            outputs = model(*inputs[:-1])
            if "target_scale" in batch:
                location = batch["target_location"].to(device).view(-1, 1, 1)
                scale = batch["target_scale"].to(device).view(-1, 1, 1)
            else:
                location = dataset.history_standardizer.mean[0]
                scale = dataset.history_standardizer.std[0]
            points.append((outputs["point_forecast"] * scale + location)[..., 0].cpu())
            quantiles.append((outputs["quantile_forecast"] * scale + location).cpu())
            actuals.append(batch["target_price_raw"][..., 0])
    point = torch.cat(points).double().numpy()
    actual = torch.cat(actuals).double().numpy()
    raw = np.asarray(dataset.target_values_raw, dtype=np.float64)
    origins = np.asarray(dataset.origin_indices)
    horizon = actual.shape[1]
    expected = np.stack([raw[origin:origin + horizon] for origin in origins])
    if not np.allclose(expected, actual):
        raise RuntimeError("Loader order does not match dataset origin indices")
    naive = np.stack([raw[origin - 24:origin - 24 + horizon] for origin in origins])
    return {
        "point": point,
        "quantile": torch.cat(quantiles).double().numpy(),
        "actual": actual,
        "naive": naive,
    }


def point_metrics(prediction: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    error = prediction - actual
    return {
        "mae": float(np.abs(error).mean()),
        "rmse_global": float(np.sqrt(np.square(error).mean())),
        # RE-Price eq. 29 defines SDE over the H horizons of one sample; RMSE is
        # reported with the same per-trajectory aggregation, then averaged.
        "rmse_window_mean": float(np.sqrt(np.square(error).mean(axis=1)).mean()),
        "rmse_horizon_mean": float(np.sqrt(np.square(error).mean(axis=0)).mean()),
        "sde_window_mean": float(error.std(axis=1, ddof=1).mean()),
        "sde_global": float(error.std(ddof=1)),
    }


def probabilistic_metrics(
    quantile: np.ndarray, actual: np.ndarray, levels: list
) -> Dict[str, float]:
    metrics = {}
    for label, (lower_level, upper_level) in INTERVALS.items():
        lower = quantile[..., levels.index(lower_level)]
        upper = quantile[..., levels.index(upper_level)]
        miscoverage = 1.0 - (upper_level - lower_level)
        width = upper - lower
        below = np.clip(lower - actual, 0.0, None)
        above = np.clip(actual - upper, 0.0, None)
        interval_score = width + 2.0 / miscoverage * (below + above)
        metrics[f"picp_{label}"] = float(((actual >= lower) & (actual <= upper)).mean())
        metrics[f"piaw_{label}"] = float(width.mean())
        metrics[f"ais_{label}"] = float(interval_score.mean())
    residual = actual[..., None] - quantile
    tau = np.asarray(levels, dtype=np.float64)
    pinball = np.maximum(tau * residual, (tau - 1.0) * residual)
    # Only five quantiles are available, so this is a coarse CRPS approximation.
    metrics["crps_quantile_approx"] = float(2.0 * pinball.mean())
    return metrics


def evaluate_checkpoint(
    config_path: Path, region: str, checkpoint_path: Path, device_name: str, batch_size: int
) -> Dict:
    config = load_forecast_config(config_path)
    device = resolve_device(device_name)
    datasets = build_region_datasets(config_path, region)
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.set_epoch(checkpoint.get("model_epoch", checkpoint["best_epoch"]))
    levels = [float(level) for level in model.quantiles]
    results = {
        "region": region,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(checkpoint["best_epoch"]),
    }
    for split in ("validation", "test"):
        predictions = collect_predictions(model, datasets[split], device, batch_size)
        results[split] = {
            "origins": int(predictions["actual"].shape[0]),
            "model": {
                **point_metrics(predictions["point"], predictions["actual"]),
                **probabilistic_metrics(predictions["quantile"], predictions["actual"], levels),
            },
            "seasonal_naive_24h": point_metrics(predictions["naive"], predictions["actual"]),
        }
    return results


def summarize(result_root: Path) -> str:
    rows = []
    for experiment_dir in sorted(path for path in result_root.iterdir() if path.is_dir()):
        per_region = {}
        for region in REGIONS:
            path = experiment_dir / f"{region}.json"
            if path.exists():
                per_region[region] = json.loads(path.read_text(encoding="utf-8"))["test"]
        if per_region:
            rows.append((experiment_dir.name, per_region))
    columns = (
        ("MAE", "mae"),
        ("RMSE global", "rmse_global"),
        ("RMSE window", "rmse_window_mean"),
        ("SDE window", "sde_window_mean"),
        ("PICP90", "picp_90"),
        ("PIAW90", "piaw_90"),
        ("AIS90", "ais_90"),
        ("CRPS~", "crps_quantile_approx"),
    )
    lines = [
        "| Experiment | Region | Origins | " + " | ".join(name for name, _ in columns) + " |",
        "|---|---|---:|" + "---:|" * len(columns),
    ]
    for name, per_region in rows:
        for region, test in per_region.items():
            values = []
            for _, key in columns:
                value = test["model"][key]
                values.append(f"{value:.2%}" if key.startswith("picp") else f"{value:.2f}")
            lines.append(f"| {name} | {region} | {test['origins']} | " + " | ".join(values) + " |")
        averages = []
        for _, key in columns:
            value = float(np.mean([test["model"][key] for test in per_region.values()]))
            averages.append(f"{value:.2%}" if key.startswith("picp") else f"{value:.2f}")
        lines.append(f"| {name} | **Mean** | - | " + " | ".join(averages) + " |")
    naive_lines = [
        "| Protocol | Region | Origins | MAE | RMSE global | RMSE window | SDE window |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    seen = set()
    for _, per_region in rows:
        for region, test in per_region.items():
            key = (region, test["origins"])
            if key in seen:
                continue
            seen.add(key)
            naive = test["seasonal_naive_24h"]
            naive_lines.append(
                f"| {'rolling hourly' if test['origins'] > 1000 else 'fixed origin'} | {region} | "
                f"{test['origins']} | {naive['mae']:.2f} | {naive['rmse_global']:.2f} | "
                f"{naive['rmse_window_mean']:.2f} | {naive['sde_window_mean']:.2f} |"
            )
    return "\n".join(lines) + "\n\nSeasonal naive (price 24 hours earlier) on the same test windows:\n\n" + "\n".join(naive_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints with RE-Price style metrics")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", required=True, type=Path)
    evaluate_parser.add_argument("--checkpoint-dir", required=True, type=Path)
    evaluate_parser.add_argument("--output-dir", required=True, type=Path)
    evaluate_parser.add_argument("--regions", nargs="+", default=list(REGIONS))
    evaluate_parser.add_argument("--device", default="cuda")
    evaluate_parser.add_argument("--batch-size", type=int, default=512)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "summarize":
        print(summarize(args.result_root), end="")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for region in args.regions:
        results = evaluate_checkpoint(
            args.config, region, args.checkpoint_dir / region / "best_model.pt", args.device, args.batch_size
        )
        path = args.output_dir / f"{region}.json"
        path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        test = results["test"]["model"]
        print(
            f"{region} test mae={test['mae']:.4f} rmse_global={test['rmse_global']:.4f} "
            f"rmse_window={test['rmse_window_mean']:.4f} crps~={test['crps_quantile_approx']:.4f}"
        )


if __name__ == "__main__":
    main()
