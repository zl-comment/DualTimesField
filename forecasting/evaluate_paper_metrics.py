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


def denormalize(values: torch.Tensor, batch: Mapping, dataset, device) -> torch.Tensor:
    if "target_scale" in batch:
        location = batch["target_location"].to(device).view(-1, 1, 1)
        scale = batch["target_scale"].to(device).view(-1, 1, 1)
        return values * scale + location
    if hasattr(dataset, "denormalize_target"):
        return dataset.denormalize_target(values)
    return values * dataset.history_standardizer.std[0] + dataset.history_standardizer.mean[0]


def collect_predictions(model, dataset, device, batch_size: int) -> Dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    points, quantiles, actuals = [], [], []
    normalized_quantiles, normalized_actuals = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = move_inputs(batch, device)
            outputs = model(*inputs[:-1])
            points.append(denormalize(outputs["point_forecast"], batch, dataset, device)[..., 0].cpu())
            if hasattr(dataset, "denormalize_quantiles"):
                quantile = dataset.denormalize_quantiles(outputs["quantile_forecast"])
            else:
                quantile = denormalize(outputs["quantile_forecast"], batch, dataset, device)
            quantiles.append(quantile.cpu())
            actuals.append(batch["target_price_raw"][..., 0])
            normalized_quantiles.append(outputs["quantile_forecast"].cpu())
            normalized_actuals.append(batch.get("target_quantile", batch["target_price"])[..., 0])
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
        "quantile_normalized": torch.cat(normalized_quantiles).double().numpy(),
        "actual_normalized": torch.cat(normalized_actuals).double().numpy(),
    }


def conformal_offsets(
    quantile: np.ndarray, actual: np.ndarray, levels: list
) -> Dict[str, Dict[str, list]]:
    # Asymmetric conformalized quantile regression (Romano et al., 2019), per horizon.
    offsets = {}
    sample_count = actual.shape[0]
    for label, (lower_level, upper_level) in INTERVALS.items():
        tail = (1.0 - (upper_level - lower_level)) / 2.0
        level = min(1.0, (1.0 - tail) * (sample_count + 1) / sample_count)
        lower_scores = quantile[..., levels.index(lower_level)] - actual
        upper_scores = actual - quantile[..., levels.index(upper_level)]
        offsets[label] = {
            "lower": np.quantile(lower_scores, level, axis=0, method="higher").tolist(),
            "upper": np.quantile(upper_scores, level, axis=0, method="higher").tolist(),
        }
    return offsets


def apply_offsets(quantile: np.ndarray, offsets: Mapping, levels: list) -> np.ndarray:
    adjusted = quantile.copy()
    for label, (lower_level, upper_level) in INTERVALS.items():
        adjusted[..., levels.index(lower_level)] -= np.asarray(offsets[label]["lower"])
        adjusted[..., levels.index(upper_level)] += np.asarray(offsets[label]["upper"])
    return np.sort(adjusted, axis=-1)


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
    config_path: Path,
    region: str,
    checkpoint_path: Path,
    device_name: str,
    batch_size: int,
    calibration_path: Path | None = None,
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
    predictions = {
        split: collect_predictions(model, datasets[split], device, batch_size)
        for split in ("validation", "test")
    }
    offsets = None
    if calibration_path is not None:
        offsets = conformal_offsets(
            predictions["validation"]["quantile_normalized"],
            predictions["validation"]["actual_normalized"],
            levels,
        )
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_path.write_text(
            json.dumps(
                {
                    "method": "asymmetric conformalized quantile regression per horizon",
                    "calibration_split": "validation",
                    "space": "normalized model output",
                    "offsets": offsets,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        results["interval_calibration"] = str(calibration_path)
    for split, split_predictions in predictions.items():
        quantile = split_predictions["quantile"]
        entry = {"origins": int(split_predictions["actual"].shape[0])}
        if offsets is not None:
            entry["model_uncalibrated"] = probabilistic_metrics(
                quantile, split_predictions["actual"], levels
            )
            adjusted = apply_offsets(split_predictions["quantile_normalized"], offsets, levels)
            quantile = datasets[split].denormalize_quantiles(torch.from_numpy(adjusted)).numpy()
        entry["model"] = {
            **point_metrics(split_predictions["point"], split_predictions["actual"]),
            **probabilistic_metrics(quantile, split_predictions["actual"], levels),
        }
        entry["seasonal_naive_24h"] = point_metrics(
            split_predictions["naive"], split_predictions["actual"]
        )
        results[split] = entry
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
    evaluate_parser.add_argument(
        "--calibration-dir",
        type=Path,
        help="Fit validation conformal interval offsets and save them under this directory",
    )
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "summarize":
        print(summarize(args.result_root), end="")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for region in args.regions:
        calibration_path = (
            args.calibration_dir / region / "interval_calibration.json"
            if args.calibration_dir is not None
            else None
        )
        results = evaluate_checkpoint(
            args.config,
            region,
            args.checkpoint_dir / region / "best_model.pt",
            args.device,
            args.batch_size,
            calibration_path,
        )
        path = args.output_dir / f"{region}.json"
        path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        test = results["test"]["model"]
        print(
            f"{region} test mae={test['mae']:.4f} rmse_global={test['rmse_global']:.4f} "
            f"rmse_window={test['rmse_window_mean']:.4f} picp90={test['picp_90']:.4f} "
            f"piaw90={test['piaw_90']:.4f} ais90={test['ais_90']:.4f} crps~={test['crps_quantile_approx']:.4f}"
        )


if __name__ == "__main__":
    main()
