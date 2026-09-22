import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from torch.utils.data import DataLoader

from .datasets import build_region_datasets, load_forecast_config
from .losses import DualFieldForecastLoss
from .models import DualFieldLinearForecaster


LOSS_NAMES = ("total", "point", "quantile", "decomposition", "smoothness", "sparsity")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_name)


def build_model(config: Mapping) -> DualFieldLinearForecaster:
    protocol = config["forecast_protocol"]
    model_config = config["model"]
    model = DualFieldLinearForecaster(
        num_variables=model_config["num_variables"],
        input_length=protocol["input_hours"],
        forecast_horizon=protocol["output_hours"],
        calendar_dim=len(protocol["calendar_features"]),
        future_exogenous_dim=(
            config.get("future_exogenous", {}).get("dimension", 0)
            if config.get("future_exogenous", {}).get("enabled", False)
            else 0
        ),
        future_exogenous_mode=model_config.get(
            "future_exogenous_mode", "query"
        ),
        exogenous_adapter_hidden_dim=model_config.get(
            "exogenous_adapter_hidden_dim", 16
        ),
        quantiles=model_config["quantiles"],
        num_frequencies=model_config["num_frequencies"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        freq_cutoff=model_config["freq_cutoff"],
        num_atoms=model_config["num_atoms"],
        sigma_base=model_config["sigma_base"],
        fusion_mode=model_config.get("fusion_mode", "concatenate"),
        forecast_head_type=model_config.get("forecast_head_type", "linear"),
        tcn_channels=model_config.get("tcn_channels", 40),
        tcn_kernel_size=model_config.get("tcn_kernel_size", 3),
        tcn_dilations=model_config.get(
            "tcn_dilations", (1, 2, 4, 8, 16)
        ),
    )
    scale_scheduler = model.dual_field.scale_scheduler
    scale_scheduler.total_epochs = config["training"]["epochs"]
    scale_scheduler.warmup_epochs = int(
        scale_scheduler.total_epochs * model_config["scale_warmup_ratio"]
    )
    return model


def build_loss(config: Mapping) -> DualFieldForecastLoss:
    return DualFieldForecastLoss(
        quantiles=config["model"]["quantiles"],
        **config["loss"],
    )


def build_loaders(datasets: Mapping, training_config: Mapping) -> Dict[str, DataLoader]:
    seed_generator = torch.Generator().manual_seed(training_config["seed"])
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=training_config["batch_size"],
            shuffle=True,
            num_workers=training_config["num_workers"],
            generator=seed_generator,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=training_config["batch_size"],
            shuffle=False,
            num_workers=training_config["num_workers"],
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=training_config["batch_size"],
            shuffle=False,
            num_workers=training_config["num_workers"],
        ),
    }


def move_inputs(batch: Mapping, device: torch.device) -> tuple:
    return (
        batch["history_values"].to(device),
        batch["future_calendar"].to(device),
        batch["future_exogenous"].to(device) if "future_exogenous" in batch else None,
        batch["target_price"].to(device),
    )


def average_losses(loss_sums: Mapping[str, float], sample_count: int) -> Dict[str, float]:
    return {name: loss_sums[name] / sample_count for name in LOSS_NAMES}


def train_epoch(
    model: DualFieldLinearForecaster,
    criterion: DualFieldForecastLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    loss_sums = {name: 0.0 for name in LOSS_NAMES}
    sample_count = 0
    for batch in loader:
        history, calendar, exogenous, target = move_inputs(batch, device)
        optimizer.zero_grad(set_to_none=True)
        losses = criterion(model(history, calendar, exogenous), history, target)
        losses["total"].backward()
        optimizer.step()
        batch_size = history.shape[0]
        sample_count += batch_size
        for name in LOSS_NAMES:
            loss_sums[name] += losses[name].item() * batch_size
    return average_losses(loss_sums, sample_count)


def evaluate(
    model: DualFieldLinearForecaster,
    criterion: DualFieldForecastLoss,
    loader: DataLoader,
    dataset,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    loss_sums = {name: 0.0 for name in LOSS_NAMES}
    sample_count = 0
    absolute_error = 0.0
    squared_error = 0.0
    target_count = 0
    interval_totals = {"80": [0.0, 0.0], "90": [0.0, 0.0]}
    fusion_weight_sum = 0.0
    ctf_fusion_weight_sum = 0.0
    fusion_weight_square_sum = 0.0
    fusion_weight_count = 0
    price_mean = dataset.history_standardizer.mean[0]
    price_std = dataset.history_standardizer.std[0]
    quantiles = list(model.quantiles)
    interval_indices = {
        "80": (quantiles.index(0.10), quantiles.index(0.90)),
        "90": (quantiles.index(0.05), quantiles.index(0.95)),
    }
    with torch.no_grad():
        for batch in loader:
            history, calendar, exogenous, target = move_inputs(batch, device)
            outputs = model(history, calendar, exogenous)
            losses = criterion(outputs, history, target)
            batch_size = history.shape[0]
            sample_count += batch_size
            for name in LOSS_NAMES:
                loss_sums[name] += losses[name].item() * batch_size
            actual = batch["target_price_raw"].to(device)
            point = outputs["point_forecast"] * price_std + price_mean
            quantile = outputs["quantile_forecast"] * price_std + price_mean
            error = point - actual
            absolute_error += error.abs().sum().item()
            squared_error += error.square().sum().item()
            target_count += actual.numel()
            if "dgf_fusion_weight" in outputs:
                dgf_weight = outputs["dgf_fusion_weight"]
                ctf_weight = outputs["ctf_fusion_weight"]
                fusion_weight_sum += dgf_weight.sum().item()
                ctf_fusion_weight_sum += ctf_weight.sum().item()
                fusion_weight_square_sum += dgf_weight.square().sum().item()
                fusion_weight_count += dgf_weight.numel()
            for label, (lower_index, upper_index) in interval_indices.items():
                lower = quantile[..., lower_index:lower_index + 1]
                upper = quantile[..., upper_index:upper_index + 1]
                interval_totals[label][0] += ((actual >= lower) & (actual <= upper)).sum().item()
                interval_totals[label][1] += (upper - lower).sum().item()
    metrics = {
        "mae_aud_per_mwh": absolute_error / target_count,
        "rmse_aud_per_mwh": (squared_error / target_count) ** 0.5,
        "coverage_80": interval_totals["80"][0] / target_count,
        "mean_width_80_aud_per_mwh": interval_totals["80"][1] / target_count,
        "coverage_90": interval_totals["90"][0] / target_count,
        "mean_width_90_aud_per_mwh": interval_totals["90"][1] / target_count,
    }
    if fusion_weight_count:
        mean_dgf_weight = fusion_weight_sum / fusion_weight_count
        variance = max(
            fusion_weight_square_sum / fusion_weight_count
            - mean_dgf_weight ** 2,
            0.0,
        )
        metrics.update(
            {
                "mean_ctf_fusion_weight": (
                    ctf_fusion_weight_sum / fusion_weight_count
                ),
                "mean_dgf_fusion_weight": mean_dgf_weight,
                "std_dgf_fusion_weight": variance ** 0.5,
            }
        )
    return {"losses": average_losses(loss_sums, sample_count), "metrics": metrics}


def environment_fingerprint() -> Dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def save_checkpoint(
    path: Path,
    model: DualFieldLinearForecaster,
    config: Mapping,
    dataset,
    region: str,
    epoch: int,
    validation_loss: float,
    model_epoch: int | None = None,
    selection_metric: str = "total",
    selection_value: float | None = None,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "region": region,
            "best_epoch": epoch,
            "model_epoch": epoch if model_epoch is None else model_epoch,
            "best_validation_loss": validation_loss,
            "selection_metric": selection_metric,
            "selection_value": (
                validation_loss if selection_value is None else selection_value
            ),
            "seed": config["training"]["seed"],
            "model_config": config["model"],
            "loss_config": config["loss"],
            "training_config": config["training"],
            "forecast_protocol": config["forecast_protocol"],
            "future_exogenous_config": config.get("future_exogenous"),
            "history_feature_names": dataset.history_feature_names,
            "calendar_feature_names": dataset.calendar_feature_names,
            "history_mean": dataset.history_standardizer.mean.tolist(),
            "history_std": dataset.history_standardizer.std.tolist(),
            "future_exogenous_feature_names": dataset.future_exogenous_feature_names,
            "future_exogenous_mean": (
                dataset.future_exogenous_standardizer.mean.tolist()
                if dataset.future_exogenous_standardizer is not None
                else None
            ),
            "future_exogenous_std": (
                dataset.future_exogenous_standardizer.std.tolist()
                if dataset.future_exogenous_standardizer is not None
                else None
            ),
        },
        path,
    )


def save_results(path: Path, results: Mapping) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2, sort_keys=True)


def initialize_from_warm_start(
    model: DualFieldLinearForecaster,
    config: Mapping,
    region: str,
    device: torch.device,
) -> dict | None:
    warm_start = config["training"].get("warm_start", {})
    if not warm_start.get("enabled", False):
        return None
    checkpoint_path = (
        Path(config["project_root"])
        / warm_start["checkpoint_directory"]
        / region
        / "best_model.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    expected_missing = {
        name for name in model.state_dict() if ".exogenous_adapter." in name
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(
            "Warm-start checkpoint has unexpected missing keys: "
            f"{sorted(set(incompatible.missing_keys) - expected_missing)}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Warm-start checkpoint has unexpected keys: "
            f"{sorted(incompatible.unexpected_keys)}"
        )
    model_epoch = int(checkpoint["best_epoch"])
    model.set_epoch(model_epoch)
    if warm_start.get("adapter_only", False):
        for parameter in model.parameters():
            parameter.requires_grad = False
        for name, parameter in model.named_parameters():
            if ".exogenous_adapter." in name:
                parameter.requires_grad = True
    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names:
        raise RuntimeError("Warm start left no trainable parameters")
    return {
        "checkpoint": str(checkpoint_path),
        "base_best_epoch": model_epoch,
        "adapter_only": bool(warm_start.get("adapter_only", False)),
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def validation_selection_values(validation: Mapping) -> Dict[str, float]:
    return {
        "total": validation["losses"]["total"],
        "mae": validation["metrics"]["mae_aud_per_mwh"],
        "rmse": validation["metrics"]["rmse_aud_per_mwh"],
    }


def run_training(config_path: Path | str, region: str) -> Path:
    config = load_forecast_config(config_path)
    training_config = config["training"]
    set_seed(training_config["seed"])
    device = resolve_device(training_config["device"])
    datasets = build_region_datasets(config_path, region)
    loaders = build_loaders(datasets, training_config)
    model = build_model(config).to(device)
    criterion = build_loss(config).to(device)
    warm_start_metadata = initialize_from_warm_start(
        model, config, region, device
    )
    if warm_start_metadata is None:
        initialization_loader = DataLoader(
            datasets["train"],
            batch_size=training_config["batch_size"],
            shuffle=False,
        )
        initialization_batch = next(iter(initialization_loader))
        model.initialize_atoms(initialization_batch["history_values"].to(device))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    output_dir = Path(config["project_root"]) / training_config["output_directory"] / region
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {
        "total": output_dir / "best_model.pt",
        "mae": output_dir / "best_mae_model.pt",
        "rmse": output_dir / "best_rmse_model.pt",
    }
    history_path = output_dir / "training_history.csv"
    history = []
    best_epochs = {name: -1 for name in checkpoint_paths}
    best_values = {name: float("inf") for name in checkpoint_paths}
    fixed_model_epoch = (
        warm_start_metadata["base_best_epoch"]
        if warm_start_metadata is not None
        else None
    )
    if warm_start_metadata is not None:
        initial_validation = evaluate(
            model,
            criterion,
            loaders["validation"],
            datasets["validation"],
            device,
        )
        history.append(
            {
                "epoch": -1,
                "stage": "warm_start",
                **{f"train_{key}": float("nan") for key in LOSS_NAMES},
                **{
                    f"validation_{key}": value
                    for key, value in initial_validation["losses"].items()
                },
                **{
                    f"validation_{key}": value
                    for key, value in initial_validation["metrics"].items()
                },
            }
        )
        initial_values = validation_selection_values(initial_validation)
        for selection_name, selection_value in initial_values.items():
            best_values[selection_name] = selection_value
            save_checkpoint(
                checkpoint_paths[selection_name],
                model,
                config,
                datasets["train"],
                region,
                -1,
                initial_validation["losses"]["total"],
                model_epoch=fixed_model_epoch,
                selection_metric=selection_name,
                selection_value=selection_value,
            )
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(
            "warm_start "
            f"validation={initial_values['total']:.6f} "
            f"mae={initial_values['mae']:.6f} "
            f"rmse={initial_values['rmse']:.6f}"
        )
    for epoch in range(training_config["epochs"]):
        model.set_epoch(
            fixed_model_epoch if fixed_model_epoch is not None else epoch
        )
        train_losses = train_epoch(model, criterion, loaders["train"], optimizer, device)
        validation = evaluate(model, criterion, loaders["validation"], datasets["validation"], device)
        history.append(
            {
                "epoch": epoch,
                "stage": "adapter_training" if warm_start_metadata else "training",
                **{f"train_{key}": value for key, value in train_losses.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation["losses"].items()
                },
                **{
                    f"validation_{key}": value
                    for key, value in validation["metrics"].items()
                },
            }
        )
        pd.DataFrame(history).to_csv(history_path, index=False)
        selection_values = validation_selection_values(validation)
        for selection_name, selection_value in selection_values.items():
            if selection_value < best_values[selection_name]:
                best_epochs[selection_name] = epoch
                best_values[selection_name] = selection_value
                save_checkpoint(
                    checkpoint_paths[selection_name],
                    model,
                    config,
                    datasets["train"],
                    region,
                    epoch,
                    validation["losses"]["total"],
                    model_epoch=(
                        fixed_model_epoch
                        if fixed_model_epoch is not None
                        else epoch
                    ),
                    selection_metric=selection_name,
                    selection_value=selection_value,
                )
        print(
            f"epoch={epoch + 1}/{training_config['epochs']} "
            f"train={train_losses['total']:.6f} "
            f"validation={selection_values['total']:.6f} "
            f"mae={selection_values['mae']:.6f} "
            f"rmse={selection_values['rmse']:.6f} "
            f"best_total_epoch={best_epochs['total'] + 1} "
            f"best_mae_epoch={best_epochs['mae'] + 1} "
            f"best_rmse_epoch={best_epochs['rmse'] + 1}"
        )
    selection_results = {}
    for selection_name, checkpoint_path in checkpoint_paths.items():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        model.set_epoch(checkpoint["model_epoch"])
        selection_results[selection_name] = {
            "best_epoch": checkpoint["best_epoch"],
            "selection_value": checkpoint["selection_value"],
            "validation": evaluate(
                model,
                criterion,
                loaders["validation"],
                datasets["validation"],
                device,
            ),
            "test": evaluate(
                model,
                criterion,
                loaders["test"],
                datasets["test"],
                device,
            ),
            "checkpoint": str(checkpoint_path),
        }
    primary = selection_results["total"]
    results = {
        "region": region,
        "seed": training_config["seed"],
        "best_epoch": primary["best_epoch"],
        "best_validation_loss": primary["validation"]["losses"]["total"],
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "validation": primary["validation"],
        "test": primary["test"],
        "selection_results": selection_results,
        "warm_start": warm_start_metadata,
        "environment": environment_fingerprint(),
        "artifacts": {
            "checkpoint": str(checkpoint_paths["total"]),
            "mae_checkpoint": str(checkpoint_paths["mae"]),
            "rmse_checkpoint": str(checkpoint_paths["rmse"]),
            "history": str(history_path),
        },
    }
    save_results(output_dir / "metrics.json", results)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the dual-field linear AEMO forecaster")
    parser.add_argument("--config", default="configs/aemo_forecast.yaml")
    parser.add_argument("--region", required=True, choices=("NSW1", "QLD1", "TAS1"))
    args = parser.parse_args()
    output_dir = run_training(args.config, args.region)
    print(f"results={output_dir}")


if __name__ == "__main__":
    main()
