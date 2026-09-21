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
        quantiles=model_config["quantiles"],
        num_frequencies=model_config["num_frequencies"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        freq_cutoff=model_config["freq_cutoff"],
        num_atoms=model_config["num_atoms"],
        sigma_base=model_config["sigma_base"],
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


def move_inputs(batch: Mapping, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch["history_values"].to(device),
        batch["future_calendar"].to(device),
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
        history, calendar, target = move_inputs(batch, device)
        optimizer.zero_grad(set_to_none=True)
        losses = criterion(model(history, calendar), history, target)
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
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    loss_sums = {name: 0.0 for name in LOSS_NAMES}
    sample_count = 0
    absolute_error = 0.0
    squared_error = 0.0
    target_count = 0
    interval_totals = {"80": [0.0, 0.0], "90": [0.0, 0.0]}
    quantiles = list(model.quantiles)
    interval_indices = {
        "80": (quantiles.index(0.10), quantiles.index(0.90)),
        "90": (quantiles.index(0.05), quantiles.index(0.95)),
    }
    with torch.no_grad():
        for batch in loader:
            history, calendar, target = move_inputs(batch, device)
            outputs = model(history, calendar)
            losses = criterion(outputs, history, target)
            batch_size = history.shape[0]
            sample_count += batch_size
            for name in LOSS_NAMES:
                loss_sums[name] += losses[name].item() * batch_size
            actual = batch["target_price_raw"].to(device)
            target_location = batch["target_location"].to(device).view(-1, 1, 1)
            target_scale = batch["target_scale"].to(device).view(-1, 1, 1)
            point = outputs["point_forecast"] * target_scale + target_location
            quantile = outputs["quantile_forecast"] * target_scale + target_location
            error = point - actual
            absolute_error += error.abs().sum().item()
            squared_error += error.square().sum().item()
            target_count += actual.numel()
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
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "region": region,
            "best_epoch": epoch,
            "best_validation_loss": validation_loss,
            "seed": config["training"]["seed"],
            "model_config": config["model"],
            "loss_config": config["loss"],
            "training_config": config["training"],
            "forecast_protocol": config["forecast_protocol"],
            "normalization_config": config["normalization"],
            "history_feature_names": dataset.history_feature_names,
            "calendar_feature_names": dataset.calendar_feature_names,
            "history_mean": dataset.history_standardizer.mean.tolist(),
            "history_std": dataset.history_standardizer.std.tolist(),
        },
        path,
    )


def save_results(path: Path, results: Mapping) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2, sort_keys=True)


def run_training(config_path: Path | str, region: str) -> Path:
    config = load_forecast_config(config_path)
    training_config = config["training"]
    set_seed(training_config["seed"])
    device = resolve_device(training_config["device"])
    datasets = build_region_datasets(config_path, region)
    loaders = build_loaders(datasets, training_config)
    model = build_model(config).to(device)
    criterion = build_loss(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    initialization_loader = DataLoader(
        datasets["train"],
        batch_size=training_config["batch_size"],
        shuffle=False,
    )
    initialization_batch = next(iter(initialization_loader))
    model.initialize_atoms(initialization_batch["history_values"].to(device))
    output_dir = Path(config["project_root"]) / training_config["output_directory"] / region
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    history_path = output_dir / "training_history.csv"
    history = []
    best_epoch = -1
    best_validation_loss = float("inf")
    for epoch in range(training_config["epochs"]):
        model.set_epoch(epoch)
        train_losses = train_epoch(model, criterion, loaders["train"], optimizer, device)
        validation = evaluate(model, criterion, loaders["validation"], device)
        history.append(
            {
                "epoch": epoch,
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
        validation_loss = validation["losses"]["total"]
        if validation_loss < best_validation_loss:
            best_epoch = epoch
            best_validation_loss = validation_loss
            save_checkpoint(checkpoint_path, model, config, datasets["train"], region, epoch, validation_loss)
        print(
            f"epoch={epoch + 1}/{training_config['epochs']} "
            f"train={train_losses['total']:.6f} "
            f"validation={validation_loss:.6f} "
            f"best_epoch={best_epoch + 1}"
        )
    if best_epoch < 0:
        raise RuntimeError("Training completed without a finite validation checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.set_epoch(checkpoint["best_epoch"])
    validation = evaluate(model, criterion, loaders["validation"], device)
    test = evaluate(model, criterion, loaders["test"], device)
    results = {
        "region": region,
        "seed": training_config["seed"],
        "normalization": config["normalization"],
        "best_epoch": checkpoint["best_epoch"],
        "best_validation_loss": checkpoint["best_validation_loss"],
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "validation": validation,
        "test": test,
        "environment": environment_fingerprint(),
        "artifacts": {"checkpoint": str(checkpoint_path), "history": str(history_path)},
    }
    save_results(output_dir / "metrics.json", results)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the dual-field linear AEMO forecaster")
    parser.add_argument("--config", default="configs/aemo_forecast_static.yaml")
    parser.add_argument("--region", required=True, choices=("NSW1", "QLD1", "TAS1"))
    args = parser.parse_args()
    output_dir = run_training(args.config, args.region)
    print(f"results={output_dir}")


if __name__ == "__main__":
    main()
