import argparse
import json
import math
import random
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
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


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be between zero and one")
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if name in self.shadow:
                    self.shadow[name].lerp_(value.detach(), 1.0 - self.decay)

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        state = model.state_dict()
        backup = {
            name: state[name].detach().clone()
            for name in self.shadow
        }
        with torch.no_grad():
            for name, value in self.shadow.items():
                state[name].copy_(value)
        try:
            yield
        finally:
            with torch.no_grad():
                state = model.state_dict()
                for name, value in backup.items():
                    state[name].copy_(value)


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


def quantile_target(batch: Mapping, device: torch.device) -> torch.Tensor | None:
    if "target_quantile" not in batch:
        return None
    return batch["target_quantile"].to(device)


def average_losses(loss_sums: Mapping[str, float], sample_count: int) -> Dict[str, float]:
    return {name: loss_sums[name] / sample_count for name in LOSS_NAMES}


def train_epoch(
    model: DualFieldLinearForecaster,
    criterion: DualFieldForecastLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None = None,
    ema: ExponentialMovingAverage | None = None,
) -> Dict[str, float]:
    model.train()
    loss_sums = {name: 0.0 for name in LOSS_NAMES}
    gradient_norm_sum = 0.0
    sample_count = 0
    for batch in loader:
        history, calendar, exogenous, target = move_inputs(batch, device)
        optimizer.zero_grad(set_to_none=True)
        losses = criterion(
            model(history, calendar, exogenous),
            history,
            target,
            quantile_target(batch, device),
        )
        losses["total"].backward()
        if gradient_clip_norm is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ],
                gradient_clip_norm,
            )
            gradient_norm_sum += float(gradient_norm) * history.shape[0]
        optimizer.step()
        if ema is not None:
            ema.update(model)
        batch_size = history.shape[0]
        sample_count += batch_size
        for name in LOSS_NAMES:
            loss_sums[name] += losses[name].item() * batch_size
    averaged = average_losses(loss_sums, sample_count)
    if gradient_clip_norm is not None:
        averaged["gradient_norm"] = gradient_norm_sum / sample_count
    return averaged


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
    quantiles = list(model.quantiles)
    interval_indices = {
        "80": (quantiles.index(0.10), quantiles.index(0.90)),
        "90": (quantiles.index(0.05), quantiles.index(0.95)),
    }
    with torch.no_grad():
        for batch in loader:
            history, calendar, exogenous, target = move_inputs(batch, device)
            outputs = model(history, calendar, exogenous)
            losses = criterion(
                outputs, history, target, quantile_target(batch, device)
            )
            batch_size = history.shape[0]
            sample_count += batch_size
            for name in LOSS_NAMES:
                loss_sums[name] += losses[name].item() * batch_size
            actual = batch["target_price_raw"].to(device)
            point = dataset.denormalize_target(outputs["point_forecast"])
            quantile = dataset.denormalize_quantiles(outputs["quantile_forecast"])
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
            "price_transform": asdict(dataset.price_transform),
            "quantile_target_mean": (
                float(dataset.quantile_standardizer.mean)
                if dataset.quantile_standardizer is not None
                else None
            ),
            "quantile_target_std": (
                float(dataset.quantile_standardizer.std)
                if dataset.quantile_standardizer is not None
                else None
            ),
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


def _fine_tuning_group_name(parameter_name: str) -> str | None:
    if parameter_name.startswith("dgf_forecast_head.exogenous_adapter."):
        return "adapter"
    if parameter_name.startswith(
        (
            "dgf_forecast_head.future_fusion.",
            "dgf_forecast_head.point_output.",
            "dgf_forecast_head.quantile_output.",
        )
    ):
        return "decoder"
    if parameter_name.startswith("dgf_forecast_head."):
        return "temporal"
    if parameter_name.startswith("fusion_gate."):
        return "fusion"
    return None


def build_optimizer(
    model: DualFieldLinearForecaster,
    training_config: Mapping,
) -> tuple[torch.optim.Optimizer, list[dict]]:
    fine_tuning = training_config.get("fine_tuning", {})
    optimizer_config = training_config.get("optimizer", {})
    betas = tuple(optimizer_config.get("betas", (0.9, 0.999)))
    epsilon = float(optimizer_config.get("eps", 1e-8))
    if not fine_tuning.get("enabled", False):
        parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
            betas=betas,
            eps=epsilon,
        )
        return optimizer, []

    configured_groups = fine_tuning["parameter_groups"]
    grouped_parameters = {name: [] for name in configured_groups}
    unmatched_trainable = []
    for parameter_name, parameter in model.named_parameters():
        group_name = _fine_tuning_group_name(parameter_name)
        parameter.requires_grad = False
        if group_name in grouped_parameters:
            grouped_parameters[group_name].append(parameter)
        elif group_name is not None:
            unmatched_trainable.append(parameter_name)
    if unmatched_trainable:
        raise RuntimeError(
            "Fine-tuning parameters have no configured group: "
            f"{unmatched_trainable}"
        )

    optimizer_groups = []
    group_metadata = []
    for group_name, group_config in configured_groups.items():
        parameters = grouped_parameters[group_name]
        if not parameters:
            raise RuntimeError(
                f"Fine-tuning group {group_name!r} has no parameters"
            )
        start_epoch = int(group_config["start_epoch"])
        max_lr = float(group_config["learning_rate"])
        weight_decay = float(
            group_config.get("weight_decay", training_config["weight_decay"])
        )
        for parameter in parameters:
            parameter.requires_grad = start_epoch == 0
        optimizer_groups.append(
            {
                "params": parameters,
                "lr": 0.0,
                "weight_decay": weight_decay,
                "group_name": group_name,
            }
        )
        group_metadata.append(
            {
                "name": group_name,
                "parameters": parameters,
                "start_epoch": start_epoch,
                "max_lr": max_lr,
                "parameter_count": sum(
                    parameter.numel() for parameter in parameters
                ),
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=betas,
        eps=epsilon,
    )
    return optimizer, group_metadata


def _scheduled_learning_rate(
    epoch: int,
    start_epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    max_lr: float,
    min_lr_ratio: float,
) -> float:
    if epoch < start_epoch:
        return 0.0
    local_epoch = epoch - start_epoch
    if warmup_epochs > 0 and local_epoch < warmup_epochs:
        return max_lr * (local_epoch + 1) / warmup_epochs
    decay_epochs = max(total_epochs - start_epoch - warmup_epochs - 1, 1)
    progress = min(max((local_epoch - warmup_epochs) / decay_epochs, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def configure_fine_tuning_epoch(
    optimizer: torch.optim.Optimizer,
    group_metadata: list[dict],
    training_config: Mapping,
    epoch: int,
) -> tuple[str, Dict[str, float], int]:
    if not group_metadata:
        schedule = training_config.get("schedule")
        learning_rate = float(training_config["learning_rate"])
        if schedule is not None:
            learning_rate = _scheduled_learning_rate(
                epoch=epoch,
                start_epoch=0,
                total_epochs=int(training_config["epochs"]),
                warmup_epochs=int(schedule.get("warmup_epochs", 0)),
                max_lr=learning_rate,
                min_lr_ratio=float(schedule.get("min_lr_ratio", 1.0)),
            )
            optimizer.param_groups[0]["lr"] = learning_rate
        return (
            "training",
            {"default": learning_rate},
            sum(
                parameter.numel()
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.requires_grad
            ),
        )
    schedule = training_config["fine_tuning"]["schedule"]
    total_epochs = int(training_config["epochs"])
    warmup_epochs = int(schedule["warmup_epochs"])
    min_lr_ratio = float(schedule["min_lr_ratio"])
    learning_rates = {}
    active_names = []
    trainable_count = 0
    for optimizer_group, metadata in zip(
        optimizer.param_groups, group_metadata
    ):
        active = epoch >= metadata["start_epoch"]
        for parameter in metadata["parameters"]:
            parameter.requires_grad = active
        learning_rate = _scheduled_learning_rate(
            epoch=epoch,
            start_epoch=metadata["start_epoch"],
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            max_lr=metadata["max_lr"],
            min_lr_ratio=min_lr_ratio,
        )
        optimizer_group["lr"] = learning_rate
        learning_rates[metadata["name"]] = learning_rate
        if active:
            active_names.append(metadata["name"])
            trainable_count += metadata["parameter_count"]
    return "+".join(active_names), learning_rates, trainable_count


def validation_selection_values(
    validation: Mapping,
    baseline_metrics: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    values = {
        "total": validation["losses"]["total"],
        "mae": validation["metrics"]["mae_aud_per_mwh"],
        "rmse": validation["metrics"]["rmse_aud_per_mwh"],
    }
    if baseline_metrics is not None:
        values["composite"] = 0.5 * (
            values["mae"] / baseline_metrics["mae"]
            + values["rmse"] / baseline_metrics["rmse"]
        )
    return values


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
    optimizer, fine_tuning_groups = build_optimizer(model, training_config)
    gradient_clip_norm = training_config.get("gradient_clip_norm")
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
        if gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
    ema_decay = training_config.get("ema_decay")
    ema = (
        ExponentialMovingAverage(model, float(ema_decay))
        if ema_decay is not None
        else None
    )
    output_dir = Path(config["project_root"]) / training_config["output_directory"] / region
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_names = tuple(
        training_config.get(
            "checkpoint_metrics", ("total", "mae", "rmse")
        )
    )
    primary_selection = training_config.get("primary_selection", "total")
    if primary_selection not in selection_names:
        raise ValueError("primary_selection must be in checkpoint_metrics")
    checkpoint_paths = {
        name: (
            output_dir / "best_model.pt"
            if name == primary_selection
            else output_dir / f"best_{name}_model.pt"
        )
        for name in selection_names
    }
    history_path = output_dir / "training_history.csv"
    history = []
    best_epochs = {name: -1 for name in checkpoint_paths}
    best_values = {name: float("inf") for name in checkpoint_paths}
    baseline_metrics = None
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
        baseline_metrics = {
            "mae": initial_validation["metrics"]["mae_aud_per_mwh"],
            "rmse": initial_validation["metrics"]["rmse_aud_per_mwh"],
        }
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
        initial_values = validation_selection_values(
            initial_validation, baseline_metrics
        )
        for selection_name in selection_names:
            selection_value = initial_values[selection_name]
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
            f"rmse={initial_values['rmse']:.6f} "
            f"composite={initial_values.get('composite', float('nan')):.6f}"
        )
    if "composite" in selection_names and baseline_metrics is None:
        raise ValueError("Composite checkpoint selection requires a warm start")

    early_stopping = training_config.get("early_stopping", {})
    early_stopping_enabled = bool(early_stopping.get("enabled", False))
    early_monitor = early_stopping.get("monitor", primary_selection)
    if early_monitor not in selection_names:
        raise ValueError("Early-stopping monitor must be checkpointed")
    minimum_epochs = int(early_stopping.get("minimum_epochs", 0))
    patience = int(early_stopping.get("patience", training_config["epochs"]))
    relative_min_delta = float(
        early_stopping.get("relative_min_delta", 0.0)
    )
    if minimum_epochs < 0 or patience <= 0 or relative_min_delta < 0:
        raise ValueError("Invalid early-stopping configuration")
    early_best = best_values.get(early_monitor, float("inf"))
    early_wait = 0
    stopped_early = False
    completed_epochs = 0
    for epoch in range(training_config["epochs"]):
        model.set_epoch(
            fixed_model_epoch if fixed_model_epoch is not None else epoch
        )
        stage, learning_rates, trainable_parameter_count = (
            configure_fine_tuning_epoch(
                optimizer,
                fine_tuning_groups,
                training_config,
                epoch,
            )
        )
        train_losses = train_epoch(
            model,
            criterion,
            loaders["train"],
            optimizer,
            device,
            gradient_clip_norm=gradient_clip_norm,
            ema=ema,
        )
        evaluation_context = (
            ema.average_parameters(model) if ema is not None else nullcontext()
        )
        with evaluation_context:
            validation = evaluate(
                model,
                criterion,
                loaders["validation"],
                datasets["validation"],
                device,
            )
            selection_values = validation_selection_values(
                validation, baseline_metrics
            )
            for selection_name in selection_names:
                selection_value = selection_values[selection_name]
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

        current_early_value = selection_values[early_monitor]
        significant_threshold = early_best * (1.0 - relative_min_delta)
        if current_early_value < significant_threshold:
            early_best = current_early_value
            early_wait = 0
        elif epoch + 1 < minimum_epochs:
            early_wait = 0
        else:
            early_wait += 1
        completed_epochs = epoch + 1
        history.append(
            {
                "epoch": epoch,
                "stage": stage,
                "trainable_parameter_count": trainable_parameter_count,
                "early_stopping_wait": early_wait,
                **{
                    f"learning_rate_{name}": value
                    for name, value in learning_rates.items()
                },
                **{f"train_{key}": value for key, value in train_losses.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation["losses"].items()
                },
                **{
                    f"validation_{key}": value
                    for key, value in validation["metrics"].items()
                },
                **{
                    f"validation_selection_{key}": value
                    for key, value in selection_values.items()
                },
            }
        )
        pd.DataFrame(history).to_csv(history_path, index=False)
        learning_rate_text = ",".join(
            f"{name}:{value:.2e}"
            for name, value in learning_rates.items()
        )
        print(
            f"epoch={epoch + 1}/{training_config['epochs']} "
            f"stage={stage} "
            f"lr={learning_rate_text} "
            f"train={train_losses['total']:.6f} "
            f"validation={selection_values['total']:.6f} "
            f"mae={selection_values['mae']:.6f} "
            f"rmse={selection_values['rmse']:.6f} "
            f"composite={selection_values.get('composite', float('nan')):.6f} "
            f"best_{primary_selection}_epoch="
            f"{best_epochs[primary_selection] + 1} "
            f"early_stop_wait={early_wait}"
        )
        if (
            early_stopping_enabled
            and completed_epochs >= minimum_epochs
            and early_wait >= patience
        ):
            stopped_early = True
            print(
                f"early_stopping epoch={completed_epochs} "
                f"monitor={early_monitor} best={early_best:.6f}"
            )
            break
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
    primary = selection_results[primary_selection]
    results = {
        "region": region,
        "seed": training_config["seed"],
        "primary_selection": primary_selection,
        "best_epoch": primary["best_epoch"],
        "best_validation_loss": primary["validation"]["losses"]["total"],
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "validation": primary["validation"],
        "test": primary["test"],
        "selection_results": selection_results,
        "warm_start": warm_start_metadata,
        "training_summary": {
            "completed_epochs": completed_epochs,
            "stopped_early": stopped_early,
            "early_stopping_monitor": early_monitor,
            "ema_decay": ema_decay,
            "gradient_clip_norm": gradient_clip_norm,
            "fine_tuning_groups": [
                {
                    key: value
                    for key, value in metadata.items()
                    if key != "parameters"
                }
                for metadata in fine_tuning_groups
            ],
        },
        "environment": environment_fingerprint(),
        "artifacts": {
            "checkpoint": str(checkpoint_paths[primary_selection]),
            **{
                f"{name}_checkpoint": str(path)
                for name, path in checkpoint_paths.items()
                if name != primary_selection
            },
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
