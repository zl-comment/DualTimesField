"""Minimal native training loop shared by the expansion's neural baselines.

Owns only the mechanics both native models need: device tensors built once from
the shared input blocks, seeded minibatch shuffling, per-epoch evaluation of a
declared raw-price selection metric, best-checkpoint keeping, and strict stage
isolation — training consumes train labels, checkpoint choice consumes
selection labels, and evaluation labels are never touched here.

Model-specific losses, heads and prediction paths stay in the model adapter:
the caller supplies ``compute_loss(batch)`` and ``predict_raw(batch)`` (the
latter returns raw-price predictions for the selection metric). The target
scaler comes from the training split metadata and is used for the inverse
transform, never refitted.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


def block_tensors(split: Any, device: torch.device, *, include_target: bool = False) -> dict[str, torch.Tensor]:
    """Move the seven shared blocks (and optionally the target) to device once."""
    from ..aligned_inputs import INPUT_KEYS

    tensors = {key: torch.from_numpy(np.ascontiguousarray(split.inputs[key])).to(device)
               for key in INPUT_KEYS}
    if include_target:
        tensors["target"] = torch.from_numpy(np.ascontiguousarray(split.target)).to(device)
    return tensors


def _batches(n: int, batch_size: int, generator: torch.Generator):
    order = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def _pinball(actual: np.ndarray, quantiles: np.ndarray, levels: tuple[float, ...]) -> float:
    error = actual[..., None] - np.sort(quantiles, axis=-1)
    return float(np.maximum(np.asarray(levels) * error, (np.asarray(levels) - 1.0) * error).mean())


def train_native(
    *,
    model: torch.nn.Module,
    compute_loss: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    predict_raw: Callable[[dict[str, torch.Tensor]], np.ndarray],
    train_tensors: dict[str, torch.Tensor],
    selection_tensors: dict[str, torch.Tensor],
    selection_actual: np.ndarray,
    quantile_levels: tuple[float, ...],
    selection_metric: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
    device: torch.device,
    checkpoint_path: Path,
    progress_path: Path,
    model_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if selection_metric not in {"qs5", "mae"}:
        raise ValueError("Native selection metric must be qs5 or mae")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_value, best_epoch, best_state, stale = float("inf"), -1, None, 0
    curve: list[float] = []
    n = train_tensors["target"].shape[0]
    events: list[dict[str, Any]] = [{"event": "start", "model": model_id, "seed": seed,
                                     "epoch_budget": epochs, "patience": patience,
                                     "batch_size": batch_size, "learning_rate": learning_rate,
                                     "selection_metric": selection_metric, "train_samples": n,
                                     "selection_samples": int(len(selection_actual))}]

    def log(event: dict[str, Any]) -> None:
        events.append(event)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("\n".join(json.dumps(e, allow_nan=False) for e in events) + "\n")

    started = time.monotonic()
    stop_reason = "epoch_budget_exhausted"
    for epoch in range(epochs):
        epoch_started = time.monotonic()
        model.train()
        loss_sum, seen = 0.0, 0
        for index in _batches(n, batch_size, generator):
            batch = {key: value[index] for key, value in train_tensors.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {model_id} loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(index.numel())
            loss_sum += float(loss.detach()) * count
            seen += count
        model.eval()
        with torch.no_grad():
            raw = predict_raw(selection_tensors)
        if selection_metric == "mae":
            centre = np.sort(raw, axis=-1)[..., 2] if raw.ndim == 3 else raw
            value = float(np.abs(centre - selection_actual).mean())
        else:
            value = _pinball(selection_actual, raw, quantile_levels)
        if not np.isfinite(value):
            raise FloatingPointError(f"Non-finite {model_id} selection metric")
        curve.append(value)
        if value < best_value:
            best_value, best_epoch, stale = value, epoch, 0
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        else:
            stale += 1
        log({"event": "epoch", "epoch": epoch, "train_loss": loss_sum / max(seen, 1),
             "selection_value": value, "best_epoch": best_epoch,
             "best_selection_value": best_value, "stale_epochs": stale,
             "seconds": time.monotonic() - epoch_started})
        if stale >= patience:
            stop_reason = "early_stopping"
            break
    model.load_state_dict(best_state)
    checkpoint = {"model_state": best_state, "model_id": model_id, "seed": seed,
                  "best_epoch": best_epoch, "best_selection_value": best_value,
                  "selection_metric": selection_metric, "metadata": metadata,
                  "optimizer_resume_supported": False}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    log({"event": "complete", "best_epoch": best_epoch, "best_selection_value": best_value,
         "epochs_executed": len(curve), "stop_reason": stop_reason,
         "seconds": time.monotonic() - started})
    return {"best_epoch": best_epoch, "best_selection_value": best_value,
            "selection_metric": selection_metric, "epoch_budget": epochs,
            "epochs_executed": len(curve), "stop_reason": stop_reason,
            "early_stopped": stop_reason == "early_stopping",
            "selection_curve": curve, "batch_size": batch_size,
            "learning_rate": learning_rate, "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_path": str(checkpoint_path)}
