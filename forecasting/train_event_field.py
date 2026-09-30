"""Train the future event field on a frozen, already-trained forecaster.

The trunk checkpoint for the same region and seed is loaded, every parameter
outside ``model.event_field`` is frozen, and only the event field is trained
with the event losses (mixture negative log-likelihood and anchoring
cross-entropy). The checkpoint with the lowest validation event loss is kept.
The point and quantile forecasts are therefore exactly the trunk's.

Configuration: the usual forecasting config with ``model.event_field: true``
and an ``event_training`` block::

    event_training:
      trunk_output_directory: outputs/forecasting/scarcity_quantile_inputs
      epochs: 40
      learning_rate: 0.001
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .datasets import build_region_datasets, load_forecast_config
from .train import build_loaders, build_loss, build_model, move_inputs, resolve_device, set_seed


def _event_loss(criterion, model, batch, device):
    *inputs, target = move_inputs(batch, device)
    outputs = model(*inputs)
    losses = criterion(outputs, inputs[0], target, None, batch["target_price_raw"].to(device))
    weighted = (
        criterion.event_nll_weight * losses["event_nll"]
        + criterion.event_anchor_weight * losses["event_anchor"]
    )
    return weighted, losses


def run(config_path: Path, region: str, seed: int | None) -> Path:
    config = load_forecast_config(config_path)
    training = config["training"]
    event_training = config["event_training"]
    trunk_directory = event_training["trunk_output_directory"]
    if seed is not None:
        training["seed"] = seed
        training["output_directory"] += f"_seed{seed}"
        trunk_directory += f"_seed{seed}"
    if training.get("torch_threads") is not None:
        torch.set_num_threads(int(training["torch_threads"]))
    set_seed(training["seed"])
    device = resolve_device(training["device"])
    datasets = build_region_datasets(config_path, region)
    loaders = build_loaders(datasets, training, device)
    model = build_model(config).to(device)
    criterion = build_loss(config).to(device)

    trunk_path = Path(config["project_root"]) / trunk_directory / region / "best_model.pt"
    trunk = torch.load(trunk_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(trunk["model_state"], strict=False)
    if unexpected or any(not key.startswith("event_field.") for key in missing):
        raise ValueError(f"Trunk checkpoint does not match: missing={missing} unexpected={unexpected}")
    model_epoch = trunk.get("model_epoch", trunk["best_epoch"])
    model.set_epoch(model_epoch)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("event_field.")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(event_training.get("learning_rate", 1e-3)),
    )

    output_dir = Path(config["project_root"]) / training["output_directory"] / region
    output_dir.mkdir(parents=True, exist_ok=True)
    best = (float("inf"), -1)
    history = []
    start = time.time()
    for epoch in range(int(event_training.get("epochs", 40))):
        # Frozen trunk stays in eval mode; only the event heads learn.
        model.eval()
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _event_loss(criterion, model, batch, device)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            totals = {"event": 0.0, "event_nll": 0.0, "event_anchor": 0.0}
            count = 0
            for batch in loaders["validation"]:
                loss, losses = _event_loss(criterion, model, batch, device)
                size = batch["history_values"].shape[0]
                totals["event"] += loss.item() * size
                totals["event_nll"] += losses["event_nll"].item() * size
                totals["event_anchor"] += losses["event_anchor"].item() * size
                count += size
        validation = {key: value / count for key, value in totals.items()}
        history.append({"epoch": epoch, **validation})
        if validation["event"] < best[0]:
            best = (validation["event"], epoch)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_epoch": epoch,
                    "model_epoch": model_epoch,
                    "trunk_checkpoint": str(trunk_path),
                    "seed": training["seed"],
                    "event_training": event_training,
                    "model_config": config["model"],
                },
                output_dir / "best_model.pt",
            )
        print(
            f"epoch={epoch + 1} validation_event={validation['event']:.6f} "
            f"nll={validation['event_nll']:.6f} anchor={validation['event_anchor']:.6f} "
            f"best_epoch={best[1] + 1}",
            flush=True,
        )
    summary = {
        "region": region,
        "seed": training["seed"],
        "trunk_checkpoint": str(trunk_path),
        "best_epoch": best[1],
        "best_validation_event_loss": best[0],
        "seconds": round(time.time() - start, 1),
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"results={output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--region", required=True, choices=("NSW1", "QLD1", "TAS1"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    run(args.config, args.region, args.seed)


if __name__ == "__main__":
    main()
