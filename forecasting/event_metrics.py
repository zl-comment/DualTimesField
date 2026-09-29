"""Score spike and trough probabilities of the future event field.

For each forecast hour, a spike is an actual price above ``spike_threshold``
and a trough an actual price below ``trough_threshold``.  The event field's
probabilities (mean over seeds) are compared with two references fitted on
the training split:

* persistence: the training rate of the event given whether the same event
  occurred in the last 24 hours;
* logistic regression on the same hour-level drivers (spare capacity,
  shortfalls, net load, calendar) plus last-24-hour event indicators.

Metrics are the Brier score, ROC-AUC, PR-AUC (average precision), and the
expected calibration error over ten probability bins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader

from .datasets import build_region_datasets, load_forecast_config
from .train import build_model, move_inputs, resolve_device

REGIONS = ("NSW1", "QLD1", "TAS1")
SEED_SUFFIXES = ("", "_seed2027", "_seed2028")


def _labels(dataset, spike_threshold: float, trough_threshold: float):
    origins = np.asarray(dataset.origin_indices)
    raw = np.asarray(dataset.target_values_raw, dtype=np.float64)
    future = np.stack([raw[i:i + dataset.output_hours] for i in origins])
    past = np.stack([raw[i - 24:i] for i in origins])
    return {
        "spike": future > spike_threshold,
        "trough": future < trough_threshold,
        "past_spike": (past > spike_threshold).any(axis=1),
        "past_trough": (past < trough_threshold).any(axis=1),
        "past_max": past.max(axis=1),
        "past_min": past.min(axis=1),
    }


def _hour_features(dataset, labels: dict) -> np.ndarray:
    origins = np.asarray(dataset.origin_indices)
    unix = dataset.delivery_unix_seconds[origins]
    horizon = dataset.output_hours
    parts = [
        np.stack([dataset.future_exogenous_by_origin[int(t)] for t in unix]),
        np.stack([dataset.quantile_exogenous_by_origin[int(t)] for t in unix]),
        np.stack([dataset.ctf_exogenous_by_origin[int(t)] for t in unix]),
        np.stack([dataset.calendar_values[i:i + horizon] for i in origins]),
    ]
    past = np.stack(
        [
            labels["past_spike"], labels["past_trough"],
            np.arcsinh(labels["past_max"] / 100.0), np.arcsinh(labels["past_min"] / 100.0),
        ],
        axis=1,
    ).astype(np.float32)
    parts.append(np.repeat(past[:, None, :], horizon, axis=1))
    return np.concatenate(parts, axis=-1)


def _ece(probability: np.ndarray, outcome: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(probability, np.linspace(0, 1, bins + 1))
    index = np.clip(np.searchsorted(edges, probability, side="right") - 1, 0, bins - 1)
    error = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            error += mask.mean() * abs(probability[mask].mean() - outcome[mask].mean())
    return float(error)


def _scores(probability: np.ndarray, outcome: np.ndarray) -> dict:
    p, y = probability.ravel(), outcome.ravel().astype(int)
    return {
        "rate": float(y.mean()),
        "mean_probability": float(p.mean()),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < y.size else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if y.sum() > 0 else float("nan"),
        "ece": _ece(p, y),
    }


def _model_probabilities(config_path: Path, checkpoint_root: str, region: str, device) -> dict:
    config = load_forecast_config(config_path)
    dataset = build_region_datasets(config_path, region)["test"]
    spikes, troughs = [], []
    for suffix in SEED_SUFFIXES:
        model = build_model(config).to(device)
        checkpoint = torch.load(
            Path(f"{checkpoint_root}{suffix}") / region / "best_model.pt",
            map_location=device, weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.set_epoch(checkpoint.get("model_epoch", checkpoint["best_epoch"]))
        model.eval()
        spike, trough = [], []
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=1024):
                *inputs, _ = move_inputs(batch, device)
                outputs = model(*inputs)
                spike.append(outputs["event_probability_spike"].cpu())
                trough.append(outputs["event_probability_trough"].cpu())
        spikes.append(torch.cat(spike).double().numpy())
        troughs.append(torch.cat(trough).double().numpy())
    return {"spike": np.mean(spikes, axis=0), "trough": np.mean(troughs, axis=0)}


def report(variants: dict[str, tuple[Path, str]], output: Path, device_name: str,
           spike_threshold: float, trough_threshold: float) -> dict:
    device = resolve_device(device_name)
    reference_config = next(iter(variants.values()))[0]
    result = {"spike_threshold": spike_threshold, "trough_threshold": trough_threshold, "regions": {}}
    for region in REGIONS:
        datasets = build_region_datasets(reference_config, region)
        train_labels = _labels(datasets["train"], spike_threshold, trough_threshold)
        test_labels = _labels(datasets["test"], spike_threshold, trough_threshold)
        region_result = {}
        for event in ("spike", "trough"):
            outcome = test_labels[event]
            horizon = outcome.shape[1]
            # Persistence: training event rate conditioned on a same-type event in the last day.
            past_train = np.repeat(train_labels[f"past_{event}"][:, None], horizon, axis=1)
            rates = {flag: train_labels[event][past_train == flag].mean() for flag in (False, True)}
            past_test = np.repeat(test_labels[f"past_{event}"][:, None], horizon, axis=1)
            persistence = np.where(past_test, rates[True], rates[False])
            # Logistic regression on hour-level drivers.
            x_train = _hour_features(datasets["train"], train_labels)
            x_train = x_train.reshape(-1, x_train.shape[-1])
            x_test = _hour_features(datasets["test"], test_labels)
            x_test = x_test.reshape(-1, x_test.shape[-1])
            classifier = LogisticRegression(max_iter=2000)
            classifier.fit(x_train, train_labels[event].ravel().astype(int))
            logistic = classifier.predict_proba(x_test)[:, 1].reshape(outcome.shape)
            region_result[event] = {
                "persistence": _scores(persistence, outcome),
                "logistic_regression": _scores(logistic, outcome),
            }
        for name, (config_path, checkpoint_root) in variants.items():
            probabilities = _model_probabilities(config_path, checkpoint_root, region, device)
            for event in ("spike", "trough"):
                region_result[event][name] = _scores(probabilities[event], test_labels[event])
        result["regions"][region] = region_result
        print(json.dumps({region: {e: {k: round(v["pr_auc"], 4) for k, v in d.items()} for e, d in region_result.items()}}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", action="append", required=True,
        help="name=config_path:checkpoint_root (seeds 2027 and 2028 use <root>_seed<N>)",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--spike-threshold", type=float, default=300.0)
    parser.add_argument("--trough-threshold", type=float, default=0.0)
    args = parser.parse_args()
    variants = {}
    for item in args.variant:
        name, spec = item.split("=", 1)
        config_path, checkpoint_root = spec.split(":", 1)
        variants[name] = (Path(config_path), checkpoint_root)
    report(variants, args.output, args.device, args.spike_threshold, args.trough_threshold)


if __name__ == "__main__":
    main()
