"""Same-information, price-space quantile controls for the exploration suite.

These are AEMO adaptations, not reproductions of published dataset scores.
Neural controls retain asinh history but optimize standardized raw-price
pinball, exactly the proposed model's quantile target. All known feature
blocks (including scarcity shortfalls) reach every control.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import general_baselines
from .baselines import SequenceWindows, QUANTILES
from .datasets import build_region_datasets, load_forecast_config
from .gbdt_baseline import _features
from .train import DeviceBatches, set_seed


class FairWindows(SequenceWindows):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if "quantile_exogenous" in sample:
            parts = [sample[k] for k in ("ctf_exogenous", "quantile_exogenous") if k in sample]
            sample["ctf_exogenous"] = torch.cat(parts, dim=-1)
        return sample


def fair_tree_features(dataset):
    shared, hourly, target = _features(dataset, True)
    origins = dataset.delivery_unix_seconds[np.asarray(dataset.origin_indices)]
    profiles = [shared]
    # Both neural heads and trees may use the complete known 24-hour profile.
    if dataset.future_exogenous_by_origin:
        profiles.append(np.stack([dataset.future_exogenous_by_origin[int(t)] for t in origins]).reshape(len(origins), -1))
    if dataset.quantile_exogenous_by_origin:
        q = np.stack([dataset.quantile_exogenous_by_origin[int(t)] for t in origins])
        hourly = np.concatenate([hourly, q], axis=-1)
        profiles.append(q.reshape(len(origins), -1))
    return np.concatenate(profiles, axis=-1), hourly, target


def run_tree(datasets, seed, device, output):
    import xgboost
    features = {s: fair_tree_features(d) for s, d in datasets.items()}
    train, validation = features["train"], features["validation"]
    predictions = {s: np.zeros((*v[2].shape, 5), dtype=np.float64) for s, v in features.items() if s != "train"}
    rounds = []
    for h in range(train[2].shape[1]):
        model = xgboost.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=np.asarray(QUANTILES),
            tree_method="hist", device=str(device), n_estimators=2000,
            learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, early_stopping_rounds=50, random_state=seed,
            n_jobs=2,
        )
        model.fit(np.hstack([train[0], train[1][:, h]]), train[2][:, h],
                  eval_set=[(np.hstack([validation[0], validation[1][:, h]]), validation[2][:, h])], verbose=False)
        model.save_model(output / f"horizon_{h:02d}.ubj")
        rounds.append(int(model.best_iteration) + 1)
        for split in predictions:
            f = features[split]
            predictions[split][:, h] = np.sort(model.predict(np.hstack([f[0], f[1][:, h]])), axis=-1)
        print(f"horizon={h + 1} rounds={rounds[-1]}", flush=True)
    return {s: {"point": q[..., 2], "quantile": q} for s, q in predictions.items()}, {"rounds": rounds, "xgboost_version": xgboost.__version__}


def run_neural(kind, datasets, seed, device, config, output):
    windows = {s: FairWindows(d) for s, d in datasets.items()}
    loaders = {s: DeviceBatches(w, 256 if s == "train" else 1024, s == "train", device,
                               torch.Generator().manual_seed(seed) if s == "train" else None)
               for s, w in windows.items()}
    example = next(iter(loaders["validation"]))
    if kind.startswith("gatets"):
        from .gatets.model import build_gatets
        model = build_gatets(kind, example)
    else:
        model = general_baselines.build(kind, example)
    model.to(device)
    lr = general_baselines.LEARNING_RATES.get(kind, 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_epoch, history = float("inf"), -1, []
    levels = torch.tensor(QUANTILES, device=device)
    def loss(batch, *, validation=False):
        q = model(batch)
        if validation:
            q = torch.sort(q, dim=-1).values
        error = batch["target_quantile"] - q
        return torch.maximum(levels * error, (levels - 1) * error).mean()
    for epoch in range(config["training"]["epochs"]):
        model.train()
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            objective = loss(batch)
            if not torch.isfinite(objective):
                raise FloatingPointError(f"Non-finite {kind} training loss")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        total, count = 0., 0
        with torch.no_grad():
            for batch in loaders["validation"]:
                n = len(batch["history_values"])
                total += loss(batch, validation=True).item() * n
                count += n
        value = total / count
        if not np.isfinite(value):
            raise FloatingPointError(f"Non-finite {kind} validation loss")
        history.append(value)
        if value < best:
            best, best_epoch = value, epoch
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "seed": seed,
                        "kind": kind, "config": config}, output / "best_model.pt")
        print(f"epoch={epoch+1} validation_pinball={value:.6f} best_epoch={best_epoch+1}", flush=True)
    if best_epoch < 0:
        raise RuntimeError("Training produced no valid checkpoint")
    model.load_state_dict(torch.load(output / "best_model.pt", map_location=device, weights_only=False)["model_state"])
    model.eval()
    predictions = {}
    with torch.no_grad():
        for split in ("validation", "test"):
            parts = [datasets[split].denormalize_quantiles(model(b)).double().cpu().numpy() for b in loaders[split]]
            q = np.sort(np.concatenate(parts), axis=-1)
            predictions[split] = {"point": q[..., 2], "quantile": q}
    return predictions, {"best_epoch": best_epoch, "validation_pinball": history, "learning_rate": lr,
                         "provenance": getattr(model, "provenance", None),
                         "parameters": sum(p.numel() for p in model.parameters())}


def run(kind, config_path, region, seed, output):
    if kind == "event_controls":
        from .exploration_events import run as run_events
        return run_events(config_path, region, seed, output)
    config = load_forecast_config(config_path)
    if kind not in {"xgboost", "linear", "dlinear", "patchtst", "gatets", "gatets_adapter"}:
        raise ValueError(f"Unsupported price-quantile control: {kind}")
    if config["data"].get("quantile_target") != "price":
        raise ValueError("Exploration baselines require quantile_target=price")
    torch.set_num_threads(2)
    set_seed(seed)
    device = torch.device(config["training"]["device"])
    start = time.monotonic()
    datasets = build_region_datasets(config_path, region)
    output.mkdir(parents=True, exist_ok=True)
    if kind == "xgboost":
        predictions, metadata = run_tree(datasets, seed, device, output)
    else:
        predictions, metadata = run_neural(kind, datasets, seed, device, config, output)
    for split, prediction in predictions.items():
        dataset = datasets[split]
        origins = np.asarray(dataset.origin_indices)
        actual = np.stack([dataset.target_values_raw[i:i + dataset.output_hours] for i in origins])
        if not all(np.isfinite(v).all() for v in prediction.values()):
            raise FloatingPointError("Non-finite baseline forecasts")
        np.savez_compressed(output / f"{split}.npz", **prediction, actual=actual,
                            origin_unix=dataset.delivery_unix_seconds[origins], quantile_levels=QUANTILES,
                            seed=seed, region=region)
    (output / "metrics.json").write_text(json.dumps({"kind": kind, "seed": seed, "region": region,
        "seconds": time.monotonic()-start, "config": str(config_path), "configuration": config,
        "target": "raw_price" if kind == "xgboost" else "standardized_raw_price",
        "information_set": "price_history_only" if kind == "gatets" else "all_known_inputs_including_shortfalls_and_24h_profiles",
        "checkpoint_selection": "unweighted_validation_pinball", **metadata}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", required=True)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--region", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    run(a.kind, a.config, a.region, a.seed, a.output)


if __name__ == "__main__":
    main()
