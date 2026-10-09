"""Cross-fitted XGBoost forecasts as a fixed base for the dual-field residual.

The fields are trained on the residual of the base, so the base forecasts of
the training origins must be out of sample: an XGBoost fitted on those origins
leaves residuals much smaller than on new data, and the fields would learn to
correct errors that do not occur at test time.

* Validation and test origins: the baseline XGBoost of ``forecasting.baselines``,
  fitted on the training split and early-stopped on validation, with the same
  settings and seed. Its test forecast is that baseline's.
* Training origins: split into ``folds`` contiguous blocks. Each block is
  forecast by an XGBoost fitted on the other blocks with the baseline's
  number of rounds for that horizon, leaving out the origins whose 96-hour
  windows (history and horizon) overlap the block.

Run it with the configuration the dual-field model will be trained with. The
output, ``<external_base.directory>/<region>.npz``, holds every origin of every
split (``forecast_origin_unix``, ``split``), ``point`` (N, 24) and ``quantile``
(N, 24, 5) in AUD/MWh, the rounds per horizon, and a fingerprint of the
configuration that the dataset checks when it reads the file.

    python -m forecasting.xgboost_base --config configs/aemo_forecast_xgb_base_tails.yaml --region NSW1
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from .baselines import QUANTILES, xgboost_regressor
from .datasets import build_region_datasets_from_config, external_base_fingerprint, load_forecast_config
from .evaluate_paper_metrics import point_metrics, probabilistic_metrics
from .gbdt_baseline import _features

SPLITS = ("train", "validation", "test")


def fold_blocks(origins: np.ndarray, folds: int, window: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous held-out blocks of origin positions and, for each, the positions to fit on.

    An origin ``j`` is used for fitting only if its window ``[j - history,
    j + horizon)`` cannot overlap the window of any held-out origin, that is
    if it lies at least ``window`` rows before the block's first origin or
    after its last.
    """
    if folds < 2:
        raise ValueError("folds must be at least 2")
    pairs = []
    for block in np.array_split(np.arange(len(origins)), folds):
        first, last = origins[block[0]], origins[block[-1]]
        fit = np.flatnonzero((origins <= first - window) | (origins >= last + window))
        if len(fit) == 0:
            raise ValueError("A fold has no origins left to fit on")
        pairs.append((block, fit))
    return pairs


def build(config_path: Path, region: str, seed: int, folds: int, device_name: str) -> dict:
    config = load_forecast_config(config_path)
    if not config.get("external_base", {}).get("enabled", False):
        raise ValueError(f"{config_path} has no enabled external_base block")
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    # The samples must not try to read the base that is being built.
    without_base = copy.deepcopy(config)
    without_base["external_base"]["enabled"] = False
    datasets = build_region_datasets_from_config(without_base, region)
    features = {name: _features(datasets[name], True) for name in SPLITS}
    x_train, c_train, y_train = features["train"]
    x_val, c_val, y_val = features["validation"]
    horizon = y_train.shape[1]
    train = datasets["train"]
    window = train.input_hours + train.output_hours
    pairs = fold_blocks(np.asarray(train.origin_indices), folds, window)
    forecasts = {name: np.zeros((len(features[name][0]), horizon, len(QUANTILES))) for name in SPLITS}
    rounds = []
    start = time.time()
    for h in range(horizon):
        train_x = np.hstack([x_train, c_train[:, h]])
        full = xgboost_regressor(seed, device)
        full.fit(train_x, y_train[:, h], eval_set=[(np.hstack([x_val, c_val[:, h]]), y_val[:, h])],
                 verbose=False)
        rounds.append(int(full.best_iteration) + 1)
        for name in ("validation", "test"):
            x, c, _ = features[name]
            forecasts[name][:, h] = full.predict(np.hstack([x, c[:, h]]))
        for block, fit in pairs:
            model = xgboost_regressor(seed, device, n_estimators=rounds[-1], early_stopping_rounds=None)
            model.fit(train_x[fit], y_train[fit, h], verbose=False)
            forecasts["train"][block, h] = model.predict(train_x[block])
        print(f"{region} horizon {h + 1}: {rounds[-1]} rounds, {time.time() - start:.0f} s", flush=True)
    for name in SPLITS:
        forecasts[name] = np.sort(forecasts[name], axis=-1)

    directory = Path(config["project_root"]) / config["external_base"]["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    median = QUANTILES.index(0.5)
    np.savez_compressed(
        directory / f"{region}.npz",
        forecast_origin_unix=np.concatenate([
            datasets[name].delivery_unix_seconds[np.asarray(datasets[name].origin_indices)] for name in SPLITS
        ]),
        split=np.concatenate([np.full(len(features[name][0]), name) for name in SPLITS]),
        point=np.concatenate([forecasts[name][..., median] for name in SPLITS]),
        quantile=np.concatenate([forecasts[name] for name in SPLITS]),
        quantiles=np.asarray(QUANTILES),
        rounds_per_horizon=np.asarray(rounds),
        config_fingerprint=np.asarray(external_base_fingerprint(config)),
    )
    summary = {
        "config": str(config_path),
        "region": region,
        "seed": seed,
        "folds": folds,
        "seconds": round(time.time() - start, 1),
        "rounds_per_horizon": rounds,
        # Training scores are out of fold; they should be close to the
        # validation scores, not far below them.
        **{
            name: point_metrics(forecasts[name][..., median], features[name][2])
            | probabilistic_metrics(forecasts[name], features[name][2], list(QUANTILES))
            for name in SPLITS
        },
    }
    (directory / f"{region}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--region", required=True, choices=("NSW1", "QLD1", "TAS1"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = build(args.config, args.region, args.seed, args.folds, args.device)
    print(
        f"xgboost base {args.region}: MAE train (out of fold) {summary['train']['mae']:.2f}, "
        f"validation {summary['validation']['mae']:.2f}, test {summary['test']['mae']:.2f}; "
        f"{summary['seconds']} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
