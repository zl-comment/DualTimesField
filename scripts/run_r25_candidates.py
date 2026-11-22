"""R25 selection-ceiling candidate generation (plan v4.1, 2026-11-19).

Trains/derives each candidate on folds A/B/C and writes prediction artifacts
for BOTH the selection and evaluation periods, subsampled to four origins
per day (local 00/06/12/18). No configuration contains a test split and data
loading stops at each fold's evaluation end; the closed 2023-2024 test split
is never touched. Run records keep only the git commit and a config copy.

Tree selection-period predictions are produced by reloading the boosters
saved during the single production fit and predicting at each horizon's
selected round (exact same fit, no second training pass).

chronos2 must be executed with the foundation venv:
  /root/workspace/Code/DualTimesField-foundation-baselines/.venvs/chronos2/bin/python
  HF_HOME=/root/workspace/Code/DualTimesField-foundation-baselines/weights/hf_home
  HF_HUB_OFFLINE=1 PYTHONPATH=<repo root>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.aligned_inputs import build_aligned_splits, flatten_inputs
from forecasting.aligned_baselines import fit_predict
from forecasting.aligned_scoring import (
    make_prediction_artifact, save_prediction_artifact, score_predictions,
)

QUANTILE_LEVELS = (0.05, 0.10, 0.50, 0.90, 0.95)
Q50_INDEX = QUANTILE_LEVELS.index(0.50)
ORIGIN_HOURS = (0, 6, 12, 18)

# Plan v4.1 section 2. Boundaries are contiguous; the 24h-inside-split origin
# rule provides the purge between adjacent splits.
FOLDS = {
    "A": {"train": ("2015-01-01", "2019-01-01"), "selection": ("2019-01-01", "2019-07-01"),
          "evaluation": ("2019-07-01", "2020-01-01")},
    "B": {"train": ("2015-01-01", "2021-01-01"), "selection": ("2021-01-01", "2021-07-01"),
          "evaluation": ("2021-07-01", "2022-01-01")},
    "C": {"train": ("2015-01-01", "2022-01-01"), "selection": ("2022-01-01", "2022-07-01"),
          "evaluation": ("2022-07-01", "2023-01-01")},
}

# Production tree parameters, copied verbatim from the archived run config
# outputs/forecasting/aligned_baselines_20261012/config.yaml (baseline_parameters).
TREE_PARAMS = {
    "lightgbm": {"n_estimators": 2000, "early_stopping_rounds": 50, "selection_interval": 1,
                 "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 50,
                 "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": 4,
                 "selection_metric": "pinball"},
    "xgboost": {"n_estimators": 2000, "early_stopping_rounds": 50, "selection_interval": 1,
                "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
                "colsample_bytree": 0.8, "n_jobs": 4, "selection_metric": "pinball"},
}

BASE_CONFIG = ROOT / "configs/lightgbm/aligned.yaml"
FOUNDATION = ROOT.parent / "DualTimesField-foundation-baselines"


def fold_config(fold: str) -> dict:
    config = yaml.safe_load(BASE_CONFIG.read_text())
    config["project_root"] = str(ROOT)
    boundaries = {}
    for name, (start, end) in FOLDS[fold].items():
        boundaries[name] = {"start": start, "end_exclusive": end}
    config["data"]["split_names"] = ["train", "selection", "evaluation"]
    config["data"]["split_boundaries"] = boundaries
    config["data"]["load_end_exclusive"] = FOLDS[fold]["evaluation"][1]
    if "test" in json.dumps(config["data"]["split_names"]):
        raise ValueError("test split must never appear in an R25 configuration")
    return config


def origin_hour_mask(origin_unix: np.ndarray) -> np.ndarray:
    stamps = pd.DatetimeIndex(pd.to_datetime(origin_unix, unit="s", utc=True))
    hours = stamps.tz_convert("Australia/Brisbane").hour.to_numpy()
    return np.isin(hours, ORIGIN_HOURS)


def save_period(out_dir: Path, split, quantiles, seed, fold, region, candidate, period):
    mask = origin_hour_mask(split.origin_unix)
    q = np.asarray(quantiles)[mask]
    repeated = bool(np.all(q == q[..., :1]))
    metadata = dict(split.metadata, model=candidate,
                    seed=int(seed) if seed is not None else 0,
                    source=f"r25:{git_commit()}", fold=fold, period=period,
                    prediction_space="raw_rrp",
                    output_type="degenerate_quantile" if repeated else "quantile",
                    deterministic=(seed is None))
    artifact = make_prediction_artifact(
        actual=split.actual[mask], origin_unix=split.origin_unix[mask],
        target_time_unix=split.target_time_unix[mask], metadata=metadata,
        quantiles=q, point=None, quantile_levels=QUANTILE_LEVELS)
    save_prediction_artifact(out_dir / f"{period}.npz", artifact)
    return score_predictions(artifact)


def git_commit() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                   text=True).strip()


def run_tree(kind, splits, out_dir, seed):
    train, selection, evaluation = (splits[s] for s in ("train", "selection", "evaluation"))
    model_dir = out_dir / "models"
    result = fit_predict(kind, train, selection, evaluation, seed=seed,
                         quantile_levels=QUANTILE_LEVELS, params=dict(TREE_PARAMS[kind]),
                         model_directory=model_dir)
    eval_quantiles = result.quantiles
    # Selection-period predictions from the saved boosters at their selected rounds.
    mean, std = float(train.metadata["target_mean"]), float(train.metadata["target_std"])
    x_select = flatten_inputs(selection)
    selected = result.metadata["selected_rounds"]
    horizons = train.target.shape[1]
    if kind == "lightgbm":
        import lightgbm as lgb
        columns = []
        for h in range(horizons):
            levels = [lgb.Booster(model_file=str(model_dir / f"h{h:02d}_q{i:02d}.txt")).predict(
                x_select, num_iteration=int(selected[h]), num_threads=4)
                for i in range(len(QUANTILE_LEVELS))]
            columns.append(np.column_stack(levels))
    else:
        import xgboost as xgb
        dmat = xgb.DMatrix(x_select)
        columns = [xgb.Booster(model_file=str(model_dir / f"h{h:02d}.ubj")).predict(
            dmat, iteration_range=(0, int(selected[h]))) for h in range(horizons)]
    selection_quantiles = np.stack(columns, axis=1) * std + mean
    return eval_quantiles, selection_quantiles


def run_naive(kind, splits, config, region):
    frame = pd.read_csv(ROOT.parent / "electricity-price-forecasting-research/data/multi_market_energy_reserve"
                        "/aemo_nem" / f"{region.lower()}_hourly.csv")
    price = frame["rrp_aud_per_mwh"].to_numpy(dtype=np.float64)
    delivery = pd.to_datetime(frame["delivery_start_aest"])
    utc_naive = delivery.dt.tz_convert("UTC").dt.tz_localize(None)
    unix = utc_naive.to_numpy(dtype="datetime64[s]").astype(np.int64)
    row_of = {int(t): i for i, t in enumerate(unix)}

    def quantiles_for(split):
        origins = [row_of[int(t)] for t in split.origin_unix]
        out = np.empty((len(origins), 24, 5), dtype=np.float64)
        for n, i in enumerate(origins):
            for h in range(24):
                if kind == "persistence":
                    value = price[i + h - 24]
                    out[n, h, :] = value
                else:  # seasonal naive: last 8 same-hour-of-week observations
                    lags = price[[i + h - 168 * k for k in range(1, 9)]]
                    out[n, h, :] = np.quantile(lags, QUANTILE_LEVELS)
        return out

    return (quantiles_for(splits["evaluation"]), quantiles_for(splits["selection"]))


def run_chronos2(splits, out_dir):
    from forecasting.chronos2_zeroshot import model as c2
    pipeline = c2.load_pipeline("cuda")

    def quantiles_for(split):
        meta = split.metadata
        mean, std = float(meta["target_mean"]), float(meta["target_std"])
        inputs = []
        for row in range(len(split.origin_unix)):
            past = split.inputs["history_values"][row]
            price_raw = past[:, 0].astype(np.float64) * std + mean
            past_cov = {"demand_std": past[:, 1].astype(np.float64)}
            future_cov = {}
            calendar_past = split.inputs["history_calendar"][row].astype(np.float64)
            calendar_future = split.inputs["future_calendar"][row].astype(np.float64)
            for column, name in enumerate(c2_adapter_calendar_names()):
                past_cov[name] = calendar_past[:, column]
                future_cov[name] = calendar_future[:, column]
            inputs.append({"target": price_raw, "past_covariates": past_cov,
                           "future_covariates": future_cov})
        quantiles = c2.predict_common_quantiles(
            pipeline, inputs, batch_size=32, context_length=72)
        assert quantiles.shape == (len(inputs), 24, 5), quantiles.shape
        return quantiles

    return quantiles_for(splits["evaluation"]), quantiles_for(splits["selection"])


def c2_adapter_calendar_names():
    from forecasting.chronos2_zeroshot.adapter import CALENDAR_NAMES
    return CALENDAR_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", default=["C", "B", "A"])
    parser.add_argument("--regions", nargs="+", default=["NSW1", "QLD1", "TAS1"])
    parser.add_argument("--candidates", nargs="+",
                        default=["persistence", "snaive", "lightgbm", "xgboost"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028],
                        help="seeds for tree candidates; naive/chronos2 ignore seeds")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs/forecasting/r25_ceiling_20261119")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "run_config.yaml").write_text(yaml.safe_dump(
        {"folds": args.folds, "regions": args.regions, "candidates": args.candidates,
         "seeds": args.seeds, "git_commit": git_commit()}))
    score_rows = []
    for fold in args.folds:
        config = fold_config(fold)
        (args.output_root / f"config_fold_{fold}.yaml").write_text(yaml.safe_dump(config))
        for region in args.regions:
            splits = build_aligned_splits(config, region,
                                          split_names=("train", "selection", "evaluation"))
            jobs = []
            for candidate in args.candidates:
                if candidate in ("lightgbm", "xgboost"):
                    seeds = [2026] if candidate == "xgboost" else sorted(set(args.seeds))
                    jobs.extend((candidate, seed) for seed in seeds)
                else:
                    jobs.append((candidate, None))
            for candidate, seed in jobs:
                out_dir = args.output_root / fold / region / (
                    f"{candidate}_seed{seed}" if seed else candidate)
                if (out_dir / "evaluation.npz").exists():
                    print(f"skip completed {fold} {region} {candidate} {seed}", flush=True)
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                print(f"run {fold} {region} {candidate} seed={seed} "
                      f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}", flush=True)
                if candidate in ("lightgbm", "xgboost"):
                    evaluation_q, selection_q = run_tree(candidate, splits, out_dir, seed)
                elif candidate in ("persistence", "snaive"):
                    evaluation_q, selection_q = run_naive(candidate, splits, config, region)
                elif candidate == "chronos2":
                    evaluation_q, selection_q = run_chronos2(splits, out_dir)
                else:
                    raise ValueError(f"unknown candidate {candidate}")
                for period, quantiles in (("evaluation", evaluation_q), ("selection", selection_q)):
                    scores = save_period(out_dir, splits[period], quantiles,
                                         seed, fold, region, candidate, period)
                    score_rows.append({"fold": fold, "region": region, "candidate": candidate,
                                       "seed": seed, "period": period, **scores})
                    pd.DataFrame(score_rows).to_csv(args.output_root / "scores.csv", index=False)
                print(f"done {fold} {region} {candidate} seed={seed}", flush=True)
    print(f"completed: {len(score_rows)} period scores -> {args.output_root / 'scores.csv'}")


if __name__ == "__main__":
    main()
