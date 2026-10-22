"""End-to-end smoke for the expansion baselines on real parent data.

Tree-calibration models (lightgbm_cp, lightgbm_qra, xgboost_cp) are cheap and
their reproduction gate requires the full test split, so their smoke runs full
fidelity. The native neural models (ddnn_jsu, nhits_qra) run with truncated
origins and a two-epoch budget. Every artifact is written with smoke=True so
the formal reporter can never admit it, and outputs live in their own smoke
directory.

Smoke checks plumbing and contracts on real data; it is never a quality
ranking and never looks at test performance to pick implementations.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from forecasting.aligned_experiment import read_json
from forecasting.aligned_scoring import (QUANTILE_LEVELS, make_prediction_artifact,
                                         save_prediction_artifact, score_predictions)
from forecasting.common.baseline_registry import REGISTRY, run_expansion_model
from forecasting.common.cache import load_split
from forecasting.datasets import load_forecast_config

DEFAULT_PARENT = Path("outputs/forecasting/aligned_baselines_20261012")
SMOKE_ROOT = Path("outputs/forecasting/expansion_smoke_20261022")


def _truncated(split, limit):
    if limit is None or len(split.origin_unix) <= limit:
        return split
    return SimpleNamespace(
        inputs={key: value[:limit] for key, value in split.inputs.items()},
        actual=np.asarray(split.actual[:limit]),
        origin_unix=np.asarray(split.origin_unix[:limit]),
        target_time_unix=np.asarray(split.target_time_unix[:limit]),
        metadata=dict(split.metadata), target=np.asarray(split.target[:limit]),
        __len__=lambda s=limit: s)


def run_smoke(model: str, *, parent=DEFAULT_PARENT, region="NSW1", seed=2026,
              neural_origins=512, epochs=2, device="cpu") -> dict:
    config = load_forecast_config(f"configs/{model}/aligned.yaml")
    if model in ("ddnn_jsu", "nhits_qra"):
        config["training"]["epochs"] = epochs
    spec = REGISTRY[model]
    manifest = read_json(parent / "jobs.json")
    wanted = {"test"}
    if spec.requires_calibration:
        wanted.add("calibration")
    if spec.native_training:
        wanted.update(("train", "selection"))
    splits = {}
    for name in ("train", "selection", "calibration", "test"):
        if name not in wanted:
            continue
        split = load_split(parent, manifest, region, name)
        splits[name] = _truncated(split, neural_origins) if spec.native_training else split

    out_dir = SMOKE_ROOT / model / f"seed{seed}" / region
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    context = SimpleNamespace(
        run_root=SMOKE_ROOT, config=config, manifest={}, region=region, seed=seed,
        device=device, splits=splits, parent_run=parent.resolve(),
        job={"model": model, "output": f"{model}/seed{seed}/{region}"},
        progress_path=out_dir / "progress.json", model_directory=out_dir / "models")
    result = run_expansion_model(model, context)
    metadata = dict(splits["test"].metadata, seed=seed, model=model, source=f"{model}:smoke",
                    prediction_space="raw_rrp", output_type="quantile", smoke=True,
                    split="test", model_metadata=result.metadata)
    artifact = make_prediction_artifact(
        actual=np.asarray(splits["test"].actual), origin_unix=np.asarray(splits["test"].origin_unix),
        target_time_unix=np.asarray(splits["test"].target_time_unix), metadata=metadata,
        quantiles=result.quantiles, point=result.point, quantile_levels=np.array(QUANTILE_LEVELS))
    save_prediction_artifact(out_dir / "test.npz", artifact)
    scores = score_predictions(artifact)
    summary = {"model": model, "region": region, "seed": seed, "smoke": True,
               "neural_origins": neural_origins if spec.native_training else None,
               "epochs": epochs if spec.native_training else None,
               "scores": {k: scores[k] for k in ("mae", "rmse_pooled", "quantile_score_proxy",
                                                 "coverage_90", "crossing_rate")},
               "output": str(out_dir / "test.npz")}
    (out_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",
                        default=["lightgbm_cp", "lightgbm_qra", "xgboost_cp", "ddnn_jsu", "nhits_qra"])
    parser.add_argument("--region", default="NSW1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--neural-origins", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    for model in args.models:
        summary = run_smoke(model, region=args.region, seed=args.seed,
                            neural_origins=args.neural_origins, epochs=args.epochs,
                            device=args.device)
        print(f"{model}: smoke passed — MAE {summary['scores']['mae']:.3f}, "
              f"QS {summary['scores']['quantile_score_proxy']:.3f}", flush=True)


if __name__ == "__main__":
    main()
