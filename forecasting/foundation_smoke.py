"""End-to-end real-weight smoke for the foundation batch on development data.

Runs each registered adapter against real official weights inside its own
model interpreter, on the selection split (development data) only: the split
is presented to the adapter exactly as the formal runner presents test — a
label-free PredictionView — and labels are loaded by this script only after
prediction to score the smoke artifact. Smoke artifacts carry smoke=True and
can never enter formal reports.

For tabpfn_v3 the train label-context is truncated (train_context_origins)
so the smoke stays cheap; the formal context policy is unchanged and recorded
in source.json.

Usage (inside the model's own venv, from the repository root):
  .venvs/<env>/bin/python -m forecasting.foundation_smoke --model <model_id> \
      [--origins 24] [--train-context-origins 2000] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np

from forecasting.aligned_experiment import read_json
from forecasting.aligned_inputs import INPUT_KEYS
from forecasting.aligned_scoring import (QUANTILE_LEVELS, make_prediction_artifact,
                                         save_prediction_artifact, score_predictions)
from forecasting.common.cache import load_split
from forecasting.common.foundation_api import PredictionView
from forecasting.common.foundation_registry import FOUNDATION_REGISTRY, load_foundation_adapter
from forecasting.foundation_experiment import _predict_indices

DEFAULT_PARENT = Path("outputs/forecasting/aligned_baselines_20261012")
SMOKE_ROOT = Path("outputs/forecasting/foundation_smoke_20261023")


def _view(split, limit):
    origins = np.asarray(split.origin_unix[:limit])
    return PredictionView(
        {key: np.asarray(split.inputs[key][:limit]) for key in INPUT_KEYS},
        origins, np.asarray(split.target_time_unix[:limit]), dict(split.metadata))


class _TruncatedSplit:
    """First-`limit`-origins view of an AlignedSplit with working len()."""

    def __init__(self, split, limit: int):
        self.inputs = {key: np.asarray(split.inputs[key][:limit]) for key in INPUT_KEYS}
        self.actual = np.asarray(split.actual[:limit])
        self.origin_unix = np.asarray(split.origin_unix[:limit])
        self.target_time_unix = np.asarray(split.target_time_unix[:limit])
        self.metadata = dict(split.metadata)
        self.target = np.asarray(split.target[:limit])
        self._limit = limit

    def __len__(self) -> int:
        return self._limit


def run_smoke(model: str, *, parent=DEFAULT_PARENT, region="NSW1", seed=2026,
              origins=24, train_context_origins=2000, device="cuda:0") -> dict:
    spec = FOUNDATION_REGISTRY[model]
    config = __import__("yaml").safe_load(
        Path(f"configs/{model}/aligned.yaml").read_text())
    config["project_root"] = str(Path.cwd().resolve())
    manifest = read_json(parent / "jobs.json")
    selection = load_split(parent, manifest, region, "selection")
    view = _view(selection, origins)
    actual = np.asarray(selection.actual[:origins])

    out_dir = SMOKE_ROOT / model / f"seed{seed}" / region
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    train = None
    if spec.fit_requires_train_labels:
        full_train = load_split(parent, manifest, region, "train")
        train = _TruncatedSplit(full_train, min(train_context_origins, len(full_train.origin_unix)))
    context = SimpleNamespace(
        run_root=SMOKE_ROOT, config=config, manifest={}, job={"model": model, "seed": seed,
        "region": region, "output": f"{model}/seed{seed}/{region}"},
        region=region, seed=seed, device=device, test_view=view, train=train,
        selection=None, calibration=None, parent_run=parent.resolve(),
        progress_path=out_dir / "progress.json", model_directory=out_dir / "models",
        prediction_directory=out_dir / "predictions")
    adapter_cls = load_foundation_adapter(model)
    started = time.monotonic()
    adapter = adapter_cls(context)
    identity = adapter.identity()
    adapter.fit()
    payload = _predict_indices(adapter, view, np.arange(len(view)), "smoke")
    seconds = time.monotonic() - started
    if hasattr(adapter, "auxiliary_quantiles"):
        np.save(out_dir / "auxiliary_quantiles.npy", adapter.auxiliary_quantiles())

    output_type = "quantile" if spec.output_table == "Q" else "point_only"
    metadata = dict(view.metadata, seed=seed, model=model, source=f"{model}:smoke",
                    prediction_space="raw_rrp", output_type=output_type, smoke=True,
                    split="selection", model_metadata={"adapter_identity": identity,
                    "smoke": {"origins": origins, "train_context_origins": train_context_origins,
                              "seconds": round(seconds, 3)}})
    artifact = make_prediction_artifact(
        actual=actual, origin_unix=view.origin_unix, target_time_unix=view.target_time_unix,
        metadata=metadata, quantiles=payload.get("quantiles"), point=payload.get("point"),
        quantile_levels=np.array(QUANTILE_LEVELS) if payload.get("quantiles") is not None else None)
    save_prediction_artifact(out_dir / "test.npz", artifact)
    scores = score_predictions(artifact)
    summary = {"model": model, "region": region, "seed": seed, "smoke": True,
               "dev_split": "selection", "origins": origins,
               "train_context_origins": train_context_origins if train is not None else None,
               "identity_weights": identity.get("weights", {}),
               "seconds": round(seconds, 3),
               "scores": {key: scores[key] for key in scores
                          if key in ("mae", "rmse_pooled", "quantile_score_proxy",
                                     "coverage_90", "crossing_rate", "point_definition")},
               "output": str(out_dir / "test.npz")}
    (out_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=["tabpfn_v3", "chronos2_zeroshot", "moirai2_zeroshot", "timesfm25_xreg"])
    parser.add_argument("--region", default="NSW1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--origins", type=int, default=24)
    parser.add_argument("--train-context-origins", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    args = parser.parse_args()
    summary = run_smoke(args.model, parent=args.parent, region=args.region, seed=args.seed,
                        origins=args.origins, train_context_origins=args.train_context_origins,
                        device=args.device)
    printable = {key: summary[key] for key in ("model", "seconds")}
    printable.update(summary["scores"])
    print(json.dumps(printable, indent=2), flush=True)


if __name__ == "__main__":
    main()
