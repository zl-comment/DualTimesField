"""Arm A runner: frozen Chronos-2 zero-shot base forecast, CPU, offline, one call.

Must run inside the dedicated chronos-2 venv:
  /root/workspace/Code/DualTimesField-foundation-baselines/.venvs/chronos2/bin/python
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np

from . import data, protocol


def _guard_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only: CUDA_VISIBLE_DEVICES must be empty")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 required")


def _snapshot_dir(hf_home: str) -> Path:
    root = Path(hf_home) / "hub" / "models--amazon--chronos-2" / "snapshots"
    revisions = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    if len(revisions) != 1:
        raise RuntimeError(f"expected exactly one pinned chronos-2 revision, found {len(revisions)}")
    return revisions[0]


def run(region: str, data_root: str, output_dir: str, hf_home: str, limit: int | None) -> int:
    _guard_environment()
    records, grid_info = data.build_origin_grid(region, data_root)
    if not records:
        print(json.dumps({"error": "no scored origins", "grid": grid_info}))
        return 3
    if limit is not None:
        records = records[:limit]
    grid_info["n_scored_origins_used"] = len(records)

    import torch
    from chronos import Chronos2Pipeline

    torch.set_num_threads(min(8, os.cpu_count() or 8))
    torch.set_grad_enabled(False)
    Chronos2Pipeline.fit = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("training forbidden"))
    snapshot = _snapshot_dir(hf_home)
    checkpoint_identity = {
        "directory": str(snapshot),
        "config_sha256": data.sha256_file(snapshot / "config.json"),
        "weights_sha256": data.sha256_file(snapshot / "model.safetensors"),
    }
    pipeline = Chronos2Pipeline.from_pretrained(str(snapshot), device_map="cpu",
                                                torch_dtype=torch.float32, local_files_only=True)
    pipeline.model.eval()
    for parameter in pipeline.model.parameters():
        parameter.requires_grad_(False)
        if parameter.device.type != "cpu":
            raise RuntimeError("non-CPU parameter detected")

    inputs = []
    for record in records:
        past_cal = {name: record.history_calendar[:, i]
                    for i, name in enumerate(data.CALENDAR_FEATURES)}
        future_cal = {name: record.future_calendar[:, i]
                      for i, name in enumerate(data.CALENDAR_FEATURES)}
        inputs.append({
            "target": record.history_prices.astype(np.float64),
            "past_covariates": {"total_demand_mw": record.history_demand.astype(np.float64), **past_cal},
            "future_covariates": future_cal,
        })

    with torch.inference_mode():
        quantiles, _ = pipeline.predict_quantiles(
            inputs, prediction_length=protocol.HORIZON_HOURS,
            quantile_levels=list(protocol.QUANTILE_LEVELS),
            batch_size=len(inputs), context_length=protocol.HISTORY_HOURS,
            cross_learning=False,
        )
    values = np.asarray(quantiles, dtype=np.float64)
    expected_shape = (len(records), 1, protocol.HORIZON_HOURS, len(protocol.QUANTILE_LEVELS))
    if values.shape != expected_shape:
        raise RuntimeError(f"unexpected chronos output shape {values.shape} != {expected_shape}")
    values = values[:, 0, :, :]

    out = Path(output_dir)
    (out / "base").mkdir(parents=True, exist_ok=True)
    metadata = {
        "arm": "A", "region": region, "model": "amazon/chronos-2 zero-shot",
        "device": "cpu", "torch_dtype": "float32", "torch_threads": torch.get_num_threads(),
        "predict_calls": 1, "batch_size": len(inputs),
        "quantile_levels": list(protocol.QUANTILE_LEVELS),
        "grid": grid_info,
        "checkpoint": checkpoint_identity,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("chronos-forecasting", "torch", "numpy", "pandas")},
    }
    npz_path = out / "base" / f"{region}_base.npz"
    with npz_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            origin_unix=np.asarray([r.origin_unix for r in records], dtype=np.int64),
            origin_local=np.asarray([r.origin_local for r in records]),
            half=np.asarray([r.half for r in records]),
            target_time_unix=np.stack([r.target_time_unix for r in records]),
            actual=np.stack([r.actual for r in records]),
            quantiles=values,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    print(json.dumps({"status": "ok", "region": region, "n_origins": len(records),
                      "output": str(npz_path),
                      "quantile_finite": bool(np.isfinite(values).all()),
                      "crossing_count": int(np.sum(np.diff(values, axis=-1) < 0))}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=protocol.REGIONS)
    parser.add_argument("--data-root", default=data.DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir",
                        default="outputs/forecasting/agentic_m2m3_dev_c2_20261030_r6")
    parser.add_argument("--hf-home", default="/root/workspace/Code/DualTimesField/weights/hf_home")
    parser.add_argument("--limit", type=int, default=None, help="smoke: first N origins")
    args = parser.parse_args(argv)
    return run(args.region, args.data_root, args.output_dir, args.hf_home, args.limit)


if __name__ == "__main__":
    sys.exit(main())
