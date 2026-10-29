"""G0 local-only, CPU-only Chronos tool; no training or scoring entry point."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .contracts import FrozenForecastInput

QUANTILES = (0.05, 0.1, 0.5, 0.9, 0.95)
INPUT_FILES = (
    "metadata.json", "origin_unix.npy", "input__history_values.npy",
    "input__history_calendar.npy", "input__future_calendar.npy",
)
TYPED_FUTURE_PAST_ENCODING = "all_nan_honest_absence"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_development_input(config: dict) -> tuple[FrozenForecastInput, dict]:
    """Read only five explicitly pinned files; never use the old split loader."""
    spec = config["input"]
    root = Path(spec["directory"])
    expected = spec["file_sha256"]
    if set(expected) != set(INPUT_FILES):
        raise ValueError("input file allowlist mismatch")
    for name in INPUT_FILES:
        if sha256_file(root / name) != expected[name]:
            raise ValueError(f"input identity mismatch: {name}")
    metadata = json.loads((root / "metadata.json").read_text())
    row = spec["row"]
    origin = int(np.load(root / "origin_unix.npy", mmap_mode="r", allow_pickle=False)[row])
    if origin != spec["origin_unix"]:
        raise ValueError("origin mismatch")
    history = np.array(np.load(root / "input__history_values.npy", mmap_mode="r",
                              allow_pickle=False)[row], dtype=np.float64, copy=True)
    past = np.array(np.load(root / "input__history_calendar.npy", mmap_mode="r",
                           allow_pickle=False)[row], dtype=np.float64, copy=True)
    future = np.array(np.load(root / "input__future_calendar.npy", mmap_mode="r",
                             allow_pickle=False)[row], dtype=np.float64, copy=True)
    if history.shape != (72, 2) or past.shape != (72, 10) or future.shape != (24, 10):
        raise ValueError("cached input shape mismatch")
    mean, std = float(metadata["target_mean"]), float(metadata["target_std"])
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError("invalid frozen decoding transform")
    covariates = {"demand_std": tuple(history[:, 1])}
    covariates.update({f"cal_{i}": tuple(past[:, i]) for i in range(10)})
    request = FrozenForecastInput(
        origin_unix=origin,
        target_time_unix=tuple(origin + 3600 * h for h in range(24)),
        price_history=tuple(history[:, 0] * std + mean),
        past_covariates=covariates,
        future_covariates={f"cal_{i}": tuple(future[:, i]) for i in range(10)},
        metadata={"region": spec["region"], "input_profile": "legacy_price_demand_calendar",
                  "transform_sha256": expected["metadata.json"]},
    )
    provenance = {
        "files_read": [str(root / name) for name in INPUT_FILES],
        "file_sha256": expected, "row": row, "origin_unix": origin,
        "availability_status": "archive_timestamp_proxies_only",
        "strict_live_asof_verified": False,
        "transform": "decode_price_frozen_train_zscore; demand_keeps_existing_train_zscore",
        "new_fitting": False, "future_labels_read": False,
    }
    return request, provenance


def validate_snapshot(spec: dict) -> dict:
    path = Path(spec["directory"])
    if not path.is_dir() or path.name != spec["revision"]:
        raise ValueError("fixed local snapshot missing or wrong revision")
    identities = {}
    for name in ("config.json", "model.safetensors"):
        digest = sha256_file(path / name)
        if digest != spec["sha256"][name]:
            raise ValueError(f"checkpoint identity mismatch: {name}")
        identities[name] = {"sha256": digest, "size_bytes": (path / name).stat().st_size}
    config = json.loads((path / "config.json").read_text())
    if not set(QUANTILES).issubset(config["chronos_config"]["quantiles"]):
        raise ValueError("requested quantiles are not native checkpoint levels")
    return identities


def validate_output(array) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.shape != (24, 5) or not np.isfinite(values).all():
        raise ValueError("expected finite 24 by 5 native quantiles")
    # No clipping, sorting, calibration or LLM modification of tool output.
    return values


def build_model_input(request: FrozenForecastInput) -> dict:
    """Map one frozen request to the Chronos-2 list-of-dicts entry format.

    A typed future forecast has no same-semantic past observations.  Chronos-2
    requires every future-covariate name to exist in the past stream and masks
    NaN context entries natively, so the honest absence is encoded as an
    all-NaN past of the same length as the target history; no past value is
    invented and the real 24 future values pass through unchanged.
    """
    history_length = len(request.price_history)
    past = {key: np.asarray(value, dtype=np.float64)
            for key, value in request.past_covariates.items()}
    future = {key: np.asarray(value, dtype=np.float64)
              for key, value in request.future_covariates.items()}
    for name, values in request.future_forecasts.items():
        past[name] = np.full(history_length, np.nan, dtype=np.float64)
        future[name] = np.asarray(values, dtype=np.float64)
    return {
        "target": np.asarray(request.price_history, dtype=np.float64),
        "past_covariates": past,
        "future_covariates": future,
    }


def _parameter_hash(model) -> str:
    import torch
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def predict_frozen(request: FrozenForecastInput, config: dict) -> tuple[np.ndarray, dict]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU tool requires explicitly hidden CUDA devices")
    if any(os.environ.get(key) != "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise RuntimeError("offline mode required")
    import importlib.metadata
    versions = {name: importlib.metadata.version(name) for name in config["packages"]}
    if versions != config["packages"]:
        raise RuntimeError("frozen dependency versions changed")
    identity = validate_snapshot(config["checkpoint"])
    import torch
    from chronos import Chronos2Pipeline

    torch.set_num_threads(config["resources"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)

    def forbidden_fit(*args, **kwargs):
        raise RuntimeError("training is forbidden in G0")

    Chronos2Pipeline.fit = forbidden_fit
    pipeline = Chronos2Pipeline.from_pretrained(
        config["checkpoint"]["directory"], device_map="cpu", torch_dtype=torch.float32,
        local_files_only=True,
    )
    pipeline.model.eval()
    for parameter in pipeline.model.parameters():
        if parameter.device.type != "cpu":
            raise RuntimeError("non-CPU parameter")
        parameter.requires_grad_(False)
    before = _parameter_hash(pipeline.model)
    model_input = build_model_input(request)
    with torch.inference_mode():
        quantiles, _ = pipeline.predict_quantiles(
            [model_input], prediction_length=24, quantile_levels=list(QUANTILES),
            batch_size=1, context_length=72, cross_learning=False,
        )
    raw = np.asarray(quantiles[0], dtype=np.float64)
    if raw.shape != (1, 24, 5):
        raise ValueError("unexpected native output shape")
    output = validate_output(raw[0])
    after = _parameter_hash(pipeline.model)
    if after != before or identity != validate_snapshot(config["checkpoint"]):
        raise RuntimeError("frozen model identity changed")
    return output, {
        "checkpoint": identity, "state_sha256_before": before, "state_sha256_after": after,
        "packages": versions,
        "device": "cpu", "dtype": "float32", "torch_threads": torch.get_num_threads(),
        "grad_enabled": torch.is_grad_enabled(), "training_calls": 0,
        "quantile_levels": list(QUANTILES), "cross_learning": False,
        "negative_values_preserved": True,
        "native_crossing_count": int(np.sum(np.diff(output, axis=1) < 0)),
        "typed_future_forecasts": {name: {
            "past_encoding": TYPED_FUTURE_PAST_ENCODING,
            "future_values": 24,
        } for name in request.future_forecasts},
    }
