"""Shared, model-independent input boundary for new aligned experiments.

All adapters receive these exact seven blocks; flattening is a layout change,
not a separate feature pipeline. Model-internal normalization is unrestricted.
Archive timestamps are availability proxies, not a publication-time audit.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .datasets import AEMOForecastDataset, _load_region_frame, load_forecast_config

INPUT_KEYS = (
    "history_values", "history_calendar", "future_calendar", "origin_context",
    "future_exogenous", "ctf_exogenous", "quantile_exogenous",
)


def _array_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, values in arrays.items():
        array = np.ascontiguousarray(values)
        digest.update(json.dumps([name, array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass
class AlignedSplit:
    inputs: dict[str, np.ndarray]
    actual: np.ndarray
    origin_unix: np.ndarray
    target_time_unix: np.ndarray
    metadata: dict
    target: np.ndarray
    dataset: AEMOForecastDataset | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.origin_unix)


def flatten_inputs(split: AlignedSplit) -> np.ndarray:
    """Return every shared block, in fixed order, without adding features."""
    if set(split.inputs) != set(INPUT_KEYS):
        raise ValueError("Aligned inputs must contain exactly the seven shared blocks")
    return np.concatenate([
        split.inputs[key].reshape(len(split), int(np.prod(split.inputs[key].shape[1:])))
        for key in INPUT_KEYS
    ], axis=1)


class AlignedWindows(Dataset):
    """Torch layout adapter; keep blocks separate and supervision out of inputs."""
    def __init__(self, split: AlignedSplit):
        self.split = split
        self.dataset = split.dataset

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        split = self.split
        sample = {key: torch.from_numpy(split.inputs[key][index].copy()) for key in INPUT_KEYS}
        target = torch.from_numpy(split.target[index, :, None].copy())
        sample.update(
            target_price=target,
            target_quantile=target.clone(),
            target_price_raw=torch.from_numpy(split.actual[index, :, None].copy()),
            forecast_origin_unix=torch.tensor(split.origin_unix[index], dtype=torch.int64),
            target_delivery_unix=torch.from_numpy(split.target_time_unix[index].copy()),
        )
        return sample


def _timestamp(value, timezone: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(timezone) if stamp.tzinfo is None else stamp.tz_convert(timezone)


def _prepare_config(config_path, split_names, end_exclusive) -> tuple[dict, list[str]]:
    config = (copy.deepcopy(dict(config_path)) if isinstance(config_path, Mapping)
              else load_forecast_config(config_path))
    data = config["data"]
    # A single affine price space for history, point and quantile supervision.
    data["price_transform"] = "identity"
    data["quantile_target"] = "transformed"
    if data.get("price_cap") is not None:
        raise ValueError("Aligned raw-price scoring forbids data.price_cap")
    names = list(split_names if split_names is not None else data["split_names"])
    if not names or len(set(names)) != len(names):
        raise ValueError("split_names must be nonempty and unique")
    unknown = set(names) - set(data["split_names"])
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")
    timezone = data["timezone"]
    # Selecting development splits automatically excludes later target rows.
    requested_end = max(_timestamp(data["split_boundaries"][s]["end_exclusive"], timezone)
                        for s in names)
    cutoffs = [requested_end]
    for value in (end_exclusive, data.get("load_end_exclusive")):
        if value is not None:
            cutoffs.append(_timestamp(value, timezone))
    cutoff = min(cutoffs)
    train_end = _timestamp(data["split_boundaries"]["train"]["end_exclusive"], timezone)
    if cutoff < train_end:
        raise ValueError("Cutoff truncates the configured training period; define a new fold explicitly")
    periods = {}
    for name, bounds in data["split_boundaries"].items():
        start = _timestamp(bounds["start"], timezone)
        if start < cutoff:
            periods[name] = dict(bounds, end_exclusive=str(min(
                _timestamp(bounds["end_exclusive"], timezone), cutoff)))
    if set(names) - set(periods):
        raise ValueError("Requested split lies entirely beyond end_exclusive")
    data["split_boundaries"] = periods
    data["split_names"] = list(periods)
    data["relabel_splits"] = True
    data["load_end_exclusive"] = str(cutoff)
    return config, names


def _preprocessing_hash(dataset, frame, config) -> str:
    data = config["data"]
    train = frame[data["split_column"]].eq("train").to_numpy()
    arrays = {
        "history_mean": dataset.history_standardizer.mean,
        "history_std": dataset.history_standardizer.std,
        "train_history": dataset.history_values[train],
        "train_calendar": dataset.calendar_values[train],
    }
    if dataset.origin_context_values is not None:
        arrays["train_context"] = dataset.origin_context_values[train]
        arrays["context_mean"] = np.asarray(dataset.origin_context_standardizer.mean)
        arrays["context_std"] = np.asarray(dataset.origin_context_standardizer.std)
    train_times = set(dataset.delivery_unix_seconds[train].tolist())
    for key in INPUT_KEYS[-3:]:
        values = getattr(dataset, f"{key}_by_origin")
        selected = [value for origin, value in sorted(values.items()) if origin in train_times]
        if selected:
            arrays[f"train_{key}"] = np.stack(selected)
    pipeline = {key: copy.deepcopy(value) for key, value in config.items()
                if key in {"data", "forecast_protocol", "origin_context", "future_exogenous",
                           "ctf_exogenous", "quantile_exogenous", "predispatch_exogenous",
                           "interconnector_exogenous"}}
    # Loading an additional evaluation split cannot alter a fitted transform's
    # identity. Preserve only the training period in this preprocessing digest.
    pipeline["data"].pop("load_end_exclusive", None)
    pipeline["data"].pop("split_names", None)
    pipeline["data"]["split_boundaries"] = {
        "train": pipeline["data"]["split_boundaries"]["train"]}
    digest = hashlib.sha256(json.dumps(pipeline, sort_keys=True, default=str).encode())
    digest.update(_array_hash(arrays).encode())
    return digest.hexdigest()


def build_aligned_splits(
    config_path: Path | str | Mapping, region: str, *,
    split_names: Sequence[str] | None = None, max_origins: int | None = None,
    end_exclusive: str | None = None,
) -> dict[str, AlignedSplit]:
    """Build once and share across models; scalers always use the complete train fold.

    ``max_origins`` takes the same earliest origins for every model, after fitting
    train-only preprocessing. It is for smoke checks, not a training-budget rule.
    Selecting train/validation excludes the later scoring period before dataset
    validation and fitting. Input files may still contain later rows on disk.
    """
    if max_origins is not None and (
        isinstance(max_origins, bool) or not isinstance(max_origins, (int, np.integer))
        or max_origins < 1
    ):
        raise ValueError("max_origins must be a positive integer")
    config, names = _prepare_config(config_path, split_names, end_exclusive)
    frame = _load_region_frame(config, region)
    if not frame[config["data"]["split_column"]].eq("train").any():
        raise ValueError("Configured training fold is empty")
    result = {}
    preprocessing_hash = None
    for name in names:
        dataset = AEMOForecastDataset(frame, config, region, name)
        # Use the very same transformed array for historical and supervised
        # prices. The legacy target path casts raw prices to float32 first,
        # introducing an avoidable rounding difference before standardization.
        dataset.target_values = dataset.history_values[:, dataset.target_index].copy()
        if max_origins is not None:
            dataset.origin_indices = dataset.origin_indices[:max_origins]
        if not len(dataset):
            raise ValueError(f"No complete forecast windows for {region}/{name}")
        if preprocessing_hash is None:
            preprocessing_hash = _preprocessing_hash(dataset, frame, config)
        starts = dataset.origin_indices
        h = dataset.output_hours
        history_rows = starts[:, None] + np.arange(-dataset.input_hours, 0)
        target_rows = starts[:, None] + np.arange(h)
        origins = dataset.delivery_unix_seconds[starts].copy()
        inputs = {
            "history_values": dataset.history_values[history_rows].copy(),
            "history_calendar": dataset.calendar_values[history_rows].copy(),
            "future_calendar": dataset.calendar_values[target_rows].copy(),
            "origin_context": (dataset.origin_context_values[starts, None].copy()
                               if dataset.origin_context_values is not None
                               else np.empty((len(starts), 0), dtype=np.float32)),
        }
        for key in INPUT_KEYS[-3:]:
            values = getattr(dataset, f"{key}_by_origin")
            inputs[key] = (np.stack([values[int(origin)] for origin in origins]) if values
                           else np.empty((len(starts), h, 0), dtype=np.float32))
        actual = dataset.target_values_raw[target_rows].copy()
        target = dataset.target_values[target_rows].copy()
        target_times = dataset.delivery_unix_seconds[target_rows].copy()
        for key, values in {**inputs, "actual": actual, "target": target}.items():
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite aligned values in {region}/{name}/{key}")
        metadata = {
            "protocol_id": config.get("aligned_protocol", {}).get("id", "aligned-v1"),
            "region": region, "split": name,
            "input_hash": _array_hash(inputs),
            "sample_hash": _array_hash({"origin": origins, "target_time": target_times,
                                         "actual": actual}),
            "preprocessing_hash": preprocessing_hash,
            "target_mean": float(dataset.history_standardizer.mean[dataset.target_index]),
            "target_std": float(dataset.history_standardizer.std[dataset.target_index]),
            "price_transform": "identity", "standardization": "train_zscore",
            "input_block_order": list(INPUT_KEYS),
            "load_end_exclusive": config["data"]["load_end_exclusive"],
            "availability_audit": "archive_timestamp_proxies_only",
        }
        result[name] = AlignedSplit(inputs, actual, origins, target_times, metadata, target, dataset)
    return result
