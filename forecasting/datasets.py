from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)


def load_forecast_config(config_path: Path | str) -> dict:
    path = Path(config_path).resolve()
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    config["project_root"] = str(path.parent.parent)
    return config


def _required_columns(config: Mapping) -> set[str]:
    data_config = config["data"]
    return {
        data_config["delivery_column"],
        data_config["available_column"],
        data_config["region_column"],
        data_config["split_column"],
        *data_config["history_columns"],
        data_config["target_column"],
    }


def _parse_timestamps(frame: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    data_config = config["data"]
    timezone = data_config["timezone"]
    parsed = frame.copy()
    for column in (data_config["delivery_column"], data_config["available_column"]):
        parsed[column] = pd.to_datetime(parsed[column], utc=True).dt.tz_convert(timezone)
    return parsed


def _validate_hourly_timeline(frame: pd.DataFrame, config: Mapping) -> None:
    data_config = config["data"]
    delivery = frame[data_config["delivery_column"]]
    available = frame[data_config["available_column"]]
    if delivery.duplicated().any() or not delivery.is_monotonic_increasing:
        raise ValueError("Delivery timestamps must be unique and strictly increasing")
    if not delivery.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Delivery timestamps must form a continuous hourly grid")
    if not available.is_monotonic_increasing:
        raise ValueError("Availability timestamps must be increasing")
    if not available.eq(delivery + pd.Timedelta(hours=1)).all():
        raise ValueError("Every hourly observation must become available one hour after delivery starts")


def _validate_splits(frame: pd.DataFrame, config: Mapping) -> None:
    data_config = config["data"]
    split_column = data_config["split_column"]
    delivery_column = data_config["delivery_column"]
    timezone = data_config["timezone"]
    actual_names = set(frame[split_column].unique())
    expected_names = set(data_config["split_names"])
    if actual_names != expected_names:
        raise ValueError(f"Split labels {actual_names} do not match {expected_names}")
    matched = np.zeros(len(frame), dtype=bool)
    for split_name, boundaries in data_config["split_boundaries"].items():
        start = pd.Timestamp(boundaries["start"], tz=timezone)
        end = pd.Timestamp(boundaries["end_exclusive"], tz=timezone)
        in_period = frame[delivery_column].ge(start) & frame[delivery_column].lt(end)
        if not frame.loc[in_period, split_column].eq(split_name).all():
            raise ValueError(f"Rows in the {split_name} period have incorrect split labels")
        matched |= in_period.to_numpy()
    if not matched.all():
        raise ValueError("Some timestamps fall outside the configured split periods")


def _load_region_frame(config: Mapping, region: str) -> pd.DataFrame:
    data_config = config["data"]
    region_files = data_config["region_files"]
    if region not in region_files:
        raise KeyError(f"Unknown AEMO region: {region}")
    path = Path(config["project_root"]) / region_files[region]
    frame = pd.read_csv(path)
    missing_columns = _required_columns(config) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    frame = _parse_timestamps(frame, config)
    if not frame[data_config["region_column"]].eq(region).all():
        raise ValueError(f"Region column does not contain only {region}")
    numeric_columns = list(dict.fromkeys(data_config["history_columns"] + [data_config["target_column"]]))
    if frame[numeric_columns].isna().any().any():
        raise ValueError(f"Missing values found in model columns for {region}")
    _validate_hourly_timeline(frame, config)
    _validate_splits(frame, config)
    return frame


def _fit_standardizer(values: np.ndarray) -> Standardizer:
    mean = values.mean(axis=0, dtype=np.float64)
    std = values.std(axis=0, dtype=np.float64)
    if np.any(std == 0):
        raise ValueError("Training features must have non-zero standard deviation")
    return Standardizer(mean=mean, std=std)


def _calendar_feature_map(delivery: pd.Series, train_mask: np.ndarray) -> Dict[str, np.ndarray]:
    hour = delivery.dt.hour.to_numpy()
    day_of_week = delivery.dt.dayofweek.to_numpy()
    day_of_year = delivery.dt.dayofyear.to_numpy()
    days_in_year = np.where(delivery.dt.is_leap_year.to_numpy(), 366.0, 365.0)
    month = delivery.dt.month.to_numpy()
    year = delivery.dt.year.to_numpy(dtype=np.float64)
    year_mean = year[train_mask].mean()
    year_std = year[train_mask].std()
    if year_std == 0:
        raise ValueError("Training years must have non-zero standard deviation")
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0),
        "day_of_week_sin": np.sin(2 * np.pi * day_of_week / 7.0),
        "day_of_week_cos": np.cos(2 * np.pi * day_of_week / 7.0),
        "day_of_year_sin": np.sin(2 * np.pi * (day_of_year - 1) / days_in_year),
        "day_of_year_cos": np.cos(2 * np.pi * (day_of_year - 1) / days_in_year),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12.0),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12.0),
        "is_weekend": (day_of_week >= 5).astype(np.float64),
        "year_standardized": (year - year_mean) / year_std,
    }


def _build_calendar_matrix(
    delivery: pd.Series,
    train_mask: np.ndarray,
    feature_names: Sequence[str],
) -> np.ndarray:
    feature_map = _calendar_feature_map(delivery, train_mask)
    unknown = set(feature_names) - set(feature_map)
    if unknown:
        raise ValueError(f"Unknown calendar features: {sorted(unknown)}")
    return np.column_stack([feature_map[name] for name in feature_names]).astype(np.float32)


def _valid_origins(
    frame: pd.DataFrame,
    config: Mapping,
    split: str,
) -> np.ndarray:
    data_config = config["data"]
    protocol = config["forecast_protocol"]
    input_hours = protocol["input_hours"]
    output_hours = protocol["output_hours"]
    stride_hours = protocol["stride_hours"]
    origin_hour = protocol.get("origin_hour")
    if origin_hour is None:
        starts = np.arange(input_hours, len(frame) - output_hours + 1, stride_hours)
    else:
        if isinstance(origin_hour, bool) or not isinstance(origin_hour, int):
            raise ValueError("forecast_protocol.origin_hour must be an integer from 0 to 23")
        if not 0 <= origin_hour <= 23:
            raise ValueError("forecast_protocol.origin_hour must be an integer from 0 to 23")
        if output_hours > 24:
            raise ValueError(
                "A daily fixed origin requires output_hours <= 24 to avoid overlapping targets"
            )
        all_starts = np.arange(input_hours, len(frame) - output_hours + 1)
        delivery_hours = frame[data_config["delivery_column"]].dt.hour.to_numpy()
        starts = all_starts[delivery_hours[all_starts] == origin_hour]
    split_values = frame[data_config["split_column"]].to_numpy()
    targets_in_split = np.array(
        [np.all(split_values[start:start + output_hours] == split) for start in starts]
    )
    delivery_ns = frame[data_config["delivery_column"]].array.asi8
    available_ns = frame[data_config["available_column"]].array.asi8
    history_is_available = available_ns[starts - 1] <= delivery_ns[starts]
    return starts[targets_in_split & history_is_available]


def _to_unix_seconds(timestamps: pd.Series) -> np.ndarray:
    utc_naive = timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
    return utc_naive.to_numpy(dtype="datetime64[s]").astype(np.int64)


class AEMOForecastDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, config: Mapping, region: str, split: str):
        data_config = config["data"]
        protocol = config["forecast_protocol"]
        if split not in data_config["split_names"]:
            raise ValueError(f"Unknown split: {split}")
        self.region = region
        self.split = split
        self.input_hours = protocol["input_hours"]
        self.output_hours = protocol["output_hours"]
        self.history_feature_names = tuple(data_config["history_columns"])
        self.calendar_feature_names = tuple(protocol["calendar_features"])
        split_values = frame[data_config["split_column"]].to_numpy()
        train_mask = split_values == "train"
        raw_history = frame[list(self.history_feature_names)].to_numpy(dtype=np.float64)
        self.history_standardizer = _fit_standardizer(raw_history[train_mask])
        self.history_values = self.history_standardizer.transform(raw_history)
        target_index = self.history_feature_names.index(data_config["target_column"])
        raw_target = frame[data_config["target_column"]].to_numpy(dtype=np.float32)
        self.target_values_raw = raw_target
        self.target_values = (
            (raw_target - self.history_standardizer.mean[target_index])
            / self.history_standardizer.std[target_index]
        ).astype(np.float32)
        self.calendar_values = _build_calendar_matrix(
            frame[data_config["delivery_column"]], train_mask, self.calendar_feature_names
        )
        self.delivery_unix_seconds = _to_unix_seconds(
            frame[data_config["delivery_column"]]
        )
        self.origin_indices = _valid_origins(frame, config, split)

    def __len__(self) -> int:
        return len(self.origin_indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        forecast_start = int(self.origin_indices[index])
        history_start = forecast_start - self.input_hours
        forecast_end = forecast_start + self.output_hours
        return {
            "history_values": torch.from_numpy(
                self.history_values[history_start:forecast_start].copy()
            ),
            "future_calendar": torch.from_numpy(
                self.calendar_values[forecast_start:forecast_end].copy()
            ),
            "target_price": torch.from_numpy(
                self.target_values[forecast_start:forecast_end, None].copy()
            ),
            "target_price_raw": torch.from_numpy(
                self.target_values_raw[forecast_start:forecast_end, None].copy()
            ),
            "forecast_origin_unix": torch.tensor(
                self.delivery_unix_seconds[forecast_start], dtype=torch.int64
            ),
            "target_delivery_unix": torch.from_numpy(
                self.delivery_unix_seconds[forecast_start:forecast_end].copy()
            ),
        }


def build_region_datasets(
    config_path: Path | str,
    region: str,
) -> Dict[str, AEMOForecastDataset]:
    config = load_forecast_config(config_path)
    frame = _load_region_frame(config, region)
    return {
        split: AEMOForecastDataset(frame, config, region, split)
        for split in config["data"]["split_names"]
    }


def build_all_region_datasets(
    config_path: Path | str,
) -> Dict[str, Dict[str, AEMOForecastDataset]]:
    config = load_forecast_config(config_path)
    return {
        region: build_region_datasets(config_path, region)
        for region in config["data"]["region_files"]
    }
