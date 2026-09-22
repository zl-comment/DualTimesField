from collections.abc import Iterator, Mapping

import numpy as np
from torch.utils.data import Sampler


class DailyHourRotationSampler(Sampler[int]):
    """Select one local forecast-origin hour per day and rotate it by epoch."""

    HOURS_PER_DAY = 24
    SECONDS_PER_DAY = 24 * 60 * 60

    def __init__(
        self,
        dataset,
        *,
        start_hour: int = 0,
        shuffle_days: bool = True,
        seed: int = 0,
    ) -> None:
        if not 0 <= start_hour < self.HOURS_PER_DAY:
            raise ValueError("start_hour must be between 0 and 23")
        if not hasattr(dataset, "forecast_origin_hours"):
            raise TypeError("Dataset must expose forecast_origin_hours")
        if not hasattr(dataset, "forecast_origin_unix_seconds"):
            raise TypeError("Dataset must expose forecast_origin_unix_seconds")

        hours = np.asarray(dataset.forecast_origin_hours, dtype=np.int64)
        origins = np.asarray(dataset.forecast_origin_unix_seconds, dtype=np.int64)
        if hours.shape != (len(dataset),) or origins.shape != (len(dataset),):
            raise ValueError("Forecast-origin metadata must have one entry per dataset item")

        self.start_hour = start_hour
        self.shuffle_days = shuffle_days
        self.seed = seed
        self.epoch = 0
        self._indices_by_hour = {
            hour: np.flatnonzero(hours == hour).astype(np.int64)
            for hour in range(self.HOURS_PER_DAY)
        }
        for hour, indices in self._indices_by_hour.items():
            if len(indices) == 0:
                raise ValueError(f"Training dataset has no forecast origins at hour {hour}")
            selected_origins = np.sort(origins[indices])
            if len(selected_origins) > 1 and not np.all(
                np.diff(selected_origins) == self.SECONDS_PER_DAY
            ):
                raise ValueError(
                    f"Forecast origins at hour {hour} are not exactly 24 hours apart"
                )

    @property
    def current_hour(self) -> int:
        return (self.start_hour + self.epoch) % self.HOURS_PER_DAY

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        indices = self._indices_by_hour[self.current_hour].copy()
        if self.shuffle_days:
            np.random.default_rng(self.seed + self.epoch).shuffle(indices)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return len(self._indices_by_hour[self.current_hour])


def build_train_sampler(dataset, training_config: Mapping):
    sampler_config = training_config.get("train_sampling")
    if not sampler_config or sampler_config.get("mode", "all_windows") == "all_windows":
        return None
    mode = sampler_config.get("mode")
    if mode != "daily_hour_rotation":
        raise ValueError(f"Unknown training sampler mode: {mode}")
    return DailyHourRotationSampler(
        dataset,
        start_hour=sampler_config.get("start_hour", 0),
        shuffle_days=sampler_config.get("shuffle_days", True),
        seed=training_config["seed"],
    )
