import unittest

import numpy as np
import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler

from forecasting.samplers import DailyHourRotationSampler
from forecasting.train import build_loaders


class HourlyDataset(Dataset):
    def __init__(self, days: int = 4):
        self.forecast_origin_hours = np.tile(np.arange(24), days)
        self.forecast_origin_unix_seconds = np.arange(days * 24) * 3600

    def __len__(self) -> int:
        return len(self.forecast_origin_hours)

    def __getitem__(self, index: int):
        return torch.tensor(index)


class DailyHourRotationSamplerTest(unittest.TestCase):
    def test_rotates_without_target_overlap_and_covers_all_windows(self):
        dataset = HourlyDataset(days=4)
        sampler = DailyHourRotationSampler(dataset, shuffle_days=True, seed=7)
        covered = set()

        for epoch in range(24):
            sampler.set_epoch(epoch)
            indices = list(sampler)
            covered.update(indices)
            self.assertEqual(sampler.current_hour, epoch)
            self.assertTrue(
                np.all(dataset.forecast_origin_hours[indices] == epoch)
            )
            selected_origins = np.sort(
                dataset.forecast_origin_unix_seconds[indices]
            )
            self.assertTrue(np.all(np.diff(selected_origins) == 24 * 3600))
            target_starts = selected_origins
            target_ends = selected_origins + 24 * 3600
            self.assertTrue(np.all(target_ends[:-1] <= target_starts[1:]))

        self.assertEqual(covered, set(range(len(dataset))))

    def test_epoch_24_restarts_the_hour_cycle(self):
        sampler = DailyHourRotationSampler(HourlyDataset(), start_hour=3)
        sampler.set_epoch(0)
        first_indices = set(sampler)
        self.assertEqual(sampler.current_hour, 3)
        sampler.set_epoch(24)
        self.assertEqual(sampler.current_hour, 3)
        self.assertEqual(set(sampler), first_indices)


class LoaderCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.datasets = {
            "train": HourlyDataset(),
            "validation": HourlyDataset(days=2),
            "test": HourlyDataset(days=3),
        }
        self.training_config = {
            "batch_size": 8,
            "num_workers": 0,
            "seed": 2026,
        }

    def test_default_loader_keeps_all_window_shuffle_behavior(self):
        loaders = build_loaders(self.datasets, self.training_config)
        self.assertIsInstance(loaders["train"].sampler, RandomSampler)
        self.assertEqual(len(loaders["train"].sampler), len(self.datasets["train"]))
        self.assertIsInstance(loaders["validation"].sampler, SequentialSampler)
        self.assertIsInstance(loaders["test"].sampler, SequentialSampler)

    def test_grouped_sampler_only_changes_training_loader(self):
        config = {
            **self.training_config,
            "train_sampling": {
                "mode": "daily_hour_rotation",
                "start_hour": 0,
                "shuffle_days": True,
            },
        }
        loaders = build_loaders(self.datasets, config)
        self.assertIsInstance(loaders["train"].sampler, DailyHourRotationSampler)
        self.assertEqual(len(loaders["train"].sampler), 4)
        self.assertEqual(len(loaders["validation"].sampler), 48)
        self.assertEqual(len(loaders["test"].sampler), 72)


if __name__ == "__main__":
    unittest.main()
