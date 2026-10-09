import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from forecasting.datasets import build_region_datasets
from forecasting.models import DualFieldLinearForecaster
from forecasting.train import build_model, move_inputs
from forecasting.xgboost_base import build, fold_blocks

QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)


def _model(**overrides):
    arguments = dict(
        num_variables=2,
        input_length=8,
        forecast_horizon=3,
        calendar_dim=2,
        quantiles=QUANTILES,
        num_frequencies=2,
        hidden_dim=8,
        num_layers=1,
        freq_cutoff=2.0,
        num_atoms=2,
        fusion_mode="additive_trigonometric_gate",
        external_base=True,
    )
    arguments.update(overrides)
    return DualFieldLinearForecaster(**arguments)


def _base_inputs(batch: int, horizon: int = 3):
    base_point = torch.randn(batch, horizon, 1)
    base_quantile = torch.sort(torch.randn(batch, horizon, len(QUANTILES)), dim=-1).values
    base_features = torch.cat([base_point, base_quantile], dim=-1)
    return base_point, base_quantile, base_features


class FoldBlocksTest(unittest.TestCase):
    def test_blocks_cover_origins_and_purge_overlapping_windows(self):
        origins = np.arange(100, 400)
        pairs = fold_blocks(origins, folds=3, window=10)
        covered = np.concatenate([block for block, _ in pairs])
        np.testing.assert_array_equal(covered, np.arange(len(origins)))
        for block, fit in pairs:
            self.assertEqual(len(np.intersect1d(block, fit)), 0)
            gaps = np.abs(origins[fit][:, None] - origins[block][None, :])
            self.assertGreaterEqual(gaps.min(), 10)


class ExternalBaseModelTest(unittest.TestCase):
    def test_point_is_the_base_without_field_point(self):
        torch.manual_seed(3)
        model = _model(field_point=False, base_inputs=True)
        history = torch.randn(4, 8, 2)
        model.initialize_atoms(history)
        base_point, base_quantile, base_features = _base_inputs(4)
        outputs = model(history, torch.randn(4, 3, 2), None, None, None, None,
                        base_point, base_quantile, base_features)
        torch.testing.assert_close(outputs["point_forecast"], base_point)
        quantile = outputs["quantile_forecast"]
        self.assertTrue(torch.all(quantile[..., 1:] >= quantile[..., :-1]))
        self.assertFalse(torch.allclose(quantile, base_quantile))

    def test_fields_correct_the_base_and_receive_gradients(self):
        torch.manual_seed(5)
        model = _model(base_inputs=True)
        history = torch.randn(4, 8, 2)
        model.initialize_atoms(history)
        base_point, base_quantile, base_features = _base_inputs(4)
        outputs = model(history, torch.randn(4, 3, 2), None, None, None, None,
                        base_point, base_quantile, base_features)
        torch.testing.assert_close(
            outputs["point_forecast"], base_point + outputs["field_point"]
        )
        (outputs["point_forecast"].square().mean() + outputs["quantile_forecast"].square().mean()).backward()
        for parameter in (model.dgf_point_head.weight, model.ctf_quantile_head.weight, model.fusion_gate.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_base_forecasts_are_required_and_exclusive_with_the_linear_base(self):
        model = _model()
        with self.assertRaises(ValueError):
            model(torch.randn(2, 8, 2), torch.randn(2, 3, 2))
        with self.assertRaises(ValueError):
            _model(linear_base=True)
        with self.assertRaises(ValueError):
            _model(external_base=False, base_inputs=True)


class ExternalBaseDatasetTest(unittest.TestCase):
    def _write_project(self, root: Path) -> Path:
        delivery = pd.date_range("2018-12-01", "2019-02-28 23:00", freq="h", tz="Australia/Brisbane")
        rng = np.random.default_rng(0)
        hour = delivery.hour.to_numpy()
        price = 60 + 30 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 8, len(delivery))
        price[rng.random(len(delivery)) < 0.02] = 900.0
        split = np.where(delivery < pd.Timestamp("2019-02-01", tz="Australia/Brisbane"), "train",
                         np.where(delivery < pd.Timestamp("2019-02-15", tz="Australia/Brisbane"),
                                  "validation", "test"))
        pd.DataFrame({
            "delivery_start_aest": delivery.astype(str),
            "available_at_aest": (delivery + pd.Timedelta(hours=1)).astype(str),
            "region": "NSW1",
            "split": split,
            "rrp_aud_per_mwh": price,
            "total_demand_mw": 7000 + 500 * np.cos(2 * np.pi * hour / 24) + rng.normal(0, 50, len(delivery)),
        }).to_csv(root / "nsw1.csv", index=False)
        config = {
            "data": {
                "timezone": "Australia/Brisbane",
                "delivery_column": "delivery_start_aest",
                "available_column": "available_at_aest",
                "region_column": "region",
                "split_column": "split",
                "history_columns": ["rrp_aud_per_mwh", "total_demand_mw"],
                "target_column": "rrp_aud_per_mwh",
                "price_transform": "asinh",
                "quantile_target": "price",
                "split_names": ["train", "validation", "test"],
                "split_boundaries": {
                    "train": {"start": "2018-12-01 00:00:00", "end_exclusive": "2019-02-01 00:00:00"},
                    "validation": {"start": "2019-02-01 00:00:00", "end_exclusive": "2019-02-15 00:00:00"},
                    "test": {"start": "2019-02-15 00:00:00", "end_exclusive": "2019-03-01 00:00:00"},
                },
                "region_files": {"NSW1": "nsw1.csv"},
            },
            "forecast_protocol": {
                "input_hours": 8, "output_hours": 3, "stride_hours": 1,
                "calendar_features": ["hour_sin", "hour_cos"],
            },
            "model": {
                "num_variables": 2, "num_frequencies": 2, "hidden_dim": 8, "num_layers": 1,
                "freq_cutoff": 2.0, "num_atoms": 2, "sigma_base": 0.05, "scale_warmup_ratio": 0.3,
                "fusion_mode": "additive_trigonometric_gate", "residual_path": True,
                "base_inputs": True, "field_point": False, "quantiles": list(QUANTILES),
            },
            "loss": {},
            "training": {"epochs": 2, "seed": 2026},
            "external_base": {"enabled": True, "directory": "base"},
        }
        configs = root / "configs"
        configs.mkdir()
        # load_forecast_config takes the project root as the config's grandparent.
        path = configs / "config.yaml"
        path.write_text(yaml.safe_dump(config))
        return path

    def test_build_and_read_the_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._write_project(root)
            summary = build(config_path, "NSW1", seed=2026, folds=3, device_name="cpu")
            self.assertEqual(len(summary["rounds_per_horizon"]), 3)
            with np.load(root / "base" / "NSW1.npz") as archive:
                origins = archive["forecast_origin_unix"]
                point = archive["point"]
                quantile = archive["quantile"]
            datasets = build_region_datasets(config_path, "NSW1")
            for split, dataset in datasets.items():
                rows = np.searchsorted(origins, dataset.delivery_unix_seconds[dataset.origin_indices])
                sample = dataset[5]
                torch.testing.assert_close(
                    dataset.denormalize_target(sample["base_point"].double())[..., 0],
                    torch.from_numpy(point[rows[5]]), rtol=1e-4, atol=1e-3,
                )
                torch.testing.assert_close(
                    dataset.denormalize_quantiles(sample["base_quantile"].double()),
                    torch.from_numpy(quantile[rows[5]]), rtol=1e-4, atol=1e-3,
                )

            # The training pipeline passes the base to the model; without
            # field_point the point forecast is the base's.
            config = yaml.safe_load(config_path.read_text())
            config["project_root"] = str(root)
            model = build_model(config)
            batch = {key: value[None] for key, value in datasets["test"][0].items()}
            *inputs, _ = move_inputs(batch, torch.device("cpu"))
            model.initialize_atoms(inputs[0])
            outputs = model(*inputs)
            torch.testing.assert_close(outputs["point_forecast"], batch["base_point"])

            # A configuration with other data cannot read this base.
            other = copy.deepcopy(yaml.safe_load(config_path.read_text()))
            other["data"]["price_cap"] = 650
            config_path.write_text(yaml.safe_dump(other))
            with self.assertRaises(ValueError):
                build_region_datasets(config_path, "NSW1")


if __name__ == "__main__":
    unittest.main()
