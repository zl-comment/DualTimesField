import unittest

import torch

from forecasting.losses import DualFieldForecastLoss
from forecasting.models import DualFieldLinearForecaster
from forecasting.train import (
    ExponentialMovingAverage,
    build_optimizer,
    configure_fine_tuning_epoch,
    validation_selection_values,
)


class StableFineTuningTest(unittest.TestCase):
    def test_blended_point_loss_combines_huber_and_mse(self):
        criterion = DualFieldForecastLoss(
            quantiles=(0.1, 0.5, 0.9),
            point_loss_type="blended_huber_mse",
            huber_delta=1.0,
            mse_fraction=0.5,
        )
        history = torch.zeros(2, 4, 1)
        target = torch.full((2, 3, 1), 2.0)
        outputs = {
            "point_forecast": torch.zeros_like(target),
            "quantile_forecast": torch.zeros(2, 3, 3),
            "ctf_signal": torch.zeros_like(history),
            "event_signal": torch.zeros_like(history),
            "event_gate": torch.zeros(2, 2),
        }

        losses = criterion(outputs, history, target)

        self.assertAlmostEqual(losses["point"].item(), 2.75, places=6)

    def test_ema_uses_average_and_restores_training_parameters(self):
        layer = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            layer.weight.zero_()
        ema = ExponentialMovingAverage(layer, decay=0.5)
        with torch.no_grad():
            layer.weight.fill_(2.0)
        ema.update(layer)

        with ema.average_parameters(layer):
            self.assertAlmostEqual(layer.weight.item(), 1.0)
        self.assertAlmostEqual(layer.weight.item(), 2.0)

    def test_progressive_groups_unfreeze_at_configured_epochs(self):
        model = DualFieldLinearForecaster(
            num_variables=2,
            input_length=8,
            forecast_horizon=3,
            calendar_dim=2,
            future_exogenous_dim=1,
            future_exogenous_mode="post_attention_residual",
            quantiles=(0.1, 0.5, 0.9),
            num_frequencies=2,
            hidden_dim=8,
            num_layers=1,
            freq_cutoff=2.0,
            num_atoms=2,
            fusion_mode="additive_trigonometric_gate",
            forecast_head_type="tcn_attention",
            tcn_channels=8,
            tcn_dilations=(1, 2),
        )
        training = {
            "epochs": 120,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "optimizer": {"betas": (0.9, 0.95), "eps": 1e-8},
            "fine_tuning": {
                "enabled": True,
                "schedule": {"warmup_epochs": 5, "min_lr_ratio": 0.01},
                "parameter_groups": {
                    "adapter": {"start_epoch": 0, "learning_rate": 3e-4},
                    "decoder": {"start_epoch": 10, "learning_rate": 5e-5},
                    "temporal": {"start_epoch": 60, "learning_rate": 5e-6},
                    "fusion": {"start_epoch": 60, "learning_rate": 5e-6},
                },
            },
        }
        optimizer, groups = build_optimizer(model, training)

        stage_0, rates_0, count_0 = configure_fine_tuning_epoch(
            optimizer, groups, training, 0
        )
        stage_10, rates_10, count_10 = configure_fine_tuning_epoch(
            optimizer, groups, training, 10
        )
        stage_60, rates_60, count_60 = configure_fine_tuning_epoch(
            optimizer, groups, training, 60
        )

        self.assertEqual(stage_0, "adapter")
        self.assertEqual(stage_10, "adapter+decoder")
        self.assertEqual(stage_60, "adapter+decoder+temporal+fusion")
        self.assertLess(count_0, count_10)
        self.assertLess(count_10, count_60)
        self.assertAlmostEqual(rates_0["adapter"], 6e-5)
        self.assertEqual(rates_0["decoder"], 0.0)
        self.assertAlmostEqual(rates_10["decoder"], 1e-5)
        self.assertAlmostEqual(rates_60["temporal"], 1e-6)

    def test_composite_selection_is_relative_to_warm_start(self):
        validation = {
            "losses": {"total": 2.0},
            "metrics": {
                "mae_aud_per_mwh": 90.0,
                "rmse_aud_per_mwh": 220.0,
            },
        }
        values = validation_selection_values(
            validation, {"mae": 100.0, "rmse": 200.0}
        )

        self.assertAlmostEqual(values["composite"], 1.0)


if __name__ == "__main__":
    unittest.main()
