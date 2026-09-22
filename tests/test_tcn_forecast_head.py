import unittest

import torch

from forecasting.models import (
    DualFieldLinearForecaster,
    TemporalConvForecastHead,
)


class TemporalConvForecastHeadTest(unittest.TestCase):
    def test_head_preserves_horizon_and_covers_history(self):
        head = TemporalConvForecastHead(
            num_variables=2,
            calendar_dim=3,
            num_quantiles=5,
            channels=8,
            dilations=(1, 2, 4, 8, 16),
        )
        point, quantile, attention = head(
            torch.randn(4, 72, 2),
            torch.randn(4, 24, 3),
        )

        self.assertEqual(head.receptive_field, 125)
        self.assertEqual(point.shape, (4, 24, 1))
        self.assertEqual(quantile.shape, (4, 24, 5))
        self.assertIsNone(attention)

    def test_attention_summary_is_horizon_specific_and_normalized(self):
        head = TemporalConvForecastHead(
            num_variables=2,
            calendar_dim=3,
            num_quantiles=5,
            channels=8,
            dilations=(1, 2),
            summary_mode="attention",
        )
        point, quantile, attention = head(
            torch.randn(4, 12, 2),
            torch.randn(4, 5, 3),
        )

        self.assertEqual(point.shape, (4, 5, 1))
        self.assertEqual(quantile.shape, (4, 5, 5))
        self.assertEqual(attention.shape, (4, 5, 12))
        torch.testing.assert_close(
            attention.sum(dim=-1),
            torch.ones(4, 5),
        )

    def test_tcn_heads_preserve_additive_fusion_and_gradients(self):
        torch.manual_seed(23)
        model = DualFieldLinearForecaster(
            num_variables=2,
            input_length=8,
            forecast_horizon=3,
            calendar_dim=2,
            quantiles=(0.1, 0.5, 0.9),
            num_frequencies=2,
            hidden_dim=8,
            num_layers=1,
            freq_cutoff=2.0,
            num_atoms=2,
            fusion_mode="additive_trigonometric_gate",
            forecast_head_type="tcn",
            tcn_channels=8,
            tcn_dilations=(1, 2),
        )
        history = torch.randn(4, 8, 2)
        calendar = torch.randn(4, 3, 2)
        model.initialize_atoms(history)
        outputs = model(history, calendar)

        self.assertEqual(outputs["point_forecast"].shape, (4, 3, 1))
        self.assertEqual(outputs["quantile_forecast"].shape, (4, 3, 3))
        torch.testing.assert_close(
            outputs["point_forecast"],
            outputs["ctf_expert_point"]
            + outputs["dgf_fusion_weight"] * outputs["dgf_expert_point"],
        )
        self.assertTrue(
            torch.all(
                outputs["quantile_forecast"][..., 1:]
                >= outputs["quantile_forecast"][..., :-1]
            )
        )

        outputs["point_forecast"].square().mean().backward()
        for head in (model.ctf_forecast_head, model.dgf_forecast_head):
            self.assertGreater(
                head.input_projection.weight.grad.abs().sum().item(), 0.0
            )
            self.assertGreater(
                head.point_output.weight.grad.abs().sum().item(), 0.0
            )
        self.assertGreater(model.fusion_gate.weight.grad.abs().sum().item(), 0.0)

    def test_attention_tcn_exposes_both_field_attention_maps(self):
        torch.manual_seed(29)
        model = DualFieldLinearForecaster(
            num_variables=2,
            input_length=8,
            forecast_horizon=3,
            calendar_dim=2,
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
        history = torch.randn(4, 8, 2)
        calendar = torch.randn(4, 3, 2)
        model.initialize_atoms(history)
        outputs = model(history, calendar)

        self.assertEqual(outputs["ctf_history_attention"].shape, (4, 3, 8))
        self.assertEqual(outputs["dgf_history_attention"].shape, (4, 3, 8))
        outputs["point_forecast"].square().mean().backward()
        for head in (model.ctf_forecast_head, model.dgf_forecast_head):
            self.assertGreater(
                head.query_projection.weight.grad.abs().sum().item(), 0.0
            )
            self.assertGreater(
                head.key_projection.weight.grad.abs().sum().item(), 0.0
            )
            self.assertGreater(
                head.value_projection.weight.grad.abs().sum().item(), 0.0
            )


if __name__ == "__main__":
    unittest.main()
