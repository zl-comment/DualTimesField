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

    def test_future_exogenous_changes_queries_and_receives_gradients(self):
        torch.manual_seed(31)
        head = TemporalConvForecastHead(
            num_variables=2,
            calendar_dim=3,
            future_exogenous_dim=1,
            num_quantiles=5,
            channels=8,
            dilations=(1, 2),
            summary_mode="attention",
        )
        history = torch.randn(4, 12, 2)
        calendar = torch.randn(4, 5, 3)
        exogenous = torch.randn(4, 5, 1, requires_grad=True)
        point, _, attention = head(history, calendar, exogenous)

        self.assertEqual(attention.shape, (4, 5, 12))
        point.square().mean().backward()
        self.assertGreater(exogenous.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            head.exogenous_projection.weight.grad.abs().sum().item(), 0.0
        )

    def test_post_attention_adapter_starts_as_exact_zero_residual(self):
        torch.manual_seed(41)
        head = TemporalConvForecastHead(
            num_variables=2,
            calendar_dim=3,
            future_exogenous_dim=1,
            future_exogenous_mode="post_attention_residual",
            num_quantiles=5,
            channels=8,
            dilations=(1, 2),
            summary_mode="attention",
        )
        history = torch.randn(4, 12, 2)
        calendar = torch.randn(4, 5, 3)
        exogenous_a = torch.randn(4, 5, 1)
        exogenous_b = torch.randn(4, 5, 1)

        point_a, quantile_a, attention_a = head(
            history, calendar, exogenous_a
        )
        point_b, quantile_b, attention_b = head(
            history, calendar, exogenous_b
        )

        torch.testing.assert_close(point_a, point_b)
        torch.testing.assert_close(quantile_a, quantile_b)
        torch.testing.assert_close(attention_a, attention_b)
        point_a.square().mean().backward()
        self.assertGreater(
            head.exogenous_adapter[-1].weight.grad.abs().sum().item(), 0.0
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

    def test_attention_model_accepts_one_future_exogenous_factor(self):
        torch.manual_seed(37)
        model = DualFieldLinearForecaster(
            num_variables=2,
            input_length=8,
            forecast_horizon=3,
            calendar_dim=2,
            future_exogenous_dim=1,
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
        exogenous = torch.randn(4, 3, 1)
        model.initialize_atoms(history)
        outputs = model(history, calendar, exogenous)

        self.assertEqual(outputs["point_forecast"].shape, (4, 3, 1))
        self.assertEqual(outputs["quantile_forecast"].shape, (4, 3, 3))
        self.assertEqual(outputs["ctf_history_attention"].shape, (4, 3, 8))

    def test_post_attention_factor_is_attached_only_to_dgf_head(self):
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

        self.assertEqual(model.ctf_forecast_head.future_exogenous_dim, 0)
        self.assertIsNone(model.ctf_forecast_head.exogenous_adapter)
        self.assertEqual(model.dgf_forecast_head.future_exogenous_dim, 1)
        self.assertIsNotNone(model.dgf_forecast_head.exogenous_adapter)


if __name__ == "__main__":
    unittest.main()
