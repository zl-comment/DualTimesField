import unittest

import torch

from forecasting.models import DualFieldLinearForecaster


class TrigonometricFusionTest(unittest.TestCase):
    def _model(
        self,
        fusion_mode="trigonometric_gate",
        forecast_head_type="linear",
    ):
        return DualFieldLinearForecaster(
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
            fusion_mode=fusion_mode,
            forecast_head_type=forecast_head_type,
            forecast_head_hidden_dim=8,
        )

    def test_initial_gate_is_equal_convex_fusion(self):
        torch.manual_seed(7)
        model = self._model()
        history = torch.randn(4, 8, 2)
        calendar = torch.randn(4, 3, 2)
        model.initialize_atoms(history)
        outputs = model(history, calendar)

        self.assertEqual(outputs["point_forecast"].shape, (4, 3, 1))
        self.assertEqual(outputs["quantile_forecast"].shape, (4, 3, 3))
        torch.testing.assert_close(
            outputs["ctf_fusion_weight"],
            torch.full((4, 3, 1), 0.5),
        )
        torch.testing.assert_close(
            outputs["ctf_fusion_weight"] + outputs["dgf_fusion_weight"],
            torch.ones(4, 3, 1),
        )
        self.assertTrue(
            torch.all(
                outputs["quantile_forecast"][..., 1:]
                >= outputs["quantile_forecast"][..., :-1]
            )
        )

    def test_gate_and_both_experts_receive_gradients(self):
        torch.manual_seed(11)
        model = self._model()
        history = torch.randn(4, 8, 2)
        calendar = torch.randn(4, 3, 2)
        model.initialize_atoms(history)
        outputs = model(history, calendar)
        loss = outputs["point_forecast"].square().mean()
        loss.backward()

        for parameter in (
            model.fusion_gate.weight,
            model.ctf_point_head.weight,
            model.dgf_point_head.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_additive_gate_keeps_ctf_and_adds_dgf_correction(self):
        torch.manual_seed(13)
        model = self._model(fusion_mode="additive_trigonometric_gate")
        history = torch.randn(4, 8, 2)
        calendar = torch.randn(4, 3, 2)
        model.initialize_atoms(history)
        outputs = model(history, calendar)

        torch.testing.assert_close(
            outputs["ctf_fusion_weight"],
            torch.ones(4, 3, 1),
        )
        torch.testing.assert_close(
            outputs["dgf_fusion_weight"],
            torch.full((4, 3, 1), 0.5),
        )
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

    def test_additive_gate_cannot_starve_ctf_gradient(self):
        torch.manual_seed(17)
        model = self._model(fusion_mode="additive_trigonometric_gate")
        with torch.no_grad():
            model.fusion_gate.bias.fill_(-100.0)
        history = torch.randn(4, 8, 2)
        calendar = torch.randn(4, 3, 2)
        model.initialize_atoms(history)
        outputs = model(history, calendar)
        loss = outputs["point_forecast"].square().mean()
        loss.backward()

        self.assertGreater(model.ctf_point_head.weight.grad.abs().sum().item(), 0.0)
        self.assertLess(outputs["dgf_fusion_weight"].max().item(), 1e-6)

    def test_nonlinear_heads_preserve_additive_interface_and_gradients(self):
        torch.manual_seed(19)
        model = self._model(
            fusion_mode="additive_trigonometric_gate",
            forecast_head_type="nonlinear",
        )
        history = torch.randn(4, 8, 2)
        calendar = torch.randn(4, 3, 2)
        model.initialize_atoms(history)
        outputs = model(history, calendar)

        self.assertIsInstance(model.ctf_point_head, torch.nn.Sequential)
        self.assertIsInstance(model.ctf_point_head[1], torch.nn.GELU)
        self.assertEqual(outputs["point_forecast"].shape, (4, 3, 1))
        self.assertEqual(outputs["quantile_forecast"].shape, (4, 3, 3))
        loss = outputs["point_forecast"].square().mean()
        loss.backward()
        for head in (model.ctf_point_head, model.dgf_point_head):
            self.assertGreater(head[0].weight.grad.abs().sum().item(), 0.0)
            self.assertGreater(head[2].weight.grad.abs().sum().item(), 0.0)

    def test_default_mode_preserves_concatenation_interface(self):
        model = self._model(fusion_mode="concatenate")
        history = torch.randn(2, 8, 2)
        calendar = torch.randn(2, 3, 2)
        model.initialize_atoms(history)
        outputs = model(history, calendar)

        self.assertEqual(outputs["point_forecast"].shape, (2, 3, 1))
        self.assertNotIn("dgf_fusion_weight", outputs)


if __name__ == "__main__":
    unittest.main()
