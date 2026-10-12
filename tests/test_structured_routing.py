import unittest

import torch

from forecasting.losses import DualFieldForecastLoss
from forecasting.models import DualFieldLinearForecaster
from forecasting.train import validation_selection_values


class StructuredRoutingTest(unittest.TestCase):
    def model(self, **kwargs):
        return DualFieldLinearForecaster(
            num_variables=2, input_length=8, forecast_horizon=3,
            calendar_dim=2, future_exogenous_dim=2,
            future_exogenous_mode="dgf_linear", ctf_exogenous_dim=1,
            quantile_exogenous_dim=1, num_frequencies=2, hidden_dim=8,
            num_layers=1, freq_cutoff=2, num_atoms=2,
            fusion_mode="additive_trigonometric_gate", **kwargs,
        )

    def inputs(self):
        return (torch.randn(4, 8, 2), torch.randn(4, 3, 2),
                torch.randn(4, 3, 2), None, torch.randn(4, 3, 1),
                torch.randn(4, 3, 1))

    def test_default_preserves_parameters_and_predictions(self):
        torch.manual_seed(9)
        legacy = self.model()
        torch.manual_seed(9)
        explicit = self.model(scarcity_router="concat", event_forecast=True)
        self.assertEqual(list(legacy.state_dict()), list(explicit.state_dict()))
        for name, value in legacy.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
        inputs = self.inputs()
        torch.testing.assert_close(legacy(*inputs)["point_forecast"], explicit(*inputs)["point_forecast"])

    def test_learned_routing_uses_future_drivers_and_independent_heads(self):
        model = self.model(scarcity_router="learned", quantile_gate=True)
        inputs = self.inputs()
        with torch.no_grad():
            # Make the point gate depend specifically on the first external
            # driver, and the quantile gates constant but distinct by level.
            model.scarcity_point_router[0].weight.zero_()
            model.scarcity_point_router[0].bias.zero_()
            model.scarcity_point_router[0].weight[0, 2] = 1
            model.scarcity_point_router[-1].weight.zero_()
            model.scarcity_point_router[-1].weight[0, 0] = 1
            model.scarcity_quantile_router[-1].bias.copy_(torch.linspace(-1, 1, 5))
        before = model(*inputs)
        changed = list(inputs)
        changed[2] = changed[2] + 5
        after = model(*changed)
        self.assertFalse(torch.allclose(before["dgf_fusion_weight"], after["dgf_fusion_weight"]))
        torch.testing.assert_close(before["dgf_quantile_fusion_weight"], after["dgf_quantile_fusion_weight"])
        self.assertFalse(torch.allclose(before["dgf_quantile_fusion_weight"][..., 0], before["dgf_quantile_fusion_weight"][..., -1]))
        self.assertTrue((after["quantile_forecast"][..., 1:] >= after["quantile_forecast"][..., :-1]).all())
        (after["point_forecast"].square().mean() + after["quantile_forecast"].square().mean()).backward()
        for router in [model.scarcity_point_router, model.scarcity_quantile_router]:
            self.assertGreater(router[-1].weight.grad.abs().sum().item(), 0)

    def test_uniform_routing_ignores_gate_parameters(self):
        model = self.model(scarcity_router="uniform", quantile_gate=True)
        with torch.no_grad():
            model.fusion_gate.bias.fill_(10)
            model.quantile_gate_bias.fill_(-10)
        output = model(*self.inputs())
        for key in ["dgf_fusion_weight", "dgf_quantile_fusion_weight"]:
            torch.testing.assert_close(output[key], torch.full_like(output[key], .5))

    def test_event_ablation_removes_predictions_but_preserves_decomposition(self):
        model = self.model(scarcity_router="learned", quantile_gate=True)
        inputs = self.inputs()
        enabled = model(*inputs)
        model.event_forecast = False
        disabled = model(*inputs)
        torch.testing.assert_close(disabled["point_forecast"], disabled["ctf_expert_point"])
        torch.testing.assert_close(disabled["quantile_forecast"], disabled["ctf_expert_quantile"])
        torch.testing.assert_close(disabled["event_signal"], enabled["event_signal"])
        self.assertEqual(disabled["dgf_fusion_weight"].count_nonzero().item(), 0)

    def test_event_ablation_has_no_indirect_path_through_remainder_or_base(self):
        from unittest.mock import patch
        model = self.model(scarcity_router="learned", quantile_gate=True,
                           event_forecast=False, residual_path=True,
                           linear_base=True, calibrator_hidden=4)
        inputs = self.inputs()
        before = model(*inputs)
        original_extract = model.dual_field.dgf.extract_events
        def changed_events(*args, **kwargs):
            signal, amplitude, gate = original_extract(*args, **kwargs)
            return signal + 100, amplitude, gate
        with patch.object(model.dual_field.dgf, "extract_events", side_effect=changed_events):
            after = model(*inputs)
        self.assertFalse(torch.allclose(before["event_signal"], after["event_signal"]))
        for key in ["point_forecast", "quantile_forecast"]:
            torch.testing.assert_close(before[key], after[key])


class TailObjectiveTest(unittest.TestCase):
    def example(self):
        history = torch.zeros(1, 3, 1)
        quantile = torch.zeros(1, 3, 3, requires_grad=True)
        output = {"point_forecast": history, "quantile_forecast": quantile,
                  "ctf_signal": history, "event_signal": history,
                  "event_gate": history}
        return output, history, torch.ones_like(history)

    def test_raw_masks_leave_main_pinball_unchanged(self):
        output, history, target = self.example()
        raw = torch.tensor([[[-1.], [100.], [301.]]])
        plain = DualFieldForecastLoss([.05, .5, .95])
        weighted = DualFieldForecastLoss([.05, .5, .95], tail_weight=.5,
                                        negative_tail_weight=2, spike_tail_weight=4)
        base = plain(output, history, target, target_price_raw=raw)
        result = weighted(output, history, target, target_price_raw=raw)
        torch.testing.assert_close(result["quantile"], base["quantile"])
        torch.testing.assert_close(result["tail"], torch.tensor(1.))
        torch.testing.assert_close(result["total"] - base["total"], torch.tensor(.5))
        result["total"].backward()
        gradient = output["quantile_forecast"].grad.abs().sum(-1).squeeze()
        torch.testing.assert_close(gradient / gradient[1], torch.tensor([2., 1., 3.]))

    def test_no_events_has_zero_auxiliary_and_missing_raw_fails(self):
        output, history, target = self.example()
        criterion = DualFieldForecastLoss([.05, .5, .95], tail_weight=1)
        result = criterion(output, history, target, target_price_raw=target * 100)
        self.assertEqual(result["tail"].item(), 0)
        self.assertTrue(torch.isfinite(result["total"]))
        with self.assertRaisesRegex(ValueError, "target_price_raw"):
            criterion(output, history, target)

    def test_selection_uses_unweighted_price_crps(self):
        validation = {"losses": {"total": 100}, "metrics": {
            "mae_aud_per_mwh": 20, "rmse_aud_per_mwh": 30,
            "crps_quantile_approx": 12,
        }}
        self.assertEqual(validation_selection_values(validation)["crps"], 12)


if __name__ == "__main__":
    unittest.main()
