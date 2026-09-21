import unittest

import torch

from forecasting.losses import DualFieldForecastLoss


def _outputs(point_values):
    point = torch.tensor(point_values, dtype=torch.float32).reshape(2, 1, 1)
    return {
        "point_forecast": point,
        "quantile_forecast": point.clone(),
        "ctf_signal": torch.zeros(2, 2, 1),
        "event_signal": torch.zeros(2, 2, 1),
        "event_gate": torch.zeros(2, 1),
    }


class TemporalWeightingLossTest(unittest.TestCase):
    def test_uniform_weights_match_unweighted_loss(self):
        criterion = DualFieldForecastLoss(
            quantiles=[0.5],
            decomposition_weight=0.0,
            smoothness_weight=0.0,
            sparsity_weight=0.0,
        )
        outputs = _outputs([1.0, 3.0])
        history = torch.zeros(2, 2, 1)
        target = torch.zeros(2, 1, 1)

        unweighted = criterion(outputs, history, target)
        weighted = criterion(
            outputs,
            history,
            target,
            sample_weight=torch.ones(2),
        )

        for name in unweighted:
            torch.testing.assert_close(weighted[name], unweighted[name])

    def test_recent_sample_can_receive_more_loss_weight(self):
        criterion = DualFieldForecastLoss(
            quantiles=[0.5],
            quantile_weight=0.0,
            decomposition_weight=0.0,
            smoothness_weight=0.0,
            sparsity_weight=0.0,
        )
        outputs = _outputs([1.0, 3.0])
        history = torch.zeros(2, 2, 1)
        target = torch.zeros(2, 1, 1)

        losses = criterion(
            outputs,
            history,
            target,
            sample_weight=torch.tensor([1.0, 3.0]),
        )

        torch.testing.assert_close(losses["point"], torch.tensor(7.0))


if __name__ == "__main__":
    unittest.main()
