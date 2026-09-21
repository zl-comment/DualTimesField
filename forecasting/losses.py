import torch
import torch.nn as nn
import torch.nn.functional as F


class DualFieldForecastLoss(nn.Module):
    """Forecast objective with auxiliary dual-field decomposition constraints."""

    def __init__(
        self,
        quantiles,
        point_weight=1.0,
        quantile_weight=1.0,
        decomposition_weight=1.0,
        smoothness_weight=0.001,
        sparsity_weight=0.001,
        target_sparsity=0.3,
        sparsity_excess_weight=10.0,
    ):
        super().__init__()
        quantile_tensor = torch.as_tensor(quantiles, dtype=torch.float32)
        if quantile_tensor.ndim != 1 or quantile_tensor.numel() == 0:
            raise ValueError("quantiles must be a non-empty one-dimensional sequence")
        if not torch.all((quantile_tensor > 0) & (quantile_tensor < 1)):
            raise ValueError("quantiles must be strictly between zero and one")
        if quantile_tensor.numel() > 1 and not torch.all(
            quantile_tensor[1:] > quantile_tensor[:-1]
        ):
            raise ValueError("quantiles must be strictly increasing")

        self.register_buffer("quantiles", quantile_tensor)
        self.point_weight = float(point_weight)
        self.quantile_weight = float(quantile_weight)
        self.decomposition_weight = float(decomposition_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.sparsity_weight = float(sparsity_weight)
        self.target_sparsity = float(target_sparsity)
        self.sparsity_excess_weight = float(sparsity_excess_weight)

    @staticmethod
    def _reduce_per_sample(values, sample_weight=None):
        per_sample = values.reshape(values.shape[0], -1).mean(dim=1)
        if sample_weight is None:
            return per_sample.mean()
        weights = sample_weight.reshape(-1).to(
            device=values.device,
            dtype=values.dtype,
        )
        if weights.shape[0] != per_sample.shape[0]:
            raise ValueError("sample_weight must contain one value per batch item")
        if torch.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("sample_weight must be non-negative with a positive sum")
        return (per_sample * weights).sum() / weights.sum()

    def forward(self, outputs, history_values, target_price, sample_weight=None):
        point_forecast = outputs["point_forecast"]
        quantile_forecast = outputs["quantile_forecast"]
        ctf_signal = outputs["ctf_signal"]
        event_signal = outputs["event_signal"]
        event_gate = outputs["event_gate"]

        if quantile_forecast.shape[-1] != self.quantiles.numel():
            raise ValueError(
                "quantile_forecast's final dimension must match configured quantiles"
            )

        point_loss = self._reduce_per_sample(
            (point_forecast - target_price).square(), sample_weight
        )

        quantile_error = target_price - quantile_forecast
        quantiles = self.quantiles.to(
            device=quantile_forecast.device,
            dtype=quantile_forecast.dtype,
        )
        quantile_loss = self._reduce_per_sample(torch.maximum(
            quantiles * quantile_error,
            (quantiles - 1.0) * quantile_error,
        ), sample_weight)

        decomposition_loss = self._reduce_per_sample(
            (ctf_signal + event_signal - history_values).square(), sample_weight
        )
        smoothness_loss = self._reduce_per_sample(
            (ctf_signal[:, 1:] - ctf_signal[:, :-1]).square(), sample_weight
        )

        mean_gate = self._reduce_per_sample(event_gate, sample_weight)
        sparsity_loss = mean_gate + self.sparsity_excess_weight * F.relu(
            mean_gate - self.target_sparsity
        )

        total_loss = (
            self.point_weight * point_loss
            + self.quantile_weight * quantile_loss
            + self.decomposition_weight * decomposition_loss
            + self.smoothness_weight * smoothness_loss
            + self.sparsity_weight * sparsity_loss
        )

        return {
            "total": total_loss,
            "point": point_loss,
            "quantile": quantile_loss,
            "decomposition": decomposition_loss,
            "smoothness": smoothness_loss,
            "sparsity": sparsity_loss,
        }
