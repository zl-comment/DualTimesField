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
        point_loss_type="mse",
        huber_delta=1.0,
        mse_fraction=0.5,
        event_nll_weight=0.0,
        event_anchor_weight=0.0,
        spike_threshold=300.0,
        trough_threshold=0.0,
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
        if point_loss_type not in {"mse", "blended_huber_mse"}:
            raise ValueError(
                "point_loss_type must be 'mse' or 'blended_huber_mse'"
            )
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if not 0.0 <= mse_fraction <= 1.0:
            raise ValueError("mse_fraction must be between zero and one")
        self.point_loss_type = point_loss_type
        self.huber_delta = float(huber_delta)
        self.mse_fraction = float(mse_fraction)
        self.event_nll_weight = float(event_nll_weight)
        self.event_anchor_weight = float(event_anchor_weight)
        self.spike_threshold = float(spike_threshold)
        self.trough_threshold = float(trough_threshold)

    def forward(self, outputs, history_values, target_price, quantile_target=None, target_price_raw=None):
        point_forecast = outputs["point_forecast"]
        quantile_forecast = outputs["quantile_forecast"]
        ctf_signal = outputs["ctf_signal"]
        event_signal = outputs["event_signal"]
        event_gate = outputs["event_gate"]

        if quantile_forecast.shape[-1] != self.quantiles.numel():
            raise ValueError(
                "quantile_forecast's final dimension must match configured quantiles"
            )

        mse_loss = F.mse_loss(point_forecast, target_price)
        if self.point_loss_type == "mse":
            point_loss = mse_loss
        else:
            huber_loss = F.huber_loss(
                point_forecast,
                target_price,
                delta=self.huber_delta,
            )
            point_loss = (
                self.mse_fraction * mse_loss
                + (1.0 - self.mse_fraction) * huber_loss
            )

        if quantile_target is None:
            quantile_target = target_price
        quantile_error = quantile_target - quantile_forecast
        quantiles = self.quantiles.to(
            device=quantile_forecast.device,
            dtype=quantile_forecast.dtype,
        )
        quantile_loss = torch.maximum(
            quantiles * quantile_error,
            (quantiles - 1.0) * quantile_error,
        ).mean()

        decomposition_target = outputs.get("field_history", history_values)
        decomposition_loss = F.mse_loss(ctf_signal + event_signal, decomposition_target)
        smoothness_loss = ((ctf_signal[:, 1:] - ctf_signal[:, :-1]) ** 2).mean()

        mean_gate = event_gate.mean()
        sparsity_loss = mean_gate + self.sparsity_excess_weight * F.relu(
            mean_gate - self.target_sparsity
        )

        event_nll = point_loss.new_zeros(())
        event_anchor = point_loss.new_zeros(())
        if "mixture_log_weights" in outputs:
            log_weights = outputs["mixture_log_weights"]
            components = torch.distributions.Normal(
                outputs["mixture_means"], outputs["mixture_scales"]
            )
            log_density = components.log_prob(target_price[..., :1].expand_as(log_weights))
            event_nll = -torch.logsumexp(log_weights + log_density, dim=-1).mean()
            if target_price_raw is not None:
                raw = target_price_raw[..., 0]
                spike = (raw > self.spike_threshold).to(log_weights.dtype)
                trough = (raw < self.trough_threshold).to(log_weights.dtype)
                not_spike = torch.logsumexp(log_weights[..., [0, 2]], dim=-1)
                not_trough = torch.logsumexp(log_weights[..., [0, 1]], dim=-1)
                event_anchor = -(
                    spike * log_weights[..., 1] + (1 - spike) * not_spike
                    + trough * log_weights[..., 2] + (1 - trough) * not_trough
                ).mean()

        total_loss = (
            self.point_weight * point_loss
            + self.quantile_weight * quantile_loss
            + self.decomposition_weight * decomposition_loss
            + self.smoothness_weight * smoothness_loss
            + self.sparsity_weight * sparsity_loss
            + self.event_nll_weight * event_nll
            + self.event_anchor_weight * event_anchor
        )

        return {
            "total": total_loss,
            "point": point_loss,
            "quantile": quantile_loss,
            "decomposition": decomposition_loss,
            "smoothness": smoothness_loss,
            "sparsity": sparsity_loss,
            "event_nll": event_nll,
            "event_anchor": event_anchor,
        }
