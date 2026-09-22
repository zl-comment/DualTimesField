import math
from typing import Dict, Sequence

import torch
import torch.nn as nn

from src.dualfield.core import DualTimesField


class CausalResidualBlock(nn.Module):
    """Two causal dilated convolutions with a residual connection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.activation = nn.GELU()

    @staticmethod
    def _trim_future(output: torch.Tensor, input_length: int) -> torch.Tensor:
        return output[..., :input_length]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_length = inputs.shape[-1]
        hidden = self._trim_future(self.conv1(inputs), sequence_length)
        hidden = self.activation(hidden)
        hidden = self._trim_future(self.conv2(hidden), sequence_length)
        return self.activation(hidden + inputs)


class TemporalConvForecastHead(nn.Module):
    """Encode history with a TCN and decode each future calendar step."""

    def __init__(
        self,
        num_variables: int,
        calendar_dim: int,
        num_quantiles: int,
        future_exogenous_dim: int = 0,
        channels: int = 40,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8, 16),
        summary_mode: str = "last",
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size <= 1:
            raise ValueError("kernel_size must be greater than one")
        if not dilations or any(dilation <= 0 for dilation in dilations):
            raise ValueError("dilations must contain positive integers")
        if summary_mode not in {"last", "attention"}:
            raise ValueError("summary_mode must be 'last' or 'attention'")
        if future_exogenous_dim < 0:
            raise ValueError("future_exogenous_dim must be non-negative")

        self.summary_mode = summary_mode
        self.future_exogenous_dim = future_exogenous_dim
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilations)
        self.input_projection = nn.Conv1d(num_variables, channels, 1)
        self.temporal_blocks = nn.Sequential(
            *[
                CausalResidualBlock(channels, kernel_size, dilation)
                for dilation in dilations
            ]
        )
        self.calendar_projection = nn.Linear(calendar_dim, channels)
        self.exogenous_projection = (
            nn.Linear(future_exogenous_dim, channels, bias=False)
            if future_exogenous_dim > 0
            else None
        )
        if summary_mode == "attention":
            self.query_projection = nn.Linear(channels, channels, bias=False)
            self.key_projection = nn.Linear(channels, channels, bias=False)
            self.value_projection = nn.Linear(channels, channels, bias=False)
            self.attention_scale = channels ** -0.5
        self.future_fusion = nn.Linear(2 * channels, channels)
        self.activation = nn.GELU()
        self.point_output = nn.Linear(channels, 1)
        self.quantile_output = nn.Linear(channels, num_quantiles)

    def forward(
        self,
        history_signal: torch.Tensor,
        future_calendar: torch.Tensor,
        future_exogenous: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        temporal = history_signal.transpose(1, 2)
        temporal = self.input_projection(temporal)
        temporal = self.temporal_blocks(temporal)

        future_projection = self.calendar_projection(future_calendar)
        if self.exogenous_projection is not None:
            if future_exogenous is None or future_exogenous.shape != (
                history_signal.shape[0],
                future_calendar.shape[1],
                self.future_exogenous_dim,
            ):
                raise ValueError(
                    "future_exogenous must have shape "
                    f"[B, {future_calendar.shape[1]}, {self.future_exogenous_dim}]"
                )
            future_projection = future_projection + self.exogenous_projection(
                future_exogenous
            )
        future_context = self.activation(future_projection)
        attention_weights = None
        if self.summary_mode == "attention":
            temporal_sequence = temporal.transpose(1, 2)
            query = self.query_projection(future_context)
            key = self.key_projection(temporal_sequence)
            value = self.value_projection(temporal_sequence)
            attention_scores = torch.matmul(
                query, key.transpose(1, 2)
            ) * self.attention_scale
            attention_weights = torch.softmax(attention_scores, dim=-1)
            history_context = torch.matmul(attention_weights, value)
        else:
            history_context = temporal[..., -1].unsqueeze(1).expand(
                -1, future_calendar.shape[1], -1
            )
        future_hidden = self.activation(
            self.future_fusion(
                torch.cat([history_context, future_context], dim=-1)
            )
        )
        return (
            self.point_output(future_hidden),
            self.quantile_output(future_hidden),
            attention_weights,
        )


class DualFieldLinearForecaster(nn.Module):
    """Forecast from constrained CTF and DGF features."""

    def __init__(
        self,
        num_variables: int,
        input_length: int = 72,
        forecast_horizon: int = 24,
        calendar_dim: int = 10,
        future_exogenous_dim: int = 0,
        quantiles: Sequence[float] = (0.05, 0.10, 0.50, 0.90, 0.95),
        num_frequencies: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 3,
        freq_cutoff: float = 10.0,
        num_atoms: int = 16,
        sigma_base: float = 0.05,
        sparsity_lambda: float = 0.001,
        smoothness_lambda: float = 0.001,
        fusion_mode: str = "concatenate",
        forecast_head_type: str = "linear",
        tcn_channels: int = 40,
        tcn_kernel_size: int = 3,
        tcn_dilations: Sequence[int] = (1, 2, 4, 8, 16),
    ):
        super().__init__()
        self.num_variables = num_variables
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.calendar_dim = calendar_dim
        self.future_exogenous_dim = future_exogenous_dim
        self.quantiles = tuple(quantiles)
        self.fusion_mode = fusion_mode
        self.forecast_head_type = forecast_head_type
        if fusion_mode not in {
            "concatenate",
            "trigonometric_gate",
            "additive_trigonometric_gate",
        }:
            raise ValueError(
                "fusion_mode must be 'concatenate', 'trigonometric_gate', "
                "or 'additive_trigonometric_gate'"
            )
        if forecast_head_type not in {"linear", "tcn", "tcn_attention"}:
            raise ValueError(
                "forecast_head_type must be 'linear', 'tcn', "
                "or 'tcn_attention'"
            )
        if fusion_mode == "concatenate" and forecast_head_type != "linear":
            raise ValueError(
                "TCN heads require a gated fusion mode with separate fields"
            )
        if future_exogenous_dim > 0 and forecast_head_type not in {
            "tcn",
            "tcn_attention",
        }:
            raise ValueError(
                "Future exogenous inputs require a TCN forecast head"
            )

        self.dual_field = DualTimesField(
            num_variables=num_variables,
            seq_length=input_length,
            num_frequencies=num_frequencies,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            freq_cutoff=freq_cutoff,
            num_atoms=num_atoms,
            sigma_base=sigma_base,
            sparsity_lambda=sparsity_lambda,
            smoothness_lambda=smoothness_lambda,
        )

        combined_feature_dim = (
            2 * input_length * num_variables
            + forecast_horizon * calendar_dim
        )
        if fusion_mode == "concatenate":
            self.point_head = nn.Linear(combined_feature_dim, forecast_horizon)
            self.quantile_head = nn.Linear(
                combined_feature_dim, forecast_horizon * len(self.quantiles)
            )
        else:
            expert_feature_dim = (
                input_length * num_variables
                + forecast_horizon * calendar_dim
            )
            if forecast_head_type in {"tcn", "tcn_attention"}:
                head_arguments = {
                    "num_variables": num_variables,
                    "calendar_dim": calendar_dim,
                    "num_quantiles": len(self.quantiles),
                    "future_exogenous_dim": future_exogenous_dim,
                    "channels": tcn_channels,
                    "kernel_size": tcn_kernel_size,
                    "dilations": tuple(tcn_dilations),
                    "summary_mode": (
                        "attention"
                        if forecast_head_type == "tcn_attention"
                        else "last"
                    ),
                }
                self.ctf_forecast_head = TemporalConvForecastHead(
                    **head_arguments
                )
                self.dgf_forecast_head = TemporalConvForecastHead(
                    **head_arguments
                )
            else:
                self.ctf_point_head = nn.Linear(
                    expert_feature_dim, forecast_horizon
                )
                self.dgf_point_head = nn.Linear(
                    expert_feature_dim, forecast_horizon
                )
                self.ctf_quantile_head = nn.Linear(
                    expert_feature_dim,
                    forecast_horizon * len(self.quantiles),
                )
                self.dgf_quantile_head = nn.Linear(
                    expert_feature_dim,
                    forecast_horizon * len(self.quantiles),
                )
            self.fusion_gate = nn.Linear(combined_feature_dim, forecast_horizon)
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.zeros_(self.fusion_gate.bias)

    def _concatenated_forecast(
        self,
        ctf_signal: torch.Tensor,
        event_signal: torch.Tensor,
        future_calendar: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        features = torch.cat(
            [
                ctf_signal.flatten(start_dim=1),
                event_signal.flatten(start_dim=1),
                future_calendar.flatten(start_dim=1),
            ],
            dim=1,
        )
        point_forecast = self.point_head(features).unsqueeze(-1)
        quantile_forecast = self.quantile_head(features).view(
            ctf_signal.shape[0],
            self.forecast_horizon,
            len(self.quantiles),
        )
        return {
            "point_forecast": point_forecast,
            "quantile_forecast": torch.sort(
                quantile_forecast, dim=-1
            ).values,
        }

    def _trigonometric_gated_forecast(
        self,
        ctf_signal: torch.Tensor,
        event_signal: torch.Tensor,
        future_calendar: torch.Tensor,
        future_exogenous: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor]:
        ctf_flat = ctf_signal.flatten(start_dim=1)
        event_flat = event_signal.flatten(start_dim=1)
        calendar_flat = future_calendar.flatten(start_dim=1)
        gate_features = torch.cat(
            [ctf_flat, event_flat, calendar_flat], dim=1
        )

        ctf_attention = None
        dgf_attention = None
        if self.forecast_head_type in {"tcn", "tcn_attention"}:
            ctf_point, ctf_quantile, ctf_attention = self.ctf_forecast_head(
                ctf_signal, future_calendar, future_exogenous
            )
            dgf_point, dgf_quantile, dgf_attention = self.dgf_forecast_head(
                event_signal, future_calendar, future_exogenous
            )
        else:
            ctf_features = torch.cat([ctf_flat, calendar_flat], dim=1)
            dgf_features = torch.cat([event_flat, calendar_flat], dim=1)
            ctf_point = self.ctf_point_head(ctf_features).unsqueeze(-1)
            dgf_point = self.dgf_point_head(dgf_features).unsqueeze(-1)
            ctf_quantile = self.ctf_quantile_head(ctf_features).view(
                ctf_signal.shape[0],
                self.forecast_horizon,
                len(self.quantiles),
            )
            dgf_quantile = self.dgf_quantile_head(dgf_features).view(
                ctf_signal.shape[0],
                self.forecast_horizon,
                len(self.quantiles),
            )
        ctf_quantile = torch.sort(ctf_quantile, dim=-1).values
        dgf_quantile = torch.sort(dgf_quantile, dim=-1).values

        fusion_angle = 0.5 * math.pi * torch.sigmoid(
            self.fusion_gate(gate_features)
        )
        dgf_weight = torch.sin(fusion_angle).square().unsqueeze(-1)
        if self.fusion_mode == "additive_trigonometric_gate":
            ctf_weight = torch.ones_like(dgf_weight)
            point_forecast = ctf_point + dgf_weight * dgf_point
            quantile_forecast = ctf_quantile + dgf_weight * dgf_quantile
        else:
            ctf_weight = torch.cos(fusion_angle).square().unsqueeze(-1)
            point_forecast = ctf_weight * ctf_point + dgf_weight * dgf_point
            quantile_forecast = (
                ctf_weight * ctf_quantile + dgf_weight * dgf_quantile
            )

        forecasts = {
            "point_forecast": point_forecast,
            "quantile_forecast": quantile_forecast,
            "ctf_expert_point": ctf_point,
            "dgf_expert_point": dgf_point,
            "ctf_expert_quantile": ctf_quantile,
            "dgf_expert_quantile": dgf_quantile,
            "fusion_angle": fusion_angle.unsqueeze(-1),
            "ctf_fusion_weight": ctf_weight,
            "dgf_fusion_weight": dgf_weight,
        }
        if ctf_attention is not None:
            forecasts.update(
                {
                    "ctf_history_attention": ctf_attention,
                    "dgf_history_attention": dgf_attention,
                }
            )
        return forecasts

    def _history_time(self, history_values: torch.Tensor) -> torch.Tensor:
        return torch.linspace(
            0.0,
            1.0,
            steps=self.input_length,
            device=history_values.device,
            dtype=history_values.dtype,
        )

    def forward(
        self,
        history_values: torch.Tensor,
        future_calendar: torch.Tensor,
        future_exogenous: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if history_values.shape[1:] != (self.input_length, self.num_variables):
            raise ValueError(
                "history_values must have shape "
                f"[B, {self.input_length}, {self.num_variables}]"
            )
        if future_calendar.shape[0] != history_values.shape[0] or (
            future_calendar.shape[1:] != (self.forecast_horizon, self.calendar_dim)
        ):
            raise ValueError(
                "future_calendar must have shape "
                f"[B, {self.forecast_horizon}, {self.calendar_dim}]"
            )
        if self.future_exogenous_dim > 0:
            expected = (
                history_values.shape[0],
                self.forecast_horizon,
                self.future_exogenous_dim,
            )
            if future_exogenous is None or future_exogenous.shape != expected:
                raise ValueError(f"future_exogenous must have shape {expected}")
        elif future_exogenous is not None:
            raise ValueError("future_exogenous was provided but the model has no exogenous inputs")

        history_time = self._history_time(history_values)
        ctf_signal = self.dual_field.ctf(history_values, history_time)
        residual = history_values - ctf_signal

        eta = self.dual_field.scale_scheduler.get_eta(
            self.dual_field.current_epoch
        )
        sigma_addition = eta * self.dual_field.scale_scheduler.sigma_base
        event_signal, amplitude, gate = self.dual_field.dgf.extract_events(
            residual, history_time, sigma_addition
        )

        if self.fusion_mode == "concatenate":
            forecasts = self._concatenated_forecast(
                ctf_signal, event_signal, future_calendar
            )
        else:
            forecasts = self._trigonometric_gated_forecast(
                ctf_signal, event_signal, future_calendar, future_exogenous
            )

        return {
            **forecasts,
            "ctf_signal": ctf_signal,
            "event_signal": event_signal,
            "event_amplitude": amplitude,
            "event_gate": gate,
        }

    def set_epoch(self, epoch: int) -> None:
        self.dual_field.set_epoch(epoch)

    def initialize_atoms(self, history_values: torch.Tensor) -> None:
        self.dual_field.initialize_atoms(
            history_values, self._history_time(history_values)
        )
