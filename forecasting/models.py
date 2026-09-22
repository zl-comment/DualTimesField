import math
from typing import Dict, Sequence

import torch
import torch.nn as nn

from src.dualfield.core import DualTimesField


class DualFieldLinearForecaster(nn.Module):
    """Forecast from constrained CTF and DGF features with compact heads."""

    def __init__(
        self,
        num_variables: int,
        input_length: int = 72,
        forecast_horizon: int = 24,
        calendar_dim: int = 10,
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
        forecast_head_hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_variables = num_variables
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.calendar_dim = calendar_dim
        self.quantiles = tuple(quantiles)
        self.fusion_mode = fusion_mode
        self.forecast_head_type = forecast_head_type
        self.forecast_head_hidden_dim = forecast_head_hidden_dim
        if fusion_mode not in {
            "concatenate",
            "trigonometric_gate",
            "additive_trigonometric_gate",
        }:
            raise ValueError(
                "fusion_mode must be 'concatenate', 'trigonometric_gate', "
                "or 'additive_trigonometric_gate'"
            )
        if forecast_head_type not in {"linear", "nonlinear"}:
            raise ValueError(
                "forecast_head_type must be 'linear' or 'nonlinear'"
            )
        if forecast_head_hidden_dim <= 0:
            raise ValueError("forecast_head_hidden_dim must be positive")

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
            self.point_head = self._build_forecast_head(
                combined_feature_dim, forecast_horizon
            )
            self.quantile_head = self._build_forecast_head(
                combined_feature_dim,
                forecast_horizon * len(self.quantiles),
            )
        else:
            expert_feature_dim = (
                input_length * num_variables
                + forecast_horizon * calendar_dim
            )
            self.ctf_point_head = self._build_forecast_head(
                expert_feature_dim, forecast_horizon
            )
            self.dgf_point_head = self._build_forecast_head(
                expert_feature_dim, forecast_horizon
            )
            self.ctf_quantile_head = self._build_forecast_head(
                expert_feature_dim,
                forecast_horizon * len(self.quantiles),
            )
            self.dgf_quantile_head = self._build_forecast_head(
                expert_feature_dim,
                forecast_horizon * len(self.quantiles),
            )
            self.fusion_gate = nn.Linear(combined_feature_dim, forecast_horizon)
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.zeros_(self.fusion_gate.bias)

    def _build_forecast_head(
        self, input_dim: int, output_dim: int
    ) -> nn.Module:
        if self.forecast_head_type == "linear":
            return nn.Linear(input_dim, output_dim)
        return nn.Sequential(
            nn.Linear(input_dim, self.forecast_head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.forecast_head_hidden_dim, output_dim),
        )

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
    ) -> Dict[str, torch.Tensor]:
        ctf_flat = ctf_signal.flatten(start_dim=1)
        event_flat = event_signal.flatten(start_dim=1)
        calendar_flat = future_calendar.flatten(start_dim=1)
        ctf_features = torch.cat([ctf_flat, calendar_flat], dim=1)
        dgf_features = torch.cat([event_flat, calendar_flat], dim=1)
        gate_features = torch.cat(
            [ctf_flat, event_flat, calendar_flat], dim=1
        )

        ctf_point = self.ctf_point_head(ctf_features).unsqueeze(-1)
        dgf_point = self.dgf_point_head(dgf_features).unsqueeze(-1)
        ctf_quantile = self.ctf_quantile_head(ctf_features).view(
            ctf_signal.shape[0], self.forecast_horizon, len(self.quantiles)
        )
        dgf_quantile = self.dgf_quantile_head(dgf_features).view(
            ctf_signal.shape[0], self.forecast_horizon, len(self.quantiles)
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

        return {
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
                ctf_signal, event_signal, future_calendar
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
