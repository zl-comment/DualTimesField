from typing import Dict, Sequence

import torch
import torch.nn as nn

from src.dualfield.core import DualTimesField


class DualFieldLinearForecaster(nn.Module):
    """Forecast from constrained CTF and DGF features with linear heads only."""

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
    ):
        super().__init__()
        self.num_variables = num_variables
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.calendar_dim = calendar_dim
        self.quantiles = tuple(quantiles)

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

        feature_dim = (
            2 * input_length * num_variables
            + forecast_horizon * calendar_dim
        )
        self.point_head = nn.Linear(feature_dim, forecast_horizon)
        self.quantile_head = nn.Linear(
            feature_dim, forecast_horizon * len(self.quantiles)
        )

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
            history_values.shape[0],
            self.forecast_horizon,
            len(self.quantiles),
        )
        quantile_forecast = torch.sort(quantile_forecast, dim=-1).values

        return {
            "point_forecast": point_forecast,
            "quantile_forecast": quantile_forecast,
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
