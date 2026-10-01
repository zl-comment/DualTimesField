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
        future_exogenous_mode: str = "query",
        exogenous_adapter_hidden_dim: int = 16,
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
        if future_exogenous_mode not in {"query", "post_attention_residual"}:
            raise ValueError(
                "future_exogenous_mode must be 'query' or "
                "'post_attention_residual'"
            )
        if exogenous_adapter_hidden_dim <= 0:
            raise ValueError("exogenous_adapter_hidden_dim must be positive")

        self.summary_mode = summary_mode
        self.future_exogenous_dim = future_exogenous_dim
        self.future_exogenous_mode = future_exogenous_mode
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
            if future_exogenous_dim > 0 and future_exogenous_mode == "query"
            else None
        )
        self.exogenous_adapter = None
        if (
            future_exogenous_dim > 0
            and future_exogenous_mode == "post_attention_residual"
        ):
            self.exogenous_adapter = nn.Sequential(
                nn.Linear(future_exogenous_dim, exogenous_adapter_hidden_dim),
                nn.GELU(),
                nn.Linear(exogenous_adapter_hidden_dim, channels, bias=False),
            )
            nn.init.zeros_(self.exogenous_adapter[-1].weight)
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

        if self.future_exogenous_dim > 0:
            if future_exogenous is None or future_exogenous.shape != (
                history_signal.shape[0],
                future_calendar.shape[1],
                self.future_exogenous_dim,
            ):
                raise ValueError(
                    "future_exogenous must have shape "
                    f"[B, {future_calendar.shape[1]}, {self.future_exogenous_dim}]"
                )
        elif future_exogenous is not None:
            raise ValueError(
                "future_exogenous was provided to a head without exogenous inputs"
            )

        future_projection = self.calendar_projection(future_calendar)
        if self.exogenous_projection is not None:
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
        if self.exogenous_adapter is not None:
            future_hidden = future_hidden + self.exogenous_adapter(
                future_exogenous
            )
        return (
            self.point_output(future_hidden),
            self.quantile_output(future_hidden),
            attention_weights,
        )


class AdaptiveEventField(nn.Module):
    """DGF whose event atoms are located per sample.

    The original DGF places its atoms at centres shared by every sample, so a
    given centre falls on a different clock hour in each rolling window and
    cannot follow where a sample's spikes and troughs actually are.  Here a
    small convolutional encoder of the residual gives each atom an attention
    distribution over the window; the atom's centre is the attention-weighted
    time, its signed amplitude the attention-weighted residual, and its width
    and gate come from the attended features.  The event field is the sum of
    Gaussian bumps at those centres.  ``extract_events`` keeps the DGF
    interface (event signal, gated amplitudes, gates); the last centres,
    widths, and amplitudes are kept for the echo into the forecast horizon.
    """

    def __init__(self, num_variables: int, num_atoms: int = 16, hidden_dim: int = 32,
                 kernel_size: int = 5, min_width: float = 0.5):
        super().__init__()
        padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv1d(num_variables, hidden_dim, kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.GELU(),
        )
        self.locator = nn.Conv1d(hidden_dim, num_atoms, 1)
        self.width_head = nn.Linear(hidden_dim, 1)
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.min_width = min_width
        self.num_atoms = num_atoms
        self.last_centres = None
        self.last_widths = None
        self.last_amplitudes = None

    def extract_events(self, x: torch.Tensor, t: torch.Tensor, sigma_addition: float = 0.0):
        batch, steps, _ = x.shape
        features = self.encoder(x.transpose(1, 2))
        attention = torch.softmax(self.locator(features), dim=-1)
        index = torch.arange(steps, device=x.device, dtype=x.dtype)
        centres = attention @ index
        amplitude_raw = attention @ x
        attended = attention @ features.transpose(1, 2)
        # sigma_addition is in window units, like the original annealed width.
        widths = (
            nn.functional.softplus(self.width_head(attended)[..., 0])
            + self.min_width
            + sigma_addition * (steps - 1)
        )
        gate = torch.sigmoid(5 * self.gate_head(attended)[..., 0])
        amplitude = amplitude_raw * gate.unsqueeze(-1)
        bumps = torch.exp(
            -((index.view(1, 1, -1) - centres.unsqueeze(-1)) ** 2)
            / (2 * widths.unsqueeze(-1) ** 2)
        )
        event_signal = torch.einsum("bkt,bkd->btd", bumps, amplitude)
        self.last_centres, self.last_widths, self.last_amplitudes = centres, widths, amplitude
        return event_signal, amplitude, gate

    def initialize_from_residual(self, residual, t):
        return None

    def echo(self, steps: int, horizon: int, lags=(24, 48)) -> torch.Tensor:
        """Project each located atom forward by whole days into the horizon.

        Returns ``[B, horizon, 2 * len(lags)]``: for each lag, the summed
        positive (spike) and negative (trough) price-channel bumps landing on
        each forecast hour.
        """
        hours = torch.arange(horizon, device=self.last_centres.device, dtype=self.last_centres.dtype)
        price = self.last_amplitudes[..., 0]
        parts = []
        for lag in lags:
            landing = self.last_centres + lag - steps
            bumps = torch.exp(
                -((hours.view(1, 1, -1) - landing.unsqueeze(-1)) ** 2)
                / (2 * self.last_widths.unsqueeze(-1) ** 2)
            )
            parts.append(torch.einsum("bkh,bk->bh", bumps, torch.relu(price)))
            parts.append(torch.einsum("bkh,bk->bh", bumps, torch.relu(-price)))
        return torch.stack(parts, dim=-1)


class DetectedEventField(nn.Module):
    """DGF built from explicitly detected price events.

    Events are found in the price channel relative to a robust baseline (the
    window median): the ``num_events`` largest absolute departures, with
    non-maximum suppression so that detected events are at least
    ``min_separation`` hours apart.  Positions are data, not parameters.
    Each event keeps its departure, soft-thresholded by a learned level so
    small departures vanish, and becomes a Gaussian bump of learned width.
    Because the event field is fixed by the data, the decomposition loss
    makes the CTF fit only what is left, so the two fields separate.
    """

    def __init__(self, num_variables: int, num_events: int = 8, min_separation: int = 3,
                 learn_threshold: bool = True, baseline: str = "median"):
        super().__init__()
        if baseline not in {"median", "mean"}:
            raise ValueError("baseline must be 'median' or 'mean'")
        self.num_variables = num_variables
        self.num_events = num_events
        self.min_separation = min_separation
        self.baseline = baseline
        self.raw_width = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        # With learn_threshold=False the threshold is fixed at zero (every
        # detected departure is kept at full size).
        self.learn_threshold = learn_threshold
        self.raw_threshold = nn.Parameter(
            torch.tensor(math.log(math.expm1(0.5))), requires_grad=learn_threshold
        )
        self.raw_echo_width = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        self.last_centres = None
        self.last_widths = None
        self.last_amplitudes = None

    def detect(self, price: torch.Tensor):
        reference = (
            price.median(dim=1, keepdim=True).values
            if self.baseline == "median"
            else price.mean(dim=1, keepdim=True)
        )
        departure = price - reference
        remaining = departure.abs()
        steps = price.shape[1]
        index = torch.arange(steps, device=price.device)
        positions, values = [], []
        for _ in range(self.num_events):
            position = remaining.argmax(dim=1)
            positions.append(position)
            values.append(departure.gather(1, position.unsqueeze(1))[:, 0])
            near = (index.unsqueeze(0) - position.unsqueeze(1)).abs() < self.min_separation
            remaining = remaining.masked_fill(near, 0.0)
        return torch.stack(positions, dim=1), torch.stack(values, dim=1)

    def extract_events(self, x: torch.Tensor, t: torch.Tensor, sigma_addition: float = 0.0):
        batch, steps, variables = x.shape
        positions, values = self.detect(x[..., 0])
        threshold = (
            nn.functional.softplus(self.raw_threshold)
            if self.learn_threshold
            else x.new_zeros(())
        )
        amplitude_price = torch.sign(values) * torch.relu(values.abs() - threshold)
        width = nn.functional.softplus(self.raw_width) + 0.25
        index = torch.arange(steps, device=x.device, dtype=x.dtype)
        centres = positions.to(x.dtype)
        bumps = torch.exp(-((index.view(1, 1, -1) - centres.unsqueeze(-1)) ** 2) / (2 * width ** 2))
        amplitude = torch.zeros(batch, self.num_events, variables, device=x.device, dtype=x.dtype)
        amplitude[..., 0] = amplitude_price
        event_signal = torch.einsum("bkt,bkd->btd", bumps, amplitude)
        gate = (amplitude_price != 0).to(x.dtype)
        self.last_centres = centres
        self.last_widths = torch.full_like(centres, 1.0) * (nn.functional.softplus(self.raw_echo_width) + 0.25)
        self.last_amplitudes = amplitude
        return event_signal, amplitude, gate

    def initialize_from_residual(self, residual, t):
        return None

    echo = AdaptiveEventField.echo


class FutureEventField(nn.Module):
    """Two-sided event process over the forecast horizon.

    For each forecast hour the target-space price follows a three-component
    Gaussian mixture: the continuous level, a scarcity-driven spike above it,
    and a surplus-driven trough below it.  Spike and trough probabilities and
    shapes come from the hour's forecast drivers and a compact state of
    recent events.  The event state is built from the DGF (atom amplitudes,
    gates, and the last day of the event signal), from the raw last day of
    history, or omitted, so the value of the DGF can be tested directly.
    """

    def __init__(
        self,
        input_length: int,
        num_variables: int,
        num_atoms: int,
        calendar_dim: int,
        spare_dim: int,
        shortfall_dim: int,
        trough_dim: int,
        state_source: str = "dgf",
        state_dim: int = 8,
        state_hours: int = 24,
        spike_rate: float = 0.005,
        trough_rate: float = 0.015,
        recent_extremes: bool = False,
    ):
        super().__init__()
        if state_source not in {"dgf", "raw", "none"}:
            raise ValueError("event state_source must be 'dgf', 'raw', or 'none'")
        self.state_source = state_source
        self.state_hours = min(state_hours, input_length)
        if state_source == "dgf":
            state_input = 2 * num_atoms + self.state_hours
        elif state_source == "raw":
            state_input = self.state_hours * num_variables
        else:
            state_input = 0
        self.state_dim = state_dim if state_input > 0 else 0
        self.state = nn.Linear(state_input, state_dim) if state_input > 0 else None
        # Maximum, minimum, and mean price of the last state_hours, read directly.
        self.recent_extremes = recent_extremes
        extra = self.state_dim + (3 if recent_extremes else 0)
        self.spike_head = nn.Linear(shortfall_dim + spare_dim + calendar_dim + extra, 3)
        self.trough_head = nn.Linear(trough_dim + calendar_dim + extra, 3)
        self.level_scale_head = nn.Linear(
            spare_dim + trough_dim + calendar_dim + extra, 1
        )

        def inverse_softplus(value: float) -> float:
            return math.log(math.expm1(value))

        with torch.no_grad():
            # logit, shift, scale; biases start at plausible event rates and sizes.
            self.spike_head.bias.copy_(torch.tensor(
                [math.log(spike_rate), inverse_softplus(3.0), inverse_softplus(1.0)]
            ))
            self.trough_head.bias.copy_(torch.tensor(
                [math.log(trough_rate), inverse_softplus(1.5), inverse_softplus(0.3)]
            ))
            self.level_scale_head.bias.fill_(inverse_softplus(0.4))

    def forward(
        self,
        level_mean: torch.Tensor,
        future_calendar: torch.Tensor,
        spare: torch.Tensor,
        shortfall: torch.Tensor,
        net_load: torch.Tensor,
        history_values: torch.Tensor,
        event_signal: torch.Tensor,
        amplitude: torch.Tensor,
        gate: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        horizon = future_calendar.shape[1]
        state_parts = []
        if self.state_source == "dgf":
            state_input = torch.cat(
                [amplitude[..., 0], gate, event_signal[:, -self.state_hours:, 0]], dim=1
            )
        elif self.state_source == "raw":
            state_input = history_values[:, -self.state_hours:].flatten(start_dim=1)
        if self.state is not None:
            state = self.state(state_input)
            state_parts = [state.unsqueeze(1).expand(-1, horizon, -1)]
        if self.recent_extremes:
            recent = history_values[:, -self.state_hours:, 0]
            extremes = torch.stack(
                [recent.max(dim=1).values, recent.min(dim=1).values, recent.mean(dim=1)], dim=-1
            )
            state_parts.append(extremes.unsqueeze(1).expand(-1, horizon, -1))
        spike = self.spike_head(torch.cat([shortfall, spare, future_calendar, *state_parts], dim=-1))
        trough = self.trough_head(torch.cat([net_load, future_calendar, *state_parts], dim=-1))
        level_scale = nn.functional.softplus(
            self.level_scale_head(torch.cat([spare, net_load, future_calendar, *state_parts], dim=-1))
        ) + 0.05
        logits = torch.stack(
            [torch.zeros_like(spike[..., 0]), spike[..., 0], trough[..., 0]], dim=-1
        )
        spike_shift = nn.functional.softplus(spike[..., 1]) + 0.1
        trough_shift = nn.functional.softplus(trough[..., 1]) + 0.1
        level = level_mean[..., 0]
        means = torch.stack([level, level + spike_shift, level - trough_shift], dim=-1)
        scales = torch.stack(
            [
                level_scale[..., 0],
                nn.functional.softplus(spike[..., 2]) + 0.05,
                nn.functional.softplus(trough[..., 2]) + 0.05,
            ],
            dim=-1,
        )
        log_weights = torch.log_softmax(logits, dim=-1)
        return {
            "mixture_log_weights": log_weights,
            "mixture_means": means,
            "mixture_scales": scales,
            "event_probability_spike": log_weights[..., 1].exp(),
            "event_probability_trough": log_weights[..., 2].exp(),
        }


class DualFieldLinearForecaster(nn.Module):
    """Forecast from constrained CTF and DGF features."""

    def __init__(
        self,
        num_variables: int,
        input_length: int = 72,
        forecast_horizon: int = 24,
        calendar_dim: int = 10,
        future_exogenous_dim: int = 0,
        future_exogenous_mode: str = "query",
        exogenous_adapter_hidden_dim: int = 16,
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
        residual_path: bool = False,
        origin_context_dim: int = 0,
        ctf_exogenous_dim: int = 0,
        quantile_gate: bool = False,
        quantile_exogenous_dim: int = 0,
        head_input: str = "fields",
        event_field: bool = False,
        event_state_source: str = "dgf",
        event_detach_level: bool = True,
        event_recent_extremes: bool = False,
        dgf_type: str = "gabor",
        event_echo: bool = False,
        echo_modulation: bool = True,
        detected_events: int = 8,
        detected_min_separation: int = 3,
        detected_learn_threshold: bool = True,
        detected_baseline: str = "median",
        linear_base: bool = False,
        field_forecast: bool = True,
    ):
        super().__init__()
        self.num_variables = num_variables
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.calendar_dim = calendar_dim
        self.future_exogenous_dim = future_exogenous_dim
        self.future_exogenous_mode = future_exogenous_mode
        self.quantiles = tuple(quantiles)
        self.fusion_mode = fusion_mode
        self.forecast_head_type = forecast_head_type
        self.residual_path = residual_path
        self.origin_context_dim = origin_context_dim
        self.ctf_exogenous_dim = ctf_exogenous_dim
        self.quantile_gate = quantile_gate
        self.quantile_exogenous_dim = quantile_exogenous_dim
        if head_input not in {"fields", "raw_history"}:
            raise ValueError("head_input must be 'fields' or 'raw_history'")
        # Ablation: "raw_history" feeds the raw history to both expert heads in
        # place of the CTF and DGF fields (and a zero remainder), keeping the
        # heads, routing, gate, parameters, and losses unchanged.
        self.head_input = head_input
        self.event_field = None
        self.event_detach_level = event_detach_level
        if event_field:
            if not (future_exogenous_dim > 0 and quantile_exogenous_dim > 0 and ctf_exogenous_dim == 1):
                raise ValueError(
                    "event_field needs spare capacity (future_exogenous), shortfalls "
                    "(quantile_exogenous), and one net-load CTF exogenous feature"
                )
            self.event_field = FutureEventField(
                input_length=input_length,
                num_variables=num_variables,
                # The detected DGF exposes one amplitude and gate per detected event.
                num_atoms=detected_events if dgf_type == "detected" else num_atoms,
                calendar_dim=calendar_dim,
                spare_dim=future_exogenous_dim,
                shortfall_dim=quantile_exogenous_dim,
                trough_dim=ctf_exogenous_dim,
                state_source=event_state_source,
                recent_extremes=event_recent_extremes,
            )
        if quantile_exogenous_dim > 0 and (
            fusion_mode == "concatenate" or forecast_head_type != "linear"
        ):
            raise ValueError("quantile exogenous inputs require gated linear heads")
        if quantile_gate and fusion_mode == "concatenate":
            raise ValueError("quantile_gate requires a gated fusion mode")
        if (residual_path or origin_context_dim > 0 or ctf_exogenous_dim > 0) and (
            fusion_mode == "concatenate" or forecast_head_type != "linear"
        ):
            raise ValueError(
                "The residual path, origin context and CTF exogenous inputs "
                "require a gated fusion mode with linear heads"
            )
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
        if future_exogenous_mode == "dgf_linear":
            if forecast_head_type != "linear" or fusion_mode == "concatenate":
                raise ValueError(
                    "dgf_linear future exogenous inputs require gated linear heads"
                )
        elif future_exogenous_dim > 0 and forecast_head_type not in {
            "tcn",
            "tcn_attention",
        }:
            raise ValueError(
                "Future exogenous inputs require a TCN forecast head"
            )
        # Forward-looking scarcity inputs reach only the DGF event expert and its gate.
        dgf_exogenous_dim = (
            forecast_horizon * future_exogenous_dim
            if future_exogenous_mode == "dgf_linear"
            else 0
        )
        if dgf_type not in {"gabor", "adaptive", "detected"}:
            raise ValueError("dgf_type must be 'gabor', 'adaptive', or 'detected'")
        if event_echo and dgf_type == "gabor":
            raise ValueError("event_echo needs an adaptive or detected DGF")
        self.dgf_type = dgf_type
        self.event_echo = event_echo
        self.echo_modulation = echo_modulation
        self.echo_lags = (24, 48)
        if event_echo:
            # Echo features reach the DGF heads and the gate, like the scarcity inputs.
            dgf_exogenous_dim += forecast_horizon * 2 * len(self.echo_lags)

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
                    **head_arguments,
                    future_exogenous_dim=(
                        future_exogenous_dim
                        if future_exogenous_mode == "query"
                        else 0
                    ),
                    future_exogenous_mode=future_exogenous_mode,
                    exogenous_adapter_hidden_dim=exogenous_adapter_hidden_dim,
                )
                self.dgf_forecast_head = TemporalConvForecastHead(
                    **head_arguments,
                    future_exogenous_dim=future_exogenous_dim,
                    future_exogenous_mode=future_exogenous_mode,
                    exogenous_adapter_hidden_dim=exogenous_adapter_hidden_dim,
                )
            else:
                # The CTF head also reads the reconstruction remainder, so
                # history not captured by either field still reaches the forecast.
                ctf_feature_dim = (
                    expert_feature_dim
                    + (input_length * num_variables if residual_path else 0)
                    + origin_context_dim
                    + forecast_horizon * ctf_exogenous_dim
                )
                self.ctf_point_head = nn.Linear(
                    ctf_feature_dim, forecast_horizon
                )
                self.dgf_point_head = nn.Linear(
                    expert_feature_dim + dgf_exogenous_dim, forecast_horizon
                )
                self.ctf_quantile_head = nn.Linear(
                    ctf_feature_dim,
                    forecast_horizon * len(self.quantiles),
                )
                self.dgf_quantile_head = nn.Linear(
                    expert_feature_dim
                    + dgf_exogenous_dim
                    + forecast_horizon * quantile_exogenous_dim,
                    forecast_horizon * len(self.quantiles),
                )
            self.fusion_gate = nn.Linear(
                combined_feature_dim + dgf_exogenous_dim, forecast_horizon
            )
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.zeros_(self.fusion_gate.bias)
            if quantile_gate:
                # Each quantile level and horizon rescales and shifts the shared
                # gate logit, so tail quantiles can lean on the event field more
                # than the point forecast does.  Starts equal to the shared gate.
                self.quantile_gate_scale = nn.Parameter(
                    torch.ones(forecast_horizon, len(self.quantiles))
                )
                self.quantile_gate_bias = nn.Parameter(
                    torch.zeros(forecast_horizon, len(self.quantiles))
                )
        if dgf_type == "adaptive":
            self.dual_field.dgf = AdaptiveEventField(num_variables, num_atoms)
        elif dgf_type == "detected":
            self.dual_field.dgf = DetectedEventField(
                num_variables,
                num_events=detected_events,
                min_separation=detected_min_separation,
                learn_threshold=detected_learn_threshold,
                baseline=detected_baseline,
            )
        self.echo_modulator = None
        if event_echo and echo_modulation:
            # Per forecast hour, spike and trough echo strengths from the hour's
            # scarcity, surplus, and calendar inputs.
            self.echo_modulator = nn.Linear(
                future_exogenous_dim + quantile_exogenous_dim + ctf_exogenous_dim + calendar_dim, 2
            )
        # Linear base: one linear map of the raw history and all known future
        # inputs (the inputs of the linear_mse baseline), to which the dual-field
        # forecast is added as a residual.  Created last so that every other
        # parameter keeps its initialization.
        if not field_forecast and not linear_base:
            raise ValueError("field_forecast=False needs linear_base")
        self.linear_base = linear_base
        self.field_forecast = field_forecast
        if linear_base:
            base_dim = (
                input_length * num_variables
                + forecast_horizon * (calendar_dim + future_exogenous_dim + ctf_exogenous_dim)
                + origin_context_dim
            )
            self.base_point_head = nn.Linear(base_dim, forecast_horizon)
            self.base_quantile_head = nn.Linear(base_dim, forecast_horizon * len(self.quantiles))

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
        remainder: torch.Tensor | None = None,
        origin_context: torch.Tensor | None = None,
        ctf_exogenous: torch.Tensor | None = None,
        quantile_exogenous: torch.Tensor | None = None,
        echo_features: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        ctf_flat = ctf_signal.flatten(start_dim=1)
        event_flat = event_signal.flatten(start_dim=1)
        calendar_flat = future_calendar.flatten(start_dim=1)
        dgf_extra = []
        if self.future_exogenous_mode == "dgf_linear":
            dgf_extra = [future_exogenous.flatten(start_dim=1)]
        if echo_features is not None:
            dgf_extra.append(echo_features.flatten(start_dim=1))
        gate_features = torch.cat(
            [ctf_flat, event_flat, calendar_flat, *dgf_extra], dim=1
        )

        ctf_attention = None
        dgf_attention = None
        if self.forecast_head_type in {"tcn", "tcn_attention"}:
            ctf_point, ctf_quantile, ctf_attention = self.ctf_forecast_head(
                ctf_signal,
                future_calendar,
                (
                    future_exogenous
                    if self.future_exogenous_mode == "query"
                    else None
                ),
            )
            dgf_point, dgf_quantile, dgf_attention = self.dgf_forecast_head(
                event_signal, future_calendar, future_exogenous
            )
        else:
            ctf_parts = [ctf_flat, calendar_flat]
            if remainder is not None:
                ctf_parts.insert(1, remainder.flatten(start_dim=1))
            if origin_context is not None:
                ctf_parts.append(origin_context)
            if ctf_exogenous is not None:
                ctf_parts.append(ctf_exogenous.flatten(start_dim=1))
            ctf_features = torch.cat(ctf_parts, dim=1)
            dgf_features = torch.cat([event_flat, calendar_flat, *dgf_extra], dim=1)
            ctf_point = self.ctf_point_head(ctf_features).unsqueeze(-1)
            dgf_point = self.dgf_point_head(dgf_features).unsqueeze(-1)
            ctf_quantile = self.ctf_quantile_head(ctf_features).view(
                ctf_signal.shape[0],
                self.forecast_horizon,
                len(self.quantiles),
            )
            dgf_quantile_features = dgf_features
            if quantile_exogenous is not None:
                dgf_quantile_features = torch.cat(
                    [dgf_features, quantile_exogenous.flatten(start_dim=1)], dim=1
                )
            dgf_quantile = self.dgf_quantile_head(dgf_quantile_features).view(
                ctf_signal.shape[0],
                self.forecast_horizon,
                len(self.quantiles),
            )
        ctf_quantile = torch.sort(ctf_quantile, dim=-1).values
        dgf_quantile = torch.sort(dgf_quantile, dim=-1).values

        gate_logit = self.fusion_gate(gate_features)
        fusion_angle = 0.5 * math.pi * torch.sigmoid(gate_logit)
        dgf_weight = torch.sin(fusion_angle).square().unsqueeze(-1)
        if self.quantile_gate:
            quantile_angle = 0.5 * math.pi * torch.sigmoid(
                gate_logit.unsqueeze(-1) * self.quantile_gate_scale
                + self.quantile_gate_bias
            )
            dgf_quantile_weight = torch.sin(quantile_angle).square()
        else:
            quantile_angle = fusion_angle.unsqueeze(-1)
            dgf_quantile_weight = dgf_weight
        if self.fusion_mode == "additive_trigonometric_gate":
            ctf_weight = torch.ones_like(dgf_weight)
            point_forecast = ctf_point + dgf_weight * dgf_point
            quantile_forecast = ctf_quantile + dgf_quantile_weight * dgf_quantile
        else:
            ctf_weight = torch.cos(fusion_angle).square().unsqueeze(-1)
            point_forecast = ctf_weight * ctf_point + dgf_weight * dgf_point
            quantile_forecast = (
                torch.cos(quantile_angle).square() * ctf_quantile
                + dgf_quantile_weight * dgf_quantile
            )
        if self.quantile_gate:
            # Different weights per level can cross; keep quantiles ordered.
            quantile_forecast = torch.sort(quantile_forecast, dim=-1).values

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
            "dgf_quantile_fusion_weight": dgf_quantile_weight,
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
        origin_context: torch.Tensor | None = None,
        ctf_exogenous: torch.Tensor | None = None,
        quantile_exogenous: torch.Tensor | None = None,
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
        if self.origin_context_dim > 0:
            expected = (history_values.shape[0], self.origin_context_dim)
            if origin_context is None or origin_context.shape != expected:
                raise ValueError(f"origin_context must have shape {expected}")
        elif origin_context is not None:
            raise ValueError("origin_context was provided but the model has no origin context")
        if self.ctf_exogenous_dim > 0:
            expected = (
                history_values.shape[0],
                self.forecast_horizon,
                self.ctf_exogenous_dim,
            )
            if ctf_exogenous is None or ctf_exogenous.shape != expected:
                raise ValueError(f"ctf_exogenous must have shape {expected}")
        elif ctf_exogenous is not None:
            raise ValueError("ctf_exogenous was provided but the model has no CTF exogenous inputs")
        if self.quantile_exogenous_dim > 0:
            expected = (
                history_values.shape[0],
                self.forecast_horizon,
                self.quantile_exogenous_dim,
            )
            if quantile_exogenous is None or quantile_exogenous.shape != expected:
                raise ValueError(f"quantile_exogenous must have shape {expected}")
        elif quantile_exogenous is not None:
            raise ValueError("quantile_exogenous was provided but the model has no quantile exogenous inputs")

        history_time = self._history_time(history_values)
        ctf_signal = self.dual_field.ctf(history_values, history_time)
        residual = history_values - ctf_signal

        eta = self.dual_field.scale_scheduler.get_eta(
            self.dual_field.current_epoch
        )
        sigma_addition = eta * self.dual_field.scale_scheduler.sigma_base
        # The detected DGF finds events in the history itself; the CTF then
        # fits what is left through the decomposition loss.
        event_source = history_values if self.dgf_type == "detected" else residual
        event_signal, amplitude, gate = self.dual_field.dgf.extract_events(
            event_source, history_time, sigma_addition
        )
        echo_features = None
        if self.event_echo:
            echo = self.dual_field.dgf.echo(self.input_length, self.forecast_horizon, self.echo_lags)
            if self.echo_modulator is not None:
                drivers = [
                    part for part in (future_exogenous, quantile_exogenous, ctf_exogenous)
                    if part is not None
                ]
                strength = torch.sigmoid(
                    self.echo_modulator(torch.cat([*drivers, future_calendar], dim=-1))
                )
                # Channels alternate spike, trough for each lag.
                echo = echo * strength.repeat(1, 1, len(self.echo_lags))
            echo_features = echo

        if self.fusion_mode == "concatenate":
            forecasts = self._concatenated_forecast(
                ctf_signal, event_signal, future_calendar
            )
        else:
            remainder = (
                history_values - ctf_signal - event_signal
                if self.residual_path
                else None
            )
            ctf_input, event_input = ctf_signal, event_signal
            if self.head_input == "raw_history":
                ctf_input, event_input = history_values, history_values
                remainder = torch.zeros_like(history_values) if self.residual_path else None
            forecasts = self._trigonometric_gated_forecast(
                ctf_input,
                event_input,
                future_calendar,
                future_exogenous,
                remainder,
                origin_context,
                ctf_exogenous,
                quantile_exogenous,
                echo_features,
            )
            if echo_features is not None:
                forecasts["event_echo"] = echo_features

        if self.linear_base:
            parts = [history_values, future_calendar, future_exogenous, ctf_exogenous]
            base_features = torch.cat(
                [part.flatten(start_dim=1) for part in parts if part is not None]
                + ([origin_context] if origin_context is not None else []),
                dim=1,
            )
            base_point = self.base_point_head(base_features).unsqueeze(-1)
            base_quantile = self.base_quantile_head(base_features).view(
                -1, self.forecast_horizon, len(self.quantiles)
            )
            forecasts["base_point"] = base_point
            if self.field_forecast:
                forecasts["field_point"] = forecasts["point_forecast"]
                forecasts["point_forecast"] = base_point + forecasts["point_forecast"]
                base_quantile = base_quantile + forecasts["quantile_forecast"]
            else:
                # Control: the linear base alone, trained in the dual-field
                # pipeline; the fields are still fitted by the decomposition loss.
                forecasts["point_forecast"] = base_point
            forecasts["quantile_forecast"] = torch.sort(base_quantile, dim=-1).values

        if self.event_field is not None:
            level_mean = forecasts["point_forecast"]
            if self.event_detach_level:
                level_mean = level_mean.detach()
            forecasts.update(
                self.event_field(
                    level_mean,
                    future_calendar,
                    future_exogenous,
                    quantile_exogenous,
                    ctf_exogenous,
                    history_values,
                    event_signal,
                    amplitude,
                    gate,
                )
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
