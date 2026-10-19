"""Joint differentiable correction of immutable chronological tree forecasts."""
import math

import torch
from torch import nn

from ..full_v5.adaptive_dual_history import AdaptiveHistoricalDualField

LEVELS = (.05, .1, .5, .9, .95)


def pinball(prediction, actual):
    levels = prediction.new_tensor(LEVELS)
    error = actual[..., None] - prediction
    return torch.maximum(levels * error, (levels - 1) * error).mean()


class StateEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        # Preserve ordered 72h observations rather than replacing them by an average.
        self.history = nn.Sequential(nn.Flatten(1), nn.Linear(72 * 12, width), nn.GELU())
        self.future = nn.Sequential(nn.Linear(31, width), nn.GELU())
        self.merge = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.LayerNorm(width))

    def forward(self, history, future):
        past = self.history(history)[:, None].expand(-1, 24, -1)
        return self.merge(torch.cat((past, self.future(future)), dim=-1))


class JointHistoryModel(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.state = StateEncoder(width)
        self.calibration = nn.Sequential(nn.Linear(width + 8, width), nn.GELU(), nn.Linear(width, 2))
        nn.init.zeros_(self.calibration[-1].weight)
        nn.init.zeros_(self.calibration[-1].bias)
        # Fresh geometry: never load an encoder fitted on later dates.
        self.dual = AdaptiveHistoricalDualField(fit_cutoff=0)
        self.match = nn.Sequential(nn.Linear(3 * width + 1, width), nn.GELU(), nn.Linear(width, 2))
        self.alignment_query = nn.Linear(width, width, bias=False)
        self.alignment_key = nn.Linear(width, width, bias=False)
        self.gates = nn.Sequential(nn.Linear(width + 8 + 2, width), nn.GELU(), nn.Linear(width, 2))
        nn.init.constant_(self.gates[-1].bias, -3.)
        time = torch.arange(24, dtype=torch.float32)
        fourier = torch.stack((torch.ones_like(time), (2 * math.pi * time / 24).cos(),
                               (2 * math.pi * time / 24).sin()), dim=-1)
        self.register_buffer('lowpass', fourier @ torch.linalg.pinv(fourier))
        self.register_buffer('time_distance', ((time[:, None] - time[None]) / 6).square())

    def forward(self, batch, *, history_off=False, event_off=False, time_shift_off=False):
        b = batch
        n, k = b['mask'].shape
        current = self.state(b['history'], b['future'])
        memory = self.state(b['memory_history'].flatten(0, 1), b['memory_future'].flatten(0, 1)).reshape(n, k, 24, -1)
        calibration = self.calibration(torch.cat((current, b['stats']), dim=-1))
        location = 150 * calibration[..., 0].tanh()
        scale = 1 + .5 * calibration[..., 1].tanh()
        baseline = b['baseline']
        anchor = baseline[..., 2:3] + location[..., None] + scale[..., None] * (baseline - baseline[..., 2:3])
        fields = self.dual(b['memory_actual'].flatten(0, 1), b['memory_z'].flatten(0, 1),
                           b['memory_lags'].flatten(0, 1))
        trend = fields['trend'].reshape(n, k, 24) * 300
        event = fields['event'].reshape(n, k, 24) * 300
        historical_q50 = b['memory_baseline'][..., 2]
        trend_error = trend - torch.einsum('ij,bkj->bki', self.lowpass, historical_q50)
        # Only transfer positive events that the historical OOF base actually missed.
        missed = (b['memory_actual'] - historical_q50 - trend_error.clamp_min(0)).clamp_min(0)
        event_error = torch.minimum(event.clamp_min(0), missed)
        query = current[:, None].expand(-1, k, -1, -1)
        age = torch.log1p(b['age'])[:, :, None, None].expand(-1, -1, 24, 1) / math.log(731)
        scores = self.match(torch.cat((query, memory, query - memory, age), dim=-1))
        scores = scores.masked_fill(~b['mask'][:, :, None, None], -torch.inf)
        # A zero-valued null memory makes rejection possible, including empty memory.
        scores = torch.cat((scores, scores.new_zeros((n, 1, 24, 2))), dim=1)
        weights = scores.softmax(dim=1)[:, :k]
        weight_trend, weight_event = weights.unbind(dim=-1)
        proposal_trend = (weight_trend * (trend_error - location[:, None])).sum(dim=1)
        if time_shift_off:
            moved = event_error
        else:
            alignment = torch.einsum('bhd,bksd->bkhs', self.alignment_query(current),
                                     self.alignment_key(memory)) / math.sqrt(current.shape[-1])
            # Each source event distributes its mass across target hours; no circular wrap.
            alignment = (alignment - self.time_distance[None, None]).softmax(dim=2)
            moved = torch.einsum('bkhs,bks->bkh', alignment, event_error)
        proposal_event = (weight_event * moved).sum(dim=1)
        gate_input = torch.cat((current, b['stats'], proposal_trend[..., None] / 300,
                                proposal_event[..., None] / 300), dim=-1)
        gates = self.gates(gate_input).sigmoid()
        correction_trend = gates[..., 0] * proposal_trend
        correction_event = gates[..., 1] * proposal_event
        if event_off:
            correction_event = torch.zeros_like(correction_event)
        correction = 150 * torch.tanh((correction_trend + correction_event) / 150)
        if history_off:
            correction = torch.zeros_like(correction)
        prediction = anchor + correction[..., None]
        reconstructed = fields['reconstruction'].reshape(n, k, 24)
        rec_error = (reconstructed - b['memory_actual'] / 300).square().mean(dim=-1)
        reconstruction = (rec_error * b['mask']).sum() / b['mask'].sum().clamp_min(1)
        return dict(prediction=prediction, anchor=anchor, correction=correction,
                    trend=correction_trend, event=correction_event, gates=gates,
                    weights=weights, reconstruction_loss=reconstruction)


def training_loss(output, targets, *, reconstruction_weight=.05, gate_penalty=.001):
    # Price normalization is a common constant, not an alternate target transform.
    prediction = pinball(output['prediction'], targets) / 300
    return prediction + reconstruction_weight * output['reconstruction_loss'] + gate_penalty * output['gates'].mean()
