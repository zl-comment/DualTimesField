"""State evidence bounds the three Anchor outputs, without choosing their sign."""
import math
import torch

from ..full_v5_calibration_robustness.model import BoundedCalibrationModel

VARIANTS = ('bounded_reference', 'constant_evidence', 'state_evidence')


class EvidenceCalibrationModel(BoundedCalibrationModel):
    def __init__(self, *, constant=False):
        super().__init__()
        self.constant = constant
        self.register_buffer('thresholds', torch.zeros(6, dtype=torch.float64))
        self.register_buffer('evidence_gates', torch.zeros(6, 3))
        self.register_buffer('constant_gates', torch.zeros(3))
        self.register_buffer('policy_fitted', torch.tensor(False))

    def set_policy(self, policy):
        self.thresholds.copy_(torch.as_tensor(policy['thresholds'], dtype=self.thresholds.dtype, device=self.thresholds.device))
        self.evidence_gates.copy_(torch.as_tensor(policy['gates'], device=self.evidence_gates.device))
        self.constant_gates.copy_(torch.as_tensor(policy['constant_gates'], device=self.constant_gates.device))
        if not torch.isfinite(self.thresholds).all() or not torch.isfinite(self.evidence_gates).all():
            raise ValueError('Nonfinite policy')
        if (self.evidence_gates < 0).any() or (self.evidence_gates > 1).any() or self.evidence_gates[5].any():
            raise ValueError('Invalid evidence gates')
        if not torch.isfinite(self.constant_gates).all() or (self.constant_gates < 0).any() or (self.constant_gates > 1).any():
            raise ValueError('Invalid constant gates')
        self.policy_fitted.fill_(True)

    def constraint(self, x):
        if not self.policy_fitted:
            raise ValueError('Training policy must be fitted or restored before prediction')
        if x.ndim != 3 or x.shape[1:] != (24, 6) or not torch.isfinite(x).all():
            raise ValueError('Invalid mechanism inputs')
        if not ((x[..., 4:] == 0) | (x[..., 4:] == 1)).all():
            raise ValueError('Coverage must be binary')
        spare20, spare50, load20, load50, import20, export20 = self.thresholds.unbind()
        spare, load, imp, exp, pc, ic = x.unbind(-1)
        state = torch.zeros_like(spare, dtype=torch.long)
        tight = (spare < spare20) & (load >= load50)
        surplus = (load < load20) & (spare >= spare50)
        state = torch.where(tight, 1, state)
        state = torch.where(tight & (imp < import20), 2, state)
        state = torch.where(surplus, 3, state)
        state = torch.where(surplus & (exp < export20), 4, state)
        state = torch.where((pc < .5) | (ic < .5), 5, state)
        gate = self.evidence_gates[state]
        if self.constant:
            gate = torch.where((state == 5)[..., None], torch.zeros_like(gate), self.constant_gates.expand_as(gate))
        return state, gate

    def calibrated(self, baseline, features, gate):
        raw = self.calibration(features)
        location = gate[..., 0] * 30 * torch.tanh(150 * raw[..., 0].tanh() / 30)
        lower = 1 + gate[..., 1] * .1 * torch.tanh(.5 * raw[..., 1].tanh() / .1)
        upper = 1 + gate[..., 2] * .1 * torch.tanh(.5 * raw[..., 2].tanh() / .1)
        scale = torch.stack((lower, lower, torch.ones_like(lower), upper, upper), dim=-1)
        return baseline[..., 2:3] + location[..., None] + scale * (baseline - baseline[..., 2:3]), location

    def forward(self, batch, *, history_off=False, event_off=False, time_shift_off=False):
        # Keep the archived base source immutable. The history computation below
        # is its unchanged forward body; only the Anchor call receives a gate.
        b = batch
        n, k = b['mask'].shape
        state, gate = self.constraint(b['mechanism_inputs'])
        current = self.state(b['history'], b['future'])
        memory = self.state(b['memory_history'].flatten(0, 1), b['memory_future'].flatten(0, 1)).reshape(n, k, 24, -1)
        baseline = b['baseline']
        anchor, location = self.calibrated(baseline, torch.cat((current, b['stats']), dim=-1), gate)
        fields = self.dual(b['memory_actual'].flatten(0, 1), b['memory_z'].flatten(0, 1), b['memory_lags'].flatten(0, 1))
        trend, event = (fields[key].reshape(n, k, 24) * 300 for key in ('trend', 'event'))
        historical_q50 = b['memory_baseline'][..., 2]
        trend_error = trend - torch.einsum('ij,bkj->bki', self.lowpass, historical_q50)
        missed = (b['memory_actual'] - historical_q50 - trend_error.clamp_min(0)).clamp_min(0)
        event_error = torch.minimum(event.clamp_min(0), missed)
        query = current[:, None].expand(-1, k, -1, -1)
        age = torch.log1p(b['age'])[:, :, None, None].expand(-1, -1, 24, 1) / math.log(731)
        scores = self.match(torch.cat((query, memory, query - memory, age), dim=-1))
        scores = scores.masked_fill(~b['mask'][:, :, None, None], -torch.inf)
        scores = torch.cat((scores, scores.new_zeros((n, 1, 24, 2))), dim=1)
        weights = scores.softmax(dim=1)[:, :k]
        weight_trend, weight_event = weights.unbind(dim=-1)
        proposal_trend = (weight_trend * (trend_error - location[:, None])).sum(dim=1)
        if time_shift_off:
            moved = event_error
        else:
            alignment = torch.einsum('bhd,bksd->bkhs', self.alignment_query(current), self.alignment_key(memory)) / math.sqrt(current.shape[-1])
            alignment = (alignment - self.time_distance[None, None]).softmax(dim=2)
            moved = torch.einsum('bkhs,bks->bkh', alignment, event_error)
        proposal_event = (weight_event * moved).sum(dim=1)
        gate_input = torch.cat((current, b['stats'], proposal_trend[..., None] / 300, proposal_event[..., None] / 300), dim=-1)
        gates = self.gates(gate_input).sigmoid()
        correction_trend, correction_event = gates[..., 0] * proposal_trend, gates[..., 1] * proposal_event
        if event_off:
            correction_event = torch.zeros_like(correction_event)
        correction = 150 * torch.tanh((correction_trend + correction_event) / 150)
        if history_off:
            correction = torch.zeros_like(correction)
        reconstructed = fields['reconstruction'].reshape(n, k, 24)
        rec_error = (reconstructed - b['memory_actual'] / 300).square().mean(dim=-1)
        reconstruction = (rec_error * b['mask']).sum() / b['mask'].sum().clamp_min(1)
        return dict(prediction=anchor + correction[..., None], anchor=anchor, correction=correction,
            trend=correction_trend, event=correction_event, gates=gates, weights=weights,
            reconstruction_loss=reconstruction, constraint_gates=gate, mechanism_state=state)


def build_model(variant, seed=2026):
    if variant not in VARIANTS:
        raise ValueError('Unknown mechanism-state arm')
    torch.manual_seed(seed)
    if variant == 'bounded_reference':
        return BoundedCalibrationModel()
    return EvidenceCalibrationModel(constant=variant == 'constant_evidence')
