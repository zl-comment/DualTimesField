"""Small causal gates for separate historical trend and positive-event transfer.

The gates only learn whether to use already-computed historical proposals.
LightGBM, C_roll, the historical encoder and all input features remain frozen.
A clamp gate can reject a proposal exactly; event-off is a fixed-head
intervention, not a refitted model or a change to the gate's input features.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import schema

HOUR = 3600
FIT_START = schema.unix_timestamp("2022-10-01")
FIT_CUTOFF = schema.unix_timestamp("2022-11-01")
MAX_FIT_CUTOFF = schema.unix_timestamp("2022-12-01")
MAX_CORRECTION = 150.


@dataclass(frozen=True)
class GateConfig:
    epochs: int = 80
    learning_rate: float = .01
    l2: float = .001
    gate_penalty: float = .001
    device: str = "cpu"

    def __post_init__(self):
        if not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")
        values = (self.learning_rate, self.l2, self.gate_penalty)
        if not np.isfinite(values).all() or self.learning_rate <= 0 or min(self.l2, self.gate_penalty) < 0:
            raise ValueError("Positive learning rate and nonnegative finite penalties required")


def _input_arrays(features, trend, event, anchor, *, feature_count=None):
    features = np.asarray(features, dtype=np.float64)
    trend, event, anchor = [np.asarray(a, dtype=np.float64) for a in (trend, event, anchor)]
    if features.ndim != 3 or features.shape[1] != schema.HORIZONS or features.shape[2] < 1:
        raise ValueError("Expected features[N,24,D] with positive D")
    if feature_count is not None and features.shape[2] != feature_count:
        raise ValueError("Feature dimension differs from the fitted gate")
    shape = features.shape[:2]
    if trend.shape != shape or event.shape != shape or anchor.shape != (*shape, 5):
        raise ValueError("Expected trend/event[N,24], anchor[N,24,5]")
    if not all(np.isfinite(a).all() for a in (features, trend, event, anchor)):
        raise ValueError("Nonfinite gate inputs")
    if np.any(np.abs(trend) > MAX_CORRECTION) or np.any(event < 0.) or np.any(event > MAX_CORRECTION):
        raise ValueError("Trend must be in [-150,150] and event in [0,150]")
    if np.any(np.diff(anchor, axis=-1) < 0):
        raise ValueError("Anchor quantiles must be ordered")
    return features, trend, event, anchor


class ConditionalTransferGate(nn.Module):
    """Two linear clamp gates shared across the 24 forecast horizons."""

    def __init__(self, feature_count, config=None, *, mode="full", fit_cutoff=FIT_CUTOFF):
        super().__init__()
        if not isinstance(feature_count, (int, np.integer)) or feature_count < 1:
            raise ValueError("feature_count must be a positive integer")
        if mode not in ("full", "trend"):
            raise ValueError("Gate mode must be full or trend")
        if not FIT_START < int(fit_cutoff) <= MAX_FIT_CUTOFF:
            raise ValueError("Gate cutoff must lie in October--November 2022")
        self.feature_count = int(feature_count)
        self.config = config or GateConfig()
        self.mode = mode
        self.fit_cutoff = int(fit_cutoff)
        self.linear = nn.Linear(self.feature_count, 2, dtype=torch.float64,
                                device=self.config.device)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, .1)

    def forward(self, features, trend, event, anchor, *, event_off=False):
        gates = self.linear(features).clamp(0., 1.)
        gate_trend = gates[..., 0]
        gate_event = gates[..., 1] if self.mode == "full" else torch.zeros_like(gates[..., 1])
        contribution = gate_trend * trend
        if not event_off:
            contribution = contribution + gate_event * event
        delta = contribution.clamp(-MAX_CORRECTION, MAX_CORRECTION)
        return dict(quantiles=anchor + delta[..., None], gate_trend=gate_trend,
                    gate_event=gate_event, delta=delta)

    def predict(self, features, trend, event, anchor, *, event_off=False):
        arrays = _input_arrays(features, trend, event, anchor, feature_count=self.feature_count)
        args = [torch.as_tensor(a, dtype=torch.float64, device=self.linear.weight.device) for a in arrays]
        self.eval()
        with torch.no_grad():
            result = self(*args, event_off=event_off)
        return {name: value.cpu().numpy() for name, value in result.items()}

    def save_checkpoint(self, path, audit):
        """JSON parameters and metadata only; no executable pickle payload."""
        payload = dict(format="conditional-transfer-gate-v1", feature_count=self.feature_count,
                       config=asdict(self.config), mode=self.mode, fit_cutoff=self.fit_cutoff,
                       weight=self.linear.weight.detach().cpu().numpy().tolist(),
                       bias=self.linear.bias.detach().cpu().numpy().tolist(), audit=audit)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, allow_nan=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, path, *, device="cpu"):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "conditional-transfer-gate-v1":
            raise ValueError("Unsupported gate checkpoint format")
        config = GateConfig(**dict(payload["config"], device=device))
        model = cls(payload["feature_count"], config, mode=payload["mode"],
                    fit_cutoff=payload["fit_cutoff"])
        weight, bias = np.asarray(payload["weight"], dtype=np.float64), np.asarray(payload["bias"], dtype=np.float64)
        if (weight.shape != (2, model.feature_count) or bias.shape != (2,)
                or not np.isfinite(weight).all() or not np.isfinite(bias).all()):
            raise ValueError("Malformed gate checkpoint parameters")
        with torch.no_grad():
            model.linear.weight.copy_(torch.as_tensor(weight, device=device))
            model.linear.bias.copy_(torch.as_tensor(bias, device=device))
        model.eval()
        return model, payload["audit"]


def fit_transfer_gate(origins, actual, anchor_quantiles, features, trend, event, *,
                      cutoff=FIT_CUTOFF, seed=2026, config=None, mode="full"):
    """Fit the final fixed-budget gate using only strictly mature fit labels."""
    config = config or GateConfig()
    if not FIT_START < int(cutoff) <= MAX_FIT_CUTOFF:
        raise ValueError("Gate cutoff must lie in October--November 2022")
    origins = np.asarray(origins, dtype=np.int64)
    if origins.ndim != 1 or np.any(np.diff(origins) <= 0):
        raise ValueError("Origins must be strictly increasing")
    # Only inspect global array dimensions. Finite-value checks and tensors are
    # constructed AFTER the temporal selection, including the target labels.
    actual = np.asarray(actual)
    all_arrays = [np.asarray(a) for a in (features, trend, event, anchor_quantiles)]
    if actual.shape != (len(origins), schema.HORIZONS):
        raise ValueError("Expected actual[N,24]")
    if any(a.ndim < 1 or len(a) != len(origins) for a in all_arrays):
        raise ValueError("Gate inputs must share origin rows")
    indices = np.flatnonzero((origins >= FIT_START) &
                            (origins + schema.HORIZONS * HOUR < int(cutoff)))
    if not len(indices):
        raise ValueError("No fully matured pre-cutoff gate fitting rows")
    selected = _input_arrays(*(a[indices] for a in all_arrays))
    target = np.asarray(actual[indices], dtype=np.float64)
    if not np.isfinite(target).all():
        raise ValueError("Nonfinite fitting labels")
    torch.set_num_threads(2)
    torch.manual_seed(int(seed))
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    model = ConditionalTransferGate(selected[0].shape[-1], config, mode=mode, fit_cutoff=cutoff)
    args = [torch.as_tensor(a, dtype=torch.float64, device=config.device) for a in selected]
    target_tensor = torch.as_tensor(target, dtype=torch.float64, device=config.device)
    levels = torch.as_tensor(schema.QUANTILE_LEVELS, dtype=torch.float64, device=config.device)
    weights = torch.as_tensor(schema.LOSS_WEIGHTS, dtype=torch.float64, device=config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def objective(output):
        residual = (target_tensor[..., None] - output["quantiles"]) / schema.PRICE_SCALE
        pinball = torch.maximum(levels * residual, (levels - 1.) * residual)
        prediction_loss = (pinball * weights).sum(dim=-1).mean()
        parameters = torch.cat([p.reshape(-1) for p in model.parameters()])
        penalty_l2 = config.l2 * parameters.square().mean()
        penalty_gate = config.gate_penalty * (output["gate_trend"].mean() + output["gate_event"].mean())
        return prediction_loss + penalty_l2 + penalty_gate, prediction_loss, penalty_l2, penalty_gate

    trace = []
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, prediction_loss, penalty_l2, penalty_gate = objective(model(*args))
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite gate loss")
        loss.backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError("Nonfinite gate gradients")
        optimizer.step()
        trace.append(dict(epoch=epoch + 1, loss=float(loss.detach()),
                          pinball=float(prediction_loss.detach()), l2=float(penalty_l2.detach()),
                          gate_penalty=float(penalty_gate.detach())))
    model.eval()
    with torch.no_grad():
        fitted = model(*args)
        final_trained_objective = float(objective(fitted)[0])
        null_residual = (target_tensor[..., None] - args[-1]) / schema.PRICE_SCALE
        null_pinball = torch.maximum(levels * null_residual, (levels - 1.) * null_residual)
        null_objective = float((null_pinball * weights).sum(dim=-1).mean())
        selected_null = final_trained_objective >= null_objective
        if selected_null:
            # Compare exactly two prespecified choices on fit labels only:
            # the fixed final epoch and the all-zero null. No epoch search or
            # development/assessment labels enter this fallback decision.
            for parameter in model.parameters():
                parameter.zero_()
        final_selected_objective = float(objective(model(*args))[0])
    audit = dict(seed=int(seed), config=asdict(config), mode=mode, fit_start=FIT_START,
                 fit_cutoff=int(cutoff), fit_rows=len(indices),
                 first_fit_origin=int(origins[indices[0]]), last_fit_origin=int(origins[indices[-1]]),
                 latest_fit_label_available=int(origins[indices[-1]] + schema.HORIZONS * HOUR),
                 feature_count=model.feature_count,
                 trainable_parameters=sum(p.numel() for p in model.parameters()),
                 gate="clamp(linear(features),0,1); shared across horizons",
                 correction="clip(gT*trend+gE*event,-150,150); same offset for all five quantiles",
                 loss_weights=list(schema.LOSS_WEIGHTS), price_scale=schema.PRICE_SCALE,
                 loss="mean weighted pinball + l2*mean(parameter^2) + gate_penalty*(mean(gT)+mean(gE))",
                 selection="fixed final epoch or exact zero null using fitting objective only; no epoch search or held-out selection",
                 final_trained_objective=final_trained_objective,
                 null_objective=null_objective, selected_null=selected_null,
                 final_selected_objective=final_selected_objective,
                 final_objective=final_selected_objective, trace=trace)
    return model, audit
