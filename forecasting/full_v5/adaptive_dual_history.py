"""Adaptive Fourier + zero-frequency Gabor fields for released historical curves.

The analytic coefficients fit *observed mature historical prices*. Only atom
geometry is learned across curves; this is not a future-price predictor or an
unchanged implementation of the canonical DualTimesField. Fixed geometry uses
exactly the same proposals, observations, basis size and coefficient solver.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn

from . import schema
from .dual_history_model import HOUR, PRICE_SCALE, TRAIN_START, _shape_check

FIT_CUTOFF = schema.unix_timestamp("2022-09-01")
ATOMS = 6
RIDGE = 1e-4


@dataclass(frozen=True)
class AdaptiveDualConfig:
    epochs: int = 12
    batch_size: int = 128
    hidden_dim: int = 32
    learning_rate: float = .003
    device: str = "cpu"

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.hidden_dim) < 1 or self.learning_rate <= 0:
            raise ValueError("Positive training dimensions and learning rate required")


class AdaptiveHistoricalDualField(nn.Module):
    """Curve-specific atom geometry, with observed-only analytic coefficients."""

    def __init__(self, config=None, *, fit_cutoff=FIT_CUTOFF):
        super().__init__()
        self.config = config or AdaptiveDualConfig()
        self.fit_cutoff = int(fit_cutoff)
        self.geometry = nn.Sequential(nn.Linear(27, self.config.hidden_dim), nn.GELU(),
                                      nn.Linear(self.config.hidden_dim, 2))
        nn.init.zeros_(self.geometry[-1].weight)
        nn.init.zeros_(self.geometry[-1].bias)
        self.register_buffer("time", torch.arange(24, dtype=torch.float32))
        angle = 2 * math.pi * self.time / 24
        self.register_buffer("fourier", torch.stack((torch.ones_like(angle),
                                                     angle.cos(), angle.sin()), dim=-1))

    @staticmethod
    def _solve(basis, target, observed):
        # Double precision avoids unstable gradients for overlapping pulses.
        design = basis.double()
        mask = observed.double()
        safe_target = torch.where(observed, target, torch.zeros_like(target)).double()
        gram = torch.einsum("bti,bt,btj->bij", design, mask, design)
        regularizer = torch.full((basis.shape[-1],), RIDGE, dtype=torch.float64,
                                 device=basis.device)
        regularizer[0] = 0.
        rhs = torch.einsum("bti,bt->bi", design, safe_target)
        coeff = torch.linalg.solve(gram + torch.diag(regularizer)[None], rhs)
        return coeff.to(basis.dtype)

    def forward(self, price, z, lag_prices, observed_mask=None, *, fixed=False):
        """Return /300 price fields and centers/widths in hours.

        Hidden prices are replaced before any arithmetic, proposals or solver.
        Every row must retain at least one observation. Exogenous and lag inputs
        are the historical record's already-known context, not hidden labels.
        """
        if price.ndim != 2 or price.shape[1] != 24:
            raise ValueError("Expected price [N,24]")
        if z.shape != (*price.shape, 16) or lag_prices.shape != (*price.shape, 3):
            raise ValueError("Expected z[N,24,16], lag_prices[N,24,3]")
        observed = (torch.ones_like(price, dtype=torch.bool) if observed_mask is None
                    else observed_mask.to(device=price.device, dtype=torch.bool))
        if observed.shape != price.shape or not bool(observed.any(dim=1).all()):
            raise ValueError("Every curve needs at least one observed price")
        safe = torch.where(observed, price, torch.zeros_like(price)) / PRICE_SCALE
        if not bool(torch.isfinite(safe).all() and torch.isfinite(z).all()
                    and torch.isfinite(lag_prices).all()):
            raise ValueError("Nonfinite visible historical inputs")
        batch = len(price)
        fourier = self.fourier.to(price.dtype)[None].expand(batch, -1, -1)
        base_coeff = self._solve(fourier, safe, observed)
        base = torch.einsum("bti,bi->bt", fourier, base_coeff)
        residual = torch.where(observed, safe - base, torch.zeros_like(safe))
        scores = residual.detach().abs().masked_fill(~observed, -1.)
        anchors, active = [], []
        hour_ids = torch.arange(24, device=price.device)[None]
        for _ in range(ATOMS):
            anchor = scores.argmax(dim=1)
            valid = scores.max(dim=1).values >= 0
            anchors.append(anchor)
            active.append(valid)
            suppress = (hour_ids - anchor[:, None]).abs() <= 1
            scores = scores.masked_fill(suppress, -1.)
        anchor_ids = torch.stack(anchors, dim=1)
        active_mask = torch.stack(active, dim=1)
        centers = anchor_ids.to(price.dtype)
        if fixed:
            geometry = price.new_zeros((batch, ATOMS, 2))
        else:
            offsets = torch.arange(-1, 2, device=price.device)
            neighbors = anchor_ids[..., None] + offsets
            inside = (neighbors >= 0) & (neighbors < 24)
            neighbors = neighbors.clamp(0, 23)
            row = torch.arange(batch, device=price.device)[:, None, None]
            visible_patch = observed[row, neighbors] & inside
            patch = torch.where(visible_patch, residual[row, neighbors], 0.)
            row_atom = torch.arange(batch, device=price.device)[:, None]
            local_z = z[row_atom, anchor_ids]
            local_lag = lag_prices[row_atom, anchor_ids] / PRICE_SCALE
            scale = torch.log1p(safe.abs().sum(dim=1) / observed.sum(dim=1))
            context = torch.cat((patch, visible_patch.to(price.dtype), local_z, local_lag,
                                 centers[..., None] / 23,
                                 scale[:, None, None].expand(-1, ATOMS, -1)), dim=-1)
            geometry = self.geometry(context)
        centers = (centers + .5 * geometry[..., 0].tanh()).clamp(0., 23.)
        widths = (.5 * torch.exp(math.log(3.) * geometry[..., 1].tanh())).clamp(min=.25)
        displacement = (self.time[None, :, None] - centers[:, None]) / widths[:, None]
        pulses = torch.exp(-.5 * displacement.square()) * active_mask[:, None]
        basis = torch.cat((fourier, pulses), dim=-1)
        coefficients = self._solve(basis, safe, observed)
        trend = torch.einsum("bti,bi->bt", fourier, coefficients[:, :3])
        event = torch.einsum("btk,bk->bt", pulses, coefficients[:, 3:])
        # Explicitly preserve the observed mean (and full mean when unmasked).
        shift = ((safe - trend - event) * observed).sum(dim=1) / observed.sum(dim=1)
        trend = trend + shift[:, None]
        return dict(trend=trend, event=event, reconstruction=trend + event,
                    centers=centers, widths=widths, amplitude=coefficients[:, 3:],
                    gates=active_mask.to(price.dtype), hidden_mask=~observed)

    def reconstruction_loss(self, price, z, lag_prices):
        output = self(price, z, lag_prices)
        weights = 1. + 3. * (price > 300).to(price.dtype)
        loss = (weights * (output["reconstruction"] - price / PRICE_SCALE).square()).mean()
        return loss

    def encode_matured(self, origins, actual, z, lag_prices, *, as_of, fixed=False):
        if int(as_of) < self.fit_cutoff:
            raise ValueError("Encoder was not fitted at requested as_of")
        origins, actual, z, lag_prices = _shape_check(origins, actual, z, lag_prices)
        indices = np.flatnonzero(origins + 24 * HOUR <= int(as_of))
        chosen = [np.asarray(a[indices], dtype=np.float32) for a in (actual, z, lag_prices)]
        if not all(np.isfinite(a).all() for a in chosen):
            raise ValueError("Nonfinite matured historical inputs")
        keys = ("trend", "event", "reconstruction", "centers", "widths", "amplitude", "gates")
        result = {key: [] for key in keys}
        self.eval()
        with torch.no_grad():
            for start in range(0, len(indices), self.config.batch_size):
                args = [torch.as_tensor(a[start:start + self.config.batch_size],
                                        device=self.time.device) for a in chosen]
                output = self(*args, fixed=fixed)
                for key in keys:
                    scale = PRICE_SCALE if key in ("trend", "event", "reconstruction", "amplitude") else 1.
                    values = output[key].cpu().numpy()
                    # Convert before physical scaling: separate float32 products
                    # can break T+E=R when large opposing fields nearly cancel.
                    if key in ("trend", "event", "reconstruction", "amplitude"):
                        values = values.astype(np.float64)
                    result[key].append(values * scale)
        for key, pieces in result.items():
            width = 24 if key in ("trend", "event", "reconstruction") else ATOMS
            dtype = np.float64 if key in ("trend", "event", "reconstruction", "amplitude") else np.float32
            result[key] = np.concatenate(pieces) if pieces else np.empty((0, width), dtype=dtype)
        # The public physical-unit reconstruction is defined by its two fields,
        # avoiding a second rounded sum exported independently from them.
        result["reconstruction"] = result["trend"] + result["event"]
        result["indices"] = indices
        result["mae"] = (np.mean(np.abs(result["reconstruction"] - chosen[0]), axis=1)
                         if len(indices) else np.empty(0, dtype=np.float32))
        return result

    def save_checkpoint(self, path, audit):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(config=asdict(self.config), fit_cutoff=self.fit_cutoff,
                        state_dict={k: v.detach().cpu() for k, v in self.state_dict().items()},
                        audit=audit), path)

    @classmethod
    def load_checkpoint(cls, path, *, device="cpu"):
        payload = torch.load(path, map_location=device, weights_only=True)
        model = cls(AdaptiveDualConfig(**dict(payload["config"], device=device)),
                    fit_cutoff=payload["fit_cutoff"]).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, payload["audit"]


def fit_adaptive_dualfield(origins, actual, z, lag_prices, *, cutoff=FIT_CUTOFF,
                           seed=2026, config=None):
    """Fit geometry on July--August only; no held-out epoch/parameter selection."""
    config = config or AdaptiveDualConfig()
    if not TRAIN_START < int(cutoff) <= FIT_CUTOFF:
        raise ValueError("Geometry fit cutoff must lie in July--August 2022")
    origins, actual, z, lag_prices = _shape_check(origins, actual, z, lag_prices)
    indices = np.flatnonzero((origins >= TRAIN_START) & (origins + 24 * HOUR < int(cutoff)))
    if not len(indices):
        raise ValueError("No fully matured pre-cutoff training curves")
    chosen = [np.asarray(a[indices], dtype=np.float32) for a in (actual, z, lag_prices)]
    if not all(np.isfinite(a).all() for a in chosen):
        raise ValueError("Nonfinite pre-cutoff training inputs")
    torch.set_num_threads(2)
    torch.manual_seed(int(seed))
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    model = AdaptiveHistoricalDualField(config, fit_cutoff=cutoff).to(config.device)
    tensors = [torch.as_tensor(a, device=config.device) for a in chosen]
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    trace = []
    for epoch in range(config.epochs):
        model.train()
        permutation = torch.randperm(len(indices), device=config.device)
        total = 0.
        for start in range(0, len(indices), config.batch_size):
            batch = permutation[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = model.reconstruction_loss(*(a[batch] for a in tensors))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite geometry training loss")
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not all(bool(torch.isfinite(g).all()) for g in gradients):
                raise ValueError("Nonfinite geometry gradient")
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        trace.append(dict(epoch=epoch + 1, weighted_mse=total / len(indices)))
    model.eval()
    audit = dict(seed=int(seed), config=asdict(config), fit_cutoff=int(cutoff),
                 fit_start=TRAIN_START, fit_rows=len(indices),
                 first_fit_origin=int(origins[indices[0]]), last_fit_origin=int(origins[indices[-1]]),
                 latest_fit_label_available=int(origins[indices[-1]] + 24 * HOUR),
                 architecture="local Fourier DC+first harmonic and adaptive zero-frequency Gabor pulses",
                 atoms=ATOMS, ridge=RIDGE, fixed_price_scale=PRICE_SCALE,
                 loss="full-visible MSE(price/300) weighted 1+3*(price>300)",
                 geometry="observed residual peak anchors; learned bounded center offset and width",
                 coefficients="observed-only joint ridge least squares; unpenalized intercept",
                 interpretation="learned geometry, analytic mature-price amplitudes; not future prediction",
                 trainable_parameters=sum(p.numel() for p in model.parameters()), trace=trace)
    return model, audit
