"""Adapted canonical dual field for *mature historical* curve reconstruction.

Uses ``src.dualfield.core.DualTimesField`` Fourier CTF and gated Gabor DGF,
not V4's spline decoder. The pointwise DGF residual shortcut is deliberately
excluded. Live gates supply the sparsity gradient (the canonical cached gates
are detached). Narrow atoms, low-frequency background and denoising are local
adaptations, not claims of an unchanged canonical implementation.

This module predicts no future labels. Training uses fully released July--
September 2022 curves. Historical representations may enter a downstream
forecast only after both encoder fitting and that curve's full release.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.dualfield.core import DualTimesField
from . import schema

HOUR = 3600
PRICE_SCALE = 300.0
TRAIN_START = schema.unix_timestamp("2022-07-01")
FIT_CUTOFF = schema.unix_timestamp("2022-10-01")
CHANNELS = 21  # historical price, 16 known states, 3 lag-day prices, price mask


@dataclass(frozen=True)
class DualHistoryConfig:
    epochs: int = 12
    batch_size: int = 128
    hidden_dim: int = 24
    num_atoms: int = 6
    num_frequencies: int = 4
    learning_rate: float = .003
    device: str = "cpu"

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.hidden_dim, self.num_atoms,
               self.num_frequencies) < 1 or self.learning_rate <= 0:
            raise ValueError("Positive training dimensions and learning rate required")


def _shape_check(origins, actual, z, lag_prices):
    origins = np.asarray(origins, dtype=np.int64)
    actual = np.asarray(actual)
    z = np.asarray(z)
    lag_prices = np.asarray(lag_prices)
    n = len(origins)
    if origins.ndim != 1 or np.any(np.diff(origins) <= 0):
        raise ValueError("Origins must be strictly increasing")
    if actual.shape != (n, 24) or z.shape != (n, 24, 16) or lag_prices.shape != (n, 24, 3):
        raise ValueError("Expected actual[N,24], z[N,24,16], lag_prices[N,24,3]")
    return origins, actual, z, lag_prices


def lowpass_price(curves):
    """Fixed DC + first harmonic, identical to canonical 24-step lowpass.

    Accepts a torch [N,24] tensor. It is the explicitly declared background
    prior and should also be evaluated as the non-learned smoothing control.
    """
    frequency = torch.fft.rfft(curves, dim=1)
    mask = torch.zeros(frequency.shape[1], dtype=curves.dtype, device=curves.device)
    mask[:2] = 1
    return torch.fft.irfft(frequency * mask[None], n=24, dim=1)


class HistoricalDualField(nn.Module):
    def __init__(self, config=None, *, fit_cutoff=FIT_CUTOFF):
        super().__init__()
        self.config = config or DualHistoryConfig()
        self.fit_cutoff = int(fit_cutoff)
        self.core = DualTimesField(
            num_variables=CHANNELS, seq_length=24,
            hidden_dim=self.config.hidden_dim,
            num_atoms=self.config.num_atoms,
            num_frequencies=self.config.num_frequencies,
            freq_cutoff=2.0,
        )
        self.register_buffer("time", torch.linspace(0, 1, 24))
        # The unused shortcut is frozen and never called, rather than merely
        # relying on loss weights to suppress an input-copy solution.
        for parameter in self.core.dgf.atoms.residual_net.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            atom = self.core.dgf.atoms
            # Canonical effective sigma = softplus(log_sigma) + .02.
            effective = torch.linspace(.06, .15, self.config.num_atoms)
            atom.log_sigma.copy_(torch.log(torch.expm1(effective - .02)))
            atom.omega.copy_(torch.linspace(8., 40., self.config.num_atoms))

    def forward(self, price, z, lag_prices, *, mask_probability=0.):
        """All inputs are already released historical curves, in price units."""
        if not 0 <= mask_probability < 1:
            raise ValueError("Mask probability must be in [0,1)")
        observed = torch.rand_like(price) >= mask_probability if mask_probability else torch.ones_like(price, dtype=torch.bool)
        scaled = price / PRICE_SCALE
        x = torch.cat((torch.where(observed, scaled, 0.)[..., None], z,
                       lag_prices / PRICE_SCALE, observed.to(price.dtype)[..., None]), dim=-1)
        background = self.core.ctf(x, self.time)
        events, amplitude, gates = self.core.dgf.extract_events(x - background, self.time)
        return dict(trend=background[..., 0], event=events[..., 0],
                    reconstruction=background[..., 0] + events[..., 0],
                    amplitude=amplitude[..., 0], gates=gates,
                    hidden_mask=~observed)

    def reconstruction_loss(self, price, z, lag_prices):
        output = self(price, z, lag_prices, mask_probability=.2)
        target = price / PRICE_SCALE
        error = output["reconstruction"] - target
        hidden = output["hidden_mask"]
        # Squared error keeps sensitivity to spikes; bounded step gradients
        # protect numerical stability without clipping the labels themselves.
        masked = error[hidden].square().mean() if hidden.any() else error.square().mean()
        full = error.square().mean()
        smooth = torch.diff(output["trend"], dim=1).square().mean()
        prior = (output["trend"] - lowpass_price(target)).square().mean()
        sparse = output["gates"].mean()  # live tensor, not canonical detached cache
        total = masked + .1 * full + .1 * prior + .01 * smooth + .001 * sparse
        return total, dict(masked_mse=masked, full_mse=full,
                           lowpass_prior=prior, smoothness=smooth, live_gate=sparse)

    def constrain_atoms(self):
        with torch.no_grad():
            atom = self.core.dgf.atoms
            atom.tau.clamp_(0., 1.)
            atom.omega.clamp_(.01, 60.)
            atom.log_sigma.clamp_(float(np.log(np.expm1(.02))),
                                  float(np.log(np.expm1(.23))))

    def encode_matured(self, origins, actual, z, lag_prices, *, as_of):
        """Filter maturity BEFORE constructing neural inputs or calling forward.

        Full-array labels may be supplied, but immature values are never fed
        to the model. The encoder itself is unavailable before fit_cutoff.
        """
        if int(as_of) < self.fit_cutoff:
            raise ValueError("Encoder was not fitted at requested as_of")
        origins, actual, z, lag_prices = _shape_check(origins, actual, z, lag_prices)
        indices = np.flatnonzero(origins + 24 * HOUR <= int(as_of))
        chosen = [np.asarray(a[indices], dtype=np.float32) for a in (actual, z, lag_prices)]
        if not all(np.isfinite(a).all() for a in chosen):
            raise ValueError("Nonfinite matured historical inputs")
        device = self.time.device
        result = {k: [] for k in ("trend", "event", "reconstruction", "amplitude", "gates")}
        self.eval()
        with torch.no_grad():
            for start in range(0, len(indices), self.config.batch_size):
                args = [torch.as_tensor(a[start:start + self.config.batch_size], device=device) for a in chosen]
                output = self(*args)
                for key in result:
                    scale = 1. if key == "gates" else PRICE_SCALE
                    result[key].append(output[key].cpu().numpy() * scale)
        for key, pieces in result.items():
            width = self.config.num_atoms if key in ("amplitude", "gates") else 24
            result[key] = np.concatenate(pieces) if pieces else np.empty((0, width), dtype=np.float32)
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
        config = dict(payload["config"], device=device)
        model = cls(DualHistoryConfig(**config), fit_cutoff=payload["fit_cutoff"]).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, payload["audit"]


def fit_historical_dualfield(origins, actual, z, lag_prices, *, fit_cutoff=FIT_CUTOFF,
                             seed=2026, config=None):
    """Fixed-budget historical denoising; no validation-based epoch selection."""
    config = config or DualHistoryConfig()
    if not TRAIN_START < int(fit_cutoff) <= FIT_CUTOFF:
        raise ValueError("Encoder fit cutoff must lie in July--September 2022")
    origins, actual, z, lag_prices = _shape_check(origins, actual, z, lag_prices)
    indices = np.flatnonzero((origins >= TRAIN_START) &
                            (origins + 24 * HOUR < int(fit_cutoff)))
    if not len(indices):
        raise ValueError("No fully matured pre-cutoff training curves")
    selected = [np.asarray(a[indices], dtype=np.float32) for a in (actual, z, lag_prices)]
    if not all(np.isfinite(a).all() for a in selected):
        raise ValueError("Nonfinite pre-cutoff training input")
    torch.set_num_threads(2)
    torch.manual_seed(int(seed))
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    model = HistoricalDualField(config, fit_cutoff=fit_cutoff).to(config.device)
    tensors = [torch.as_tensor(a, device=config.device) for a in selected]
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=config.learning_rate)
    trace = []
    for epoch in range(config.epochs):
        model.train()
        permutation = torch.randperm(len(indices), device=config.device)
        weighted = 0.
        terms_sum = {}
        for start in range(0, len(indices), config.batch_size):
            batch = permutation[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss, terms = model.reconstruction_loss(*(a[batch] for a in tensors))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite reconstruction training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optimizer.step()
            model.constrain_atoms()
            weighted += float(loss.detach()) * len(batch)
            for key, value in terms.items():
                terms_sum[key] = terms_sum.get(key, 0.) + float(value.detach()) * len(batch)
        trace.append(dict(epoch=epoch + 1, loss=weighted / len(indices),
                          **{k: v / len(indices) for k, v in terms_sum.items()}))
    model.eval()
    audit = dict(seed=int(seed), config=asdict(config), fit_cutoff=int(fit_cutoff),
                 fit_start=TRAIN_START, fit_rows=len(indices), first_fit_origin=int(origins[indices[0]]),
                 last_fit_origin=int(origins[indices[-1]]),
                 latest_fit_label_available=int(origins[indices[-1]] + 24 * HOUR),
                 input_channels=CHANNELS, fixed_price_scale=PRICE_SCALE,
                 reconstruction_space="raw price divided by 300; no target clipping",
                 architecture="adapted canonical Fourier CTF + gated Gabor DGF; no residual shortcut",
                 loss="masked_MSE + 0.1 full_MSE + 0.1 lowpass_prior + 0.01 smoothness + 0.001 live_gate",
                 lowpass_prior="DC + first harmonic; fixed smoothing control required",
                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                 trace=trace)
    return model, audit
