"""DDNN-JSU: distributional deep network with a Johnson SU parameter head.

Method reimplementation. The J34 companion repository (lebeand/uq-epf) ships a
DDNN with a Normal parameterization; the Johnson SU parameterization follows
the paper's distributional formulation implemented locally from its
parametrization:

    z = gamma + delta * asinh((x - xi) / lambda),  z ~ N(0, 1)
    x_q = xi + lambda * sinh((Phi^-1(q) - gamma) / delta)

The network shape follows the official ddnn_normal.py pattern (Kaiming-hidden /
Xavier-output MLP, optional batch-norm/dropout, Adam). Training maximizes the
exact JSU likelihood (NLL) in the shared standardized price space; the five
common quantiles come from the analytic inverse CDF and are inverted to raw
RRP. Selection between epochs uses the declared raw-price common five-quantile
QS, so checkpoint choice matches the public scoring even though training
optimizes NLL. The JSU location is not claimed to be the median: the q50 used
for point metrics is the true distribution median.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..aligned_baselines import BaselinePrediction
from ..aligned_scoring import QUANTILE_LEVELS
from ..common.frozen_base import _expansion_artifacts
from ..common.native_trainer import block_tensors, train_native

MIN_SHAPE = 1e-4  # softplus floor keeping delta and lambda strictly positive


class JohnsonSUHead(nn.Module):
    """MLP emitting (gamma, log-unconstrained delta, xi, log-unconstrained lambda) per horizon."""

    def __init__(self, input_dim: int, horizon: int, *, hidden_dims: list[int],
                 use_batch_norm: bool, dropout_rate: float):
        super().__init__()
        self.horizon = horizon
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            linear = nn.Linear(previous, width)
            nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
            nn.init.constant_(linear.bias, 0.0)
            block: list[nn.Module] = [linear]
            if use_batch_norm:
                block.append(nn.BatchNorm1d(width))
            block.append(nn.ReLU())
            if dropout_rate > 0:
                block.append(nn.Dropout(dropout_rate))
            layers.append(nn.Sequential(*block))
            previous = width
        output = nn.Linear(previous, 4 * horizon)
        nn.init.xavier_uniform_(output.weight)
        nn.init.constant_(output.bias, 0.0)
        self.hidden = nn.Sequential(*layers) if layers else nn.Identity()
        self.output = output

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        raw = self.output(self.hidden(x)).view(-1, self.horizon, 4)
        gamma, delta_t, xi, lam_t = raw.unbind(dim=-1)
        delta = torch.nn.functional.softplus(delta_t) + MIN_SHAPE
        lam = torch.nn.functional.softplus(lam_t) + MIN_SHAPE
        return gamma, delta, xi, lam


def jsu_negative_log_likelihood(target: torch.Tensor, gamma: torch.Tensor, delta: torch.Tensor,
                                xi: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Exact JSU NLL, numerically stabilized; mean over all origin/horizon cells."""
    z = gamma + delta * torch.asinh((target - xi) / lam)
    log_phi = -0.5 * (z * z + math.log(2.0 * math.pi))
    # log cosh(u) = logaddexp(u, -u) - log 2, stable for extreme u.
    log_cosh = torch.logaddexp(z, -z) - math.log(2.0)
    return (-log_phi + torch.log(delta) - torch.log(lam) + log_cosh).mean()


def jsu_quantiles(gamma: torch.Tensor, delta: torch.Tensor, xi: torch.Tensor, lam: torch.Tensor,
                  levels: torch.Tensor) -> torch.Tensor:
    """Analytic quantiles via the standard-normal inverse CDF; supports negative prices.

    The sinh argument is clamped to ±30: float32 sinh overflows to inf beyond
    |arg| ~ 89, which early-training parameter outliers can produce. The clamp
    engages only far outside the operating regime and keeps every quantile
    finite; nothing is silently clipped in the normal range.
    """
    z = torch.erfinv(2.0 * levels - 1.0) * math.sqrt(2.0)  # [Q] -> Phi^-1(q)
    argument = (z.view(*[1] * gamma.dim(), -1) - gamma.unsqueeze(-1)) / delta.unsqueeze(-1)
    argument = torch.clamp(argument, -30.0, 30.0)
    return xi.unsqueeze(-1) + lam.unsqueeze(-1) * torch.sinh(argument)  # [N, H, Q]


class DDNNJSU(nn.Module):
    """Consumes every shared block flattened in the canonical INPUT_KEYS order."""

    def __init__(self, input_dim: int, horizon: int, params: dict[str, Any]):
        super().__init__()
        self.head = JohnsonSUHead(input_dim, horizon, hidden_dims=list(params["hidden_dims"]),
                                  use_batch_norm=bool(params["use_batch_norm"]),
                                  dropout_rate=float(params["dropout_rate"]))
        self.horizon = horizon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.head(x)


def _flatten(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    from ..aligned_inputs import INPUT_KEYS

    return torch.cat([batch[key].reshape(batch[key].shape[0], -1) for key in INPUT_KEYS], dim=1)


def run_expansion(context: Any) -> BaselinePrediction:
    params = dict(context.config["model_baselines"]["ddnn_jsu"])
    expected = {"hidden_dims", "use_batch_norm", "dropout_rate", "selection_metric"}
    if set(params) != expected:
        raise ValueError(f"Unexpected ddnn_jsu model parameters: {sorted(params)}")
    if params["selection_metric"] != "qs5":
        raise ValueError("ddnn_jsu first version fixes selection_metric=qs5")
    training = context.config["training"]
    artifacts = _expansion_artifacts("ddnn_jsu", context)
    device = torch.device(context.device)
    torch.manual_seed(context.seed)
    train, selection, test = context.splits["train"], context.splits["selection"], context.splits["test"]
    horizon = train.target.shape[1]
    train_tensors = block_tensors(train, device, include_target=True)
    selection_tensors = block_tensors(selection, device, include_target=True)
    input_dim = _flatten(train_tensors).shape[1]
    model = DDNNJSU(input_dim, horizon, params).to(device)
    mean, std = float(train.metadata["target_mean"]), float(train.metadata["target_std"])
    levels = torch.tensor(QUANTILE_LEVELS, dtype=torch.float32, device=device)

    def compute_loss(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        gamma, delta, xi, lam = model(_flatten(batch))
        return jsu_negative_log_likelihood(batch["target"], gamma, delta, xi, lam)

    def predict_std_quantiles(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        gamma, delta, xi, lam = model(_flatten(tensors))
        return jsu_quantiles(gamma, delta, xi, lam, levels)

    def predict_raw(tensors: dict[str, torch.Tensor]) -> np.ndarray:
        return predict_std_quantiles(tensors).cpu().numpy().astype(np.float64) * std + mean

    artifacts.progress("train_start", epochs=training["epochs"], seed=context.seed)
    checkpoint_path = Path(context.run_root) / context.job["output"] / "best_model.pt"
    history = train_native(
        model=model, compute_loss=compute_loss, predict_raw=predict_raw,
        train_tensors=train_tensors, selection_tensors=selection_tensors,
        selection_actual=selection.actual, quantile_levels=tuple(QUANTILE_LEVELS),
        selection_metric="qs5", epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]), learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]), patience=int(training["patience"]),
        seed=context.seed, device=device, checkpoint_path=checkpoint_path,
        progress_path=artifacts.progress_path, model_id="ddnn_jsu",
        metadata={"distribution": "johnson_su", "params": params,
                  "target_space": "shared_standardized_price", "train_mean": mean, "train_std": std})
    model.eval()
    with torch.no_grad():
        test_tensors = block_tensors(test, device, include_target=False)
        quantiles = predict_raw(test_tensors)
    point = np.sort(quantiles, axis=-1)[..., 2].copy()
    metadata = {
        "model": "ddnn_jsu", "task": "quantile", "seed": context.seed,
        "distribution": "johnson_su", "distribution_parametrization":
            "z=gamma+delta*asinh((x-xi)/lambda); quantiles via analytic inverse CDF",
        "native_training_objective": "exact_jsu_negative_log_likelihood",
        "selection_metric": "raw_price_common_five_quantile_qs5",
        "paper_selection_difference": "J34 selects by validation NLL and tunes hyperparameters; "
                                      "this adaptation selects by the public raw-price QS with fixed parameters",
        "prediction_space": "raw_rrp", "point_definition": "analytic_jsu_median",
        "quantile_levels": list(QUANTILE_LEVELS),
        "training_history": history,
        "package_versions": {"torch": torch.__version__, "numpy": np.__version__},
        "implementation_identity": "method_reimplementation_from_paper_formulas",
        "official_reference_implementation": "https://github.com/lebeand/uq-epf (Normal head only)",
        "requested_params": params,
        "supports_negative_prices": True,
        "calibration": "none_first_version_no_post_hoc_calibrator",
    }
    artifacts.finish(metadata, dict(train=train.metadata, selection=selection.metadata,
                                    test=test.metadata))
    return BaselinePrediction(point, quantiles, metadata)
