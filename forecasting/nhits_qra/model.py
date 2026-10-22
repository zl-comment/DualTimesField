"""NHITS backbone + fixed-seed MC-dropout + QRA calibration.

Official NHITS blocks (vendored, see vendor/PROVENANCE.md) trained once on the
shared standardized price space with the upstream-default MAE loss. After the
backbone checkpoint is chosen on the selection split (declared metric:
raw-price MAE of the eval-mode point forecast), MC-dropout sampling with
dropout-only train mode produces K point forecasts per origin; their empirical
quantiles at the five common levels are the QRA features. QRA coefficients are
fitted on the 2022H2 calibration labels only and applied, frozen, to the test
period's MC features. MC sampling and dropout never update any training state.

Input mapping (every shared block reaches the backbone, no block dropped):
    insample_y : history_values[..., 0]                     (price, [B, 72])
    hist_exog  : history_values[..., 1:] + history_calendar   ([B, 72, 11])
    futr_exog  : future_calendar + future_exogenous +
                 ctf_exogenous + quantile_exogenous           ([B, 24, 16])
    stat_exog  : origin_context                              ([B, 1])
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..aligned_baselines import BaselinePrediction
from ..aligned_scoring import QUANTILE_LEVELS
from ..common.calibration import apply_qra, fit_qra
from ..common.frozen_base import _expansion_artifacts
from ..common.native_trainer import block_tensors, train_native
from .vendor.nhits_blocks import NHITSBlock, _IdentityBasis


class NHITSBackbone(nn.Module):
    """Residual stack of vendored NHITS blocks with a point (n_out=1) head."""

    def __init__(self, *, input_hours: int, horizon: int, hist_channels: int,
                 futr_channels: int, stat_channels: int, params: dict[str, Any]):
        super().__init__()
        self.horizon = horizon
        self.input_hours = input_hours
        blocks = []
        for stack in range(len(params["n_pool_kernel_size"])):
            basis = _IdentityBasis(backcast_size=input_hours, forecast_size=horizon,
                                   out_features=1, interpolation_mode=params["interpolation_mode"])
            n_theta = input_hours + max(horizon // params["n_freq_downsample"][stack], 1)
            blocks.append(NHITSBlock(
                input_size=input_hours, h=horizon, n_theta=n_theta,
                mlp_units=list(params["mlp_units"]), basis=basis,
                futr_input_size=futr_channels, hist_input_size=hist_channels,
                stat_input_size=stat_channels,
                n_pool_kernel_size=int(params["n_pool_kernel_size"][stack]),
                pooling_mode=params["pooling_mode"],
                dropout_prob=float(params["dropout_prob_theta"]),
                activation=params["activation"],
                futr_length=horizon))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, insample_y: torch.Tensor, hist_exog: torch.Tensor,
                futr_exog: torch.Tensor, stat_exog: torch.Tensor) -> torch.Tensor:
        residuals = insample_y.flip(dims=(-1,))  # official backcast initialization
        mask = torch.ones_like(residuals)
        forecast = insample_y[:, -1:, None]      # official Naive1 level initialization
        forecast = forecast.repeat(1, self.horizon, 1)
        for block in self.blocks:
            backcast, block_forecast = block(residuals, futr_exog, hist_exog, stat_exog)
            residuals = (residuals - backcast) * mask
            forecast = forecast + block_forecast
        return forecast.squeeze(-1)


def _split_views(tensors: dict[str, torch.Tensor]):
    insample_y = tensors["history_values"][..., 0]
    hist_exog = torch.cat([tensors["history_values"][..., 1:], tensors["history_calendar"]], dim=-1)
    futr_exog = torch.cat([tensors["future_calendar"], tensors["future_exogenous"],
                           tensors["ctf_exogenous"], tensors["quantile_exogenous"]], dim=-1)
    stat_exog = tensors["origin_context"]
    return insample_y, hist_exog, futr_exog, stat_exog


def _mc_quantile_features(model: NHITSBackbone, tensors: dict[str, torch.Tensor],
                          mc_samples: int, batch_size: int) -> np.ndarray:
    """K MC-dropout point forecasts -> empirical type-7 quantiles [N, H, Q]."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train(True)  # dropout-only train mode; nothing else updates
    count = tensors["history_values"].shape[0]
    samples = []
    with torch.no_grad():
        for _ in range(mc_samples):
            chunks = []
            for start in range(0, count, batch_size):
                view = {key: value[start:start + batch_size] for key, value in tensors.items()}
                y, hist, futr, stat = _split_views(view)
                chunks.append(model(y, hist, futr, stat))
            samples.append(torch.cat(chunks, dim=0))
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train(False)
    stacked = torch.stack(samples, dim=0).cpu().numpy()  # [K, N, H]
    return np.quantile(stacked, np.asarray(QUANTILE_LEVELS), axis=0,
                       method="linear").transpose(1, 2, 0)


def run_expansion(context: Any) -> BaselinePrediction:
    params = dict(context.config["model_baselines"]["nhits_qra"])
    expected = {"n_pool_kernel_size", "n_freq_downsample", "mlp_units", "pooling_mode",
                "interpolation_mode", "dropout_prob_theta", "activation",
                "mc_samples", "mc_batch_size", "selection_metric"}
    if set(params) != expected:
        raise ValueError(f"Unexpected nhits_qra model parameters: {sorted(params)}")
    if params["selection_metric"] != "mae_eval_point":
        raise ValueError("nhits_qra first version fixes selection_metric=mae_eval_point")
    training = context.config["training"]
    artifacts = _expansion_artifacts("nhits_qra", context)
    device = torch.device(context.device)
    torch.manual_seed(context.seed)
    train, selection, calibration, test = (context.splits[name] for name in
                                           ("train", "selection", "calibration", "test"))
    train_tensors = block_tensors(train, device, include_target=True)
    selection_tensors = block_tensors(selection, device, include_target=True)
    model = NHITSBackbone(input_hours=train.inputs["history_values"].shape[1],
                          horizon=train.target.shape[1],
                          hist_channels=train.inputs["history_values"].shape[2] - 1
                          + train.inputs["history_calendar"].shape[2],
                          futr_channels=train.inputs["future_calendar"].shape[2]
                          + train.inputs["future_exogenous"].shape[2]
                          + train.inputs["ctf_exogenous"].shape[2]
                          + train.inputs["quantile_exogenous"].shape[2],
                          stat_channels=train.inputs["origin_context"].shape[1],
                          params=params).to(device)
    mean, std = float(train.metadata["target_mean"]), float(train.metadata["target_std"])

    def compute_loss(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        y, hist, futr, stat = _split_views(batch)
        return torch.abs(model(y, hist, futr, stat) - batch["target"]).mean()

    def predict_raw_point(tensors: dict[str, torch.Tensor]) -> np.ndarray:
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, tensors["history_values"].shape[0], int(training["batch_size"])):
                view = {key: value[start:start + int(training["batch_size"])]
                        for key, value in tensors.items() if key != "target"}
                y, hist, futr, stat = _split_views(view)
                chunks.append(model(y, hist, futr, stat))
        return torch.cat(chunks, dim=0).cpu().numpy().astype(np.float64) * std + mean

    artifacts.progress("train_start", epochs=training["epochs"], seed=context.seed)
    checkpoint_path = Path(context.run_root) / context.job["output"] / "best_model.pt"
    history = train_native(
        model=model, compute_loss=compute_loss,
        predict_raw=lambda tensors: predict_raw_point(tensors),
        train_tensors=train_tensors, selection_tensors=selection_tensors,
        selection_actual=selection.actual, quantile_levels=tuple(QUANTILE_LEVELS),
        selection_metric="mae", epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]), learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]), patience=int(training["patience"]),
        seed=context.seed, device=device, checkpoint_path=checkpoint_path,
        progress_path=artifacts.progress_path, model_id="nhits_qra",
        metadata={"architecture": "vendored_neuralforecast_nhits", "params": params,
                  "target_space": "shared_standardized_price", "train_mean": mean, "train_std": std})

    # Frozen backbone: MC-dropout quantile features per split with fixed seeds.
    mc = int(params["mc_samples"])
    artifacts.progress("mc_calibration", mc_samples=mc)
    torch.manual_seed(context.seed + 7919)  # calibration draw; declared fixed rule
    cal_tensors = block_tensors(calibration, device, include_target=False)
    cal_features = _mc_quantile_features(model, cal_tensors, mc, int(params["mc_batch_size"])) * std + mean
    artifacts.progress("mc_test", mc_samples=mc)
    torch.manual_seed(context.seed + 2 * 7919)  # test draw; differs from calibration
    test_tensors = block_tensors(test, device, include_target=False)
    test_features = _mc_quantile_features(model, test_tensors, mc, int(params["mc_batch_size"])) * std + mean

    artifacts.progress("fit_qra", origins=len(calibration.origin_unix))
    fit = fit_qra(cal_features, calibration.actual, tuple(QUANTILE_LEVELS),
                  alpha=float(context.config["experiment"]["baseline_parameters"]["nhits_qra"]["qra_alpha"]),
                  solver=context.config["experiment"]["baseline_parameters"]["nhits_qra"]["qra_solver"])
    quantiles = apply_qra(fit, test_features)
    point = np.sort(quantiles, axis=-1)[..., 2].copy()
    artifacts.save_npz("qra_fit.npz",
                       dict(coefficients=fit["coefficients"],
                            feature_order=np.array(fit["feature_order"]),
                            quantile_levels=np.array(fit["quantile_levels"]),
                            calibration_origin_unix=calibration.origin_unix,
                            calibration_target_time_unix=calibration.target_time_unix),
                       content="QRA coefficients fitted on MC-dropout quantile features",
                       mc_samples=mc, alpha=fit["alpha"], solver=fit["solver"])
    metadata = {
        "model": "nhits_qra", "task": "quantile", "seed": context.seed,
        "architecture": "vendored_neuralforecast_nhits_point_head",
        "upstream_commit": "6c11accfbcfdd62a40ef69a7c81e3d9c7f272682",
        "input_mapping": "insample_y=price history; hist=demand+history calendar; "
                         "futr=future calendar+future/ctf/quantile exogenous over horizon; stat=origin_context",
        "native_training_objective": "upstream_default_mae_in_standardized_space",
        "selection_metric": "raw_price_mae_of_eval_mode_point_forecast",
        "mc_dropout": {"samples": mc, "dropout_only_train_mode": True,
                       "seeds": "job_seed+7919 (calibration), job_seed+2*7919 (test)",
                       "quantile_features": "empirical type-7 quantiles of MC point forecasts",
                       "updates_training_state": False},
        "calibration": "linear QRA on calibration labels after backbone frozen",
        "qra_alpha": fit["alpha"], "qra_solver": fit["solver"],
        "qra_feature_order": fit["feature_order"],
        "calibration_origins": int(len(calibration.origin_unix)),
        "prediction_space": "raw_rrp", "point_definition": "rearranged_q50_of_qra_output",
        "quantile_levels": list(QUANTILE_LEVELS),
        "training_history": history,
        "package_versions": {"torch": torch.__version__, "numpy": np.__version__},
        "requested_params": params,
        "paper_claim": "P08/P16-style NHITS+MC+QRA adaptation; not a reproduction of their pipelines",
    }
    artifacts.finish(metadata, dict(train=train.metadata, selection=selection.metadata,
                                    calibration=calibration.metadata, test=test.metadata))
    return BaselinePrediction(point, quantiles, metadata)
