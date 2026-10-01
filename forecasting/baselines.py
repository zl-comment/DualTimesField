"""RE-Price benchmark baselines trained on the local datasets.

RE-Price (Chen et al., Applied Energy 426 (2026) 128712) compares against
XGBoost with quantile regression, a GRU, and DeepAR.  This module re-implements
those three on the same splits, inputs, and scoring as the dual-field model:

* ``xgboost``: one multi-quantile XGBoost model per horizon on flattened
  history, the origin context, and the horizon's calendar and future
  exogenous values (the tabular features of ``gbdt_baseline``); the 0.5
  quantile is the point forecast.
* ``gru``: a GRU encodes the 72-hour history with its calendar; for each
  horizon, an MLP maps the final state and the horizon's known inputs to a
  Gaussian mean and scale.
* ``deepar``: an autoregressive LSTM with a Gaussian likelihood, trained with
  teacher forcing over the whole 96-hour window and forecast with 200
  ancestral sample paths.

RE-Price gives the GRU and DeepAR a Gamma likelihood on shifted prices. Here
the Gaussian is placed on the standardized asinh price, which is the target
space of the dual-field model; its inverse gives a skewed price distribution
and handles negative prices. None of the baselines receive news.

The general time-series baselines of ``general_baselines`` (``dlinear``,
``patchtst``, ``itransformer``, ``informer``) run through the same training
loop, with a pinball loss on five quantiles instead of a likelihood.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .datasets import build_region_datasets, load_forecast_config
from . import general_baselines
from .evaluate_paper_metrics import point_metrics, probabilistic_metrics
from .gbdt_baseline import _features

QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)


class SequenceWindows(Dataset):
    """Adds the history calendar to the forecasting samples."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.calendar = torch.from_numpy(dataset.calendar_values.astype(np.float32))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        start = int(self.dataset.origin_indices[index])
        sample["history_calendar"] = self.calendar[start - self.dataset.input_hours:start]
        return sample


def _known_future(batch: dict) -> torch.Tensor:
    """Calendar, future exogenous, CTF exogenous, and origin context per horizon."""
    parts = [batch["future_calendar"]]
    horizon = batch["future_calendar"].shape[1]
    for key in ("future_exogenous", "ctf_exogenous"):
        if key in batch:
            parts.append(batch[key])
    if "origin_context" in batch:
        parts.append(batch["origin_context"][:, None, :].expand(-1, horizon, -1))
    return torch.cat(parts, dim=-1)


def _known_history(batch: dict) -> torch.Tensor:
    parts = [batch["history_values"][..., 1:], batch["history_calendar"]]
    steps = batch["history_values"].shape[1]
    if "origin_context" in batch:
        parts.append(batch["origin_context"][:, None, :].expand(-1, steps, -1))
    return torch.cat(parts, dim=-1)


class GRUForecaster(nn.Module):
    def __init__(self, history_dim: int, future_dim: int, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.encoder = nn.GRU(history_dim, hidden, num_layers=layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden + future_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2)
        )

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        history = torch.cat([batch["history_values"], batch["history_calendar"]], dim=-1)
        _, state = self.encoder(history)
        future = _known_future(batch)
        summary = state[-1][:, None, :].expand(-1, future.shape[1], -1)
        mean, raw_scale = self.head(torch.cat([summary, future], dim=-1)).unbind(-1)
        return mean, nn.functional.softplus(raw_scale) + 1e-3

    def loss(self, batch: dict) -> torch.Tensor:
        mean, scale = self(batch)
        return -torch.distributions.Normal(mean, scale).log_prob(batch["target_price"][..., 0]).mean()

    @torch.no_grad()
    def predict(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        mean, scale = self(batch)
        levels = torch.tensor(QUANTILES, device=mean.device)
        z = torch.distributions.Normal(0.0, 1.0).icdf(levels)
        return mean, mean[..., None] + scale[..., None] * z


class DeepARForecaster(nn.Module):
    def __init__(self, history_dim: int, future_dim: int, hidden: int = 64, layers: int = 2,
                 samples: int = 200):
        super().__init__()
        # Covariates per step: known history or known future inputs, and a flag.
        self.covariate_dim = max(history_dim, future_dim) + 1
        self.history_dim = history_dim
        self.future_dim = future_dim
        self.samples = samples
        self.rnn = nn.LSTM(1 + self.covariate_dim, hidden, num_layers=layers, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, 2)

    def _pad(self, values: torch.Tensor, flag: float) -> torch.Tensor:
        padding = self.covariate_dim - 1 - values.shape[-1]
        flag_column = torch.full_like(values[..., :1], flag)
        return torch.cat([values, values.new_zeros(*values.shape[:-1], padding), flag_column], dim=-1)

    def _covariates(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self._pad(_known_history(batch), 0.0), self._pad(_known_future(batch), 1.0)

    def _step(self, output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_scale = self.head(output).unbind(-1)
        return mean, nn.functional.softplus(raw_scale) + 1e-3

    def loss(self, batch: dict) -> torch.Tensor:
        history_cov, future_cov = self._covariates(batch)
        target = torch.cat([batch["history_values"][..., 0], batch["target_price"][..., 0]], dim=1)
        covariates = torch.cat([history_cov, future_cov], dim=1)
        # Step t sees the price at t-1 and the covariates of t, and predicts t.
        inputs = torch.cat([target[:, :-1, None], covariates[:, 1:]], dim=-1)
        output, _ = self.rnn(inputs)
        mean, scale = self._step(output)
        return -torch.distributions.Normal(mean, scale).log_prob(target[:, 1:]).mean()

    @torch.no_grad()
    def predict(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        history_cov, future_cov = self._covariates(batch)
        history = batch["history_values"][..., 0]
        inputs = torch.cat([history[:, :-1, None], history_cov[:, 1:]], dim=-1)
        _, state = self.rnn(inputs)
        batch_size, horizon = future_cov.shape[:2]
        state = tuple(s.repeat_interleave(self.samples, dim=1) for s in state)
        previous = history[:, -1].repeat_interleave(self.samples)
        future_cov = future_cov.repeat_interleave(self.samples, dim=0)
        paths = []
        for step in range(horizon):
            step_input = torch.cat([previous[:, None], future_cov[:, step]], dim=-1)[:, None]
            output, state = self.rnn(step_input, state)
            mean, scale = self._step(output[:, 0])
            previous = torch.distributions.Normal(mean, scale).sample()
            paths.append(previous)
        paths = torch.stack(paths, dim=1).view(batch_size, self.samples, horizon)
        levels = torch.tensor(QUANTILES, device=paths.device)
        quantile = torch.quantile(paths, levels, dim=1).permute(1, 2, 0)
        return quantile[..., QUANTILES.index(0.5)], quantile


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}


def _neural_run(kind: str, datasets: dict, seed: int, device: torch.device, epochs: int,
                learning_rate: float | None = None) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    windows = {name: SequenceWindows(datasets[name]) for name in ("train", "validation", "test")}
    example = windows["train"][0]
    example_batch = {k: v[None] for k, v in example.items() if torch.is_tensor(v)}
    history_dim = example["history_values"].shape[-1] + example["history_calendar"].shape[-1]
    future_dim = _known_future(example_batch).shape[-1]
    if kind == "gru":
        model = GRUForecaster(history_dim, future_dim)
    elif kind == "deepar":
        model = DeepARForecaster(_known_history(example_batch).shape[-1], future_dim)
    else:
        model = general_baselines.build(kind, example_batch)
    model.to(device)
    if learning_rate is None:
        learning_rate = general_baselines.LEARNING_RATES.get(kind, 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(windows["train"], batch_size=256, shuffle=True, generator=generator)
    validation_loader = DataLoader(windows["validation"], batch_size=1024)
    best_loss, best_state, best_epoch = math.inf, None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(np.mean([
                model.loss(_to_device(batch, device)).item() for batch in validation_loader
            ]))
        history.append(validation_loss)
        print(f"epoch {epoch + 1} validation {validation_loss:.5f}", flush=True)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    test = datasets["test"]
    points, quantiles = [], []
    for batch in DataLoader(windows["test"], batch_size=512):
        point, quantile = model.predict(_to_device(batch, device))
        points.append(test.denormalize_target(point).cpu())
        quantiles.append(test.denormalize_target(quantile).cpu())
    return {
        "point": torch.cat(points).double().numpy(),
        "quantile": np.sort(torch.cat(quantiles).double().numpy(), axis=-1),
        "best_epoch": best_epoch,
        "validation_nll" if kind in ("gru", "deepar") else "validation_pinball": history,
        "learning_rate": learning_rate,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def _xgboost_run(datasets: dict, seed: int, device: torch.device) -> dict:
    import xgboost

    (x_train, c_train, y_train) = _features(datasets["train"], True)
    (x_val, c_val, y_val) = _features(datasets["validation"], True)
    (x_test, c_test, _) = _features(datasets["test"], True)
    horizon = y_train.shape[1]
    point = np.zeros((x_test.shape[0], horizon))
    quantile = np.zeros((x_test.shape[0], horizon, len(QUANTILES)))
    rounds = []
    for h in range(horizon):
        model = xgboost.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=np.asarray(QUANTILES),
            tree_method="hist",
            device="cuda" if device.type == "cuda" else "cpu",
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            random_state=seed,
        )
        model.fit(
            np.hstack([x_train, c_train[:, h]]), y_train[:, h],
            eval_set=[(np.hstack([x_val, c_val[:, h]]), y_val[:, h])],
            verbose=False,
        )
        predicted = np.sort(model.predict(np.hstack([x_test, c_test[:, h]])), axis=-1)
        quantile[:, h] = predicted
        point[:, h] = predicted[:, QUANTILES.index(0.5)]
        rounds.append(int(model.best_iteration) + 1)
    return {"point": point, "quantile": quantile, "rounds_per_horizon": rounds}


def run(kind: str, config: Path, region: str, seed: int, output_dir: Path, device_name: str,
        epochs: int, learning_rate: float | None = None) -> dict:
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    datasets = build_region_datasets(config, region)
    start = time.time()
    if kind == "xgboost":
        result = _xgboost_run(datasets, seed, device)
    else:
        result = _neural_run(kind, datasets, seed, device, epochs, learning_rate)
    test = datasets["test"]
    origins = np.asarray(test.origin_indices)
    raw = np.asarray(test.target_values_raw, dtype=np.float64)
    actual = np.stack([raw[i:i + test.output_hours] for i in origins])
    metrics = point_metrics(result["point"], actual) | probabilistic_metrics(
        result["quantile"], actual, list(QUANTILES)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{region}.npz",
        point=result["point"],
        quantile=result["quantile"],
        actual=actual,
        origin_unix=test.delivery_unix_seconds[origins],
    )
    summary = {
        "baseline": kind,
        "config": str(config),
        "region": region,
        "seed": seed,
        "seconds": round(time.time() - start, 1),
        "test": metrics,
        **{k: v for k, v in result.items() if k not in ("point", "quantile")},
    }
    (output_dir / f"{region}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=("xgboost", "gru", "deepar", "dlinear", "patchtst", "itransformer", "informer"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--region", required=True, choices=("NSW1", "QLD1", "TAS1"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="defaults to 1e-3, or general_baselines.LEARNING_RATES")
    args = parser.parse_args()
    summary = run(args.baseline, args.config, args.region, args.seed, args.output_dir,
                  args.device, args.epochs, args.learning_rate)
    test = summary["test"]
    print(
        f"{args.baseline} {args.region} seed={args.seed} mae={test['mae']:.4f} "
        f"rmse_window={test['rmse_window_mean']:.4f} crps~={test['crps_quantile_approx']:.4f} "
        f"seconds={summary['seconds']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
