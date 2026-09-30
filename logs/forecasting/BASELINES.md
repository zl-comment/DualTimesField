# Re-implemented RE-Price baselines on the local data

RE-Price ([Chen et al., Applied Energy 426 (2026)
128712](https://doi.org/10.1016/j.apenergy.2026.128712)) compares against
XGBoost, GRU, and DeepAR, among others. Its reported numbers come from its own
data treatment and give every baseline self-collected news embeddings, so they
are not a same-data comparison. This file re-runs those three baselines on the
same splits, inputs, and scoring as the dual-field trunk.

| Item | Setting |
|---|---|
| Code | [`forecasting/baselines.py`](../../forecasting/baselines.py), tables from [`forecasting/summarize_baselines.py`](../../forecasting/summarize_baselines.py) |
| Inputs, all models | 72-hour price and demand history, calendar, DWGM gas-price context, PD PASA spare capacity, soft-saturated PD PASA net load; no news, no temperature |
| Protocols | Raw RRP (`configs/aemo_forecast_pdpasa_netload_softclip_ctf.yaml`) and RRP capped at 650 AUD/MWh before training and scoring (`configs/aemo_forecast_capped650_trunk.yaml`); see [`PROTOCOL_ALIGNMENT.md`](PROTOCOL_ALIGNMENT.md) |
| Seeds | 2026, 2027, 2028 for every model and protocol (54 baseline runs) |
| Test set | 17,521 rolling hourly origins per region, 2023-2024 |
| Dual-field trunk | Soft-saturated net-load trunk (`bfd5f55`) on raw prices; the same configuration retrained with the 650 cap (`e6c06d3`) on capped prices |
| XGBoost 3.2.0 | One multi-quantile model per horizon (`reg:quantileerror`, quantiles 0.05/0.10/0.50/0.90/0.95), depth 6, learning rate 0.05, subsample and column sample 0.8, up to 2,000 rounds with early stopping on validation; the 0.5 quantile is the point forecast |
| GRU | Two-layer GRU (64 units) over history and its calendar; per-horizon MLP on the final state and known future inputs gives a Gaussian mean and scale; 45,058 parameters |
| DeepAR | Two-layer autoregressive LSTM (64 units), Gaussian likelihood, teacher forcing over the 96-hour window, 200 ancestral sample paths at test time; 54,146 parameters |
| Neural training | Adam, learning rate 0.001, batch 256, 30 epochs, checkpoint with the lowest validation negative log-likelihood |
| Likelihood | RE-Price gives GRU and DeepAR a Gamma likelihood on shifted prices; here the Gaussian is on the standardized asinh price, the trunk's target space, whose inverse is a skewed price distribution that allows negative prices |

Cells are three-seed means; MAE also shows the seed standard deviation.
CRPS~ is twice the mean pinball loss over the five quantiles. Rows marked
"RE-Price paper, with news" are copied from RE-Price Tables 2 and 3; they use
news and an unstated price treatment.

## Raw prices

### NSW1

| Model | MAE | Window RMSE | Window SDE | 90% PICP | 90% PIAW | 90% AIS | CRPS~ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dual-field trunk | 48.53 ± 0.32 | 100.10 | 93.51 | 88.98 | 131.90 | 572.16 | 35.68 |
| XGBoost | 50.85 ± 0.03 | 103.97 | 94.93 | 84.34 | 159.62 | 563.12 | 35.32 |
| GRU | 52.35 ± 0.87 | 105.26 | 96.12 | 91.13 | 237.16 | 558.80 | 35.75 |
| DeepAR | 54.24 ± 2.55 | 107.31 | 97.89 | 84.58 | 150.34 | 579.76 | 36.74 |
| RE-Price (RE-Price paper, with news) | 23.48 | 34.15 | 34.14 | - | - | - | 17.36 |
| XGBoost (RE-Price paper, with news) | 38.63 | 51.99 | 50.61 | - | - | - | 28.79 |
| GRU (RE-Price paper, with news) | 33.24 | 48.98 | 45.58 | - | - | - | 25.12 |
| DeepAR (RE-Price paper, with news) | 33.77 | 50.20 | 47.87 | - | - | - | 26.25 |

### QLD1

| Model | MAE | Window RMSE | Window SDE | 90% PICP | 90% PIAW | 90% AIS | CRPS~ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dual-field trunk | 42.51 ± 0.49 | 87.26 | 81.82 | 91.51 | 155.02 | 453.94 | 29.96 |
| XGBoost | 44.70 ± 0.19 | 91.50 | 84.31 | 91.22 | 230.33 | 431.18 | 28.79 |
| GRU | 46.21 ± 2.37 | 91.57 | 83.67 | 94.69 | 400.99 | 505.55 | 32.77 |
| DeepAR | 50.84 ± 2.46 | 96.72 | 88.70 | 87.67 | 250.79 | 474.56 | 32.32 |
| RE-Price (RE-Price paper, with news) | 25.85 | 37.15 | 36.89 | - | - | - | 20.58 |
| XGBoost (RE-Price paper, with news) | 44.70 | 64.05 | 61.36 | - | - | - | 29.17 |
| GRU (RE-Price paper, with news) | 43.24 | 52.70 | 43.06 | - | - | - | 28.48 |
| DeepAR (RE-Price paper, with news) | 36.67 | 42.32 | 36.73 | - | - | - | 25.94 |

### TAS1

| Model | MAE | Window RMSE | Window SDE | 90% PICP | 90% PIAW | 90% AIS | CRPS~ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dual-field trunk | 34.55 ± 0.51 | 49.96 | 43.44 | 79.35 | 98.91 | 274.49 | 21.20 |
| XGBoost | 33.89 ± 0.06 | 49.49 | 42.62 | 77.89 | 102.49 | 248.66 | 18.94 |
| GRU | 38.39 ± 1.06 | 53.45 | 44.61 | 66.36 | 83.05 | 293.15 | 21.85 |
| DeepAR | 38.51 ± 0.75 | 54.24 | 45.24 | 76.09 | 105.39 | 280.15 | 21.45 |
| RE-Price (RE-Price paper, with news) | 18.96 | 22.81 | 21.57 | - | - | - | 13.13 |
| XGBoost (RE-Price paper, with news) | 37.56 | 40.33 | 38.96 | - | - | - | 23.43 |
| GRU (RE-Price paper, with news) | 30.40 | 41.80 | 33.94 | - | - | - | 19.66 |
| DeepAR (RE-Price paper, with news) | 26.51 | 41.90 | 35.87 | - | - | - | 20.18 |

### Mean

| Model | MAE | Window RMSE | Window SDE | 90% PICP | 90% PIAW | 90% AIS | CRPS~ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dual-field trunk | 41.86 ± 0.24 | 79.11 | 72.92 | 86.62 | 128.61 | 433.53 | 28.95 |
| XGBoost | 43.15 ± 0.07 | 81.65 | 73.96 | 84.48 | 164.15 | 414.32 | 27.69 |
| GRU | 45.65 ± 1.16 | 83.43 | 74.80 | 84.06 | 240.40 | 452.50 | 30.12 |
| DeepAR | 47.86 ± 1.69 | 86.09 | 77.28 | 82.78 | 168.84 | 444.82 | 30.17 |

## Prices capped at 650 AUD/MWh

### NSW1

| Model | MAE | Window RMSE | Window SDE | 90% PICP | 90% PIAW | 90% AIS | CRPS~ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dual-field trunk | 30.05 ± 0.32 | 43.05 | 38.20 | 88.66 | 125.54 | 213.87 | 17.69 |
| XGBoost | 32.88 ± 0.02 | 47.15 | 39.75 | 82.81 | 104.54 | 214.55 | 17.51 |
| GRU | 34.25 ± 0.67 | 48.32 | 41.03 | 91.18 | 198.65 | 245.96 | 18.93 |
| DeepAR | 37.09 ± 3.68 | 51.40 | 42.80 | 82.61 | 128.61 | 239.44 | 19.59 |
| RE-Price (RE-Price paper, with news) | 23.48 | 34.15 | 34.14 | - | - | - | 17.36 |
| XGBoost (RE-Price paper, with news) | 38.63 | 51.99 | 50.61 | - | - | - | 28.79 |
| GRU (RE-Price paper, with news) | 33.24 | 48.98 | 45.58 | - | - | - | 25.12 |
| DeepAR (RE-Price paper, with news) | 33.77 | 50.20 | 47.87 | - | - | - | 26.25 |

### QLD1

| Model | MAE | Window RMSE | Window SDE | 90% PICP | 90% PIAW | 90% AIS | CRPS~ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dual-field trunk | 30.69 ± 0.11 | 43.69 | 38.13 | 94.92 | 183.69 | 229.95 | 18.12 |
| XGBoost | 31.18 ± 0.03 | 44.86 | 39.08 | 89.40 | 142.55 | 205.10 | 16.87 |
| GRU | 34.06 ± 1.07 | 47.61 | 40.99 | 93.65 | 323.23 | 355.83 | 23.53 |
| DeepAR | 38.75 ± 4.48 | 53.07 | 45.84 | 88.24 | 202.30 | 265.56 | 20.95 |
| RE-Price (RE-Price paper, with news) | 25.85 | 37.15 | 36.89 | - | - | - | 20.58 |
| XGBoost (RE-Price paper, with news) | 44.70 | 64.05 | 61.36 | - | - | - | 29.17 |
| GRU (RE-Price paper, with news) | 43.24 | 52.70 | 43.06 | - | - | - | 28.48 |
| DeepAR (RE-Price paper, with news) | 36.67 | 42.32 | 36.73 | - | - | - | 25.94 |

### TAS1

| Model | MAE | Window RMSE | Window SDE | 90% PICP | 90% PIAW | 90% AIS | CRPS~ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dual-field trunk | 31.94 ± 0.25 | 39.07 | 32.68 | 82.11 | 106.31 | 208.58 | 17.76 |
| XGBoost | 30.77 ± 0.06 | 38.10 | 31.54 | 76.81 | 92.10 | 186.13 | 15.85 |
| GRU | 35.54 ± 1.06 | 42.36 | 33.45 | 69.32 | 86.24 | 227.75 | 18.71 |
| DeepAR | 35.19 ± 1.37 | 42.62 | 33.96 | 76.22 | 102.47 | 220.79 | 18.38 |
| RE-Price (RE-Price paper, with news) | 18.96 | 22.81 | 21.57 | - | - | - | 13.13 |
| XGBoost (RE-Price paper, with news) | 37.56 | 40.33 | 38.96 | - | - | - | 23.43 |
| GRU (RE-Price paper, with news) | 30.40 | 41.80 | 33.94 | - | - | - | 19.66 |
| DeepAR (RE-Price paper, with news) | 26.51 | 41.90 | 35.87 | - | - | - | 20.18 |

### Mean

| Model | MAE | Window RMSE | Window SDE | 90% PICP | 90% PIAW | 90% AIS | CRPS~ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dual-field trunk | 30.89 ± 0.13 | 41.94 | 36.34 | 88.56 | 138.51 | 217.47 | 17.86 |
| XGBoost | 31.61 ± 0.02 | 43.37 | 36.79 | 83.01 | 113.06 | 201.93 | 16.74 |
| GRU | 34.62 ± 0.62 | 46.09 | 38.49 | 84.72 | 202.71 | 276.51 | 20.39 |
| DeepAR | 37.01 ± 3.02 | 49.03 | 40.87 | 82.36 | 144.46 | 241.93 | 19.64 |

## Findings

| Finding | Evidence |
|---|---|
| The dual-field trunk has the best point accuracy on average | Mean MAE 41.86 on raw prices against 43.15 (XGBoost), 45.65 (GRU), and 47.86 (DeepAR); 30.89 on capped prices against 31.61, 34.62, and 37.01. It also has the lowest mean window RMSE and SDE under both protocols |
| It leads in NSW1 and QLD1 but not TAS1 | XGBoost is ahead in TAS1 by 0.66 (raw) and 1.17 (capped) MAE; the trunk leads in NSW1 by 2.32 and 2.83 and in QLD1 by 2.19 and 0.49 |
| XGBoost has the best probabilistic scores | Mean CRPS~ 27.69 against the trunk's 28.95 on raw prices and 16.74 against 17.86 on capped prices; mean 90% AIS 414.32 against 433.53 and 201.93 against 217.47. The trunk's 90% coverage is closer to nominal (86.6% and 88.6% against 84.5% and 83.0%) but its intervals are less sharp where it matters |
| Recurrent baselines are weaker and less stable | GRU and DeepAR trail on every point metric; DeepAR's seed spread of MAE reaches 2.5-4.5 in NSW1 and QLD1 |
| On capped prices, news-free re-implementations are close to or better than RE-Price's reported baselines in NSW1 and QLD1 | Capped NSW1 / QLD1 MAE: XGBoost 32.88 / 31.18 against reported 38.63 / 44.70, GRU 34.25 / 34.06 against 33.24 / 43.24, DeepAR 37.09 / 38.75 against 33.77 / 36.67 |
| TAS1 again does not match | RE-Price reports 26.51 (DeepAR) and 30.40 (GRU) in TAS1, below every local TAS1 model under either protocol (at least 30.77) |

Under the same data and inputs, the dual-field trunk beats all three
re-implemented baselines on mean point accuracy, but not on the probabilistic
scores, where the multi-quantile XGBoost is ahead. The comparison with the
published RE-Price numbers still depends on the unstated price treatment and
on news, which none of these models use.

## Update: detected-event DGF trunk

The research trunk is now the detected-event DGF (`e2cd5f8`). Against the
same re-implemented baselines, three-seed means:

| Raw prices | MAE (NSW1 / QLD1 / TAS1) | Mean MAE | Window RMSE | 90% AIS | CRPS~ |
|---|---|---:|---:|---:|---:|
| Detected-event DGF | **48.31 / 42.53 / 33.79** | **41.54 ± 0.06** | **78.80** | **403.48** | 28.15 |
| XGBoost | 50.85 / 44.70 / 33.89 | 43.15 ± 0.07 | 81.65 | 414.32 | **27.69** |
| GRU | 52.35 / 46.21 / 38.39 | 45.65 ± 1.16 | 83.43 | 452.50 | 30.12 |
| DeepAR | 54.24 / 50.84 / 38.51 | 47.86 ± 1.69 | 86.09 | 444.82 | 30.17 |

| Capped at 650 | MAE (NSW1 / QLD1 / TAS1) | Mean MAE | Window RMSE | 90% AIS | CRPS~ |
|---|---|---:|---:|---:|---:|
| Detected-event DGF | **29.79 / 30.38** / 30.80 | **30.32 ± 0.12** | **41.29** | 213.16 | 17.86 |
| XGBoost | 32.88 / 31.18 / **30.77** | 31.61 ± 0.02 | 43.37 | **201.93** | **16.74** |
| GRU | 34.25 / 34.06 / 35.54 | 34.62 ± 0.62 | 46.09 | 276.51 | 20.39 |
| DeepAR | 37.09 / 38.75 / 35.19 | 37.01 ± 3.02 | 49.03 | 241.93 | 19.64 |

The trunk now has the lowest MAE in every region on raw prices, and in NSW1
and QLD1 on capped prices. Pooled Diebold-Mariano tests favour it over every
baseline under both treatments (p < 0.0001). The TAS1 differences to XGBoost
(-0.09 raw, +0.03 capped) are not significant. XGBoost keeps the best CRPS~,
and the best AIS on capped prices. Details are in `EXPERIMENT_RESULTS.md`,
"Step 5".

## Reproduction

```bash
python -m forecasting.baselines --baseline xgboost \
  --config configs/aemo_forecast_pdpasa_netload_softclip_ctf.yaml \
  --region NSW1 --seed 2026 --output-dir outputs/forecasting/baselines/raw/xgboost/seed2026
python -m forecasting.summarize_baselines
```

Repeat for `--baseline gru` and `deepar`, regions NSW1/QLD1/TAS1, seeds
2026-2028, and `configs/aemo_forecast_capped650_trunk.yaml` with
`outputs/forecasting/baselines/capped650/...`. Each run took 2-8 minutes on
one L40 GPU (one capped TAS1 XGBoost run took 33 minutes on a GPU shared with
another job). Run summaries are in [`baselines/`](baselines/), and console
logs in [`baselines/logs/`](baselines/logs/).
