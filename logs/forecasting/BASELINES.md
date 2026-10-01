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

## General time-series baselines

DLinear (Zeng et al., AAAI 2023), PatchTST (Nie et al., ICLR 2023),
iTransformer (Liu et al., ICLR 2024), and Informer (Zhou et al., AAAI 2021),
re-implemented in [`forecasting/general_baselines.py`](../../forecasting/general_baselines.py)
and trained by the same loop as GRU and DeepAR.

| Item | Setting |
|---|---|
| Inputs | Same as every other model: 72-hour price and demand history with calendar; known future calendar, PD PASA spare capacity, soft-saturated net load, and gas-price context. All are used in the "MS" setting of their code bases (multivariate inputs, price target) |
| Known inputs | Informer: extra decoder channels and calendar time features. iTransformer: each history series, calendar feature, and known future series is a token. DLinear and PatchTST are channel-independent and would forecast the price from its own history alone, so both add a linear known-input adapter (flattened known future inputs and demand history to the 24 x 5 outputs), added to the backbone forecast. Their rows are therefore "backbone + adapter" |
| Output and loss | 0.05/0.10/0.50/0.90/0.95 quantiles of the standardized asinh price, pinball loss; the median is the point forecast |
| Architecture | DLinear: moving average 25, 63,720 parameters. PatchTST: patch 16, stride 8, RevIN, d_model 128, 16 heads, 3 layers, 585,330 parameters. iTransformer: non-stationary normalization, d_model 128, 8 heads, 2 layers, 227,448 parameters. Informer: ProbSparse factor 5, d_model 512, 8 heads, 2+1 layers with distilling, label length 36, 11,325,445 parameters |
| Learning rate | Chosen from 1e-4, 5e-4, 1e-3 by the lowest validation pinball loss on raw NSW1, seed 2026, then fixed: DLinear 1e-3, PatchTST 1e-4, iTransformer 1e-3, Informer 1e-3 |
| Training | Adam, batch 256, 30 epochs, weight decay 1e-5, checkpoint with the lowest validation loss; seeds 2026-2028, both price treatments (72 runs, logs in [`baselines/logs/general/`](baselines/logs/general/)) |

Three-seed means, with the detected-event DGF (the trunk) and the RE-Price
baselines for reference:

| raw | MAE (NSW1 / QLD1 / TAS1) | Mean MAE | Window RMSE | 90% PICP | 90% AIS | CRPS~ |
|---|---|---:|---:|---:|---:|---:|
| Detected-event DGF | 48.31 / 42.53 / 33.79 | 41.54 ± 0.06 | 78.80 | 84.19 | 403.48 | 28.15 |
| XGBoost | 50.85 / 44.70 / 33.89 | 43.15 ± 0.07 | 81.65 | 84.48 | 414.32 | 27.69 |
| GRU | 52.35 / 46.21 / 38.39 | 45.65 ± 1.16 | 83.43 | 84.06 | 452.50 | 30.12 |
| DeepAR | 54.24 / 50.84 / 38.51 | 47.86 ± 1.69 | 86.09 | 82.78 | 444.82 | 30.17 |
| DLinear | 48.58 / 41.99 / 32.17 | 40.91 ± 0.02 | 78.59 | 83.81 | 389.26 | 26.31 |
| PatchTST | 48.82 / 42.30 / 31.95 | 41.02 ± 0.03 | 78.78 | 89.32 | 389.71 | 26.22 |
| iTransformer | 51.15 / 45.04 / 33.42 | 43.20 ± 0.06 | 81.43 | 86.94 | 648.84 | 34.36 |
| Informer | 54.92 / 47.81 / 37.20 | 46.65 ± 2.38 | 85.32 | 86.31 | 450.81 | 30.12 |

| capped650 | MAE (NSW1 / QLD1 / TAS1) | Mean MAE | Window RMSE | 90% PICP | 90% AIS | CRPS~ |
|---|---|---:|---:|---:|---:|---:|
| Detected-event DGF | 29.79 / 30.38 / 30.80 | 30.32 ± 0.12 | 41.29 | 84.26 | 213.16 | 17.86 |
| XGBoost | 32.88 / 31.18 / 30.77 | 31.61 ± 0.02 | 43.37 | 83.01 | 201.93 | 16.74 |
| GRU | 34.25 / 34.06 / 35.54 | 34.62 ± 0.62 | 46.09 | 84.72 | 276.51 | 20.39 |
| DeepAR | 37.09 / 38.75 / 35.19 | 37.01 ± 3.02 | 49.03 | 82.36 | 241.93 | 19.64 |
| DLinear | 29.77 / 29.95 / 29.09 | 29.61 ± 0.04 | 40.82 | 83.82 | 209.10 | 16.40 |
| PatchTST | 30.39 / 30.56 / 29.01 | 29.99 ± 0.05 | 41.43 | 89.65 | 208.78 | 16.37 |
| iTransformer | 32.27 / 33.55 / 30.18 | 32.00 ± 0.31 | 44.04 | 86.65 | 339.89 | 21.30 |
| Informer | 34.15 / 33.85 / 34.13 | 34.05 ± 0.48 | 45.85 | 86.46 | 209.97 | 17.58 |

Diebold-Mariano tests (per-origin MAE, seed-averaged, Newey-West lag 48;
positive means the detected-event DGF is worse;
[`significance/raw_general_baselines.json`](significance/raw_general_baselines.json),
[`significance/capped650_general_baselines.json`](significance/capped650_general_baselines.json)):

| Detected-event DGF minus | Raw pooled | Raw NSW1 / QLD1 / TAS1 | Capped pooled | Capped NSW1 / QLD1 / TAS1 |
|---|---|---|---|---|
| DLinear | +0.63 (p = 0.0003) | -0.27 (0.41) / +0.54 (0.002) / +1.63 (<0.001) | +0.72 (p < 0.0001) | +0.02 (0.92) / +0.42 (0.004) / +1.71 (<0.001) |
| PatchTST | +0.52 (p = 0.008) | -0.52 (0.051) / +0.24 (0.30) / +1.84 (<0.001) | +0.34 (p = 0.065) | -0.60 (0.005) / -0.18 (0.37) / +1.79 (<0.001) |
| iTransformer | -1.66 (p < 0.0001) | -2.85 / -2.50 / +0.38 (0.03) | -1.68 (p < 0.0001) | -2.48 / -3.17 / +0.62 (all < 0.001) |
| Informer | -5.10 (p < 0.0001) | -6.62 / -5.28 / -3.41 (all < 0.001) | -3.72 (p < 0.0001) | -4.36 / -3.48 / -3.33 (all < 0.001) |

| Finding | Evidence |
|---|---|
| DLinear and PatchTST with the known-input adapter beat the detected-event DGF on mean MAE, significantly for DLinear | Raw mean MAE 40.91 and 41.02 against 41.54; capped 29.61 and 29.99 against 30.32. Pooled DM p = 0.0003 and 0.008 (raw), < 0.0001 and 0.065 (capped) |
| The gap is mostly TAS1, then QLD1 | TAS1 differences of +1.6 to +1.8 MAE under both treatments. The detected-event DGF is ahead of PatchTST in NSW1 (-0.52 raw, -0.60 capped) and ties DLinear there |
| They are also better probabilistically | Mean CRPS~ 26.31 (DLinear) and 26.22 (PatchTST) against 28.15 on raw prices, below XGBoost's 27.69; mean 90% AIS 389 against 403. Only in NSW1 is the detected-event DGF's AIS lower (519.8 against 532.2 and 528.6) |
| iTransformer and Informer are weaker | iTransformer's point accuracy is close to XGBoost's but its upper quantiles are too wide (raw AIS 648.84). Informer's validation loss is lowest after 1-5 epochs: without instance normalization it overfits the 2015-2021 level and does not follow the 2022 validation regime |

The strongest same-data baselines are now linear or patch-based models on the
price history plus a linear map of the same known inputs. The claim that the
dual-field model has the lowest MAE of all same-data baselines no longer holds;
it holds against XGBoost, GRU, DeepAR, iTransformer, and Informer.

### Attribution: what makes DLinear and PatchTST strong

Variants on the same grid (raw and capped prices, seeds 2026-2028, three
regions), each with the learning rate of the model it modifies. "Adapter" is
the linear known-input adapter; `linear` is DLinear without the trend/seasonal
decomposition (one linear map of the price history); `linear_mse` is `linear`
with a separate point output trained by mean squared error in the target
space, as the dual-field point head is (quantiles keep the pinball loss).

| Variant | Price history | Known inputs | Raw MAE (NSW1 / QLD1 / TAS1) | Raw mean MAE | Capped mean MAE |
|---|---|---|---|---:|---:|
| Detected-event DGF | dual field | field-routed | 48.31 / 42.53 / 33.79 | 41.54 ± 0.06 | 30.32 ± 0.12 |
| `dlinear` | decomposed linear | adapter | 48.58 / 41.99 / 32.17 | 40.91 ± 0.02 | 29.61 ± 0.04 |
| `linear` | linear | adapter | 48.57 / 41.93 / 32.18 | 40.89 ± 0.04 | 29.59 ± 0.04 |
| `linear_mse` | linear, MSE point | adapter | 47.88 / 41.53 / 32.14 | **40.52 ± 0.03** | **29.33 ± 0.06** |
| `patchtst` | PatchTST | adapter | 48.82 / 42.30 / 31.95 | 41.02 ± 0.03 | 29.99 ± 0.05 |
| `known_linear` | none | adapter | 56.64 / 55.44 / 46.74 | 52.94 ± 0.43 | 42.14 ± 0.43 |
| `dlinear_noadapter` | decomposed linear | none | 52.68 / 45.53 / 33.28 | 43.83 ± 0.03 | 32.18 ± 0.05 |
| `patchtst_noadapter` | PatchTST | none | 52.38 / 46.26 / 32.96 | 43.87 ± 0.08 | 32.53 ± 0.06 |

Diebold-Mariano, detected-event DGF minus variant (positive: the detected-event
DGF is worse; [`significance/raw_attribution.json`](significance/raw_attribution.json),
[`significance/capped650_attribution.json`](significance/capped650_attribution.json)):

| Variant | Raw pooled | Raw NSW1 / QLD1 / TAS1 | Capped pooled | Capped NSW1 / QLD1 / TAS1 |
|---|---|---|---|---|
| `linear_mse` | +1.03 (p < 0.0001) | +0.42 (0.08) / +1.01 / +1.66 | +0.99 (p < 0.0001) | +0.52 / +0.74 / +1.71 (all < 0.001) |
| `linear` | +0.65 (p = 0.0002) | -0.26 (0.42) / +0.60 / +1.62 | +0.73 (p < 0.0001) | +0.04 (0.81) / +0.44 / +1.70 |
| `dlinear_noadapter` | -2.28 (p < 0.0001) | -4.37 / -2.99 / +0.52 (0.02) | -1.86 (p < 0.0001) | -3.45 / -2.78 / +0.64 (0.002) |
| `patchtst_noadapter` | -2.32 (p < 0.0001) | -4.08 / -3.73 / +0.83 (0.003) | -2.21 (p < 0.0001) | -3.71 / -3.72 / +0.79 (0.003) |

| Finding | Evidence |
|---|---|
| Neither the decomposition nor the Transformer matters | `linear` equals `dlinear` (40.89 against 40.91) and is ahead of `patchtst` |
| The known inputs and the price history are both needed | Without the adapter, MAE rises by about 2.9 (raw) and 2.6 (capped); without the price history, by 12 |
| The pinball loss is not the reason | The MSE point output is better still: `linear_mse` 40.52 raw, 29.33 capped |
| A linear regression on the same inputs beats the detected-event DGF | `linear_mse` is ahead in every region under both treatments, pooled DM p < 0.0001; the gap is largest in TAS1 (1.7) and QLD1 (0.7-1.0) |
| The dual-field model beats the published price-only DLinear and PatchTST | By 2.3 (raw) and 1.9-2.2 (capped), except in TAS1, where even price-only DLinear is ahead |

Absolute error by actual-price band, seed-ensemble forecasts, raw prices
([`forecasting/error_by_price_band.py`](../../forecasting/error_by_price_band.py);
[`baselines/raw_price_band_detected_vs_linear_mse.json`](baselines/raw_price_band_detected_vs_linear_mse.json),
[`baselines/capped650_price_band_detected_vs_linear_mse.json`](baselines/capped650_price_band_detected_vs_linear_mse.json)):

| Band (AUD/MWh) | Share of hours (NSW1 / QLD1 / TAS1) | Detected-event DGF MAE | `linear_mse` MAE | Share of the total error gap (raw / capped) |
|---|---|---|---|---:|
| below 0 | 6.2 / 13.2 / 7.4% | 50.20 / 30.94 / 35.51 | 46.24 / 28.86 / 35.16 | 18% / 13% |
| 0-100 | 58.3 / 50.6 / 65.8% | 16.21 / 15.37 / 22.74 | 16.76 / 15.87 / 21.79 | 2% / -1% |
| 100-300 | 33.7 / 34.4 / 26.3% | 41.78 / 44.82 / 44.97 | 40.03 / 41.69 / 41.57 | 85% / 91% |
| above 300 | 1.8 / 1.9 / 0.5% | 1217 / 806 / 838 | 1225 / 811 / 817 | -5% / -3% |

The gap is the elevated but ordinary level of 100-300 AUD/MWh and negative
prices, not the spikes: in NSW1 and QLD1 the detected-event DGF is slightly
better above 300 and between 0 and 100. Neither model forecasts spikes (errors
of 800-1,200 AUD/MWh above 300).

The dual-field model's advantage over the general baselines comes from the
known inputs, which a linear regression uses better. Its field structure, as
trained now, costs about 1 MAE relative to that regression.

## Reproduction

```bash
python -m forecasting.baselines --baseline xgboost \
  --config configs/aemo_forecast_pdpasa_netload_softclip_ctf.yaml \
  --region NSW1 --seed 2026 --output-dir outputs/forecasting/baselines/raw/xgboost/seed2026
python -m forecasting.summarize_baselines
python -m forecasting.summarize_baselines --compact   # all seven baselines
scripts/run_general_baselines.sh "6 7" 4             # DLinear, PatchTST, iTransformer, Informer grid
KINDS="linear linear_mse known_linear dlinear_noadapter patchtst_noadapter" scripts/run_general_baselines.sh "6 7" 4
```

Repeat for `--baseline gru` and `deepar`, regions NSW1/QLD1/TAS1, seeds
2026-2028, and `configs/aemo_forecast_capped650_trunk.yaml` with
`outputs/forecasting/baselines/capped650/...`. Each run took 2-8 minutes on
one L40 GPU (one capped TAS1 XGBoost run took 33 minutes on a GPU shared with
another job). Run summaries are in [`baselines/`](baselines/), and console
logs in [`baselines/logs/`](baselines/logs/).
