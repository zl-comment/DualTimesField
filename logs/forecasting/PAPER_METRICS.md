# RE-Price style evaluation of archived checkpoints

This file re-scores the archived checkpoints reachable from this branch with the
metric definitions of RE-Price ([Chen et al., Applied Energy 426 (2026)
128712](https://doi.org/10.1016/j.apenergy.2026.128712)). No model was
retrained. Every experiment was evaluated with the code at its own tag, and the
global MAE and RMSE reproduce each run's recorded `metrics.json` within 0.003
AUD/MWh.

## Protocol alignment

| Item | RE-Price | Local |
|---|---|---|
| Data | AEMO hourly prices, 2015-2024, NSW/QLD/TAS | Same |
| Split | Chronological 70/10/20 | 2015-2021 / 2022 / 2023-2024; 70/10/20 of the same series starts the test split at 2022-12-31 23:00 |
| Window | 72 hours in, 24 hours out | Same |
| Inputs | Price, load forecast, temperature proxy, more than 2,000 WattClarity articles | Price, demand, calendar |
| Price preprocessing | None stated | None |

## Metric definitions

| Metric | Definition |
|---|---|
| MAE | Mean absolute error over all origins and horizons; independent of aggregation order |
| RMSE global | Root mean squared error over all origins and horizons; the metric stored in `metrics.json` |
| RMSE window | RMSE of each 24-hour trajectory, averaged over origins. RE-Price defines SDE over the horizons of one sample (eq. 29) and reports RMSE almost equal to SDE, so this is the closest reading of its RMSE |
| SDE window | Eq. 29 per trajectory with `H - 1` in the denominator, averaged over origins |
| PICP90 / PIAW90 / AIS90 | Central 90% interval from the 0.05 and 0.95 quantiles; AIS follows eqs. 31-32 with gamma 0.1. The 80% values are in the JSON files |
| CRPS~ | Twice the mean pinball loss over the five quantiles 0.05, 0.10, 0.50, 0.90, 0.95. This is a coarse approximation, not RE-Price's CRPS from an adaptive Gamma-kernel density |

NLL is not reported because the local models do not output a density. RE-Price
does not state the nominal level behind its PICP, so interval metrics should be
compared with caution.

## RE-Price reported test results

| Region | MAE | RMSE | SDE | PICP | PIAW | AIS | CRPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| NSW | 23.48 | 34.15 | 34.14 | 84.61% | 81.85 | 125.94 | 17.36 |
| QLD | 25.85 | 37.15 | 36.89 | 81.26% | 97.46 | 152.68 | 20.58 |
| TAS | 18.96 | 22.81 | 21.57 | 90.18% | 80.43 | 89.35 | 13.13 |
| Mean | 22.76 | 31.37 | 30.87 | 85.35% | 86.58 | 122.66 | 17.02 |

## Local test results

| Experiment | Region | Origins | MAE | RMSE global | RMSE window | SDE window | PICP90 | PIAW90 | AIS90 | CRPS~ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01_static_normalization_baseline | NSW1 | 17521 | 60.93 | 400.59 | 114.07 | 105.17 | 83.97% | 174.05 | 635.32 | 41.48 |
| 01_static_normalization_baseline | QLD1 | 17521 | 65.62 | 279.97 | 110.15 | 97.17 | 90.61% | 208.95 | 511.18 | 35.00 |
| 01_static_normalization_baseline | TAS1 | 17521 | 37.13 | 161.69 | 53.03 | 45.28 | 68.66% | 92.27 | 312.51 | 23.81 |
| 01_static_normalization_baseline | **Mean** | - | 54.56 | 280.75 | 92.42 | 82.54 | 81.08% | 158.42 | 486.34 | 33.43 |
| 09_additive_trigonometric_fusion | NSW1 | 17521 | 67.59 | 400.33 | 121.64 | 111.94 | 84.22% | 167.17 | 650.65 | 41.44 |
| 09_additive_trigonometric_fusion | QLD1 | 17521 | 59.56 | 278.28 | 106.47 | 100.76 | 92.38% | 218.04 | 526.33 | 36.53 |
| 09_additive_trigonometric_fusion | TAS1 | 17521 | 37.75 | 161.76 | 53.78 | 46.27 | 78.56% | 110.83 | 279.23 | 22.14 |
| 09_additive_trigonometric_fusion | **Mean** | - | 54.97 | 280.12 | 93.96 | 86.32 | 85.05% | 165.34 | 485.41 | 33.37 |
| 11_additive_tcn_head | NSW1 | 17521 | 65.81 | 401.53 | 119.28 | 109.22 | 68.33% | 108.66 | 730.95 | 45.90 |
| 11_additive_tcn_head | QLD1 | 17521 | 75.24 | 286.07 | 123.85 | 106.81 | 90.09% | 245.44 | 541.74 | 39.05 |
| 11_additive_tcn_head | TAS1 | 17521 | 41.13 | 162.62 | 57.20 | 46.15 | 65.23% | 89.08 | 331.72 | 24.36 |
| 11_additive_tcn_head | **Mean** | - | 60.73 | 283.41 | 100.11 | 87.39 | 74.55% | 147.73 | 534.80 | 36.43 |
| 12_additive_tcn_attention_head | NSW1 | 17521 | 70.91 | 409.32 | 125.67 | 107.44 | 74.03% | 162.67 | 721.23 | 46.49 |
| 12_additive_tcn_attention_head | QLD1 | 17521 | 60.26 | 280.36 | 106.88 | 97.24 | 77.01% | 180.70 | 604.00 | 49.32 |
| 12_additive_tcn_attention_head | TAS1 | 17521 | 41.97 | 163.92 | 57.85 | 44.97 | 69.58% | 93.36 | 323.21 | 25.36 |
| 12_additive_tcn_attention_head | **Mean** | - | 57.71 | 284.53 | 96.80 | 83.22 | 73.54% | 145.58 | 549.48 | 40.39 |
| 13_max_spare_query_injection | NSW1 | 17521 | 74.94 | 395.10 | 127.21 | 112.84 | 78.23% | 141.74 | 647.83 | 42.29 |
| 13_max_spare_query_injection | QLD1 | 17521 | 75.48 | 277.20 | 120.55 | 98.82 | 80.08% | 156.09 | 560.84 | 43.26 |
| 13_max_spare_query_injection | TAS1 | 17521 | 41.85 | 163.37 | 57.73 | 45.72 | 73.84% | 108.11 | 294.22 | 23.56 |
| 13_max_spare_query_injection | **Mean** | - | 64.09 | 278.56 | 101.83 | 85.79 | 77.38% | 135.31 | 500.96 | 36.37 |
| 14_max_spare_dgf_residual_adapter | NSW1 | 17521 | 70.06 | 406.72 | 124.26 | 106.93 | 76.49% | 169.17 | 696.72 | 45.43 |
| 14_max_spare_dgf_residual_adapter | QLD1 | 17521 | 63.20 | 276.83 | 109.37 | 98.15 | 77.81% | 186.43 | 588.51 | 49.03 |
| 14_max_spare_dgf_residual_adapter | TAS1 | 17521 | 41.84 | 163.69 | 57.66 | 44.93 | 70.23% | 94.02 | 316.19 | 25.11 |
| 14_max_spare_dgf_residual_adapter | **Mean** | - | 58.37 | 282.41 | 97.09 | 83.34 | 74.84% | 149.87 | 533.81 | 39.86 |

Seasonal naive (price 24 hours earlier) on the same test windows:

| Protocol | Region | Origins | MAE | RMSE global | RMSE window | SDE window |
|---|---|---:|---:|---:|---:|---:|
| rolling hourly | NSW1 | 17521 | 72.51 | 528.03 | 157.01 | 148.22 |
| rolling hourly | QLD1 | 17521 | 59.17 | 368.82 | 133.54 | 127.35 |
| rolling hourly | TAS1 | 17521 | 39.83 | 222.81 | 67.37 | 61.29 |

Fixed-origin experiments score 731 daily origins per region and are not directly comparable with rolling-hour rows.

## Findings

| Finding | Evidence |
|---|---|
| Aggregation explains most of the RMSE gap | The static baseline's mean RMSE falls from 280.75 (global) to 92.42 (window). RE-Price reports 31.37 |
| Aggregation does not explain the MAE gap | MAE is aggregation-invariant: 54.56 locally versus 22.76 for RE-Price |
| The seasonal naive forecast is a strong floor | On the rolling windows it reaches mean MAE 57.17 and window RMSE 119.31; in QLD1 its MAE (59.17) is below the static baseline's (65.62) |
| RE-Price's MAE is hard to reach on raw prices | In the NSW1 test period, hours above 300 AUD/MWh alone add at least 20.7 to the MAE of any forecast that stays at or below 300 in those hours, versus RE-Price's reported 23.48 overall. The reported values imply either strong spike prediction or data processing not described in the paper |

## Reproduction

Run from the worktree checked out at each experiment's evaluated commit:

```bash
python -m forecasting.evaluate_paper_metrics evaluate --config <config> --checkpoint-dir <checkpoint directory> --output-dir logs/forecasting/paper_metrics/<experiment>
python -m forecasting.evaluate_paper_metrics summarize --result-root logs/forecasting/paper_metrics
```

| Experiment | Tag | Evaluated commit | Config | Checkpoints |
|---|---|---|---|---|
| 01_static_normalization_baseline | `aemo-static-normalization-baseline` | `1bda1f1` | `configs/aemo_forecast.yaml` | `outputs/forecasting/static_train` |
| 09_additive_trigonometric_fusion | `aemo-additive-trigonometric-fusion-30epoch` | `39676db` | `configs/aemo_forecast_additive_trigonometric_gate.yaml` | `outputs/forecasting/additive_trigonometric_gate` |
| 11_additive_tcn_head | `aemo-additive-tcn-head-30epoch` | `556bbfc` | `configs/aemo_forecast_additive_tcn_head.yaml` | `outputs/forecasting/additive_tcn_head` |
| 12_additive_tcn_attention_head | `aemo-additive-tcn-attention-head-30epoch` | `8fe917b` | `configs/aemo_forecast_additive_tcn_attention_head.yaml` | `outputs/forecasting/additive_tcn_attention_head` |
| 13_max_spare_query_injection | `aemo-max-spare-query-injection-30epoch` | `a9d6913` | `configs/aemo_forecast_max_spare.yaml` | `outputs/forecasting/max_spare_future_context` |
| 14_max_spare_dgf_residual_adapter | `aemo-best-max-spare-dgf-residual-adapter-30epoch` | `ad9792a` | `configs/aemo_forecast_max_spare_dgf_adapter.yaml` | `outputs/forecasting/max_spare_dgf_residual_adapter` |
