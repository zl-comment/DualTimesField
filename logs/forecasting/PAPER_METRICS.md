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
| 06_daily_grouped_30epoch | NSW1 | 17521 | 74.96 | 404.24 | 132.45 | 125.18 | 76.35% | 168.26 | 738.04 | 47.97 |
| 06_daily_grouped_30epoch | QLD1 | 17521 | 75.96 | 286.06 | 127.05 | 120.03 | 85.93% | 237.64 | 608.39 | 43.58 |
| 06_daily_grouped_30epoch | TAS1 | 17521 | 38.86 | 161.90 | 55.13 | 47.84 | 70.01% | 98.14 | 318.40 | 24.61 |
| 06_daily_grouped_30epoch | **Mean** | - | 63.26 | 284.07 | 104.88 | 97.68 | 77.43% | 168.01 | 554.94 | 38.72 |
| 07_daily_grouped_240epoch | NSW1 | 17521 | 81.06 | 408.64 | 138.83 | 130.03 | 77.93% | 198.30 | 724.35 | 47.17 |
| 07_daily_grouped_240epoch | QLD1 | 17521 | 81.14 | 284.99 | 131.52 | 122.62 | 81.12% | 219.70 | 599.92 | 43.14 |
| 07_daily_grouped_240epoch | TAS1 | 17521 | 39.11 | 162.40 | 55.55 | 48.10 | 76.51% | 117.88 | 301.12 | 23.67 |
| 07_daily_grouped_240epoch | **Mean** | - | 67.10 | 285.34 | 108.63 | 100.25 | 78.52% | 178.63 | 541.80 | 37.99 |

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
| 06_daily_grouped_30epoch | `aemo-daily-grouped-sampling-30epoch` | `c9725b9` | `configs/aemo_forecast_daily_grouped_sampling.yaml` | `outputs/forecasting/daily_grouped_sampling` |
| 07_daily_grouped_240epoch | `aemo-daily-grouped-sampling-240epoch-lr1e-3` | `0c52d73` | `configs/aemo_forecast_daily_grouped_sampling_extended.yaml` | `outputs/forecasting/daily_grouped_sampling_extended` |
