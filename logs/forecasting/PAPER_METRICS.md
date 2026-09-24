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
| 10_additive_nonlinear_head | NSW1 | 17521 | 72.33 | 401.07 | 127.06 | 117.79 | 87.88% | 190.46 | 658.08 | 44.47 |
| 10_additive_nonlinear_head | QLD1 | 17521 | 64.85 | 279.32 | 111.19 | 104.25 | 89.45% | 240.39 | 565.26 | 40.57 |
| 10_additive_nonlinear_head | TAS1 | 17521 | 40.94 | 162.92 | 57.14 | 46.69 | 75.49% | 110.34 | 290.00 | 23.30 |
| 10_additive_nonlinear_head | **Mean** | - | 59.37 | 281.10 | 98.46 | 89.58 | 84.28% | 180.40 | 504.45 | 36.11 |

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
| 10_additive_nonlinear_head | `aemo-additive-nonlinear-head-30epoch` | `3b6981a` | `configs/aemo_forecast_additive_nonlinear_head.yaml` | `outputs/forecasting/additive_nonlinear_head` |
