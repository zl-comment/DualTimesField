# Full minus XGBoost: decomposition (static protocol, seed means)

input: logs/forecasting/exploration_20261011/reports/report.json.gz
sha256: 07ade862cf1a556ea9b1f89681d39065e41254b578fb5bdd782515c7fca215f9

## Macro and regional gaps (Full - XGBoost; positive = Full worse)

| metric | Full | XGBoost | macro gap | NSW1 (share of macro) | QLD1 (share of macro) | TAS1 (share of macro) |
|---|---|---|---|---|---|---|
| mae | 38.923 | 37.968 | +0.955 | +0.522 (18%) | +0.054 (2%) | +2.289 (80%) |
| crps_quantile_approx | 25.936 | 25.527 | +0.409 | +0.956 (78%) | -0.219 (-18%) | +0.491 (40%) |

## Price-band contributions to the regional gap (weight x slice gap)

| metric | region | actual band | obs share | Full | XGBoost | contribution | share of macro gap |
|---|---|---|---|---|---|---|---|
| mae | NSW1 | negative | 6.2% | 45.53 | 46.31 | -0.048 | -2% |
| mae | NSW1 | 0_to_100 | 58.3% | 14.00 | 13.32 | +0.394 | +14% |
| mae | NSW1 | 100_to_300 | 33.7% | 39.13 | 36.10 | +1.019 | +36% |
| mae | NSW1 | spike | 1.8% | 1220.97 | 1269.05 | -0.842 | -29% |
| mae | QLD1 | negative | 13.2% | 30.77 | 30.59 | +0.024 | +1% |
| mae | QLD1 | 0_to_100 | 50.6% | 14.61 | 13.21 | +0.706 | +25% |
| mae | QLD1 | 100_to_300 | 34.3% | 39.90 | 38.19 | +0.586 | +20% |
| mae | QLD1 | spike | 1.8% | 801.98 | 870.21 | -1.262 | -44% |
| mae | TAS1 | negative | 7.4% | 32.55 | 31.12 | +0.105 | +4% |
| mae | TAS1 | 0_to_100 | 65.8% | 18.61 | 16.50 | +1.390 | +49% |
| mae | TAS1 | 100_to_300 | 26.3% | 47.04 | 44.01 | +0.796 | +28% |
| mae | TAS1 | spike | 0.5% | 842.76 | 843.29 | -0.003 | -0% |
| crps_quantile_approx | NSW1 | negative | 6.2% | 26.16 | 31.28 | -0.317 | -26% |
| crps_quantile_approx | NSW1 | 0_to_100 | 58.3% | 10.17 | 7.58 | +1.515 | +124% |
| crps_quantile_approx | NSW1 | 100_to_300 | 33.7% | 25.75 | 21.85 | +1.315 | +107% |
| crps_quantile_approx | NSW1 | spike | 1.8% | 1001.98 | 1090.82 | -1.557 | -127% |
| crps_quantile_approx | QLD1 | negative | 13.2% | 12.91 | 17.34 | -0.585 | -48% |
| crps_quantile_approx | QLD1 | 0_to_100 | 50.6% | 10.21 | 8.94 | +0.643 | +52% |
| crps_quantile_approx | QLD1 | 100_to_300 | 34.3% | 25.39 | 25.48 | -0.030 | -2% |
| crps_quantile_approx | QLD1 | spike | 1.8% | 602.69 | 616.07 | -0.247 | -20% |
| crps_quantile_approx | TAS1 | negative | 7.4% | 15.28 | 19.22 | -0.291 | -24% |
| crps_quantile_approx | TAS1 | 0_to_100 | 65.8% | 9.78 | 9.21 | +0.377 | +31% |
| crps_quantile_approx | TAS1 | 100_to_300 | 26.3% | 22.14 | 21.12 | +0.270 | +22% |
| crps_quantile_approx | TAS1 | spike | 0.5% | 783.33 | 756.22 | +0.135 | +11% |

## Macro band contribution (mean over regions)

| metric | negative | 0_to_100 | 100_to_300 | spike | sum |
|---|---|---|---|---|---|
| mae | +0.027 | +0.830 | +0.800 | -0.702 | +0.955 |
| crps_quantile_approx | -0.398 | +0.845 | +0.518 | -0.557 | +0.409 |

## Horizon blocks (macro contribution over regions, hours 1-6/7-12/13-18/19-24)

| metric | 1-6 | 7-12 | 13-18 | 19-24 |
|---|---|---|---|---|
| mae | +0.315 | +0.253 | +0.267 | +0.120 |
| crps_quantile_approx | +0.093 | +0.071 | +0.159 | +0.087 |

## Per-horizon slice gap (Full-XGBoost, macro over regions)

| horizon | MAE gap | CRPS~ gap |
|---|---|---|
| 1 | +2.235 | +1.140 |
| 2 | +1.204 | +0.245 |
| 3 | +0.721 | +0.180 |
| 4 | +0.916 | -0.113 |
| 5 | +1.099 | +0.289 |
| 6 | +1.383 | +0.484 |
| 7 | +0.767 | +0.116 |
| 8 | +0.969 | +0.028 |
| 9 | +1.186 | +0.279 |
| 10 | +0.899 | +0.487 |
| 11 | +1.308 | +0.382 |
| 12 | +0.954 | +0.412 |
| 13 | +1.217 | +0.735 |
| 14 | +1.153 | +0.659 |
| 15 | +1.349 | +0.736 |
| 16 | +0.763 | +0.487 |
| 17 | +0.748 | +0.412 |
| 18 | +1.170 | +0.778 |
| 19 | +1.305 | +0.638 |
| 20 | +0.710 | +0.186 |
| 21 | +0.726 | +0.437 |
| 22 | -0.005 | +0.197 |
| 23 | +0.311 | +0.305 |
| 24 | -0.169 | +0.315 |

## Top slices by macro contribution (positive = Full worse)

### mae / months: top 8 positive slices cover 47% of the macro gap; all positive slices sum to 134%

| region | slice | macro contribution | share |
|---|---|---|---|
| QLD1 | 2024-10 | +0.0918 | +9.6% |
| TAS1 | 2024-11 | +0.0608 | +6.4% |
| TAS1 | 2024-09 | +0.0562 | +5.9% |
| TAS1 | 2023-03 | +0.0549 | +5.7% |
| TAS1 | 2023-01 | +0.0490 | +5.1% |
| TAS1 | 2023-02 | +0.0475 | +5.0% |
| TAS1 | 2024-10 | +0.0456 | +4.8% |
| NSW1 | 2024-11 | +0.0455 | +4.8% |

### mae / price_bands: top 8 positive slices cover 175% of the macro gap; all positive slices sum to 175%

| region | slice | macro contribution | share |
|---|---|---|---|
| TAS1 | 0_to_100 | +0.4633 | +48.5% |
| NSW1 | 100_to_300 | +0.3397 | +35.6% |
| TAS1 | 100_to_300 | +0.2653 | +27.8% |
| QLD1 | 0_to_100 | +0.2353 | +24.6% |
| QLD1 | 100_to_300 | +0.1954 | +20.5% |
| NSW1 | 0_to_100 | +0.1313 | +13.7% |
| TAS1 | negative | +0.0351 | +3.7% |
| QLD1 | negative | +0.0078 | +0.8% |

### crps_quantile_approx / months: top 8 positive slices cover 69% of the macro gap; all positive slices sum to 210%

| region | slice | macro contribution | share |
|---|---|---|---|
| NSW1 | 2023-07 | +0.0404 | +9.9% |
| NSW1 | 2023-12 | +0.0381 | +9.3% |
| QLD1 | 2024-05 | +0.0375 | +9.2% |
| NSW1 | 2023-05 | +0.0362 | +8.8% |
| NSW1 | 2024-10 | +0.0352 | +8.6% |
| NSW1 | 2024-03 | +0.0347 | +8.5% |
| NSW1 | 2023-04 | +0.0334 | +8.2% |
| NSW1 | 2023-06 | +0.0282 | +6.9% |

### crps_quantile_approx / price_bands: top 8 positive slices cover 324% of the macro gap; all positive slices sum to 347%

| region | slice | macro contribution | share |
|---|---|---|---|
| NSW1 | 0_to_100 | +0.5050 | +123.5% |
| NSW1 | 100_to_300 | +0.4382 | +107.2% |
| QLD1 | 0_to_100 | +0.2143 | +52.4% |
| TAS1 | 0_to_100 | +0.1257 | +30.7% |
| TAS1 | 100_to_300 | +0.0900 | +22.0% |
| TAS1 | spike | +0.0449 | +11.0% |
| QLD1 | 100_to_300 | -0.0099 | -2.4% |
| QLD1 | spike | -0.0825 | -20.2% |

## Validation (2022) vs test gap (Full-XGBoost, unclipped, seed means)

| region | metric | Full val | XGB val | val gap | test gap |
|---|---|---|---|---|---|
| NSW1 | mae | 75.30 | 64.09 | +11.21 | +0.52 |
| NSW1 | crps_quantile_approx | 42.31 | 37.80 | +4.51 | +0.96 |
| QLD1 | mae | 97.38 | 93.08 | +4.31 | +0.05 |
| QLD1 | crps_quantile_approx | 62.04 | 60.67 | +1.37 | -0.22 |
| TAS1 | mae | 56.12 | 56.45 | -0.33 | +2.29 |
| TAS1 | crps_quantile_approx | 35.92 | 33.35 | +2.57 | +0.49 |

## Calibration / tail descriptors (seed means)

| region | metric | Full | XGBoost |
|---|---|---|---|
| NSW1 | coverage_90 | 0.870 | 0.836 |
| NSW1 | interval_width_90 | 183.590 | 151.762 |
| NSW1 | ais_90 | 517.617 | 540.183 |
| NSW1 | crps_negative | 26.155 | 31.282 |
| NSW1 | crps_spike | 1001.976 | 1090.819 |
| NSW1 | pinball_q05_negative_hours | 10.565 | 18.198 |
| NSW1 | pinball_q95_spike_hours | 804.413 | 912.108 |
| NSW1 | rmse_window_mean | 96.640 | 98.060 |
| QLD1 | coverage_90 | 0.915 | 0.912 |
| QLD1 | interval_width_90 | 190.485 | 239.148 |
| QLD1 | ais_90 | 368.654 | 422.792 |
| QLD1 | crps_negative | 12.911 | 17.342 |
| QLD1 | crps_spike | 602.688 | 616.068 |
| QLD1 | pinball_q05_negative_hours | 3.630 | 5.239 |
| QLD1 | pinball_q95_spike_hours | 418.784 | 434.787 |
| QLD1 | rmse_window_mean | 84.583 | 86.378 |
| TAS1 | coverage_90 | 0.866 | 0.805 |
| TAS1 | interval_width_90 | 101.371 | 97.959 |
| TAS1 | ais_90 | 223.180 | 230.503 |
| TAS1 | crps_negative | 15.277 | 19.225 |
| TAS1 | crps_spike | 783.329 | 756.220 |
| TAS1 | pinball_q05_negative_hours | 5.887 | 9.388 |
| TAS1 | pinball_q95_spike_hours | 698.506 | 668.381 |
| TAS1 | rmse_window_mean | 46.641 | 44.663 |

## Seed dispersion (macro MAE / CRPS~ per seed)

| model | seed | MAE | CRPS~ |
|---|---|---|---|
| full | 2026 | 38.788 | 25.930 |
| full | 2027 | 39.263 | 25.955 |
| full | 2028 | 38.718 | 25.923 |
| xgboost | 2026 | 37.990 | 25.519 |
| xgboost | 2027 | 37.955 | 25.544 |
| xgboost | 2028 | 37.958 | 25.519 |

## Per-region per-seed MAE

| model | region | seed 2026 | seed 2027 | seed 2028 |
|---|---|---|---|---|
| full | NSW1 | 46.025 | 45.402 | 45.289 |
| full | QLD1 | 39.170 | 40.310 | 40.488 |
| full | TAS1 | 31.168 | 32.076 | 30.377 |
| xgboost | NSW1 | 45.081 | 45.021 | 45.047 |
| xgboost | QLD1 | 39.928 | 39.944 | 39.935 |
| xgboost | TAS1 | 28.961 | 28.901 | 28.894 |

## Earlier tracked runs of the same architecture family (not the frozen run): training curves

Validation MAE per epoch is not monotone although training loss is; the epoch minimising validation MAE usually differs from the one minimising total loss.

| run | region | best ep (val MAE) | best ep (val total) | val MAE min | max | SD ep>=4 | mean |d epoch| | final loss share point/quantile/decomp % |
|---|---|---|---|---|---|---|---|---|
| pdpasa_dgf | NSW1 | 3 | 16 | 60.2 | 74.7 | 3.37 | 3.10 | 58/8/34 |
| pdpasa_dgf | QLD1 | 4 | 13 | 84.0 | 96.2 | 3.67 | 2.08 | 60/8/32 |
| pdpasa_dgf | TAS1 | 2 | 21 | 55.4 | 59.3 | 0.85 | 0.67 | 54/17/29 |
| pdpasa_dgf_seed2027 | NSW1 | 4 | 6 | 57.1 | 73.7 | 4.31 | 2.68 | 58/8/34 |
| pdpasa_dgf_seed2027 | QLD1 | 4 | 4 | 82.2 | 94.0 | 2.94 | 2.85 | 59/8/32 |
| pdpasa_dgf_seed2027 | TAS1 | 17 | 24 | 55.2 | 59.4 | 1.04 | 1.01 | 54/16/30 |
| pdpasa_dgf_seed2028 | NSW1 | 2 | 15 | 61.7 | 70.1 | 2.13 | 2.42 | 58/8/34 |
| pdpasa_dgf_seed2028 | QLD1 | 5 | 6 | 84.4 | 97.1 | 3.42 | 1.73 | 59/8/32 |
| pdpasa_dgf_seed2028 | TAS1 | 11 | 24 | 56.0 | 60.6 | 1.23 | 0.91 | 54/17/30 |
| gas_price_ctf | NSW1 | 3 | 3 | 61.3 | 74.4 | 3.13 | 3.19 | 59/8/33 |
| gas_price_ctf | QLD1 | 4 | 13 | 82.8 | 97.7 | 4.00 | 2.16 | 61/8/31 |
| gas_price_ctf | TAS1 | 2 | 5 | 56.8 | 61.9 | 1.09 | 0.92 | 54/17/29 |
| gas_price_ctf_seed2027 | NSW1 | 4 | 7 | 55.2 | 64.8 | 2.32 | 2.50 | 60/8/32 |
| gas_price_ctf_seed2027 | QLD1 | 7 | 7 | 80.4 | 117.1 | 7.58 | 6.28 | 60/8/31 |
| gas_price_ctf_seed2027 | TAS1 | 17 | 24 | 56.8 | 61.8 | 1.18 | 1.26 | 54/16/29 |
| gas_price_ctf_seed2028 | NSW1 | 2 | 15 | 61.4 | 82.1 | 5.80 | 3.17 | 60/8/32 |
| gas_price_ctf_seed2028 | QLD1 | 2 | 6 | 86.1 | 95.1 | 2.34 | 1.68 | 61/8/31 |
| gas_price_ctf_seed2028 | TAS1 | 6 | 24 | 57.4 | 63.2 | 1.36 | 1.27 | 54/17/29 |
| residual_skip_path | NSW1 | 3 | 2 | 63.2 | 84.3 | 4.25 | 3.78 | 59/8/33 |
| residual_skip_path | QLD1 | 4 | 4 | 85.8 | 102.6 | 4.10 | 3.37 | 61/8/31 |
| residual_skip_path | TAS1 | 2 | 2 | 57.1 | 64.2 | 1.36 | 1.17 | 54/17/29 |
| residual_skip_path_seed2027 | NSW1 | 1 | 1 | 59.2 | 73.3 | 3.13 | 3.12 | 60/8/32 |
| residual_skip_path_seed2027 | QLD1 | 7 | 7 | 82.6 | 118.3 | 7.40 | 6.36 | 61/8/31 |
| residual_skip_path_seed2027 | TAS1 | 17 | 24 | 57.9 | 62.7 | 1.09 | 1.25 | 54/16/29 |
| residual_skip_path_seed2028 | NSW1 | 2 | 9 | 66.4 | 91.4 | 6.77 | 3.90 | 59/8/32 |
| residual_skip_path_seed2028 | QLD1 | 2 | 28 | 88.8 | 101.6 | 3.37 | 3.09 | 61/8/31 |
| residual_skip_path_seed2028 | TAS1 | 1 | 11 | 58.5 | 64.3 | 1.21 | 1.04 | 54/17/29 |

## Macro Full-XGBoost gap by protocol (positive = Full worse)

| protocol | MAE gap | CRPS~ gap | window-RMSE gap |
|---|---|---|---|
| static | +0.955 | +0.409 | -0.412 |
| rolling | +1.636 | +1.643 | +0.825 |
| capped650 | +1.547 | +0.190 | +1.318 |

## Static variants minus XGBoost: macro gap and band contributions

| variant | metric | macro gap | negative | 0_to_100 | 100_to_300 | spike |
|---|---|---|---|---|---|---|
| full | mae | +0.955 | +0.027 | +0.830 | +0.800 | -0.702 |
| full | crps_quantile_approx | +0.409 | -0.398 | +0.845 | +0.518 | -0.557 |
| no_tail | mae | +0.969 | -0.081 | +0.798 | +0.918 | -0.666 |
| no_tail | crps_quantile_approx | +0.738 | -0.239 | +0.536 | +0.115 | +0.327 |
| raw_history | mae | +1.615 | -0.192 | +0.747 | +1.663 | -0.603 |
| raw_history | crps_quantile_approx | +0.492 | -0.352 | +0.769 | +0.631 | -0.556 |
| no_fields | mae | +2.050 | -0.065 | +0.649 | +1.966 | -0.500 |
| no_fields | crps_quantile_approx | +1.161 | -0.144 | +0.787 | +0.750 | -0.232 |
| no_pd | mae | +4.905 | +0.340 | +2.677 | +2.450 | -0.561 |
| no_pd | crps_quantile_approx | +2.418 | -0.427 | +1.719 | +1.148 | -0.022 |
| linear | mae | +3.708 | +0.033 | +1.858 | +1.419 | +0.399 |
| linear | crps_quantile_approx | +1.328 | -0.076 | +0.769 | -0.152 | +0.787 |
| dlinear | mae | +3.863 | -0.010 | +1.981 | +1.487 | +0.405 |
| dlinear | crps_quantile_approx | +1.489 | -0.094 | +0.793 | -0.118 | +0.908 |
| gatets_adapter | mae | +3.737 | +0.333 | +1.795 | +1.327 | +0.282 |
| gatets_adapter | crps_quantile_approx | +0.783 | -0.050 | +0.902 | -0.202 | +0.134 |
| pd_calibrator | mae | +1.542 | +0.050 | +0.788 | +1.329 | -0.625 |
| pd_calibrator | crps_quantile_approx | +1.038 | -0.148 | +0.583 | -0.085 | +0.687 |

## Breadth: region-slices where Full is worse than XGBoost

| metric/family | slices | Full worse |
|---|---|---|
| mae/months | 72 | 51 |
| mae/horizons | 72 | 58 |
| crps_quantile_approx/months | 72 | 56 |
| crps_quantile_approx/horizons | 72 | 57 |

## Squared-error comparison (Full better on RMSE in NSW1 and QLD1)

| region | Full window RMSE | XGB window RMSE | Full pooled RMSE | XGB pooled RMSE |
|---|---|---|---|---|
| NSW1 | 96.64 | 98.06 | 388.24 | 399.11 |
| QLD1 | 84.58 | 86.38 | 261.74 | 275.38 |
| TAS1 | 46.64 | 44.66 | 158.92 | 158.79 |

## Coverage and width of the 90% interval within realised-price bands (descriptive only)

| region | band | cov90 Full | cov90 XGB | width90 Full | width90 XGB |
|---|---|---|---|---|---|
| NSW1 | negative | 0.512 | 0.162 | 100.8 | 94.6 |
| NSW1 | 0_to_100 | 0.893 | 0.892 | 101.2 | 89.8 |
| NSW1 | 100_to_300 | 0.907 | 0.873 | 292.1 | 231.2 |
| NSW1 | spike | 0.672 | 0.625 | 1132.7 | 887.9 |
| QLD1 | negative | 0.910 | 0.768 | 123.4 | 136.6 |
| QLD1 | 0_to_100 | 0.923 | 0.935 | 111.3 | 144.5 |
| QLD1 | 100_to_300 | 0.918 | 0.942 | 274.0 | 368.8 |
| QLD1 | spike | 0.673 | 0.741 | 1284.8 | 1154.0 |
| TAS1 | negative | 0.776 | 0.528 | 89.7 | 87.4 |
| TAS1 | 0_to_100 | 0.908 | 0.836 | 89.0 | 78.4 |
| TAS1 | 100_to_300 | 0.795 | 0.807 | 132.8 | 144.7 |
| TAS1 | spike | 0.367 | 0.639 | 250.3 | 372.4 |

## Gap by half-year of the test period (macro contribution; positive = Full worse)

| metric | 2023H1 | 2023H2 | 2024H1 | 2024H2 |
|---|---|---|---|---|
| mae | +0.290 | +0.111 | +0.204 | +0.350 |
| crps_quantile_approx | +0.259 | +0.181 | +0.046 | -0.077 |

## Absolute macro scores by protocol

| protocol | Full MAE | XGB MAE | Full CRPS~ | XGB CRPS~ |
|---|---|---|---|---|
| static | 38.923 | 37.968 | 25.936 | 25.527 |
| rolling | 36.868 | 35.232 | 24.749 | 23.106 |
| capped650 | 28.005 | 26.457 | 14.742 | 14.552 |

## MAE seed spread per region (static)

| region | Full min | Full max | XGB min | XGB max | Full seeds (of 3) below XGB mean |
|---|---|---|---|---|---|
| NSW1 | 45.289 | 46.025 | 45.021 | 45.081 | 0 |
| QLD1 | 39.170 | 40.488 | 39.928 | 39.944 | 1 |
| TAS1 | 30.377 | 32.076 | 28.894 | 28.961 | 0 |

Linear -> XGBoost macro MAE gap recovered by Full: linear 41.676, Full 38.923, XGBoost 37.968, recovered 74%

## Paired block bootstrap in the report (candidate XGBoost minus Full; negative favours XGBoost)

| block h | metric | region | diff | 95% CI | Holm p |
|---|---|---|---|---|---|
| 168 | mae | NSW1 | -0.522 | [-1.007, -0.059] | 1.0000 |
| 168 | mae | QLD1 | -0.054 | [-0.677, +0.577] | 1.0000 |
| 168 | mae | TAS1 | -2.289 | [-2.626, -1.958] | 0.0492 |
| 168 | mae | macro_regions | -0.955 | [-1.317, -0.596] | 0.0492 |
| 168 | crps_quantile_approx | NSW1 | -0.956 | [-1.635, -0.156] | 1.0000 |
| 168 | crps_quantile_approx | QLD1 | +0.219 | [-0.477, +0.993] | 1.0000 |
| 168 | crps_quantile_approx | TAS1 | -0.491 | [-0.677, -0.302] | 0.0492 |
| 168 | crps_quantile_approx | macro_regions | -0.409 | [-0.803, +0.044] | 1.0000 |
| 24 | mae | NSW1 | -0.522 | [-0.959, -0.090] | 1.0000 |
| 24 | mae | QLD1 | -0.054 | [-0.596, +0.484] | 1.0000 |
| 24 | mae | TAS1 | -2.289 | [-2.487, -2.086] | 0.0120 |
| 24 | mae | macro_regions | -0.955 | [-1.251, -0.656] | 0.0120 |
| 24 | crps_quantile_approx | NSW1 | -0.956 | [-1.522, -0.303] | 0.2370 |
| 24 | crps_quantile_approx | QLD1 | +0.219 | [-0.315, +0.866] | 1.0000 |
| 24 | crps_quantile_approx | TAS1 | -0.491 | [-0.638, -0.348] | 0.0120 |
| 24 | crps_quantile_approx | macro_regions | -0.409 | [-0.721, -0.039] | 1.0000 |

## Earlier tracked Full-family runs: epoch behaviour and selection-rule sensitivity

- 9 runs x 3 regions = 27 curves; validation-MAE-optimal epoch <= 5 in 20 of 27 (median epoch 4); best epoch differs between val-MAE and val-total in 20; mean (last - min) val MAE = 9.06; mean val-MAE range over epochs = 13.02.
- 39 stored runs: test MAE of the val-MAE-selected checkpoint vs the val-total-selected checkpoint differ by 0.581 on average (max 6.154); different epoch in 26; MAE rule better on test in 11.


## Internal consistency checks (should be ~0)

max |sum of slice contributions - regional gap| over 18 checks = 1.10e-14
