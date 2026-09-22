# AEMO NSW/QLD/TAS forecasting experiments

This file records the first completed three-region forecasting run. All values under **Local results** were produced by this repository. The RE-Price values are external `paper-reported` references and were not reproduced here.

## Run scope

| Item | Value |
|---|---|
| Regions | NSW1, QLD1, TAS1 |
| Data period | 2015-01-01 00:00 to 2024-12-31 23:00, Australia/Brisbane |
| Train | 2015-01-01 to 2021-12-31 |
| Validation | 2022-01-01 to 2022-12-31 |
| Test | 2023-01-01 to 2024-12-31 |
| History / horizon / stride | 72 hours / 24 hours / 1 hour |
| Random seed | 2026 |
| Result timestamp | 2026-09-21 UTC |

## Reproduction commands

```bash
CUDA_VISIBLE_DEVICES=0 python -m forecasting.train --config configs/aemo_forecast.yaml --region NSW1 2>&1 | tee logs/forecasting/nsw1.log
CUDA_VISIBLE_DEVICES=1 python -m forecasting.train --config configs/aemo_forecast.yaml --region QLD1 2>&1 | tee logs/forecasting/qld1.log
CUDA_VISIBLE_DEVICES=2 python -m forecasting.train --config configs/aemo_forecast.yaml --region TAS1 2>&1 | tee logs/forecasting/tas1.log
```

These commands reproduce the three-GPU layout. The original log files do not record `CUDA_VISIBLE_DEVICES`, so the exact GPU indices used by the completed run are unknown.

## Configuration

Source: [`configs/aemo_forecast.yaml`](../../configs/aemo_forecast.yaml)

| Group | Setting | Value |
|---|---|---|
| Inputs | Historical variables | `rrp_aud_per_mwh`, `total_demand_mw` |
| Inputs | Known future variables | hour, weekday, day-of-year and month sine/cosine; weekend flag; standardized year |
| Target | Variable | `rrp_aud_per_mwh` |
| Model | CTF frequencies / cutoff | 16 / 10 |
| Model | Hidden dimension / layers | 64 / 3 |
| Model | DGF atoms / base sigma | 16 / 0.05 |
| Model | Quantiles | 0.05, 0.10, 0.50, 0.90, 0.95 |
| Loss | Point / quantile / decomposition weights | 1.0 / 1.0 / 1.0 |
| Loss | Smoothness / sparsity weights | 0.001 / 0.001 |
| Loss | Target sparsity / excess weight | 0.3 / 10.0 |
| Training | Batch size / epochs | 128 / 30 |
| Training | Learning rate / weight decay | 0.0003 / 0.0001 |
| Training | Optimizer | AdamW |
| Training | Device / workers | CUDA / 0 |

## Environment

| Python | PyTorch | CUDA runtime | NumPy | pandas | SciPy | scikit-learn |
|---:|---:|---:|---:|---:|---:|---:|
| 3.11.0 | 2.5.1+cu124 | 12.4 | 2.4.6 | 3.0.6 | 1.17.1 | 1.9.1 |

The machine exposes eight NVIDIA L40 GPUs with 46,068 MiB each. The exact physical GPU assigned to each completed run was not persisted.

## Files

| Region | Console log | Metrics | Training history | Best checkpoint |
|---|---|---|---|---|
| NSW1 | [`nsw1.log`](nsw1.log) | [`metrics.json`](../../outputs/forecasting/static_train/NSW1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/static_train/NSW1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/static_train/NSW1/best_model.pt) |
| QLD1 | [`qld1.log`](qld1.log) | [`metrics.json`](../../outputs/forecasting/static_train/QLD1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/static_train/QLD1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/static_train/QLD1/best_model.pt) |
| TAS1 | [`tas1.log`](tas1.log) | [`metrics.json`](../../outputs/forecasting/static_train/TAS1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/static_train/TAS1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/static_train/TAS1/best_model.pt) |

## Local results

### Training curves

![Training and validation total loss](training_loss_curves.png)

![Validation loss components](validation_loss_components.png)

### Dataset sizes and model selection

| Region | Train windows | Validation windows | Test windows | Best epoch | Best validation total loss | Final training total loss |
|---|---:|---:|---:|---:|---:|---:|
| NSW1 | 61,273 | 8,737 | 17,521 | 8/30 | 1.774037 | 1.268737 |
| QLD1 | 61,273 | 8,737 | 17,521 | 6/30 | 6.076369 | 1.202618 |
| TAS1 | 61,273 | 8,737 | 17,521 | 2/30 | 21.200955 | 0.912223 |

`metrics.json` stores zero-based best epochs (7, 5, and 1); the table uses human-readable epoch numbers (8, 6, and 2).

### Validation metrics at the selected checkpoint

| Region | MAE (AUD/MWh) | RMSE (AUD/MWh) | 80% coverage | 80% mean width | 90% coverage | 90% mean width |
|---|---:|---:|---:|---:|---:|---:|
| NSW1 | 56.3191 | 179.3832 | 64.15% | 125.3292 | 85.85% | 218.4768 |
| QLD1 | 96.1624 | 427.3042 | 64.87% | 188.0268 | 90.47% | 355.4294 |
| TAS1 | 62.2210 | 273.1233 | 38.49% | 71.8134 | 65.61% | 142.1390 |

### Test metrics at the selected checkpoint

| Region | MAE (AUD/MWh) | RMSE (AUD/MWh) | 80% coverage | 80% mean width | 90% coverage | 90% mean width |
|---|---:|---:|---:|---:|---:|---:|
| NSW1 | 60.9251 | 400.5916 | 62.55% | 89.1850 | 83.97% | 174.0480 |
| QLD1 | 65.6180 | 279.9669 | 67.37% | 106.1958 | 90.61% | 208.9505 |
| TAS1 | 37.1264 | 161.6860 | 39.37% | 46.2584 | 68.66% | 92.2706 |

### Test loss components

| Region | Total | Point | Quantile | Decomposition | Smoothness | Sparsity |
|---|---:|---:|---:|---:|---:|---:|
| NSW1 | 7.856348 | 5.903404 | 0.125783 | 1.826783 | 0.126397 | 0.252075 |
| QLD1 | 2.728067 | 1.912101 | 0.086434 | 0.729157 | 0.068527 | 0.307171 |
| TAS1 | 7.622208 | 5.533191 | 0.173170 | 1.914390 | 0.075824 | 1.381370 |

## Daily grouped-sampling experiment

This run retains every hourly forecast window but trains on one global local-hour
group per epoch. The selected hour rotates from 00 through 23, so the 24-hour
targets inside an epoch do not overlap. Validation and test remain identical to
the static baseline's full rolling protocol.

| Item | Value |
|---|---|
| Configuration | [`configs/aemo_forecast_daily_grouped_sampling.yaml`](../../configs/aemo_forecast_daily_grouped_sampling.yaml) |
| Full train / validation / test windows | 61,273 / 8,737 / 17,521 per region |
| Windows per training epoch | 2,553 or 2,554 |
| Coverage completed in 30 epochs | One 24-hour rotation plus six hour groups |
| Best epochs, NSW1 / QLD1 / TAS1 | 29 / 30 / 24 |
| GPU mapping | NSW1 / QLD1 / TAS1 on physical GPUs 5 / 6 / 7 |
| Output directory | `outputs/forecasting/daily_grouped_sampling/` |

### Grouped-sampling artifacts

| Region | Console log | Metrics | Training history | Best checkpoint |
|---|---|---|---|---|
| NSW1 | [`nsw1.log`](daily_grouped_sampling/nsw1.log) | [`metrics.json`](../../outputs/forecasting/daily_grouped_sampling/NSW1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/daily_grouped_sampling/NSW1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/daily_grouped_sampling/NSW1/best_model.pt) |
| QLD1 | [`qld1.log`](daily_grouped_sampling/qld1.log) | [`metrics.json`](../../outputs/forecasting/daily_grouped_sampling/QLD1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/daily_grouped_sampling/QLD1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/daily_grouped_sampling/QLD1/best_model.pt) |
| TAS1 | [`tas1.log`](daily_grouped_sampling/tas1.log) | [`metrics.json`](../../outputs/forecasting/daily_grouped_sampling/TAS1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/daily_grouped_sampling/TAS1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/daily_grouped_sampling/TAS1/best_model.pt) |

### Grouped-sampling validation metrics

| Region | MAE | RMSE | 80% coverage | 80% width | 90% coverage | 90% width |
|---|---:|---:|---:|---:|---:|---:|
| NSW1 | 73.9278 | 192.7844 | 50.19% | 108.4290 | 77.17% | 200.4791 |
| QLD1 | 110.0947 | 439.3766 | 51.81% | 141.2803 | 82.45% | 292.5864 |
| TAS1 | 62.8204 | 272.9647 | 39.54% | 73.4055 | 67.04% | 145.5246 |

### Grouped-sampling test metrics

| Region | MAE | RMSE | 80% coverage | 80% width | 90% coverage | 90% width |
|---|---:|---:|---:|---:|---:|---:|
| NSW1 | 74.9601 | 404.2411 | 47.07% | 78.4644 | 76.35% | 168.2612 |
| QLD1 | 75.9566 | 286.0601 | 56.38% | 108.2197 | 85.93% | 237.6447 |
| TAS1 | 38.8590 | 161.9001 | 40.17% | 48.8170 | 70.01% | 98.1377 |

### Change from the static baseline

| Region | MAE | MAE change | RMSE | RMSE change | 80% coverage change | 90% coverage change |
|---|---:|---:|---:|---:|---:|---:|
| NSW1 | 74.9601 | +23.04% | 404.2411 | +0.91% | -15.48 pp | -7.61 pp |
| QLD1 | 75.9566 | +15.76% | 286.0601 | +2.18% | -10.98 pp | -4.69 pp |
| TAS1 | 38.8590 | +4.67% | 161.9001 | +0.13% | +0.80 pp | +1.35 pp |

This 30-epoch run is materially undertrained relative to the baseline: one grouped
epoch contains about one twenty-fourth as many optimizer steps, and the selected
checkpoints occur at epochs 29, 30, and 24 rather than early in training. The run
shows that grouped sampling removes within-epoch target duplication, but it does
not establish that the sampling method harms accuracy. A compute-matched run needs
multiple complete 24-hour rotations or a scheduler defined in rotation cycles.

### Extended grouped-sampling run: 240 epochs, learning rate 0.001

The extended run covers ten complete 24-hour rotations. It also raises the
learning rate from 0.0003 to 0.001, so it measures the joint effect of a larger
training budget and faster optimization rather than epochs alone.

| Item | Value |
|---|---|
| Configuration | [`configs/aemo_forecast_daily_grouped_sampling_extended.yaml`](../../configs/aemo_forecast_daily_grouped_sampling_extended.yaml) |
| Epochs / rotations / learning rate | 240 / 10 / 0.001 |
| Best epochs, NSW1 / QLD1 / TAS1 | 69 / 151 / 182 |
| Output directory | `outputs/forecasting/daily_grouped_sampling_extended/` |

| Region | Console log | Metrics | Training history | Best checkpoint |
|---|---|---|---|---|
| NSW1 | [`nsw1.log`](daily_grouped_sampling_extended/nsw1.log) | [`metrics.json`](../../outputs/forecasting/daily_grouped_sampling_extended/NSW1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/daily_grouped_sampling_extended/NSW1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/daily_grouped_sampling_extended/NSW1/best_model.pt) |
| QLD1 | [`qld1.log`](daily_grouped_sampling_extended/qld1.log) | [`metrics.json`](../../outputs/forecasting/daily_grouped_sampling_extended/QLD1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/daily_grouped_sampling_extended/QLD1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/daily_grouped_sampling_extended/QLD1/best_model.pt) |
| TAS1 | [`tas1.log`](daily_grouped_sampling_extended/tas1.log) | [`metrics.json`](../../outputs/forecasting/daily_grouped_sampling_extended/TAS1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/daily_grouped_sampling_extended/TAS1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/daily_grouped_sampling_extended/TAS1/best_model.pt) |

| Region | Best validation loss | Test MAE | Test RMSE | 80% coverage | 80% width | 90% coverage | 90% width |
|---|---:|---:|---:|---:|---:|---:|---:|
| NSW1 | 1.888758 | 81.0640 | 408.6400 | 48.44% | 84.3306 | 77.93% | 198.3038 |
| QLD1 | 6.275600 | 81.1377 | 284.9892 | 49.55% | 105.5124 | 81.12% | 219.6974 |
| TAS1 | 21.103945 | 39.1098 | 162.3975 | 46.34% | 58.3218 | 76.51% | 117.8816 |

More rotations move checkpoint selection to substantially later epochs and lower
validation loss, confirming that 30 grouped epochs were insufficient. Test MAE,
however, worsens in all three regions relative to the 30-epoch run. The expanded
budget plus higher learning rate therefore does not improve point forecasting and
is retained as a negative hyperparameter result.

## External RE-Price reference

| Region | Local MAE | RE-Price MAE | MAE reduction needed | Local RMSE | RE-Price RMSE | RMSE reduction needed | RE-Price CRPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| NSW | 60.9251 | 23.48 | 61.5% | 400.5916 | 34.15 | 91.5% | 17.36 |
| QLD | 65.6180 | 25.85 | 60.6% | 279.9669 | 37.15 | 86.7% | 20.58 |
| TAS | 37.1264 | 18.96 | 48.9% | 161.6860 | 22.81 | 85.9% | 13.13 |

RE-Price uses price, load forecast, temperature, and more than 2,000 WattClarity news articles. The local pipeline uses price, demand, and known calendar variables without news. The external values are therefore a closely related reference, not a directly comparable local leaderboard. Reference: [Chen et al., Applied Energy (2026), DOI 10.1016/j.apenergy.2026.128712](https://doi.org/10.1016/j.apenergy.2026.128712).

## Interpretation and missing measurements

| Finding | Evidence |
|---|---|
| Extreme errors dominate | Test RMSE is 4.35–6.58 times MAE across the three regions. |
| More epochs alone did not improve selection | The best checkpoints occur at epochs 8, 6, and 2 while training loss continues to decrease. |
| Interval calibration is incomplete | NSW and TAS under-cover both nominal intervals; QLD reaches 90.61% coverage for the nominal 90% interval but under-covers the 80% interval. |
| Probability comparison is incomplete | The local evaluator does not yet calculate CRPS, AIS, NLL, or PIAW under the RE-Price definitions. |
| Replication metadata is incomplete | Start/end timestamps, wall-clock duration, exact GPU mapping, GPU driver, and git revision were not saved by this run. |
