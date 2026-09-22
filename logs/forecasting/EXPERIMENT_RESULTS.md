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

## Trigonometric gated-fusion experiment

This experiment keeps the static normalization, rolling hourly protocol, split,
loss weights, optimizer, learning rate, epoch budget, and seed unchanged. It
replaces concatenation followed by one linear forecast head with separate linear
CTF and DGF experts followed by a horizon-specific nonlinear gate. The gate uses
`cos(theta)^2` and `sin(theta)^2` weights, initialized to an equal mixture.

| Item | Value |
|---|---|
| Configuration | [`configs/aemo_forecast_trigonometric_gate.yaml`](../../configs/aemo_forecast_trigonometric_gate.yaml) |
| Epochs / learning rate | 30 / 0.0003 |
| Best epochs, NSW1 / QLD1 / TAS1 | 24 / 4 / 29 |
| GPU mapping | NSW1 / QLD1 / TAS1 on physical GPUs 5 / 6 / 7 |
| Output directory | `outputs/forecasting/trigonometric_gate/` |

### Artifacts

| Region | Console log | Metrics | Training history | Best checkpoint |
|---|---|---|---|---|
| NSW1 | [`nsw1.log`](trigonometric_gate/nsw1.log) | [`metrics.json`](../../outputs/forecasting/trigonometric_gate/NSW1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/trigonometric_gate/NSW1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/trigonometric_gate/NSW1/best_model.pt) |
| QLD1 | [`qld1.log`](trigonometric_gate/qld1.log) | [`metrics.json`](../../outputs/forecasting/trigonometric_gate/QLD1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/trigonometric_gate/QLD1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/trigonometric_gate/QLD1/best_model.pt) |
| TAS1 | [`tas1.log`](trigonometric_gate/tas1.log) | [`metrics.json`](../../outputs/forecasting/trigonometric_gate/TAS1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/trigonometric_gate/TAS1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/trigonometric_gate/TAS1/best_model.pt) |

### Validation metrics at the selected checkpoint

| Region | MAE | RMSE | 80% coverage | 90% coverage | Mean DGF weight | DGF weight std. |
|---|---:|---:|---:|---:|---:|---:|
| NSW1 | 64.4656 | 189.4906 | 67.20% | 79.83% | 99.85% | 1.48% |
| QLD1 | 108.6560 | 433.0052 | 66.81% | 85.77% | 69.38% | 25.10% |
| TAS1 | 73.4418 | 284.2861 | 43.43% | 66.56% | 48.90% | 30.28% |

### Test metrics and static-baseline change

| Region | MAE | MAE change | RMSE | RMSE change | 80% coverage / width | 90% coverage / width | Mean DGF weight | DGF weight std. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NSW1 | 84.4436 | +38.60% | 445.4598 | +11.20% | 59.09% / 126.7615 | 77.91% / 198.7455 | 99.03% | 5.94% |
| QLD1 | 68.8983 | +5.00% | 280.5253 | +0.20% | 67.94% / 124.8967 | 90.80% / 225.3520 | 78.88% | 19.89% |
| TAS1 | 39.0713 | +5.24% | 162.7952 | +0.69% | 50.43% / 56.9452 | 75.90% / 101.2208 | 51.60% | 22.62% |

The gated model does not improve point forecasting in any region. NSW1 collapses
almost completely onto the DGF expert, while QLD1 also strongly favors DGF. TAS1
retains a balanced and variable gate, but still loses point accuracy. The result
shows that unconstrained trigonometric gating can collapse to one expert and is
retained as a negative fusion ablation.

## Complementary additive trigonometric-fusion experiment

This experiment changes the competitive convex mixture into an additive
residual form: `forecast = CTF forecast + sin(theta)^2 * DGF correction`.
The CTF path therefore always has weight 1 and receives an unscaled gradient;
the dynamic gate controls only the non-negative strength of the DGF event
correction. All data, normalization, split, loss, optimizer, learning rate,
epoch budget, and random seed settings remain unchanged.

| Item | Value |
|---|---|
| Configuration | [`configs/aemo_forecast_additive_trigonometric_gate.yaml`](../../configs/aemo_forecast_additive_trigonometric_gate.yaml) |
| Epochs / learning rate | 30 / 0.0003 |
| Best epochs, NSW1 / QLD1 / TAS1 | 1 / 1 / 2 |
| GPU mapping | NSW1 / QLD1 / TAS1 on physical GPUs 5 / 6 / 7 |
| Output directory | `outputs/forecasting/additive_trigonometric_gate/` |

### Artifacts

| Region | Console log | Metrics | Training history | Best checkpoint |
|---|---|---|---|---|
| NSW1 | [`nsw1.log`](additive_trigonometric_gate/nsw1.log) | [`metrics.json`](../../outputs/forecasting/additive_trigonometric_gate/NSW1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/additive_trigonometric_gate/NSW1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/additive_trigonometric_gate/NSW1/best_model.pt) |
| QLD1 | [`qld1.log`](additive_trigonometric_gate/qld1.log) | [`metrics.json`](../../outputs/forecasting/additive_trigonometric_gate/QLD1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/additive_trigonometric_gate/QLD1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/additive_trigonometric_gate/QLD1/best_model.pt) |
| TAS1 | [`tas1.log`](additive_trigonometric_gate/tas1.log) | [`metrics.json`](../../outputs/forecasting/additive_trigonometric_gate/TAS1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/additive_trigonometric_gate/TAS1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/additive_trigonometric_gate/TAS1/best_model.pt) |

![Additive fusion training and validation loss](additive_trigonometric_gate/training_loss_curves.png)

![Additive fusion validation DGF gate](additive_trigonometric_gate/dgf_gate_curves.png)

### Validation metrics at the selected checkpoint

| Region | MAE | RMSE | 80% coverage | 90% coverage | Mean DGF correction weight | DGF weight std. |
|---|---:|---:|---:|---:|---:|---:|
| NSW1 | 69.0112 | 187.8241 | 60.12% | 78.60% | 47.24% | 17.51% |
| QLD1 | 95.9668 | 434.4366 | 62.20% | 85.61% | 51.87% | 20.30% |
| TAS1 | 63.7030 | 273.8954 | 46.75% | 73.65% | 66.09% | 20.33% |

### Test metrics and comparisons

| Region | MAE | vs. static | vs. competitive gate | RMSE | vs. static | vs. competitive gate | Mean DGF correction weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| NSW1 | 67.5852 | +10.93% | -19.96% | 400.3284 | -0.07% | -10.13% | 45.43% |
| QLD1 | 59.5623 | -9.23% | -13.55% | 278.2809 | -0.60% | -0.80% | 50.73% |
| TAS1 | 37.7484 | +1.68% | -3.39% | 161.7574 | +0.04% | -0.64% | 56.51% |

Negative comparison values are improvements. The additive design outperforms
the competitive trigonometric gate on both point metrics in all three regions.
Against the original concatenation baseline, it improves both point metrics on
QLD1, slightly improves NSW1 RMSE but worsens NSW1 MAE, and is effectively tied
on TAS1 RMSE while slightly worsening TAS1 MAE.

### CTF backbone and DGF correction ablation

| Region | CTF-only MAE | Fused MAE | CTF-only RMSE | Fused RMSE | Mean absolute DGF correction (AUD/MWh) |
|---|---:|---:|---:|---:|---:|
| NSW1 | 79.8787 | 67.5852 | 406.3610 | 400.3284 | 52.5739 |
| QLD1 | 73.6634 | 59.5623 | 285.0789 | 278.2809 | 47.9608 |
| TAS1 | 39.0819 | 37.7484 | 164.3688 | 161.7574 | 17.0854 |

The DGF correction improves its paired CTF backbone in every region on both
MAE and RMSE, which supports the intended complementary interpretation. The
test DGF weights remain distributed around 45--57% instead of collapsing near
100% as in competitive NSW1. However, the best validation checkpoints occur at
epochs 1, 1, and 2, so the new parameterization still overfits quickly and does
not consistently beat the original concatenation head.

## Temporal convolutional forecast-head experiment

This experiment preserves the complementary additive CTF/DGF fusion and
replaces the flattened linear heads with separate temporal convolutional
heads. Each head uses 40 channels and five causal residual blocks with kernel
size 3 and dilations 1, 2, 4, 8, and 16. Two convolutions per block give a
125-step receptive field, covering the full 72-hour history. The final temporal
state is combined independently with each future hour's calendar vector before
point and quantile decoding. The two TCN heads contain 104,892 parameters,
compared with 110,880 for the additive linear heads and 117,280 for the minimal
GELU heads.

| Item | Value |
|---|---|
| Configuration | [`configs/aemo_forecast_additive_tcn_head.yaml`](../../configs/aemo_forecast_additive_tcn_head.yaml) |
| Epochs / learning rate | 30 / 0.0003 |
| Best epochs, NSW1 / QLD1 / TAS1 | 1 / 6 / 7 |
| GPU mapping | NSW1 / QLD1 / TAS1 on physical GPUs 5 / 6 / 7 |
| Output directory | `outputs/forecasting/additive_tcn_head/` |

### Artifacts

| Region | Console log | Metrics | Training history | Best checkpoint |
|---|---|---|---|---|
| NSW1 | [`nsw1.log`](additive_tcn_head/nsw1.log) | [`metrics.json`](../../outputs/forecasting/additive_tcn_head/NSW1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/additive_tcn_head/NSW1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/additive_tcn_head/NSW1/best_model.pt) |
| QLD1 | [`qld1.log`](additive_tcn_head/qld1.log) | [`metrics.json`](../../outputs/forecasting/additive_tcn_head/QLD1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/additive_tcn_head/QLD1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/additive_tcn_head/QLD1/best_model.pt) |
| TAS1 | [`tas1.log`](additive_tcn_head/tas1.log) | [`metrics.json`](../../outputs/forecasting/additive_tcn_head/TAS1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/additive_tcn_head/TAS1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/additive_tcn_head/TAS1/best_model.pt) |

![TCN-head training and validation loss](additive_tcn_head/training_loss_curves.png)

![TCN-head validation DGF gate](additive_tcn_head/dgf_gate_curves.png)

### Test metrics and comparisons

| Region | MAE | vs. GELU head | vs. additive linear head | RMSE | vs. GELU head | vs. additive linear head | Mean DGF correction weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| NSW1 | 65.8118 | -9.01% | -2.62% | 401.5297 | +0.11% | +0.30% | 33.98% |
| QLD1 | 75.2342 | +16.01% | +26.31% | 286.0702 | +2.42% | +2.80% | 37.44% |
| TAS1 | 41.1317 | +0.46% | +8.96% | 162.6147 | -0.19% | +0.53% | 48.30% |

Negative comparison values are improvements. The TCN improves NSW1 MAE over
both matched alternatives and slightly improves TAS1 RMSE over the GELU head,
but substantially degrades QLD1 and does not improve consistently across
regions. Training loss falls much faster than validation loss, showing that
the TCN learns stronger temporal patterns but also widens the generalization
gap. Retaining only the final TCN state as a single summary of all 72 hours is
a likely information bottleneck and is recorded as part of this ablation.

## Horizon-specific TCN history-attention experiment

This experiment retains the complete 72-step output from each CTF/DGF TCN.
Each of the 24 future calendar embeddings produces a query over historical
key/value vectors, yielding separate `[24, 72]` attention maps for the two
fields. The convolutional encoder, additive trigonometric gate, data, loss,
optimizer, learning rate, epoch budget, and seed remain unchanged. The two
attention TCN heads contain 114,492 parameters, close to the 104,892-parameter
last-state TCN and 117,280-parameter GELU alternatives.

| Item | Value |
|---|---|
| Configuration | [`configs/aemo_forecast_additive_tcn_attention_head.yaml`](../../configs/aemo_forecast_additive_tcn_attention_head.yaml) |
| Epochs / learning rate | 30 / 0.0003 |
| Best epochs, NSW1 / QLD1 / TAS1 | 7 / 4 / 2 |
| GPU mapping | NSW1 / QLD1 / TAS1 on physical GPUs 5 / 6 / 7 |
| Output directory | `outputs/forecasting/additive_tcn_attention_head/` |

### Artifacts

| Region | Console log | Metrics | Training history | Attention diagnostics | Best checkpoint |
|---|---|---|---|---|---|
| NSW1 | [`nsw1.log`](additive_tcn_attention_head/nsw1.log) | [`metrics.json`](../../outputs/forecasting/additive_tcn_attention_head/NSW1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/additive_tcn_attention_head/NSW1/training_history.csv) | [`attention_diagnostics.json`](../../outputs/forecasting/additive_tcn_attention_head/NSW1/attention_diagnostics.json) | [`best_model.pt`](../../outputs/forecasting/additive_tcn_attention_head/NSW1/best_model.pt) |
| QLD1 | [`qld1.log`](additive_tcn_attention_head/qld1.log) | [`metrics.json`](../../outputs/forecasting/additive_tcn_attention_head/QLD1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/additive_tcn_attention_head/QLD1/training_history.csv) | [`attention_diagnostics.json`](../../outputs/forecasting/additive_tcn_attention_head/QLD1/attention_diagnostics.json) | [`best_model.pt`](../../outputs/forecasting/additive_tcn_attention_head/QLD1/best_model.pt) |
| TAS1 | [`tas1.log`](additive_tcn_attention_head/tas1.log) | [`metrics.json`](../../outputs/forecasting/additive_tcn_attention_head/TAS1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/additive_tcn_attention_head/TAS1/training_history.csv) | [`attention_diagnostics.json`](../../outputs/forecasting/additive_tcn_attention_head/TAS1/attention_diagnostics.json) | [`best_model.pt`](../../outputs/forecasting/additive_tcn_attention_head/TAS1/best_model.pt) |

![Attention-TCN training and validation loss](additive_tcn_attention_head/training_loss_curves.png)

![Attention-TCN validation DGF gate](additive_tcn_attention_head/dgf_gate_curves.png)

![Mean test attention maps](additive_tcn_attention_head/mean_attention_heatmaps.png)

### Test metrics and last-state TCN comparison

| Region | MAE | MAE change | RMSE | RMSE change | Mean DGF correction weight |
|---|---:|---:|---:|---:|---:|
| NSW1 | 70.9050 | +7.74% | 409.3162 | +1.94% | 40.27% |
| QLD1 | 60.2609 | -19.90% | 280.3577 | -2.00% | 47.07% |
| TAS1 | 41.9712 | +2.04% | 163.9216 | +0.80% | 59.12% |

Negative changes are improvements. Full-history attention substantially fixes
the QLD1 degradation of the last-state TCN and approaches the additive linear
head (59.5623 MAE), but it degrades NSW1 and TAS1. The effect of removing the
single-state bottleneck is therefore market-dependent.

### Attention selectivity on the test split

| Region | CTF normalized entropy | DGF normalized entropy | CTF max weight | DGF max weight | CTF adjacent-horizon TV | DGF adjacent-horizon TV |
|---|---:|---:|---:|---:|---:|---:|
| NSW1 | 0.9361 | 0.8562 | 3.78% | 6.11% | 0.0166 | 0.0480 |
| QLD1 | 0.9033 | 0.9485 | 4.16% | 2.75% | 0.0148 | 0.0106 |
| TAS1 | 0.9781 | 0.9458 | 2.72% | 4.07% | 0.0102 | 0.0112 |

Normalized entropy is one for uniform attention; a uniform maximum over 72
positions is 1.39%. The learned maps are non-uniform but remain diffuse, and
adjacent forecast hours have very similar distributions. Thus the QLD1 gain is
mainly attributable to full-history weighted pooling rather than strongly
horizon-specific historical selection.

## Maximum-spare future-context experiment

This experiment adds one point-in-time known-future AEMO factor,
`MAXSPARECAPACITY`, to the horizon-specific TCN attention model. Each hourly
forecast uses the latest PD PASA publication available at its origin, with the
remaining market-day tail supplied by the latest already-published ST PASA
run. Two half-hour intervals are aggregated by their minimum. The factor is
standardized with training data only and projected into both the CTF and DGF
future attention contexts. Historical inputs, target, split, loss weights,
optimizer, epoch budget, and random seed remain unchanged.

| Item | Value |
|---|---|
| Configuration | [`configs/aemo_forecast_max_spare.yaml`](../../configs/aemo_forecast_max_spare.yaml) |
| Future factor | AEMO `MAXSPARECAPACITY`, hourly minimum |
| Vintage rule | Latest available PD PASA with an already-published ST PASA tail |
| Epochs / learning rate | 30 / 0.0003 |
| Best epochs, NSW1 / QLD1 / TAS1 | 1 / 1 / 2 |
| GPU mapping | NSW1 / QLD1 / TAS1 on physical GPUs 5 / 6 / 7 |
| Output directory | `outputs/forecasting/max_spare_future_context/` |

### Artifacts

| Region | Console log | Metrics | Training history | Best checkpoint |
|---|---|---|---|---|
| NSW1 | [`train.log`](../../outputs/forecasting/max_spare_future_context/NSW1/train.log) | [`metrics.json`](../../outputs/forecasting/max_spare_future_context/NSW1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/max_spare_future_context/NSW1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/max_spare_future_context/NSW1/best_model.pt) |
| QLD1 | [`train.log`](../../outputs/forecasting/max_spare_future_context/QLD1/train.log) | [`metrics.json`](../../outputs/forecasting/max_spare_future_context/QLD1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/max_spare_future_context/QLD1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/max_spare_future_context/QLD1/best_model.pt) |
| TAS1 | [`train.log`](../../outputs/forecasting/max_spare_future_context/TAS1/train.log) | [`metrics.json`](../../outputs/forecasting/max_spare_future_context/TAS1/metrics.json) | [`training_history.csv`](../../outputs/forecasting/max_spare_future_context/TAS1/training_history.csv) | [`best_model.pt`](../../outputs/forecasting/max_spare_future_context/TAS1/best_model.pt) |

### Validation metrics at the selected checkpoint

| Region | MAE | RMSE | 80% coverage | 80% mean width | 90% coverage | 90% mean width |
|---|---:|---:|---:|---:|---:|---:|
| NSW1 | 88.1851 | 198.1391 | 42.20% | 109.4059 | 63.50% | 165.0661 |
| QLD1 | 175.8848 | 465.3082 | 56.13% | 180.2693 | 80.10% | 301.7551 |
| TAS1 | 68.0684 | 276.7955 | 53.47% | 114.0317 | 76.21% | 192.4702 |

### Test metrics and attention-baseline comparison

| Region | MAE | MAE change | RMSE | RMSE change | 80% coverage | 90% coverage |
|---|---:|---:|---:|---:|---:|---:|
| NSW1 | 74.9394 | +5.69% | 395.0977 | -3.47% | 60.50% | 78.23% |
| QLD1 | 75.4752 | +25.25% | 277.1987 | -1.13% | 44.79% | 80.08% |
| TAS1 | 41.8536 | -0.28% | 163.3747 | -0.33% | 54.15% | 73.84% |

Negative changes are improvements. `MAXSPARECAPACITY` reduces RMSE in every
region and reduces the MAE of the highest-price one percent by 8.6%, 5.8%, and
3.5% in NSW1, QLD1, and TAS1, respectively. It nevertheless increases ordinary
hour errors enough to worsen overall MAE in NSW1 and QLD1. NSW1 and QLD1 select
the first epoch while their training losses continue to fall, showing rapid
overfitting. The one-dimensional exogenous projection is also initialized with
substantially greater per-feature norm than a calendar column and directly
changes both experts' attention queries. The follow-up experiment therefore
keeps attention calendar-driven and introduces the factor only as a
zero-initialized post-attention residual adapter in the DGF event branch.

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
