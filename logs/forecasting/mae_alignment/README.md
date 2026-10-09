# MAE-aligned checkpoint selection and a no-dual-field control (2026-10-09)

Setting: raw prices, quarterly refits, predispatch inputs, seeds 2026-2028, same 17,521 test origins
per region as the paper. All variants take the validation-chosen price floor of
`forecasting/price_floor.py` (floor re-selected for each variant on the refit's validation quarter).

* **A** final model in the manuscript: dual fields + calibrator, checkpoint with the lowest total validation loss.
* **B** A with the checkpoint chosen by validation MAE (`best_mae_model.pt`, no retraining).
* **C** no dual fields (linear base + calibrator, `field_forecast: false`), 72 new refits
  (`configs/aemo_forecast_pd_calibrator_nofields.yaml`), total-loss checkpoint.
* **C-mae** C with the validation-MAE checkpoint.

| variant | NSW1 | QLD1 | TAS1 | mean MAE (seed std) | CRPS~ | 90% AIS |
|---|---|---|---|---|---|---|
| A fields + calibrator + floor (reproduces the paper) | 44.27 | 38.57 | 27.07 | 36.64 (0.14) | 25.07 | 367.8 |
| B  A, MAE checkpoint | 43.87 | 38.14 | 27.08 | 36.36 (0.14) | 25.14 | 372.1 |
| C  no fields + calibrator + floor | 43.86 | 38.03 | 27.13 | 36.34 (0.21) | 24.87 | 372.1 |
| C-mae  C, MAE checkpoint | 43.45 | 37.56 | 27.21 | 36.08 (0.18) | 24.90 | 376.2 |
| XGBoost (paper) | 43.34 | 36.47 | 26.76 | 35.52 (0.02) | 23.11 | 347.6 |

Without the floor: A 36.95, B 36.60, C 36.56, C-mae 36.29.

Pooled Diebold-Mariano differences (first minus second; negative favours the first; Newey-West 48 lags, `significance/`):

| comparison | MAE | CRPS~ | negative-price CRPS~ | spike (>300) CRPS~ |
|---|---|---|---|---|
| B - A (MAE checkpoint) | -0.28 (p<0.001) | +0.01 (0.91) | -0.07 (0.39) | +15.5 (0.004) |
| C - A (remove the fields) | -0.29 (p<0.001) | +0.15 (0.16) | -0.33 (0.02) | +29.7 (<0.001) |
| C-mae - B (remove the fields, MAE checkpoint) | -0.29 (p<0.001) | +0.16 (0.08) | -0.45 (<0.001) | +24.8 (<0.001) |
| C-mae - XGBoost | +0.55 (p<0.001; NSW1 +0.11, p=0.51; QLD1 +1.09; TAS1 +0.46) | +1.28 | -2.23 | +44.6 |
| A - XGBoost | +1.12 | +1.11 | | |

## Reading

1. Selecting checkpoints by validation MAE lowers MAE by 0.28 and leaves CRPS unchanged, but worsens
   spike-hour CRPS. Part of the gap to XGBoost (1.12) was model selection (now 0.84 for B).
2. **Removing the dual fields improves MAE by 0.29 (p<0.001)** once the calibrator and floor are in both
   models. The pooled MAE gain over "linear base only" credited to the fields (-0.34) is therefore
   explained by the calibrator and floor. With them, the fields cost MAE.
3. The fields' remaining measurable value is in spike hours: spike-hour CRPS is 25-30 better with them
   (NSW1 +64 without), pooled p<0.001. Negative-price CRPS is better without the fields.
4. The best variant, C-mae (36.08), still trails XGBoost by 0.55; NSW1 is tied (p=0.51), QLD1 is not.

Caveats: single design pass on the 2023-2024 test period that earlier work already used for
selection; the floor was re-selected on validation for every variant, so MAE checkpoint and floor
share the validation quarter. Seed std of C is larger (0.21 vs 0.14).
