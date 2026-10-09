# Plan: XGBoost base with a dual-field residual

Status: run (2026-10-09); results at the end. Branch `feature/xgboost-base-fields`, from
`main` (`5ff4853`).

## Why

Under quarterly refits with predispatch, XGBoost is the best model
(35.52 raw MAE, 23.11 CRPS~), ahead of our calibrator model with the price floor
(36.64, 25.07). The fields have been tested only on a linear base, where they
improve CRPS~ and the negative-price tail but not the point forecast by much.
A reviewer will ask whether that gain survives on the strongest base. This
experiment answers it: XGBoost is the base, the dual field is trained as its
residual, and a control with the same residual heads reading the raw history
separates the effect of the fields from that of any learned residual
(stacking).

## Design

**Base.** `forecasting/xgboost_base.py` builds, for one configuration and
region, the multi-quantile XGBoost of `forecasting.baselines` (same features,
settings, and seed 2026) and stores its forecasts for every origin:

- validation and test origins: the model fitted on the training split and
  early-stopped on validation, so its test forecast is the seed-2026 XGBoost
  baseline;
- training origins: out of fold. The training origins are split into 5
  contiguous blocks; each block is forecast by an XGBoost fitted on the other
  blocks with the full model's rounds for that horizon, leaving out every
  origin within 96 hours (history plus horizon) of the block, so no fitted
  window overlaps a forecast one.

In-sample base forecasts would leave training residuals much smaller than at
test time, and the fields would learn to correct errors that do not occur.
The `.npz` stores a fingerprint of the data, split, and base-input
configuration; the dataset refuses a base built from other settings.

**Residual model.** `external_base` in the configuration replaces the linear
base of the calibrator model (`pd_calibrator`) by the fixed XGBoost forecasts:
the point is added in the point target space (standardized asinh price), the
quantiles in the quantile target space (standardized price), as for the linear
base. `model.base_inputs: true` also feeds the base's point and quantiles
(point target space) to the DGF heads and the fusion gate, so the correction
can depend on what the base forecasts (for example, widen the lower tail when
the base predicts a negative price). Everything else is the trunk's dual field
(detected-event DGF, residual path, quantile gate, predispatch, PD PASA inputs).
The calibrator and the price floor are left out, so that any change is the
fields'.

**Variants** (raw and 650-capped prices; configurations
`configs/aemo_forecast_{,capped650_}<variant>.yaml`):

| Variant | Point | Quantiles | Residual heads read |
|---|---|---|---|
| `xgb_base_tails` (main) | XGBoost | XGBoost + residual | CTF and DGF fields |
| `xgb_base_fields` | XGBoost + residual | XGBoost + residual | CTF and DGF fields |
| `xgb_base_raw_tails` (control) | XGBoost | XGBoost + residual | raw history |
| `xgb_base_raw` (control) | XGBoost + residual | XGBoost + residual | raw history |

The point forecast of the two `tails` variants is XGBoost's exactly, so their
MAE equals the base's; they test the probabilistic forecast only.

**Protocol.** Quarterly refits over 2023-2024 as in "Predispatch inputs under
quarterly recalibration" (`forecasting.rolling`, quarter configurations
`configs/aemo_forecast_rolling_{,capped650_}<variant>_q<k>.yaml`). Each refit
has its own base, fitted on its own training period
(`outputs/forecasting/xgboost_base/pd_{raw,capped650}/rolling_q<k>/`). Seeds
2026-2028 for the residual model; the base keeps seed 2026, so the seed spread
is that of the fields.

## Running

```bash
# Prerequisite: quarterly XGBoost-with-predispatch forecasts for the comparison.
CONFIG_NAME=pd_baseline OUT=outputs/forecasting/baselines_rolling_pd \
  LOG_ROOT=logs/forecasting/baselines_rolling_pd \
  scripts/run_rolling_baselines.sh "0 1 2 3" 1 xgboost

scripts/run_xgboost_base_fields.sh "0 1 2 3" 2
```

The script builds the 48 bases (8 quarters, 3 regions, 2 price treatments; about
six XGBoost fits each, roughly 30 GPU minutes), trains the 576 refits, stitches
them, and runs the Diebold-Mariano tests (`logs/forecasting/significance/
{raw,capped650}_xgb_base_{tails,fields}_<loss>.json`) and tail metrics
(`logs/forecasting/xgb_base/`). A single base can be built with

```bash
python -m forecasting.xgboost_base \
  --config configs/aemo_forecast_rolling_xgb_base_tails_q0.yaml --region NSW1
```

## Checks before reading the results

- Each base's `<region>.json` reports the out-of-fold training MAE; it should
  be close to the validation MAE, not far below it.
- The stitched test MAE of `xgb_base_tails` must equal the seed-2026 XGBoost
  baseline's (35.52 raw, 24.54 capped, up to GPU nondeterminism).

## Success criteria, fixed before the run

Pooled Diebold-Mariano over the three regions, p < 0.05, under both price
treatments unless stated.

| Question | Criterion |
|---|---|
| Do the fields improve the strongest model's probabilistic forecast? | `xgb_base_tails` CRPS~ below XGBoost |
| Is that the fields, not stacking? | `xgb_base_tails` CRPS~ below `xgb_base_raw_tails` |
| Does the negative-price advantage survive on XGBoost? | `xgb_base_tails` q05 pinball and CRPS~ on negative-price hours below XGBoost and below `xgb_base_raw_tails` |
| Do the fields improve the point forecast? | `xgb_base_fields` MAE below XGBoost and below `xgb_base_raw` |

Every outcome is reported. If the fields beat XGBoost but not the raw-history
control, the gain is stacking and the paper must say so; if they beat neither,
the paper reports that the decomposition's tail advantage holds only on a
linear base.

## After this run

The 2023-2024 test period has guided many design choices. Once the final model
is fixed, it and the baselines should be scored once on 2025, which no
decision has seen.

## Results (run 2026-10-09)

All 48 cross-fitted bases and 576 residual refits were completed. Raw prices unless noted; three seeds for the
residual model, pooled Diebold-Mariano over the three regions (first minus second, negative favours the first;
`logs/forecasting/significance/{raw,capped650}_xgb_base_*`, `logs/forecasting/xgb_base/`).

| Variant | raw MAE | raw CRPS~ | capped MAE | capped CRPS~ |
|---|---:|---:|---:|---:|
| XGBoost (archived baseline) | 35.52 | 23.11 | 24.54 | 13.10 |
| `xgb_base_tails` (XGBoost point, fields tail correction) | 35.53 | 23.87 | 24.58 | 14.11 |
| `xgb_base_raw_tails` (control) | 35.53 | 23.75 | 24.58 | 13.75 |
| `xgb_base_fields` (point and quantiles corrected) | 37.48 | 23.87 | 27.02 | 13.78 |
| `xgb_base_raw` (control) | 37.06 | 23.60 | 26.08 | 13.41 |

Check of the base: `xgb_base_tails` MAE equals XGBoost's (DM difference +0.007, p=0.49).

| Pre-registered criterion | raw | capped650 | Met? |
|---|---|---|---|
| tails CRPS~ below XGBoost | +0.60 (p=0.003) | +0.89 (p<0.001) | **No** (worse) |
| tails CRPS~ below `raw_tails` | +0.11 (p=0.012) | +0.32 (p<0.001) | **No** (worse) |
| negative-hour CRPS~ / q05 below XGBoost | -2.66 / -2.99 (p<0.001) | -2.65 / -2.67 (p<0.001) | Yes |
| negative-hour CRPS~ / q05 below `raw_tails` | +0.25 / +0.18 (p=0.001 / 0.03) | +0.06 / +0.08 (p=0.30 / 0.22) | **No** |
| fields MAE below XGBoost | +1.96 (p<0.001) | +2.48 (p<0.001) | **No** (worse) |
| fields MAE below `raw` | +0.43 (p<0.001) | +0.94 (p<0.001) | **No** (worse) |

Not pre-registered (exploratory): spike-hour (>300) CRPS~ of `xgb_base_tails` is below XGBoost by 40.5 (raw) and
20.2 (capped), and below `raw_tails` by 6.5 (raw, p=0.002) but not under capping (+0.37, p=0.24).

Reading: a learned residual on XGBoost's quantiles improves the tails (negative and spike hours) against XGBoost
itself, but costs overall CRPS~ and the point forecast; the fields do not beat the raw-history control on any
pre-registered criterion, so the tail gain is stacking, not the decomposition. By the plan's own rule, the
decomposition's tail advantage holds only on a linear base (and there only for spike hours).

Run notes: bases of quarters 5-7 (some) were built with the CPU XGBoost (`--device cpu`), the rest on the GPU;
the capped650 collection used `rolling_capped650_pd_linear_base` as the origin reference because the default
`capped650_pd_linear_base` does not exist in this checkout. The 2025 hold-out has not been used.
