# Plan: XGBoost base with a dual-field residual

Status: implemented, not yet run. Branch `feature/xgboost-base-fields`, from
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
