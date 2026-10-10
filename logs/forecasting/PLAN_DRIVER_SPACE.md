# Plan: a dual field in driver space

Status: run (2026-10-10); results at the end. Branch `feature/driver-space-dual-field`, from `research/trunk`.

## Why

Earlier controls (`logs/forecasting/mae_alignment/`, `logs/forecasting/PLAN_XGBOOST_BASE_FIELDS.md`) found no
independent value of the price-space dual field for MAE or overall CRPS; its one visible edge is spike-hour
CRPS on raw prices. A price spike is a result. Its causes are slow system variables (spare capacity, net
load, supply margin) whose sudden departures push the system up the supply stack. Here the event field
detects events in those driver variables, the CTF still reconstructs the price, and a persistence summary of
the recent price is given to the event heads as context instead of being decomposed into "events".

## Design

History channels (72 h before the origin): price (asinh, standardized), demand, and three driver channels:
PD PASA `max_spare_capacity_mw`, `net_load_mw` (demand50 - UIGF) and `reserve_margin_mw` (available capacity -
demand50). For each past hour the value is the lead-0 value of that hour's own PD PASA run, published by that
hour (`history_drivers`, with the leakage check of the future exogenous inputs; hours without a run carry the
last value forward). Everything else is as in `pd_calibrator`: linear base, additive trigonometric gate,
quantile gate, calibrator, quarterly refits, predispatch and PD PASA inputs. The linear base and the calibrator
read the driver channels too, so all variants below share the same information.

| Variant | Event field | Heads read | Purpose |
|---|---|---|---|
| `ds_events` (main) | `DriverEventField`: 3 largest departures in each driver channel (9 events); price and demand carry none | fields + 4-number price state (last, 24 h max and mean, 72 h max) | the hypothesis |
| `ds_raw` | same | raw 5-channel history in place of the fields, same heads and state | same inputs, no decomposition |
| `ps_drivers` | events detected in the price (8, as the current model); driver channels are in the history | fields + price state | isolates where the events are detected |
| `ds_nostate` | as `ds_events` | fields, no price state | what the persistence summary adds |
| `base_drivers` | none (`field_forecast: false`) | - | what the driver channels add to the linear base alone |

Reference results already stored: `C-mae` (no fields, 2 channels, `nofields_floor_best_mae_model`), `A` (the
manuscript model), XGBoost.

## Protocol

Raw prices, quarterly refits over 2023-2024, seeds 2026-2028, as `forecasting.rolling`. Primary pipeline: the
validation-MAE checkpoint with the validation-chosen price floor (`forecasting.price_floor --checkpoint
best_mae_model.pt`), the pipeline of `C-mae`. The total-loss checkpoint is reported as well.

## Success criteria, fixed before the run

Pooled Diebold-Mariano over the three regions, p < 0.05, primary pipeline.

| Question | Criterion |
|---|---|
| Does decomposing the drivers help beyond the same inputs? | `ds_events` below `ds_raw` in MAE, or below it in spike-hour CRPS~ without worse MAE and overall CRPS~ |
| Is it the driver space? | `ds_events` below `ps_drivers` in MAE or in spike-hour CRPS~, neither worse |
| Do any fields help once drivers are in the base? | `ds_events` MAE below `base_drivers` |
| Does the persistence summary matter? | `ds_events` below `ds_nostate` in spike-hour CRPS~ |
| External reference (descriptive) | `ds_events` against `C-mae` and XGBoost: MAE, CRPS~, negative-hour and spike-hour CRPS~ |

If the first and third are not met, the decomposition has no independent value on the information tested, and the
dual field should not be the centre of the paper. Every outcome is reported. The 2023-2024 test period has guided
many choices; a final model should be scored once on 2025, which no decision has seen.

## Results (run 2026-10-10)

All 360 refits (5 variants x 72) completed; `forecasting.price_floor` (`--checkpoint`) and the pooled
Diebold-Mariano tests (`logs/forecasting/driver_space/significance/`) were run afterwards. Raw prices, three
seeds, floor chosen on each refit's validation quarter.

| Variant | MAE (NSW1 / QLD1 / TAS1) | mean MAE | CRPS~ |
|---|---|---:|---:|
| `ds_events` (main), MAE checkpoint | 44.18 / 38.73 / 27.62 | 36.84 | 26.20 |
| `ds_raw` | 44.14 / 38.39 / 27.64 | 36.72 | 25.96 |
| `ps_drivers` | 44.06 / 38.53 / 27.50 | 36.70 | 26.09 |
| `ds_nostate` | 44.10 / 38.62 / 27.34 | 36.68 | 26.14 |
| `base_drivers` (no fields) | 43.39 / 37.82 / 27.15 | 36.12 | 25.48 |
| reference `C-mae` (no fields, 2 channels) | 43.45 / 37.56 / 27.21 | 36.08 | 24.90 |
| reference XGBoost | 43.34 / 36.47 / 26.76 | 35.52 | 23.11 |

Pooled DM, `ds_events` minus the other (negative favours `ds_events`), MAE checkpoint; the total-loss checkpoint
gives the same picture (`significance/best_model_*`):

| Other | MAE | CRPS~ | negative-hour CRPS~ | spike-hour CRPS~ |
|---|---|---|---|---|
| `ds_raw` | +0.12 (p=0.008) | +0.18 (p=0.001) | +0.04 (0.62) | +7.4 (p=0.017) |
| `ps_drivers` | +0.15 (p<0.001) | +0.06 (p=0.014) | -0.60 (p<0.001) | +0.8 (0.51) |
| `ds_nostate` | +0.16 (p<0.001) | +0.04 (0.16) | -0.30 (p=0.009) | +0.4 (0.75) |
| `base_drivers` | +0.72 (p<0.001) | +0.05 (0.59) | +0.59 (p<0.001) | -25.4 (p<0.001) |
| `C-mae` | +0.77 (p<0.001) | +0.47 (p<0.001) | +1.44 (p<0.001) | -12.3 (p=0.004) |
| XGBoost | +1.32 (p<0.001) | +1.75 (p<0.001) | -0.79 (p=0.001) | +32.3 (p=0.001) |

| Pre-registered criterion | Met? |
|---|---|
| `ds_events` below `ds_raw` in MAE, or in spike-hour CRPS~ without worse MAE and CRPS~ | **No**: worse in MAE, CRPS~ and spike-hour CRPS~ |
| `ds_events` below `ps_drivers` in MAE or spike-hour CRPS~, neither worse | **No**: MAE worse, spike-hour no difference |
| `ds_events` MAE below `base_drivers` | **No**: worse by 0.72 |
| `ds_events` spike-hour CRPS~ below `ds_nostate` | **No** (p=0.75) |

Reading. Both decisive criteria fail: detecting events in driver channels does not beat feeding the same
channels undecomposed, nor price-space events, nor no fields at all. The persistence summary has no measurable
effect. Fields still lower spike-hour CRPS~ against the no-field base (-25), but the raw-channel control does
better still (+7.4 for `ds_events`), so that gain is a nonlinear head reading more inputs, not the decomposition.
The driver history channels alone add nothing: `base_drivers` (36.12 MAE, 25.48 CRPS~) is no better than `C-mae`
(36.08, 24.90), and worse in CRPS~. By the plan's rule, the dual field should not be the centre of the paper.

Run notes: `price_floor --checkpoint` is taken from `feature/mae-aligned-selection`; the first floor run failed
without it and was rerun. 2025 has not been used.
