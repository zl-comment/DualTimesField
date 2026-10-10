# Plan: a dual field in driver space

Status: implemented, not yet run. Branch `feature/driver-space-dual-field`, from `research/trunk`.

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
