# V5 history correction diagnosis and bounded repair attempts

Branch: `research/v5-history-repair-20261017`, parent `219937d`.
Existing LightGBM, C_roll, input processing and public scoring files are unchanged.
Three regions, seed 2026. No 2023–2024 final-test data was used.

## What was wrong

The original matched-history branch was available on 95.63% of query origins,
but all its location and scale coefficients were exactly zero. All 12 regional
lead-group location L1 budgets were binding. History competed with established
state corrections for a shared unit budget, while describing original
LightGBM residuals rather than remaining C_roll errors.

An independent, bounded history correction removes this competition and uses
`actual - frozen_C_roll_q50`. The base predictions and parameters remain frozen;
the old production path is unchanged. The new branch is opt-in.

## Attempt 1: activation succeeds, prediction value fails

Weights fit October–November; the following table is December only. December
was already inspected, so these are retrospective development results.

| Model | NSW1 MAE | QLD1 MAE | TAS1 MAE | Macro MAE | Macro QS |
|---|---:|---:|---:|---:|---:|
| Frozen LightGBM | 25.710190 | 26.291141 | 27.044704 | 26.348678 | 15.219256 |
| Frozen C_roll | 31.216519 | 29.512742 | 27.226839 | 29.318700 | 17.428757 |
| Recent-history correction | 25.850057 | 25.771022 | 27.284182 | 26.301754 | 15.480187 |
| State-matched correction | 34.245663 | 33.199710 | 28.009585 | 31.818319 | 19.991982 |

All three matched branches have nonzero fitted weights and change every
December prediction. Mean absolute shifts are 14.42 / 11.99 / 7.19 AUD/MWh.
Thus the module operates, but it worsens every region's MAE. Recent-history
correction is substantially better, but slightly worsens TAS1 MAE and does not
establish that state matching adds value.

## Why activation was insufficient

V1-only diagnostics found strong residual-regime drift: early October weekly
median remaining errors were roughly +54 to +77, later changing to near zero
or negative. State-matched neighbors averaged 12.82–16.18 days old versus
8.27 days for recent neighbors. Matched and recent corrections disagreed in
sign on 15.74–17.97% of V1 cells. Lower distance in the six available state
coordinates does not ensure a comparable current error regime.

## Bounded follow-up: rejected inside V1

Two fixed policies were declared before comparison: restrict candidates to
the existing 28-day window, or additionally reject sign disagreement and cap
the matched shift by recent-history magnitude. October weights were evaluated
in November. Both policies failed the predeclared QS eligibility requirement.

| Inner V1 model | November macro MAE | November macro QS |
|---|---:|---:|
| Frozen C_roll | 44.971383 | 30.934971 |
| 28-day matching | 55.127147 | 39.898890 |
| 28-day matching with consistency guard | 51.292099 | 36.590027 |

`selected_policy` is null. No second December evaluation, final-test run or
parameter search followed. This follows the stopping rule instead of choosing
a policy based on repeated December results.

## Interpretation and disposition

Validation: original history/calibrator tests plus new correction/selection
tests: **52 passed in 44.60 seconds**. Independent review found two provenance
issues, both corrected. All 34 parent source/config/document/test hashes remain
identical, and both archived executed-source snapshots match their run hashes.

Mechanism success and accuracy success are separate. This work demonstrates
how history can influence predictions without changing the calibrated anchor;
it does not demonstrate a useful state-matching correction. The existing six
coordinates and static transferred weight are insufficient under the observed
regime changes. The next scientific question is whether a causally measured
forecast-error regime can establish when matching is reliable; further work
requires a separately specified protocol, not a forced nonzero history weight.

The new implementation remains experimental and uncommitted because forecast
validation failed. Saved summaries, full region JSON and executed source
snapshots accompany this report; large prediction arrays remain under
`outputs/forecasting/v5_history_repair_20261017_*`.

## Reproduction

For attempt 1, use its archived `r1/source` implementation and run:

```bash
.venv/bin/python -m forecasting.full_v5.history_quickcheck --output outputs/forecasting/NEW_r1
```

For the inner comparison, use its archived `inner_v1/source` implementation:

```bash
.venv/bin/python -m forecasting.full_v5.history_quickcheck --select-policy --output outputs/forecasting/NEW_inner
```

The current runner additionally rejects a stale implementation, an altered
inner split, or an overridden/failed selection before allowing a follow-up.
No numerical changes were made after the failed inner comparison.
