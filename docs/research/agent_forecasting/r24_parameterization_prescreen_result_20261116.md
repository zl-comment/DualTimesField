# R24 parameterization pre-screen result

Plan frozen in `r24_parameterization_prescreen_plan_20261116.md` (commit
`8e9eafa`) before computation. Screen:
`scripts/screen_parameterized_demand_operations.py`, output
`outputs/forecasting/r24_parameterized_prescreen_20261116/screen.json`.
2022H1 selection split, all 4321 origins per region, frozen seed-2026
LightGBM, offline inference only, zero API calls. Deltas versus identity on
identical origins; lower is better.

## Per-window region-aggregate deltas

| Window (keep hours) | NSW1 QS / MAE | QLD1 QS / MAE | TAS1 QS / MAE |
|---|---:|---:|---:|
| all replaced | +0.305 / +0.167 | +0.139 / +0.439 | **-0.113 / -0.511** |
| keep 12 | +0.158 / +0.091 | +0.163 / +0.374 | **-0.118 / -0.502** |
| keep 24 | +0.128 / +0.071 | +0.115 / +0.319 | **-0.094 / -0.414** |
| keep 36 | +0.049 / +0.030 | +0.088 / +0.259 | -0.028 / -0.179 |
| keep 48 | +0.037 / +0.009 | +0.058 / +0.196 | -0.018 / -0.080 |
| keep 60 | +0.039 / -0.009 | +0.058 / +0.136 | -0.000 / -0.030 |

Oracle QS gain (post-hoc per-origin best): region-equal 0.422 expanded versus
0.160 on the baseline keep-24/48 grid — condition 2 (growth ≥ 0.05) passes
easily. Condition 1 fails: every window co-improves QS and MAE in exactly one
region (TAS1) and in zero others.

## Pre-registered decision

`parameterization_proceeds: false`. The headroom added by the window grid at
aggregate level is confined to TAS1; NSW1 and QLD1 show no window with a QS
gain at all on H1. Notably their expanded oracle gains (0.450/0.553) rise
purely through per-origin cherry-picking among windows whose aggregate
deltas are uniformly worse — the selection-noise pattern the oracle caveat
exists for, and exactly why condition 1 required aggregate co-improvement.
The rule did its job.

## What the screen does teach

1. TAS1 consistently and strongly benefits from demand-history dampening
   across both halves (H1 here, H2 in the R23/R24a runs), with shorter
   windows better.
2. The same operations flip sign between regions and periods: QLD1 favored
   keep-24 on H2 development data yet every window hurts both metrics on H1.
   Aggregate window effects are region- and period-dependent in opposite
   directions — precisely the regime a fixed global admission cannot capture
   and per-state selection exists for.
3. Window granularity adds little beyond TAS1's existing preference; the
   binding constraint is knowing WHEN dampening helps, not how much to
   dampen.

## Next step

Per the frozen plan order: R24b, the cross-time candidate ledger. Record
every materialized candidate per origin (not only the committed forecast),
mature outcomes at +86400, expose per-(operation, state-bucket) support
counts in the evidence payload. The fresher, state-conditional evidence is
the remaining mechanism that can distinguish the TAS1-type regimes from the
QLD1-type regimes at decision time. The formal test split stays closed.
