# Diagnosis of the dual-field contribution check

## Outcome

The requested dual field is implemented and executed in the historical
representation path. It is the local repository's Fourier CTF + Gabor DGF,
with documented adaptations and no pointwise residual shortcut. Existing
LightGBM, C_roll and shared scoring code/artifacts remain unchanged.

The three fixed12epoch encoder fits and causal feature generation took about
70.2 seconds combined. Each encoder has7501 trainable parameters and2184
pre-October training windows. This is a bounded pilot, not an optimized
architecture or evidence against every possible dual-field model.

December2022 macroMAE: B26.348678; C_roll29.318700; recent26.301754;
rawmatched31.818319; fixedsmooth31.906064; dualtrend29.654605;
dualfull29.849720; fixed-head eventoff29.705454.

Dualfull is6.19% better than raw historical matching, but1.81% worse than the
unchanged C_roll. Both its macroMAE and QS are worse than original LightGBM.
TAS1 MAE slightly improves relative to C_roll, but its QS worsens; NSW and
QLD worsen on both metrics. The predefined incremental-value check fails.

## Reconstruction and forecast contribution are different

| Region | Fixed-smoothing historical reconstruction MAE | Dual historical reconstruction MAE | Mean absolute learned event |
|---|---:|---:|---:|
| NSW1 |29.707556|50.999680|3.86|
| QLD1 |35.084275|62.557497|5.32|
| TAS1 |48.608796|58.049482|4.06|

The learned historical reconstructor itself has not beaten fixed smoothing.
Its event component is nonzero but too weak to justify a claim of successful
historical spike reconstruction. Training loss falls, but no independent
masked reconstruction validation was used; undertraining, distribution shift
and capacity limitations cannot be distinguished from these traces alone.

Forecast shifts are substantially smaller than raw history: mean absolute
shifts decrease from14.42/11.99/7.19 to2.84/3.95/2.32 AUD/MWh. Therefore beating
raw history can reflect more conservative correction; it is not by itself
proof of a useful learned representation.

## Direct event intervention

Holding all full-model downstream weights and neighbor IDs fixed, removing E
changes macroMAE29.849720 to29.705454 and QS17.808058 to17.707603: removing the
event component helps overall. Per-region MAE contributions of adding E are:
NSW +0.2800, QLD +0.1694 (worse), TAS -0.0166 (better). Separately refitted
trend-only results are also reported and must not be conflated with this
fixed-head intervention.

## Tail slices

Conditional MAE above300 improves relative to C_roll: NSW934.0461→929.8199,
QLD143.3201→135.2563, TAS1278.2508→1278.0517. However, negative/ordinary-price
errors offset these gains in NSW/QLD. For example, QLD ordinary-price MAE
increases27.0851→28.3785; its whole-sample error contribution increases by
1.1050, larger than the tail contribution reduction of0.1118.

The December tail counts are only2/10/1 distinct target hours in NSW/QLD/TAS
(48/240/24 origin-horizon cells due to rolling overlap). These slices are
descriptive and cannot support a robust spike-prediction advantage claim.

## Integrity and limits

68 tests pass. Independent review verifies21 source snapshots,12 frozen
parent artifacts,3 checkpoints,24 rescored prediction files and18 exact
forecast replays. Each region's744576 active candidate assignments satisfy
same-hour and24h maturity rules;2161 records were causally encoded, with zero
maturity violations. Current query actual prices never enter the historical
encoder before release. Frozen C_roll forecasts match their original files
exactly. No2023–2024 test split was opened.

The data audit remains conditional on existing archived timestamps and
delivery/gas-publication proxies; actual first-publication times are not fully
verified. December was already inspected; the experiment is retrospective
development. Bootstrap intervals have only five weekly blocks and are
descriptive; no formal SOTA or blind-test claim is made.

## Disposition

No parameter search or further December-driven repair followed these results.
Keep the new dual-history entry point and checkpoints as an experimental
implementation. Do not replace the original forecast path or commit/push it
as a validated improvement. A next experiment should first demonstrate
held-out historical reconstruction quality and localized event recovery
before attributing forecast gains to the dual-field representation.
