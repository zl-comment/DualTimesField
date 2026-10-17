# Dual field in historical features: fixed contribution experiment

This work restores the requested dual field to historical feature learning;
it does not replace the frozen LightGBM/C_roll predictive path or change the
shared inputs, samples, point definition or scoring. Existing failed repair
attempts remain archived, not overwritten.

## Exact data flow

1. Frozen aligned LightGBM consumes the unchanged seven canonical input blocks.
2. Frozen C_roll consumes its original baseline/state/rolling-error features.
3. A historical encoder trains only on fully matured July–September 2022
   actual 24-hour price curves. Each curve is conditioned on the historical
   origin's 16 V5 state variables and its three prior 24-hour price blocks.
   Price/mask/state/context inputs total 21 channels.
4. The encoder uses the repository's Fourier continuous field and Gabor event
   field (`src/dualfield/core.py`). This is an adapted local implementation:
   the pointwise residual shortcut is excluded, live gate regularization is
   used, and atom widths/frequencies are constrained for 24-hour curves.
5. A masked-price reconstruction loss learns a background C and event E, with
   an explicitly declared weak DC+first-harmonic background prior. Only C+E
   reconstructs the historical price. Fixed smoothing is a required control.
6. After October 1, a historical record enters the reconstruction/feature
   cache only when all its 24 outcomes are available. Never encode the current
   query's unknown actual price. Retrieval uses the existing six origin-known
   state variables; every matched variant uses exactly the same candidate IDs.
7. The historical feature is a 24-hour reconstructed price curve. For selected
   neighbors, subtract their saved C_roll q50 forecasts, take median residuals
   per horizon, shrink by k/16, and clip to +/-150. Four nonnegative weights in
   [0,1] are fitted on fully matured October–November labels. Add each weighted
   shift equally to the five frozen C_roll quantiles.

The current query's past 72 hours and available external forecasts enter the
frozen predictor and matching state. The *historical* 72 hours and external
forecasts condition dual-field reconstruction. The encoder is a historical
reconstructor, not a newly claimed causal future-price predictor.

## Fixed controls and hypotheses

All three regions, seed 2026. Each historical variant has four downstream
weights, the same fitting rows, clipping, state matching and public loss.

| Variant | Historical curve / selection |
|---|---|
| B | frozen LightGBM, no new correction |
| C_roll | frozen existing calibrated forecast |
| H_recent | actual historical prices; latest16 legal records |
| H_raw | actual historical prices; state-matched16 |
| H_smooth | fixed DC+first-harmonic price curves; same matched16 |
| H_dual_trend | learned C only; same matched16 |
| H_dual_full | learned C+E; same matched16 |
| H_dual_event_off | C only, but reuse full-model weights without refitting |

H_raw and H_smooth separate added historical information from learned fields.
H_dual_trend measures whether the event field helps beyond learned background
when each representation can fit its own four weights. H_dual_event_off also
removes only E while keeping full-model weights fixed, providing a direct
event intervention without another training run.
H_recent tests the earlier evidence that recency outperforms state matching.
Every variant is reported, including zero weights and adverse results. No
candidate is selected by December performance.

Primary incremental-value requirement: H_dual_full lowers December macro MAE
and QS against C_roll, H_raw, H_smooth and H_recent; no regional MAE/QS worsening
against C_roll. Event contribution is reported separately against H_dual_trend.
Nonzero features/weights only establish activation. Historical reconstruction
quality only establishes historical reconstruction; neither alone proves
forecast contribution. Also report negative/ordinary/spike error slices and
paired 168-hour block-bootstrap intervals (24-hour sensitivity; small-sample
descriptive uncertainty, not a new blind-test/SOTA claim).

## Time boundaries and budget

- Field training: origin >= July1, origin+24h < October1. Fixed12epochs,
  batch128, width24, 6Gaboratoms, 4Fourierfrequencies, Adam .003, CPU2threads.
- Field fit never reads October or later targets or uses validation to choose
  epochs. Linear raw-price /300 scaling; inherited state standardization is
  original train-only.
- Residual memory: origins >= October1, released only at origin+24h.
- Four correction weights: origin >= October1, origin+24h < December1.
- Score: exact original December V2 sample/label grid, no tuning afterward.
- No 2023–2024 files are opened. December has already been examined; this is
  retrospective development. Three small encoder trainings replace any need
  to rerun the baseline or expensive calibration LPs.

Code-path causality is tested with future-label perturbations, strict cutoff
and exact-maturity boundaries, checkpoint round trips, common candidate IDs,
and null correction identity. External actual first-publication times remain
unaudited: the existing archive timestamps/delivery+1h/gasday-end proxies are
retained and disclosed in `full_v5_dual_history_audit_20261017.md`.
