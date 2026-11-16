# R24 parameterization pre-screen plan (pre-registered)

Frozen 2026-11-16 before any screen result is computed. Branch
`research/r24-iterative-loop`. This document fixes the operation grid, the
data, the metrics and the decision rule. The only code constraint carried
over is the data-time constraint: the selection-split contexts already encode
each origin's information cutoff, and offline inference touches no future
data by construction. No further defensive machinery is added.

## Motivation

Steps 2-3 and the R24a evaluation closed admission and within-prediction
selection as bottlenecks. The remaining open losses are the operation space
(G = 0.214 QS region-equal, post-hoc over keep24/keep48) and evidence
freshness (R24b). This pre-screen measures whether a parameterized demand
window family grows the realizable headroom before any live engineering.

## Grid, data, computation

- Operations: demand-history mean-replacement window, keep the most recent
  {12, 24, 36, 48, 60} hours, plus replace-the-whole-window. Existing
  registered operations cover keep-24 and keep-48; four new registrations are
  added to `OPERATIONS` (keep-12, keep-36, keep-60, whole-window). Identity is
  the reference.
- Data: 2022H1 selection split, all 4321 origins per region, seed-2026 frozen
  LightGBM checkpoints, offline `predict_matrix` only. Zero API calls.
- Metrics per region and region-equal, against identity on the same origins:
  QS / MAE / RMSE deltas, origin-level QS improvement fraction, and the
  post-hoc per-origin oracle QS gain G(grid) — the same oracle definition as
  step 2, upward-biased, diagnostic only.

## Pre-registered decision rule

Parameterization proceeds to live integration (new operator-admission
artifact, then the R24a loop over the expanded menu) only if BOTH:

1. At least one window value improves QS and MAE against identity in at least
   two of the three regions simultaneously (region-aggregate deltas).
2. G(expanded grid) exceeds G(keep24, keep48) by at least 0.05 QS region-equal
   on the same origins.

Otherwise the headroom is judged oracle noise at this granularity and R24b
(cross-time candidate ledger) is next per the frozen plan. The formal test
split stays closed either way.

## Honesty clauses

The screen reuses the development-period data of a different half (H1) than
the R21 operator screen (H2); selection-split exposure still counts as design
exposure and is recorded. Oracle gains remain upward-biased upper bounds and
never enter live decision inputs.
