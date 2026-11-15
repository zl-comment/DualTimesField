# R24 step 3 plan: pre-registered R23 admission ablation

Frozen 2026-11-15, before any ablation result is computed. Branch
`research/r24-iterative-loop`, base `08766f9`. This document pre-registers the
variant list, the deterministic evaluation rule and the adoption decision rule.
Nothing in it may change after results are observed.

## Motivation and data

Step 2 (`r24_step2_opportunity_loss_20261114.md`) located the dominant loss in
admission filtering: region-equal A = 0.140 QS versus S = 0.046. This step asks
one question with saved data only: are the candidates the contract blocks
genuine headroom or noise?

Source: `outputs/forecasting/r23_dev_181_seed2026_20261111` (seed 2026, 181
origins per region, NSW1/QLD1/TAS1, 2022H2 calibration). Per origin the trace
stores the structured evidence report; every admission input field
(`common_support`, `paired_qs_gain` mean/median/win-fraction, the four
six-hour bucket means, `similarity_weighted` availability/effective
support/mean gain, current-disagreement median, `cohort_quality`) is saved
inside `operation_effects`. Materialized candidate forecasts and aligned
actuals come from the same loaders as step 2. Zero model inference, zero API
calls, fully deterministic.

Three origins per region have no evidence report (no mature cohort or paired
support); every variant commits identity there.

## Pre-registered variants

R23 admission (V0) rejects a candidate on any of: common support < 5, mean
paired QS gain ≤ 0, median paired QS gain ≤ 0, win fraction < 0.625, first-12h
gain ≤ 0, last-12h gain ≤ 0, similarity evidence unavailable or effective
support < 5 or similarity-weighted gain ≤ 0, zero current disagreement,
cohort quality grade ≠ sufficient.

| Variant | support | mean>0 | median>0 | win frac | half-day | similarity | current change |
|---|---|---|---|---|---|---|---|
| V0 baseline (R23) | ≥5 | yes | yes | ≥.625 | yes | yes | yes |
| V1 drop half-day | ≥5 | yes | yes | ≥.625 | no | yes | yes |
| V2 support 5→3 | ≥3 | yes | yes | ≥.625 | yes | ≥3 | yes |
| V3 drop current-change | ≥5 | yes | yes | ≥.625 | yes | yes | no |
| V4 state-gate only | ≥5 | yes | no | no | no | yes | yes |

V0 is recomputed from the saved fields and must reproduce the stored
`eligible_candidate_ids` exactly for every origin with a report; any mismatch
aborts the ablation (implementation-consistency guard, not a variant).

## Pre-registered deterministic evaluation

For each variant and region, simulate a fixed policy on the same 181 origins:

- No eligible operation → commit identity.
- Exactly one → commit it.
- Two → the one with higher `similarity_weighted.mean_qs_gain`; tie-break by
  higher `paired_qs_gain.mean_qs_gain`, then lexicographically smallest id.

Committed forecasts are the saved materialized candidate quantiles. Score
QS, MAE, RMSE and coverage90 over all origins per region against the same
actuals, and report deltas versus identity. Selection uses only
decision-time evidence, so no oracle-selection bias applies; the policy
bypasses Kev Stage A/B and risk review, whose separate effect is measured by
the veto audit below. Reported alongside: eligibility counts, committed
non-identity counts, and per-reason binding counts under V0.

Veto audit (observational, not a variant): for origins stopped by
`risk_hard_veto_revert_identity`, commit the Stage B selection that the veto
reverted and score it against identity, grouped by veto reason code. This is
what-if diagnostics; it does not license removing vetoes by itself.

## Pre-registered adoption decision

A variant qualifies only if, in every one of the three regions:

1. Eligible-nonempty origins increase versus V0, and
2. Deterministic-commit QS and MAE both improve against identity (delta < 0),
   and
3. Coverage90 loss versus identity is at most 0.005.

If several variants qualify, adopt the most conservative: the one with the
fewest newly eligible origins region-equal; tie-break by order V3, V2, V1, V4.
If none qualifies, the R23 contract stands unchanged and R24a proceeds
directly per the frozen plan. Adoption means a new admission contract version
inside R24a (as R22→R23 did), never an in-place edit of the R23 policy.

## Caveats fixed in advance

Development-split repeated exposure applies; these numbers justify at most a
development decision. The deterministic proxy differs from the deployed agent
(Kev choice, reviewer behaviour); the veto audit bounds only the measured
subset. The formal 2023–2024 test stays closed. Oracle quantities from step 2
are not used anywhere in this ablation.
