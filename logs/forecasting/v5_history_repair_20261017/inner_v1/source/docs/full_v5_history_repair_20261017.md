# V5 history-only repair: retrospective development check

## Scope and diagnosis

Parent: `219937d`, frozen r3 development run
`outputs/forecasting/v5_calibrated_history_20261015_r3`.
Keep LightGBM, C_roll parameters, existing inputs, sample grids, q50 definition,
and shared scoring unchanged. Do not rerun or relax the original V5 gates.

In the saved models, all 12 region/horizon-group location L1 budgets equal one.
Every matched-history location/scale coefficient is zero. History queries are
open for 95.63% of origins, so the branch is not simply missing. The shared
budget forces history to displace current-state corrections. Furthermore the
old history branch describes errors of the original LightGBM, whereas the
remaining task is to correct errors after C_roll. These are plausible causes,
not proof that matching improves prediction.

## Fixed repair and controls (declared before the new check)

Use the existing C_roll forecast as an immutable anchor. Build memory from
`Y - C_roll_q50`, only for origins on/after 2022-10-01 (after C_roll training).
At origin t, eligible records satisfy j < t and j + 24 hours <= t. Retain the
existing same-origin-hour, same-horizon, 365-day lookback, six state coordinates,
16 neighbors and minimum eight candidates. State coordinates are available at
the original forecast origin; no realized future state enters matching.

The candidate shift is the neighbor median remaining residual, shrunk by
min(candidate_count, 16)/16 and bounded to +/-150 AUD/MWh. A matched-neighbor
branch and an equally sized most-recent-neighbor control share this formula.
There is no new neural network, tree, or global recalibration.

Fit one nonnegative weight in [0,1] for each of the four existing lead groups,
using the original five-quantile weighted pinball objective. Fit only origins
in October-November whose complete labels mature strictly before December 1.
Zero weight is available and is selected if the training objective does not
improve. Add the weighted shift equally to all five anchor quantiles; this
preserves ordering and interval width. The history module has its own bounded
correction, without changing the anchor's coefficients or constraints.

## Fast evaluation and acceptance

Use all three regions, seed 2026. Score the original December V2 sample grid
with the unchanged public scorer, comparing frozen LightGBM, frozen C_roll,
recent-history correction and matched-history correction. Historical labels
in December may enter memory only once fully mature; December labels cannot
fit weights. Preserve all October/November origins in memory, including the
boundary origins omitted from segment scoring.

Report nonzero weights, prediction-change fractions, MAE, QS and price bands.
Distinguish three outcomes:

1. Mechanism works: actual predictions change with valid historical data.
2. Incremental value: matched correction improves macro MAE and QS over
   frozen C_roll, with neither metric worse in any region (tolerance 1e-9).
3. Matching value: matched correction also improves macro MAE and QS over
   the same-capacity recent-history control. This is additional evidence,
   not implied by improvement over C_roll alone.

Require causality, cutoff, null-branch, quantile-order and optimizer tests.
Use a single fixed candidate and report failures without retuning on December.
No 2023-2024 test files are read. December was already examined, so this is
retrospective development evidence, not a blind holdout or formal SOTA claim.
Changing the training split means these scores cannot be presented as passing
the original r3 G1/G2 protocol. Production V5 remains unchanged; the repair is
an opt-in history-only module and reproducible development runner.

## Bounded follow-up after r1 failure

The first fixed candidate activated history but failed incremental value in
December: matched MAE 31.8183 versus C_roll 29.3187 and recent 26.3018.
Keep this failed result and its source snapshot. Do not present activation as
accuracy success.

A V1-only diagnosis found October/November residual-regime reversal: weekly
median errors changed from roughly +60/+75 to negative values in NSW/QLD,
and from +65/+77 to near zero in TAS. Matched neighbors were older than recent
neighbors (mean 12.8–16.2 versus 8.3 days), and the two corrections disagreed
in sign on 15.7–18.0% of V1 cells. Low state distance does not establish that
the residual regime remains comparable.

Before a second December check, compare exactly two fixed policies using an
inner V1 split: fit weights in October (labels strictly before November 1),
score November's full-label-window origins. `fresh28` restricts candidate age
to the already-declared 28-day window. `recent_consistent` uses that same
pool, sets the matched shift to zero on sign disagreement with recent-history
shift, and otherwise caps its magnitude by the recent shift magnitude. Neither
policy changes the anchor or adds learned parameters. Both use clip(eta*median,
-150,150), so shrinkage precedes clipping.

Choose the policy with lowest November macro MAE among policies whose macro
QS does not exceed frozen C_roll; if neither qualifies, stop. Bind the inner
selection report/hash before reading December evaluation labels for the new
variant. Refit only its four weights on all V1, then evaluate December once.
The acceptance criteria above remain unchanged, and both failed and successful
attempts must be reported. This follow-up is adaptive exploratory development,
not evidence from an untouched validation set.
