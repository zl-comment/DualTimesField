# R24a 2022H2 development evaluation

Implementation commits `25ebcc5`/`92aabd3` on `research/r24-iterative-loop`.
Run `outputs/forecasting/r24a_dev_181_seed2026_20261115` (seed 2026, 181
origins per region), paired analysis
`development_comparison.json` via `scripts/analyze_r24_development.py`
(flat-layout loader, 4 synthetic tests). Gate references frozen in
`r24_execution_plan_20261114.md`: beat identity and R23 on QS and MAE in every
region, RMSE not worse, coverage90 loss at most 0.005.

## Result: the pre-registered gate fails in all three regions

Region-equal QS/MAE: identity 39.2779/68.4853, R23 39.2503/68.4316,
R24a 39.2528/68.4444.

| Region | vs identity QS / MAE | vs R23 QS / MAE |
|---|---:|---:|
| NSW1 | +0.0000 / +0.0000 | +0.0000 / +0.0000 |
| QLD1 | -0.0320 / +0.0085 | +0.0000 / +0.0000 |
| TAS1 | -0.0433 / -0.1312 | +0.0075 / +0.0384 |

NSW1 and QLD1 are bit-identical to R23: the loop examined the same candidates
on the same cohort and reached the same report and the same commits. All of
the R24a-versus-R23 difference concentrates in TAS1 block 1 (+0.0145 QS
region-equal in that block; all other blocks exactly zero). Engineering
health is clean: 543/543 COMPLETE, zero fallback, zero API failure, loop
latency p50 ~6.7 s per origin.

## Interpretation

With two admitted operations and an eight-episode cohort, the agent used the
loop conservatively everywhere: one incremental `diagnose_one` round, then
`stop_and_evaluate`; no cohort appends at all. Given that behavior, the
completed diagnostic, the evidence report and Stages A/B/C are exactly the
R23 machinery, and identical decisions follow except where the v8.0.0 skill
wording nudged a handful of TAS1 commits.

This is consistent with the step-2 and step-3 evidence chain, now closed:
the within-prediction selection loss S was the smallest loss (0.046), the
admission contract was shown to block noise rather than headroom, and
iteration over the same two-operation space reproduces R23. The loop is not
the bottleneck; the operation space and the evidence feeding it are.

Per the frozen plan, a failed development gate means development continues
and the formal test split stays closed. Multi-seed and attribution baselines
(R24-single-round, greedy, random) are moot for this gate: a failure at seed
2026 cannot be repaired by more seeds of the same policy.

## Next levers, in the frozen plan's order

1. R24b: extend the FeedbackLedger to every diagnosed candidate with
   matured-at-+86400 outcomes and per-(operation, state-bucket) support
   counts (the plan's bias guards). This targets evidence quality, the
   surviving suspect for the blocked headroom.
2. Operation parameterization (keep{12..72}) with the step-3 ablation
   harness, targeting the generation headroom G = 0.214 that the current
   two-operation space converts poorly (NSW1 QS-vs-MAE conflict, QLD1 MAE
   cost). The loop built here is the delivery mechanism once either lever
   gives it something to select.
