# R24a selection smoke

Implementation commit: `25ebcc5` plus replay CLI registration fixes on
`research/r24-iterative-loop`. Policy `r24_iterative_loop_input`, skills
`8.0.0`, admission and risk contracts unchanged from R23.

Command: selection split, NSW1/QLD1/TAS1, 8 origins per region, seed 2026,
R21 operation admission artifact unchanged. Output:
`outputs/forecasting/r24a_selection_smoke_seed2026_20261115`.

## Result

- 24/24 origins COMPLETE; zero fallback, zero API failure.
- Loop engaged on all 21 origins with a mature cohort: each ran one productive
  `diagnose_one` round and then chose `stop_and_evaluate` in round two. No
  cohort appends in this sample; with two candidates and an eight-episode
  cohort the agent treated one incremental look as sufficient.
- Downstream decisions match R23 semantics: 12 structured-evidence rejections,
  4 hard vetoes, 3 Stage A identity choices, 2 approved non-identity commits,
  3 no-mature-cohort early stops.
- Budgets: loop rounds stayed within the three-round cap; completion
  diagnostics charged only remaining pairs as designed.

Engineering validation only; no accuracy or generalization claim. The
development evaluation on 2022H2 follows the pre-registered protocol in
`r24_execution_plan_20261114.md`.
