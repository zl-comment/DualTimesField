# R24b 2022H2 development evaluation

Implementation commit `74e5970`. Run
`outputs/forecasting/r24b_dev_181_seed2026_20261116` (seed 2026, 181 origins
per region, 543 total, zero fallback, zero API failure). Analysis:
`scripts/analyze_r24b_development.py`, paired against identity, R21, R22,
R23 and R24a on identical origins. Gate references frozen in
`r24_execution_plan_20261114.md`.

## Result: the gate fails; R24b commits identity everywhere

Region-equal QS/MAE: R24b 39.2779/68.4853 — numerically identical to
identity. R24b committed identity in all 543 origins (R23 committed 26
non-identity forecasts, R24a 25). Versus R23 the deltas are +0.0280 QS
region-equal (NSW1 0.0000, QLD1 +0.0320, TAS1 +0.0508), concentrated in the
first two blocks.

The ledger itself worked exactly as engineered: by run end each region held
538 matured candidate rows covering identity and both operations, with
demand-regime state buckets. The counterfactual evidence pipeline is
functional; the agent's response to it was to stop taking the bets that R23's
analog-cohort evidence encouraged. The v8.1.0 instructions weight sparse
support as weak evidence and forbid ledger override of deterministic
admission; under that contract Stage A chose identity everywhere.

## Where the R24 programme now stands

Four pre-registered experiments have mapped the development ceiling of the
frozen architecture (single LightGBM, two-operation space, R23 admission and
risk contracts):

| Experiment | Outcome |
|---|---|
| Admission ablation (step 3) | Contract blocks noise; keep unchanged |
| R24a iterative loop | Gate fails; loop reproduces R23 on this space |
| Parameterization pre-screen | No aggregate co-improvement beyond TAS1 |
| R24b candidate ledger | Gate fails; agent turns fully conservative, loses R23's small gains |

R23 remains the best measured configuration (region-equal QS 39.2503, a
0.0276 improvement over identity from TAS1 and QLD1 commits). No arm has
passed the all-region gate, which requires strict QS and MAE improvement
against identity and R23 in all three regions — an outcome the observed
region-period heterogeneity (TAS1 helps, NSW1 never, QLD1 trades QS against
MAE) makes structurally very hard for any conservative policy, since not
acting leaves the delta at exactly zero rather than negative.

## Honest conclusions

1. The selection machinery — admission, iteration, fresh within-arm evidence
   — is not where the remaining accuracy lives. Each mechanism was tested in
   isolation under pre-registered rules and each failed to add value on this
   operation space.
2. The gains that do exist (about 0.03-0.05 QS in TAS1, partly QLD1) come
   from the operations themselves and are region- and state-conditional.
   Converting them into an all-region win requires a different upstream
   lever: model or feature changes, or operation design beyond demand-window
   mean replacement.
3. The formal 2023-2024 test split stays closed: no development arm passed
   its frozen gate, and opening it now would measure an unimproved policy.

## Recommendation

Close the R24 line here with this retrospective. The reproducible artefacts
(diagnosis, ablation, screen, loop, ledger, six-arm paired analyses) stay on
`research/r24-iterative-loop`. If research continues on this problem, the
evidence points upstream of agent design: candidate forecasting quality and
operation design, not evidence admission or decision policy.
