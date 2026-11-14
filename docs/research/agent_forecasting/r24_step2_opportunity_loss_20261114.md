# R24 step 2: R23 opportunity-loss diagnosis

Diagnostic only. Per-origin post-hoc oracle selection over saved candidates is
an upward-biased upper bound; it is not a deployable score, and these values
must never enter any decision input for the same origins.

Source run: `outputs/forecasting/r23_dev_181_seed2026_20261111`
(seed 2026, 181 origins per region, NSW1/QLD1/TAS1, 2022H2 calibration split).
Analysis: `scripts/diagnose_r23_opportunity_loss.py`, output
`outputs/forecasting/r24_step2_diagnosis_20261114/opportunity_loss.json`.
Branch rule: frozen in `r24_execution_plan_20261114.md` before this run. The
A/S sign convention in that plan was corrected before any results were
observed; QS(identity) − QS(committed) = G − A − S holds exactly.

## QS decomposition (lower is better; G/A/S are losses, larger = worse)

| Region | G generation headroom | A admission loss | S selection+risk loss | identity − committed (realized) |
|---|---:|---:|---:|---:|
| NSW1 | 0.1248 | 0.1066 | 0.0182 | 0.0000 |
| QLD1 | 0.2884 | 0.1837 | 0.0727 | 0.0320 |
| TAS1 | 0.2298 | 0.1306 | 0.0485 | 0.0508 |
| Region-equal | 0.2143 | 0.1403 | 0.0465 | 0.0276 |

MAE region-equal: G 0.2460, A 0.1156, S 0.0767 (identity − committed 0.0537).
TAS1 MAE carries the largest headroom: G 0.6099, A 0.2581, S 0.1822.

## Facts behind the numbers

- Materialized space is identity + keep-24h + keep-48h everywhere. The post-hoc
  materialized oracle picks keep-24h in 37/76/77 origins (NSW1/QLD1/TAS1),
  keep-48h in 34/30/30, identity in 110/75/74.
- Admission is where headroom dies: 160/124/128 origins per region stop at
  `identity_no_operation_passed_structured_evidence`; eligible sets are
  nonempty in only 18/54/50 origins. NSW1 never commits a non-identity input.
- Among eligible origins, committed matches the eligible-set oracle in
  173/149/162 origins; the residual selection+risk loss S is the smallest of
  the three losses in every region.
- NSW1's realized gain is exactly zero: nothing passes admission, so the loop
  would have no input there under the current contract.

## Pre-registered branch decision

`admission_contract_first` — G = 0.214 ≥ 0.05 and A = 0.140 > S = 0.046.

The R24a iteration loop targets S, the smallest loss. The dominant reachable
loss is admission filtering, so the contract is reviewed first. Direction for
that review, consistent with this diagnosis and the R23 report eligibility
counts (5→21 / 15→42 / 31→53 when R23 admission replaced R22's):

1. Re-examine the state-aware admission requirements (first/second half-day
   gains, similarity-weighted mean gain, effective support ≥ 5) and the four
   six-hour buckets' role, via pre-registered ablations on saved evidence
   reports, as done in the R23 implementation checkpoint. Do not fit
   thresholds to the oracle numbers above.
2. Audit hard-veto reasons on eligible origins (7/11/16 hard vetoes per
   region) for rules that fire on mechanically similar candidates; the
   zero-current-disagreement veto in particular cannot distinguish a harmless
   from a pointless operation.
3. Only after admission reaches the eligible origins that carry headroom does
   the R24a loop have something to select; then proceed per the frozen plan.

QS and MAE do not conflict here: both decompositions rank A above S in all
regions. The QS-vs-MAE tension visible in R23 results (NSW1 QS better than
R21 while MAE worse) lives inside the realized term, not in the loss ranking.

## Consequences for the frozen plan

No plan gate or threshold changes. Sequence adjustment per the plan's own
branch rule: step 3 executes the `admission_contract_first` branch before
R24a loop implementation. R24a and R24b remain separately evaluated with
three sampling settings and the pre-registered gate.
