# R24 step 3 result: R23 admission ablation

Ablation plan frozen in `r24_step3_admission_ablation_plan_20261115.md`
(commit `91779de`) before computation. Source run
`outputs/forecasting/r23_dev_181_seed2026_20261111`, output
`outputs/forecasting/r24_step3_admission_ablation_20261115/ablation.json`.
The V0 reproduction guard passed: recomputed R23 admission equals the stored
eligibility for all 534 report-bearing origins.

Deltas are versus identity on identical origins; negative QS/MAE is better.
The deterministic commit policy and adoption rule are those pre-registered in
the plan. Diagnostic only; repeated-exposure development split; formal test
closed.

## Region-equal summary

| Variant | eligible origins | committed non-identity | QS delta | MAE delta | coverage90 delta |
|---|---:|---:|---:|---:|---:|
| V0 baseline (R23) | 40.7 | 40.7 | -0.0194 | -0.0666 | -0.00115 |
| V1 drop half-day | 57.7 | 57.7 | -0.0179 | -0.0651 | -0.00207 |
| V2 support 5→3 | 41.0 | 41.0 | -0.0211 | -0.0732 | -0.00115 |
| V3 drop current-change | 40.7 | 40.7 | -0.0194 | -0.0666 | -0.00115 |
| V4 state-gate only | 73.0 | 73.0 | -0.0124 | -0.0671 | -0.00261 |

Per region the picture is structural, not threshold-driven:

- NSW1: every variant's QS delta is positive (+0.030 to +0.049). The
  operations that pass admission in NSW1 hurt QS there; relaxing admission
  admits more of the same (V1 +0.0415, V4 +0.0488 are worse than V0).
- QLD1: QS improves (-0.025 to -0.044) but MAE worsens under every variant
  (+0.14 to +0.19).
- TAS1: QS and MAE both improve under every variant; relaxations add
  eligibility at declining QS value per origin.
- V2 changes almost nothing because support thresholds almost never bind
  (4-8 blocked candidates per region out of ~358). V3 changes nothing at all:
  the zero-current-change condition never binds on report-bearing origins.

The binding-reason counts agree: candidates are blocked by broadly negative
historical gains (mean, similarity-weighted, median, win fraction and half-day
conditions each fire on 150-300 of ~358 candidates per region), not by
threshold edges. The evidence itself says these operations do not help most
blocked origins. Step 2's admission "loss" A = 0.140 was therefore dominated
by post-hoc selection noise, exactly the caveat carried there.

## Veto audit (observational, small samples)

Hard vetoes reverted 7/11/16 selections (NSW1/QLD1/TAS1), dominated by
`risk_material_downside` (7/11/13) with `risk_calibration_damage` (4/6/8) and
`risk_width_inflation` (2/0/0). Committing the vetoed selections instead:

| Region | origins | QS delta | MAE delta |
|---|---:|---:|---:|
| NSW1 | 7 | -0.3960 | -0.0017 |
| QLD1 | 11 | -0.0849 | +0.3194 |
| TAS1 | 16 | -0.2063 | -0.9883 |

The vetoes blocked QS-improving selections in all three regions on these few
origins, with an MAE cost in QLD1. Samples are tiny and the same H2 data is
reused; per the plan this does not license removing vetoes. It feeds the
per-rule veto review already scheduled inside R24a, with
`risk_material_downside` as the first rule to examine.

One validation of the deployed pipeline: in NSW1 the R23 agent actually
committed identity everywhere (Stage A refusals plus seven vetoes), while the
deterministic V0 policy would have committed 18 operations at +0.0301 QS
damage. Stage A choices and vetoes protected NSW1's QS this run.

## Pre-registered decision

`keep_r23_admission_proceed_r24a` — no variant satisfies the frozen adoption
rule (QS and MAE both improve against identity in every region while
eligibility rises). The R23 admission contract stands unchanged. The blocked
candidates are predominantly noise, not wrongly filtered headroom.

Consequences:

1. R24a proceeds directly per the frozen plan (`r24_execution_plan_20261114.md`):
   the loop, its stop conditions, error semantics and `skills_r24`.
2. The per-rule hard-veto review inside R24a starts from
   `risk_material_downside` on the strength of the veto audit above.
3. The structural finding — NSW1 gains only MAE while QLD1 gains only QS from
   the current operation space — is an operation-space property. If R24a's
   loop cannot navigate it, the fallback lever is operation parameterization,
   reusing this ablation harness.
