# R23 2022H2 development evaluation

Implementation commit: `52d8135a7a356763dc4d2f5c82e99bfd9161fcd9`.

Experiment branch: `research/r23-dev-2022h2-20261111`.

Prediction source SHA256:
`499c6a6352731143f47a727b84a5fcc321624187a995e1f2eefe03af61026e13`.

## Scope

Frozen R23 policy, seed 2026, 181 origins per region in NSW1/QLD1/TAS1,
543 predictions total. Six balanced contiguous blocks over the already exposed
2022H2 calibration split. Identity, R21, R22 and R23 share the same origin and target
axes, identity forecasts, frozen LightGBM models and operation-admission artifact.
All committed bundle hashes and completion identities passed validation.

This is development evaluation with no within-block fitting. It is not the formal
2023-2024 test, and does not establish superiority over traditional methods. No
traditional development exports are required for this development decision.

## QS (lower is better)

| Region | Identity | R21 | R22 | R23 |
|---|---:|---:|---:|---:|
| NSW1 | 34.789365 | 34.819741 | 34.789365 | 34.789365 |
| QLD1 | 46.851195 | 46.800960 | 46.839616 | 46.819154 |
| TAS1 | 36.193094 | 36.154850 | 36.154096 | 36.142335 |
| Region-equal | 39.277885 | 39.258517 | 39.261026 | 39.250285 |

R23 improves region-equal QS by 0.027600 against identity, 0.008232 against R21
and 0.010741 against R22. These are numerical development improvements, without
a statistical-significance claim.

## Point accuracy (lower is better)

| Region | Identity MAE | R21 MAE | R22 MAE | R23 MAE |
|---|---:|---:|---:|---:|
| NSW1 | 63.994656 | 63.982784 | 63.994656 | 63.994656 |
| QLD1 | 76.240331 | 76.299990 | 76.238876 | 76.248797 |
| TAS1 | 65.220894 | 64.978429 | 65.081503 | 65.051311 |
| Region-equal | 68.485294 | 68.420401 | 68.438345 | 68.431588 |

| Region | Identity RMSE | R21 RMSE | R22 RMSE | R23 RMSE |
|---|---:|---:|---:|---:|
| NSW1 | 117.704922 | 117.718351 | 117.704922 | 117.704922 |
| QLD1 | 271.489424 | 271.528973 | 271.489450 | 271.492395 |
| TAS1 | 242.687950 | 242.560206 | 242.607014 | 242.591015 |
| Region-equal | 210.627432 | 210.602510 | 210.600462 | 210.596110 |

The region-equal RMSE is the arithmetic mean of regional RMSE values, not a single
RMSE recomputed over all regions. QS and MAE improvements need not move together:
the input operations modify the entire quantile forecast, while MAE uses its median.

## Per-region interpretation

- NSW1: R23 equals identity and R22 exactly. QS improves against R21 by 0.030376,
  but MAE is worse by 0.011871. There is no additional accuracy gain over identity.
- QLD1: QS improves over R22 by 0.020462, recovering about 53% of R22's QS deficit
  against R21. R23 still trails R21 by 0.018195 QS. MAE/RMSE improve against R21,
  but worsen slightly against identity and R22.
- TAS1: QS improves over identity, R21 and R22. MAE/RMSE also improve over identity
  and R22, but remain worse than R21 by 0.072882 MAE and 0.030809 RMSE.

All regional coverage90 changes against identity/R21 stay within the frozen 0.005
loss tolerance. This does not mean nominal 90% coverage has been achieved.

## Block stability

| Block | R23 minus identity QS | R23 minus R21 QS | R23 minus R22 QS |
|---|---:|---:|---:|
| 1 | -0.105813 | -0.030076 | -0.030859 |
| 2 | -0.056161 | -0.019615 | -0.032915 |
| 3 | 0.000000 | +0.011861 | 0.000000 |
| 4 | 0.000000 | -0.002916 | 0.000000 |
| 5 | 0.000000 | -0.010692 | 0.000000 |
| 6 | -0.001017 | +0.002777 | 0.000000 |

R23 beats R21 on region-equal QS in four of six blocks. Improvements over R22
concentrate in the first two blocks; the remaining four tie at pooled metric level.

## Decision and operational evidence

| Region | R22 nonidentity commits | R23 nonidentity commits | R23 hard vetoes | R23 Stage A identity |
|---|---:|---:|---:|---:|
| NSW1 | 0 | 0 | 7 | 11 |
| QLD1 | 3 | 15 | 11 | 28 |
| TAS1 | 10 | 11 | 16 | 23 |

R23 commits nonidentity 26 times versus R22's 13. Of 543 origins, 412 bypass
selection because no operation passes evidence admission, nine lack mature support,
62 select identity at Stage A, and 60 propose a nonidentity candidate. Of those 60,
34 are hard-vetoed and 26 reach Stage C; all 26 are approved. Thus Stage C still
does not change any decision in this run. Hard-veto primary reasons are material
downside (31) and calibration damage (3); individual cases can have multiple vetoes.

NSW1 has 18 origins reaching Stage A, but none produces a final nonidentity forecast.
QLD1 restores additional switching. The policy and skills also affect cohort choice,
so differences from R22 are not a controlled causal estimate of one admission rule.

All three runs are `COMPLETE`, with zero fallback and zero API failure. Regional
p95 origin latency is 6.66 s (NSW1), 11.57 s (QLD1), and 11.13 s (TAS1).
The analyzer and previous R22 analyzer regression suite passed eight tests.

## Development decision

The implementation checkpoint required all regions to improve QS and MAE against
identity and R21, with no regional RMSE regression and coverage loss at most 0.005.
No region satisfies that full conjunction in this run. R23 therefore does not yet
qualify for formal test evaluation despite its improved aggregate QS.

Keep the formal test closed. The next development diagnosis should examine NSW1's
18 rejected/abstained opportunities, QLD1's remaining QS deficit, and the QS versus
median-error tradeoff in QLD1/TAS1 before proposing another frozen variant.

Machine-readable results and prediction manifests:
`outputs/forecasting/r23_dev_181_seed2026_20261111/development_comparison.json`.
The JSON includes hashes of the analysis sources and manifests for all three arms.
