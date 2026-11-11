# R22 2022H2 fixed-policy blocked evaluation

Date: 2026-11-11

Implementation commit: `64f5adf958ca951790048120c32199c0e000b479`

Branch: `research/r22-state-aligned-risk-review`

## Evaluation boundary

- Split: exposed 2022H2 `calibration` period.
- Regions: NSW1, QLD1, and TAS1.
- Origins: 181 evenly spaced origins per region, 543 total.
- Reporting: six contiguous balanced chronological blocks.
- Seed: 2026.
- Models and operation admission: frozen R21 LightGBM checkpoints and frozen R21
  operation-admission artifact.
- Comparison: identical-origin identity, R21 v2, and R22 forecasts.

This is a repeated-exposure fixed-policy development evaluation. It has no within-block
fitting and is not independent cross-validation or a blind test. No 2023-2024 labels
were opened.

## Pooled result

Lower QS, MAE, RMSE, and width are better. Coverage deltas are absolute fractions.

| System | Region-equal QS | MAE | RMSE | Coverage90 | Width90 |
|---|---:|---:|---:|---:|---:|
| Identity | 39.277885 | 68.485294 | 210.627432 | 0.776703 | 264.483618 |
| R21 v2 | 39.258517 | 68.420401 | 210.602510 | 0.776243 | 263.399102 |
| R22 | 39.261026 | 68.438345 | 210.600462 | 0.776473 | 264.361640 |

R22 minus R21 is `+0.002509` QS, `+0.017944` MAE, `-0.002048` RMSE,
`+0.000230` coverage90, and `+0.962538` width90. R22 therefore does not beat R21
on the primary pooled QS metric.

R22 minus identity is `-0.016859` QS, `-0.046949` MAE, `-0.026970` RMSE,
`-0.000230` coverage90, and `-0.121978` width90. The new layer remains beneficial
relative to always using identity.

## Regional result

| Region | R22 QS | R22 minus R21 QS | R22 minus identity QS | R22 minus R21 coverage90 |
|---|---:|---:|---:|---:|
| NSW1 | 34.789365 | -0.030376 | 0.000000 | +0.000691 |
| QLD1 | 46.839616 | +0.038656 | -0.011579 | -0.001151 |
| TAS1 | 36.154096 | -0.000753 | -0.038998 | +0.001151 |

NSW1 reverted completely to identity and removed R21's loss. TAS1 remained slightly
better than R21. QLD1 retained a small benefit over identity but rejected too much of
R21's useful adaptation, causing the overall R21 comparison to fail.

## Block stability

The six region-equal R22-minus-R21 QS deltas were `+0.000783`, `+0.013300`,
`+0.011861`, `-0.002916`, `-0.010692`, and `+0.002777`. R22 was better in two of
six blocks. Its R22-minus-identity QS delta was negative in three blocks and exactly
zero in the other three, so it never lost to identity at block level.

## Agent decisions

- R21 changed input on 94 of 543 origins; R22 changed input on 13, an 86.2% reduction.
- R22 committed identity on 530 origins and a nonidentity candidate on 13 origins.
- Twenty-two candidates reached the risk boundary: 13 passed Stage C and nine were
  hard-vetoed before review.
- Hard vetoes comprised six `risk_material_downside` and three
  `risk_calibration_damage` decisions.
- Stage C approved all 13 candidates it received and issued no reviewer reversion.
- Final nonidentity selections were eight keep-24h and five keep-48h candidates.
- All 543 origins completed with zero fallback and zero API failure.

NSW1 keep-24h had zero R22 selections, so its realized effect is undefined rather
than positive. TAS1 keep-48h fell from 22 R21 selections to two R22 selections; on
those two origins R22 minus identity QS was `-0.201167`.

## Operational result

| Region | API calls | Calls/origin | p95 origin latency | Fallbacks | API failures |
|---|---:|---:|---:|---:|---:|
| NSW1 | 187 | 1.033 | 4.82 s | 0 | 0 |
| QLD1 | 205 | 1.133 | 5.65 s | 0 | 0 |
| TAS1 | 234 | 1.293 | 9.17 s | 0 | 0 |

All runs remained below the eight-call and 180-second per-origin limits. Prediction
bundles, manifests, hashes, schedules, and identity forecasts passed the analyzer's
integrity checks.

## Promotion gates

| Gate | Result | Evidence |
|---|---|---|
| Region-equal QS beats R21 | Fail | R22 minus R21 QS is +0.002509. |
| No region degrades by more than 0.01 QS | Fail | QLD1 degrades by +0.038656. |
| NSW1 keep-24h is not harmful | Operational pass, effect N/A | R22 selected it zero times. |
| TAS1 keep-48h is reduced or non-harmful | Pass | 22 to 2 selections; QS delta versus identity is -0.201167. |
| Coverage loss no worse than 0.5 percentage points | Pass | Worst regional change is -0.115 percentage points. |
| Zero protocol/fallback/atomicity failures | Pass | Zero fallback/API failure; integrity validation passed. |
| Latency and API use within budget | Pass | Worst p95 is 9.17 s; worst mean is 1.293 calls/origin. |

R22 is **not promoted over R21 v2** because the two primary accuracy gates fail.
The risk-review architecture is technically sound and safer than R21, but its current
evidence boundary is too conservative in QLD1.

## Next action

Keep this R22 artifact frozen. Audit the QLD1 proposals, hard vetoes, and the 163
`identity_no_operation_passed_structured_evidence` decisions to identify which report
field suppresses useful adaptation. Any threshold or evidence change must become a
new R23 hypothesis and be evaluated without tuning R22 on these same blocked results.
The 2023-2024 blind split remains closed.

Machine-readable result:
`outputs/forecasting/r22_blocked_cv_181_seed2026_20261111/blocked_cv_6fold.json`
