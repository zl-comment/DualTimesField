# R23 state-aware admission

Base: `42cc556` (R22 implementation and blocked development result).

Research branch: `research/r23-state-aware-admission`.

## Objective and acceptance

The objective is improved forecasting accuracy in each of NSW1, QLD1 and TAS1.
A region-equal average alone cannot establish success. Report per-region QS, MAE,
RMSE, coverage90 and width90 on identical origins, forecast horizons and price units.
Compare identity LightGBM, R21, R22 and the frozen traditional controls: linear,
LASSO-ARX and XGBoost. Point-only controls must be evaluated on point metrics;
do not invent probabilistic scores for them.

R23 is an implementation hypothesis, not a demonstrated accuracy improvement.
Promotion requires improvement in QS and MAE against identity and R21 in all three
regions, with no regional RMSE regression or coverage90 loss exceeding 0.005.
Claims against other traditional controls require complete matched predictions and
their per-region comparisons. Missing controls mean incomplete evaluation, not a
win. Report six chronological block deltas and paired uncertainty alongside pooled
metrics; numerical improvement alone does not imply statistical significance.

## Implemented hypothesis

R22 rejected every operation with any nonpositive six-hour historical mean gain.
R23 instead requires positive first-half and second-half twelve-hour gains. The
four six-hour summaries remain visible to selection and risk review, allowing the
agent to inspect localized loss without treating every negative bucket as an
automatic prohibition.

State matching now has admission authority: similarity-weighted mean gain must be
positive, and effective support must be at least five. The existing runtime uses
weights proportional to `1 / (1 + distance)`; these weights provide a limited
state-similarity signal, not proof that an episode predicts the current outcome.
Overall mean/median gain, win fraction, mature support and nonzero current forecast
change remain required. No new numerical threshold was fitted on the H2 results.

This first increment retains the R22 hard risk limits and independent approve/revert
review, enabling a focused test of admission changes. It does not implement regional
risk thresholds or selective review invocation. A negative six-hour bucket can reach
review only when all other admission conditions pass; it cannot override a risk veto.

## Version and replay contract

- Policy: `r23_state_aware_input`.
- Release: `r23-state-aware-input-agent-v1`.
- Replay manifest: `r23-state-aware-replay-v1`.
- Admission contract: `r23-state-aware-admission-v1`.
- Skills: isolated `skills_r23`, version `7.0.0`.
- Shared evidence payload: `r22-structured-evidence-v1` (same field definitions).
- The distinct admission contract is included in the manifest, trace, selection
  request and review request; payload-schema compatibility does not imply identical
  R22/R23 admission semantics.
- Source fingerprints cover the R23 policy/admission implementation, shared report,
  runtime and risk implementation. Replay retains all materialized candidates.

R21 and R22 policies remain separately registered. No model retraining or new input
operation is introduced in this increment.

## Validation and evaluation sequence

1. Regression tests exercise recovery of a mixed-six-hour candidate, rejection of
   a nonpositive half day, similarity sign/support checks, hard-veto preservation,
   reviewer approval/reversion, timeouts, manifest contract and replay registration.
2. Run small engineering replays for all three regions on the 2022H1 selection
   split. These establish runtime behavior, not generalization or accuracy claims.
3. Freeze the code and compare R21/R22/R23 on the same H2 origins. Treat this as
   repeated-exposure development evaluation. Report admission, selection and veto
   attribution separately; changed retrieval choices can also change the evidence.
4. Run preregistered admission ablations on identical locked cohorts: R22 admission,
   half-day admission, and half-day plus similarity admission. Offline candidate
   outcomes are diagnostic and must not be used to choose each origin's action.
5. Export matched development predictions from frozen traditional controls before
   claiming three-region superiority; keep the 2023-2024 evaluation split closed.

The existing aligned baseline directory currently contains test prediction artifacts
but no selection/calibration NPZ exports for linear, LASSO-ARX, LightGBM or XGBoost.
Additionally, R21 operation admission was selected on H2 calibration data. Reusing it
on H1 does not turn H1 into an independent holdout. Model selection and previous
project exposure must remain explicit in every result description.

## Entry point

```bash
.venv/bin/python -m forecasting.adaptive_agent.replay \
  --policy r23_state_aware_input \
  --split selection --regions NSW1 QLD1 TAS1 --origins 8 --seed 2026 \
  --operator-admission outputs/forecasting/r21_input_operator_screen_20261105_calibration/calibration_seed2026.json \
  --output outputs/forecasting/r23_selection_smoke_seed2026_20261111
```

The command is plan-only unless `--execute` is supplied. A live replay binds its
source fingerprints, branch, commit, input artifacts and model identities in the
run manifest.

## Implementation validation result

- Adaptive-agent regression: 215 tests and 40 subtests passed.
- Live selection smoke: eight origins per region, 24 total, status `COMPLETE`;
  zero fallback and zero API failure.
- The smoke exercised two risk-review approvals, three hard vetoes and four
  Stage A identity decisions. Final predictions used keep-24h once, keep-48h once
  and identity 22 times. These counts are engineering validation only.
- Output: `outputs/forecasting/r23_selection_smoke_seed2026_20261111`.
- Smoke source SHA256:
  `499c6a6352731143f47a727b84a5fcc321624187a995e1f2eefe03af61026e13`.
- The smoke manifest records base commit `42cc556`, the R23 branch, tracked edits
  and hashes of the actual source/skill files, including newly added files.

Applying only the new admission function to each saved R22 evidence report, without
reading evaluation labels or changing cohorts, produces the following diagnostic:

| Region | Reports | R22 eligible origins | R23 eligible origins |
|---|---:|---:|---:|
| NSW1 | 178 | 5 | 21 |
| QLD1 | 178 | 15 | 42 |
| TAS1 | 178 | 31 | 53 |

This confirms that the rule opens additional decisions for the agent. It is not a
fresh R23 replay or an accuracy estimate; retrieval, selection and risk review can
still change the final actions. Full R23 paired accuracy evaluation and matched
traditional development baselines remain pending.
