# R22 implementation record

Date: 2026-11-11

Base commit: `c87d966553d0a5d7fc8899616bb7fa1bf3667293`

Branch: `research/r22-state-aligned-risk-review`

## Implemented boundary

R22 keeps the frozen regional LightGBM checkpoints, the R21 operation-admission
artifact, the identity/keep-48h/keep-24h candidate set, and the first locked mature
cohort protocol. It adds only a new decision layer:

- `r22-structured-evidence-v1` with current regime, cohort quality, four 6-hour
  QS-gain buckets, similarity weighting, calibration/width effects, downside,
  fixed-seed small-sample bootstrap, and leave-one-episode-out stability;
- a separate fail-closed risk packet with immutable support, downside, coverage,
  width, data-quality, provenance, and current-disagreement checks;
- a fresh Stage C Kev call that may only approve the selected candidate or revert
  to identity;
- an R22 replay policy and manifest contract that bind report versions, bootstrap
  settings, risk thresholds, reason codes, skills, source, models, and inputs;
- a paired fixed-policy blocked evaluator for R21/R22/identity results.

R21 modules and skills remain registered separately. The base
`IterativeToolHost` retains its previous observation schema; only
`R22IterativeToolHost` exposes the additional aggregates.

## Frozen risk limits

- common paired support at least 5;
- material paired-loss fraction at most 0.25 for QS loss greater than 0.01;
- lower-tail mean QS gain at least 0;
- mean coverage90 damage no greater than 0.005;
- mean width90 inflation no greater than 0.10 absolute target-price units;
- positive current median forecast disagreement;
- complete valid provenance, report, cohort, and current data-quality fields.

These limits are host-side constants. Kev cannot alter or override them.

## Validation before the development replay

- R22 and adjacent adaptive-agent test suite: 201 tests and 40 subtests passed.
- R22 policy tests cover approval, reviewer reversion, deterministic hard veto,
  review timeout, and evidence rejection.
- A two-origin NSW1 live smoke completed atomically with zero fallback or API
  failure.
- A 24-origin QLD1 live smoke completed atomically with zero fallback or API
  failure, exercised one Stage C approval and one deterministic calibration-damage
  veto, and produced a nonidentity forecast without changing model weights.
- The first 24-origin attempt exposed oversized Stage A/B requests. R22 was corrected
  to send only the validated structured report while retaining raw retrieval and
  diagnostics in the trace. The repeated smoke had no byte-budget errors.

Smoke scores are engineering evidence only and are not the R22 development result.

## Development evaluation

The run used 181 evenly spaced 2022H2 calibration origins per region, seed 2026, the
same R21 admitted operations, and six contiguous reporting blocks. All 543 origins
completed with zero fallback and zero API failure. R22 improved region-equal QS over
identity by 0.016859 but was 0.002509 worse than R21; QLD1 was 0.038656 worse than
R21 and caused the primary promotion gates to fail. R22 is therefore frozen but not
promoted. See `r22_blocked_cv_result_20261111.md` for the complete result.

The period was already exposed to earlier project iterations. The result is a
repeated-exposure fixed-policy blocked development evaluation, not independent
cross-validation and not a blind test. No 2023-2024 labels were opened.
