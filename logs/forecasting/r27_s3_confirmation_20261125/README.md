# R27 S3 confirmation (2026-11-25)

Pre-registration: `docs/research/agent_forecasting/r27_s3_confirmation_preregistration_20261124.md`
(commit `0f2778e`). Result: `docs/research/agent_forecasting/r27_s3_confirmation_result_20261125.md`.

G-C1 PASS (dQS +0.358, +1.47%, DM/bootstrap p=0, jackknife 4.7%);
G-C2 FAIL (cross-family +0.041 < 0.05, CI includes zero) —
conclusion tier: seed averaging effective, cross-family combination unproven.

Files: scores.csv (per-arm per-region metrics), frozen_rules.json (bucket
boundaries, admission sets, git commits), results.json (full statistics and
verdict). Large per-arm prediction archives stay in
`outputs/forecasting/r27_s3_confirmation_20261125/` (not tracked).
Test-set ledger: spend #2 consumed; the S3 line is permanently closed on
the 2023-24 split.
