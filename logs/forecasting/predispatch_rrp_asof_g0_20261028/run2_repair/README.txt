G0-A predispatch_rrp as-of preaudit — run2 (post-repair verification)
Date: 2026-10-28 UTC. Source commit: 5157786 (repair), branch research/predispatch-rrp-asof-g0-20261028.

Run2 re-executes the frozen G0-A pipeline after the minimal repair addressing
the external read-only audit (findings 2 and 4; see ../REPAIR_20261028.md):

- check_coverage now verifies uncovered non-stale tails at value level against
  the last covered hour. The four vetoed origins carry measured evidence:
    2015-01-01 05:00 AEST  23/24  tail=[23]    repeats last covered 18.960400
    2018-03-08 07:00 AEST  21/24  tail=[21..23] repeats last covered 62.459999
    2021-05-13 08:00 AEST  20/24  tail=[20..23] repeats last covered 20.984711
    2024-07-18 10:00 AEST  18/24  tail=[18..23] repeats last covered 59.265049
  Each row in selected_origins.json records tail_repeat_verified=true,
  last_covered_hour, last_covered_value and the tail hours (exact stored-value
  equality). quick_veto=TRIGGERED as in run1; veto semantics unchanged.

- Resource guards now cover the whole audit: BudgetGuard and a SIGALRM hard
  deadline start before config load and the frozen-binding hashes and remain
  active through the final source hashes; HEAD is read in-process from .git
  (run_manifest.git_head_source; budget processes=1); budget_report shows
  0.39s of the 300s protocol budget.

Verdict (g0a_verdict.json): ENGINEERING_PASS / STRICT_ASOF_BLOCKED /
quick_veto TRIGGERED — identical semantics to run1. publication_basis stays
archive_proxy: source_last_changed_unix is an MMSDM archive proxy, NOT a
historical first-seen capture; strict point-in-time use remains unproven and
unclaimable from this artifact.

NPZ (6e5e6d73...) and builder (8ef49cdf...) sha256 unchanged, still bound by
the frozen config. Preflight: 49 tests passed, exit code 0 (run1: 34).
Replayed protocol §7 commands verbatim (pytest redirect, record-tests, audit,
archive --archive-subdir run2_repair). Run1 evidence (commit d934b30) remains
untouched in this archive directory.
