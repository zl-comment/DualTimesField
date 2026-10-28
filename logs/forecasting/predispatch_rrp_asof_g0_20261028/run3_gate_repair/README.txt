G0-A predispatch_rrp as-of preaudit — run3 (gate-reliability repair verification)
Date: 2026-10-28 UTC. Source commit: 5626086 (freeze), branch
research/predispatch-rrp-g0a-gate-repair-20261028 (from 2b3044c on
research/predispatch-rrp-asof-g0-20261028). Run1/run2 evidence untouched.

Run3 re-executes the frozen G0-A pipeline after the second repair round,
which fixed AUDIT TOOL RELIABILITY only (external audit gate findings):
verdict semantics, thresholds, selection rule, and data are unchanged.

What changed in the tooling (commit 5626086, RED->GREEN->REFACTOR):
- Failure finalization: any exception or BudgetExceeded in the final
  source_hashes/HEAD/budget phases (or anywhere before publication) exits
  non-zero (code 3) and atomically writes an explicit ENGINEERING_FAIL
  record into <run_dir>_failures/ next to the run dir. No ENGINEERING_PASS
  verdict, success budget report, or success manifest can survive a failed
  run (they are never written into the run dir before publication).
- Atomic publication: the complete success evidence set is staged in a
  temporary sibling directory and published with a single atomic rename
  after ALL final checks pass. Partial success sets are impossible; a
  publication fault leaves the staged set for recovery and a failure record.
- Budget accounting now covers process start (best-effort /proc/self/stat,
  else first module import line, always before the numpy/harness imports),
  module imports, CLI parsing, config load, checks, final hashes/HEAD and
  evidence finalization. run_manifest.wall_clock_anchor=process_start;
  budget_report.wall_seconds_elapsed >= run_manifest.wall_seconds is
  enforced (here 3.057s vs 3.049s of the 300s budget).
- Evidence protection: a rerun refuses to overwrite a run dir that already
  holds published evidence (refusal is itself a failure record); the
  record-tests stage refuses evidence-bearing run dirs too.

Process discipline: 10 new gate-reliability tests were written FIRST and
observed failing against 2b3044c (10 failed, pytest exit 1; log preserved
at ../repair2_red_gate_tests.txt in the archive root), then the minimal
implementation made them green; 59/59 tests pass (exit 0, see
preflight_tests.txt, returncode recorded in preflight_tests.json).

Verdict (g0a_verdict.json): ENGINEERING_PASS / STRICT_ASOF_BLOCKED /
quick_veto TRIGGERED — identical semantics to run1/run2. The same 4 of 12
stride-selected NSW1 origins carry value-verified tail-ffill repeats
(23/24, 21/24, 20/24, 18/24 covered hours; tail_repeat_verified=true).
publication_basis stays archive_proxy: source_last_changed_unix is an MMSDM
archive proxy, NOT historical first-seen; strict point-in-time use remains
BLOCKED regardless of tool repair. Predispatch_rrp rows with covered<24
must not be used as fill-free as-of features.

NPZ (6e5e6d73...) and builder (8ef49cdf...) sha256 unchanged, still bound
by the frozen configs. Config used: copies of the frozen config with only
output_dir/branch/lineage changed (this dir), so run1/run2 outputs and
archives stay read-only. Protocol §7 replay commands replayed verbatim
(pytest redirect with true exit code, record-tests, audit, archive
--archive-subdir run3_gate_repair).
