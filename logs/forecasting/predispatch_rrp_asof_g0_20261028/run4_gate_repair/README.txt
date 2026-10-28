G0-A predispatch_rrp as-of preaudit — run4 (gate-repair round 3)
Date: 2026-10-28 UTC. Source commit: 2f7c9e1 (freeze), branch
research/predispatch-rrp-g0a-gate-repair-20261028 (from 500c317, which
holds run3 evidence). Run1/run2/run3 evidence untouched.

Run4 re-executes the frozen G0-A pipeline after the third repair round,
again fixing AUDIT TOOL RELIABILITY only; verdict semantics, thresholds,
selection rule, and data are unchanged.

What changed in the tooling (commit 2f7c9e1, RED->GREEN->REFACTOR; 7
tests observed failing first, log preserved at
../../repair3_red_gate_tests.txt, exit 1):
- Process-start anchor (audit gap 1): /proc/self/stat starttime is now
  converted into the monotonic domain via the process age on a since-boot
  clock — CLOCK_BOOTTIME when available, /proc/uptime otherwise (this
  interpreter has no CLOCK_BOOTTIME, which made the old code return the
  process AGE as the anchor, degrading elapsed to system uptime, while
  the old fallback anchored after the stdlib imports and missed
  interpreter startup/pre-import work). run4 pre-audit phase now
  accounts 2.667s of interpreter startup + imports that run3 could not
  see.
- Finalization-tail accounting (audit gap 2): the staged
  budget_report/run_manifest are resampled AFTER the last tree fsync and
  rewritten without a further fsync, and a publication-complete record
  (run4_gate_repair.publication_record.json, archived as
  publication_record.json) is sampled after the atomic commit rename
  with no fsynced write after it. Chain enforced and recorded:
  publication-complete 3.101966s >= budget_report 3.100941s >=
  run_manifest wall 3.100647s of the 300s budget; no fsync or
  publication tail can go unrecorded.
- Staged-evidence scrub (audit gap 3): any failure after staging exists
  (final budget.check included, BudgetExceeded or generic) now deletes
  the staged success set before writing the ENGINEERING_FAIL record
  (record fields staging_scrubbed/evidence_published disclose the
  outcome); no ENGINEERING_PASS content survives a failed audit anywhere
  on disk. The round-2 "staging retained for recovery" behavior is
  intentionally superseded.
- 65/65 tests pass (exit 0, preflight_tests.txt/.exit/.json).

Verdict (g0a_verdict.json): ENGINEERING_PASS / STRICT_ASOF_BLOCKED /
quick_veto TRIGGERED — identical semantics to run1/run2/run3. The same
4 of 12 stride-selected NSW1 origins carry value-verified tail-ffill
repeats (23/24, 21/24, 20/24, 18/24 covered hours;
tail_repeat_verified=true). publication_basis stays archive_proxy;
strict point-in-time use remains BLOCKED regardless of tool repair.
Predispatch_rrp rows with covered<24 must not be used as fill-free
as-of features.

NPZ (6e5e6d73...) and builder (8ef49cdf...) sha256 unchanged, still
bound by the frozen configs. Config used: copy of the run3 config with
only output_dir/config_lineage changed, so earlier outputs and archives
stay read-only. Protocol §7 replay commands replayed verbatim (pytest
redirect with true exit code, record-tests, audit, archive
--archive-subdir run4_gate_repair).
