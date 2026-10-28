# G0-A gate-reliability repair record — round 3 (2026-10-28)

Sole-writer repair on branch `research/predispatch-rrp-g0a-gate-repair-20261028`,
based on `500c317` (run3 evidence; verified clean at start, sole writer held
throughout). Run1/run2/run3 evidence and commits untouched; nothing pushed.

## Scope

Audit TOOL reliability only, restricted to the three gaps listed by the
external Codex audit of run3 (BLOCKED verdict). No scientific thresholds,
envelopes, selection rule, or verdict semantics changed; no-fill/coverage not
relaxed; archive_proxy NOT upgraded to first-seen; no labels/final tests
read; no network, model calls, training, GPU, or G0-B/G0-C/G1/G2 activity;
no tags, no push. Strict as-of remains BLOCKED (see run4 g0a_verdict.json).

## RED (tests first, real failures observed)

7 tests in tests/test_g0a_predispatch_rrp.py (GateRound3AnchorTests,
GateRound3FinalizationTailTests, GateRound3StagingScrubTests, plus the
round-2 rename-fault test whose retention expectation is intentionally
superseded) pinning the audited gaps:

1. anchor: real child with a 4.0s pre-import delay must land inside the
   anchored elapsed; BOOTTIME branch must stay in the monotonic domain
   (controlled fake /proc + clock); missing CLOCK_BOOTTIME must fall back
   to /proc/uptime, not to the post-stdlib import line;
2. finalization tail: with +0.25s injected into _fsync_tree, the published
   budget_report must cover the child wall time (baseline-cancelled spawn
   overhead) and a publication-complete record must exist and dominate the
   staged accounting;
3. staging scrub: BudgetExceeded and generic exceptions at the final
   budget.check() must leave NO ENGINEERING_PASS content anywhere (run dir,
   staging siblings, failures dir) and the failure record must disclose
   staging_scrubbed/evidence_published.

Observed against 500c317: `7 failed, exit 1` (repair3_red_gate_tests.txt /
.exit in this directory). Root causes confirmed: the anchor function
returned the process AGE on the BOOTTIME branch (elapsed degraded to system
uptime) and anchored after the stdlib imports on the fallback branch (this
interpreter has no CLOCK_BOOTTIME, so the fallback was the live path); the
budget report was sampled before the second tree fsync and publication; the
staged success set was retained "for recovery" on failure.

## GREEN (minimal implementation, then run4)

Commit 2f7c9e1 (freeze: runner + tests + run4 config):
- _process_start_monotonic(): /proc/self/stat starttime converted into the
  monotonic domain via process age on a since-boot clock — CLOCK_BOOTTIME
  when present, else /proc/uptime; module import line remains the
  last-resort fallback only when /proc is unavailable.
- cmd_audit(): staged accounting (run_manifest/budget_report) is resampled
  AFTER the last tree fsync and rewritten without a further fsync (a crash
  can then only expose the older sample, itself a true lower bound); a
  publication-complete record is sampled after the atomic commit rename
  with no fsynced write after it and written as a sibling
  (<run>.publication_record.json, archived as publication_record.json);
  invariants completion >= budget_report >= manifest enforced in-process.
- Failure path: any BaseException with staging present deletes the staged
  success set (shutil.rmtree) before writing the ENGINEERING_FAIL record;
  record fields staging_scrubbed and evidence_published disclose the
  outcome (evidence_published now uses an explicit published flag).
- 65/65 tests pass, pytest exit 0 (run4 preflight_tests.txt/.json).

## Run4 (fresh G0-A on the frozen commit)

Executed at HEAD 2f7c9e184120 with
configs/agent_forecasting/g0a_predispatch_rrp_20261028_run4_gate_repair.json
(copy of the run3 config; only output_dir/config_lineage changed) into
outputs/forecasting/predispatch_rrp_asof_g0_20261028/run4_gate_repair, then
archived to run4_gate_repair/ in this directory. Never touched run3.
Budget chain: publication-complete 3.101966s >= budget_report 3.100941s >=
run_manifest 3.100647s (of 300s); pre-audit phase (interpreter start
through CLI) now accounts 2.666895s that the run3 anchor could not see.
Layered verdict (unchanged semantics vs run1/run2/run3):
- engineering: ENGINEERING_PASS (structure checks + preflight 0 + budget ok)
- strict_asof: STRICT_ASOF_BLOCKED (archive_proxy != first-seen)
- quick_veto: TRIGGERED — 4/12 origins carry value-verified tail-ffill
  repeats (23/24, 21/24, 20/24, 18/24 covered hours)

## Deviation disclosure

This repair modifies harness/tests after the original 7255d5d freeze and
after the round-2 freeze 5626086, continuing the disclosures in
REPAIR_20261028.md and REPAIR_ROUND2_20261028.md; the original freeze
process remains non-compliant as recorded there. Run4 executes on its own
frozen commit (2f7c9e1) with source hashes recorded in run_manifest.json;
the source was NOT modified after that commit before the run.
