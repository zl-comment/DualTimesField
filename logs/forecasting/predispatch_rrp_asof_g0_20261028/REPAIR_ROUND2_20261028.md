# G0-A gate-reliability repair record — round 2 (2026-10-28)

Sole-writer repair on branch `research/predispatch-rrp-g0a-gate-repair-20261028`,
based on `research/predispatch-rrp-asof-g0-20261028@2b3044c` (verified clean at
start). Run1/run2 evidence and commits untouched; nothing pushed.

## Scope

Audit TOOL reliability only. No scientific thresholds, envelopes, selection
rule, or verdict semantics changed; no-fill/coverage not relaxed;
archive_proxy NOT upgraded to first-seen; no labels/final tests read; no
network, model calls, training, GPU, or G0-B/G0-C/G1/G2 activity. Strict
as-of remains BLOCKED (see run3 g0a_verdict.json).

## RED (tests first, real failures observed)

10 new tests in tests/test_g0a_predispatch_rrp.py (classes
GateFinalizationFaultTests, GatePublicationAtomicityTests,
GateEvidenceProtectionTests) pinning:

1. final source_hashes/HEAD/budget-phase faults (RuntimeError,
   BudgetExceeded, exhausted budget at process start) leave no
   ENGINEERING_PASS/success budget/manifest and produce an atomic
   ENGINEERING_FAIL record with non-zero exit;
2. success evidence published only after ALL final checks via temp dir +
   single atomic rename (write interception at source_hashes; injected
   os.rename failure leaves no partial set);
3. budget timing covers process start, imports, CLI parse, checks, final
   hashes, evidence finalization; budget elapsed >= full run elapsed;
4. failure/rerun paths never damage or overwrite published evidence
   (record-tests refusal included);
5. real-behavior tests: synthetic frozen worlds on disk + subprocess
   fault-injection drivers preserving true exit codes.

Observed against 2b3044c: `10 failed, 49 deselected`, pytest exit 1
(repair2_red_gate_tests.txt / .exit in this archive root). Root causes
confirmed: verdict/budget written before final hashes; no staging/atomic
publication; budget anchored at cmd_audit entry (imports/CLI uncovered);
no failure record; reruns overwrote published evidence.

## GREEN (minimal implementation, then run3)

Commit 5626086 (freeze: harness + runner + tests + run3 config):
- BudgetGuard accepts a process-start monotonic anchor; the runner anchors
  the budget and SIGALRM deadline at process start (best-effort
  /proc/self/stat, else pre-import module line).
- cmd_audit stages the complete success set in a temp sibling dir, fsyncs,
  final budget check, then publishes with one os.rename (deadline disarmed
  immediately before the commit rename). Any BaseException -> timer
  disarmed, atomic ENGINEERING_FAIL record in <run_dir>_failures/ via
  tmp+os.replace, exit code 3 (4 if the record itself cannot be written).
- Publication refuses run dirs holding evidence or unknown occupants;
  record-tests refuses evidence-bearing run dirs; load_config no longer
  creates the run dir; source_hashes/verdict record the config actually
  used.
- 59/59 tests pass, pytest exit 0 (run3 preflight_tests.txt/.json).

## Run3 (fresh G0-A on the frozen commit)

Executed at HEAD 562608698f634e6dcebe9c9899558833648758bf with
configs/agent_forecasting/g0a_predispatch_rrp_20261028_run3_gate_repair.json
(copy of frozen config; only output_dir, branch, config_lineage changed) into
outputs/forecasting/predispatch_rrp_asof_g0_20261028/run3_gate_repair, then
archived to this directory (run3_gate_repair/). Manifest: git_head 5626086...,
wall_clock_anchor=process_start, evidence_publication mode
staging_dir_single_atomic_rename, budget 3.057s/300s, wall 3.049s,
preflight returncode 0.

Layered verdict (unchanged semantics vs run1/run2):
- engineering: ENGINEERING_PASS (structure checks + preflight 0 + budget ok)
- strict_asof: STRICT_ASOF_BLOCKED (archive_proxy != first-seen; backdated
  revisions cannot be excluded; no prospective live capture started)
- quick_veto: TRIGGERED — 4/12 origins carry value-verified tail-ffill
  repeats (23/24, 21/24, 20/24, 18/24 covered hours)

## Deviation disclosure

This repair again modifies harness/tests after the original 7255d5d freeze,
continuing the disclosure in REPAIR_20261028.md; the original freeze process
remains non-compliant as recorded there. Run3 executes on its own frozen
commit (5626086) with source hashes recorded in run_manifest.json; the
source was NOT modified after that commit before the run.
