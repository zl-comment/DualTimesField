# M1 evidence — frozen Agent contract, candidate 2 (m1.agent_contract.c2.v1)

- round: agent-m1-frozen-contract-c2-20261029 (branch
  research/agentic-m1-bounded-actions-c2-20261029-r5, base e4580163, provenance
  9feb95fd)
- scope: contract completeness and enforcement ONLY.  No labels, no scores, no
  accuracy, no D-C evaluation, no fitting/calibration, no model download, no GPU,
  no live OpenAI call, no live source fetch, no collector, no G1/G2, no push.
- gate verdict (gate.json, machine-derived): engineering GREEN (189 tests,
  suite exit 0 at the final repair HEAD) AND reducibility
  FIXED_RULES_SUFFICIENT_AGENT_NOT_NEEDED =>
  scientific_decision STOP, metrics_status NOT_EVALUATED, advance_m1 false.
  Engineering greenness never converts sufficiency into ADVANCE_M1.

## Commits
- RED ef4d68e6: tests (test_agent_m1_contract.py, test_agent_m1_reducer.py),
  scenario harness, prompts/schemas, protocol doc, config, run_logged_m1.sh.
  Logged run 01_red_m1_suite: 63 per-test failures (guarded imports, no
  collection error), EXIT_CODE=1.
- GREEN 31828a1d: m1_contract/m1_digest/m1_budget/m1_actions/m1_llm/m1_asof/
  m1_runtime/m1_publication/m1_reducer + evidence scripts; digest-renderer hash
  bound at implementation (c70fd1a36e92b2ac2a787c6e83d132a4dec314b9ed40fb96d80b895dd4233f87).
  Logged run 02_green_full_suite: 133 passed (70 M0 + 63 M1), EXIT_CODE=0.
  Three post-RED test reconciliations, no assertion weakened: (a) state-machine
  kwarg listed_refs->available_refs matching the module API; (b) the reducer
  scenario count corrected to the exact exhaustive cross (181); (c) a forged
  raw-body case accepts either fail-closed code (SCHEMA_DRIFT or
  BODY_HASH_MISMATCH) because hash binding is checked before parsing.
- EVIDENCE: this directory (capture-free round: probes are synthetic/protocol
  only).

## Commands (commands.jsonl, unique log paths, real EXIT_CODE tails)
- 00_baseline_agent_suites (provenance HEAD 9feb95fd): 70 passed, exit 0.
- 01_red_m1_suite (RED HEAD): 63 failed per-test, exit 1.
- 02_green_full_suite (GREEN HEAD): 133 passed, exit 0.
- 03_reducer_full (GREEN HEAD): 181 scenarios, 1712 legal D traces, 1629 LLM
  failure routings, internal consistency ok, witnesses 0, non-vacuity control
  detected (10 scenarios), verdict FIXED_RULES_SUFFICIENT_AGENT_NOT_NEEDED,
  exit 0.  Full report: reducibility/reducer_report.json.
- 04_gate_build: gate.json written from 02 + 03, exit 0.
- 05_evidence_closure: bidirectional closure over commands/logs/runs, exit 0.
- 06_final_state: branch/HEAD/status/writers snapshot, exit 0.

## Manifest rule (M0 audit-1 pattern)
artifact_manifest.json is regenerated DIRECTLY as the last write into this root
after every logged command; it covers every file except itself (sole
self-exclusion).  Nothing is appended to this root after that regeneration; the
closure checker runs outside the evidence root.

## Reducibility gate summary
The executable, label-free reducer crossed the declared axes (listing
EMPTY/ONE_REF/TWO_REFS x fetch outcomes COMPLETE_VALID/INCOMPLETE/SCHEMA_DRIFT/
TIMEOUT/NOT_FOUND x newest-receipt relation BEFORE/EQUAL/AFTER x two-body
relation DISTINCT/DUPLICATE/FORK x LLM result VALID/INVALID/REFUSAL/INCOMPLETE/
TIMEOUT = 181 scenarios), enumerated every legal D trace of at most four valid
proposals (1712 traces) and every LLM failure routing (1629 checks), and
compared terminals (verdict, input_hash, forecast-tool trace hash) with the
maximal B plan, the one-shot C policies and the common A fallback on the same
immutable snapshot.  Every useful D terminal is reproduced by B/C; no witness
exists.  The seeded weak-B control (spends its body GET only on the superseded
ref) is classified non-reducible in 10 scenarios, proving the detector is not
vacuous.  Therefore the frozen contract's scientific terminal state for
candidate 2 is STOP / FIXED_RULES_SUFFICIENT_AGENT_NOT_NEEDED; metrics remain
NOT_EVALUATED and no ADVANCE_M1 is emitted by this round.

## Evidence repair record
- repair 1 (fb550713): verify_m1_evidence.py repo-root path order fix; the
  05 closure run executed these exact bytes from the working tree before the
  fix was committed.  05b_evidence_closure_repair re-ran the closure check at
  the committed fix HEAD: CLOSURE OK (commands=8, runs=0), exit 0.
- the artifact manifest is regenerated directly after repair logging as the
  last write into this root; nothing is appended afterwards.

## Audit-2 repair record (second and final audit repair)
- trigger: the first read-only audit rejected the archived PASS/STOP as
  BLOCKED / REDUCIBILITY_PROOF_INCOMPLETE.  Decisive counterexample (reproduced
  before any edit): a legal same-body duplicate whose two mirrors have
  DIFFERENT per-ref transport outcomes (one TIMEOUT, one COMPLETE_VALID) made
  maximal B fall back (HTTP_TIMEOUT) while D published typed over the same
  pre-origin body — reducer over 181+12 scenarios returned
  REDUCIBILITY_REFUTED with 8 witnesses.  Root cause: the duplicate harness
  axis forced one outcome per body, B/C deduped by body hash before fetching
  (wasting the second GET), and typed derivation took the first same-body
  fetch record instead of a byte-validated mirror.
- RED commit 86d0f89 (log red_green/10_red2_m1_suite.txt, exit 1): 38
  per-test failures / 59 passed — fairness (5), fail-closed budgets (6),
  arm-A snapshot boundary (2), trust boundaries (6), publication semantic
  closure (7+4), closure verifier (5), reducer independent cross (1),
  request binding (1), budget archive tamper (1).  Failures are real
  assertion/knob failures against the then-current implementation, not a
  shared import guard.
- GREEN commit 2e3305f (log red_green/11_green2_full_suite.txt, exit 0):
  167 passed + 89 subtests.  Post-RED test reconciliations (disclosed, none
  weakens an assertion): (1) the oversized-body test helper gained the
  required snapshot_id field (malformed fixture data failed with KeyError,
  not an assertion); (2) the publication-closure helper now builds the full
  15-class budget fold from a real ledger (the RED helper omitted 14 classes;
  the valid case must publish with a complete fold); (3) call sites adapted
  to that helper; (4) the reducer expectation table's asymmetric entries had
  a wrong old_outcome field, corrected to the harness default
  COMPLETE_VALID (the axis itself is unchanged, still 2x2x3=12).
- fixes (implementation only in the GREEN commit):
  (a) B/C maximal mirror policy — fetch both visible mirrors of the newest
      body, fill the remaining GET from the listing, select a byte-validated
      mirror order-independently, anchor the receipt at the earliest VISIBLE
      receipt (never the fetched mirror's clock, which may be post-origin);
  (b) reserve-before-effect everywhere (LLM envelope reserved before the
      transport; FETCH reserves before the replay read; per-body byte cap is
      standalone, no cross-body accumulation; transport timeouts billed);
  (c) every adapter/tool/transport/publication exception maps to a frozen
      reason (new declared token EVIDENCE_PUBLICATION_FAILED); arm A never
      parses the receipt chain;
  (d) typed_input re-derives the tensor from raw bytes (forged
      M1TypedEvidence with 999999.0x24 is rejected); the __function__
      reserved key cannot override the envelope; requests bind the frozen
      schema sha256 and per-arm max_output_tokens;
  (e) 15 execution/protocol/harness/scripts files pinned in config
      (execution_modules.files_sha256) and config fields cross-checked
      module-by-module incl. M0 checkpoint/packages inheritance; identity
      recompute is no longer cached by path; from_entries verifies the
      archived hash chain verbatim;
  (f) publish_run verifies budget chain/fold/caps (per-body max), deadline,
      as-of origin and input/tool identity BEFORE any write, cleans its own
      staging+lock on crash, and every published run archives the raw LLM
      wire (base64) plus rendered digests;
  (g) verify_m1_evidence.py checks ledger/log bijection, header provenance,
      exact EXIT_CODE tails, bidirectional manifest closure, orphan
      staging/locks, gate-vs-reducer mapping, and (by default) recomputes
      the full reducer.
- new commands: 10_red2_m1_suite (exit 1), 11_green2_full_suite (exit 0,
  167 passed + 89 subtests), 12_reducer2_full (exit 0), 13_gate2_build
  (exit 1: wrong suite-log path resolution, gate.json untouched) and
  13b_gate2_build (exit 0).  The failed 13 record stays as the honest
  record of the mis-invocation.
- verdict after repair (12/13b over the extended fair cross): 193 scenarios
  (181 + 12 asymmetric mirror scenarios), 1832 legal D traces, 1737 LLM
  failure routings, internal consistency ok, witnesses 0, weak-B control
  detected in 16 scenarios => gate.json STOP /
  FIXED_RULES_SUFFICIENT_AGENT_NOT_NEEDED, metrics NOT_EVALUATED,
  advance_m1 false.  The zero-witness claim now rests on the fair scenario
  space that includes the audit's counterexample class.
- final closure protocol (audit-2): after the last logged command and this
  README, artifact_manifest.json is regenerated directly as the last write
  into this root; the closure verifier then runs DIRECTLY from outside the
  root (no ledger append, output kept outside the root at
  logs/forecasting/m1_agent_contract_c2_20261029_final_closure.txt) with
  full reducer recompute; the final clean-tree/writers/compute snapshot is
  taken at the final HEAD and recorded at
  logs/forecasting/m1_agent_contract_c2_20261029_final_state.txt.

## Repair-2 record (second read-only audit blockers, final audit-directed repair)
- trigger: the second read-only audit returned BLOCKED (H1-H5) at then-HEAD
  49a9e93.  Repaired under the frozen contract only; with the initial
  implementation, this is the second and final audit-directed repair (the
  round budget carries no third repair).
- history disclosure (audit H5): reflog shows RED commit 86d0f89 of the
  previous repair was created by commit --amend over e655077 (same message,
  same tree intent) BEFORE the archived 10_red2 run, so log HEAD lines stay
  consistent; that amend was not disclosed in the previous record.  Archived
  history is frozen (no rebase/rewrite permitted); the disclosure is made
  here instead.  The 10_red2 scaffold-dependent failures are likewise
  historical facts and are not rewritten.
- RED commit 0861e42 (log red_green/14_red3_m1_suite.txt, exit 1: 21
  per-test behavioral failures / 98 passed): execution-pin KEYSET equality
  and full config-field closure (audit H1), run-entry binding of git HEAD +
  tracked-clean tree + committed config bytes, arm A surviving corrupt
  receipt structure, unchained caller receipts rejected at the typed
  boundary, post-origin corruption unable to fail an earlier origin (H2),
  catch-all frozen mapping for transport/adapter and replay-source
  exceptions (H3), publication caps equal to the frozen CAPS with
  recomputed tool/input identity and a mandatory model ledger (H4).
  Failures are per-assertion against the then-current implementation.
- GREEN commit 611bd2c (implementation + config pins only; tests unchanged
  from RED) and GREEN-fix commit b6e3623: the verify_run_entry clean check
  is scoped to the contract surface, exempting the append-only evidence
  root because the logged-command pipeline appends to the tracked ledger
  during runs BY DESIGN (the root closes via manifest/ledger bijection
  instead); config pin for m1_contract.py regenerated.  This fix is
  implementation only; no test changed after RED.
- 15_green3_full_suite (log red_green/15_green3_full_suite.txt, exit 1: 60
  cascade failures, all run-entry tree-not-clean over commands.jsonl) is
  kept as the honest failure record; 16_green3b_full_suite at final HEAD
  b6e3623 (exit 0: 189 passed + 89 subtests, 646.85s) closes it.
- 17_reducer3_full (log reducibility/17_reducer3_full.txt, exit 0): 193
  scenarios, 1832 legal D traces, 1737 LLM failure routings, witnesses 0,
  weak-B control 16, internal consistency ok; reducer_report.json
  regenerated byte-identical (git shows no diff) => the repair did not
  change the frozen scientific surface.
- 18_gate3_build (log gate/18_gate3_build.txt, exit 0) is kept as the
  honest record of a mis-invocation: its relative output path wrote a stray
  gate.json at the worktree root (deleted; never part of evidence).  18b
  rebuilt gate.json at the evidence root from log 16 + the reducer report:
  engineering suite_exit_code 0, tests_passed 189, suite_log
  red_green/16_green3b_full_suite.txt; scientific_decision STOP /
  FIXED_RULES_SUFFICIENT_AGENT_NOT_NEEDED, metrics_status NOT_EVALUATED,
  advance_m1 false (unchanged).
- new commands: 14_red3_m1_suite (exit 1), 15_green3_full_suite (exit 1),
  16_green3b_full_suite (exit 0), 17_reducer3_full (exit 0),
  18_gate3_build (exit 0, mis-targeted output), 18b_gate3_build (exit 0).
- final closure protocol (same as the audit-2 repair): after this README,
  artifact_manifest.json is regenerated directly as the last write into
  this root; the closure verifier then runs DIRECTLY from outside the root
  (no ledger append, full reducer recompute, record at
  logs/forecasting/m1_agent_contract_c2_20261029_final_closure_r2.txt);
  the final clean-tree/writers/compute snapshot is recorded at
  logs/forecasting/m1_agent_contract_c2_20261029_final_state_r2.txt.
