# M1 evidence — frozen Agent contract, candidate 2 (m1.agent_contract.c2.v1)

- round: agent-m1-frozen-contract-c2-20261029 (branch
  research/agentic-m1-bounded-actions-c2-20261029-r5, base e4580163, provenance
  9feb95fd)
- scope: contract completeness and enforcement ONLY.  No labels, no scores, no
  accuracy, no D-C evaluation, no fitting/calibration, no model download, no GPU,
  no live OpenAI call, no live source fetch, no collector, no G1/G2, no push.
- gate verdict (gate.json, machine-derived): engineering GREEN (133 tests,
  suite exit 0) AND reducibility FIXED_RULES_SUFFICIENT_AGENT_NOT_NEEDED =>
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
