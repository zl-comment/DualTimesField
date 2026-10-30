"""R9 behavioral regressions: finite reduction and actual structured requests.

Only synthetic source bytes and the deterministic forecast stub are used.  The
run-entry identity check is isolated in runtime tests because RED runs on a new
branch while the previous M1 identity still names r5.  Source validation, action
execution, budgets, typed derivation and reducer comparisons are not mocked.
"""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "fixtures" / "agent_forecasting"))
import m1_scenario_harness as harness

from forecasting.agent_forecasting import m1_contract, m1_llm, m1_reducer, m1_runtime
from forecasting.agent_forecasting.contracts import FrozenForecastInput


def three_mirror_snapshot() -> dict:
    """One visible body; two earlier mirrors fail, a third delivers its bytes."""
    snapshot = harness.build_snapshot(
        listing="TWO_REFS", new_outcome="COMPLETE_VALID",
        relation="BEFORE_ORIGIN", body_relation="DUPLICATE_SAME_BODY")
    filename = "SYNTHETIC_M1_PUBLIC_PREDISPATCHIS_202201032350_20220103235500.zip"
    ref = harness.opaque_ref(filename)
    events = [{key: value for key, value in event.items()
               if key not in ("chain_seq", "prev_sha256", "event_sha256")}
              for event in snapshot["events"]]
    events.append({**events[-1], "ref": ref, "filename": filename,
                   "receipt_complete_unix_ceil": harness.ORIGIN_UNIX - 100})
    snapshot["listing"] = (*snapshot["listing"], {"ref": ref, "filename": filename})
    snapshot["events"] = harness._chain(events)
    snapshot["fetch_behaviours"] = dict(zip(
        (entry["ref"] for entry in snapshot["listing"]),
        ("TIMEOUT", "NOT_FOUND", "COMPLETE_VALID")))
    snapshot["scenario"] = {"listing": "THREE_REFS",
                            "body_relation": "DUPLICATE_SAME_BODY",
                            "outcomes": ["TIMEOUT", "NOT_FOUND", "COMPLETE_VALID"],
                            "relation": "BEFORE_ORIGIN"}
    identity = {key: snapshot[key] for key in
                ("origin_unix", "listing", "events", "fetch_behaviours")}
    snapshot["snapshot_id"] = "snap_" + hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    return snapshot


def response(payload: bytes) -> m1_llm.M1LlmResponse:
    return m1_llm.M1LlmResponse(status="COMPLETED", model=m1_contract.LLM_MODEL,
                                output_raw=payload)


class StructuredRequestTests(unittest.TestCase):
    def test_c_request_carries_full_frozen_schema(self):
        self._check_request("C", "m1_c_decision", "m1_c_decision_schema.json")

    def test_d_request_carries_full_frozen_schema(self):
        self._check_request("D", "m1_d_action", "m1_d_action_schema.json")

    def _check_request(self, arm, name, filename):
        request = json.loads(m1_llm.build_request_bytes(arm, "{}", ROOT))
        schema = json.loads((ROOT / m1_contract.PROMPT_DIRECTORY / filename).read_text())
        output_format = request.get("text", {}).get("format", {})
        self.assertEqual(output_format.get("schema"), schema,
                         "a filename/hash cannot constrain provider output structure")
        self.assertEqual(output_format.get("type"), "json_schema")
        self.assertEqual(output_format.get("name"), name)
        self.assertIs(output_format.get("strict"), True)
        self.assertNotIn("response_schema", request,
                         "internal schema-reference metadata is not a Responses field")
        self.assertEqual(request["model"], m1_contract.LLM_MODEL)
        self.assertEqual(request["max_output_tokens"],
                         m1_contract.PER_CALL_OUTPUT_TOKENS[arm])


class ThreeMirrorReductionTests(unittest.TestCase):
    def setUp(self):
        self.entry_check = patch.object(
            m1_contract, "verify_run_entry", return_value={"contract_id": m1_contract.CONTRACT_ID})
        self.entry_check.start()
        self.addCleanup(self.entry_check.stop)

    def test_three_mirrors_admit_a_legal_useful_d_terminal(self):
        snapshot = three_mirror_snapshot()
        base = FrozenForecastInput(**harness.synthetic_base_input_payload())
        successful_ref = snapshot["listing"][2]["ref"]
        b = m1_runtime.run_arm("B", snapshot, base_input=base)
        c = m1_runtime.run_arm("C", snapshot, base_input=base,
            transport=m1_runtime.ScriptedTransport([
                response(harness.llm_valid_c_decision_bytes("USE_TYPED", "NONE"))]))
        trace = (("LIST", "NONE"), ("FETCH_REF", successful_ref),
                 ("FINALIZE_TYPED", "NONE"))
        self.assertIn(trace, m1_reducer.d_legal_traces(snapshot))
        d = m1_runtime.run_arm("D", snapshot, base_input=base,
            transport=m1_runtime.ScriptedTransport([
                response(harness.llm_valid_d_action_bytes(*action)) for action in trace]))
        self.assertEqual(b.verdict, "PUBLISHED_BASE_FALLBACK")
        self.assertEqual(c.verdict, "PUBLISHED_BASE_FALLBACK")
        self.assertEqual(d.verdict, "PUBLISHED_TYPED")
        self.assertLessEqual(d.source_replay_calls, 3)
        self.assertLessEqual(d.llm_calls, 4)
        self.assertNotEqual(d.input_hash, b.input_hash)

    def test_default_reducer_includes_the_legal_three_mirror_counterexample(self):
        report = m1_reducer.run_reducer()
        self.assertEqual(report["verdict"], "REDUCIBILITY_REFUTED",
                         "the default corpus omits a legal three-mirror witness")
        self.assertTrue(any(witness["scenario"].get("listing") == "THREE_REFS"
                            and witness["terminal"]["verdict"] == "PUBLISHED_TYPED"
                            for witness in report["witnesses"]))
        self.assertFalse(report["uses_labels_or_scores"])
        self.assertFalse(report.get("global_sufficiency_proven", True))


class ScopedGateTests(unittest.TestCase):
    @staticmethod
    def witness_report(engineering_green):
        return {"verdict": "REDUCIBILITY_REFUTED", "engineering_green": engineering_green,
                "witnesses": [{"terminal": {"verdict": "PUBLISHED_TYPED"}}],
                "global_sufficiency_proven": False}

    def test_engineering_pass_with_witness_does_not_claim_llm_advantage(self):
        gate = m1_contract.map_gate(self.witness_report(True))
        self.assertEqual(gate["scientific_decision"],
                         "ENGINEERING_PASS_REDUCIBILITY_REFUTED")
        self.assertFalse(gate["advance_m1"])
        self.assertFalse(gate["mission_goal_achieved"])
        self.assertEqual(gate["metrics_status"], "NOT_EVALUATED")
        self.assertFalse(gate.get("llm_advantage_demonstrated", True))

    def test_witness_does_not_replace_engineering_acceptance(self):
        gate = m1_contract.map_gate(self.witness_report(False))
        self.assertEqual(gate["scientific_decision"], "BLOCKED")

    def test_no_witness_on_finite_subspace_is_not_a_global_stop(self):
        gate = m1_contract.map_gate({
            "verdict": "FIXED_RULES_SUFFICIENT_AGENT_NOT_NEEDED",
            "engineering_green": True, "witnesses": [],
            "global_sufficiency_proven": False})
        self.assertEqual(gate["scientific_decision"], "INCONCLUSIVE_REDUCIBILITY")
        self.assertIsNone(gate["stop_token"])
        self.assertFalse(gate["advance_m1"])


if __name__ == "__main__":
    unittest.main()
