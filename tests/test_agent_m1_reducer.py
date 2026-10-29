"""M1 label-free reducibility gate tests (candidate 2).

Audit #18: exhaust every legal D trace over the declared finite scenario axes and
compare terminals with the maximal deterministic B/C policy on the SAME immutable
snapshot; a seeded weak-B control must prove the detector is not vacuous.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "fixtures" / "agent_forecasting"))

import m1_scenario_harness as harness  # type: ignore

try:  # RED granularity: fail per test while the M1 modules are absent
    from forecasting.agent_forecasting import m1_actions
    from forecasting.agent_forecasting import m1_contract
    from forecasting.agent_forecasting import m1_reducer
    _M1_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only during the RED commit
    _M1_IMPORT_ERROR = exc


def _require_m1():
    if _M1_IMPORT_ERROR is not None:
        raise AssertionError(f"M1 reducer chain unavailable: {_M1_IMPORT_ERROR}")


class ReducerScenarioSpaceTests(unittest.TestCase):
    def setUp(self):
        _require_m1()

    def test_declared_axes_are_fully_crossed(self):
        scenarios = harness.scenario_space()
        listing_kinds = {snapshot["scenario"]["listing"] for snapshot in scenarios}
        self.assertEqual(listing_kinds, set(harness.LISTING_KINDS))
        two_ref = [snapshot for snapshot in scenarios
                   if snapshot["scenario"]["listing"] == "TWO_REFS"]
        relations = {snapshot["scenario"]["relation"] for snapshot in two_ref}
        self.assertEqual(relations, set(harness.RELATIONS))
        body_relations = {snapshot["scenario"]["body_relation"]
                          for snapshot in two_ref}
        self.assertEqual(body_relations, set(harness.BODY_RELATIONS))
        outcomes = {snapshot["scenario"]["new_outcome"] for snapshot in two_ref}
        self.assertEqual(outcomes, set(harness.FETCH_OUTCOMES))
        # duplicate same-body scenarios either repeat one outcome for both refs
        # (the symmetric axis) or break at transport level on exactly one mirror
        # (the asymmetric fairness axis)
        duplicates = [snapshot for snapshot in two_ref
                      if snapshot["scenario"]["body_relation"] == "DUPLICATE_SAME_BODY"]
        self.assertTrue(duplicates)
        for snapshot in duplicates:
            scenario = snapshot["scenario"]
            new_outcome, dup_outcome = (scenario["new_outcome"],
                                        scenario["dup_outcome"])
            if new_outcome == dup_outcome:
                self.assertEqual(scenario["old_outcome"], new_outcome)
            else:  # exactly one mirror broke at transport level
                self.assertEqual(
                    {new_outcome, dup_outcome} - {"COMPLETE_VALID"},
                    {new_outcome, dup_outcome} & set(harness.TRANSPORT_FAILURES),
                    msg=str(scenario))
                self.assertIn("COMPLETE_VALID", {new_outcome, dup_outcome})

    def test_scenario_space_equals_the_independently_derived_protocol_cross(self):
        """No hardcoded total: the expectation set is rebuilt axis by axis."""
        expected = [
            {"listing": "EMPTY", "new_outcome": "COMPLETE_VALID",
             "old_outcome": "COMPLETE_VALID", "relation": "BEFORE_ORIGIN",
             "body_relation": "DISTINCT_VINTAGES", "dup_outcome": ""},
        ]
        for outcome in harness.FETCH_OUTCOMES:          # ONE_REF: 5 x 3
            for relation in harness.RELATIONS:
                expected.append({"listing": "ONE_REF", "new_outcome": outcome,
                                 "old_outcome": "COMPLETE_VALID",
                                 "relation": relation,
                                 "body_relation": "DISTINCT_VINTAGES",
                                 "dup_outcome": ""})
        for body_relation in ("DISTINCT_VINTAGES",   # TWO_REFS: 2 x 25 x 3
                              "FORK_SAME_VINTAGE"):
            for old in harness.FETCH_OUTCOMES:
                for new in harness.FETCH_OUTCOMES:
                    for relation in harness.RELATIONS:
                        expected.append({"listing": "TWO_REFS",
                                         "new_outcome": new,
                                         "old_outcome": old,
                                         "relation": relation,
                                         "body_relation": body_relation,
                                         "dup_outcome": ""})
        for outcome in harness.FETCH_OUTCOMES:          # symmetric duplicates: 5 x 3
            for relation in harness.RELATIONS:
                expected.append({"listing": "TWO_REFS", "new_outcome": outcome,
                                 "old_outcome": outcome, "relation": relation,
                                 "body_relation": "DUPLICATE_SAME_BODY",
                                 "dup_outcome": outcome})
        for failure in harness.TRANSPORT_FAILURES:      # asymmetric mirrors: 2 x 2 x 3
            for failing in ("NEW", "DUP"):
                for relation in harness.RELATIONS:
                    new_outcome = failure if failing == "NEW" else "COMPLETE_VALID"
                    dup_outcome = "COMPLETE_VALID" if failing == "NEW" else failure
                    expected.append({"listing": "TWO_REFS",
                                     "new_outcome": new_outcome,
                                     "old_outcome": "COMPLETE_VALID",
                                     "relation": relation,
                                     "body_relation": "DUPLICATE_SAME_BODY",
                                     "dup_outcome": dup_outcome})
        self.assertEqual(len(expected),
                         1 + 5 * 3 + 2 * 25 * 3 + 5 * 3 + 2 * 2 * 3)
        actual = [json.loads(json.dumps(snapshot["scenario"], sort_keys=True))
                  for snapshot in harness.scenario_space()]
        key = lambda scenario: json.dumps(scenario, sort_keys=True)  # noqa: E731
        self.assertEqual(sorted(map(key, actual)), sorted(map(key, expected)))

    def test_d_trace_enumeration_is_closed_under_the_action_graph(self):
        snapshot = harness.build_snapshot(
            listing="TWO_REFS", new_outcome="COMPLETE_VALID",
            old_outcome="COMPLETE_VALID", relation="BEFORE_ORIGIN",
            body_relation="DISTINCT_VINTAGES")
        traces = m1_reducer.d_legal_traces(snapshot)
        self.assertTrue(traces)
        for trace in traces:
            self.assertLessEqual(len(trace), 4)
            for action, _ref in trace:
                self.assertIn(action, {"LIST", "FETCH_REF", "FINALIZE_TYPED",
                                       "FINALIZE_BASE"})
        serialized = {tuple(trace) for trace in traces}
        self.assertEqual(len(serialized), len(traces))  # no duplicates
        # every trace ends in a terminal action or exhausts the proposal cap
        for trace in traces:
            self.assertIn(trace[-1][0], ("FINALIZE_TYPED", "FINALIZE_BASE", "FETCH_REF"))


class ReducerGateTests(unittest.TestCase):
    def setUp(self):
        _require_m1()

    def test_full_reducer_proves_fixed_rule_sufficiency_or_reports_witness(self):
        report = m1_reducer.run_reducer()
        self.assertIn(report["verdict"],
                      ("FIXED_RULES_SUFFICIENT_AGENT_NOT_NEEDED",
                       "REDUCIBILITY_REFUTED", "REDUCER_NON_VACUITY_FAILED"))
        # the exhaustive cross is the independently derived expectation set
        # (see test_scenario_space_equals_the_independently_derived_protocol_cross)
        self.assertEqual(report["scenarios_total"], len(harness.scenario_space()))
        self.assertGreater(report["d_traces_enumerated"], 1500)
        self.assertGreater(report["llm_failures_checked"], 1400)
        self.assertTrue(report["internal_consistency_ok"])
        self.assertFalse(report["uses_labels_or_scores"])
        if report["verdict"] == "FIXED_RULES_SUFFICIENT_AGENT_NOT_NEEDED":
            self.assertEqual(report["witnesses"], [])

    def test_non_vacuity_control_detects_a_weak_b_policy(self):
        report = m1_reducer.run_reducer()
        control = report["non_vacuity_control"]
        self.assertTrue(control["detected"],
                        "the weak-B control MUST be classified non-reducible, "
                        "otherwise the reducer test is invalid")
        self.assertGreater(control["weak_b_diff_scenarios"], 0)

    def test_gate_maps_reducer_verdict_to_the_scientific_state(self):
        gate = m1_contract.map_gate(
            {"verdict": "FIXED_RULES_SUFFICIENT_AGENT_NOT_NEEDED",
             "engineering_green": True})
        self.assertEqual(gate["scientific_decision"], "STOP")
        self.assertEqual(gate["metrics_status"], "NOT_EVALUATED")
        self.assertIn("advance_m1", gate)
        self.assertFalse(gate["advance_m1"])

    def test_reducer_terminal_metric_covers_verdict_input_and_tool_trace(self):
        snapshot = harness.build_snapshot(
            listing="ONE_REF", new_outcome="COMPLETE_VALID",
            relation="BEFORE_ORIGIN", body_relation="DISTINCT_VINTAGES")
        terminals = m1_reducer.terminals_for_scenario(snapshot)
        self.assertIn("B", terminals)
        self.assertIn("C", terminals)
        self.assertIn("D", terminals)
        for arm in ("B", "C", "D"):
            metric = terminals[arm]
            self.assertEqual(
                set(metric),
                {"verdict", "input_hash", "tool_trace_hash", "reason_code"})
            self.assertIn(metric["verdict"],
                          ("PUBLISHED_TYPED", "PUBLISHED_BASE_FALLBACK",
                           "SYSTEM_MISSING", "TERMINAL_FAILED_NO_COMMIT"))


if __name__ == "__main__":
    unittest.main()
