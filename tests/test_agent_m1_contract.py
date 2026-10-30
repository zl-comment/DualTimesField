"""M1 frozen Agent contract tests (candidate 2): RED-first enforcement suite.

These tests are committed BEFORE the M1 implementation.  At the frozen base they fail
per-test (guarded imports, never a collection error) because the contract/executor
modules do not exist.  Every source fact comes from synthetic PREDISPATCHIS-format ZIP
bytes built by the fixture harness; no labels, no scores, no network, no model download,
no GPU, no live LLM call.
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "fixtures" / "agent_forecasting"))

from forecasting.agent_forecasting.contracts import FrozenForecastInput
import m1_scenario_harness as harness  # type: ignore

try:  # RED granularity: fail per test while the M1 modules are absent
    from forecasting.agent_forecasting import m1_actions
    from forecasting.agent_forecasting import m1_asof
    from forecasting.agent_forecasting import m1_budget
    from forecasting.agent_forecasting import m1_contract
    from forecasting.agent_forecasting import m1_digest
    from forecasting.agent_forecasting import m1_llm
    from forecasting.agent_forecasting import m1_publication
    from forecasting.agent_forecasting import m1_runtime
    _M1_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only during the RED commit
    _M1_IMPORT_ERROR = exc


def _require_m1():
    if _M1_IMPORT_ERROR is not None:
        raise AssertionError(f"M1 contract chain unavailable: {_M1_IMPORT_ERROR}")


def _base_input(region: str = "NSW1") -> FrozenForecastInput:
    return FrozenForecastInput(**harness.synthetic_base_input_payload(region=region))


def _clean_snapshot() -> dict:
    return harness.build_snapshot(listing="ONE_REF", new_outcome="COMPLETE_VALID",
                                  relation="BEFORE_ORIGIN",
                                  body_relation="DISTINCT_VINTAGES")


def _two_ref_snapshot() -> dict:
    return harness.build_snapshot(listing="TWO_REFS", new_outcome="COMPLETE_VALID",
                                  old_outcome="COMPLETE_VALID",
                                  relation="BEFORE_ORIGIN",
                                  body_relation="DISTINCT_VINTAGES")


def _asym_snapshot(failing: str, failure: str,
                   relation: str = "BEFORE_ORIGIN") -> dict:
    """Duplicate same-body snapshot whose `failing` mirror breaks at transport."""
    return harness.build_snapshot(
        listing="TWO_REFS", body_relation="DUPLICATE_SAME_BODY",
        new_outcome=(failure if failing == "NEW" else "COMPLETE_VALID"),
        dup_outcome=("COMPLETE_VALID" if failing == "NEW" else failure),
        relation=relation)


def _ok_response(payload: bytes) -> "m1_llm.M1LlmResponse":
    return m1_llm.M1LlmResponse(status="COMPLETED", model=m1_contract.LLM_MODEL,
                                output_raw=payload)


def _d_transport(*pairs) -> "m1_runtime.ScriptedTransport":
    _require_m1()
    return m1_runtime.ScriptedTransport([
        _ok_response(harness.llm_valid_d_action_bytes(action, ref))
        for action, ref in pairs])


def _c_transport(decision: str) -> "m1_runtime.ScriptedTransport":
    _require_m1()
    return m1_runtime.ScriptedTransport([
        _ok_response(harness.llm_valid_c_decision_bytes(decision, "NONE"))])


def _run(arm, snapshot, *, transport=None, tool=None, clock=None,
         publish_root=None, region="NSW1", budget=None):
    _require_m1()
    extra = {"budget": budget} if budget is not None else {}
    return m1_runtime.run_arm(
        arm, snapshot, base_input=_base_input(region),
        tool=tool if tool is not None else m1_runtime.DeterministicForecastStub(),
        transport=transport, clock=clock, publish_root=publish_root, **extra)


def _fetch_records_for(snapshot: dict, visible: tuple) -> dict:
    """Kernel fetch records for every visible event (honest in-run validation)."""
    _require_m1()
    records = {}
    for event in visible:
        body = snapshot["bodies"][event["body_sha256"]]
        records[event["ref"]] = m1_asof.M1FetchRecord(
            ref=event["ref"], filename=event["filename"],
            behaviour="COMPLETE_VALID",
            receipt_complete_unix_ceil=event["receipt_complete_unix_ceil"],
            body_sha256=event["body_sha256"], raw_body=body)
    return records


class IdentityPinningTests(unittest.TestCase):
    """Audit #1: recompute and pin M0 identity; drift refuses before any action."""

    def setUp(self):
        _require_m1()

    def test_config_pins_recompute_against_the_worktree(self):
        report = m1_contract.verify_run_identity(ROOT)
        self.assertEqual(report["status"], "VERIFIED")

    def test_mutated_m0_file_refuses_before_any_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp)
            for relative in (*m1_contract.PINNED_RELATIVE_PATHS,
                              m1_contract.CONFIG_PATH):
                source = ROOT / relative
                target = tree / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(source, target)
            victim = tree / "forecasting/agent_forecasting/predispatch_td_abs.py"
            victim.write_bytes(victim.read_bytes() + b"\n# drift\n")
            with self.assertRaises(m1_contract.M1IdentityDrift):
                m1_contract.verify_run_identity(tree)

    def test_prompt_and_renderer_hashes_are_pinned_in_config(self):
        config = m1_contract.load_config(ROOT)
        for name, digest in config["prompt_artifacts"]["sha256"].items():
            path = ROOT / config["prompt_artifacts"]["directory"] / name
            self.assertEqual(m1_contract.sha256_file(path), digest)
        renderer = config["digest_renderer"]
        self.assertEqual(m1_contract.sha256_file(ROOT / renderer["path"]),
                         renderer["sha256"])

    def test_withdrawal_semantics_declared_not_applicable(self):
        self.assertEqual(m1_contract.WITHDRAWAL_SEMANTICS, "NOT_APPLICABLE")

    def test_import_guard_blocks_label_score_and_training_modules(self):
        import importlib
        loaded_before = set(sys.modules)
        for name in ("m1_contract", "m1_digest", "m1_budget", "m1_actions",
                     "m1_llm", "m1_asof", "m1_runtime", "m1_publication"):
            importlib.import_module(f"forecasting.agent_forecasting.{name}")
        newly = {name for name in set(sys.modules) - loaded_before
                 if name.startswith("forecasting.")}
        allowed = {name for name in newly
                   if name in {"forecasting", "forecasting.agent_forecasting"}
                   or name.rsplit(".", 1)[-1].startswith("m1_")
                   or name.endswith(("contracts", "predispatch_td_abs", "frozen_chronos"))}
        self.assertEqual(newly - allowed, set(),
                         "M1 modules must not import label/score/training code")


class ArmLeafDiffTests(unittest.TestCase):
    """Audit #2/#4: canonical arm leaf diff equals the declared whitelist; equal budgets."""

    def setUp(self):
        _require_m1()

    def test_leaf_diffs_equal_declared_whitelist(self):
        leaves = m1_contract.ARM_LEAVES
        declared = m1_contract.declared_arm_leaf_diffs()
        for arm, previous, key in (("B", "A", "B_minus_A"), ("C", "B", "C_minus_B"),
                                   ("D", "C", "D_minus_C")):
            diff = {leaf for leaf, value in leaves[arm].items()
                    if leaves[previous].get(leaf) != value}
            self.assertEqual(diff, set(declared[key]),
                             f"{key} leaf diff drifted: {diff}")

    def test_shared_leaves_are_identical_across_arms(self):
        leaves = m1_contract.ARM_LEAVES
        declared = set().union(*m1_contract.declared_arm_leaf_diffs().values())
        shared = set(leaves["A"]) - declared
        for arm in ("B", "C", "D"):
            self.assertTrue(shared <= set(leaves[arm]))
        for leaf in shared:
            values = {leaves[arm][leaf] for arm in "ABCD"}
            self.assertEqual(len(values), 1, f"shared leaf {leaf} differs across arms")

    def test_common_budget_vector_is_equal_across_arms_and_fully_charged(self):
        for arm in "ABCD":
            self.assertEqual(m1_contract.COMMON_BUDGET_VECTOR[arm],
                             m1_contract.COMMON_BUDGET_VECTOR["A"])
            self.assertEqual(m1_contract.COMMON_BUDGET_VECTOR[arm]["retries"], 0)

    def test_c_one_call_envelope_equals_d_episode_envelope(self):
        envelope = m1_contract.load_config(ROOT)["llm_envelopes"]
        self.assertEqual(envelope["C"]["input_tokens_max"],
                         envelope["shared_input_tokens_max"])
        self.assertEqual(envelope["C"]["output_tokens_max"],
                         envelope["shared_output_tokens_max"])
        self.assertEqual(envelope["D"]["input_tokens_max_per_call"]
                         * envelope["D"]["generation_calls_max"],
                         envelope["shared_input_tokens_max"])
        self.assertEqual(envelope["D"]["output_tokens_max_per_call"]
                         * envelope["D"]["generation_calls_max"],
                         envelope["shared_output_tokens_max"])


class AsOfWrapperTests(unittest.TestCase):
    """Audit #3/#7/#9/#10: one immutable snapshot, receipt clocks, post-origin closure."""

    def setUp(self):
        _require_m1()

    def test_bcd_share_one_immutable_snapshot_identity(self):
        snapshot = _clean_snapshot()
        results = {arm: _run(arm, snapshot,
                             transport=_c_transport("USE_TYPED") if arm == "C" else
                             (_d_transport(("LIST", "NONE"),
                                           ("FETCH_REF", harness.opaque_ref(harness.NEW_FILENAME)),
                                           ("FINALIZE_TYPED", "NONE")) if arm == "D" else None))
                   for arm in "BCD"}
        for arm, result in results.items():
            self.assertEqual(result.snapshot_id, snapshot["snapshot_id"], msg=arm)
            self.assertEqual(result.origin_unix, snapshot["origin_unix"], msg=arm)
            self.assertEqual(result.cutoff, snapshot["origin_unix"], msg=arm)

    def test_chain_and_body_binding_verified_before_any_clock_trust(self):
        snapshot = _clean_snapshot()
        m1_asof.verify_event_chain(snapshot["events"])
        broken = [dict(event) for event in snapshot["events"]]
        broken[0]["receipt_complete_unix_ceil"] += 1  # stale hashes now
        with self.assertRaises(ValueError):
            m1_asof.verify_event_chain(tuple(broken))

    def test_post_origin_receipts_and_forks_cannot_change_origin_identity(self):
        base = _two_ref_snapshot()
        visible = m1_asof.materialize_visible_events(base["events"], harness.ORIGIN_UNIX)
        fetch = _fetch_records_for(base, visible)
        evidence = m1_asof.derive_typed_evidence(harness.ORIGIN_UNIX, visible, fetch)
        # a post-origin FORK of the newest vintage and a post-origin duplicate receipt
        fork_body = harness.build_body(seqno=harness.NEW_SEQNO, outcome="COMPLETE_VALID",
                                       fork_shift=True)
        late_events = list(base["events"]) + [{
            "event": "body_complete", "ref": harness.opaque_ref("LATE_FORK.zip"),
            "filename": "LATE_FORK.zip",
            "body_sha256": hashlib.sha256(fork_body).hexdigest(),
            "seqno": harness.NEW_SEQNO, "runno": "1",
            "receipt_complete_unix_ceil": harness.ORIGIN_UNIX + 600,
            "size_bytes": len(fork_body),
            "clock_health": {"wall_monotonic_consistent": True}}]
        visible_late = m1_asof.materialize_visible_events(
            m1_asof.chain_events(late_events), harness.ORIGIN_UNIX)
        evidence_late = m1_asof.derive_typed_evidence(
            harness.ORIGIN_UNIX, visible_late, fetch)
        self.assertEqual(evidence.series_hash, evidence_late.series_hash)
        self.assertEqual(evidence.values, evidence_late.values)
        self.assertEqual(evidence.selected["seqno"], evidence_late.selected["seqno"])

    def test_materialize_filters_before_parsing_and_accepts_any_iterable(self):
        snapshot = _two_ref_snapshot()
        events = snapshot["events"]
        by_tuple = m1_asof.materialize_visible_events(events, harness.ORIGIN_UNIX)
        by_generator = m1_asof.materialize_visible_events(
            (event for event in events), harness.ORIGIN_UNIX)
        self.assertEqual(by_tuple, by_generator)
        self.assertTrue(all(event["receipt_complete_unix_ceil"] <= harness.ORIGIN_UNIX
                            for event in by_tuple))

    def test_receipt_after_origin_is_invisible(self):
        snapshot = harness.build_snapshot(listing="ONE_REF",
                                          new_outcome="COMPLETE_VALID",
                                          relation="AFTER_ORIGIN",
                                          body_relation="DISTINCT_VINTAGES")
        visible = m1_asof.materialize_visible_events(snapshot["events"],
                                                     harness.ORIGIN_UNIX)
        self.assertEqual(visible, ())
        result = _run("B", snapshot)
        self.assertEqual(result.verdict, "PUBLISHED_BASE_FALLBACK")
        self.assertEqual(result.reason_code, "NO_VISIBLE_RECEIPT")

    def test_newest_visible_only_and_omission_cannot_select_older_vintage(self):
        incomplete = harness.build_snapshot(listing="TWO_REFS",
                                            new_outcome="INCOMPLETE",
                                            old_outcome="COMPLETE_VALID",
                                            relation="BEFORE_ORIGIN",
                                            body_relation="DISTINCT_VINTAGES")
        for arm, transport in (("B", None), ("C", _c_transport("USE_TYPED"))):
            result = _run(arm, incomplete, transport=transport)
            self.assertEqual(result.verdict, "PUBLISHED_BASE_FALLBACK", msg=arm)
            self.assertEqual(result.reason_code, "INCOMPLETE_NEWEST_VINTAGE", msg=arm)
        # D refusing to fetch the newest visible ref cannot select the older vintage
        snapshot = _two_ref_snapshot()
        old_ref = harness.opaque_ref(harness.OLD_FILENAME)
        result = _run("D", snapshot,
                      transport=_d_transport(("LIST", "NONE"),
                                             ("FETCH_REF", old_ref),
                                             ("FINALIZE_TYPED", "NONE")))
        self.assertEqual(result.verdict, "PUBLISHED_BASE_FALLBACK")
        self.assertEqual(result.reason_code, "ACTION_PRECONDITION")
        self.assertEqual(result.tool_calls, 1)  # only the base forecast ran

    def test_relation_equal_origin_is_visible(self):
        snapshot = harness.build_snapshot(listing="ONE_REF",
                                          new_outcome="COMPLETE_VALID",
                                          relation="EQUAL_ORIGIN",
                                          body_relation="DISTINCT_VINTAGES")
        result = _run("B", snapshot)
        self.assertEqual(result.verdict, "PUBLISHED_TYPED")

    def test_duplicate_folds_to_earliest_and_fork_fails_closed(self):
        duplicate = harness.build_snapshot(listing="TWO_REFS",
                                           new_outcome="COMPLETE_VALID",
                                           relation="BEFORE_ORIGIN",
                                           body_relation="DUPLICATE_SAME_BODY")
        result = _run("B", duplicate)
        self.assertEqual(result.verdict, "PUBLISHED_TYPED")
        later_ref = harness.opaque_ref(harness.NEW_FILENAME_B)
        d_later = _run("D", duplicate,
                       transport=_d_transport(("LIST", "NONE"),
                                              ("FETCH_REF", later_ref),
                                              ("FINALIZE_TYPED", "NONE")))
        self.assertEqual(d_later.series_hash, result.series_hash)
        fork = harness.build_snapshot(listing="TWO_REFS",
                                      new_outcome="COMPLETE_VALID",
                                      relation="BEFORE_ORIGIN",
                                      body_relation="FORK_SAME_VINTAGE")
        result = _run("B", fork)
        self.assertEqual(result.verdict, "PUBLISHED_BASE_FALLBACK")
        self.assertEqual(result.reason_code, "FORK")


class TypedDerivationTests(unittest.TestCase):
    """Audit #8/#11/#12/#13: adversarial receipts, exact mapping, tool boundary."""

    def setUp(self):
        _require_m1()

    def test_exact_48_to_24_mapping_from_the_kernel(self):
        result = _run("B", _clean_snapshot())
        self.assertEqual(result.verdict, "PUBLISHED_TYPED")
        self.assertEqual(len(result.typed_values), 24)
        from predispatchis_td_v2_fixture import fixture_values
        raw = fixture_values(70, seqno=harness.NEW_SEQNO)
        expected = [(raw[2 * h] + raw[2 * h + 1]) / 2.0 for h in range(24)]
        self.assertEqual(tuple(result.typed_values), tuple(expected))

    def test_gap_drift_and_timeout_refuse_with_no_fallback(self):
        gap = harness.build_snapshot(listing="ONE_REF", new_outcome="INCOMPLETE",
                                     relation="BEFORE_ORIGIN",
                                     body_relation="DISTINCT_VINTAGES")
        self.assertEqual(_run("B", gap).reason_code, "INCOMPLETE_NEWEST_VINTAGE")
        drift = harness.build_snapshot(listing="ONE_REF", new_outcome="SCHEMA_DRIFT",
                                       relation="BEFORE_ORIGIN",
                                       body_relation="DISTINCT_VINTAGES")
        self.assertEqual(_run("B", drift).reason_code, "SCHEMA_DRIFT")
        timeout = harness.build_snapshot(listing="ONE_REF", new_outcome="TIMEOUT",
                                         relation="BEFORE_ORIGIN",
                                         body_relation="DISTINCT_VINTAGES")
        self.assertEqual(_run("B", timeout).reason_code, "NO_VISIBLE_RECEIPT")

    def test_no_fabricated_same_semantic_history(self):
        result = _run("B", _clean_snapshot())
        self.assertEqual(result.verdict, "PUBLISHED_TYPED")
        from forecasting.agent_forecasting.frozen_chronos import build_model_input
        typed = result.typed_input_object
        self.assertNotIn("predispatchis_totaldemand_mw", typed.past_covariates)
        model_input = build_model_input(typed)
        import math
        past = model_input["past_covariates"]["predispatchis_totaldemand_mw"]
        self.assertEqual(len(past), 72)
        self.assertTrue(all(math.isnan(value) for value in past))
        future = model_input["future_covariates"]["predispatchis_totaldemand_mw"]
        self.assertEqual(list(future), list(result.typed_values))

    def test_tool_boundary_rederives_and_rejects_forged_inputs(self):
        snapshot = _clean_snapshot()
        visible = m1_asof.materialize_visible_events(snapshot["events"],
                                                     harness.ORIGIN_UNIX)
        event = visible[0]
        record = m1_asof.M1FetchRecord(
            ref=event["ref"], filename=event["filename"], behaviour="COMPLETE_VALID",
            receipt_complete_unix_ceil=event["receipt_complete_unix_ceil"],
            body_sha256=event["body_sha256"], raw_body=b"not a zip body")
        with self.assertRaises(m1_asof.M1TypedRejection) as caught:
            m1_asof.derive_typed_evidence(harness.ORIGIN_UNIX, visible,
                                          {event["ref"]: record})
        self.assertIn(caught.exception.reason_code,
                      ("SCHEMA_DRIFT", "BODY_HASH_MISMATCH"))

        from forecasting.agent_forecasting import predispatch_td_abs as m0
        body = snapshot["bodies"][event["body_sha256"]]
        receipt = m0.receipt_from_raw_body(
            body, event["receipt_complete_unix_ceil"], event["filename"])
        with self.assertRaises(ValueError):
            dataclasses.replace(receipt, rows=[(end, 999999.0) for end, _ in receipt.rows])
        # the runtime accepts no caller-supplied series: the knob does not exist
        with self.assertRaises(TypeError):
            m1_runtime.run_arm("B", snapshot, base_input=_base_input(),
                               tool=m1_runtime.DeterministicForecastStub(),
                               series_from_caller=object())

    def test_value_mutation_with_fixed_body_hash_fails_closed(self):
        snapshot = _clean_snapshot()
        visible = m1_asof.materialize_visible_events(snapshot["events"],
                                                     harness.ORIGIN_UNIX)
        event = visible[0]
        body = bytearray(snapshot["bodies"][event["body_sha256"]])
        body[-40] ^= 0x01  # different bytes, event still claims the old hash
        record = m1_asof.M1FetchRecord(
            ref=event["ref"], filename=event["filename"], behaviour="COMPLETE_VALID",
            receipt_complete_unix_ceil=event["receipt_complete_unix_ceil"],
            body_sha256=event["body_sha256"], raw_body=bytes(body))
        with self.assertRaises(m1_asof.M1TypedRejection) as caught:
            m1_asof.derive_typed_evidence(harness.ORIGIN_UNIX, visible,
                                          {event["ref"]: record})
        self.assertIn(caught.exception.reason_code,
                      ("SCHEMA_DRIFT", "BODY_HASH_MISMATCH"))

    def test_region_mismatch_refuses_attach_as_identity_drift(self):
        result = _run("B", _clean_snapshot(), region="QLD1")
        self.assertEqual(result.verdict, "TERMINAL_FAILED_NO_COMMIT")
        self.assertEqual(result.reason_code, "CONTRACT_IDENTITY_DRIFT")
        self.assertFalse(result.published)

    def test_series_identity_is_fetch_subset_independent(self):
        snapshot = _two_ref_snapshot()
        b_result = _run("B", snapshot)
        new_ref = harness.opaque_ref(harness.NEW_FILENAME)
        d_result = _run("D", snapshot,
                        transport=_d_transport(("LIST", "NONE"),
                                               ("FETCH_REF", new_ref),
                                               ("FINALIZE_TYPED", "NONE")))
        self.assertEqual(d_result.verdict, "PUBLISHED_TYPED")
        self.assertEqual(d_result.series_hash, b_result.series_hash)
        self.assertEqual(d_result.input_hash, b_result.input_hash)


class DigestAndPromptTests(unittest.TestCase):
    """Audit #3 digest fairness + frozen prompt/model/parser identity."""

    def setUp(self):
        _require_m1()

    def _assert_no_numbers(self, value):
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            raise AssertionError(f"numeric fact leaked into the digest: {value!r}")
        if isinstance(value, dict):
            for item in value.values():
                self._assert_no_numbers(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._assert_no_numbers(item)

    def test_digest_renders_only_enums_bools_and_opaque_refs(self):
        result = _run("D", _two_ref_snapshot(),
                      transport=_d_transport(("LIST", "NONE"),
                                             ("FINALIZE_BASE", "NONE")))
        self.assertTrue(result.digests)
        for digest in result.digests:
            facts = json.loads(digest)
            self.assertEqual(set(facts), set(m1_digest.ALLOWED_FACT_KEYS))
            self._assert_no_numbers(facts)

    def test_digest_fact_space_is_shared_between_c_and_d(self):
        snapshot = _clean_snapshot()
        c_digest = _run("C", snapshot, transport=_c_transport("USE_TYPED")).digests[0]
        d_digests = _run("D", snapshot,
                         transport=_d_transport(("LIST", "NONE"),
                                                ("FINALIZE_BASE", "NONE"))).digests
        self.assertEqual(set(json.loads(c_digest)), set(m1_digest.ALLOWED_FACT_KEYS))
        for digest in d_digests:
            self.assertEqual(set(json.loads(digest)), set(json.loads(c_digest)))

    def test_digest_renderer_rejects_forbidden_facts(self):
        with self.assertRaises(m1_digest.DigestViolation):
            m1_digest.render_digest({"state": "LISTED", "totaldemand_mw": 7000.5})
        with self.assertRaises(m1_digest.DigestViolation):
            m1_digest.render_digest({"state": "LISTED",
                                     "timestamp": "2026-10-29T00:00:00Z"})

    def test_request_bytes_use_the_frozen_prompts(self):
        request = m1_llm.build_request_bytes("C", '{"state":"FETCHED"}', ROOT)
        payload = json.loads(request)
        config = m1_contract.load_config(ROOT)
        directory = ROOT / config["prompt_artifacts"]["directory"]
        system = (directory / "m1_common_system_prompt.txt").read_text()
        template = (directory / "m1_c_prompt_template.txt").read_text()
        self.assertEqual(payload["instructions"], system)
        self.assertEqual(payload["input"],
                         template.replace("{{canonical_digest}}",
                                          '{"state":"FETCHED"}'))
        self.assertEqual(payload["model"], m1_contract.LLM_MODEL)
        self.assertEqual(payload["reasoning"], {"effort": "none"})
        self.assertFalse(payload["store"])
        self.assertEqual(payload["truncation"], "disabled")
        self.assertFalse(payload["parallel_tool_calls"])
        self.assertEqual(set(payload),
                         {"model", "instructions", "input", "reasoning", "store",
                          "truncation", "parallel_tool_calls", "max_output_tokens",
                          "text"})
        self.assertEqual(payload["max_output_tokens"],
                         m1_contract.PER_CALL_OUTPUT_TOKENS["C"])
        output_format = payload["text"]["format"]
        self.assertEqual(output_format, {
            "type": "json_schema", "name": "m1_c_decision", "strict": True,
            "schema": json.loads((directory / "m1_c_decision_schema.json").read_text())})
        self.assertEqual(config["prompt_artifacts"]["sha256"]["m1_c_decision_schema.json"],
                         m1_contract.sha256_file(directory / "m1_c_decision_schema.json"))

    def test_parser_rejects_the_full_forbidden_zoo(self):
        bad = [
            b'{"action":"LIST","ref":"NONE","extra":1}',
            b'{"action":"LIST","ref":"NONE","ref":"NONE"}',
            b'{"action":"LIST","ref":"NONE"} trailing',
            b'{"action":"LIST","ref":123}',
            b'{"action":"LIST","ref":7000}',
            b'{"action":"HACK","ref":"NONE"}',
            b'{"action":"FETCH_REF","ref":"v_ffffffffffffffff"}',
            b'{"action":"LIST","ref":"v_0000000000000000"}',
            b'{"action":"FINALIZE_TYPED","ref":"v_0000000000000000"}',
            b'{"action":"LIST","ref":"https://nemweb.example"}',
            b'{"action":"LIST","ref":"`sh -c calc`"}',
            b'{"action":"LIST","ref":"__import__(\'os\')"}',
            b'{"action":"FETCH_REF","ref":"NONE","mw":7000.0}',
            b'{"action":"LIST","ref":"NaN"}',
            b'{"action":"LIST","ref":"Infinity"}',
        ]
        enabled = ("v_abcd0123ef456789",)
        for payload in bad:
            with self.assertRaises(m1_llm.LlmParseError, msg=payload.decode()):
                m1_llm.parse_d_action(_ok_response(payload), enabled_refs=enabled)
        good = m1_llm.parse_d_action(
            _ok_response(b'{"action":"FETCH_REF","ref":"v_abcd0123ef456789"}'),
            enabled_refs=enabled)
        self.assertEqual((good[0].name, good[1]), ("FETCH_REF", "v_abcd0123ef456789"))
        for broken in (
            m1_llm.M1LlmResponse(status="COMPLETED", model="gpt-5.4-alias",
                                 output_raw=b'{"action":"LIST","ref":"NONE"}'),
            m1_llm.M1LlmResponse(status="REFUSED", model=m1_contract.LLM_MODEL,
                                 output_raw=b""),
            m1_llm.M1LlmResponse(status="TIMEOUT", model=m1_contract.LLM_MODEL,
                                 output_raw=b""),
            m1_llm.M1LlmResponse(status="INCOMPLETE", model=m1_contract.LLM_MODEL,
                                 output_raw=b""),
            m1_llm.M1LlmResponse(status="COMPLETED", model=m1_contract.LLM_MODEL,
                                 output_raw=b"x" * 4097),
        ):
            with self.assertRaises(m1_llm.LlmParseError):
                m1_llm.parse_d_action(broken, enabled_refs=enabled)

    def test_function_call_envelope_is_accepted_strictly(self):
        payload = (b'{"name":"m1_d_action",'
                   b'"arguments":"{\\"action\\":\\"LIST\\",\\"ref\\":\\"NONE\\"}"}')
        good = m1_llm.parse_d_action(_ok_response(payload), enabled_refs=())
        self.assertEqual((good[0].name, good[1]), ("LIST", "NONE"))
        prose = b'{"name":"m1_d_action","arguments":"{}"} surely'
        with self.assertRaises(m1_llm.LlmParseError):
            m1_llm.parse_d_action(_ok_response(prose), enabled_refs=())

    def test_c_parser_accepts_only_the_two_decisions(self):
        good = m1_llm.parse_c_decision(
            _ok_response(harness.llm_valid_c_decision_bytes("USE_TYPED", "NONE")),
            enabled_evidence_refs=())
        self.assertEqual(good, ("USE_TYPED", "NONE"))
        with self.assertRaises(m1_llm.LlmParseError):
            m1_llm.parse_c_decision(
                _ok_response(b'{"decision":"RE_QUERY","evidence_ref":"NONE"}'),
                enabled_evidence_refs=())

    def test_cost_accounting_uses_frozen_rates_in_micro_usd(self):
        self.assertEqual(m1_llm.cost_micro_usd(16384, 1024), 56320)
        self.assertEqual(m1_llm.cost_micro_usd(0, 0), 0)


class ActionRegistryTests(unittest.TestCase):
    """Audit #6: closed typed vocabulary; numeric/free-form input fails closed."""

    def setUp(self):
        _require_m1()
        self.snapshot = _two_ref_snapshot()

    def test_unknown_and_out_of_order_actions_have_zero_side_effects(self):
        old = harness.opaque_ref(harness.OLD_FILENAME)
        bad_traces = [
            (("FETCH_REF", old),),
            (("FINALIZE_TYPED", "NONE"),),
            (("LIST", "NONE"), ("LIST", "NONE")),
            (("LIST", "NONE"), ("FINALIZE_TYPED", "NONE")),
            (("LIST", "NONE"), ("FETCH_REF", old), ("FINALIZE_TYPED", "NONE")),
        ]
        expected_replay = [0, 0, 1, 1, 2]
        for trace, replay in zip(bad_traces, expected_replay):
            result = _run("D", self.snapshot, transport=_d_transport(*trace))
            self.assertEqual(result.verdict, "PUBLISHED_BASE_FALLBACK",
                             msg=str(trace))
            self.assertEqual(result.reason_code, "ACTION_PRECONDITION", msg=str(trace))
            self.assertEqual(result.tool_calls, 1, msg=str(trace))
            self.assertEqual(result.source_replay_calls, replay, msg=str(trace))

    def test_fetch_refuses_repeats_and_respects_the_two_get_cap(self):
        old = harness.opaque_ref(harness.OLD_FILENAME)
        result = _run("D", self.snapshot,
                      transport=_d_transport(("LIST", "NONE"),
                                             ("FETCH_REF", old),
                                             ("FETCH_REF", old)))
        self.assertEqual(result.reason_code, "ACTION_PRECONDITION")
        self.assertEqual(result.source_replay_calls, 2)

    def test_four_proposal_cap_is_charged_and_a_terminal_fourth_is_legal(self):
        old = harness.opaque_ref(harness.OLD_FILENAME)
        new = harness.opaque_ref(harness.NEW_FILENAME)
        result = _run("D", self.snapshot,
                      transport=_d_transport(("LIST", "NONE"),
                                             ("FETCH_REF", old),
                                             ("FETCH_REF", new),
                                             ("FINALIZE_TYPED", "NONE")))
        self.assertEqual(result.verdict, "PUBLISHED_TYPED")
        self.assertEqual(result.llm_calls, 4)

    def test_state_machine_enums_and_terminal_absorption(self):
        states = {state.name for state in m1_actions.M1State}
        self.assertEqual(states, {"INIT", "A_READY", "LISTED", "FETCHED", "TYPED_READY",
                                  "TERMINAL_TYPED", "TERMINAL_BASE_FALLBACK",
                                  "TERMINAL_SYSTEM_MISSING",
                                  "TERMINAL_FAILED_NO_COMMIT"})
        kinds = {kind.name for kind in m1_actions.M1ActionKind}
        self.assertEqual(kinds, {"LIST", "FETCH_REF", "FINALIZE_TYPED", "FINALIZE_BASE"})
        for state in m1_actions.TERMINAL_STATES:
            self.assertEqual(
                m1_actions.enabled_actions(state, list_calls=0, get_calls=0,
                                           available_refs=()), ())
        self.assertEqual(
            m1_actions.enabled_actions(m1_actions.M1State.A_READY, list_calls=0,
                                       get_calls=0, available_refs=()),
            (m1_actions.M1ActionKind.LIST, m1_actions.M1ActionKind.FINALIZE_BASE))


class BudgetDeadlineTests(unittest.TestCase):
    """Audit #4/#16: ledger enforcement, deadlines, failure matrix, origin retention."""

    def setUp(self):
        _require_m1()

    def test_every_cap_fails_at_cap_plus_one_with_no_side_effects(self):
        for budget_class, cap in m1_budget.CAPS.items():
            ledger = m1_budget.BudgetLedger()
            ledger.charge(budget_class, cap, "at cap")
            try:
                ledger.reserve(budget_class, 1, "cap+1 must refuse")
            except m1_budget.BudgetRefused:
                pass
            else:
                self.fail(f"{budget_class.name}: reserve leaked past cap {cap}")
            self.assertEqual(ledger.fold()[budget_class], cap, msg=budget_class.name)
            self.assertTrue(any(entry["op"] == "REFUSE" for entry in ledger.entries),
                            msg=budget_class.name)

    def test_reserve_without_settle_is_fully_consumed_and_tamper_detected(self):
        ledger = m1_budget.BudgetLedger()
        ledger.reserve(m1_budget.BudgetClass.LLM_INPUT_TOKENS, 100, "reserve")
        ledger.close()
        self.assertEqual(ledger.fold()[m1_budget.BudgetClass.LLM_INPUT_TOKENS], 100)
        other = m1_budget.BudgetLedger()
        seq = other.reserve(m1_budget.BudgetClass.LLM_OUTPUT_TOKENS, 10, "reserve")
        other.settle(seq, 25, "provider used more than reserved")
        self.assertIn(m1_budget.BudgetClass.LLM_OUTPUT_TOKENS, other.overrun_classes)
        other.entries[0]["amount"] = 0
        with self.assertRaises(ValueError):
            other.verify_chain()

    def test_ledger_fold_equals_reported_totals(self):
        result = _run("B", _clean_snapshot())
        fold = result.budget_fold
        ledger = m1_budget.BudgetLedger.from_entries(result.budget_entries)
        self.assertEqual(ledger.fold(), fold)
        self.assertEqual(fold[m1_budget.BudgetClass.SOURCE_REPLAY_CALLS], 2)
        self.assertEqual(fold[m1_budget.BudgetClass.FORECAST_CALLS], 2)
        self.assertEqual(fold[m1_budget.BudgetClass.LLM_GENERATIONS], 0)

    def test_deadline_miss_routes_to_system_missing_or_exact_base(self):
        snapshot = _clean_snapshot()
        too_late = m1_runtime.DeterministicClock(
            start=harness.ORIGIN_UNIX + 601, step=0.0)
        result = _run("B", snapshot, clock=too_late)
        self.assertEqual(result.verdict, "SYSTEM_MISSING")
        self.assertEqual(result.reason_code, "DEADLINE_MISSED")
        no_typed_reserve = m1_runtime.DeterministicClock(
            start=harness.ORIGIN_UNIX + 400, step=0.0)
        result = _run("B", snapshot, clock=no_typed_reserve)
        self.assertEqual(result.verdict, "PUBLISHED_BASE_FALLBACK")
        self.assertEqual(result.reason_code, "DEADLINE_MISSED")

    def test_failure_matrix_retains_every_origin(self):
        cases = [
            ("B", "tool_timeout", {"tool": m1_runtime.TimeoutForecastTool()}),
            ("C", "llm_timeout", {"transport": m1_runtime.ScriptedTransport([
                m1_llm.M1LlmResponse(status="TIMEOUT", model=m1_contract.LLM_MODEL,
                                     output_raw=b"")])}),
            ("C", "llm_refusal", {"transport": m1_runtime.ScriptedTransport([
                m1_llm.M1LlmResponse(status="REFUSED", model=m1_contract.LLM_MODEL,
                                     output_raw=b"")])}),
            ("C", "llm_malformed", {"transport": m1_runtime.ScriptedTransport([
                _ok_response(b'{"decision":"USE_TYPED"}')])}),
            ("C", "c_abstain", {"transport": _c_transport("USE_BASE")}),
            ("D", "invalid_action", {"transport": m1_runtime.ScriptedTransport([
                _ok_response(b'{"action":"RUN_PYTHON","ref":"NONE"}')])}),
        ]
        snapshot = _clean_snapshot()
        for arm, name, kwargs in cases:
            result = _run(arm, snapshot, **kwargs)
            self.assertIn(result.verdict,
                          ("PUBLISHED_BASE_FALLBACK", "SYSTEM_MISSING"), msg=name)
            self.assertIn(result.reason_code, m1_contract.ORIGIN_REASON_CODES, msg=name)
            self.assertIsNotNone(result.input_hash, msg=name)  # never dropped

    def test_overrun_forbids_typed_success(self):
        greedy = m1_runtime.SimulatedUsageTransport(
            decision="USE_TYPED", input_tokens=20000, output_tokens=2000)
        result = _run("C", _clean_snapshot(), transport=greedy)
        self.assertEqual(result.verdict, "PUBLISHED_BASE_FALLBACK")
        self.assertEqual(result.reason_code, "BUDGET_OVERRUN")


class COneShotTests(unittest.TestCase):
    """Audit #5: fixed retrieval, exactly one LLM call, no adaptive follow-up."""

    def setUp(self):
        _require_m1()

    def test_retrieval_trace_is_independent_of_the_llm_output(self):
        snapshot = _two_ref_snapshot()
        typed = _run("C", snapshot, transport=_c_transport("USE_TYPED"))
        abstain = _run("C", snapshot, transport=_c_transport("USE_BASE"))
        b_run = _run("B", snapshot)
        self.assertEqual(typed.retrieval_trace, b_run.retrieval_trace)
        self.assertEqual(abstain.retrieval_trace, b_run.retrieval_trace)
        self.assertEqual(typed.llm_calls, 1)
        self.assertEqual(abstain.llm_calls, 1)

    def test_bad_json_timeout_or_requery_cannot_trigger_followup(self):
        for response in (
            m1_llm.M1LlmResponse(status="TIMEOUT", model=m1_contract.LLM_MODEL,
                                 output_raw=b""),
            _ok_response(b"not json at all"),
            _ok_response(b'{"decision":"RE_QUERY","evidence_ref":"NONE"}'),
        ):
            transport = m1_runtime.ScriptedTransport([response])
            result = _run("C", _clean_snapshot(), transport=transport)
            self.assertEqual(transport.calls, 1)
            self.assertEqual(result.verdict, "PUBLISHED_BASE_FALLBACK")


class ByteIdenticalTerminalTests(unittest.TestCase):
    """Audit #14: same decision => byte-identical inputs and tool traces."""

    def setUp(self):
        _require_m1()

    def test_bcd_same_decision_produces_identical_hashes(self):
        snapshot = _two_ref_snapshot()
        new = harness.opaque_ref(harness.NEW_FILENAME)
        results = {"B": _run("B", snapshot),
                   "C": _run("C", snapshot, transport=_c_transport("USE_TYPED")),
                   "D": _run("D", snapshot,
                             transport=_d_transport(("LIST", "NONE"),
                                                    ("FETCH_REF", new),
                                                    ("FINALIZE_TYPED", "NONE")))}
        base = results["B"]
        for arm, result in results.items():
            self.assertEqual(result.verdict, "PUBLISHED_TYPED", msg=arm)
            self.assertEqual(result.input_hash, base.input_hash, msg=arm)
            self.assertEqual(result.series_hash, base.series_hash, msg=arm)
            self.assertEqual(result.tool_trace_hash, base.tool_trace_hash, msg=arm)
            self.assertEqual(result.forecast_sha256, base.forecast_sha256, msg=arm)
            self.assertNotIn(arm, result.canonical_model_input_json)
            self.assertNotIn(arm, result.canonical_model_input_json.lower())

    def test_fallback_terminals_are_byte_identical_to_a(self):
        snapshot = harness.build_snapshot(listing="ONE_REF", new_outcome="TIMEOUT",
                                          relation="BEFORE_ORIGIN",
                                          body_relation="DISTINCT_VINTAGES")
        a_run = _run("A", snapshot)
        for arm, transport in (("B", None), ("C", _c_transport("USE_TYPED")),
                               ("D", _d_transport(("LIST", "NONE"),
                                                  ("FINALIZE_BASE", "NONE")))):
            result = _run(arm, snapshot, transport=transport)
            self.assertEqual(result.forecast_sha256, a_run.forecast_sha256, msg=arm)
            self.assertEqual(result.input_hash, a_run.input_hash, msg=arm)


class SingleSinkTests(unittest.TestCase):
    """Audit #15: frozen tool is the single sink; output unmodified; unique inputs."""

    def setUp(self):
        _require_m1()

    def test_output_bytes_are_published_unchanged_without_sorting_or_clipping(self):
        tool = m1_runtime.DeterministicForecastStub()
        result = _run("B", _clean_snapshot(), tool=tool)
        self.assertEqual(result.forecast_bytes, tool.last_output)
        quantiles = json.loads(result.forecast_bytes)["quantiles"]
        crossings = sum(1 for row in quantiles for i in range(4)
                        if row[i] > row[i + 1])
        self.assertGreater(crossings, 0,
                           "the stub emits non-monotone columns; none may be sorted")

    def test_unique_input_forecast_once_and_second_call_refused(self):
        sink = m1_runtime.ForecastSink(m1_runtime.DeterministicForecastStub())
        model_input = {"target": [1.0] * 72}
        first = sink.forecast_once(model_input, params={"prediction_length": 24})
        self.assertIsNotNone(first)
        with self.assertRaises(m1_runtime.ForecastReuseForbidden):
            sink.forecast_once(model_input, params={"prediction_length": 24})
        self.assertEqual(sink.call_count, 1)

    def test_max_two_forecast_calls_enforced(self):
        result = _run("B", _clean_snapshot())
        self.assertLessEqual(result.tool_calls, 2)
        self.assertEqual(result.budget_fold[m1_budget.BudgetClass.FORECAST_CALLS], 2)


class AtomicPublicationTests(unittest.TestCase):
    """Audit #17: atomic ledger, injectivity, manifest closure, real exit codes."""

    def setUp(self):
        _require_m1()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _payloads(self):
        # audit-3/H4 fixture honesty: the closure must recompute from real
        # archives (model ledger + forecast trace), so the probe payloads carry
        # a self-consistent tool/trace/input identity instead of fake hexes
        ledger = m1_budget.BudgetLedger()
        ledger.charge(m1_budget.BudgetClass.SOURCE_REPLAY_CALLS, 1, "probe")
        entries = ("\n".join(json.dumps(entry, sort_keys=True,
                                        separators=(",", ":"))
                             for entry in ledger.entries) + "\n").encode()
        self._ledger = ledger
        self._forecast = b"\x00\x01quantile-bytes"
        self._input_hash = hashlib.sha256(b"canonical-probe-input").hexdigest()
        tool = m1_runtime.DeterministicForecastStub.identity
        self._tool_hash = hashlib.sha256(json.dumps(
            tool, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        model_ledger = json.dumps(
            {"tool": tool,
             "trace": [{"phase": "base", "input_hash": self._input_hash,
                        "params_hash": hashlib.sha256(b"{}").hexdigest(),
                        "output_sha256": hashlib.sha256(
                            self._forecast).hexdigest(),
                        "output_bytes": len(self._forecast)}],
             "llm_calls": 0}, sort_keys=True, separators=(",", ":")).encode()
        return {"result.json": json.dumps(
                    {"verdict": "PUBLISHED_TYPED", "origin_unix": harness.ORIGIN_UNIX,
                     "input_hash": self._input_hash}, sort_keys=True,
                    separators=(",", ":")).encode(),
                "forecast.bin": self._forecast,
                "budget_ledger.jsonl": entries,
                "model_ledger.jsonl": model_ledger}

    def _closure(self, **overrides):
        ledger = getattr(self, "_ledger", None) or m1_budget.BudgetLedger()
        closure = {
            "origin_unix": harness.ORIGIN_UNIX,
            "deadline_unix": harness.ORIGIN_UNIX + m1_contract.DEADLINE_OFFSET,
            "clock_final": harness.ORIGIN_UNIX + 1,
            "budget_fold": {cls.name: value
                            for cls, value in ledger.fold().items()},
            "budget_caps": {cls.name: value
                            for cls, value in m1_budget.CAPS.items()},
            "budget_chain_ok": True,
            "tool_identity_sha256": getattr(self, "_tool_hash", None)
            or hashlib.sha256(json.dumps(
                m1_runtime.DeterministicForecastStub.identity,
                sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "input_hash": getattr(self, "_input_hash", None)
            or hashlib.sha256(b"canonical-probe-input").hexdigest(),
            "canonical_model_input_sha256": getattr(self, "_input_hash", None)
            or hashlib.sha256(b"canonical-probe-input").hexdigest(),
        }
        closure.update(overrides)
        return closure

    def test_publish_verify_and_refuse_duplicates(self):
        final = m1_publication.publish_run(
            self.root, "B_1641218400_x", self._payloads(),
            identity={"contract_id": m1_contract.CONTRACT_ID},
            closure=self._closure())
        report = m1_publication.verify_run_closure(final)
        self.assertEqual(report["status"], "CLOSED")
        self.assertEqual(sorted(report["inventory"]),
                         ["COMMIT.json", "budget_ledger.jsonl", "forecast.bin",
                          "manifest.json", "model_ledger.jsonl", "result.json"])
        with self.assertRaises(m1_publication.PublicationError):
            m1_publication.publish_run(self.root, "B_1641218400_x", self._payloads(),
                                       identity={"contract_id": m1_contract.CONTRACT_ID},
                                       closure=self._closure())

    def test_tampered_payload_breaks_closure(self):
        final = m1_publication.publish_run(self.root, "run_tamper", self._payloads(),
                                           identity={"contract_id": "x"},
                                           closure=self._closure())
        (final / "forecast.bin").write_bytes(b"edited")
        with self.assertRaises(m1_publication.PublicationError):
            m1_publication.verify_run_closure(final)

    def test_extra_file_after_commit_breaks_closure(self):
        final = m1_publication.publish_run(self.root, "run_extra", self._payloads(),
                                           identity={"contract_id": "x"},
                                           closure=self._closure())
        (final / "smuggled.txt").write_bytes(b"x")
        with self.assertRaises(m1_publication.PublicationError):
            m1_publication.verify_run_closure(final)

    def test_crash_before_commit_leaves_no_visible_success(self):
        import forecasting.agent_forecasting.m1_publication as module
        original = module.os.rename
        counter = {"n": 0}

        def flaky(src, dst):
            counter["n"] += 1
            if counter["n"] == 1:  # crash at the staging -> final rename
                raise OSError("injected crash")
            return original(src, dst)

        module.os.rename = flaky
        try:
            with self.assertRaises(OSError):
                m1_publication.publish_run(self.root, "run_crash", self._payloads(),
                                           identity={"contract_id": "x"},
                                           closure=self._closure())
        finally:
            module.os.rename = original
        runs = (self.root / "runs")
        if runs.exists():
            for orphan in runs.glob("*"):
                self.assertFalse((orphan / "COMMIT.json").exists(),
                                 "an uncommitted directory is never a success")
        record = m1_publication.write_failure_record(
            self.root, "run_crash", {"reason": "INJECTED_CRASH"})
        self.assertTrue(record.exists())

    def test_run_arm_publishes_atomically_with_real_verdict(self):
        publish_root = self.root / "evidence"
        result = _run("B", _clean_snapshot(), publish_root=publish_root)
        self.assertTrue(result.published)
        report = m1_publication.verify_run_closure(Path(result.publish_dir))
        self.assertEqual(report["status"], "CLOSED")
        committed = json.loads((Path(result.publish_dir) / "result.json").read_text())
        self.assertEqual(committed["verdict"], "PUBLISHED_TYPED")
        manifest = json.loads((Path(result.publish_dir) / "manifest.json").read_text())
        self.assertIn("forecast.bin", manifest["payloads"])
        self.assertNotIn("COMMIT.json", manifest["payloads"])

    def test_command_logger_unique_paths_and_real_exit_codes(self):
        script = ROOT / "scripts" / "run_logged_m1.sh"
        if not script.exists():
            self.fail("scripts/run_logged_m1.sh must ship with the RED commit")
        env = dict(os.environ)
        env["M1_EVIDENCE_ROOT"] = str(self.root / "cmdroot")
        env["M1_WT"] = str(ROOT)
        run = subprocess.run(
            ["bash", str(script), "t_ok", "t_ok.txt", "--", "sh", "-c", "exit 7"],
            env=env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 7)
        log = self.root / "cmdroot" / "t_ok.txt"
        self.assertIn("EXIT_CODE=7", log.read_text())
        again = subprocess.run(
            ["bash", str(script), "t_dup", "t_ok.txt", "--", "true"],
            env=env, capture_output=True, text=True)
        self.assertEqual(again.returncode, 99)  # unique log path refused
        ledger = (self.root / "cmdroot" / "commands.jsonl").read_text().splitlines()
        records = [json.loads(line) for line in ledger]
        self.assertEqual([record["id"] for record in records], ["t_ok"])
        self.assertEqual(records[0]["exit_code"], 7)
        self.assertEqual(records[0]["log_path"], "t_ok.txt")


class MirrorFairnessTests(unittest.TestCase):
    """Audit-2 blocker 1: same body, multiple refs, per-ref transport outcomes.

    The duplicate-same-body harness axis used to force both mirrors to one
    outcome, so B/C never faced a legitimately half-dead mirror pair; B deduped
    by body hash before fetching, wasted its second GET, and fell back to the
    base forecast while D published typed.  B must be the maximal fixed policy.
    """

    def setUp(self):
        _require_m1()

    def test_b_fetches_both_mirrors_and_survives_one_transport_death(self):
        for failing in ("NEW", "DUP"):
            for failure in harness.TRANSPORT_FAILURES:
                snapshot = _asym_snapshot(failing, failure)
                result = _run("B", snapshot)
                self.assertEqual(result.verdict, "PUBLISHED_TYPED",
                                 msg=f"{failing}/{failure}")
                self.assertEqual(result.reason_code, "PUBLISHED_TYPED",
                                 msg=f"{failing}/{failure}")
                self.assertEqual(result.source_replay_calls, 3,
                                 msg=f"{failing}/{failure}")  # LIST + both GETs
                dead_name = (harness.NEW_FILENAME if failing == "NEW"
                             else harness.NEW_FILENAME_B)
                dead = harness.opaque_ref(dead_name)
                outcomes = {entry["ref"]: entry["outcome"]
                            for entry in result.retrieval_trace
                            if entry["step"] == "FETCH_REF"}
                self.assertEqual(outcomes[dead], failure, msg=f"{failing}/{failure}")

    def test_b_survives_when_only_the_dead_mirror_is_visible(self):
        # AFTER cutoff: only the dead mirror's receipt is visible while the live
        # mirror is a listed ref with a post-origin receipt; the fixed plan still
        # spends its second GET there and publishes the same pre-origin body
        result = _run("B", _asym_snapshot("DUP", "TIMEOUT", "AFTER_ORIGIN"))
        self.assertEqual(result.verdict, "PUBLISHED_TYPED")
        self.assertEqual(result.reason_code, "PUBLISHED_TYPED")
        self.assertEqual(result.source_replay_calls, 3)

    def test_asymmetric_mirrors_keep_bcd_terminals_identical(self):
        snapshot = _asym_snapshot("NEW", "TIMEOUT")
        dup = harness.opaque_ref(harness.NEW_FILENAME_B)
        b = _run("B", snapshot)
        c = _run("C", snapshot, transport=_c_transport("USE_TYPED"))
        d = _run("D", snapshot,
                 transport=_d_transport(("LIST", "NONE"), ("FETCH_REF", dup),
                                        ("FINALIZE_TYPED", "NONE")))
        self.assertEqual(b.verdict, "PUBLISHED_TYPED")
        for arm, result in (("C", c), ("D", d)):
            self.assertEqual(
                (result.verdict, result.input_hash, result.series_hash,
                 result.tool_trace_hash),
                (b.verdict, b.input_hash, b.series_hash, b.tool_trace_hash),
                msg=arm)

    def test_typed_derivation_is_fetch_order_independent(self):
        snapshot = _asym_snapshot("NEW", "TIMEOUT")
        visible = m1_asof.materialize_visible_events(snapshot["events"],
                                                     harness.ORIGIN_UNIX)
        records = {}
        for event in visible:  # honest in-run fetch of every visible mirror
            behaviour = snapshot["fetch_behaviours"][event["ref"]]
            records[event["ref"]] = m1_asof.M1FetchRecord(
                ref=event["ref"], filename=event["filename"], behaviour=behaviour,
                receipt_complete_unix_ceil=event["receipt_complete_unix_ceil"],
                body_sha256=event["body_sha256"],
                raw_body=(snapshot["bodies"][event["body_sha256"]]
                          if behaviour in harness.BODY_LEVEL_OUTCOMES else None))
        first = m1_asof.derive_typed_evidence(harness.ORIGIN_UNIX, visible, records)
        flipped = m1_asof.derive_typed_evidence(
            harness.ORIGIN_UNIX, visible,
            dict(reversed(list(records.items()))))
        self.assertEqual(first.values, flipped.values)
        self.assertEqual(first.series_hash, flipped.series_hash)
        self.assertEqual(first.selected, flipped.selected)
        self.assertEqual(first.publication_basis, flipped.publication_basis)

    def test_canonical_plan_prefers_both_mirrors_of_the_newest_visible_body(self):
        snapshot = m1_runtime.M1SourceSnapshot.from_mapping(
            _asym_snapshot("NEW", "TIMEOUT"))
        plan = snapshot.canonical_fetch_refs(harness.ORIGIN_UNIX)
        self.assertEqual(set(plan), {harness.opaque_ref(harness.NEW_FILENAME),
                                     harness.opaque_ref(harness.NEW_FILENAME_B)})
        late = m1_runtime.M1SourceSnapshot.from_mapping(
            _asym_snapshot("DUP", "TIMEOUT", "AFTER_ORIGIN"))
        self.assertEqual(len(late.canonical_fetch_refs(harness.ORIGIN_UNIX)), 2)

    def test_harness_rejects_same_body_with_different_body_level_outcomes(self):
        with self.assertRaises(ValueError):
            harness.build_snapshot(listing="TWO_REFS",
                                   body_relation="DUPLICATE_SAME_BODY",
                                   new_outcome="COMPLETE_VALID",
                                   dup_outcome="INCOMPLETE",
                                   relation="BEFORE_ORIGIN")
        with self.assertRaises(ValueError):
            harness.build_snapshot(listing="ONE_REF", new_outcome="COMPLETE_VALID",
                                   relation="BEFORE_ORIGIN",
                                   body_relation="DISTINCT_VINTAGES",
                                   dup_outcome="COMPLETE_VALID")


class FailClosedBudgetTests(unittest.TestCase):
    """Audit-2 blocker 2: reserve precedes every effect; refusals never escape."""

    def setUp(self):
        _require_m1()

    def test_generation_cap_zero_never_touches_the_transport(self):
        caps = dict(m1_budget.CAPS)
        caps[m1_budget.BudgetClass.LLM_GENERATIONS] = 0
        transport = _c_transport("USE_TYPED")
        result = _run("C", _clean_snapshot(), transport=transport,
                      budget=m1_budget.BudgetLedger(caps=caps))
        self.assertEqual(transport.calls, 0)  # fail closed BEFORE the effect
        self.assertEqual(result.verdict, "PUBLISHED_BASE_FALLBACK")
        self.assertEqual(result.reason_code, "BUDGET_EXHAUSTED")

    def test_transport_timeout_is_billed_like_a_success(self):
        class RaisingTransport:
            identity = {"kind": "raising_probe_transport", "version": 1}

            def __init__(self):
                self.calls = 0

            def generate(self, arm, turn, request_bytes):
                self.calls += 1
                raise TimeoutError("provider timeout")

        transport = RaisingTransport()
        result = _run("C", _clean_snapshot(), transport=transport)
        self.assertEqual(transport.calls, 1)
        self.assertEqual(result.reason_code, "LLM_TIMEOUT")
        fold = result.budget_fold
        self.assertEqual(fold[m1_budget.BudgetClass.LLM_GENERATIONS], 1)
        self.assertEqual(fold[m1_budget.BudgetClass.LLM_INPUT_TOKENS],
                         m1_runtime.C_PER_CALL_INPUT_TOKENS)
        self.assertEqual(fold[m1_budget.BudgetClass.LLM_OUTPUT_TOKENS],
                         m1_runtime.C_PER_CALL_OUTPUT_TOKENS)

    def test_forecast_budget_refusal_maps_to_a_frozen_reason(self):
        for arm in ("A", "B"):
            caps = dict(m1_budget.CAPS)
            caps[m1_budget.BudgetClass.FORECAST_CALLS] = 0
            result = _run(arm, _clean_snapshot(),
                          budget=m1_budget.BudgetLedger(caps=caps))
            self.assertIn(result.verdict,
                          ("SYSTEM_MISSING", "PUBLISHED_BASE_FALLBACK"), msg=arm)
            self.assertEqual(result.reason_code, "BUDGET_EXHAUSTED", msg=arm)

    def test_forecast_tool_runtime_error_is_contained(self):
        class ExplodingTool:
            identity = {"kind": "exploding_probe_stub", "version": 1}

            def predict(self, model_input, params=None):
                raise RuntimeError("boom")

        base_result = _run("A", _clean_snapshot(), tool=ExplodingTool())
        self.assertEqual(base_result.verdict, "SYSTEM_MISSING")
        self.assertEqual(base_result.reason_code, "FORECAST_TOOL_FAILURE")

        class SecondCallExplodes:
            identity = {"kind": "second_call_explodes", "version": 1}

            def __init__(self):
                self.calls = 0

            def predict(self, model_input, params=None):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("boom on the typed call")
                return b'{"quantiles": [[1.0] * 5] * 24}'

        typed_result = _run("B", _two_ref_snapshot(), tool=SecondCallExplodes())
        self.assertEqual(typed_result.verdict, "PUBLISHED_BASE_FALLBACK")
        self.assertEqual(typed_result.reason_code, "FORECAST_TOOL_FAILURE")

    def _body_snapshot(self, bodies_spec):
        events, listing, bodies, behaviours = [], [], {}, {}
        for filename, seqno, clock, body, behaviour in bodies_spec:
            ref = harness.opaque_ref(filename)
            digest = hashlib.sha256(body).hexdigest()
            listing.append({"ref": ref, "filename": filename})
            bodies[digest] = body
            behaviours[ref] = behaviour
            events.append({"event": "body_complete", "ref": ref,
                           "filename": filename, "body_sha256": digest,
                           "seqno": seqno, "runno": "1",
                           "receipt_complete_unix_ceil": clock,
                           "size_bytes": len(body),
                           "clock_health": {"wall_monotonic_consistent": True}})
        snapshot = {"origin_unix": harness.ORIGIN_UNIX,
                    "listing": tuple(listing),
                    "events": m1_asof.chain_events(events), "bodies": bodies,
                    "fetch_behaviours": behaviours}
        snapshot["snapshot_id"] = "snap_" + hashlib.sha256(json.dumps(
            {key: snapshot[key] for key in ("origin_unix", "listing", "events",
                                            "fetch_behaviours")},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
        return snapshot

    def test_oversized_body_refused_before_any_byte_is_read(self):
        cap = m1_budget.CAPS[m1_budget.BudgetClass.SOURCE_BYTES_ONE_BODY]
        body = b"\x00" * (cap + 1)
        snapshot = self._body_snapshot(
            [("BIG.zip", harness.NEW_SEQNO, harness.ORIGIN_UNIX - 600, body,
              "COMPLETE_VALID")])
        result = _run("B", snapshot)
        self.assertEqual(result.verdict, "PUBLISHED_BASE_FALLBACK")
        self.assertEqual(result.reason_code, "BUDGET_EXHAUSTED")
        self.assertEqual(
            result.budget_fold[m1_budget.BudgetClass.SOURCE_BYTES_TOTAL], 0)
        self.assertEqual(
            result.budget_fold[m1_budget.BudgetClass.SOURCE_REPLAY_CALLS], 1)

    def test_two_mid_size_bodies_do_not_accumulate_into_the_per_body_cap(self):
        cap = m1_budget.CAPS[m1_budget.BudgetClass.SOURCE_BYTES_ONE_BODY]
        size = cap // 2 + 1  # two of them exceed the cap when summed
        first = b"\x01" * size
        second = b"\x02" * size
        snapshot = self._body_snapshot([
            ("MID_OLD.zip", harness.OLD_SEQNO, harness.ORIGIN_UNIX - 3600, first,
             "COMPLETE_VALID"),
            ("MID_NEW.zip", harness.NEW_SEQNO, harness.ORIGIN_UNIX - 600, second,
             "COMPLETE_VALID")])
        result = _run("B", snapshot)
        self.assertEqual(result.verdict, "PUBLISHED_BASE_FALLBACK")
        self.assertEqual(result.reason_code, "SCHEMA_DRIFT")  # probe bytes, not ZIPs
        fold = result.budget_fold
        self.assertEqual(fold[m1_budget.BudgetClass.SOURCE_REPLAY_CALLS], 3)
        self.assertEqual(fold[m1_budget.BudgetClass.SOURCE_BYTES_TOTAL], size * 2)


class ArmSnapshotBoundaryTests(unittest.TestCase):
    """Audit-2: A never touches the snapshot; broken chains map to frozen reasons."""

    def setUp(self):
        _require_m1()
        self.tmp = tempfile.TemporaryDirectory()
        self.publish_root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _broken_snapshot(self) -> dict:
        snapshot = _clean_snapshot()
        broken = [dict(event) for event in snapshot["events"]]
        broken[0]["receipt_complete_unix_ceil"] += 1  # stale chain hashes now
        snapshot["events"] = tuple(broken)
        return snapshot

    def test_arm_a_never_touches_the_snapshot_receipt_chain(self):
        result = _run("A", self._broken_snapshot(),
                      publish_root=self.publish_root / "ev_a")
        self.assertEqual(result.verdict, "PUBLISHED_BASE_FALLBACK")
        self.assertEqual(result.reason_code, "AGENT_ABSTAIN")
        names = set(os.listdir(Path(result.publish_dir)))
        self.assertNotIn("asof_snapshot.json", names)

    def test_broken_chain_maps_to_a_frozen_reason_for_b_and_d(self):
        snapshot = self._broken_snapshot()
        b = _run("B", snapshot)
        self.assertEqual(b.verdict, "PUBLISHED_BASE_FALLBACK")
        self.assertEqual(b.reason_code, "SCHEMA_DRIFT")
        d = _run("D", snapshot,
                 transport=_d_transport(("LIST", "NONE"), ("FINALIZE_BASE", "NONE")))
        self.assertEqual(d.verdict, "PUBLISHED_BASE_FALLBACK")
        self.assertIn(d.reason_code, m1_contract.ORIGIN_REASON_CODES)


class TrustBoundaryTests(unittest.TestCase):
    """Audit-2 blockers 3/4: numeric injection, parser shell games, request binding."""

    def setUp(self):
        _require_m1()

    def test_forged_typed_evidence_cannot_attach_arbitrary_values(self):
        snapshot = _clean_snapshot()
        visible = m1_asof.materialize_visible_events(snapshot["events"],
                                                     harness.ORIGIN_UNIX)
        fetch = _fetch_records_for(snapshot, visible)
        evidence = m1_asof.derive_typed_evidence(harness.ORIGIN_UNIX, visible, fetch)
        forged = dataclasses.replace(evidence, values=(999999.0,) * 24)
        with self.assertRaises(m1_asof.BaseBoundaryError):
            m1_asof.typed_input(_base_input(), forged)
        honest = m1_asof.typed_input(_base_input(), evidence)
        from forecasting.agent_forecasting.predispatch_td_abs import COVARIATE_NAME
        self.assertEqual(tuple(honest.future_forecasts[COVARIATE_NAME]),
                         evidence.values)

    def test_function_envelope_cannot_override_the_function_identity(self):
        shell = (b'{"name":"m1_d_action","arguments":"{\\"__function__\\":null,'
                 b'\\"decision\\":\\"USE_TYPED\\",\\"evidence_ref\\":\\"NONE\\"}"}')
        with self.assertRaises(m1_llm.LlmParseError):
            m1_llm.parse_c_decision(_ok_response(shell),
                                    enabled_evidence_refs=())
        inner_named = (b'{"name":"m1_c_decision","arguments":"{\\"__function__\\":'
                       b'\\"m1_d_action\\",\\"action\\":\\"LIST\\",\\"ref\\":'
                       b'\\"NONE\\"}"}')
        with self.assertRaises(m1_llm.LlmParseError):
            m1_llm.parse_d_action(_ok_response(inner_named), enabled_refs=())
        plain = b'{"__function__":null,"action":"LIST","ref":"NONE"}'
        with self.assertRaises(m1_llm.LlmParseError):
            m1_llm.parse_d_action(_ok_response(plain), enabled_refs=())

    def test_request_binds_the_frozen_schema_and_output_cap_for_both_arms(self):
        config = m1_contract.load_config(ROOT)
        directory = ROOT / config["prompt_artifacts"]["directory"]
        for arm in ("C", "D"):
            request = json.loads(m1_llm.build_request_bytes(
                arm, '{"state":"FETCHED"}', ROOT))
            self.assertEqual(request["max_output_tokens"],
                             m1_contract.PER_CALL_OUTPUT_TOKENS[arm], msg=arm)
            schema_name = ("m1_c_decision_schema.json" if arm == "C"
                           else "m1_d_action_schema.json")
            output_format = request["text"]["format"]
            self.assertEqual(output_format["schema"],
                             json.loads((directory / schema_name).read_text()), msg=arm)
            self.assertTrue(output_format["strict"])
            self.assertEqual(output_format["type"], "json_schema")
            self.assertEqual(config["prompt_artifacts"]["sha256"][schema_name],
                             m1_contract.sha256_file(directory / schema_name),
                             msg=arm)

    def test_identity_recomputation_is_not_cached_by_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp)
            for relative in (*m1_contract.PINNED_RELATIVE_PATHS,
                              m1_contract.CONFIG_PATH):
                source = ROOT / relative
                target = tree / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(source, target)
            self.assertEqual(
                m1_contract.verify_run_identity(tree)["status"], "VERIFIED")
            victim = tree / "forecasting/agent_forecasting/m1_runtime.py"
            victim.write_bytes(victim.read_bytes() + b"\n# drift\n")
            with self.assertRaises(m1_contract.M1IdentityDrift):
                m1_contract.verify_run_identity(tree)

    def test_execution_modules_protocol_and_config_fields_are_pinned(self):
        config = m1_contract.load_config(ROOT)
        pins = config["execution_modules"]["files_sha256"]
        for required in ("forecasting/agent_forecasting/m1_runtime.py",
                         "forecasting/agent_forecasting/m1_budget.py",
                         "forecasting/agent_forecasting/m1_reducer.py",
                         config["protocol_doc"],
                         "tests/fixtures/agent_forecasting/m1_scenario_harness.py"):
            self.assertIn(required, pins)
        for relative, digest in pins.items():
            self.assertEqual(m1_contract.sha256_file(ROOT / relative), digest,
                             msg=relative)
        report = m1_contract.verify_run_identity(ROOT)
        self.assertEqual(report["status"], "VERIFIED")
        self.assertGreaterEqual(report["pinned_files"], len(pins))

    def test_budget_ledger_rebuild_detects_a_tampered_archive(self):
        result = _run("B", _clean_snapshot())
        tampered = [dict(entry) for entry in result.budget_entries]
        victim = next(entry for entry in tampered if entry["op"] == "SETTLE")
        victim["amount"] = 0  # silent discount attempt
        with self.assertRaises(ValueError):
            m1_budget.BudgetLedger.from_entries(tampered)


class PublicationSemanticClosureTests(unittest.TestCase):
    """Audit-2 blocker 5: publication verifies budget/deadline/identity semantics."""

    def setUp(self):
        _require_m1()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _real_archives(self):
        """Honest closure/payloads read back from one real published B run.

        audit-3/H4 fixture honesty: the semantic-closure tests must exercise a
        closure whose identities recompute from real archives, so the source of
        truth is a genuine run_arm publication, not fabricated hex digests.
        """
        if getattr(self, "_archives_cache", None) is None:
            publish_root = self.root / "ev_src"
            result = _run("B", _clean_snapshot(), publish_root=publish_root)
            self.assertTrue(result.published)
            final = Path(result.publish_dir)
            payloads = {name: (final / name).read_bytes()
                        for name in os.listdir(final)
                        if name not in ("manifest.json", "COMMIT.json")}
            manifest = json.loads((final / "manifest.json").read_text())
            self._archives_cache = (dict(manifest["closure"]), payloads)
        return self._archives_cache

    def _publish(self, run_id, closure_overrides=None, drop=(), payload_drop=()):
        closure, payloads = self._real_archives()
        closure = dict(closure)
        payloads = dict(payloads)
        closure.update(closure_overrides or {})
        for key in drop:
            del closure[key]
        for name in payload_drop:
            payloads.pop(name, None)
        return m1_publication.publish_run(self.root, run_id, payloads,
                                          identity={"contract_id": "probe"},
                                          closure=closure)

    def test_valid_closure_publishes_and_is_recorded_in_the_manifest(self):
        closure, _payloads = self._real_archives()
        final = self._publish("run_ok")
        manifest = json.loads((final / "manifest.json").read_text())
        self.assertEqual(manifest["closure"]["input_hash"],
                         closure["input_hash"])
        self.assertTrue(m1_publication.verify_run_closure(final)["status"] == "CLOSED")

    def test_fold_over_cap_refuses_publication(self):
        caps = {cls.name: value for cls, value in m1_budget.CAPS.items()}
        fold = dict(caps)
        fold["LLM_INPUT_TOKENS"] = caps["LLM_INPUT_TOKENS"] + 1
        with self.assertRaises(m1_publication.PublicationError):
            self._publish("run_fold", {"budget_fold": fold})

    def test_deadline_breach_refuses_publication(self):
        with self.assertRaises(m1_publication.PublicationError):
            self._publish("run_late", {
                "clock_final": harness.ORIGIN_UNIX + m1_contract.DEADLINE_OFFSET + 1})

    def test_broken_budget_chain_refuses_publication(self):
        with self.assertRaises(m1_publication.PublicationError):
            self._publish("run_chain", {"budget_chain_ok": False})

    def test_missing_closure_field_refuses_publication(self):
        with self.assertRaises(m1_publication.PublicationError):
            self._publish("run_missing", drop=("input_hash",))

    def test_run_arm_converts_publication_failure_into_a_failure_record(self):
        publish_root = self.root / "ev"
        original = m1_runtime.publish_run
        try:
            m1_runtime.publish_run = lambda *a, **k: (_ for _ in ()).throw(
                m1_publication.PublicationError("injected refusal"))
            result = _run("B", _clean_snapshot(), publish_root=publish_root)
        finally:
            m1_runtime.publish_run = original
        self.assertFalse(result.published)
        self.assertEqual(result.verdict, "TERMINAL_FAILED_NO_COMMIT")
        self.assertEqual(result.reason_code, "EVIDENCE_PUBLICATION_FAILED")
        self.assertIn("EVIDENCE_PUBLICATION_FAILED",
                      m1_contract.ORIGIN_REASON_CODES)
        self.assertTrue(list((publish_root / "failures").glob("*.json")))

    def test_published_runs_archive_llm_wire_and_digests(self):
        publish_root = self.root / "ev"
        dup = harness.opaque_ref(harness.NEW_FILENAME_B)
        result = _run("D", _asym_snapshot("NEW", "TIMEOUT"),
                      transport=_d_transport(("LIST", "NONE"),
                                             ("FETCH_REF", dup),
                                             ("FINALIZE_TYPED", "NONE")),
                      publish_root=publish_root)
        self.assertEqual(result.verdict, "PUBLISHED_TYPED")
        final = Path(result.publish_dir)
        wire = [json.loads(line) for line in
                (final / "llm_wire.jsonl").read_text().splitlines()]
        self.assertEqual(len(wire), result.llm_calls)
        for entry in wire:
            request = base64.b64decode(entry["request_b64"])
            self.assertEqual(hashlib.sha256(request).hexdigest(),
                             entry["request_sha256"])
            self.assertIn(b"max_output_tokens", request)
        digests = (final / "digests.jsonl").read_text().splitlines()
        self.assertEqual(len(digests), len(result.digests))

    def test_crash_cleanup_removes_staging_and_lock(self):
        import forecasting.agent_forecasting.m1_publication as module
        original = module._write_exclusive
        # warm the honest-archive cache first: the injected crash must hit the
        # TARGET publication's staging writes, not the fixture-producing run
        self._real_archives()
        counter = {"n": 0}

        def flaky(path, payload):
            counter["n"] += 1
            if counter["n"] == 2:
                raise OSError("injected crash mid staging")
            return original(path, payload)

        module._write_exclusive = flaky
        try:
            with self.assertRaises(OSError):
                self._publish("run_clean")
        finally:
            module._write_exclusive = original
        staging = self.root / "staging"
        locks = self.root / "locks"
        self.assertTrue(not staging.exists() or not any(staging.iterdir()))
        self.assertTrue(not locks.exists() or not any(locks.iterdir()))
        self.assertFalse((self.root / "runs" / "run_clean").exists())


class EvidenceClosureVerifierTests(unittest.TestCase):
    """Audit-2 blocker 6: the closure verifier checks provenance and manifest."""

    def setUp(self):
        _require_m1()
        sys.path.insert(0, str(ROOT / "scripts"))
        import verify_m1_evidence as verifier  # noqa: E402
        self.verifier = verifier
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "evidence"
        self.root.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_log(self, relative, *, ident, head, exit_code=0, tail=None):
        log = self.root / relative
        log.parent.mkdir(parents=True, exist_ok=True)
        body = (f"=== run_logged_m1 id={ident} ===\nargv: probe\ngit_head: {head}\n"
                f"branch: research/agentic-m1-bounded-actions-c2-20261029-r5\n"
                f"start_utc: t0\nend_utc: t1\n--- output begin ---\nprobe\n"
                f"--- output end ---\n")
        log.write_text(body + (tail if tail is not None
                               else f"EXIT_CODE={exit_code}\n"))
        return {"id": ident, "argv": ["probe"], "log_path": relative,
                "git_head": head, "start_utc": "t0", "end_utc": "t1",
                "exit_code": exit_code}

    def _write_ledger(self, *records):
        (self.root / "commands.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records))

    def _write_manifest(self):
        manifest = self.root / "artifact_manifest.json"
        files = {}
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path != manifest:
                files[str(path.relative_to(self.root))] = {
                    "sha256": m1_contract.sha256_file(path),
                    "size_bytes": path.stat().st_size}
        manifest.write_text(json.dumps(
            {"evidence_root": str(self.root), "files": files}, indent=2,
            sort_keys=True) + "\n")

    def test_clean_root_passes(self):
        record = self._write_log("red_green/g.txt", ident="g", head="H" * 40)
        self._write_ledger(record)
        self._write_manifest()
        self.assertEqual(self.verifier.verify_evidence_root(
            self.root, recompute=False), [])

    def test_detects_duplicate_log_path_and_missing_tail(self):
        first = self._write_log("a.txt", ident="a", head="H" * 40)
        second = self._write_log("a.txt", ident="b", head="H" * 40,
                                 tail="EXIT_CODE=3\nEXTRA\n")
        self._write_ledger(first, second)
        self._write_manifest()
        problems = self.verifier.verify_evidence_root(self.root, recompute=False)
        self.assertTrue(any("injective" in problem for problem in problems))
        self.assertTrue(any("final line" in problem for problem in problems))

    def test_detects_header_provenance_mismatch(self):
        record = self._write_log("a.txt", ident="a", head="H" * 40)
        record["git_head"] = "O" * 40  # ledger disagrees with the log header
        self._write_ledger(record)
        self._write_manifest()
        problems = self.verifier.verify_evidence_root(self.root, recompute=False)
        self.assertTrue(any("provenance" in problem for problem in problems))

    def test_detects_manifest_drift(self):
        record = self._write_log("a.txt", ident="a", head="H" * 40)
        self._write_ledger(record)
        self._write_manifest()
        (self.root / "smuggled.txt").write_bytes(b"x")
        problems = self.verifier.verify_evidence_root(self.root, recompute=False)
        self.assertTrue(any("smuggled" in problem for problem in problems))

    def test_detects_gate_inconsistent_with_the_reducer_report(self):
        record = self._write_log("red_green/02_green_full_suite.txt", ident="g",
                                 head="H" * 40, tail="81 passed in 2s\nEXIT_CODE=0\n")
        self._write_ledger(record)
        report = {"verdict": "FIXED_RULES_SUFFICIENT_AGENT_NOT_NEEDED",
                  "scenarios_total": 193, "d_traces_enumerated": 1700,
                  "llm_failures_checked": 1500, "witnesses": [],
                  "uses_labels_or_scores": False, "internal_consistency_ok": True}
        (self.root / "reducibility").mkdir()
        (self.root / "reducibility" / "reducer_report.json").write_text(
            json.dumps(report))
        gate = m1_contract.map_gate({"verdict": report["verdict"],
                                     "engineering_green": True})
        gate["engineering"] = {"suite_exit_code": 0,
                               "suite_log": "red_green/02_green_full_suite.txt",
                               "tests_passed": 81}
        gate["advance_m1"] = True  # the exact forbidden mutation
        (self.root / "gate.json").write_text(json.dumps(gate))
        self._write_manifest()
        problems = self.verifier.verify_evidence_root(self.root, recompute=False)
        self.assertTrue(any("advance_m1" in problem for problem in problems))


class GateMappingTests(unittest.TestCase):
    """Audit #19: engineering greenness never maps sufficiency to ADVANCE_M1."""

    def setUp(self):
        _require_m1()

    def test_finite_sufficiency_is_inconclusive_with_metrics_not_evaluated(self):
        gate = m1_contract.map_gate(
            {"verdict": "FIXED_RULES_SUFFICIENT_AGENT_NOT_NEEDED",
             "engineering_green": True})
        self.assertEqual(gate["scientific_decision"], "INCONCLUSIVE_REDUCIBILITY")
        self.assertIsNone(gate["stop_token"])
        self.assertEqual(gate["metrics_status"], "NOT_EVALUATED")
        self.assertFalse(gate["advance_m1"])
        self.assertFalse(gate["mission_goal_achieved"])

    def test_useful_witness_and_green_mean_engineering_acceptance_only(self):
        gate = m1_contract.map_gate({"verdict": "REDUCIBILITY_REFUTED",
                                     "witnesses": [{"terminal": {"verdict": "PUBLISHED_TYPED"}}],
                                     "engineering_green": True})
        self.assertEqual(gate["scientific_decision"],
                         "ENGINEERING_PASS_REDUCIBILITY_REFUTED")
        gate = m1_contract.map_gate({"verdict": "REDUCER_NON_VACUITY_FAILED",
                                     "engineering_green": True})
        self.assertEqual(gate["scientific_decision"], "BLOCKED")


class Audit3IdentityClosureTests(unittest.TestCase):
    """Read-only audit #2 / H1: identity must fail closed over the WHOLE config.

    A deleted execution pin, an extra pin, a mutated source contract, branch,
    arms declaration, deadline reserve, LLM cost, forecast-tool binding,
    evidence root or source-file list must each refuse the run BEFORE any
    action; the run entry must additionally bind git HEAD, a tracked-clean
    tree and the config file's own committed bytes.
    """

    def setUp(self):
        _require_m1()

    def _assert_config_mutation_refuses(self, mutate, label):
        config = m1_contract.load_config(ROOT)
        mutate(config)
        original = m1_contract.load_config
        m1_contract.load_config = lambda _root: config
        try:
            with self.assertRaises(m1_contract.M1IdentityDrift, msg=label):
                m1_contract.verify_run_identity(ROOT)
        finally:
            m1_contract.load_config = original

    def test_deleted_execution_pin_refuses_instead_of_verifying(self):
        def mutate(config):
            del config["execution_modules"]["files_sha256"][
                "forecasting/agent_forecasting/m1_runtime.py"]
        self._assert_config_mutation_refuses(
            mutate, "a deleted pin must be drift, not a silently unchecked file")

    def test_extra_execution_pin_refuses(self):
        def mutate(config):
            # a REAL existing file outside the frozen surface: the pin hash
            # itself is honest, only the keyset is smuggled
            config["execution_modules"]["files_sha256"][
                "tests/test_agent_m1_reducer.py"] = m1_contract.sha256_file(
                    ROOT / "tests/test_agent_m1_reducer.py")
        self._assert_config_mutation_refuses(
            mutate, "an extra pin is an undeclared execution-surface change")

    def test_source_contract_field_mutation_refuses(self):
        def mutate(config):
            config["source_contract"]["unit"] = "BANANAS"
        self._assert_config_mutation_refuses(
            mutate, "the frozen source contract (unit/region/field) is identity")

    def test_branch_and_provenance_mutation_refuses(self):
        def mutate(config):
            config["branch"] = "research/somebody-else"
            config["run_provenance_commit"] = "0" * 40
        self._assert_config_mutation_refuses(
            mutate, "run ancestry (branch/base/provenance) is identity")

    def test_arms_declaration_mutation_refuses(self):
        def mutate(config):
            config["arms"]["D"].append("FREE_FORM_PYTHON")
        self._assert_config_mutation_refuses(
            mutate, "the frozen arms declaration is identity")

    def test_deadline_and_llm_cost_and_tool_binding_mutations_refuse(self):
        def mutate(config):
            config["deadline"]["http_timeout_seconds"] = 999
            config["llm_cost"]["rate_input_usd_per_million"] = 0.0
            config["forecast_tool"]["calls_max"] = 99
        self._assert_config_mutation_refuses(
            mutate, "deadline/llm_cost/forecast_tool bindings are identity")

    def test_evidence_root_and_source_files_mutations_refuse(self):
        def mutate(config):
            config["evidence_root"] = "logs/elsewhere"
            config["source_files"].append("smuggled_file.py")
        self._assert_config_mutation_refuses(
            mutate, "the evidence root and the frozen source-file list are identity")

    def test_run_entry_binds_git_head_tree_and_config_bytes(self):
        entry = getattr(m1_contract, "verify_run_entry", None)
        self.assertIsNotNone(
            entry, "audit-3/H1: the run entry must bind git HEAD, a "
            "tracked-clean tree and the config file's committed bytes")
        report = entry(ROOT)
        self.assertEqual(report["status"], "VERIFIED")
        self.assertTrue(report["tracked_tree_clean"])
        self.assertTrue(report["git_head"])
        with self.assertRaises(m1_contract.M1IdentityDrift):
            entry(ROOT, expected_head="0" * 40)

    def test_run_entry_refuses_a_dirty_tracked_tree(self):
        entry = getattr(m1_contract, "verify_run_entry", None)
        self.assertIsNotNone(
            entry, "audit-3/H1: the run entry must bind git HEAD, a "
            "tracked-clean tree and the config file's committed bytes")
        victim = ROOT / "tests" / "test_agent_m1_reducer.py"
        original_bytes = victim.read_bytes()
        victim.write_bytes(original_bytes + b"\n# transient audit-3 probe\n")
        try:
            with self.assertRaises(m1_contract.M1IdentityDrift):
                entry(ROOT)
        finally:
            victim.write_bytes(original_bytes)

    def test_run_entry_refuses_edited_config_bytes(self):
        entry = getattr(m1_contract, "verify_run_entry", None)
        self.assertIsNotNone(
            entry, "audit-3/H1: the run entry must bind git HEAD, a "
            "tracked-clean tree and the config file's committed bytes")
        victim = ROOT / m1_contract.CONFIG_PATH
        original_bytes = victim.read_bytes()
        victim.write_bytes(original_bytes + b"\n")  # same JSON, different bytes
        try:
            with self.assertRaises(m1_contract.M1IdentityDrift):
                entry(ROOT)
        finally:
            victim.write_bytes(original_bytes)


class Audit3AsOfBoundaryTests(unittest.TestCase):
    """Read-only audit #2 / H2: A isolation and typed provenance closure."""

    def setUp(self):
        _require_m1()

    def test_arm_a_completes_when_receipt_structure_is_corrupt(self):
        snapshot = _clean_snapshot()
        stripped = tuple(
            {key: value for key, value in event.items() if key != "ref"}
            for event in snapshot["events"])
        snapshot["events"] = stripped
        try:
            result = _run("A", snapshot)
        except Exception as error:  # noqa: BLE001 - the failure IS the finding
            self.fail(f"arm A must not read the receipt chain: {error!r}")
        self.assertEqual(result.verdict, "PUBLISHED_BASE_FALLBACK")
        self.assertEqual(result.reason_code, "AGENT_ABSTAIN")

    def test_derive_typed_evidence_rejects_unchained_caller_events(self):
        snapshot = _two_ref_snapshot()
        visible = m1_asof.materialize_visible_events(snapshot["events"],
                                                     harness.ORIGIN_UNIX)
        stripped = tuple(
            {key: value for key, value in event.items()
             if key not in ("chain_seq", "prev_sha256", "event_sha256")}
            for event in visible)
        fetch = _fetch_records_for(snapshot, visible)
        with self.assertRaises(m1_asof.M1TypedRejection) as ctx:
            m1_asof.derive_typed_evidence(harness.ORIGIN_UNIX, stripped, fetch)
        self.assertEqual(ctx.exception.reason_code, "SCHEMA_DRIFT")
        # the re-derivation boundary refuses the same forgery, never escapes
        forged = m1_asof.derive_typed_evidence(
            harness.ORIGIN_UNIX, visible, fetch)
        poisoned = dataclasses.replace(forged, derivation_events=stripped)
        with self.assertRaises(m1_asof.BaseBoundaryError):
            m1_asof.verify_typed_evidence(poisoned)

    def test_post_origin_corruption_preserves_earlier_origin(self):
        base = _two_ref_snapshot()
        events = [dict(event) for event in base["events"]]
        extra = dict(events[-1])
        extra.update({"ref": harness.opaque_ref("POST_ORIGIN_CORRUPT.zip"),
                      "filename": "POST_ORIGIN_CORRUPT.zip",
                      "receipt_complete_unix_ceil": harness.ORIGIN_UNIX + 600,
                      "chain_seq": len(events),
                      "prev_sha256": events[-1]["event_sha256"],
                      "event_sha256": "0" * 64})  # corrupt: never recomputes
        chained = tuple(events) + (extra,)
        try:
            visible_late = m1_asof.materialize_visible_events(
                chained, harness.ORIGIN_UNIX)
        except ValueError as error:
            self.fail("a corrupt post-origin receipt must not fail an earlier "
                      f"origin: {error}")
        visible_base = m1_asof.materialize_visible_events(
            tuple(events), harness.ORIGIN_UNIX)
        self.assertEqual(visible_late, visible_base)
        self.assertEqual(m1_asof.visible_ledger_hash(visible_late),
                         m1_asof.visible_ledger_hash(visible_base))
        # at the later origin the corruption IS visible and must still fail
        with self.assertRaises(ValueError):
            m1_asof.materialize_visible_events(chained,
                                               harness.ORIGIN_UNIX + 1200)


class Audit3FailureMappingTests(unittest.TestCase):
    """Read-only audit #2 / H3: every adapter/transport exception is frozen."""

    def setUp(self):
        _require_m1()

    class ExplodingTransport:
        identity = {"kind": "exploding_probe_transport", "version": 1}

        def __init__(self):
            self.calls = 0

        def generate(self, arm, turn, request_bytes):
            self.calls += 1
            raise RuntimeError("adapter exploded")

    def test_c_transport_internal_failure_maps_to_a_frozen_reason(self):
        transport = self.ExplodingTransport()
        try:
            result = _run("C", _clean_snapshot(), transport=transport)
        except Exception as error:  # noqa: BLE001 - the escape IS the finding
            self.fail(f"a transport-internal failure must map to a frozen "
                      f"reason, never escape: {error!r}")
        self.assertEqual(transport.calls, 1)
        self.assertEqual(result.verdict, "PUBLISHED_BASE_FALLBACK")
        self.assertEqual(result.reason_code, "LLM_TIMEOUT")
        self.assertIn("LLM_TIMEOUT", m1_contract.ORIGIN_REASON_CODES)

    def test_d_transport_internal_failure_maps_to_a_frozen_reason(self):
        transport = self.ExplodingTransport()
        try:
            result = _run("D", _clean_snapshot(), transport=transport)
        except Exception as error:  # noqa: BLE001 - the escape IS the finding
            self.fail(f"a transport-internal failure must map to a frozen "
                      f"reason, never escape: {error!r}")
        self.assertEqual(transport.calls, 1)
        self.assertEqual(result.verdict, "PUBLISHED_BASE_FALLBACK")
        self.assertEqual(result.reason_code, "LLM_TIMEOUT")

    def test_missing_fetch_behaviour_maps_to_a_frozen_reason(self):
        snapshot = _clean_snapshot()
        victim = snapshot["listing"][0]["ref"]
        del snapshot["fetch_behaviours"][victim]
        try:
            result = _run("B", snapshot)
        except Exception as error:  # noqa: BLE001 - the escape IS the finding
            self.fail(f"a replay adapter that omits a declared ref's behaviour "
                      f"must map to a frozen reason, never escape: {error!r}")
        self.assertEqual(result.verdict, "PUBLISHED_BASE_FALLBACK")
        self.assertEqual(result.reason_code, "SCHEMA_DRIFT")


class Audit3PublicationIdentityTests(unittest.TestCase):
    """Read-only audit #2 / H4: publication binds FROZEN caps and recomputes
    tool/input identity from the archived model ledger and forecast trace."""

    def setUp(self):
        _require_m1()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _real_archives(self):
        publish_root = self.root / "ev_src"
        result = _run("B", _clean_snapshot(), publish_root=publish_root)
        self.assertTrue(result.published)
        final = Path(result.publish_dir)
        payloads = {name: (final / name).read_bytes()
                    for name in os.listdir(final)
                    if name not in ("manifest.json", "COMMIT.json")}
        manifest = json.loads((final / "manifest.json").read_text())
        return dict(manifest["closure"]), payloads

    def test_honest_archives_verify(self):
        closure, payloads = self._real_archives()
        report = m1_publication.verify_publication_closure(closure, payloads)
        self.assertEqual(report["budget_fold"]["LLM_INPUT_TOKENS"],
                         closure["budget_fold"]["LLM_INPUT_TOKENS"])

    def test_inflated_self_reported_caps_refuse_publication(self):
        closure, payloads = self._real_archives()
        caps = dict(closure["budget_caps"])
        caps["LLM_INPUT_TOKENS"] = caps["LLM_INPUT_TOKENS"] + 1
        closure["budget_caps"] = caps
        with self.assertRaises(m1_publication.PublicationError):
            m1_publication.verify_publication_closure(closure, payloads)

    def test_forged_tool_identity_refuses_publication(self):
        closure, payloads = self._real_archives()
        closure["tool_identity_sha256"] = "f" * 64
        with self.assertRaises(m1_publication.PublicationError):
            m1_publication.verify_publication_closure(closure, payloads)

    def test_forged_canonical_input_identity_refuses_publication(self):
        closure, payloads = self._real_archives()
        closure["canonical_model_input_sha256"] = "e" * 64
        with self.assertRaises(m1_publication.PublicationError):
            m1_publication.verify_publication_closure(closure, payloads)

    def test_missing_model_ledger_refuses_publication(self):
        closure, payloads = self._real_archives()
        del payloads["model_ledger.jsonl"]
        with self.assertRaises(m1_publication.PublicationError):
            m1_publication.verify_publication_closure(closure, payloads)

    def test_forecast_bytes_must_rebind_to_exactly_one_trace_entry(self):
        closure, payloads = self._real_archives()
        payloads["forecast.bin"] = payloads["forecast.bin"] + b"x"
        with self.assertRaises(m1_publication.PublicationError):
            m1_publication.verify_publication_closure(closure, payloads)


if __name__ == "__main__":
    unittest.main()
