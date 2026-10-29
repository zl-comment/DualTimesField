"""Label-free exhaustive reducibility gate for the M1 contract.

Enumerates the declared finite scenario axes, every legal D trace of at most four
valid proposals under the frozen state machine and envelope, and the LLM failure
modes, then compares each terminal (verdict, input_hash, forecast-tool trace hash)
with the maximal deterministic B plan, the one-shot maximal/abstaining C policies
and the common A fallback on the SAME immutable snapshot.

If every useful D terminal is reproduced by B/C the verdict is
FIXED_RULES_SUFFICIENT_AGENT_NOT_NEEDED.  A seeded weak-B control that wastes its
body GET on a superseded ref must be classified non-reducible, otherwise the
reducer itself is invalid (REDUCER_NON_VACUITY_FAILED).  No labels, no scores.
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import m1_actions
from .contracts import FrozenForecastInput
from .m1_actions import M1ActionKind, M1State
from .m1_runtime import M1SourceSnapshot, ScriptedTransport, run_arm

# the synthetic scenario harness is test-fixture data shared with the suite
_FIXTURES = str(Path(__file__).resolve().parents[2] / "tests" / "fixtures"
                / "agent_forecasting")
if _FIXTURES not in sys.path:  # pragma: no cover - idempotent path setup
    sys.path.insert(0, _FIXTURES)
import m1_scenario_harness as harness  # type: ignore

VERDICT_SUFFICIENT = "FIXED_RULES_SUFFICIENT_AGENT_NOT_NEEDED"
VERDICT_REFUTED = "REDUCIBILITY_REFUTED"
VERDICT_INVALID = "REDUCER_NON_VACUITY_FAILED"


def _base_input() -> FrozenForecastInput:
    return FrozenForecastInput(**harness.synthetic_base_input_payload())


def _metric(result) -> dict:
    return {"verdict": result.verdict, "input_hash": result.input_hash,
            "tool_trace_hash": result.tool_trace_hash,
            "reason_code": result.reason_code}


def _key(metric: dict) -> tuple:
    return (metric["verdict"], metric["input_hash"], metric["tool_trace_hash"])


def _llm_response(payload: bytes):
    from . import m1_contract, m1_llm
    return m1_llm.M1LlmResponse(status="COMPLETED", model=m1_contract.LLM_MODEL,
                                output_raw=payload)


def d_legal_traces(snapshot_data: dict) -> list[tuple[tuple[str, str], ...]]:
    """Every legal D action sequence of at most four valid proposals."""
    snapshot = M1SourceSnapshot.from_mapping(snapshot_data)
    listed = tuple(entry["ref"] for entry in snapshot.listing)
    traces: list[tuple[tuple[str, str], ...]] = []

    def walk(prefix: list[tuple[str, str]], list_calls: int, get_calls: int,
             fetched: tuple[str, ...]) -> None:
        if len(prefix) == 4:
            traces.append(tuple(prefix))
            return
        state = (M1State.A_READY if not list_calls else
                 M1State.FETCHED if fetched else M1State.LISTED)
        available = (tuple(ref for ref in listed if ref not in fetched)
                     if get_calls < 2 else ())
        enabled = m1_actions.enabled_actions(
            state, list_calls=list_calls, get_calls=get_calls,
            available_refs=available, fetched_refs=fetched)
        if not enabled:
            traces.append(tuple(prefix))
            return
        for kind in enabled:
            if kind is M1ActionKind.FETCH_REF:
                pairs = [(kind.name, ref) for ref in available]
            else:
                pairs = [(kind.name, "NONE")]
            for pair in pairs:
                extended = prefix + [pair]
                if kind in (M1ActionKind.FINALIZE_TYPED, M1ActionKind.FINALIZE_BASE):
                    traces.append(tuple(extended))
                    continue
                walk(extended,
                     list_calls + (1 if kind is M1ActionKind.LIST else 0),
                     get_calls + (1 if kind is M1ActionKind.FETCH_REF else 0),
                     fetched + ((pair[1],) if kind is M1ActionKind.FETCH_REF else ()))

    walk([], 0, 0, ())
    return traces


def _canonical_d_trace(snapshot_data: dict) -> tuple[tuple[str, str], ...]:
    """The maximal D policy: LIST, fetch newest-visible refs, finalize typed."""
    snapshot = M1SourceSnapshot.from_mapping(snapshot_data)
    origin = snapshot.origin_unix
    trace: list[tuple[str, str]] = [("LIST", "NONE")]
    fetched: list[str] = []
    for ref in snapshot.canonical_fetch_refs(origin):
        trace.append(("FETCH_REF", ref))
        fetched.append(ref)
    trace.append(("FINALIZE_TYPED", "NONE"))
    transport = ScriptedTransport([
        _llm_response(harness.llm_valid_d_action_bytes(action, ref))
        for action, ref in trace])
    result = run_arm("D", snapshot_data, base_input=_base_input(),
                     transport=transport)
    if result.verdict == "PUBLISHED_TYPED":
        return tuple(trace)
    return tuple(trace[:-1]) + (("FINALIZE_BASE", "NONE"),)


def terminals_for_scenario(snapshot_data: dict) -> dict:
    """Terminal metrics for A, maximal B, maximal C and the canonical maximal D."""
    a = _metric(run_arm("A", snapshot_data, base_input=_base_input()))
    b = _metric(run_arm("B", snapshot_data, base_input=_base_input()))
    c = _metric(run_arm("C", snapshot_data, base_input=_base_input(),
                        transport=ScriptedTransport([
                            _llm_response(harness.llm_valid_c_decision_bytes(
                                "USE_TYPED", "NONE"))])))
    d = _metric(run_arm("D", snapshot_data, base_input=_base_input(),
                        transport=ScriptedTransport([
                            _llm_response(harness.llm_valid_d_action_bytes(action, ref))
                            for action, ref in _canonical_d_trace(snapshot_data)])))
    return {"A": a, "B": b, "C": c, "D": d}


def _weak_b_metric(snapshot_data: dict) -> dict:
    """Seeded weak-B: spends its body GETs only on the superseded (oldest) ref."""
    snapshot = M1SourceSnapshot.from_mapping(snapshot_data)
    origin = snapshot.origin_unix
    visible_refs = []
    for record in snapshot.visible_events(origin):
        if record["ref"] not in visible_refs:
            visible_refs.append(record["ref"])
    ordered = sorted(visible_refs, key=lambda ref: (
        (snapshot.event_for(ref) or {}).get("seqno", ""),
        int((snapshot.event_for(ref) or {}).get("runno", "0") or 0)))
    oldest = ordered[0] if ordered else None
    trace = [("LIST", "NONE")] + (
        [("FETCH_REF", oldest)] if oldest else []) + [("FINALIZE_TYPED", "NONE")]
    transport = ScriptedTransport([
        _llm_response(harness.llm_valid_d_action_bytes(action, ref))
        for action, ref in trace])
    return _metric(run_arm("D", snapshot_data, base_input=_base_input(),
                           transport=transport))


def _c_llm_failure_metric(snapshot_data: dict, kind: str) -> dict:
    from . import m1_contract, m1_llm
    if kind == "VALID_USE_BASE":
        response = _llm_response(
            harness.llm_valid_c_decision_bytes("USE_BASE", "NONE"))
    elif kind == "INVALID_ACTION":
        response = _llm_response(b'{"decision":"RE_QUERY","evidence_ref":"NONE"}')
    else:
        response = m1_llm.M1LlmResponse(
            status={"REFUSAL": "REFUSED", "INCOMPLETE": "INCOMPLETE",
                    "TIMEOUT": "TIMEOUT"}[kind],
            model=m1_contract.LLM_MODEL, output_raw=b"")
    return _metric(run_arm("C", snapshot_data, base_input=_base_input(),
                           transport=ScriptedTransport([response])))


def _d_failure_metric(snapshot_data: dict, kind: str) -> dict:
    from . import m1_contract, m1_llm
    trace = list(_canonical_d_trace(snapshot_data))
    if kind == "INVALID_ACTION":
        response = _llm_response(b'{"action":"RUN_PYTHON","ref":"NONE"}')
    else:
        response = m1_llm.M1LlmResponse(
            status={"REFUSAL": "REFUSED", "INCOMPLETE": "INCOMPLETE",
                    "TIMEOUT": "TIMEOUT"}[kind],
            model=m1_contract.LLM_MODEL, output_raw=b"")
    # failure injected at the second proposal, after one valid action
    transport = ScriptedTransport([
        _llm_response(harness.llm_valid_d_action_bytes(*trace[0])), response])
    return _metric(run_arm("D", snapshot_data, base_input=_base_input(),
                           transport=transport))


def run_reducer(repo_root=None, scenarios=None) -> dict:
    """Execute the full label-free reducibility gate over the declared axes."""
    scenario_list = scenarios if scenarios is not None else harness.scenario_space()
    witnesses: list[dict] = []
    consistency_ok = True
    weak_b_diff_scenarios = 0
    d_traces_enumerated = 0
    llm_failures_checked = 0
    for snapshot in scenario_list:
        label = snapshot["scenario"]
        a = _metric(run_arm("A", snapshot, base_input=_base_input()))
        b = _metric(run_arm("B", snapshot, base_input=_base_input()))
        c_typed = _metric(run_arm("C", snapshot, base_input=_base_input(),
                                  transport=ScriptedTransport([
                                      _llm_response(
                                          harness.llm_valid_c_decision_bytes(
                                              "USE_TYPED", "NONE"))])))
        c_base = _metric(run_arm("C", snapshot, base_input=_base_input(),
                                 transport=ScriptedTransport([
                                     _llm_response(
                                         harness.llm_valid_c_decision_bytes(
                                             "USE_BASE", "NONE"))])))
        if _key(c_typed) != _key(b):
            consistency_ok = False
        if c_base["verdict"] != "PUBLISHED_BASE_FALLBACK":
            consistency_ok = False
        fixed_keys = {_key(b), _key(a), _key(c_typed), _key(c_base)}
        weak = _weak_b_metric(snapshot)
        weak_keys = {_key(weak), _key(a)}

        useful_d_keys = set()
        for trace in d_legal_traces(snapshot):
            d_traces_enumerated += 1
            result = run_arm("D", snapshot, base_input=_base_input(),
                             transport=ScriptedTransport([
                                 _llm_response(
                                     harness.llm_valid_d_action_bytes(action, ref))
                                 for action, ref in trace]))
            if result.verdict == "PUBLISHED_TYPED":
                useful_d_keys.add(_key(_metric(result)))
                if _key(_metric(result)) not in fixed_keys:
                    witnesses.append({"scenario": label, "trace": [list(step) for
                                                                   step in trace],
                                      "terminal": _metric(result)})
            elif _key(_metric(result)) not in fixed_keys:
                # a non-useful terminal outside the fixed set still needs explaining
                witnesses.append({"scenario": label, "trace": [list(step) for
                                                               step in trace],
                                  "terminal": _metric(result), "useful": False})
        if useful_d_keys - weak_keys:
            weak_b_diff_scenarios += 1

        for kind in ("VALID_USE_BASE", "INVALID_ACTION", "REFUSAL", "INCOMPLETE",
                     "TIMEOUT"):
            failure = _c_llm_failure_metric(snapshot, kind)
            llm_failures_checked += 1
            if _key(failure) not in fixed_keys:
                witnesses.append({"scenario": label, "llm": kind, "arm": "C",
                                  "terminal": failure})
        for kind in ("INVALID_ACTION", "REFUSAL", "INCOMPLETE", "TIMEOUT"):
            failure = _d_failure_metric(snapshot, kind)
            llm_failures_checked += 1
            if _key(failure) not in fixed_keys:
                witnesses.append({"scenario": label, "llm": kind, "arm": "D",
                                  "terminal": failure})

    if not consistency_ok:
        verdict = VERDICT_INVALID
    elif witnesses:
        verdict = VERDICT_REFUTED
    elif weak_b_diff_scenarios > 0:
        verdict = VERDICT_SUFFICIENT
    else:
        verdict = VERDICT_INVALID
    return {
        "contract": "m1.agent_contract.c2.v1",
        "verdict": verdict,
        "uses_labels_or_scores": False,
        "scenarios_total": len(scenario_list),
        "d_traces_enumerated": d_traces_enumerated,
        "llm_failures_checked": llm_failures_checked,
        "witnesses": witnesses,
        "internal_consistency_ok": consistency_ok,
        "non_vacuity_control": {
            "policy": "weak_B_fetches_only_the_superseded_ref",
            "weak_b_diff_scenarios": weak_b_diff_scenarios,
            "detected": weak_b_diff_scenarios > 0,
        },
    }
