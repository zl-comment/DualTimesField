"""Arm deciders: B fixed rules, C one-shot LLM, D bounded allowlisted agent."""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Mapping

import numpy as np

from . import data as eval_data
from . import llm_transport, protocol
from .protocol import ACTIONS, DecisionParseError

Decision = dict  # {action, exit_code, failure_reason, latency_ms, llm_calls, tool_calls, tokens, detail}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "history_window",
        "description": "Last N visible hourly RRP prices (AUD/MWh) before the origin.",
        "parameters": {"type": "object", "properties": {
            "hours": {"type": "integer", "enum": [24, 48, 72]}},
            "required": ["hours"]}}},
    {"type": "function", "function": {
        "name": "demand_window",
        "description": "Last N visible hourly total demand values (MW) before the origin.",
        "parameters": {"type": "object", "properties": {
            "hours": {"type": "integer", "enum": [24, 48, 72]}},
            "required": ["hours"]}}},
    {"type": "function", "function": {
        "name": "hourly_profile",
        "description": "Mean RRP by hour-of-day over the last N days of visible history.",
        "parameters": {"type": "object", "properties": {
            "days_back": {"type": "integer", "enum": [7, 14]}},
            "required": ["days_back"]}}},
    {"type": "function", "function": {
        "name": "base_quantiles",
        "description": "The frozen base numeric 24h x 5-quantile forecast the action applies to.",
        "parameters": {"type": "object", "properties": {}}}},
]

# r8 predeclared change: the existing final action JSON structure from the frozen
# system prompt ({"action": <ACTIONS>, "reason": <one short sentence>}) bound to
# the provider request as a structured-output constraint for the decision-only
# turn. The action set itself is unchanged.
FINAL_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "final_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(ACTIONS)},
                "reason": {"type": "string"},
            },
            "required": ["action", "reason"],
            "additionalProperties": False,
        },
    },
}

FINAL_TURN_INSTRUCTION = (
    "Your investigation is over and no further tool calls are allowed. Reply now "
    "with your final decision: exactly one JSON object "
    '{"action": "<KEEP|SHIFT_UP|SHIFT_DOWN|WIDEN|NARROW>", '
    '"reason": "<one short sentence>"} and nothing else.'
)


def _ok(action: str, latency_ms: float, llm_calls: int = 0, tool_calls: int = 0,
        tokens: dict | None = None, detail: str = "") -> Decision:
    return {"action": action, "exit_code": 0, "failure_reason": "",
            "latency_ms": round(latency_ms, 1), "llm_calls": llm_calls,
            "tool_calls": tool_calls,
            "tokens": tokens or {"prompt": 0, "completion": 0}, "detail": detail}


def _fail(reason: str, latency_ms: float, llm_calls: int = 0, tool_calls: int = 0,
          tokens: dict | None = None, detail: str = "") -> Decision:
    return {"action": "", "exit_code": 1, "failure_reason": reason,
            "latency_ms": round(latency_ms, 1), "llm_calls": llm_calls,
            "tool_calls": tool_calls,
            "tokens": tokens or {"prompt": 0, "completion": 0}, "detail": detail[:200]}


def _action_menu() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in protocol.ACTION_DEFINITIONS.items())


def decide_b(record: Mapping[str, Any], base_quantiles: np.ndarray) -> Decision:
    start = time.perf_counter()
    action = protocol.rule_b(record["history_prices"])
    return _ok(action, (time.perf_counter() - start) * 1000.0, detail="fixed_rule_b")


def _system_prompt(agent: bool) -> str:
    text = (
        "You are a frozen decision module in a 2022 Australian electricity price "
        "forecasting experiment. A frozen numeric model produced a 24-hour-ahead "
        "five-quantile RRP forecast for one region from the last 72 visible hourly "
        "prices. Your only job is to choose ONE action applied to that base forecast:\n"
        f"{_action_menu()}\n"
        "You see one evidence digest computed strictly from data available at the "
        "origin; you never see the labels. There is no single correct answer; choose "
        "the action that best corrects the base forecast given the evidence."
    )
    if agent:
        text += (
            "\nYou may first investigate the as-of-visible history with the provided "
            f"tools: at most {protocol.D_MAX_TOOL_CALLS} tool calls across at most "
            f"{protocol.D_MAX_TURNS} turns. Exceeding the budget or calling a tool "
            "outside the allowlist fails the cell. After your investigation (or "
            "immediately if the digest is enough) you MUST answer with the final JSON."
        )
    text += (
        "\nYour final reply must be exactly one JSON object with no extra text:\n"
        '{"action": "<KEEP|SHIFT_UP|SHIFT_DOWN|WIDEN|NARROW>", "reason": "<one short sentence>"}'
    )
    return text


def decide_c(record: Mapping[str, Any], base_quantiles: np.ndarray,
             transport: Callable = llm_transport.chat) -> Decision:
    digest = protocol.evidence_digest(record, base_quantiles)
    messages = [{"role": "system", "content": _system_prompt(agent=False)},
                {"role": "user", "content": json.dumps(digest, sort_keys=True)}]
    start = time.perf_counter()
    try:
        response = transport(messages, max_tokens=protocol.C_MAX_TOKENS, tag="C")
    except llm_transport.TransportError as exc:
        return _fail(exc.reason_code, (time.perf_counter() - start) * 1000.0,
                     llm_calls=1, detail=str(exc))
    latency = (time.perf_counter() - start) * 1000.0
    tokens = {"prompt": response["usage"]["prompt_tokens"],
              "completion": response["usage"]["completion_tokens"]}
    try:
        action = protocol.parse_decision(response["content"])
    except DecisionParseError as exc:
        return _fail(exc.reason_code, latency, llm_calls=1, tokens=tokens,
                     detail=f"raw={response['content'][:120]!r}")
    return _ok(action, latency, llm_calls=1, tokens=tokens, detail=response["content"][:200])


def _make_tool_executor(record: Mapping[str, Any], base_quantiles: np.ndarray,
                        frame) -> tuple[Callable, dict]:
    """Allowlisted read-only tools over as-of-visible data only."""
    def run_tool(name: str, arguments) -> str:
        args = arguments if isinstance(arguments, dict) else {}
        if name == "history_window":
            hours = args.get("hours")
            if hours not in (24, 48, 72):
                return json.dumps({"error": "hours must be 24, 48 or 72"})
            prices, _, _ = eval_data.visible_slice(frame, record["origin_local"], hours)
            return json.dumps({"prices": [round(float(v), 2) for v in prices]})
        if name == "demand_window":
            hours = args.get("hours")
            if hours not in (24, 48, 72):
                return json.dumps({"error": "hours must be 24, 48 or 72"})
            _, demand, _ = eval_data.visible_slice(frame, record["origin_local"], hours)
            return json.dumps({"demand": [round(float(v), 1) for v in demand]})
        if name == "hourly_profile":
            days = args.get("days_back")
            if days not in (7, 14):
                return json.dumps({"error": "days_back must be 7 or 14"})
            prices, _, hours_of_day = eval_data.visible_slice(
                frame, record["origin_local"], days * 24)
            profile = {str(h): round(float(np.mean(prices[hours_of_day == h])), 2)
                       for h in range(24)}
            return json.dumps({"hourly_mean_rrp": profile})
        if name == "base_quantiles":
            q = np.sort(np.asarray(base_quantiles, dtype=np.float64), axis=-1)
            return json.dumps({"quantile_levels": list(protocol.QUANTILE_LEVELS),
                               "forecast": [[round(float(v), 2) for v in row] for row in q]})
        return json.dumps({"error": f"tool {name!r} is not allowlisted"})

    stats = {"tool_calls": 0}
    def bounded(name: str, arguments) -> str:
        stats["tool_calls"] += 1
        return run_tool(name, arguments)
    return bounded, stats


def decide_d(record: Mapping[str, Any], base_quantiles: np.ndarray, frame,
             transport: Callable = llm_transport.chat) -> Decision:
    """Bounded agent: <=3 turns, <=4 allowlisted tool calls, then strict JSON.

    r8 predeclared change: when the adaptive phase ends without a legal final
    decision, exactly one decision-only LLM turn runs with all tool calls
    forbidden and FINAL_RESPONSE_FORMAT bound to the request; an exhausted
    tool budget does not block entry. The original per-call token limit, LLM
    timeout and transport-failure semantics are retained."""
    executor, stats = _make_tool_executor(record, base_quantiles, frame)
    digest = protocol.evidence_digest(record, base_quantiles)
    messages = [{"role": "system", "content": _system_prompt(agent=True)},
                {"role": "user", "content": json.dumps(digest, sort_keys=True)}]
    tokens = {"prompt": 0, "completion": 0}
    llm_calls = 0
    start = time.perf_counter()
    phase_reason = ""
    for turn in range(1, protocol.D_MAX_TURNS + 1):
        try:
            response = transport(messages, tools=TOOL_SCHEMAS,
                                 max_tokens=protocol.D_MAX_TOKENS, tag=f"D{turn}")
        except llm_transport.TransportError as exc:
            return _fail(exc.reason_code, (time.perf_counter() - start) * 1000.0,
                         llm_calls=llm_calls, tool_calls=stats["tool_calls"],
                         tokens=tokens, detail=str(exc))
        llm_calls += 1
        tokens["prompt"] += response["usage"]["prompt_tokens"]
        tokens["completion"] += response["usage"]["completion_tokens"]
        calls = response["tool_calls"]
        if not calls:
            try:
                action = protocol.parse_decision(response["content"])
            except DecisionParseError as exc:
                phase_reason = exc.reason_code
                break
            return _ok(action, (time.perf_counter() - start) * 1000.0,
                       llm_calls=llm_calls, tool_calls=stats["tool_calls"],
                       tokens=tokens, detail=response["content"][:200])
        if turn == protocol.D_MAX_TURNS:
            phase_reason = "BUDGET_EXHAUSTED_NO_DECISION"
            break
        messages.append({"role": "assistant", "content": response["content"] or None,
                         "tool_calls": [
                             {"id": c["id"], "type": "function",
                              "function": {"name": c["name"],
                                           "arguments": json.dumps(c["arguments"] or {})}}
                             for c in calls]})
        phase_start = len(messages)
        aborted = False
        for call in calls:
            if stats["tool_calls"] >= protocol.D_MAX_TOOL_CALLS:
                phase_reason = "BUDGET_EXHAUSTED_TOO_MANY_TOOL_CALLS"
                aborted = True
                break
            if call["name"] not in ("history_window", "demand_window", "hourly_profile",
                                    "base_quantiles"):
                phase_reason = "TOOL_NOT_ALLOWLISTED"
                aborted = True
                break
            result = executor(call["name"], call["arguments"])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        if aborted:
            del messages[phase_start - 1:]  # drop the assistant turn with unanswerable calls
            break

    # Final decision-only turn: no tools offered, final action JSON schema bound
    # to the request; entry is not blocked by the exhausted tool budget.
    messages.append({"role": "user", "content": FINAL_TURN_INSTRUCTION})
    try:
        response = transport(messages, response_format=FINAL_RESPONSE_FORMAT,
                             max_tokens=protocol.D_MAX_TOKENS, tag="D-final")
    except llm_transport.TransportError as exc:
        return _fail(exc.reason_code, (time.perf_counter() - start) * 1000.0,
                     llm_calls=llm_calls, tool_calls=stats["tool_calls"],
                     tokens=tokens, detail=str(exc))
    llm_calls += 1
    tokens["prompt"] += response["usage"]["prompt_tokens"]
    tokens["completion"] += response["usage"]["completion_tokens"]
    elapsed = (time.perf_counter() - start) * 1000.0
    if response["tool_calls"]:
        return _fail("FINAL_TURN_TOOL_CALL", elapsed, llm_calls=llm_calls,
                     tool_calls=stats["tool_calls"], tokens=tokens,
                     detail=f"phase={phase_reason} "
                            f"tools={[c['name'] for c in response['tool_calls']]}")
    try:
        action = protocol.parse_decision(response["content"])
    except DecisionParseError as exc:
        return _fail(exc.reason_code, elapsed, llm_calls=llm_calls,
                     tool_calls=stats["tool_calls"], tokens=tokens,
                     detail=f"phase={phase_reason} raw={response['content'][:100]!r}")
    return _ok(action, elapsed, llm_calls=llm_calls, tool_calls=stats["tool_calls"],
               tokens=tokens, detail=f"phase={phase_reason} -> {response['content'][:160]}")


def allowed_actions() -> tuple[str, ...]:
    return ACTIONS
