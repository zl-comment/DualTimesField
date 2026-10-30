"""Frozen LLM identity, strict parser and request builder for the M1 contract.

No live call happens in M1: a transport injects provider-shaped responses and this
module enforces the frozen wire discipline.  The parser accepts exactly one JSON
object or one strict function call with no prose, rejects duplicate keys, NaN/
Infinity, ANY numeric literal, unknown/missing keys and oversize payloads, resolves
opaque refs against the enabled set and never synthesizes sampling controls.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import m1_contract
from .m1_actions import M1ActionKind

C_DECISIONS = ("USE_TYPED", "USE_BASE")
C_FUNCTION_NAME = "m1_c_decision"
D_FUNCTION_NAME = "m1_d_action"


class LlmParseError(Exception):
    def __init__(self, reason_code: str, detail: str = ""):
        super().__init__(f"LLM output rejected: {reason_code} {detail}")
        self.reason_code = reason_code


@dataclass(frozen=True)
class M1LlmResponse:
    """Provider-shaped response record; raw bytes plus usage are the evidence."""
    status: str
    model: str
    output_raw: bytes
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0
    refusal: bool = False
    incomplete_details: bool = False


def _no_duplicate_keys(pairs):
    seen: dict = {}
    for key, value in pairs:
        if key in seen:
            raise LlmParseError("LLM_MALFORMED", f"duplicate key {key!r}")
        seen[key] = value
    return seen


def _no_numbers(value_token: str):
    raise LlmParseError("LLM_MALFORMED",
                        f"numeric literals are forbidden: {value_token}")


def parse_strict_json_object(raw: bytes) -> dict:
    if not isinstance(raw, (bytes, bytearray)):
        raise LlmParseError("LLM_MALFORMED", "output must be raw bytes")
    if len(raw) > m1_contract.LLM_RAW_OUTPUT_CAP_BYTES:
        raise LlmParseError("LLM_MALFORMED", "output exceeds the 4096-byte cap")
    try:
        text = bytes(raw).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise LlmParseError("LLM_MALFORMED", "output is not strict UTF-8") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_no_duplicate_keys,
            parse_int=_no_numbers,
            parse_float=_no_numbers,
            parse_constant=_no_numbers,
        )
    except LlmParseError:
        raise
    except ValueError as exc:
        raise LlmParseError("LLM_MALFORMED", f"not one strict JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise LlmParseError("LLM_MALFORMED", "output is not a JSON object")
    return payload


def _parse_payload(response: M1LlmResponse) -> dict:
    if response.status == "TIMEOUT":
        raise LlmParseError("LLM_TIMEOUT")
    if response.status == "REFUSED" or response.refusal:
        raise LlmParseError("LLM_REFUSAL")
    if response.status == "INCOMPLETE" or response.incomplete_details:
        raise LlmParseError("LLM_INCOMPLETE")
    if response.status != "COMPLETED":
        raise LlmParseError("LLM_INCOMPLETE", f"status {response.status!r}")
    if response.model != m1_contract.LLM_MODEL:
        raise LlmParseError("LLM_MALFORMED",
                            f"response.model {response.model!r} is not the frozen "
                            f"snapshot {m1_contract.LLM_MODEL!r}")
    payload = parse_strict_json_object(response.output_raw)
    if "name" in payload and "arguments" in payload:  # strict function-call envelope
        if set(payload) != {"name", "arguments"}:
            raise LlmParseError("LLM_MALFORMED", "unknown keys in the function call")
        if payload["name"] not in (C_FUNCTION_NAME, D_FUNCTION_NAME):
            raise LlmParseError("LLM_MALFORMED",
                                f"unknown function {payload['name']!r}")
        if not isinstance(payload["arguments"], str):
            raise LlmParseError("LLM_MALFORMED", "function arguments must be a string")
        inner = parse_strict_json_object(payload["arguments"].encode())
        # __function__ is reserved for the envelope: an inner key with this name
        # must never override or erase the outer function identity
        if "__function__" in inner:
            raise LlmParseError(
                "LLM_MALFORMED", "__function__ is reserved for the envelope")
        return {"__function__": payload["name"], **inner}
    if "__function__" in payload:
        raise LlmParseError(
            "LLM_MALFORMED",
            "__function__ may only come from a function-call envelope")
    return payload


def parse_c_decision(response: M1LlmResponse, *,
                     enabled_evidence_refs: tuple[str, ...]) -> tuple[str, str]:
    payload = _parse_payload(response)
    function = payload.pop("__function__", None)
    if function not in (None, C_FUNCTION_NAME):
        raise LlmParseError("LLM_MALFORMED", "C received a non-C function call")
    if set(payload) != {"decision", "evidence_ref"}:
        raise LlmParseError("LLM_MALFORMED",
                            f"C payload keys must be exactly decision/evidence_ref, "
                            f"got {sorted(payload)}")
    decision, ref = payload["decision"], payload["evidence_ref"]
    if decision not in C_DECISIONS:
        raise LlmParseError("LLM_MALFORMED", f"unknown decision {decision!r}")
    if ref != "NONE" and ref not in enabled_evidence_refs:
        raise LlmParseError("LLM_MALFORMED", f"evidence ref {ref!r} is not enabled")
    return decision, ref


def parse_d_action(response: M1LlmResponse, *,
                   enabled_refs: tuple[str, ...]) -> tuple[M1ActionKind, str]:
    payload = _parse_payload(response)
    function = payload.pop("__function__", None)
    if function not in (None, D_FUNCTION_NAME):
        raise LlmParseError("LLM_MALFORMED", "D received a non-D function call")
    if set(payload) != {"action", "ref"}:
        raise LlmParseError("LLM_MALFORMED",
                            f"D payload keys must be exactly action/ref, "
                            f"got {sorted(payload)}")
    action, ref = payload["action"], payload["ref"]
    try:
        kind = M1ActionKind(action)
    except ValueError as exc:
        raise LlmParseError("LLM_MALFORMED", f"unknown action {action!r}") from exc
    if kind is M1ActionKind.FETCH_REF:
        if ref not in enabled_refs:
            raise LlmParseError("LLM_MALFORMED", f"ref {ref!r} is not enabled")
    elif ref != "NONE":
        raise LlmParseError("LLM_MALFORMED",
                            f"{kind.name} must carry ref NONE, got {ref!r}")
    return kind, ref


def build_request_bytes(arm: str, digest: str, repo_root: str | Path) -> bytes:
    """Render the frozen request: byte-exact prompt plus bound schema and cap."""
    if arm not in ("C", "D"):
        raise ValueError("LLM requests exist only for arms C and D")
    directory = Path(repo_root) / m1_contract.PROMPT_DIRECTORY
    system = (directory / "m1_common_system_prompt.txt").read_text()
    template_name = ("m1_c_prompt_template.txt" if arm == "C"
                     else "m1_d_prompt_template.txt")
    template = (directory / template_name).read_text()
    placeholder = "{{canonical_digest}}" if arm == "C" else "{{canonical_state}}"
    if placeholder not in template:
        raise ValueError(f"frozen {arm} template lost its digest placeholder")
    schema_name = ("m1_c_decision_schema.json" if arm == "C"
                   else "m1_d_action_schema.json")
    schema = json.loads((directory / schema_name).read_text())
    user_message = template.replace(placeholder, digest)
    request = {
        "model": m1_contract.LLM_MODEL,
        "instructions": system,
        "input": user_message,
        "reasoning": {"effort": m1_contract.LLM_REASONING_EFFORT},
        "store": m1_contract.LLM_STORE,
        "truncation": m1_contract.LLM_TRUNCATION,
        "parallel_tool_calls": m1_contract.LLM_PARALLEL_TOOL_CALLS,
        "max_output_tokens": m1_contract.PER_CALL_OUTPUT_TOKENS[arm],
        "text": {"format": {
            "type": "json_schema",
            "name": (C_FUNCTION_NAME if arm == "C" else D_FUNCTION_NAME),
            "strict": True,
            "schema": schema,
        }},
    }
    return json.dumps(request, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode()


def cost_micro_usd(input_tokens: int, output_tokens: int) -> int:
    """Integer micro-USD at the frozen rates (2.5 / 15.0 USD per million tokens)."""
    return int(round(input_tokens * 2.5 + output_tokens * 15.0))
