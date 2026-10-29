"""Frozen digest renderer: the ONLY channel from run state to an LLM prompt.

The renderer accepts a fixed fact space and emits canonical JSON whose values are
exclusively enum strings, opaque ref strings and booleans.  Timestamps, counts,
numeric values, raw bytes, TOTALDEMAND values, prices, forecasts, labels, scores,
URLs, paths and free source text are structurally impossible: every render walks the
value tree and rejects any number before serialization.
"""
from __future__ import annotations

import json
from typing import Mapping

ALLOWED_FACT_KEYS = frozenset({
    "contract", "arm", "state", "enabled_actions", "enabled_refs", "budget",
    "receipt_cutoff_relation", "coverage", "schema", "fork", "typed_ready",
    "evidence_refs",
})
ENUM_STATES = frozenset({"INIT", "A_READY", "LISTED", "FETCHED", "TYPED_READY"})
ENUM_ACTIONS = frozenset({"LIST", "FETCH_REF", "FINALIZE_TYPED", "FINALIZE_BASE"})
ENUM_BUDGET_CLASSES = frozenset({
    "source_replay", "llm_generation", "llm_tokens", "forecast_calls",
    "wall_clock", "evidence_bytes"})
ENUM_BUDGET_LEVELS = frozenset({"AVAILABLE", "EXHAUSTED"})
ENUM_RELATIONS = frozenset({"BEFORE_CUTOFF", "EQUAL_CUTOFF", "AFTER_CUTOFF"})
ENUM_COVERAGE = frozenset({"COMPLETE", "INCOMPLETE", "NOT_FETCHED"})
ENUM_SCHEMA = frozenset({"VALID", "REJECTED", "NOT_FETCHED"})
ENUM_FORK = frozenset({"PRESENT", "ABSENT", "UNKNOWN"})
ENUM_READY = frozenset({"YES", "NO"})
_ENUMS_BY_KEY = {
    "state": ENUM_STATES,
    "fork": ENUM_FORK,
    "typed_ready": ENUM_READY,
}
_ENUM_BY_SUBKEY = {
    "receipt_cutoff_relation": ENUM_RELATIONS,
    "coverage": ENUM_COVERAGE,
    "schema": ENUM_SCHEMA,
}
_REF_PATTERN_CHARS = set("0123456789abcdef")


class DigestViolation(Exception):
    """Raised when a fact is outside the frozen fact space or carries a number."""


def _check_ref(value: object) -> None:
    if not isinstance(value, str):
        raise DigestViolation(f"ref facts must be strings, got {type(value)!r}")
    if value in ("NONE",):
        return
    for prefix in ("v_", "snap_", "ev_"):
        if value.startswith(prefix) and len(value) == len(prefix) + 16:
            if set(value[len(prefix):]) <= _REF_PATTERN_CHARS:
                return
    raise DigestViolation(f"opaque ref {value!r} is not an enabled ref token")


def _walk_no_numbers(value: object, where: str) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        raise DigestViolation(f"numeric fact is forbidden in digests ({where})")
    if isinstance(value, str):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _walk_no_numbers(key, f"{where}.{key}")
            _walk_no_numbers(item, f"{where}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _walk_no_numbers(item, where)
        return
    raise DigestViolation(f"unsupported fact type {type(value)!r} at {where}")


def render_digest(facts: Mapping) -> str:
    """Render the canonical digest; anything outside the fact space fails closed."""
    if not isinstance(facts, Mapping):
        raise DigestViolation("digest facts must be a mapping")
    missing = ALLOWED_FACT_KEYS - set(facts)
    unknown = set(facts) - ALLOWED_FACT_KEYS
    if missing or unknown:
        raise DigestViolation(
            f"digest fact space violated: missing={sorted(missing)} "
            f"unknown={sorted(unknown)}")
    _walk_no_numbers(facts, "root")
    clean = {
        "contract": facts["contract"],
        "arm": facts["arm"],
        "state": facts["state"],
        "enabled_actions": tuple(facts["enabled_actions"]),
        "enabled_refs": tuple(facts["enabled_refs"]),
        "budget": dict(facts["budget"]),
        "receipt_cutoff_relation": dict(facts["receipt_cutoff_relation"]),
        "coverage": dict(facts["coverage"]),
        "schema": dict(facts["schema"]),
        "fork": facts["fork"],
        "typed_ready": facts["typed_ready"],
        "evidence_refs": tuple(facts["evidence_refs"]),
    }
    if clean["contract"] != "m1.agent_contract.c2.v1":
        raise DigestViolation("digest contract identity drifted")
    if clean["arm"] not in ("C", "D"):
        raise DigestViolation("digest arm must be C or D")
    for key, allowed in _ENUMS_BY_KEY.items():
        if clean[key] not in allowed:
            raise DigestViolation(f"{key}={clean[key]!r} is not a frozen enum")
    for action in clean["enabled_actions"]:
        if action not in ENUM_ACTIONS:
            raise DigestViolation(f"enabled action {action!r} is not a frozen enum")
    for ref in (*clean["enabled_refs"], *clean["evidence_refs"]):
        _check_ref(ref)
    if set(clean["budget"]) != ENUM_BUDGET_CLASSES:
        raise DigestViolation("budget fact classes drifted")
    for level in clean["budget"].values():
        if level not in ENUM_BUDGET_LEVELS:
            raise DigestViolation(f"budget level {level!r} is not a frozen enum")
    for key, allowed in _ENUM_BY_SUBKEY.items():
        for ref, value in clean[key].items():
            _check_ref(ref)
            if value not in allowed:
                raise DigestViolation(f"{key}[{ref}]={value!r} is not a frozen enum")
    return json.dumps(clean, sort_keys=True, separators=(",", ":"))
