"""Frozen M1 agent-contract identity, budgets, arms and pinning (candidate 2).

Everything in this module is contract identity: the inherited M0 file hashes, the
frozen prompt/schema artifacts, the frozen LLM model identity, the shared budget
vector, the declared arm leaves and the reason vocabularies.  Drift against any pin
is a run hard stop raised BEFORE any action executes.  The module never imports
label, scoring or training code and never touches the network.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

CONTRACT_ID = "m1.agent_contract.c2.v1"
CANDIDATE_ID = "predispatchis_absolute_totaldemand_future_v1"
WITHDRAWAL_SEMANTICS = "NOT_APPLICABLE"
CONFIG_PATH = "configs/agent_forecasting/m1_agent_contract_c2_20261029.json"
PROMPT_DIRECTORY = "docs/research/agent_forecasting/m1_prompts"
RENDERER_PATH = "forecasting/agent_forecasting/m1_digest.py"

M0_PINNED_FILES = (
    "docs/research/agent_forecasting/m0_predispatch_td_abs_protocol_20261029.txt",
    "configs/agent_forecasting/m0_predispatch_td_abs_c2_20261029.json",
    "forecasting/agent_forecasting/predispatch_td_abs.py",
    "forecasting/agent_forecasting/contracts.py",
    "forecasting/agent_forecasting/frozen_chronos.py",
)
PROMPT_ARTIFACTS = (
    "m1_common_system_prompt.txt",
    "m1_c_prompt_template.txt",
    "m1_d_prompt_template.txt",
    "m1_c_decision_schema.json",
    "m1_d_action_schema.json",
)
PINNED_RELATIVE_PATHS = M0_PINNED_FILES + tuple(
    f"{PROMPT_DIRECTORY}/{name}" for name in PROMPT_ARTIFACTS) + (RENDERER_PATH,)

LLM_MODEL = "gpt-5.4-2026-03-05"
LLM_ENDPOINT = "POST https://api.openai.com/v1/responses"
LLM_INPUT_TOKEN_ENDPOINT = "POST https://api.openai.com/v1/responses/input_tokens"
LLM_REASONING_EFFORT = "none"
LLM_STORE = False
LLM_TRUNCATION = "disabled"
LLM_PARALLEL_TOOL_CALLS = False
LLM_RAW_OUTPUT_CAP_BYTES = 4096
LLM_TIMEOUT_SECONDS = 30

ORIGIN_REASON_CODES = frozenset({
    "NO_VISIBLE_RECEIPT", "RECEIPT_AFTER_ORIGIN", "INCOMPLETE_NEWEST_VINTAGE",
    "SCHEMA_DRIFT", "BODY_HASH_MISMATCH", "FORK", "CLOCK_UNHEALTHY",
    "HTTP_TIMEOUT", "HTTP_STATUS", "LLM_TIMEOUT", "LLM_REFUSAL", "LLM_INCOMPLETE",
    "LLM_MALFORMED", "ACTION_PRECONDITION", "BUDGET_EXHAUSTED", "BUDGET_OVERRUN",
    "DEADLINE_MISSED", "FORECAST_TOOL_FAILURE", "AGENT_ABSTAIN",
    "CONTRACT_IDENTITY_DRIFT",
})
RUN_HARD_STOP_REASONS = frozenset({
    "CONTRACT_IDENTITY_DRIFT", "LABEL_OR_SCORE_ACCESS",
    "FORBIDDEN_SOURCE_TOOL_ACTION", "SOURCE_TERMS_CHANGE",
})

# The common per-origin envelope: identical for A/B/C/D in every field.
COMMON_BUDGET_VECTOR = {arm: {
    "logical_source_replay_calls": 3,
    "logical_source_bytes_total": 16777216,
    "logical_source_bytes_one_body": 8388608,
    "llm_input_token_count_requests": 4,
    "llm_input_tokens": 16384,
    "llm_output_tokens": 1024,
    "llm_cost_micro_usd": 60000,
    "agent_action_proposals": 4,
    "local_deterministic_tool_calls": 8,
    "forecast_calls": 2,
    "evidence_total_bytes": 19398656,
    "retries": 0,
    "deadline_success_publish_by_offset": 600,
} for arm in ("A", "B", "C", "D")}
ARM_LLM_GENERATIONS = {"A": 0, "B": 0, "C": 1, "D": 4}

# Declared leaf diffs (audit #2): everything else is shared and must be identical.
ARM_LEAVES = {
    "A": {
        "source": "FROZEN_SNAPSHOT_REPLAY", "snapshot": "SHARED_IMMUTABLE",
        "forecast_tool": "FROZEN_POOL", "base_input": "FROZEN_IDENTITY",
        "numeric_adapter": "FROZEN_M0", "fallback": "EXACT_A_BYTES",
        "deadline": "ORIGIN_PLUS_600", "budget_vector": "COMMON_EQUAL",
        "retries": 0, "typed_pipeline": None, "llm_decision": None,
        "llm_envelope_shape": None, "policy": "FIXED_A",
    },
    "B": {
        "source": "FROZEN_SNAPSHOT_REPLAY", "snapshot": "SHARED_IMMUTABLE",
        "forecast_tool": "FROZEN_POOL", "base_input": "FROZEN_IDENTITY",
        "numeric_adapter": "FROZEN_M0", "fallback": "EXACT_A_BYTES",
        "deadline": "ORIGIN_PLUS_600", "budget_vector": "COMMON_EQUAL",
        "retries": 0, "typed_pipeline": "PREDISPATCH_DETERMINISTIC_KERNEL",
        "llm_decision": None, "llm_envelope_shape": None, "policy": "FIXED_MAXIMAL",
    },
    "C": {
        "source": "FROZEN_SNAPSHOT_REPLAY", "snapshot": "SHARED_IMMUTABLE",
        "forecast_tool": "FROZEN_POOL", "base_input": "FROZEN_IDENTITY",
        "numeric_adapter": "FROZEN_M0", "fallback": "EXACT_A_BYTES",
        "deadline": "ORIGIN_PLUS_600", "budget_vector": "COMMON_EQUAL",
        "retries": 0, "typed_pipeline": "PREDISPATCH_DETERMINISTIC_KERNEL",
        "llm_decision": "ONE_SHOT_USE_TYPED_OR_BASE",
        "llm_envelope_shape": None, "policy": "FIXED_B_PLUS_ONE_SHOT",
    },
    "D": {
        "source": "FROZEN_SNAPSHOT_REPLAY", "snapshot": "SHARED_IMMUTABLE",
        "forecast_tool": "FROZEN_POOL", "base_input": "FROZEN_IDENTITY",
        "numeric_adapter": "FROZEN_M0", "fallback": "EXACT_A_BYTES",
        "deadline": "ORIGIN_PLUS_600", "budget_vector": "COMMON_EQUAL",
        "retries": 0, "typed_pipeline": "PREDISPATCH_DETERMINISTIC_KERNEL",
        "llm_decision": "ONE_SHOT_USE_TYPED_OR_BASE",
        "llm_envelope_shape": "FOUR_CALLS_SPLIT_EQUAL_TOTALS",
        "policy": "AGENT_ORDER_SUBSET",
    },
}


class M1IdentityDrift(Exception):
    """Raised when any pinned identity fails to recompute; precedes every action."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(repo_root: str | Path) -> dict:
    return json.loads((Path(repo_root) / CONFIG_PATH).read_text())


def declared_arm_leaf_diffs() -> dict[str, tuple[str, ...]]:
    return {
        "B_minus_A": ("policy", "typed_pipeline"),
        "C_minus_B": ("llm_decision", "policy"),
        "D_minus_C": ("llm_envelope_shape", "policy"),
    }


@lru_cache(maxsize=8)
def _verify_run_identity_cached(root_key: str) -> dict:
    root = Path(root_key)
    config = load_config(root)
    checked: dict[str, str] = {}
    for relative in M0_PINNED_FILES:
        pinned = config["frozen_base"]["files_sha256"][relative]
        actual = sha256_file(root / relative)
        if actual != pinned:
            raise M1IdentityDrift(
                f"inherited M0 identity drift: {relative} {actual} != {pinned}")
        checked[relative] = actual
    for name in PROMPT_ARTIFACTS:
        pinned = config["prompt_artifacts"]["sha256"][name]
        actual = sha256_file(root / PROMPT_DIRECTORY / name)
        if actual != pinned:
            raise M1IdentityDrift(
                f"prompt artifact drift: {name} {actual} != {pinned}")
        checked[f"{PROMPT_DIRECTORY}/{name}"] = actual
    renderer_pin = config["digest_renderer"]["sha256"]
    if renderer_pin == "MUST_BIND_AT_IMPLEMENTATION_COMMIT":
        raise M1IdentityDrift("digest renderer hash was never bound at implementation")
    actual_renderer = sha256_file(root / RENDERER_PATH)
    if actual_renderer != renderer_pin:
        raise M1IdentityDrift(
            f"digest renderer drift: {actual_renderer} != {renderer_pin}")
    checked[RENDERER_PATH] = actual_renderer
    if config.get("contract_id") != CONTRACT_ID:
        raise M1IdentityDrift("config contract_id drifted")
    return {"status": "VERIFIED", "pinned_files": len(checked),
            "contract_id": CONTRACT_ID}


def verify_run_identity(repo_root: str | Path) -> dict:
    """Recompute every pin against the tree; refuse (before any action) on drift.

    Results are cached per resolved tree for the process lifetime: within one
    single-writer process a tree that already verified stays verified; a different
    or mutated tree is a different cache key only if its path differs, so tests
    that mutate copies in temporary trees always recompute from bytes.
    """
    return _verify_run_identity_cached(str(Path(repo_root).resolve()))


def map_gate(reducer_report: dict) -> dict:
    """Map the executable reducer verdict to the scientific decision.

    Engineering greenness NEVER converts sufficiency into ADVANCE_M1: a sufficiency
    verdict is a scientific STOP with metrics explicitly not evaluated.
    """
    verdict = reducer_report.get("verdict")
    gate = {
        "contract_id": CONTRACT_ID,
        "candidate_id": CANDIDATE_ID,
        "engineering_green": bool(reducer_report.get("engineering_green", False)),
        "reducibility_verdict": verdict,
        "metrics_status": "NOT_EVALUATED",
        "mission_goal_achieved": False,
        "advance_m1": False,
    }
    if verdict == "FIXED_RULES_SUFFICIENT_AGENT_NOT_NEEDED":
        gate.update({"scientific_decision": "STOP",
                     "stop_token": verdict})
    elif verdict == "REDUCIBILITY_REFUTED":
        gate.update({"scientific_decision": "PENDING_REVIEW",
                     "stop_token": None})
    elif verdict == "REDUCER_NON_VACUITY_FAILED":
        gate.update({"scientific_decision": "BLOCKED",
                     "stop_token": None})
    else:
        gate.update({"scientific_decision": "BLOCKED",
                     "stop_token": None,
                     "note": "no executable reducer verdict; ADVANCE_M1 is forbidden"})
    return gate
