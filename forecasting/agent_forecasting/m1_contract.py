"""Frozen M1 agent-contract identity, budgets, arms and pinning (candidate 2).

Everything in this module is contract identity: the inherited M0 file hashes, the
frozen prompt/schema artifacts, the frozen LLM model identity, the shared budget
vector, the declared arm leaves and the reason vocabularies.  Drift against any pin
is a run hard stop raised BEFORE any action executes.  The module never imports
label, scoring or training code and never touches the network.

Audit-3/H1 closure: identity is FAIL-CLOSED over the complete config.  The pin
maps must carry EXACTLY the frozen keysets (a deleted pin is drift, an extra
pin is drift), every config field — source contract, arms declaration, branch
ancestry, LLM endpoints/cost, envelopes, forecast tool binding, deadline
reserves, evidence root, interpreter, status — is cross-checked against the
constants below, and verify_run_entry additionally binds the git HEAD, a
tracked-clean tree and the config file's own committed bytes.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

CONTRACT_ID = "m1.agent_contract.c2.v1"
CANDIDATE_ID = "predispatchis_absolute_totaldemand_future_v1"
WITHDRAWAL_SEMANTICS = "NOT_APPLICABLE"
CONFIG_PATH = "configs/agent_forecasting/m1_agent_contract_c2_20261029.json"
PROTOCOL_DOC = "docs/research/agent_forecasting/m1_agent_contract_c2_20261029.txt"
PROMPT_DIRECTORY = "docs/research/agent_forecasting/m1_prompts"
RENDERER_PATH = "forecasting/agent_forecasting/m1_digest.py"

# run ancestry and frozen surface identity (audit-3/H1: every scalar config
# field is contract identity; drift in ANY of them refuses the run)
BRANCH = "research/agentic-m1-bounded-actions-c2-20261029-r5"
BASE_BRANCH = "research/agentic-asof-abs-predispatch-c2-20261029-r4"
BASE_SHA = "e458016386311b277e193ec5a35b94aad1436c92"
RUN_PROVENANCE_COMMIT = "9feb95fd41ee43b37cbb08b1d9ae3ef8428412f1"
EVIDENCE_ROOT = "logs/forecasting/m1_agent_contract_c2_20261029"
EXECUTION_SCOPE = "m1_contract_enforcement_only"
INTERPRETER_PATH = "/root/workspace/Code/DualTimesField/.venv/bin/python"
CONFIG_STATUS = "frozen_m1_contract"
METRICS_STATUS = "NOT_EVALUATED"
ADVANCE_TO_G1_G2 = False
M0_HEADER_LINE_SHA256 = (
    "2f013ba453d6ef05959d74895fa6db6f9cd2522b6c0f7d4356af09fadaaa529a")

SOURCE_CONTRACT = {
    "csv_interface": "PREDISPATCH,REGION_SOLUTION,9",
    "field": "TOTALDEMAND",
    "header_field_count": 131,
    "hour_mapping": ("hour h is the mean of half-hour ends origin+(2h+1)*1800 "
                     "and origin+(2h+2)*1800"),
    "intervention": 0,
    "non_clocks": ["filename timestamp", "HTTP Date", "HTTP Last-Modified",
                   "LASTCHANGED", "directory order"],
    "official_table": "PREDISPATCHREGIONSUM",
    "region": "NSW1",
    "unit": "MW",
    "url": "https://www.nemweb.com.au/REPORTS/CURRENT/PredispatchIS_Reports/",
    "vintage_key": ["PREDISPATCHSEQNO", "RUNNO"],
    "visibility_rule": "receipt_complete_unix_ceil <= origin_unix",
    "withdrawal_semantics": "NOT_APPLICABLE",
}

ARMS_DECLARATION = {
    "A": ["RUN_BASE_FORECAST", "ATOMIC_PUBLISH_BASE"],
    "B": ["RUN_BASE_FORECAST", "LIST", "FETCH_CANONICAL_NEWEST_VISIBLE_1",
          "FETCH_CANONICAL_NEWEST_VISIBLE_2_IF_PRESENT",
          "DETERMINISTIC_VALIDATE_DERIVE_ATTACH",
          "RUN_TYPED_FORECAST_IF_READY", "ATOMIC_PUBLISH_TYPED_ELSE_EXACT_BASE"],
    "C": ["RUN_BASE_FORECAST", "BYTE_IDENTICAL_B_RETRIEVAL_VALIDATION",
          "ONE_FROZEN_LLM_DECISION",
          "RUN_TYPED_FORECAST_IF_READY_AND_SELECTED",
          "ATOMIC_PUBLISH_TYPED_ELSE_EXACT_BASE"],
    "D": ["RUN_BASE_FORECAST", "UP_TO_FOUR_TYPED_ACTION_PROPOSALS",
          "CONTROLLER_EXECUTES_ONLY_PRECONDITION_VALID_ACTIONS",
          "RUN_TYPED_FORECAST_AT_MOST_ONCE",
          "ATOMIC_PUBLISH_TYPED_ELSE_EXACT_BASE"],
}

LLM_PROVIDER = "OpenAI"
LLM_RETRIES = 0
LLM_SAMPLING_NOTE = (
    "No seed or determinism claim; unspecified sampling controls are not "
    "synthesized; raw request/response bytes and provider usage are the evidence.")

LLM_COST = {
    "accounting_unit": "integer micro_usd",
    "hard_cap_usd": 0.06,
    "rate_input_usd_per_million": 2.5,
    "rate_output_usd_per_million": 15.0,
    "worst_case_usd_at_token_caps": 0.05632,
}

FORECAST_TOOL_BINDING = {
    "binding": ("identity-only; the frozen Chronos-2 CPU path from M0 is pinned "
                "via frozen_base checkpoint/packages and is NOT executed in M1 "
                "probes"),
    "calls_max": 2,
    "output_rule": ("published bytes are the frozen tool bytes; no clip/sort/"
                    "calibration/edit"),
    "unique_input_only": True,
}

HTTP_TIMEOUT_SECONDS = 15
LOCAL_TOOL_TIMEOUT_SECONDS = 30
FORECAST_CALL_WALL_SECONDS = 300
WHOLE_PROCESS_MAX_SECONDS = 900

# the exact frozen source-file list of the contract surface, in config order
SOURCE_FILE_LIST = (
    CONFIG_PATH,
    PROTOCOL_DOC,
    f"{PROMPT_DIRECTORY}/m1_common_system_prompt.txt",
    f"{PROMPT_DIRECTORY}/m1_c_prompt_template.txt",
    f"{PROMPT_DIRECTORY}/m1_d_prompt_template.txt",
    f"{PROMPT_DIRECTORY}/m1_c_decision_schema.json",
    f"{PROMPT_DIRECTORY}/m1_d_action_schema.json",
    "forecasting/agent_forecasting/m1_contract.py",
    RENDERER_PATH,
    "forecasting/agent_forecasting/m1_budget.py",
    "forecasting/agent_forecasting/m1_actions.py",
    "forecasting/agent_forecasting/m1_llm.py",
    "forecasting/agent_forecasting/m1_asof.py",
    "forecasting/agent_forecasting/m1_runtime.py",
    "forecasting/agent_forecasting/m1_publication.py",
    "forecasting/agent_forecasting/m1_reducer.py",
    "scripts/run_logged_m1.sh",
    "scripts/run_m1_reducer.py",
    "scripts/build_m1_gate.py",
    "scripts/build_m1_manifest.py",
    "scripts/verify_m1_evidence.py",
    "tests/fixtures/agent_forecasting/m1_scenario_harness.py",
    "tests/test_agent_m1_contract.py",
    "tests/test_agent_m1_reducer.py",
)

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
# the M1 execution surface closed at run time: every module that can influence a
# verdict, plus the protocol document and the scenario harness that defines the
# reducibility cross (all hashed from config; the config's own fields are
# cross-checked against these constants below)
EXECUTION_PINNED_FILES = (
    "forecasting/agent_forecasting/m1_contract.py",
    "forecasting/agent_forecasting/m1_actions.py",
    "forecasting/agent_forecasting/m1_asof.py",
    "forecasting/agent_forecasting/m1_budget.py",
    "forecasting/agent_forecasting/m1_llm.py",
    "forecasting/agent_forecasting/m1_publication.py",
    "forecasting/agent_forecasting/m1_reducer.py",
    "forecasting/agent_forecasting/m1_runtime.py",
    PROTOCOL_DOC,
    "tests/fixtures/agent_forecasting/m1_scenario_harness.py",
    "scripts/run_m1_reducer.py",
    "scripts/build_m1_gate.py",
    "scripts/build_m1_manifest.py",
    "scripts/verify_m1_evidence.py",
    "scripts/run_logged_m1.sh",
)
PINNED_RELATIVE_PATHS = M0_PINNED_FILES + tuple(
    f"{PROMPT_DIRECTORY}/{name}" for name in PROMPT_ARTIFACTS) + (
    RENDERER_PATH,) + EXECUTION_PINNED_FILES

LLM_MODEL = "gpt-5.4-2026-03-05"
LLM_ENDPOINT = "POST https://api.openai.com/v1/responses"
LLM_INPUT_TOKEN_ENDPOINT = "POST https://api.openai.com/v1/responses/input_tokens"
LLM_REASONING_EFFORT = "none"
LLM_STORE = False
LLM_TRUNCATION = "disabled"
LLM_PARALLEL_TOOL_CALLS = False
LLM_RAW_OUTPUT_CAP_BYTES = 4096
LLM_TIMEOUT_SECONDS = 30

# per-call LLM envelopes and deadline reserves: the single source of truth for
# m1_runtime, m1_llm request building and the config field closure below
PER_CALL_INPUT_TOKENS = {"C": 16384, "D": 4096}
PER_CALL_OUTPUT_TOKENS = {"C": 1024, "D": 256}
C_PER_CALL_INPUT_TOKENS = PER_CALL_INPUT_TOKENS["C"]
C_PER_CALL_OUTPUT_TOKENS = PER_CALL_OUTPUT_TOKENS["C"]
D_PER_CALL_INPUT_TOKENS = PER_CALL_INPUT_TOKENS["D"]
D_PER_CALL_OUTPUT_TOKENS = PER_CALL_OUTPUT_TOKENS["D"]
DEADLINE_OFFSET = 600
TYPED_RESERVE = 300
PUBLICATION_RESERVE = 30

ORIGIN_REASON_CODES = frozenset({
    "NO_VISIBLE_RECEIPT", "RECEIPT_AFTER_ORIGIN", "INCOMPLETE_NEWEST_VINTAGE",
    "SCHEMA_DRIFT", "BODY_HASH_MISMATCH", "FORK", "CLOCK_UNHEALTHY",
    "HTTP_TIMEOUT", "HTTP_STATUS", "LLM_TIMEOUT", "LLM_REFUSAL", "LLM_INCOMPLETE",
    "LLM_MALFORMED", "ACTION_PRECONDITION", "BUDGET_EXHAUSTED", "BUDGET_OVERRUN",
    "DEADLINE_MISSED", "FORECAST_TOOL_FAILURE", "AGENT_ABSTAIN",
    "CONTRACT_IDENTITY_DRIFT", "EVIDENCE_PUBLICATION_FAILED",
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


def _verify_execution_pins(root: Path, config: dict, checked: dict) -> None:
    pins = config.get("execution_modules", {}).get("files_sha256")
    if not isinstance(pins, dict) or not pins:
        raise M1IdentityDrift(
            "config carries no execution_modules pins: the M1 execution surface "
            "is not closed")
    if set(pins) != set(EXECUTION_PINNED_FILES):
        # fail-closed over the EXACT frozen keyset: a deleted pin is drift
        # (the module would silently escape hashing) and an extra pin is drift
        # (an undeclared file would be smuggled into the frozen surface)
        missing = sorted(set(EXECUTION_PINNED_FILES) - set(pins))
        extra = sorted(set(pins) - set(EXECUTION_PINNED_FILES))
        raise M1IdentityDrift(
            "execution_modules.files_sha256 keyset != the frozen execution "
            f"surface (missing={missing}, extra={extra})")
    for relative, pinned in sorted(pins.items()):
        actual = sha256_file(root / relative)
        if actual != pinned:
            raise M1IdentityDrift(
                f"execution module drift: {relative} {actual} != {pinned}")
        checked[relative] = actual


def _verify_config_fields(config: dict) -> int:
    """Cross-check every frozen config field against these module constants."""
    from . import m1_budget  # local import: budget caps are contract identity

    checks = 0
    llm = config["llm"]
    expected_llm = {
        "model": LLM_MODEL, "response_model_must_equal": LLM_MODEL,
        "raw_output_cap_bytes": LLM_RAW_OUTPUT_CAP_BYTES,
        "timeout_seconds_each": LLM_TIMEOUT_SECONDS,
    }
    for key, value in expected_llm.items():
        if llm.get(key) != value:
            raise M1IdentityDrift(f"config llm.{key} drifted from the contract")
        checks += 1
    if llm.get("reasoning") != {"effort": LLM_REASONING_EFFORT}:
        raise M1IdentityDrift("config llm.reasoning drifted")
    if llm.get("store") is not LLM_STORE or llm.get("truncation") != LLM_TRUNCATION:
        raise M1IdentityDrift("config llm store/truncation drifted")
    if llm.get("parallel_tool_calls") is not LLM_PARALLEL_TOOL_CALLS:
        raise M1IdentityDrift("config llm.parallel_tool_calls drifted")
    checks += 3

    budgets = config["budgets"]
    budget_key_to_class = {
        "logical_source_replay_calls": m1_budget.BudgetClass.SOURCE_REPLAY_CALLS,
        "logical_source_bytes_total": m1_budget.BudgetClass.SOURCE_BYTES_TOTAL,
        "logical_source_bytes_one_body": m1_budget.BudgetClass.SOURCE_BYTES_ONE_BODY,
        "llm_input_token_count_requests":
            m1_budget.BudgetClass.LLM_TOKEN_COUNT_REQUESTS,
        "llm_generation_calls": m1_budget.BudgetClass.LLM_GENERATIONS,
        "llm_input_tokens": m1_budget.BudgetClass.LLM_INPUT_TOKENS,
        "llm_output_tokens": m1_budget.BudgetClass.LLM_OUTPUT_TOKENS,
        "llm_cost_micro_usd": m1_budget.BudgetClass.LLM_COST_MICRO_USD,
        "agent_action_proposals": m1_budget.BudgetClass.ACTION_PROPOSALS,
        "local_deterministic_tool_calls": m1_budget.BudgetClass.LOCAL_TOOL_CALLS,
        "forecast_calls": m1_budget.BudgetClass.FORECAST_CALLS,
        "evidence_source_bytes": m1_budget.BudgetClass.EVIDENCE_SOURCE_BYTES,
        "evidence_llm_bytes": m1_budget.BudgetClass.EVIDENCE_LLM_BYTES,
        "evidence_ledger_bytes": m1_budget.BudgetClass.EVIDENCE_LEDGER_BYTES,
        "evidence_total_bytes": m1_budget.BudgetClass.EVIDENCE_TOTAL_BYTES,
    }
    if set(budgets) != set(budget_key_to_class) | {"retries"}:
        raise M1IdentityDrift("config budgets key set drifted")
    for key, budget_class in sorted(budget_key_to_class.items()):
        if budgets[key] != m1_budget.CAPS[budget_class]:
            raise M1IdentityDrift(
                f"config budgets.{key} drifted from the ledger cap")
        checks += 1
    if budgets["retries"] != 0:
        raise M1IdentityDrift("config budgets.retries must be zero")

    envelopes = config["llm_envelopes"]
    if envelopes["C"]["input_tokens_max"] != PER_CALL_INPUT_TOKENS["C"]:
        raise M1IdentityDrift("config C input envelope drifted")
    if envelopes["C"]["output_tokens_max"] != PER_CALL_OUTPUT_TOKENS["C"]:
        raise M1IdentityDrift("config C output envelope drifted")
    if envelopes["D"]["input_tokens_max_per_call"] != PER_CALL_INPUT_TOKENS["D"]:
        raise M1IdentityDrift("config D per-call input envelope drifted")
    if envelopes["D"]["output_tokens_max_per_call"] != PER_CALL_OUTPUT_TOKENS["D"]:
        raise M1IdentityDrift("config D per-call output envelope drifted")
    checks += 4

    deadline = config["deadline"]
    if deadline["success_publish_by_offset_seconds"] != DEADLINE_OFFSET:
        raise M1IdentityDrift("config deadline offset drifted")
    if deadline["typed_forecast_reserve_seconds"] != TYPED_RESERVE:
        raise M1IdentityDrift("config typed reserve drifted")
    if deadline["publication_reserve_seconds"] != PUBLICATION_RESERVE:
        raise M1IdentityDrift("config publication reserve drifted")
    expected_deadline = {
        "forecast_call_wall_seconds": FORECAST_CALL_WALL_SECONDS,
        "http_timeout_seconds": HTTP_TIMEOUT_SECONDS,
        "local_tool_timeout_seconds": LOCAL_TOOL_TIMEOUT_SECONDS,
        "publication_reserve_seconds": PUBLICATION_RESERVE,
        "success_publish_by_offset_seconds": DEADLINE_OFFSET,
        "typed_forecast_reserve_seconds": TYPED_RESERVE,
        "whole_process_max_seconds": WHOLE_PROCESS_MAX_SECONDS,
    }
    if deadline != expected_deadline:
        raise M1IdentityDrift("config deadline vector drifted from the contract")
    checks += 7

    if config.get("reason_codes") is not None:
        if sorted(config["reason_codes"]) != sorted(ORIGIN_REASON_CODES):
            raise M1IdentityDrift("config reason_codes drifted from the contract")
        checks += 1
    if config.get("run_hard_stop_reasons") is not None:
        if sorted(config["run_hard_stop_reasons"]) != sorted(RUN_HARD_STOP_REASONS):
            raise M1IdentityDrift("config run_hard_stop_reasons drifted")
        checks += 1

    # ---- audit-3/H1: the REMAINING config fields are contract identity too ----
    if config.get("branch") != BRANCH:
        raise M1IdentityDrift("config branch drifted from the run branch")
    if config.get("base_branch") != BASE_BRANCH:
        raise M1IdentityDrift("config base_branch drifted")
    if config.get("base_sha") != BASE_SHA:
        raise M1IdentityDrift("config base_sha drifted")
    if config.get("run_provenance_commit") != RUN_PROVENANCE_COMMIT:
        raise M1IdentityDrift("config run_provenance_commit drifted")
    if config.get("evidence_root") != EVIDENCE_ROOT:
        raise M1IdentityDrift("config evidence_root drifted")
    if config.get("execution_scope") != EXECUTION_SCOPE:
        raise M1IdentityDrift("config execution_scope drifted")
    if config.get("interpreter") != INTERPRETER_PATH:
        raise M1IdentityDrift("config interpreter drifted")
    if config.get("status") != CONFIG_STATUS:
        raise M1IdentityDrift("config status drifted")
    if config.get("metrics_status") != METRICS_STATUS:
        raise M1IdentityDrift("config metrics_status drifted")
    if config.get("advance_to_g1_g2") is not ADVANCE_TO_G1_G2:
        raise M1IdentityDrift("config advance_to_g1_g2 drifted")
    if config.get("protocol_doc") != PROTOCOL_DOC:
        raise M1IdentityDrift("config protocol_doc drifted")
    if config.get("source_files") != list(SOURCE_FILE_LIST):
        raise M1IdentityDrift(
            "config source_files list drifted from the frozen contract surface")
    checks += 12

    if config.get("source_contract") != SOURCE_CONTRACT:
        raise M1IdentityDrift(
            "config source_contract drifted from the frozen source contract")
    checks += 1
    if config.get("arms") != ARMS_DECLARATION:
        raise M1IdentityDrift("config arms declaration drifted")
    checks += 1
    expected_diffs = {key: list(value) for key, value
                      in declared_arm_leaf_diffs().items()}
    if config.get("arm_leaf_whitelist_diffs") != expected_diffs:
        raise M1IdentityDrift("config arm_leaf_whitelist_diffs drifted")
    checks += 1

    if llm.get("endpoint") != LLM_ENDPOINT:
        raise M1IdentityDrift("config llm.endpoint drifted")
    if llm.get("input_token_endpoint") != LLM_INPUT_TOKEN_ENDPOINT:
        raise M1IdentityDrift("config llm.input_token_endpoint drifted")
    if llm.get("provider") != LLM_PROVIDER:
        raise M1IdentityDrift("config llm.provider drifted")
    if llm.get("retries") != LLM_RETRIES:
        raise M1IdentityDrift("config llm.retries drifted")
    if llm.get("sampling_note") != LLM_SAMPLING_NOTE:
        raise M1IdentityDrift("config llm.sampling_note drifted")
    checks += 5
    if config.get("llm_cost") != LLM_COST:
        raise M1IdentityDrift("config llm_cost drifted from the frozen rates")
    checks += 1
    if (envelopes.get("input_token_count_requests_max")
            != COMMON_BUDGET_VECTOR["A"]["llm_input_token_count_requests"]):
        raise M1IdentityDrift("config shared token-count-request cap drifted")
    if envelopes.get("shared_generation_calls_max") != ARM_LLM_GENERATIONS["D"]:
        raise M1IdentityDrift("config shared generation cap drifted")
    if envelopes.get("shared_input_tokens_max") != m1_budget.CAPS[
            m1_budget.BudgetClass.LLM_INPUT_TOKENS]:
        raise M1IdentityDrift("config shared input-token cap drifted")
    if envelopes.get("shared_output_tokens_max") != m1_budget.CAPS[
            m1_budget.BudgetClass.LLM_OUTPUT_TOKENS]:
        raise M1IdentityDrift("config shared output-token cap drifted")
    checks += 4
    if envelopes.get("C") != {"generation_calls": ARM_LLM_GENERATIONS["C"],
                              "input_tokens_max": PER_CALL_INPUT_TOKENS["C"],
                              "output_tokens_max": PER_CALL_OUTPUT_TOKENS["C"]}:
        raise M1IdentityDrift("config C envelope drifted")
    if envelopes.get("D") != {
            "generation_calls_max": ARM_LLM_GENERATIONS["D"],
            "input_tokens_max_per_call": PER_CALL_INPUT_TOKENS["D"],
            "output_tokens_max_per_call": PER_CALL_OUTPUT_TOKENS["D"]}:
        raise M1IdentityDrift("config D envelope drifted")
    checks += 2

    if config.get("forecast_tool") != FORECAST_TOOL_BINDING:
        raise M1IdentityDrift("config forecast_tool binding drifted")
    if FORECAST_TOOL_BINDING["calls_max"] != m1_budget.CAPS[
            m1_budget.BudgetClass.FORECAST_CALLS]:
        raise M1IdentityDrift("config forecast_tool calls_max drifted from the cap")
    checks += 2
    if config.get("prompt_artifacts", {}).get("directory") != PROMPT_DIRECTORY:
        raise M1IdentityDrift("config prompt_artifacts.directory drifted")
    if config.get("digest_renderer", {}).get("path") != RENDERER_PATH:
        raise M1IdentityDrift("config digest_renderer.path drifted")
    checks += 2

    for arm, vector in sorted(COMMON_BUDGET_VECTOR.items()):
        for key, budget_class in sorted(budget_key_to_class.items()):
            if key in vector and vector[key] != m1_budget.CAPS[budget_class]:
                raise M1IdentityDrift(
                    f"arm {arm} budget vector drifted from the ledger caps")
        checks += 1
    if ARM_LLM_GENERATIONS["C"] != config["llm_envelopes"]["C"]["generation_calls"]:
        raise M1IdentityDrift("config C generation calls drifted")
    if ARM_LLM_GENERATIONS["D"] != config["llm_envelopes"]["D"]["generation_calls_max"]:
        raise M1IdentityDrift("config D generation calls drifted")
    checks += 2
    return checks


def _verify_m0_inheritance(root: Path, config: dict) -> int:
    """The frozen_base checkpoint/packages must equal the pinned M0 config."""
    m0 = json.loads((root / "configs/agent_forecasting/"
                     "m0_predispatch_td_abs_c2_20261029.json").read_text())
    base = config["frozen_base"]
    checks = 0
    if base["checkpoint"]["revision"] != m0["checkpoint"]["revision"]:
        raise M1IdentityDrift("frozen_base checkpoint revision drifted from M0")
    m0_files = m0["checkpoint"]["sha256"]
    if (base["checkpoint"]["config_json_sha256"] != m0_files["config.json"]
            or base["checkpoint"]["model_safetensors_sha256"]
            != m0_files["model.safetensors"]):
        raise M1IdentityDrift(
            "frozen_base checkpoint file hashes drifted from M0")
    checks += 3
    if base["packages"] != m0["packages"]:
        raise M1IdentityDrift("frozen_base packages drifted from M0")
    if base.get("header_line_sha256") != M0_HEADER_LINE_SHA256:
        raise M1IdentityDrift("frozen_base header_line_sha256 drifted")
    return checks + 2


def verify_run_identity(repo_root: str | Path) -> dict:
    """Recompute every pin against the tree; refuse (before any action) on drift.

    Nothing is cached: every call re-hashes the pinned files, so a mutated tree
    at an already-verified path is detected immediately (audit-2: caching by
    mutable directory path is not an identity).
    """
    root = Path(repo_root)
    config = load_config(root)
    checked: dict[str, str] = {}
    if set(config.get("frozen_base", {}).get("files_sha256", {})) != set(
            M0_PINNED_FILES):
        raise M1IdentityDrift(
            "frozen_base.files_sha256 keyset != the inherited M0 surface")
    for relative in M0_PINNED_FILES:
        pinned = config["frozen_base"]["files_sha256"][relative]
        actual = sha256_file(root / relative)
        if actual != pinned:
            raise M1IdentityDrift(
                f"inherited M0 identity drift: {relative} {actual} != {pinned}")
        checked[relative] = actual
    if set(config.get("prompt_artifacts", {}).get("sha256", {})) != set(
            PROMPT_ARTIFACTS):
        raise M1IdentityDrift(
            "prompt_artifacts.sha256 keyset != the frozen prompt artifacts")
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
    _verify_execution_pins(root, config, checked)
    if config.get("contract_id") != CONTRACT_ID:
        raise M1IdentityDrift("config contract_id drifted")
    if config.get("candidate_id") != CANDIDATE_ID:
        raise M1IdentityDrift("config candidate_id drifted")
    field_checks = _verify_config_fields(config)
    field_checks += _verify_m0_inheritance(root, config)
    return {"status": "VERIFIED", "pinned_files": len(checked),
            "config_field_checks": field_checks, "contract_id": CONTRACT_ID}


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo_root), *args],
                          capture_output=True, text=True, check=False)


def verify_run_entry(repo_root: str | Path,
                     expected_head: str | None = None) -> dict:
    """The run ENTRY: bind git HEAD, a tracked-clean tree and the config file
    itself to the pinned identity BEFORE any action executes.

    Audit-3/H1: per-file pins alone are not a run identity.  The entry (a) runs
    at a resolvable git HEAD, optionally equal to an explicitly expected head
    (evidence replay binds the head its logs were produced at), (b) refuses any
    modification, staged change or tracked-file drift — untracked files cannot
    reach any pinned path — and (c) requires the frozen config file's working
    bytes to equal its committed bytes, so the config that closes the execution
    surface is itself closed by the commit it ships in.  Nothing is cached.
    """
    root = Path(repo_root)
    head = _git(root, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise M1IdentityDrift(
            f"run entry cannot resolve git HEAD under {root}: "
            f"{head.stderr.strip()}")
    git_head = head.stdout.strip()
    if expected_head is not None and git_head != expected_head:
        raise M1IdentityDrift(
            f"run entry HEAD drift: {git_head} != expected {expected_head}")
    status = _git(root, "status", "--porcelain")
    if status.returncode != 0:
        raise M1IdentityDrift("run entry cannot read git status")
    tracked_changes = [line for line in status.stdout.splitlines()
                       if line and not line.startswith("??")]
    if tracked_changes:
        raise M1IdentityDrift(
            "run entry tree is not clean over tracked files: "
            f"{tracked_changes[:3]}")
    config_binding = _git(root, "diff-index", "--quiet", "HEAD", "--", CONFIG_PATH)
    if config_binding.returncode != 0:
        raise M1IdentityDrift(
            "the frozen config file's bytes differ from its committed version")
    report = verify_run_identity(root)
    report["git_head"] = git_head
    report["tracked_tree_clean"] = True
    return report


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
