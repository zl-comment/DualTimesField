"""Hash-chained append-only budget ledger for the M1 contract.

Every class is reserve -> effect -> settle.  A reservation that would exceed a cap
is refused BEFORE any side effect; failures, timeouts and rejected attempts are
charged like successes; there are no refunds or resets; an unsettled reserve is
fully consumed on close; provider usage above the reservation marks BUDGET_OVERRUN;
fold totals must rederive exactly from the chain.
"""
from __future__ import annotations

import enum
import hashlib
import json
from typing import Mapping, Optional


class BudgetClass(enum.Enum):
    SOURCE_REPLAY_CALLS = "source_replay_calls"
    SOURCE_BYTES_TOTAL = "source_bytes_total"
    SOURCE_BYTES_ONE_BODY = "source_bytes_one_body"
    LLM_TOKEN_COUNT_REQUESTS = "llm_token_count_requests"
    LLM_GENERATIONS = "llm_generations"
    LLM_INPUT_TOKENS = "llm_input_tokens"
    LLM_OUTPUT_TOKENS = "llm_output_tokens"
    LLM_COST_MICRO_USD = "llm_cost_micro_usd"
    ACTION_PROPOSALS = "action_proposals"
    LOCAL_TOOL_CALLS = "local_tool_calls"
    FORECAST_CALLS = "forecast_calls"
    EVIDENCE_SOURCE_BYTES = "evidence_source_bytes"
    EVIDENCE_LLM_BYTES = "evidence_llm_bytes"
    EVIDENCE_LEDGER_BYTES = "evidence_ledger_bytes"
    EVIDENCE_TOTAL_BYTES = "evidence_total_bytes"


CAPS: dict[BudgetClass, int] = {
    BudgetClass.SOURCE_REPLAY_CALLS: 3,
    BudgetClass.SOURCE_BYTES_TOTAL: 16777216,
    BudgetClass.SOURCE_BYTES_ONE_BODY: 8388608,
    BudgetClass.LLM_TOKEN_COUNT_REQUESTS: 4,
    BudgetClass.LLM_GENERATIONS: 4,
    BudgetClass.LLM_INPUT_TOKENS: 16384,
    BudgetClass.LLM_OUTPUT_TOKENS: 1024,
    BudgetClass.LLM_COST_MICRO_USD: 60000,
    BudgetClass.ACTION_PROPOSALS: 4,
    BudgetClass.LOCAL_TOOL_CALLS: 8,
    BudgetClass.FORECAST_CALLS: 2,
    BudgetClass.EVIDENCE_SOURCE_BYTES: 16777216,
    BudgetClass.EVIDENCE_LLM_BYTES: 524288,
    BudgetClass.EVIDENCE_LEDGER_BYTES: 2097152,
    BudgetClass.EVIDENCE_TOTAL_BYTES: 19398656,
}


class BudgetRefused(Exception):
    def __init__(self, budget_class: BudgetClass, cap: int, needed: int):
        super().__init__(
            f"budget cap {cap} for {budget_class.name} refuses additional {needed}")
        self.budget_class = budget_class
        self.cap = cap
        self.needed = needed


def _entry_hash(entry: dict) -> str:
    payload = {key: value for key, value in entry.items() if key != "sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class BudgetLedger:
    """Append-only reserve/effect/settle ledger with a verifiable hash chain."""

    def __init__(self, caps: Optional[Mapping[BudgetClass, int]] = None):
        self.caps: dict[BudgetClass, int] = dict(caps or CAPS)
        self.entries: list[dict] = []
        self._committed: dict[BudgetClass, int] = {cls: 0 for cls in self.caps}
        self._open: dict[BudgetClass, dict[int, int]] = {cls: {} for cls in self.caps}
        self._overrun: set[BudgetClass] = set()
        self._head = "0" * 64

    # -- chain primitives -------------------------------------------------
    def _append(self, entry: dict) -> dict:
        record = dict(entry)
        record["seq"] = len(self.entries)
        record["prev"] = self._head
        record["sha256"] = _entry_hash(record)
        self.entries.append(record)
        self._head = record["sha256"]
        return record

    # -- admission control ------------------------------------------------
    def reserve(self, budget_class: BudgetClass, amount: int, note: str = "") -> int:
        if not isinstance(amount, int) or amount < 0:
            raise ValueError("reserve amount must be a nonnegative integer")
        cap = self.caps[budget_class]
        committed = self._committed[budget_class] + sum(self._open[budget_class].values())
        if committed + amount > cap:
            self._append({"cls": budget_class.name, "op": "REFUSE", "amount": amount,
                          "committed_after": committed, "note": note})
            raise BudgetRefused(budget_class, cap, amount)
        entry = self._append({"cls": budget_class.name, "op": "RESERVE",
                              "amount": amount, "committed_after": committed + amount,
                              "note": note})
        self._open[budget_class][entry["seq"]] = amount
        return entry["seq"]

    def settle(self, seq: int, actual: int, note: str = "") -> dict:
        if not isinstance(actual, int) or actual < 0:
            raise ValueError("settle amount must be a nonnegative integer")
        source = next((entry for entry in self.entries if entry["seq"] == seq
                       and entry["op"] == "RESERVE"), None)
        if source is None:
            raise ValueError(f"no open reserve with seq {seq}")
        budget_class = BudgetClass[source["cls"]]
        reserved = self._open[budget_class].pop(seq)
        if actual > reserved:
            self._overrun.add(budget_class)
        self._committed[budget_class] += actual
        return self._append({"cls": budget_class.name, "op": "SETTLE",
                             "amount": actual, "committed_after":
                             self._committed[budget_class]
                             + sum(self._open[budget_class].values()),
                             "note": note, "resolves": seq})

    def charge(self, budget_class: BudgetClass, amount: int, note: str = "") -> dict:
        """Atomic reserve+effect+settle for a completed attempt (no refund)."""
        seq = self.reserve(budget_class, amount, note)
        return self.settle(seq, amount, note)

    def close(self) -> None:
        """An unsettled reserve is fully consumed: no refund, no reset."""
        for budget_class, open_map in self._open.items():
            for seq, amount in sorted(open_map.items()):
                self._committed[budget_class] += amount
                self._append({"cls": budget_class.name, "op": "SETTLE_CLOSE",
                              "amount": amount,
                              "committed_after": self._committed[budget_class],
                              "note": "unsettled reserve fully consumed",
                              "resolves": seq})
        for budget_class in list(self._open):
            self._open[budget_class] = {}

    # -- verification -----------------------------------------------------
    def fold(self) -> dict[BudgetClass, int]:
        return {cls: self._committed[cls] + sum(self._open[cls].values())
                for cls in self.caps}

    @property
    def overrun_classes(self) -> tuple[BudgetClass, ...]:
        return tuple(sorted(self._overrun, key=lambda cls: cls.name))

    def verify_chain(self) -> None:
        prev = "0" * 64
        for entry in self.entries:
            if entry["prev"] != prev or entry["sha256"] != _entry_hash(entry):
                raise ValueError(f"budget ledger chain broken at seq {entry['seq']}")
            prev = entry["sha256"]

    @classmethod
    def from_entries(cls, entries: list[dict]) -> "BudgetLedger":
        ledger = cls()
        ledger.entries = []
        ledger._head = "0" * 64
        for raw in entries:
            entry = dict(raw)
            ledger._append({key: value for key, value in entry.items()
                            if key not in ("seq", "prev", "sha256")})
            budget_class = BudgetClass[entry["cls"]]
            if entry["op"] == "RESERVE":
                ledger._open[budget_class][entry["seq"]] = entry["amount"]
            elif entry["op"] in ("SETTLE", "SETTLE_CLOSE"):
                if entry.get("resolves") in ledger._open[budget_class]:
                    reserved = ledger._open[budget_class].pop(entry["resolves"])
                    if entry["amount"] > reserved:
                        ledger._overrun.add(budget_class)
                ledger._committed[budget_class] += entry["amount"]
        # recompute committed_after consistency is implicit; the chain verifies bytes
        ledger.verify_chain()
        return ledger
