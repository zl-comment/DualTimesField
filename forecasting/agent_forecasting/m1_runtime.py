"""M1 runtime: the A/B/C/D executors over one shared immutable snapshot replay.

Every numeric decision is made by the deterministic safety kernel; the LLM (when
present) only selects among typed bounded actions.  Budgets are reserve->effect->
settle, retries are zero everywhere, deadlines route to the exact A fallback or a
SYSTEM_MISSING record, and evidence publication is atomic.  M1 runs bounded
synthetic/protocol probes only: no labels, no scores, no live LLM, no live source.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional

from . import m1_actions, m1_asof, m1_budget, m1_contract, m1_digest, m1_llm
from .contracts import FrozenForecastInput
from .frozen_chronos import build_model_input
from .m1_actions import M1ActionKind, M1State
from .m1_asof import M1FetchRecord
from .m1_budget import BudgetClass, BudgetLedger
from .m1_publication import publish_run, write_failure_record

# deadline reserves and per-call envelopes are contract identity (m1_contract);
# re-exported here for the executors
DEADLINE_OFFSET = m1_contract.DEADLINE_OFFSET
TYPED_RESERVE = m1_contract.TYPED_RESERVE
PUBLICATION_RESERVE = m1_contract.PUBLICATION_RESERVE
D_PER_CALL_INPUT_TOKENS = m1_contract.D_PER_CALL_INPUT_TOKENS
D_PER_CALL_OUTPUT_TOKENS = m1_contract.D_PER_CALL_OUTPUT_TOKENS
C_PER_CALL_INPUT_TOKENS = m1_contract.C_PER_CALL_INPUT_TOKENS
C_PER_CALL_OUTPUT_TOKENS = m1_contract.C_PER_CALL_OUTPUT_TOKENS
_FETCHABLE_BEHAVIOURS = ("COMPLETE_VALID", "INCOMPLETE", "SCHEMA_DRIFT")


class ForecastReuseForbidden(Exception):
    """Each unique model input may be forecast at most once."""


class DeterministicClock:
    """Injected wall clock: every call advances by ``step`` seconds."""

    def __init__(self, start: float, step: float = 1.0):
        self._now = float(start)
        self._step = float(step)

    def __call__(self) -> float:
        current = self._now
        self._now += self._step
        return current


def _normalize(value):
    if hasattr(value, "tolist"):
        return _normalize(value.tolist())
    if isinstance(value, float):
        return "<NaN>" if math.isnan(value) else ("<Inf>" if math.isinf(value) else value)
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    return str(value)


def canonical_model_input_json(model_input: Mapping, params: Mapping | None = None) -> str:
    """Arm-independent canonical JSON of the numeric model input (NaN as a token)."""
    payload = {"model_input": _normalize(model_input)}
    if params is not None:
        payload["params"] = _normalize(params)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def model_input_sha256(model_input: Mapping, params: Mapping | None = None) -> str:
    return hashlib.sha256(
        canonical_model_input_json(model_input, params).encode()).hexdigest()


class DeterministicForecastStub:
    """Probe-only frozen tool: deterministic, deliberately non-monotone columns.

    Output bytes are a function of the canonical input only, so identical inputs
    give byte-identical outputs across arms.  The real frozen Chronos-2 path stays
    identity-pinned via the config and is NOT executed in M1 probes.
    """

    identity = {"kind": "deterministic_probe_stub", "version": 1, "quantiles": 5,
                "horizon": 24, "sorting": "none", "clipping": "none",
                "calibration": "none"}

    def __init__(self):
        self.last_output: bytes | None = None

    def predict(self, model_input: Mapping, params: Mapping | None = None) -> bytes:
        digest = model_input_sha256(model_input, params)
        rows = []
        for hour in range(24):
            row = [40.0 + (int(hashlib.sha256(
                f"{digest}:{hour}:{column}".encode()).hexdigest()[:8], 16) % 800) / 10.0
                for column in range(5)]
            if hour % 2 == 0:  # deliberate non-monotone ordering that must survive
                row = [row[2], row[0], row[4], row[1], row[3]]
            rows.append(row)
        output = json.dumps({"input_digest": digest, "quantiles": rows},
                            sort_keys=True, separators=(",", ":")).encode()
        self.last_output = output
        return output


class TimeoutForecastTool:
    identity = {"kind": "timeout_probe_stub", "version": 1}

    def predict(self, model_input: Mapping, params: Mapping | None = None) -> bytes:
        raise TimeoutError("forecast tool timed out")


class ForecastSink:
    """The single forecast sink: at most one call per unique input, traced."""

    def __init__(self, tool):
        self.tool = tool
        self._outputs: dict[str, bytes] = {}
        self.call_count = 0
        self.trace: list[dict] = []

    def forecast_once(self, model_input: Mapping, params: Mapping | None = None,
                      *, phase: str = "base",
                      budget: Optional[BudgetLedger] = None) -> bytes:
        key = model_input_sha256(model_input, params)
        if key in self._outputs:
            raise ForecastReuseForbidden(f"input {key} was already forecast")
        seq = budget.reserve(BudgetClass.FORECAST_CALLS, 1, phase) if budget else None
        output = self.tool.predict(model_input, params)
        if seq is not None:
            budget.settle(seq, 1, phase)
        self._outputs[key] = output
        self.call_count += 1
        self.trace.append({
            "phase": phase, "input_hash": key,
            "params_hash": hashlib.sha256(json.dumps(
                _normalize(params or {}), sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(),
            "output_sha256": hashlib.sha256(output).hexdigest(),
            "output_bytes": len(output),
        })
        return output


class ScriptedTransport:
    """Probe transport popping prepared responses; counts every call."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls = 0

    def generate(self, arm: str, turn: int, request_bytes: bytes):
        self.calls += 1
        if not self._responses:
            raise RuntimeError("scripted transport exhausted")
        return self._responses.pop(0)


class SimulatedUsageTransport:
    """Valid C decision but provider usage above the reserved envelope."""

    def __init__(self, decision: str, input_tokens: int, output_tokens: int):
        self._decision = decision
        self._input = int(input_tokens)
        self._output = int(output_tokens)
        self.calls = 0

    def generate(self, arm: str, turn: int, request_bytes: bytes):
        self.calls += 1
        return m1_llm.M1LlmResponse(
            status="COMPLETED", model=m1_contract.LLM_MODEL,
            output_raw=json.dumps({"decision": self._decision,
                                   "evidence_ref": "NONE"},
                                  sort_keys=True, separators=(",", ":")).encode(),
            usage_input_tokens=self._input, usage_output_tokens=self._output)


class M1SourceSnapshot:
    """Immutable read-only replay of one captured source state; no live access."""

    def __init__(self, *, snapshot_id: str, origin_unix: int, listing: tuple,
                 events: tuple, bodies: Mapping[str, bytes],
                 fetch_behaviours: Mapping[str, str]):
        self.snapshot_id = snapshot_id
        self.origin_unix = int(origin_unix)
        self.listing = tuple(listing)
        self.events = tuple(events)
        self._bodies = dict(bodies)
        self._behaviours = dict(fetch_behaviours)
        self._event_by_ref = {event["ref"]: event for event in self.events}

    @classmethod
    def from_mapping(cls, data: Mapping) -> "M1SourceSnapshot":
        return cls(snapshot_id=data["snapshot_id"], origin_unix=data["origin_unix"],
                   listing=tuple(data["listing"]), events=tuple(data["events"]),
                   bodies=data["bodies"], fetch_behaviours=data["fetch_behaviours"])

    def filename_for(self, ref: str) -> str:
        return next(entry["filename"] for entry in self.listing
                    if entry["ref"] == ref)

    def fetch(self, ref: str) -> tuple[str, Optional[bytes]]:
        behaviour = self._behaviours[ref]
        if behaviour not in _FETCHABLE_BEHAVIOURS:
            return behaviour, None
        event = self._event_by_ref.get(ref)
        if event is None:
            return behaviour, None
        return behaviour, self._bodies[event["body_sha256"]]

    def event_for(self, ref: str) -> Optional[dict]:
        return self._event_by_ref.get(ref)

    def visible_events(self, origin_unix: int) -> tuple:
        return m1_asof.materialize_visible_events(self.events, origin_unix)

    def canonical_fetch_refs(self, origin_unix: int) -> tuple[str, ...]:
        """The B/C fixed maximal plan over the two allowed body GETs.

        Mirrors of the newest VISIBLE body come first (earliest receipt, then
        filename, then canonical listing order): one dead mirror must not waste
        the plan's second GET.  When the newest body has fewer than two visible
        mirrors, the remaining GET is spent on the next listed ref in canonical
        listing order — a listed ref whose own receipt may be post-origin never
        contributes identity, but its byte-identical body may still validate.
        Never consults post-origin receipts for planning.
        """
        visible = self.visible_events(origin_unix)
        if not visible:
            return ()
        order = {entry["ref"]: index for index, entry in enumerate(self.listing)}
        earliest = {}
        for record in visible:
            known = earliest.get(record["body_sha256"])
            if known is None or (record["receipt_complete_unix_ceil"],
                                 record["filename"]) < (
                    known["receipt_complete_unix_ceil"], known["filename"]):
                earliest[record["body_sha256"]] = record
        newest = max(earliest.values(), key=lambda record: (
            record["seqno"][:8], int(record["seqno"][8:10]), int(record["runno"])))
        mirrors = sorted(
            (record for record in visible
             if record["body_sha256"] == newest["body_sha256"]),
            key=lambda record: (record["receipt_complete_unix_ceil"],
                                record["filename"],
                                order.get(record["ref"], 1 << 30)))
        plan = [record["ref"] for record in mirrors[:2]]
        if len(plan) < 2:
            for entry in self.listing:
                if entry["ref"] not in plan:
                    plan.append(entry["ref"])
                    if len(plan) == 2:
                        break
        return tuple(plan)


@dataclass
class M1RunResult:
    arm: str
    verdict: str
    reason_code: str
    origin_unix: int
    cutoff: int
    snapshot_id: str
    input_hash: str
    series_hash: Optional[str]
    typed_values: Optional[tuple[float, ...]]
    forecast_bytes: bytes
    tool_calls: int
    tool_trace_hash: str
    llm_calls: int
    retrieval_trace: tuple
    action_trace: tuple
    digests: tuple
    budget_fold: dict
    budget_entries: list
    published: bool
    publish_dir: Optional[str]
    source_replay_calls: int
    selected_seqno: Optional[str]
    canonical_model_input_json: str
    typed_input_object: Optional[FrozenForecastInput] = None
    identity: dict = field(default_factory=dict)

    @property
    def forecast_sha256(self) -> str:
        return hashlib.sha256(self.forecast_bytes).hexdigest()


class _OverrunMarker(Exception):
    pass


def _budget_facts(budget: BudgetLedger, clock_now: float, deadline: float) -> dict:
    fold = budget.fold()

    def level(budget_class):
        return ("EXHAUSTED" if fold[budget_class] >= budget.caps[budget_class]
                else "AVAILABLE")

    return {
        "source_replay": level(BudgetClass.SOURCE_REPLAY_CALLS),
        "llm_generation": level(BudgetClass.LLM_GENERATIONS),
        "llm_tokens": "EXHAUSTED" if (
            fold[BudgetClass.LLM_INPUT_TOKENS] >= budget.caps[
                BudgetClass.LLM_INPUT_TOKENS]
            or fold[BudgetClass.LLM_OUTPUT_TOKENS] >= budget.caps[
                BudgetClass.LLM_OUTPUT_TOKENS]) else "AVAILABLE",
        "forecast_calls": level(BudgetClass.FORECAST_CALLS),
        "wall_clock": "EXHAUSTED" if clock_now > deadline else "AVAILABLE",
        "evidence_bytes": level(BudgetClass.EVIDENCE_TOTAL_BYTES),
    }


def _per_ref_facts(snapshot: M1SourceSnapshot, origin: int,
                   fetched: Mapping[str, M1FetchRecord]) -> tuple[dict, dict, dict]:
    relation, coverage, schema = {}, {}, {}
    for entry in snapshot.listing:
        ref = entry["ref"]
        event = snapshot.event_for(ref)
        if event is None:
            relation[ref] = "AFTER_CUTOFF"  # no body-complete receipt: never visible
        else:
            clock = event["receipt_complete_unix_ceil"]
            relation[ref] = ("BEFORE_CUTOFF" if clock < origin else
                             "EQUAL_CUTOFF" if clock == origin else "AFTER_CUTOFF")
        record = fetched.get(ref)
        if record is None:
            coverage[ref] = "NOT_FETCHED"
            schema[ref] = "NOT_FETCHED"
        else:
            coverage[ref] = ("COMPLETE" if record.behaviour == "COMPLETE_VALID"
                             else "INCOMPLETE")
            schema[ref] = ("VALID" if record.behaviour in
                           ("COMPLETE_VALID", "INCOMPLETE") else "REJECTED")
    return relation, coverage, schema


def _fork_present(snapshot: M1SourceSnapshot, origin: int) -> str:
    per_vintage: dict[tuple, set] = {}
    try:
        visible = snapshot.visible_events(origin)
    except ValueError:
        return "UNKNOWN"  # a broken receipt chain cannot support a fork claim
    for record in visible:
        per_vintage.setdefault((record["seqno"], record["runno"]), set()).add(
            record["body_sha256"])
    return ("PRESENT" if any(len(bodies) > 1 for bodies in per_vintage.values())
            else "ABSENT")


def _render_digest(*, arm: str, state: M1State, snapshot: M1SourceSnapshot,
                   origin: int, fetched: Mapping[str, M1FetchRecord], list_calls: int,
                   get_calls: int, typed_ready: bool, evidence_refs: tuple,
                   budget: BudgetLedger, clock_now: float, deadline: float) -> str:
    available = (tuple(entry["ref"] for entry in snapshot.listing
                       if entry["ref"] not in fetched) if get_calls < 2 else ())
    relation, coverage, schema = _per_ref_facts(snapshot, origin, fetched)
    return m1_digest.render_digest({
        "contract": m1_contract.CONTRACT_ID,
        "arm": arm,
        "state": state.name,
        "enabled_actions": tuple(action.name for action in m1_actions.enabled_actions(
            state, list_calls=list_calls, get_calls=get_calls,
            available_refs=available, fetched_refs=tuple(fetched))),
        "enabled_refs": available,
        "budget": _budget_facts(budget, clock_now, deadline),
        "receipt_cutoff_relation": relation,
        "coverage": coverage,
        "schema": schema,
        "fork": _fork_present(snapshot, origin),
        "typed_ready": "YES" if typed_ready else "NO",
        "evidence_refs": evidence_refs,
    })


def _wire_entry(arm: str, turn: int, request: bytes, response) -> dict:
    """The archived LLM wire record: raw request/response bytes are the evidence."""
    return {
        "arm": arm, "turn": turn,
        "request_sha256": hashlib.sha256(request).hexdigest(),
        "request_len": len(request),
        "request_b64": base64.b64encode(request).decode(),
        "response_sha256": hashlib.sha256(response.output_raw).hexdigest(),
        "response_len": len(response.output_raw),
        "response_b64": base64.b64encode(response.output_raw).decode(),
        "status": response.status, "model": response.model,
        "usage_input_tokens": response.usage_input_tokens,
        "usage_output_tokens": response.usage_output_tokens,
    }


def _derive_or_none(origin: int, snapshot: M1SourceSnapshot,
                    fetched: Mapping[str, M1FetchRecord]) -> tuple:
    try:
        return (m1_asof.derive_typed_evidence(origin, snapshot.visible_events(origin),
                                              fetched), None)
    except m1_asof.M1TypedRejection as rejection:
        return None, rejection.reason_code
    except ValueError:
        # a receipt chain that fails verification is a frozen rejection, never
        # an escaping exception
        return None, "SCHEMA_DRIFT"


def _reserve_llm_call(budget: BudgetLedger, *, input_cap: int, output_cap: int,
                      request_len: int) -> dict:
    """Reserve the whole LLM envelope BEFORE the transport is touched.

    Refusal here means the provider call never happens (fail closed before the
    effect); the evidence-byte reservation uses the worst case (request plus the
    raw output cap) so even the bytes a response could add are pre-admitted.
    """
    seqs = {
        "input": budget.reserve(BudgetClass.LLM_INPUT_TOKENS, input_cap,
                                "llm input"),
        "output": budget.reserve(BudgetClass.LLM_OUTPUT_TOKENS, output_cap,
                                 "llm output"),
        "ev_llm": budget.reserve(
            BudgetClass.EVIDENCE_LLM_BYTES,
            request_len + m1_contract.LLM_RAW_OUTPUT_CAP_BYTES, "llm evidence"),
        "ev_total": budget.reserve(
            BudgetClass.EVIDENCE_TOTAL_BYTES,
            request_len + m1_contract.LLM_RAW_OUTPUT_CAP_BYTES,
            "llm evidence total"),
    }
    budget.charge(BudgetClass.LLM_TOKEN_COUNT_REQUESTS, 1, "input_tokens endpoint")
    budget.charge(BudgetClass.LLM_GENERATIONS, 1, "generation")
    budget.charge(BudgetClass.ACTION_PROPOSALS, 1, "proposal")
    return seqs


def _settle_llm_call(budget: BudgetLedger, seqs: dict, *, request: bytes,
                     response) -> None:
    """Settle actuals after the transport; an unsettled reserve (timeout raised
    by the transport) is fully consumed at close, so failures are billed."""
    budget.settle(seqs["input"], response.usage_input_tokens, "llm input actual")
    budget.settle(seqs["output"], response.usage_output_tokens, "llm output actual")
    budget.charge(BudgetClass.LLM_COST_MICRO_USD,
                  m1_llm.cost_micro_usd(response.usage_input_tokens,
                                        response.usage_output_tokens), "llm cost")
    evidence_bytes = len(request) + len(response.output_raw)
    budget.settle(seqs["ev_llm"], evidence_bytes, "llm evidence")
    budget.settle(seqs["ev_total"], evidence_bytes, "llm evidence total")
    if budget.overrun_classes:
        raise _OverrunMarker()


def run_arm(arm: str, snapshot_data: Mapping, *, base_input: FrozenForecastInput,
            tool=None, transport=None, clock: Optional[Callable[[], float]] = None,
            publish_root: Optional[Path] = None,
            repo_root: Optional[str | Path] = None,
            budget: Optional[BudgetLedger] = None) -> M1RunResult:
    """Execute one arm over the shared snapshot; every origin lands in a verdict."""
    if arm not in ("A", "B", "C", "D"):
        raise ValueError("arm must be one of A/B/C/D")
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    identity_report = m1_contract.verify_run_identity(repo_root)
    snapshot = M1SourceSnapshot.from_mapping(snapshot_data)
    origin = snapshot.origin_unix
    deadline = origin + DEADLINE_OFFSET
    budget = budget if budget is not None else BudgetLedger()
    the_clock = clock if clock is not None else DeterministicClock(start=origin,
                                                                   step=1.0)
    sink = ForecastSink(tool if tool is not None else DeterministicForecastStub())
    if arm in ("C", "D") and transport is None:
        raise ValueError(f"arm {arm} requires an LLM transport")
    if arm in ("A", "B") and transport is not None:
        raise ValueError(f"arm {arm} must not receive an LLM transport")

    run_id = f"{arm}_{origin}_{snapshot.snapshot_id}"
    action_trace: list[dict] = []
    retrieval_trace: list[dict] = []
    digests: list[str] = []
    fetched: dict[str, M1FetchRecord] = {}
    llm_wire: list[dict] = []

    def _fetch(ref: str) -> None:
        # reserve BEFORE the replay read; the per-body byte cap is checked per
        # single body (standalone), never accumulated across bodies
        event = snapshot.event_for(ref)
        expected_size = event["size_bytes"] if event else 0
        seq_body = budget.reserve(BudgetClass.SOURCE_BYTES_ONE_BODY, expected_size,
                                  f"body:{ref}", standalone=True)
        seq_total = budget.reserve(BudgetClass.SOURCE_BYTES_TOTAL, expected_size,
                                   f"body:{ref}")
        seq_call = budget.reserve(BudgetClass.SOURCE_REPLAY_CALLS, 1, f"fetch:{ref}")
        behaviour, body = snapshot.fetch(ref)  # the effect, after every reserve
        actual = len(body) if body is not None else 0
        budget.settle(seq_body, actual, f"body:{ref}")
        budget.settle(seq_total, actual, f"body:{ref}")
        budget.settle(seq_call, 1, f"fetch:{ref}")
        fetched[ref] = M1FetchRecord(
            ref=ref, filename=snapshot.filename_for(ref), behaviour=behaviour,
            receipt_complete_unix_ceil=(event["receipt_complete_unix_ceil"]
                                        if event else 0),
            body_sha256=(event["body_sha256"] if event else "0" * 64),
            raw_body=body)
        retrieval_trace.append({"step": "FETCH_REF", "ref": ref,
                                "filename": fetched[ref].filename,
                                "outcome": behaviour,
                                "body_sha256": fetched[ref].body_sha256})

    def finish(verdict: str, reason: str, *, forecast_bytes: bytes,
               canonical_json: str, input_hash: Optional[str] = None,
               series_hash: str | None = None,
               typed_values: tuple | None = None,
               typed_input: FrozenForecastInput | None = None) -> M1RunResult:
        budget.close()
        deadline_ok = the_clock() <= deadline
        if not deadline_ok and verdict != "SYSTEM_MISSING":
            verdict, reason = "SYSTEM_MISSING", "DEADLINE_MISSED"
        try:
            budget.verify_chain()
            budget_chain_ok = True
        except ValueError:
            budget_chain_ok = False
        result = M1RunResult(
            arm=arm, verdict=verdict, reason_code=reason, origin_unix=origin,
            cutoff=origin, snapshot_id=snapshot.snapshot_id,
            input_hash=input_hash if input_hash is not None else "",
            series_hash=series_hash, typed_values=typed_values,
            forecast_bytes=forecast_bytes, tool_calls=sink.call_count,
            tool_trace_hash=hashlib.sha256(json.dumps(
                sink.trace, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(),
            llm_calls=transport.calls if transport is not None else 0,
            retrieval_trace=tuple(retrieval_trace),
            action_trace=tuple(action_trace), digests=tuple(digests),
            budget_fold=budget.fold(), budget_entries=[dict(entry)
                                                       for entry in budget.entries],
            published=False, publish_dir=None,
            source_replay_calls=budget.fold()[BudgetClass.SOURCE_REPLAY_CALLS],
            selected_seqno=None, canonical_model_input_json=canonical_json,
            typed_input_object=typed_input,
            identity={"contract": identity_report["contract_id"], "arm": arm,
                      "origin_unix": origin, "snapshot_id": snapshot.snapshot_id})
        record = {"arm": arm, "verdict": verdict, "reason_code": reason,
                  "origin_unix": origin, "snapshot_id": snapshot.snapshot_id,
                  "input_hash": result.input_hash, "series_hash": series_hash,
                  "tool_trace_hash": result.tool_trace_hash,
                  "llm_calls": result.llm_calls, "deadline_ok": deadline_ok,
                  "source_replay_calls": result.source_replay_calls}

        def _failed(reason_code: str) -> M1RunResult:
            result.verdict = "TERMINAL_FAILED_NO_COMMIT"
            result.reason_code = reason_code
            record["verdict"], record["reason_code"] = result.verdict, reason_code
            if publish_root is not None:
                try:
                    write_failure_record(publish_root, run_id, record)
                except Exception:  # noqa: BLE001 - the run already failed closed
                    pass
            return result

        if verdict == "TERMINAL_FAILED_NO_COMMIT":
            if publish_root is not None:
                try:
                    write_failure_record(publish_root, run_id, record)
                except Exception:  # noqa: BLE001 - the run already failed closed
                    pass
            return result
        payloads = {
            "result.json": json.dumps(record, sort_keys=True,
                                      separators=(",", ":")).encode(),
            "action_ledger.jsonl": ("\n".join(
                json.dumps(entry, sort_keys=True, separators=(",", ":"))
                for entry in action_trace).encode() + (b"\n" if action_trace else b"")),
            "budget_ledger.jsonl": ("\n".join(
                json.dumps(entry, sort_keys=True, separators=(",", ":"))
                for entry in budget.entries).encode() + (b"\n" if budget.entries
                                                         else b"")),
            "model_ledger.jsonl": json.dumps(
                {"tool": getattr(sink.tool, "identity", {}), "trace": sink.trace,
                 "llm_calls": result.llm_calls},
                sort_keys=True, separators=(",", ":")).encode(),
        }
        # arm A never touches the snapshot: no receipt-chain parse, no as-of view
        if arm != "A":
            try:
                visible = [dict(event) for event in
                           snapshot.visible_events(origin)]
                chain_verified = True
            except ValueError:
                visible, chain_verified = [], False
            payloads["asof_snapshot.json"] = json.dumps(
                {"snapshot_id": snapshot.snapshot_id, "origin_unix": origin,
                 "receipt_chain_verified": chain_verified,
                 "visible_events": visible},
                sort_keys=True, separators=(",", ":")).encode()
        if llm_wire:
            payloads["llm_wire.jsonl"] = ("\n".join(
                json.dumps(entry, sort_keys=True, separators=(",", ":"))
                for entry in llm_wire).encode() + b"\n")
        if digests:
            payloads["digests.jsonl"] = ("\n".join(
                json.dumps({"digest_sha256": hashlib.sha256(digest.encode())
                            .hexdigest(), "digest": digest}
                           , sort_keys=True, separators=(",", ":"))
                for digest in digests).encode() + b"\n")
        if forecast_bytes:
            payloads["forecast.bin"] = forecast_bytes
        if publish_root is not None:
            closure = {
                "origin_unix": origin,
                "deadline_unix": deadline,
                "clock_final": the_clock(),
                "budget_fold": {cls.name: value for cls, value in
                                budget.fold().items()},
                "budget_caps": {cls.name: value for cls, value in
                                budget.caps.items()},
                "budget_chain_ok": budget_chain_ok,
                "tool_identity_sha256": hashlib.sha256(json.dumps(
                    getattr(sink.tool, "identity", {}), sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest(),
                "input_hash": result.input_hash,
                "canonical_model_input_sha256": hashlib.sha256(
                    canonical_json.encode()).hexdigest(),
            }
            try:
                final = publish_run(publish_root, run_id, payloads,
                                    identity=result.identity, closure=closure)
            except Exception:  # noqa: BLE001 - publication failure fails closed
                return _failed("EVIDENCE_PUBLICATION_FAILED")
            result.published = True
            result.publish_dir = str(final)
        return result

    # ---- phase 0: the deadline gate --------------------------------------------
    if the_clock() > deadline:
        return finish("SYSTEM_MISSING", "DEADLINE_MISSED", forecast_bytes=b"",
                      canonical_json="")

    # ---- phase A: the frozen base forecast (the only numeric fallback) ---------
    base_model_input = build_model_input(base_input)
    base_canonical = canonical_model_input_json(base_model_input)
    base_hash = model_input_sha256(base_model_input)
    try:
        base_bytes = sink.forecast_once(base_model_input, phase="base", budget=budget)
    except m1_budget.BudgetRefused:
        return finish("SYSTEM_MISSING", "BUDGET_EXHAUSTED", forecast_bytes=b"",
                      canonical_json=base_canonical, input_hash=base_hash)
    except (TimeoutError, ForecastReuseForbidden):
        return finish("SYSTEM_MISSING", "FORECAST_TOOL_FAILURE", forecast_bytes=b"",
                      canonical_json=base_canonical, input_hash=base_hash)
    except Exception:  # noqa: BLE001 - a broken tool is a system miss, never a leak
        return finish("SYSTEM_MISSING", "FORECAST_TOOL_FAILURE", forecast_bytes=b"",
                      canonical_json=base_canonical, input_hash=base_hash)

    def base_terminal(reason: str) -> M1RunResult:
        return finish("PUBLISHED_BASE_FALLBACK", reason, forecast_bytes=base_bytes,
                      canonical_json=base_canonical, input_hash=base_hash)

    def typed_terminal(evidence) -> M1RunResult:
        try:
            typed_request = m1_asof.typed_input(base_input, evidence)
        except m1_asof.BaseBoundaryError as error:
            return finish("TERMINAL_FAILED_NO_COMMIT", error.reason_code,
                          forecast_bytes=b"", canonical_json="",
                          input_hash=base_hash)
        typed_model_input = build_model_input(typed_request)
        typed_hash = model_input_sha256(typed_model_input)
        try:
            typed_bytes = sink.forecast_once(typed_model_input, phase="typed",
                                             budget=budget)
        except m1_budget.BudgetRefused:
            return base_terminal("BUDGET_OVERRUN" if budget.overrun_classes
                                 else "BUDGET_EXHAUSTED")
        except (TimeoutError, ForecastReuseForbidden):
            return base_terminal("FORECAST_TOOL_FAILURE")
        except Exception:  # noqa: BLE001 - a broken tool falls back, never leaks
            return base_terminal("FORECAST_TOOL_FAILURE")
        result = finish("PUBLISHED_TYPED", "PUBLISHED_TYPED",
                        forecast_bytes=typed_bytes,
                        canonical_json=canonical_model_input_json(typed_model_input),
                        input_hash=typed_hash, series_hash=evidence.series_hash,
                        typed_values=evidence.values, typed_input=typed_request)
        result.selected_seqno = evidence.selected["seqno"]
        return result

    if arm == "A":
        return finish("PUBLISHED_BASE_FALLBACK", "AGENT_ABSTAIN",
                      forecast_bytes=base_bytes, canonical_json=base_canonical,
                      input_hash=base_hash)

    if arm in ("B", "C"):
        # ---- the fixed maximal retrieval plan (byte-identical for B and C) ----
        try:
            plan = snapshot.canonical_fetch_refs(origin)
        except ValueError:
            # the receipt chain failed verification: a frozen rejection, no escape
            return base_terminal("SCHEMA_DRIFT")
        try:
            budget.charge(BudgetClass.SOURCE_REPLAY_CALLS, 1, "list")
        except m1_budget.BudgetRefused:
            return base_terminal("BUDGET_OVERRUN" if budget.overrun_classes
                                 else "BUDGET_EXHAUSTED")
        retrieval_trace.append({"step": "LIST", "snapshot_id": snapshot.snapshot_id,
                                "refs": tuple(entry["ref"]
                                              for entry in snapshot.listing)})
        action_trace.append({"arm": arm, "action": "LIST", "ref": "NONE",
                             "mode": "fixed_plan"})
        for ref in plan:
            try:
                _fetch(ref)
            except m1_budget.BudgetRefused:
                action_trace.append({"arm": arm, "action": "FETCH_REF", "ref": ref,
                                     "rejected": "BUDGET_EXHAUSTED"})
                return base_terminal("BUDGET_OVERRUN" if budget.overrun_classes
                                     else "BUDGET_EXHAUSTED")
            action_trace.append({"arm": arm, "action": "FETCH_REF", "ref": ref,
                                 "mode": "fixed_plan"})
        evidence, rejection_reason = _derive_or_none(origin, snapshot, fetched)
        if evidence is None:
            return base_terminal(rejection_reason)
        if arm == "B":
            if the_clock() + TYPED_RESERVE + PUBLICATION_RESERVE > deadline:
                return base_terminal("DEADLINE_MISSED")
            return typed_terminal(evidence)
        # ---- arm C: exactly one frozen LLM decision, no follow-up --------------
        if the_clock() + PUBLICATION_RESERVE > deadline:
            return base_terminal("DEADLINE_MISSED")
        digest = _render_digest(
            arm="C", state=M1State.FETCHED, snapshot=snapshot, origin=origin,
            fetched=fetched, list_calls=1, get_calls=len(fetched), typed_ready=True,
            evidence_refs=(f"ev_{evidence.series_hash[:16]}",), budget=budget,
            clock_now=the_clock(), deadline=deadline)
        digests.append(digest)
        request = m1_llm.build_request_bytes("C", digest, repo_root)
        try:
            seqs = _reserve_llm_call(budget, input_cap=C_PER_CALL_INPUT_TOKENS,
                                     output_cap=C_PER_CALL_OUTPUT_TOKENS,
                                     request_len=len(request))
        except m1_budget.BudgetRefused:
            return base_terminal("BUDGET_OVERRUN" if budget.overrun_classes
                                 else "BUDGET_EXHAUSTED")
        try:
            response = transport.generate("C", 1, request)
        except TimeoutError:
            return base_terminal("LLM_TIMEOUT")  # reserves stay consumed at close
        llm_wire.append(_wire_entry("C", 1, request, response))
        try:
            _settle_llm_call(budget, seqs, request=request, response=response)
        except _OverrunMarker:
            return base_terminal("BUDGET_OVERRUN")
        except m1_budget.BudgetRefused:
            return base_terminal("BUDGET_OVERRUN" if budget.overrun_classes
                                 else "BUDGET_EXHAUSTED")
        try:
            decision, _ref = m1_llm.parse_c_decision(
                response, enabled_evidence_refs=(f"ev_{evidence.series_hash[:16]}",))
        except m1_llm.LlmParseError as error:
            return base_terminal(error.reason_code)
        if decision != "USE_TYPED":
            return base_terminal("AGENT_ABSTAIN")
        if the_clock() + TYPED_RESERVE + PUBLICATION_RESERVE > deadline:
            return base_terminal("DEADLINE_MISSED")
        return typed_terminal(evidence)

    # ---- arm D: bounded typed action proposals ---------------------------------
    list_calls, get_calls = 0, 0
    typed_cache: dict[tuple, tuple] = {}

    def typed_state() -> tuple:
        key = tuple(sorted(fetched))
        if key not in typed_cache:
            typed_cache[key] = _derive_or_none(origin, snapshot, fetched)
        return typed_cache[key]

    for turn in range(1, 5):
        if the_clock() + PUBLICATION_RESERVE > deadline:
            return base_terminal("DEADLINE_MISSED")
        evidence, rejection_reason = typed_state()
        state = (M1State.FETCHED if fetched else
                 M1State.LISTED if list_calls else M1State.A_READY)
        available = (tuple(entry["ref"] for entry in snapshot.listing
                           if entry["ref"] not in fetched) if get_calls < 2 else ())
        digest = _render_digest(
            arm="D", state=state, snapshot=snapshot, origin=origin, fetched=fetched,
            list_calls=list_calls, get_calls=get_calls,
            typed_ready=evidence is not None, evidence_refs=(), budget=budget,
            clock_now=the_clock(), deadline=deadline)
        digests.append(digest)
        request = m1_llm.build_request_bytes("D", digest, repo_root)
        try:
            seqs = _reserve_llm_call(budget, input_cap=D_PER_CALL_INPUT_TOKENS,
                                     output_cap=D_PER_CALL_OUTPUT_TOKENS,
                                     request_len=len(request))
        except m1_budget.BudgetRefused:
            return base_terminal("BUDGET_OVERRUN" if budget.overrun_classes
                                 else "BUDGET_EXHAUSTED")
        try:
            response = transport.generate("D", turn, request)
        except TimeoutError:
            return base_terminal("LLM_TIMEOUT")  # reserves stay consumed at close
        llm_wire.append(_wire_entry("D", turn, request, response))
        try:
            _settle_llm_call(budget, seqs, request=request, response=response)
        except _OverrunMarker:
            return base_terminal("BUDGET_OVERRUN")
        except m1_budget.BudgetRefused:
            return base_terminal("BUDGET_OVERRUN" if budget.overrun_classes
                                 else "BUDGET_EXHAUSTED")
        try:
            kind, ref = m1_llm.parse_d_action(
                response, enabled_refs=tuple(entry["ref"]
                                             for entry in snapshot.listing))
        except m1_llm.LlmParseError as error:
            return base_terminal(error.reason_code)
        try:
            m1_actions.validate_proposal(
                state, kind, ref, list_calls=list_calls, get_calls=get_calls,
                available_refs=available, fetched_refs=tuple(fetched))
        except m1_actions.ProposalRejected as rejection:
            action_trace.append({"arm": "D", "action": kind.name, "ref": ref,
                                 "rejected": rejection.reason_code})
            return base_terminal(rejection.reason_code)
        action_trace.append({"arm": "D", "action": kind.name, "ref": ref,
                             "mode": "agent", "turn": turn})
        if kind is M1ActionKind.LIST:
            list_calls += 1
            try:
                budget.charge(BudgetClass.SOURCE_REPLAY_CALLS, 1, "agent list")
            except m1_budget.BudgetRefused:
                return base_terminal("BUDGET_OVERRUN" if budget.overrun_classes
                                     else "BUDGET_EXHAUSTED")
            retrieval_trace.append({"step": "LIST", "snapshot_id": snapshot.snapshot_id,
                                    "refs": tuple(entry["ref"]
                                                  for entry in snapshot.listing)})
        elif kind is M1ActionKind.FETCH_REF:
            get_calls += 1
            try:
                _fetch(ref)
            except m1_budget.BudgetRefused:
                action_trace.append({"arm": "D", "action": kind.name, "ref": ref,
                                     "rejected": "BUDGET_EXHAUSTED"})
                return base_terminal("BUDGET_OVERRUN" if budget.overrun_classes
                                     else "BUDGET_EXHAUSTED")
        elif kind is M1ActionKind.FINALIZE_TYPED:
            if evidence is None:
                return base_terminal(rejection_reason or "ACTION_PRECONDITION")
            if the_clock() + TYPED_RESERVE + PUBLICATION_RESERVE > deadline:
                return base_terminal("DEADLINE_MISSED")
            return typed_terminal(evidence)
        else:  # FINALIZE_BASE
            return base_terminal("AGENT_ABSTAIN")
    return base_terminal("BUDGET_EXHAUSTED")
