"""M1 as-of wrapper: strict receipt-chain verification and typed derivation.

This is the mandatory M1 wrapper over the frozen M0 adapter.  It (a) verifies the
append-only receipt-event hash chain before trusting any clock, (b) filters events
by receipt_complete_unix_ceil <= origin BEFORE any body is opened or parsed,
(c) materializes the visible prefix exactly once as an immutable tuple, and (d)
derives typed evidence whose identity depends only on the visible event ledger and
the selected raw body — never on the fetch subset or on post-origin receipts.

The kernel re-parses raw bytes itself: caller-supplied rows, values, hashes, basis
strings or hand-built series objects cannot enter the typed path.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from . import m1_contract
from .contracts import FrozenForecastInput, object_hash
from .predispatch_td_abs import (
    CANDIDATE_ID, COVARIATE_NAME, REAL_BASIS, REGION, STRICT_ASOF_VERIFIED,
    SYNTHETIC_STATUS, build_future_series, synthetic_receipt_from_raw_body,
)


class M1TypedRejection(Exception):
    def __init__(self, reason_code: str, detail: str = ""):
        super().__init__(f"typed path rejected: {reason_code} {detail}")
        self.reason_code = reason_code


class BaseBoundaryError(Exception):
    """The base input violates the candidate boundary (run hard stop)."""

    def __init__(self, reason_code: str, detail: str = ""):
        super().__init__(f"base boundary violation: {reason_code} {detail}")
        self.reason_code = reason_code


@dataclass(frozen=True)
class M1FetchRecord:
    """One in-run FETCH_REF result; raw_body is re-parsed by the kernel itself."""
    ref: str
    filename: str
    behaviour: str            # COMPLETE_VALID | INCOMPLETE | SCHEMA_DRIFT | TIMEOUT | NOT_FOUND
    receipt_complete_unix_ceil: int
    body_sha256: str
    raw_body: bytes | None


@dataclass(frozen=True)
class M1TypedEvidence:
    origin_unix: int
    values: tuple[float, ...]
    per_point: tuple[dict, ...]
    series_hash: str
    visible_ledger_hash: str
    selected: dict
    publication_basis: str
    strict_asof: str
    # the kernel derivation inputs: typed_input re-derives the tensor from these
    # raw bytes and rejects any evidence whose claims do not recompute exactly
    derivation_events: tuple = ()
    derivation_fetch: Optional["M1FetchRecord"] = None


def chain_events(events: Iterable[Mapping]) -> tuple[dict, ...]:
    """(Re)chain plain event records into the append-only hash chain."""
    prev = "0" * 64
    chained: list[dict] = []
    for seq, raw in enumerate(events):
        record = {key: value for key, value in dict(raw).items()
                  if key not in ("chain_seq", "prev_sha256", "event_sha256")}
        record["chain_seq"] = seq
        record["prev_sha256"] = prev
        record["event_sha256"] = hashlib.sha256(
            (json.dumps(record, sort_keys=True, separators=(",", ":")) + prev)
            .encode()).hexdigest()
        chained.append(record)
        prev = record["event_sha256"]
    return tuple(chained)


def verify_event_chain(events: Iterable[Mapping]) -> None:
    """Verify append-only sequencing and every event hash before any clock is used."""
    records = tuple(events)
    prev = "0" * 64
    seen_seq = -1
    for record in records:
        if record.get("event") != "body_complete":
            raise ValueError("receipt chain carries an unknown event kind")
        if record.get("chain_seq") != seen_seq + 1:
            raise ValueError("receipt chain is not append-only sequential")
        if record.get("prev_sha256") != prev:
            raise ValueError("receipt chain prev hash mismatch")
        payload = {key: value for key, value in record.items()
                   if key != "event_sha256"}
        expected = hashlib.sha256(
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + prev)
            .encode()).hexdigest()
        if record.get("event_sha256") != expected:
            raise ValueError("receipt event hash mismatch")
        digest = record.get("body_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("event body hash is malformed")
        filename = record.get("filename")
        if not isinstance(filename, str) or "/" in filename or "\\" in filename:
            raise ValueError("event filename is malformed")
        seen_seq += 1
        prev = record["event_sha256"]
    if seen_seq >= 0 and records[0]["chain_seq"] != 0:
        raise ValueError("receipt chain must start at seq 0")


def materialize_visible_events(events: Iterable[Mapping],
                               origin_unix: int) -> tuple[dict, ...]:
    """Filter by the receipt clock BEFORE any body is opened; materialize once.

    Accepts any iterable (tuple, list, generator): the sequence is consumed exactly
    once and the result is an immutable tuple shared by every later stage.
    """
    records = tuple(events)
    verify_event_chain(records)
    return tuple(record for record in records
                 if record["receipt_complete_unix_ceil"] <= origin_unix)


def visible_ledger_hash(visible_events: tuple[Mapping, ...]) -> str:
    return object_hash({
        "contract_id": m1_contract.CONTRACT_ID,
        "events": [dict(record) for record in visible_events],
    })


def _vintage_key(seqno: str, runno: str) -> tuple[str, int, int]:
    return (seqno[:8], int(seqno[8:10]), int(runno))


def _dedupe_visible(visible_events: tuple[dict, ...]) -> dict[str, dict]:
    """Fold duplicate bodies to the earliest receipt event, keyed by body hash."""
    earliest: dict[str, dict] = {}
    for record in visible_events:
        known = earliest.get(record["body_sha256"])
        if known is None or (record["receipt_complete_unix_ceil"], record["filename"]) < (
                known["receipt_complete_unix_ceil"], known["filename"]):
            earliest[record["body_sha256"]] = record
    return earliest


def _fork_check(visible_events: tuple[dict, ...]) -> None:
    bodies_per_vintage: dict[tuple[str, str], set[str]] = {}
    for record in visible_events:
        bodies_per_vintage.setdefault(
            (record["seqno"], record["runno"]), set()).add(record["body_sha256"])
    for vintage, bodies in bodies_per_vintage.items():
        if len(bodies) > 1:
            raise M1TypedRejection("FORK", f"vintage {vintage} has {len(bodies)} bodies")


def derive_typed_evidence(origin_unix: int,
                          visible_events: tuple[dict, ...],
                          fetch_records: Mapping[str, M1FetchRecord],
                          ) -> M1TypedEvidence:
    """Derive the 24-value typed tensor from raw bytes under newest-visible-only."""
    if any(record["receipt_complete_unix_ceil"] > origin_unix
           for record in visible_events):
        raise M1TypedRejection("RECEIPT_AFTER_ORIGIN",
                               "visible tuple must be pre-filtered at origin")
    if not visible_events:
        raise M1TypedRejection("NO_VISIBLE_RECEIPT", "no body-complete receipt at "
                               "or before origin")
    _fork_check(visible_events)
    ledger = visible_ledger_hash(visible_events)
    earliest = _dedupe_visible(visible_events)
    newest_event = max(
        earliest.values(),
        key=lambda record: _vintage_key(record["seqno"], record["runno"]))
    # resolve the in-run validation by BODY, not by ref: a duplicate republication
    # of the same body may have been fetched through either of its refs.  Among
    # the fetched mirrors of the newest body, any byte-validated fetch wins —
    # a dead mirror must not veto a live one — and the choice is deterministic
    # and independent of fetch order.
    candidates = sorted(
        (record for record in fetch_records.values()
         if record.body_sha256 == newest_event["body_sha256"]),
        key=lambda record: (record.ref, record.filename))
    if not candidates:
        raise M1TypedRejection(
            "ACTION_PRECONDITION",
            f"the newest visible vintage {newest_event['seqno']}/"
            f"{newest_event['runno']} was never fetched and validated in-run")
    byte_validated = [
        record for record in candidates
        if record.raw_body is not None and hashlib.sha256(
            record.raw_body).hexdigest() == newest_event["body_sha256"]]
    if byte_validated:
        fetch = byte_validated[0]
    elif any(record.behaviour == "TIMEOUT" for record in candidates):
        raise M1TypedRejection(
            "HTTP_TIMEOUT",
            min(record.ref for record in candidates
                if record.behaviour == "TIMEOUT"))
    elif any(record.behaviour == "NOT_FOUND" for record in candidates):
        raise M1TypedRejection(
            "HTTP_STATUS",
            min(record.ref for record in candidates
                if record.behaviour == "NOT_FOUND"))
    else:
        with_bytes = [record for record in candidates
                      if record.raw_body is not None]
        if not with_bytes:
            raise M1TypedRejection("SCHEMA_DRIFT", "no body bytes to validate")
        fetch = with_bytes[0]
    actual_digest = hashlib.sha256(fetch.raw_body).hexdigest()
    if actual_digest != newest_event["body_sha256"] or actual_digest != fetch.body_sha256:
        raise M1TypedRejection("BODY_HASH_MISMATCH", fetch.ref)
    try:
        # M1 exercises synthetic/protocol probe replay only: the honest basis is the
        # synthetic fixture builder and can never claim first-seen strictness.  The
        # basis is granted by the frozen builder from the raw bytes, never chosen.
        # The receipt is anchored at the EARLIEST VISIBLE receipt of the selected
        # body (never at the fetched mirror's own clock, which may be post-origin
        # for a late republication of the same pre-origin body).
        anchor = earliest[newest_event["body_sha256"]]
        receipt = synthetic_receipt_from_raw_body(
            fetch.raw_body, anchor["receipt_complete_unix_ceil"], anchor["filename"])
    except ValueError:
        raise M1TypedRejection(
            "SCHEMA_DRIFT", f"raw bytes of {fetch.ref} failed the frozen parser") from None
    if (receipt.body_sha256 != newest_event["body_sha256"]
            or receipt.seqno != newest_event["seqno"]
            or receipt.runno != newest_event["runno"]):
        raise M1TypedRejection("BODY_HASH_MISMATCH",
                               "fetched body does not match the newest visible event")
    try:
        series = build_future_series(origin_unix, [receipt])
    except ValueError as exc:
        message = str(exc)
        if "no fill" in message:
            raise M1TypedRejection(
                "INCOMPLETE_NEWEST_VINTAGE",
                "the newest visible vintage lacks required half-hour ends") from exc
        raise M1TypedRejection("SCHEMA_DRIFT", message) from exc
    identity = {
        "candidate_id": CANDIDATE_ID,
        "contract_id": m1_contract.CONTRACT_ID,
        "origin_unix": origin_unix,
        "covariate_name": COVARIATE_NAME,
        "future_values": list(series.future_values),
        "selected_seqno": receipt.seqno,
        "selected_runno": receipt.runno,
        "selected_body_sha256": receipt.body_sha256,
        "selected_header_line_sha256": receipt.header_line_sha256,
        "selected_receipt_complete_unix": earliest[receipt.body_sha256][
            "receipt_complete_unix_ceil"],
        "visible_ledger_hash": ledger,
    }
    return M1TypedEvidence(
        origin_unix=origin_unix,
        values=tuple(series.future_values),
        per_point=tuple(series.per_point),
        series_hash=object_hash(identity),
        visible_ledger_hash=ledger,
        selected={"seqno": receipt.seqno, "runno": receipt.runno,
                  "body_sha256": receipt.body_sha256,
                  "header_line_sha256": receipt.header_line_sha256,
                  "receipt_complete_unix": earliest[receipt.body_sha256][
                      "receipt_complete_unix_ceil"]},
        publication_basis=receipt.basis,
        strict_asof=(STRICT_ASOF_VERIFIED if receipt.basis == REAL_BASIS
                     else SYNTHETIC_STATUS),
        derivation_events=tuple(visible_events),
        derivation_fetch=fetch,
    )


def verify_typed_evidence(evidence: "M1TypedEvidence") -> "M1TypedEvidence":
    """Re-derive the tensor from the carried raw bytes; reject any mismatch.

    The Python/tool boundary cannot inject numbers: a hand-built or edited
    M1TypedEvidence must recompute every claimed field from its own derivation
    inputs (visible ledger + selected raw body) or be refused.
    """
    if not isinstance(evidence, M1TypedEvidence):
        raise BaseBoundaryError("CONTRACT_IDENTITY_DRIFT",
                                "typed evidence must be kernel evidence")
    if not evidence.derivation_events or evidence.derivation_fetch is None:
        raise BaseBoundaryError(
            "CONTRACT_IDENTITY_DRIFT",
            "typed evidence carries no kernel derivation inputs")
    rederived = derive_typed_evidence(
        evidence.origin_unix, tuple(evidence.derivation_events),
        {evidence.derivation_fetch.ref: evidence.derivation_fetch})
    for name in ("origin_unix", "values", "per_point", "series_hash",
                 "visible_ledger_hash", "selected", "publication_basis",
                 "strict_asof"):
        if getattr(rederived, name) != getattr(evidence, name):
            raise BaseBoundaryError(
                "CONTRACT_IDENTITY_DRIFT",
                f"typed evidence field {name} failed kernel re-derivation")
    return rederived


def typed_input(base: FrozenForecastInput,
                evidence: M1TypedEvidence) -> FrozenForecastInput:
    """Attach the kernel-derived tensor; the base boundary is enforced here."""
    verify_typed_evidence(evidence)
    if not isinstance(base, FrozenForecastInput):
        raise BaseBoundaryError("CONTRACT_IDENTITY_DRIFT", "base must be frozen input")
    if base.metadata.get("region") != REGION:
        raise BaseBoundaryError(
            "CONTRACT_IDENTITY_DRIFT",
            f"candidate region is {REGION}, base input carries "
            f"{base.metadata.get('region')!r}")
    if "source_manifest_sha256" in base.metadata:
        raise BaseBoundaryError("CONTRACT_IDENTITY_DRIFT",
                                "base input already carries a source manifest")
    if evidence.origin_unix != base.origin_unix:
        raise BaseBoundaryError("CONTRACT_IDENTITY_DRIFT", "origin mismatch")
    metadata = dict(base.metadata)
    metadata["source_manifest_sha256"] = evidence.series_hash
    metadata["input_profile"] = f"{metadata.get('input_profile', 'base')}__{CANDIDATE_ID}"
    return FrozenForecastInput(
        origin_unix=base.origin_unix,
        target_time_unix=base.target_time_unix,
        price_history=base.price_history,
        past_covariates=base.past_covariates,
        future_covariates=base.future_covariates,
        future_forecasts={COVARIATE_NAME: evidence.values},
        metadata=metadata,
    )
