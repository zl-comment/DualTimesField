"""Synthetic M1 protocol-probe harness (candidate 2, contract m1.agent_contract.c2.v1).

Builds plain-data source snapshots (listing, chained receipt events, bodies, replay
behaviours) for the bounded A/B/C/D enforcement probes and the label-free reducibility
reducer.  It produces ONLY plain dictionaries/tuples — no M1 module import — so the RED
commit can ship tests that fail per-test instead of erroring at collection.

All bodies are deterministic synthetic PREDISPATCHIS-format ZIP bytes from the frozen M0
fixture builder; no network, no labels, no scores, no randomness.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "fixtures" / "agent_forecasting"))

from predispatchis_td_v2_fixture import (  # type: ignore
    FIXTURE_ORIGIN_UNIX, build_body_bytes,
)

ORIGIN_UNIX = FIXTURE_ORIGIN_UNIX  # 1641218400, hour-aligned UTC
OLD_SEQNO = "2022010342"
NEW_SEQNO = "2022010344"
OLD_FILENAME = "SYNTHETIC_M1_PUBLIC_PREDISPATCHIS_202201032200_20220103220000.zip"
NEW_FILENAME = "SYNTHETIC_M1_PUBLIC_PREDISPATCHIS_202201032330_20220103233000.zip"
NEW_FILENAME_B = "SYNTHETIC_M1_PUBLIC_PREDISPATCHIS_202201032340_20220103234500.zip"

FETCH_OUTCOMES = ("COMPLETE_VALID", "INCOMPLETE", "SCHEMA_DRIFT", "TIMEOUT", "NOT_FOUND")
RELATIONS = ("BEFORE_ORIGIN", "EQUAL_ORIGIN", "AFTER_ORIGIN")
BODY_RELATIONS = ("DISTINCT_VINTAGES", "DUPLICATE_SAME_BODY", "FORK_SAME_VINTAGE")
LLM_RESULTS = ("VALID_ENABLED_ACTION", "INVALID_ACTION", "REFUSAL", "INCOMPLETE", "TIMEOUT")
LISTING_KINDS = ("EMPTY", "ONE_REF", "TWO_REFS")


def _rezip_text(text: str, *, member: str = "SYNTHETIC_PREDISPATCHIS_SECTION.csv") -> bytes:
    buffer = io.BytesIO()
    info = zipfile.ZipInfo(member, date_time=(2022, 1, 3, 23, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(info, text.encode())
    return buffer.getvalue()


def _body_text(body: bytes) -> str:
    archive = zipfile.ZipFile(io.BytesIO(body))
    text = archive.read(archive.namelist()[0]).decode()
    archive.close()
    return text


def _schema_drift_body(base_body: bytes) -> bytes:
    text = _body_text(base_body).replace(",DEMANDFORECAST,", ",DEMANDFORECAST2,", 1)
    return _rezip_text(text)


def opaque_ref(filename: str) -> str:
    return "v_" + hashlib.sha256(filename.encode()).hexdigest()[:16]


def _chain(events: list[dict]) -> tuple[dict, ...]:
    prev = "0" * 64
    chained = []
    for seq, event in enumerate(events):
        record = dict(event)
        record["chain_seq"] = seq
        record["prev_sha256"] = prev
        record["event_sha256"] = hashlib.sha256(
            (json.dumps(record, sort_keys=True, separators=(",", ":")) + prev).encode()
        ).hexdigest()
        chained.append(record)
        prev = record["event_sha256"]
    return tuple(chained)


def _relation_clock(relation: str) -> int:
    return {"BEFORE_ORIGIN": ORIGIN_UNIX - 600,
            "EQUAL_ORIGIN": ORIGIN_UNIX,
            "AFTER_ORIGIN": ORIGIN_UNIX + 600}[relation]


def build_body(*, seqno: str, outcome: str, fork_shift: bool = False) -> bytes:
    """One synthetic body realizing the requested fetch outcome."""
    if outcome == "SCHEMA_DRIFT":
        return _schema_drift_body(build_body_bytes(origin_unix=ORIGIN_UNIX, seqno=seqno))
    kwargs = {"origin_unix": ORIGIN_UNIX, "seqno": seqno}
    if outcome == "INCOMPLETE":
        kwargs["drop_half_hour_end"] = ORIGIN_UNIX + 5 * 1800
    if fork_shift:
        kwargs["shift_seconds"] = 1800
    return build_body_bytes(**kwargs)


def build_snapshot(*, listing: str, new_outcome: str, relation: str,
                   body_relation: str, old_outcome: str = "COMPLETE_VALID",
                   origin_unix: int = ORIGIN_UNIX) -> dict:
    """A plain-data immutable source snapshot for one reducer scenario.

    Refs whose replay behaviour is TIMEOUT/NOT_FOUND never produced a body-complete
    receipt, so they carry no receipt event and are invisible at any origin.
    """
    bodies: dict[str, bytes] = {}
    behaviours: dict[str, str] = {}
    events: list[dict] = []
    listing_refs: list[dict] = []

    def add_ref(filename: str, seqno: str, outcome: str, clock: int,
                body: bytes | None) -> None:
        ref = opaque_ref(filename)
        digest = hashlib.sha256(body).hexdigest() if body is not None else None
        listing_refs.append({"ref": ref, "filename": filename})
        behaviours[ref] = outcome
        if body is not None:
            bodies[digest] = body
            events.append({
                "event": "body_complete", "ref": ref, "filename": filename,
                "body_sha256": digest, "seqno": seqno, "runno": "1",
                "receipt_complete_unix_ceil": clock, "size_bytes": len(body),
                "clock_health": {"wall_monotonic_consistent": True},
            })

    if listing == "TWO_REFS":
        if body_relation == "DUPLICATE_SAME_BODY":
            # two filenames republishing one identical body: both refs realize the
            # same outcome because the bytes are the same
            body = build_body(seqno=NEW_SEQNO, outcome=new_outcome)
            add_ref(NEW_FILENAME, NEW_SEQNO, new_outcome, _relation_clock(relation), body)
            add_ref(NEW_FILENAME_B, NEW_SEQNO, new_outcome, ORIGIN_UNIX - 300, body)
        elif body_relation == "FORK_SAME_VINTAGE":
            add_ref(NEW_FILENAME, NEW_SEQNO, new_outcome, _relation_clock(relation),
                    build_body(seqno=NEW_SEQNO, outcome=new_outcome))
            add_ref(NEW_FILENAME_B, NEW_SEQNO, old_outcome, ORIGIN_UNIX - 3600,
                    build_body(seqno=NEW_SEQNO, outcome=old_outcome, fork_shift=True))
        else:  # DISTINCT_VINTAGES
            add_ref(OLD_FILENAME, OLD_SEQNO, old_outcome, ORIGIN_UNIX - 3600,
                    build_body(seqno=OLD_SEQNO, outcome=old_outcome))
            add_ref(NEW_FILENAME, NEW_SEQNO, new_outcome, _relation_clock(relation),
                    build_body(seqno=NEW_SEQNO, outcome=new_outcome))
    elif listing == "ONE_REF":
        add_ref(NEW_FILENAME, NEW_SEQNO, new_outcome, _relation_clock(relation),
                None if new_outcome in ("TIMEOUT", "NOT_FOUND")
                else build_body(seqno=NEW_SEQNO, outcome=new_outcome))
    elif listing != "EMPTY":
        raise ValueError(f"unknown listing kind {listing!r}")

    snapshot = {
        "origin_unix": origin_unix,
        "listing": tuple(listing_refs),
        "events": _chain(events),
        "bodies": dict(bodies),
        "fetch_behaviours": dict(behaviours),
        "scenario": {"listing": listing, "new_outcome": new_outcome,
                     "old_outcome": old_outcome, "relation": relation,
                     "body_relation": body_relation},
    }
    snapshot["snapshot_id"] = "snap_" + hashlib.sha256(
        json.dumps({key: snapshot[key] for key in
                    ("origin_unix", "listing", "events", "fetch_behaviours")},
                   sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    return snapshot


def scenario_space() -> list[dict]:
    """The exhaustive finite scenario cross for the reducer."""
    scenarios: list[dict] = []
    for listing in LISTING_KINDS:
        if listing == "EMPTY":
            scenarios.append(build_snapshot(
                listing=listing, new_outcome="COMPLETE_VALID",
                relation="BEFORE_ORIGIN", body_relation="DISTINCT_VINTAGES"))
            continue
        if listing == "ONE_REF":
            for outcome in FETCH_OUTCOMES:
                for relation in RELATIONS:
                    scenarios.append(build_snapshot(
                        listing=listing, new_outcome=outcome,
                        relation=relation, body_relation="DISTINCT_VINTAGES"))
            continue
        for body_relation in BODY_RELATIONS:
            pairs = ([(old, new) for old in FETCH_OUTCOMES for new in FETCH_OUTCOMES]
                     if body_relation != "DUPLICATE_SAME_BODY"
                     else [(outcome, outcome) for outcome in FETCH_OUTCOMES])
            for old_outcome, new_outcome in pairs:
                for relation in RELATIONS:
                    scenarios.append(build_snapshot(
                        listing=listing, new_outcome=new_outcome,
                        relation=relation, body_relation=body_relation,
                        old_outcome=old_outcome))
    return scenarios


def synthetic_base_input_payload(region: str = "NSW1") -> dict:
    """Plain payload for a deterministic frozen base input (no labels anywhere)."""
    return {
        "origin_unix": ORIGIN_UNIX,
        "target_time_unix": tuple(ORIGIN_UNIX + 3600 * h for h in range(24)),
        "price_history": tuple(float(v) for v in range(1, 73)),
        "metadata": {"region": region, "input_profile": "legacy_price_demand_calendar"},
    }


def llm_valid_d_action_bytes(action: str, ref: str) -> bytes:
    return json.dumps({"action": action, "ref": ref},
                      sort_keys=True, separators=(",", ":")).encode()


def llm_valid_c_decision_bytes(decision: str, evidence_ref: str) -> bytes:
    return json.dumps({"decision": decision, "evidence_ref": evidence_ref},
                      sort_keys=True, separators=(",", ":")).encode()


def llm_failure_response(kind: str) -> dict:
    """Scripted non-COMPLETED provider responses (status fields only)."""
    if kind == "REFUSAL":
        return {"status": "REFUSED", "refusal": True}
    if kind == "INCOMPLETE":
        return {"status": "INCOMPLETE", "incomplete_details": True}
    if kind == "TIMEOUT":
        return {"status": "TIMEOUT"}
    raise ValueError(kind)


def main() -> int:  # pragma: no cover - smoke helper
    space = scenario_space()
    print(f"scenario_space size={len(space)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
