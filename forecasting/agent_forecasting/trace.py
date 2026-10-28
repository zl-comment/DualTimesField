"""Append-only, locked, idempotent JSONL request events without data payloads.

Failures use a fixed reason code, not exception text that may contain inputs.
The chain detects edits relative to its retained tail; external manifest hashes
remain necessary to detect truncation or replacement of the entire log.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

from .contracts import canonical_json, identifier, object_hash, sha256_string


REASON_CODES = frozenset({"invalid_input", "unavailable_source", "revoked",
                          "conflicting_versions", "timeout", "memory_limit", "model_error",
                          "missing_base_input", "unverified_time", "no_numeric_path"})
_PAYLOAD_KEYS = frozenset({"request_id", "event", "input_hash", "output_hash", "reason"})
_RECORD_KEYS = _PAYLOAD_KEYS | {"sequence", "previous_hash", "record_hash"}


def _payload(request_id, event, input_hash, output_hash, reason) -> dict:
    identifier(request_id, "request_id")
    identifier(event, "event")
    sha256_string(input_hash, "input_hash")
    if output_hash is not None:
        sha256_string(output_hash, "output_hash")
    if reason is not None and reason not in REASON_CODES:
        raise ValueError("trace reason must be an approved code, not arbitrary text")
    return dict(request_id=request_id, event=event, input_hash=input_hash,
                output_hash=output_hash, reason=reason)


def _read_records(handle) -> list[dict]:
    handle.seek(0)
    records = []
    previous = "0" * 64
    identities = set()
    request_inputs = {}
    for raw in handle:
        if not raw.endswith("\n"):
            raise ValueError("incomplete trace line; refusing to append")
        try:
            record = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid trace JSON; refusing to append") from exc
        if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
            raise ValueError("unapproved trace schema")
        _payload(**{key: record[key] for key in _PAYLOAD_KEYS})
        identity = (record["request_id"], record["event"])
        if identity in identities:
            raise ValueError("duplicate trace identity")
        identities.add(identity)
        if record["request_id"] in request_inputs and request_inputs[record["request_id"]] != record["input_hash"]:
            raise ValueError("request input changed between trace events")
        request_inputs[record["request_id"]] = record["input_hash"]
        if type(record["sequence"]) is not int or record["sequence"] != len(records) or record["previous_hash"] != previous:
            raise ValueError("broken trace sequence")
        hashed = {k: v for k, v in record.items() if k != "record_hash"}
        if record["record_hash"] != object_hash(hashed):
            raise ValueError("broken trace hash")
        previous = record["record_hash"]
        records.append(record)
    return records


class AppendOnlyTrace:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, request_id: str, event: str, input_hash: str,
               output_hash: str | None = None, reason: str | None = None) -> dict:
        payload = _payload(request_id, event, input_hash, output_hash, reason)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            records = _read_records(handle)
            for record in records:
                if record["request_id"] == request_id and record["input_hash"] != input_hash:
                    raise ValueError("idempotency conflict: request input changed")
                if (record["request_id"], record["event"]) == (request_id, event):
                    if any(record[key] != value for key, value in payload.items()):
                        raise ValueError("idempotency conflict: request/event payload changed")
                    return dict(record)
            record = {**payload, "sequence": len(records),
                      "previous_hash": records[-1]["record_hash"] if records else "0" * 64}
            record["record_hash"] = object_hash(record)
            handle.seek(0, os.SEEK_END)
            handle.write(canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return record

    def read(self) -> tuple[dict, ...]:
        with self.path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            return tuple(_read_records(handle))
