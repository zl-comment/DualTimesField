"""Frozen candidate-2 M0 adapter: AEMO PREDISPATCHIS NSW1 TOTALDEMAND.

Candidate ``predispatchis_absolute_totaldemand_future_v1`` maps one exact
PREDISPATCHIS vintage to 24 hourly future TOTALDEMAND values in MW for a
declared origin.  Every number is re-derived from immutable raw ZIP bytes;
callers cannot submit rows, values, a body hash or a basis string.  The
module never touches the network, never reads labels, never scores, never
calls a model.

Frozen source identity (discovered 2026-10-29 from authenticated raw bytes
of PUBLIC_PREDISPATCHIS_202610291730_20261029170208.zip and
PUBLIC_PREDISPATCHIS_202610291800_20261029173204.zip; both bodies carry the
identical REGION_SOLUTION header):

* table ``PREDISPATCH,REGION_SOLUTION,9``, 131 CSV fields per I-row,
  127 data column names, header-line SHA-256 ``2f013ba4...``
* field ``TOTALDEMAND`` (MW), region NSW1, ``INTERVENTION=0``
* one vintage per body: exactly one (PREDISPATCHSEQNO, RUNNO)
* DATETIME is AEST (UTC+10, no DST): UTC = wall - 10h; the value is the
  period END on the half-hour grid
* target hour h in [0,24) is the delivery hour [origin+h, origin+(h+1));
  its value is the exact arithmetic mean of the two half-hour rows ending
  at origin+(2h+1)*1800 and origin+(2h+2)*1800 — no splice, no fallback,
  no fill

Visibility uses only the local body-complete receipt clock (wall clock read
in the same process at stream EOF, ceiled to the next whole second so the
recorded instant is never earlier than the true EOF).  File name timestamps,
HTTP Date/Last-Modified and the LASTCHANGED column are declared metadata
and never clocks.

Companion protocol: docs/research/agent_forecasting/m0_predispatch_td_abs_protocol_20261029.txt
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .contracts import (
    FrozenForecastInput, identifier, object_hash, sha256_string, utc_unix,
)

CANDIDATE_ID = "predispatchis_absolute_totaldemand_future_v1"
SOURCE_URL = "https://www.nemweb.com.au/REPORTS/CURRENT/PredispatchIS_Reports/"
TABLE_ID = "PREDISPATCH,REGION_SOLUTION,9"
TABLE_PARTS = ("PREDISPATCH", "REGION_SOLUTION", "9")
TABLE_VERSION = "9"
FIELD_NAME = "TOTALDEMAND"
UNIT = "MW"
REGION = "NSW1"
INTERVENTION = 0
COVARIATE_NAME = "predispatchis_totaldemand_mw"
AEST_OFFSET_SECONDS = 10 * 3600
AEMO_TIME_FORMAT = "%Y/%m/%d %H:%M:%S"
HOUR_SECONDS = 3600
HALF_HOUR_SECONDS = 1800
FUTURE_POSITIONS = 24
HEADER_TOTAL_FIELDS = 131
HEADER_LINE_SHA256 = "2f013ba453d6ef05959d74895fa6db6f9cd2522b6c0f7d4356af09fadaaa529a"
REQUIRED_COLUMNS = ("PREDISPATCHSEQNO", "RUNNO", "REGIONID", "PERIODID",
                    "INTERVENTION", "TOTALDEMAND", "DATETIME")
END_OF_REPORT = "END OF REPORT"
FILE_PREAMBLE_IDENTITY = ("NEMP.WORLD", "PREDISPATCHIS", "AEMO", "PUBLIC")
RECEIPT_BASES = frozenset({"first_seen_receipt", "synthetic_fixture"})
REAL_BASIS = "first_seen_receipt"
SYNTHETIC_BASIS = "synthetic_fixture"
STRICT_ASOF_VERIFIED = "STRICT_ASOF_VERIFIED"
SYNTHETIC_STATUS = "SYNTHETIC_FIXTURE"
_SEQNO_RE = re.compile(r"\d{10}\Z")
_RUNNO_RE = re.compile(r"\d{1,3}\Z")

_RAW_BODY_TRUST = object()

HEADER_COLUMNS = (
    "PREDISPATCHSEQNO", "RUNNO", "REGIONID", "PERIODID", "INTERVENTION",
    "TOTALDEMAND", "AVAILABLEGENERATION", "AVAILABLELOAD", "DEMANDFORECAST",
    "DISPATCHABLEGENERATION", "DISPATCHABLELOAD", "NETINTERCHANGE",
    "EXCESSGENERATION", "LOWER5MINDISPATCH", "LOWER5MINIMPORT",
    "LOWER5MINLOCALDISPATCH", "LOWER5MINLOCALPRICE", "LOWER5MINLOCALREQ",
    "LOWER5MINPRICE", "LOWER5MINREQ", "LOWER5MINSUPPLYPRICE",
    "LOWER60SECDISPATCH", "LOWER60SECIMPORT", "LOWER60SECLOCALDISPATCH",
    "LOWER60SECLOCALPRICE", "LOWER60SECLOCALREQ", "LOWER60SECPRICE",
    "LOWER60SECREQ", "LOWER60SECSUPPLYPRICE", "LOWER6SECDISPATCH",
    "LOWER6SECIMPORT", "LOWER6SECLOCALDISPATCH", "LOWER6SECLOCALPRICE",
    "LOWER6SECLOCALREQ", "LOWER6SECPRICE", "LOWER6SECREQ",
    "LOWER6SECSUPPLYPRICE", "RAISE5MINDISPATCH", "RAISE5MINIMPORT",
    "RAISE5MINLOCALDISPATCH", "RAISE5MINLOCALPRICE", "RAISE5MINLOCALREQ",
    "RAISE5MINPRICE", "RAISE5MINREQ", "RAISE5MINSUPPLYPRICE",
    "RAISE60SECDISPATCH", "RAISE60SECIMPORT", "RAISE60SECLOCALDISPATCH",
    "RAISE60SECLOCALPRICE", "RAISE60SECLOCALREQ", "RAISE60SECPRICE",
    "RAISE60SECREQ", "RAISE60SECSUPPLYPRICE", "RAISE6SECDISPATCH",
    "RAISE6SECIMPORT", "RAISE6SECLOCALDISPATCH", "RAISE6SECLOCALPRICE",
    "RAISE6SECLOCALREQ", "RAISE6SECPRICE", "RAISE6SECREQ",
    "RAISE6SECSUPPLYPRICE", "LASTCHANGED", "DATETIME", "INITIALSUPPLY",
    "CLEAREDSUPPLY", "LOWERREGIMPORT", "LOWERREGLOCALDISPATCH",
    "LOWERREGLOCALREQ", "LOWERREGREQ", "RAISEREGIMPORT",
    "RAISEREGLOCALDISPATCH", "RAISEREGLOCALREQ", "RAISEREGREQ",
    "RAISE5MINLOCALVIOLATION", "RAISEREGLOCALVIOLATION",
    "RAISE60SECLOCALVIOLATION", "RAISE6SECLOCALVIOLATION",
    "LOWER5MINLOCALVIOLATION", "LOWERREGLOCALVIOLATION",
    "LOWER60SECLOCALVIOLATION", "LOWER6SECLOCALVIOLATION",
    "RAISE5MINVIOLATION", "RAISEREGVIOLATION", "RAISE60SECVIOLATION",
    "RAISE6SECVIOLATION", "LOWER5MINVIOLATION", "LOWERREGVIOLATION",
    "LOWER60SECVIOLATION", "LOWER6SECVIOLATION", "RAISE6SECACTUALAVAILABILITY",
    "RAISE60SECACTUALAVAILABILITY", "RAISE5MINACTUALAVAILABILITY",
    "RAISEREGACTUALAVAILABILITY", "LOWER6SECACTUALAVAILABILITY",
    "LOWER60SECACTUALAVAILABILITY", "LOWER5MINACTUALAVAILABILITY",
    "LOWERREGACTUALAVAILABILITY", "DECAVAILABILITY", "LORSURPLUS",
    "LRCSURPLUS", "TOTALINTERMITTENTGENERATION", "DEMAND_AND_NONSCHEDGEN",
    "UIGF", "SEMISCHEDULE_CLEAREDMW", "SEMISCHEDULE_COMPLIANCEMW",
    "SS_SOLAR_UIGF", "SS_WIND_UIGF", "SS_SOLAR_CLEAREDMW", "SS_WIND_CLEAREDMW",
    "SS_SOLAR_COMPLIANCEMW", "SS_WIND_COMPLIANCEMW", "WDR_INITIALMW",
    "WDR_AVAILABLE", "WDR_DISPATCHED", "LOWER1SECLOCALDISPATCH",
    "RAISE1SECLOCALDISPATCH", "RAISE1SECACTUALAVAILABILITY",
    "LOWER1SECACTUALAVAILABILITY", "SS_SOLAR_AVAILABILITY",
    "SS_WIND_AVAILABILITY", "BDU_ENERGY_STORAGE", "BDU_MIN_AVAIL",
    "BDU_MAX_AVAIL", "BDU_CLEAREDMW_GEN", "BDU_CLEAREDMW_LOAD",
    "BDU_INITIAL_ENERGY_STORAGE", "DECGEN_INITIAL_ENERGY_STORAGE",
)


def parse_aest(value: str) -> int:
    """Parse an AEMO AEST wall timestamp to integral UTC Unix seconds."""
    if not isinstance(value, str):
        raise ValueError("AEST timestamp must be a string")
    try:
        naive = datetime.strptime(value.strip(), AEMO_TIME_FORMAT)
    except ValueError as exc:
        raise ValueError(f"invalid AEMO timestamp {value!r}") from exc
    # AEST is UTC+10 with no DST: UTC = wall - 10h.
    return utc_unix(int(naive.replace(tzinfo=timezone.utc).timestamp()) - AEST_OFFSET_SECONDS)


def _vintage_key(seqno: str, runno: str) -> tuple[str, int, int]:
    return (seqno[:8], int(seqno[8:10]), int(runno))


@dataclass(frozen=True, slots=True)
class PredispatchBody:
    """Immutable parse result of one published PREDISPATCHIS ZIP body."""
    seqno: str
    runno: str
    body_sha256: str
    header_line_sha256: str
    rows: tuple[tuple[int, float], ...]


def parse_body(raw_body: bytes) -> PredispatchBody:
    """Strictly parse and authenticate one PREDISPATCHIS ZIP body.

    Fail-closed structure discovered from the authenticated raw bytes: the
    CSV opens with a ``C,NEMP.WORLD,PREDISPATCHIS,AEMO,PUBLIC,...`` preamble
    row, then carries I/D sections for several tables (each D row repeats
    its table identity in fields 1-3), and closes with a single
    ``C,END OF REPORT,<N>`` trailer whose N equals the total row count of
    the file.  The ``PREDISPATCH,REGION_SOLUTION,9`` I-row appears exactly
    once with the frozen 131-field header (127 pinned column names plus the
    header-line hash); its D rows match the header width and the current
    section identity; the (seqno, runno, regionid, intervention, datetime)
    primary key never repeats; the whole table carries exactly one
    (PREDISPATCHSEQNO, RUNNO); scope rows are NSW1 with INTERVENTION=0 and
    every TOTALDEMAND cell is a finite number on the half-hour DATETIME
    grid.  Anything else is rejected — no lenient column matching, no
    version migration, no value repair.
    """
    if not isinstance(raw_body, bytes) or not raw_body:
        raise ValueError("raw body must be nonempty bytes")
    body_sha256 = hashlib.sha256(raw_body).hexdigest()
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw_body))
    except zipfile.BadZipFile as exc:
        raise ValueError("raw body is not the published PREDISPATCHIS zip") from exc
    try:
        members = archive.namelist()
        if len(members) != 1 or not members[0].lower().endswith(".csv"):
            raise ValueError("raw body must contain exactly one CSV ZIP member")
        text = archive.read(members[0]).decode("utf-8-sig")
    finally:
        archive.close()
    rows = [row for row in csv.reader(io.StringIO(text, newline="")) if row]
    if len(rows) < 3:
        raise ValueError("table file is too small to be the published report")
    preamble = [cell.strip() for cell in rows[0]]
    if (len(rows[0]) != 10 or tuple(preamble[1:5]) != FILE_PREAMBLE_IDENTITY
            or preamble[8] != "PREDISPATCHIS"):
        raise ValueError("table file must open with the PREDISPATCHIS preamble row")
    trailer = [cell.strip() for cell in rows[-1]]
    if len(rows[-1]) != 3 or trailer[:2] != ["C", END_OF_REPORT] or trailer[2] != str(len(rows)):
        raise ValueError(
            f"table file must end with '{END_OF_REPORT}' carrying the total row "
            f"count; got {rows[-1]!r}")

    current_section: tuple[str, ...] | None = None
    target_index: dict[str, int] | None = None
    target_seen = False
    data_count = 0
    primary_keys: set[tuple[str, str, str, str, str]] = set()
    seqnos: set[str] = set()
    runnos: set[str] = set()
    scope: list[tuple[int, float]] = []
    for row in rows[1:-1]:
        kind = row[0].strip()
        if kind == "C":
            raise ValueError("a control row inside the data area is rejected")
        if kind == "I":
            current_section = tuple(part.strip() for part in row[1:4])
            if current_section == TABLE_PARTS:
                if target_seen:
                    raise ValueError("the REGION_SOLUTION section appears twice")
                if len(row) != HEADER_TOTAL_FIELDS:
                    raise ValueError(
                        f"header field count {len(row)} is not the frozen "
                        f"{HEADER_TOTAL_FIELDS}")
                if tuple(part.strip() for part in row[4:]) != HEADER_COLUMNS:
                    raise ValueError("header column names drifted from the frozen schema")
                header_line = ",".join(row)
                if hashlib.sha256(header_line.encode()).hexdigest() != HEADER_LINE_SHA256:
                    raise ValueError("header line hash drifted from the frozen schema")
                target_index = {name.strip(): pos for pos, name in enumerate(row)}
                missing = [name for name in REQUIRED_COLUMNS if name not in target_index]
                if missing:
                    raise ValueError(f"raw body is missing required columns {missing}")
                target_seen = True
            continue
        if kind != "D":
            raise ValueError(f"unknown row kind {kind!r} in the published table")
        if len(row) < 4:
            raise ValueError("a data row shorter than its identity prefix is rejected")
        identity = tuple(part.strip() for part in row[1:4])
        if identity == TABLE_PARTS or current_section == TABLE_PARTS:
            if identity != TABLE_PARTS or current_section != TABLE_PARTS:
                raise ValueError(
                    f"a REGION_SOLUTION data row outside its own section is rejected: {identity}")
            assert target_index is not None
            if len(row) != HEADER_TOTAL_FIELDS:
                raise ValueError(
                    f"data row width {len(row)} does not match the header width "
                    f"{HEADER_TOTAL_FIELDS}")
            seqno = row[target_index["PREDISPATCHSEQNO"]].strip()
            runno = row[target_index["RUNNO"]].strip()
            regionid = row[target_index["REGIONID"]].strip()
            intervention = row[target_index["INTERVENTION"]].strip()
            datetime_cell = row[target_index["DATETIME"]].strip()
            if not _SEQNO_RE.fullmatch(seqno) or not _RUNNO_RE.fullmatch(runno):
                raise ValueError(f"malformed vintage identity ({seqno!r}, {runno!r})")
            key = (seqno, runno, regionid, intervention, datetime_cell)
            if key in primary_keys:
                raise ValueError(f"duplicate primary key {key}; the published table cannot repeat a row")
            primary_keys.add(key)
            seqnos.add(seqno)
            runnos.add(runno)
            data_count += 1
            if regionid != REGION or intervention != str(INTERVENTION):
                continue
            demand = row[target_index["TOTALDEMAND"]].strip()
            if not demand:
                raise ValueError("a TOTALDEMAND cell is empty; no value fill is permitted")
            try:
                value = float(demand)
            except ValueError as exc:
                raise ValueError(f"TOTALDEMAND {demand!r} is not a number") from exc
            if not isfinite(value):
                raise ValueError("TOTALDEMAND must be finite")
            scope.append((parse_aest(datetime_cell), value))
    if not target_seen:
        raise ValueError(f"body carries no {TABLE_ID} section")
    if not scope:
        raise ValueError(f"body carries no {REGION} INTERVENTION={INTERVENTION} rows")
    if len(seqnos) != 1 or len(runnos) != 1:
        raise ValueError(
            f"a raw body must carry exactly one vintage; found {sorted(seqnos)} x {sorted(runnos)}")
    scope_rows = tuple(sorted(scope, key=lambda item: item[0]))
    previous = None
    for end, _value in scope_rows:
        if end % HALF_HOUR_SECONDS:
            raise ValueError("DATETIME values must sit on the half-hour grid")
        if previous is not None and end <= previous:
            raise ValueError("scope DATETIME values must be unique and strictly increasing")
        previous = end
    return PredispatchBody(
        seqno=seqnos.pop(),
        runno=runnos.pop(),
        body_sha256=body_sha256,
        header_line_sha256=HEADER_LINE_SHA256,
        rows=scope_rows,
    )


@dataclass(frozen=True, slots=True)
class PredispatchTdReceipt:
    """One forward-captured PREDISPATCHIS vintage, bound to its raw bytes.

    ``receipt_complete_unix`` is the local body-complete instant (wall clock
    read at stream EOF, ceiled to the next whole second) and the only
    visibility clock.  Every receipt — synthetic fixture or first-seen — is
    raw-body-bound: construction re-parses the immutable zip bytes and
    rejects any drift between the claimed rows/hashes/vintage and the bytes.
    ``basis`` cannot be chosen by callers or flipped via ``replace``: it is
    granted together with the raw-body trust token by the two builders.
    """

    seqno: str
    runno: str
    receipt_complete_unix: int
    body_sha256: str
    header_line_sha256: str
    filename: str
    table_id: str
    region: str
    intervention: int
    unit: str
    basis: str
    rows: tuple[tuple[int, float], ...]
    raw_body: bytes = field(default=None, compare=False, repr=False)  # type: ignore[assignment]
    _trust: object | None = field(default=None, compare=False, repr=False)
    _granted_basis: str | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.seqno, str) or not _SEQNO_RE.fullmatch(self.seqno):
            raise ValueError("PREDISPATCHSEQNO must be 10 digits")
        if not isinstance(self.runno, str) or not _RUNNO_RE.fullmatch(self.runno):
            raise ValueError("RUNNO must be 1 to 3 digits")
        if self.table_id != TABLE_ID:
            raise ValueError(f"table must be the frozen identity {TABLE_ID!r}")
        if self.region != REGION:
            raise ValueError(f"region must be {REGION!r}")
        if self.intervention != INTERVENTION:
            raise ValueError(f"intervention must be {INTERVENTION}")
        if self.unit != UNIT:
            raise ValueError(f"unit must be {UNIT!r}")
        if self.basis not in RECEIPT_BASES:
            raise ValueError("unknown receipt basis")
        if self._trust is not _RAW_BODY_TRUST or self._granted_basis != self.basis:
            raise ValueError(
                "receipts exist only through the module builders; basis, rows "
                "and hashes are granted from the raw body, never assigned")
        sha256_string(self.body_sha256, "body_sha256")
        sha256_string(self.header_line_sha256, "header_line_sha256")
        if self.header_line_sha256 != HEADER_LINE_SHA256:
            raise ValueError("header hash drifted from the frozen schema")
        identifier(self.filename, "filename")
        receipt = utc_unix(self.receipt_complete_unix, "receipt_complete_unix")
        if isinstance(self.rows, (str, bytes, Mapping)) or not self.rows:
            raise ValueError("rows must be a nonempty (interval_end, MW) sequence")
        try:
            parsed = tuple(self.rows)
        except TypeError as exc:
            raise ValueError("rows must be a nonempty (interval_end, MW) sequence") from exc
        cleaned = []
        previous_end = None
        for row in parsed:
            if len(row) != 2:
                raise ValueError("each row must be (interval_end, TOTALDEMAND)")
            end, value = row
            end = utc_unix(end, "interval_end")
            if end % HALF_HOUR_SECONDS:
                raise ValueError("interval ends must sit on the half-hour grid")
            if isinstance(value, bool) or not isinstance(value, float) or not isfinite(value):
                raise ValueError("TOTALDEMAND must be a finite float in MW")
            if previous_end is not None and end <= previous_end:
                raise ValueError("interval ends must be unique and strictly increasing")
            previous_end = end
            cleaned.append((end, value))
        if not isinstance(self.raw_body, bytes):
            raise ValueError("a receipt must carry its raw body")
        verified = parse_body(self.raw_body)
        if (verified.seqno, verified.runno, verified.body_sha256, verified.rows) != (
                self.seqno, self.runno, self.body_sha256, tuple(cleaned)):
            raise ValueError("rows, hashes or vintage do not match the verified raw body")
        object.__setattr__(self, "receipt_complete_unix", receipt)
        object.__setattr__(self, "rows", tuple(cleaned))

    def visible_at(self, origin_unix: int) -> bool:
        return self.receipt_complete_unix <= utc_unix(origin_unix, "origin_unix")

    def to_dict(self) -> dict:
        return {"seqno": self.seqno,
                "runno": self.runno,
                "receipt_complete_unix": self.receipt_complete_unix,
                "body_sha256": self.body_sha256,
                "header_line_sha256": self.header_line_sha256,
                "filename": self.filename,
                "table_id": self.table_id,
                "region": self.region,
                "intervention": self.intervention,
                "unit": self.unit,
                "basis": self.basis,
                "rows": [[end, value] for end, value in self.rows]}


def _receipt_from_bytes(
    raw_body: bytes, receipt_complete_unix: int, filename: str, basis: str,
) -> PredispatchTdReceipt:
    if not isinstance(raw_body, (bytes, bytearray)):
        raise ValueError("raw body must be bytes")
    raw = bytes(raw_body)
    parsed = parse_body(raw)
    return PredispatchTdReceipt(
        seqno=parsed.seqno,
        runno=parsed.runno,
        receipt_complete_unix=receipt_complete_unix,
        body_sha256=parsed.body_sha256,
        header_line_sha256=parsed.header_line_sha256,
        filename=identifier(filename, "filename"),
        table_id=TABLE_ID,
        region=REGION,
        intervention=INTERVENTION,
        unit=UNIT,
        basis=basis,
        rows=parsed.rows,
        raw_body=raw,
        _trust=_RAW_BODY_TRUST,
        _granted_basis=basis,
    )


def receipt_from_raw_body(
    raw_body: bytes | bytearray, receipt_complete_unix: int, filename: str,
) -> PredispatchTdReceipt:
    """The ONLY first-seen path: every value derived from the raw zip bytes."""
    return _receipt_from_bytes(raw_body, receipt_complete_unix, filename, REAL_BASIS)


def synthetic_receipt_from_raw_body(
    raw_body: bytes | bytearray, receipt_complete_unix: int, filename: str,
) -> PredispatchTdReceipt:
    """Fixture evidence: fully parsed and byte-bound, never first-seen."""
    return _receipt_from_bytes(raw_body, receipt_complete_unix, filename, SYNTHETIC_BASIS)


def load_receipt_store(events_path: str | Path) -> tuple[tuple[PredispatchTdReceipt, ...], tuple[dict, ...]]:
    """Re-verify a capture store; each event body is re-parsed from disk.

    Receipt times come from the append-only event ledger; hashes, vintage
    identity and every TOTALDEMAND value are re-derived from the immutable
    body file and cross-checked against the event record.
    """
    events = []
    for line in Path(events_path).read_text().splitlines():
        if line.strip():
            events.append(json.loads(line))
    receipts = []
    for event in events:
        if event.get("event") != "body_complete":
            continue
        body_path = Path(event["body_path"])
        if body_path.parent != Path(events_path).parent / "bodies":
            raise ValueError(f"event body path escapes the immutable store: {body_path}")
        if body_path.name != f"{event['body_sha256']}.zip":
            raise ValueError("event body file name does not match its declared hash")
        body = body_path.read_bytes()
        if hashlib.sha256(body).hexdigest() != event["body_sha256"]:
            raise ValueError("stored body hash does not match the event record")
        receipt = receipt_from_raw_body(
            body, int(event["receipt_complete_unix_ceil"]), event["filename"])
        if receipt.body_sha256 != event["body_sha256"]:
            raise ValueError("re-parsed body hash does not match the event record")
        receipts.append(receipt)
    if not receipts:
        raise ValueError("capture store carries no body-complete events")
    return tuple(receipts), tuple(events)


def _dedupe(receipts: Sequence[PredispatchTdReceipt]) -> tuple[PredispatchTdReceipt, ...]:
    """Fold duplicate bodies to the earliest receipt; forks fail closed.

    Two receipts of the same published body are one signal: they collapse to
    the EARLIEST body-complete instant.  One vintage (seqno, runno) carrying
    two different body hashes is a fork and fails closed — no last-write
    wins, no merge.
    """
    by_body: dict[tuple[str, str, str], PredispatchTdReceipt] = {}
    for record in receipts:
        if not isinstance(record, PredispatchTdReceipt):
            raise ValueError("ledger accepts only validated PredispatchTdReceipt records")
        key = (record.seqno, record.runno, record.body_sha256)
        known = by_body.get(key)
        if known is None or (record.receipt_complete_unix, record.filename) < (
                known.receipt_complete_unix, known.filename):
            by_body[key] = record
    bodies_per_vintage: dict[tuple[str, str], set[str]] = {}
    for seqno, runno, body_sha256 in by_body:
        bodies_per_vintage.setdefault((seqno, runno), set()).add(body_sha256)
    for vintage, bodies in bodies_per_vintage.items():
        if len(bodies) > 1:
            raise ValueError(
                f"conflicting content for vintage {vintage}: two different body "
                "hashes are a fork and fail closed")
    return tuple(by_body[key] for key in sorted(by_body))


def select_vintage(
    origin_unix: int, receipts: Iterable[PredispatchTdReceipt],
) -> PredispatchTdReceipt:
    """Newest vintage whose body-complete receipt is visible at origin.

    Exactly one run is examined — the newest visible (seqno, runno).  If it
    does not carry the 48 target half-hours, the origin is uncovered: no
    fallback to an older-but-longer run, no archive, no splice.
    """
    origin = utc_unix(origin_unix, "origin_unix")
    deduped = _dedupe(tuple(receipts))
    visible = [record for record in deduped if record.visible_at(origin)]
    if not visible:
        raise ValueError(f"origin {origin} is uncovered: no body-complete receipt at or before it")
    return max(visible, key=lambda record: _vintage_key(record.seqno, record.runno))


@dataclass(frozen=True, slots=True)
class PredispatchTdFutureSeries:
    origin_unix: int
    candidate_id: str
    covariate_name: str
    future_values: tuple[float, ...]
    seqno: str
    runno: str
    receipt_complete_unix: int
    body_sha256: str
    header_line_sha256: str
    publication_basis: str
    strict_asof: str
    ledger_hash: str
    series_hash: str
    per_point: tuple[dict, ...]


def build_future_series(
    origin_unix: int, receipts: Iterable[PredispatchTdReceipt],
) -> PredispatchTdFutureSeries:
    """Build the explicit 24-value future TOTALDEMAND tensor or refuse."""
    origin = utc_unix(origin_unix, "origin_unix")
    if origin % HOUR_SECONDS:
        raise ValueError("origin must be aligned to a UTC hour")
    selected = select_vintage(origin, receipts)
    by_end = dict(selected.rows)
    values, provenance = [], []
    for hour in range(FUTURE_POSITIONS):
        first_end = origin + (2 * hour + 1) * HALF_HOUR_SECONDS
        second_end = origin + (2 * hour + 2) * HALF_HOUR_SECONDS
        missing = [end for end in (first_end, second_end) if end not in by_end]
        if missing:
            raise ValueError(
                f"vintage {selected.seqno}/{selected.runno} lacks half-hour ends "
                f"{missing} for delivery hour {origin + hour * HOUR_SECONDS}; "
                "no fill, splice or fallback is permitted")
        mean = (by_end[first_end] + by_end[second_end]) / 2.0
        values.append(mean)
        provenance.append({
            "hour_index": hour,
            "delivery_hour_start_unix": origin + hour * HOUR_SECONDS,
            "half_hour_end_unix": [first_end, second_end],
            "half_hour_totaldemand_mw": [by_end[first_end], by_end[second_end]],
            "hour_mean_mw": mean,
        })
    ledger = _dedupe(tuple(receipts))
    ledger_hash = object_hash({"candidate_id": CANDIDATE_ID,
                               "receipts": [record.to_dict() for record in ledger]})
    series_hash = object_hash({
        "candidate_id": CANDIDATE_ID,
        "origin_unix": origin,
        "covariate_name": COVARIATE_NAME,
        "future_values": values,
        "selected_seqno": selected.seqno,
        "selected_runno": selected.runno,
        "selected_body_sha256": selected.body_sha256,
        "selected_header_line_sha256": selected.header_line_sha256,
        "selected_receipt_complete_unix": selected.receipt_complete_unix,
        "ledger_hash": ledger_hash,
    })
    return PredispatchTdFutureSeries(
        origin_unix=origin,
        candidate_id=CANDIDATE_ID,
        covariate_name=COVARIATE_NAME,
        future_values=tuple(values),
        seqno=selected.seqno,
        runno=selected.runno,
        receipt_complete_unix=selected.receipt_complete_unix,
        body_sha256=selected.body_sha256,
        header_line_sha256=selected.header_line_sha256,
        publication_basis=selected.basis,
        strict_asof=(STRICT_ASOF_VERIFIED if selected.basis == REAL_BASIS
                     else SYNTHETIC_STATUS),
        ledger_hash=ledger_hash,
        series_hash=series_hash,
        per_point=tuple(provenance),
    )


def attach_future_forecast(
    base: FrozenForecastInput, series: PredispatchTdFutureSeries,
) -> FrozenForecastInput:
    """Extend one frozen input by the typed future tensor; nothing else changes."""
    if not isinstance(base, FrozenForecastInput):
        raise ValueError("base must be a FrozenForecastInput")
    if not isinstance(series, PredispatchTdFutureSeries):
        raise ValueError("series must be a PredispatchTdFutureSeries")
    if series.origin_unix != base.origin_unix:
        raise ValueError("series origin does not match the base input origin")
    if "source_manifest_sha256" in base.metadata:
        raise ValueError("base input already carries a source manifest; refusing "
                         "to overwrite a second manifest slot")
    metadata = dict(base.metadata)
    metadata["source_manifest_sha256"] = series.series_hash
    metadata["input_profile"] = f"{metadata.get('input_profile', 'base')}__{CANDIDATE_ID}"
    return FrozenForecastInput(
        origin_unix=base.origin_unix,
        target_time_unix=base.target_time_unix,
        price_history=base.price_history,
        past_covariates=base.past_covariates,
        future_covariates=base.future_covariates,
        future_forecasts={series.covariate_name: series.future_values},
        metadata=metadata,
    )
