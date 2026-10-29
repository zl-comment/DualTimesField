"""Deterministic synthetic PREDISPATCHIS test-fixture builder (candidate 2).

Produces byte-deterministic ZIP bodies in the exact published layout
discovered from authenticated raw bytes: a ``C,NEMP.WORLD,PREDISPATCHIS,
AEMO,PUBLIC,...`` preamble row, I/D sections for a decoy table and the
frozen ``PREDISPATCH,REGION_SOLUTION,9`` table (131-field I header with the
127 pinned column names; every D row repeats its table identity in fields
1-3), and a final ``C,END OF REPORT,<total rows>`` trailer.  Unit tests and
the offline response check exercise the real parser with these bytes
instead of hand-fed rows.  This is SYNTHETIC FIXTURE evidence only: every
receipt built from it carries ``basis="synthetic_fixture"`` and can never
claim first-seen.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from forecasting.agent_forecasting.predispatch_td_abs import (
    HEADER_COLUMNS, REGION, TABLE_PARTS,
)

FIXTURE_ORIGIN_UNIX = 1641218400  # matches the frozen base development input
FIXTURE_SEQNO = "2022010343"
FIXTURE_RUNNO = "1"
FIXTURE_FILENAME = "SYNTHETIC_PREDISPATCHIS_202201032330_20220103230000.zip"
FIXTURE_RECEIPT_UNIX = FIXTURE_ORIGIN_UNIX - 600
AEMO_TIME_FORMAT = "%Y/%m/%d %H:%M:%S"
AEST_OFFSET_SECONDS = 10 * 3600


def _aest_wall(unix: int) -> str:
    return datetime.fromtimestamp(unix + AEST_OFFSET_SECONDS, tz=timezone.utc).strftime(AEMO_TIME_FORMAT)


def fixture_values(count: int, *, seqno: str = FIXTURE_SEQNO) -> list[float]:
    """Deterministic, varied, finite MW magnitudes; no randomness."""
    seed = int(seqno[8:10])
    return [5000.0 + 37.5 * (index + 1) + (50.0 if (index + seed) % 7 == 0 else 0.0)
            for index in range(count)]


def _target_row(seqno: str, runno: str, region: str, periodid: str,
                demand: str, lastchanged: str, datetime_cell: str) -> list[str]:
    row = ["D", *TABLE_PARTS, seqno, runno, region, periodid, "0", demand]
    row += ["0"] * (65 - len(row))            # pad through LASTCHANGED position
    row.append(lastchanged)                   # column 65: LASTCHANGED (metadata only)
    row.append(datetime_cell)                 # column 66: DATETIME
    row += ["0"] * (131 - len(row))           # remaining frozen columns
    return row


def build_body_bytes(
    *,
    origin_unix: int = FIXTURE_ORIGIN_UNIX,
    seqno: str = FIXTURE_SEQNO,
    runno: str = FIXTURE_RUNNO,
    n_rows: int = 70,
    drop_half_hour_end: int | None = None,
    shift_seconds: int = 0,
    duplicate_row_end: int | None = None,
) -> bytes:
    """One synthetic PREDISPATCHIS ZIP body around ``origin_unix``.

    Scope rows (NSW1, INTERVENTION=0) end on the half-hour grid from
    ``origin_unix`` (PERIODID 1) forward; two QLD1 rows ride along to prove
    region scoping; a decoy foreign section precedes the target section.
    Optional mutations serve the RED fail-closed tests.
    """
    lines: list[list[str]] = []
    lines.append(["C", "NEMP.WORLD", "PREDISPATCHIS", "AEMO", "PUBLIC",
                  "2022/01/03", "23:00:00", "0000000000000001",
                  "PREDISPATCHIS", "0000000000000000"])
    # decoy foreign table (different identity and width)
    lines.append(["I", "PREDISPATCH", "LOCAL_PRICE", "1"] + [f"X{i}" for i in range(7)])
    lines.append(["D", "PREDISPATCH", "LOCAL_PRICE", "1", "a", "b", "c", "d", "e", "f", "g"])
    # target section
    lines.append(["I", *TABLE_PARTS, *HEADER_COLUMNS])
    lastchanged = "2022/01/03 23:00:00"
    data: list[list[str]] = []
    values = fixture_values(n_rows, seqno=seqno)
    for index in range(n_rows):
        end = origin_unix + (index + 1) * 1800 + shift_seconds
        if drop_half_hour_end is not None and end == drop_half_hour_end:
            continue
        row = _target_row(seqno, runno, REGION, f"{index + 1:02d}",
                          f"{values[index]:.2f}", lastchanged, _aest_wall(end))
        data.append(row)
        if duplicate_row_end is not None and end == duplicate_row_end:
            data.append(list(row))
    # two out-of-scope rows prove the parser filters region before mapping
    for offset in (1, 2):
        end = origin_unix + offset * 1800
        data.append(_target_row(seqno, runno, "QLD1", "71", "1.00",
                                lastchanged, _aest_wall(end)))
    lines.extend(data)
    lines.append(["C", "END OF REPORT", "0"])  # placeholder, fixed below

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    for line in lines:
        writer.writerow(line)
    text = buffer.getvalue()
    # the trailer count is the total row count of the file, exactly like the
    # published bytes; rewrite it now that every row exists.
    rows = [row for row in csv.reader(io.StringIO(text, newline="")) if row]
    rows[-1][2] = str(len(rows))
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    for row in rows:
        writer.writerow(row)
    payload = buffer.getvalue().encode()

    zip_info = zipfile.ZipInfo("SYNTHETIC_PREDISPATCHIS_SECTION.csv",
                               date_time=(2022, 1, 3, 23, 0, 0))
    zip_info.compress_type = zipfile.ZIP_STORED
    zip_info.external_attr = 0o644 << 16
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr(zip_info, payload)
    return out.getvalue()


def build_fixture_receipt():
    from forecasting.agent_forecasting.predispatch_td_abs import (
        synthetic_receipt_from_raw_body,
    )
    body = build_body_bytes()
    return synthetic_receipt_from_raw_body(body, FIXTURE_RECEIPT_UNIX, FIXTURE_FILENAME)


def main() -> int:
    import hashlib
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    body = build_body_bytes()
    zip_path = root / "predispatchis_td_v2_synthetic.zip"
    zip_path.write_bytes(body)
    descriptor = {
        "fixture_basis": "synthetic_fixture",
        "origin_unix": FIXTURE_ORIGIN_UNIX,
        "receipt_complete_unix": FIXTURE_RECEIPT_UNIX,
        "filename": FIXTURE_FILENAME,
        "zip_sha256": hashlib.sha256(body).hexdigest(),
        "zip_size_bytes": len(body),
        "expected_seqno": FIXTURE_SEQNO,
        "expected_runno": FIXTURE_RUNNO,
        "note": "synthetic PREDISPATCHIS-format bytes; parser-verified, never first-seen",
    }
    (root / "predispatchis_td_v2_synthetic.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    print(f"wrote {zip_path} sha256={descriptor['zip_sha256']} bytes={len(body)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
