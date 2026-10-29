"""Transport-only capture of AEMO PREDISPATCHIS raw bodies (candidate 2 M0).

This script is the network edge: one directory listing plus at most
``--max-gets`` exact ZIP GETs, strict budget, zero retries, no parsing
authority.  It durably stores each raw body under its SHA-256 in an
immutable bodies directory and appends a body-complete receipt event with
the local UTC wall clock and monotonic clock recorded in the same process
at stream EOF.  It never computes scores, never reads labels, never calls
any model.

The analytical trust boundary is the frozen raw-body parser in
``forecasting.agent_forecasting.predispatch_td_abs``; a separate offline
verify step re-derives every value from the stored bytes after the header
freeze.  Nothing in this file grants evidence status to a body.

Companion protocol: docs/research/agent_forecasting/m0_predispatch_td_abs_protocol_20261029.txt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SOURCE_URL = "https://www.nemweb.com.au/REPORTS/CURRENT/PredispatchIS_Reports/"
_NAME_RE = re.compile(r"PUBLIC_PREDISPATCHIS_(\d{12})_(\d+)\.zip")
AEST_OFFSET_SECONDS = 10 * 3600
AEMO_TIME_FORMAT = "%Y/%m/%d %H:%M:%S"
CHUNK = 1 << 16


def _http_get(url: str, timeout: float) -> tuple[bytes, dict, dict]:
    """One single GET; no retry; wall+monotonic clocks recorded at EOF."""
    mono_before = time.monotonic()
    wall_before = time.time()
    request = urllib.request.Request(url, headers={"User-Agent": "DualTimesField-M0-capture/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # default cert verification
        status = response.status
        declared = {k: v for k, v in response.headers.items()
                    if k in ("Date", "Content-Length", "Last-Modified")}
        buffer = io.BytesIO()
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            buffer.write(block)
        body = buffer.getvalue()
    mono_eof = time.monotonic()
    wall_eof = time.time()
    health = {"wall_before": wall_before, "wall_eof": wall_eof,
              "monotonic_before": mono_before, "monotonic_eof": mono_eof,
              "wall_advance_seconds": wall_eof - wall_before,
              "monotonic_advance_seconds": mono_eof - mono_before}
    if health["wall_advance_seconds"] < 0 or health["monotonic_advance_seconds"] <= 0:
        raise RuntimeError("local clock regressed during download; refusing to certify")
    return body, {"status": status, "declared_headers": declared}, health


def _fsync_write(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        path.chmod(0o445)
        import os
        os.fsync(stream.fileno())


def _soft_discovery(body: bytes) -> dict:
    """Pre-freeze inventory only; NOT the frozen parser, grants nothing."""
    archive = zipfile.ZipFile(io.BytesIO(body))
    try:
        members = archive.namelist()
        if len(members) != 1:
            return {"members": len(members), "region_solution_section": False}
        text = archive.read(members[0]).decode("utf-8-sig", errors="replace")
    finally:
        archive.close()
    rows = [r for r in csv.reader(io.StringIO(text, newline="")) if r]
    sections = [r for r in rows if r[0].strip() == "I"]
    wanted = None
    for row in rows:
        if row[0].strip() != "I":
            continue
        if tuple(part.strip() for part in row[1:4]) == ("PREDISPATCH", "REGION_SOLUTION", "9"):
            wanted = row
    if wanted is None:
        return {"region_solution_section": False, "i_sections": len(sections)}
    columns = [c.strip() for c in wanted[4:]]
    index = {name: pos + 4 for pos, name in enumerate(columns)}
    missing = [n for n in ("PREDISPATCHSEQNO", "RUNNO", "REGIONID", "PERIODID",
                           "INTERVENTION", "TOTALDEMAND", "DATETIME") if n not in index]
    scope_rows, ends = [], []
    for row in rows:
        if row[0].strip() != "D" or len(row) != len(wanted):
            continue
        if row[index["REGIONID"]].strip() != "NSW1" or row[index["INTERVENTION"]].strip() != "0":
            continue
        scope_rows.append(row)
        try:
            naive = datetime.strptime(row[index["DATETIME"]].strip(), AEMO_TIME_FORMAT)
            # AEST wall time is UTC+10 with no DST: UTC = wall - 10h.
            ends.append(int(naive.replace(tzinfo=timezone.utc).timestamp()) - AEST_OFFSET_SECONDS)
        except ValueError:
            pass
    origin = max(ends) - 24 * 3600 if ends else None
    header_line = ",".join(wanted)
    return {
        "region_solution_section": True,
        "header_field_count": len(wanted),
        "header_columns": columns,
        "header_line_sha256": hashlib.sha256(header_line.encode()).hexdigest(),
        "header_columns_sha256": hashlib.sha256(",".join(columns).encode()).hexdigest(),
        "required_columns_missing": missing,
        "nsw1_int0_rows": len(scope_rows),
        "seqnos": sorted({r[index["PREDISPATCHSEQNO"]].strip() for r in scope_rows}),
        "runnos": sorted({r[index["RUNNO"]].strip() for r in scope_rows}),
        "datetime_min_max_unix": [min(ends), max(ends)] if ends else None,
        "file_end_anchored_origin_unix": origin,
        "file_end_anchored_origin_hour_aligned": bool(origin is not None and origin % 3600 == 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir",
                        default="logs/forecasting/m0_predispatch_td_abs_c2_20261029/capture")
    parser.add_argument("--max-gets", type=int, default=2)
    parser.add_argument("--max-total-bytes", type=int, default=16777216)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--re-discover-from-store", action="store_true",
                        help="offline: regenerate discovery_*.json from stored bodies; no network")
    args = parser.parse_args()

    root = Path(args.evidence_dir)

    if args.re_discover_from_store:
        events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
        for event in events:
            body = Path(event["body_path"]).read_bytes()
            summary = _soft_discovery(body)
            summary.update({"filename": event["filename"], "body_sha256": event["body_sha256"],
                            "receipt_complete_unix_ceil": event["receipt_complete_unix_ceil"]})
            (root / f"discovery_{event['filename']}.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"re-discovered {event['filename']} origin_end_anchored="
                  f"{summary.get('file_end_anchored_origin_unix')} "
                  f"aligned={summary.get('file_end_anchored_origin_hour_aligned')}")
        return 0

    bodies = root / "bodies"
    bodies.mkdir(parents=True, exist_ok=True)
    ledger = root / "events.jsonl"

    listing_body, listing_meta, listing_health = _http_get(SOURCE_URL, args.timeout_seconds)
    names = sorted(set(_NAME_RE.findall(listing_body.decode("utf-8", errors="replace"))))
    # each tuple is (runstamp12, published14); newest runstamp first
    names.sort(key=lambda pair: pair[0], reverse=True)
    chosen = [f"PUBLIC_PREDISPATCHIS_{run}_{pub}.zip" for run, pub in names[:args.max_gets]]
    listing_record = {
        "url": SOURCE_URL,
        "fetched_wall_utc": datetime.fromtimestamp(listing_health["wall_eof"], timezone.utc).isoformat(),
        "status": listing_meta["status"],
        "body_sha256": hashlib.sha256(listing_body).hexdigest(),
        "size_bytes": len(listing_body),
        "matching_files_total": len(names),
        "newest_runstamps": [pair[0] for pair in names[:8]],
        "chosen_files": chosen,
    }
    (root / "listing.json").write_text(json.dumps(listing_record, indent=2) + "\n")
    total_bytes = len(listing_body)

    for rank, name in enumerate(chosen):
        if rank:
            time.sleep(args.interval_seconds)
        url = SOURCE_URL + name
        body, meta, health = _http_get(url, args.timeout_seconds)
        total_bytes += len(body)
        if total_bytes > args.max_total_bytes:
            raise SystemExit(f"budget exceeded: {total_bytes} bytes > {args.max_total_bytes}")
        digest = hashlib.sha256(body).hexdigest()
        target = bodies / f"{digest}.zip"
        if not target.exists():
            _fsync_write(target, body)
        elif hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"immutable body store collision at {target}")
        receipt_second = math.ceil(health["wall_eof"])
        event = {
            "event": "body_complete",
            "rank": rank,
            "url": url,
            "filename": name,
            "http_status": meta["status"],
            "body_sha256": digest,
            "size_bytes": len(body),
            "body_path": str(target),
            "receipt_complete_wall_utc": datetime.fromtimestamp(health["wall_eof"], timezone.utc).isoformat(),
            "receipt_complete_unix_ceil": receipt_second,
            "monotonic_eof": health["monotonic_eof"],
            "clock_health": health,
            "declared_headers_not_used_as_clock": meta["declared_headers"],
        }
        with ledger.open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
        summary = _soft_discovery(body)
        summary.update({"filename": name, "body_sha256": digest,
                        "receipt_complete_unix_ceil": receipt_second})
        (root / f"discovery_{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"captured {name} sha256={digest} bytes={len(body)} "
              f"receipt_complete_ceil={receipt_second} "
              f"origin={summary.get('derived_origin_unix')} "
              f"aligned={summary.get('derived_origin_hour_aligned')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
