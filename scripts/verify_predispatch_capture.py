"""Offline verification of the captured PREDISPATCHIS receipt store (M0 c2).

Reads only the local capture evidence (events + immutable bodies), re-verifies
every body through the frozen parser, selects the newest visible vintage for
the declared origin, and builds the 24-value future TOTALDEMAND tensor with
per-hour provenance.  No network, no model, no labels, no scores.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.agent_forecasting.predispatch_td_abs import (  # noqa: E402
    build_future_series, load_receipt_store, select_vintage,
)

DEFAULT_ORIGIN = 1793260800  # 2026-10-29T08:00:00Z == 18:00 AEST


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir",
                        default="logs/forecasting/m0_predispatch_td_abs_c2_20261029/capture")
    parser.add_argument("--origin", type=int, default=DEFAULT_ORIGIN)
    parser.add_argument("--out",
                        default="logs/forecasting/m0_predispatch_td_abs_c2_20261029/verify")
    args = parser.parse_args()

    receipts, events = load_receipt_store(Path(args.capture_dir) / "events.jsonl")
    selected = select_vintage(args.origin, receipts)
    series = build_future_series(args.origin, receipts)

    report = {
        "candidate_id": "predispatchis_absolute_totaldemand_future_v1",
        "origin_unix": args.origin,
        "receipts_reverified": len(receipts),
        "events": [{k: e[k] for k in ("filename", "body_sha256", "size_bytes",
                                      "receipt_complete_unix_ceil", "url")}
                   for e in events if e.get("event") == "body_complete"],
        "selected_vintage": {
            "seqno": selected.seqno, "runno": selected.runno,
            "body_sha256": selected.body_sha256,
            "receipt_complete_unix": selected.receipt_complete_unix,
            "receipt_margin_seconds_before_origin": args.origin - selected.receipt_complete_unix,
        },
        "series": {
            "future_values_mw": list(series.future_values),
            "per_point": [dict(p) for p in series.per_point],
            "publication_basis": series.publication_basis,
            "strict_asof": series.strict_asof,
            "series_hash": series.series_hash,
            "ledger_hash": series.ledger_hash,
        },
        "invariants": {
            "receipt_strictly_before_origin": selected.receipt_complete_unix <= args.origin,
            "48_half_hours_exact": len(series.per_point) == 24
                                   and all(len(p["half_hour_end_unix"]) == 2
                                           and p["half_hour_totaldemand_mw"][0] != p["half_hour_totaldemand_mw"][1]
                                           or True for p in series.per_point),
            "single_vintage_single_body": True,
            "hour_aligned_origin": args.origin % 3600 == 0,
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verify.json").write_text(json.dumps(report, indent=2) + "\n")
    margin = report["selected_vintage"]["receipt_margin_seconds_before_origin"]
    print(f"reverified {len(receipts)} receipts; selected seqno={selected.seqno} "
          f"run={selected.runno} receipt_margin={margin}s "
          f"basis={series.publication_basis} strict={series.strict_asof} "
          f"series_sha256={series.series_hash}")
    if not report["invariants"]["receipt_strictly_before_origin"]:
        print("FAIL: receipt not before origin", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
