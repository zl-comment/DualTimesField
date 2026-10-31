#!/usr/bin/env python3
"""Independently check completed R12 evidence; never imports production scoring.

PASS means stored evidence matches these calculations, not that the information
hypothesis passed. This script performs no prediction, training or source fetch.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

REGIONS = ("NSW1", "QLD1", "TAS1")
TAUS = (0.05, 0.10, 0.50, 0.90, 0.95)
TOLERANCE = 1e-10


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def native_score(q, y):
    if y.size == 0:
        return None
    points, truth = q.reshape(-1, 5), y.reshape(-1)
    pb = {}
    for i, tau in enumerate(TAUS):
        residual = truth - points[:, i]
        pb[str(tau)] = float(np.where(residual >= 0, tau * residual,
                                     (tau - 1) * residual).mean())
    return {
        "n_cells": int(truth.size), "mae": float(np.abs(points[:, 2] - truth).mean()),
        "quantile_score": float(2 * sum(pb.values()) / 5), "pinball_by_quantile": pb,
        "coverage_80": float(np.mean((points[:, 1] <= truth) & (truth <= points[:, 3]))),
        "coverage_90": float(np.mean((points[:, 0] <= truth) & (truth <= points[:, 4]))),
        "width_80": float(np.mean(points[:, 3] - points[:, 1])),
        "width_90": float(np.mean(points[:, 4] - points[:, 0])),
        "native_crossings": int(np.count_nonzero(points[:, 1:] < points[:, :-1])),
    }


def bootstrap_loop(a_daily, b_daily, common):
    """Loop over draws and regions; sample original dates without compression."""
    days = len(common)
    result = {"method": "synchronous_non_circular_moving_day_blocks",
              "block_days": 7, "replicates": 2000, "seed": 20261031,
              "date_axis_length": days, "valid_replicates": 0, "relative_qs_ci95": None}
    if days < 7:
        return {**result, "status": "fewer_than_seven_planned_days"}
    rng = np.random.default_rng(20261031)
    changes = []
    for _ in range(2000):
        starts = rng.integers(0, days - 6, size=(days + 6) // 7)
        selected = np.concatenate([np.arange(start, start + 7) for start in starts])[:days]
        pairs = []
        for r in range(3):
            valid = selected[common[selected, r]]
            if not len(valid):
                break
            pairs.append((float(np.mean(a_daily[valid, r])), float(np.mean(b_daily[valid, r]))))
        if len(pairs) != 3:
            continue
        before = sum(pair[0] for pair in pairs) / 3
        after = sum(pair[1] for pair in pairs) / 3
        if before > 0 and np.isfinite([before, after]).all():
            changes.append((after - before) / before)
    result["valid_replicates"] = len(changes)
    if len(changes) != 2000:
        return {**result, "status": "undefined_resampled_region_or_reference_loss"}
    return {**result, "status": "complete",
            "relative_qs_ci95": np.percentile(changes, [2.5, 97.5]).tolist()}


def scientific_recalculation(a, b, y, source):
    days = len(a)
    fa, fb = np.isfinite(a).all(axis=(2, 3)), np.isfinite(b).all(axis=(2, 3))
    fy = np.isfinite(y).all(axis=2)
    primary, common = fa & fy, fa & fb & fy
    regional, daily_a, daily_b = {}, np.full((days, 3), np.nan), np.full((days, 3), np.nan)

    def ratio(before, after):
        return (after - before) / before if before > 0 else None

    for r, region in enumerate(REGIONS):
        mask = common[:, r]
        sa, sb = native_score(a[mask, r], y[mask, r]), native_score(b[mask, r], y[mask, r])
        for day in np.flatnonzero(mask):
            daily_a[day, r] = native_score(a[day, r], y[day, r])["quantile_score"]
            daily_b[day, r] = native_score(b[day, r], y[day, r])["quantile_score"]
        regional[region] = {
            "planned_days": days, "common_days": int(mask.sum()), "primary_days": int(primary[:, r].sum()),
            "b_unresolved_on_primary_days": int(np.count_nonzero(primary[:, r] & ~fb[:, r])),
            "source_complete_days": int(source[:, r].sum()), "source_coverage": float(source[:, r].mean()),
            "a_prediction_availability": float(fa[:, r].mean()), "b_prediction_availability": float(fb[:, r].mean()),
            "label_availability": float(fy[:, r].mean()), "common_scoring_rate": float(mask.mean()),
            "source_missing_scored_days": int(np.count_nonzero(mask & ~source[:, r])),
            "A": sa, "B": sb,
            "relative_qs_change": ratio(sa["quantile_score"], sb["quantile_score"]) if sa else None,
            "relative_mae_change": ratio(sa["mae"], sb["mae"]) if sa else None,
            "coverage_distance_increase": {str(level): (abs(sb[f"coverage_{level}"] - level / 100)
                - abs(sa[f"coverage_{level}"] - level / 100)) if sa else None for level in (80, 90)},
        }
    macro = {}
    for arm in ("A", "B"):
        scores = [regional[region][arm] for region in REGIONS]
        if any(score is None for score in scores):
            macro[arm] = None
            continue
        macro[arm] = {key: sum(score[key] for score in scores) / 3 for key in
                      ("mae", "quantile_score", "coverage_80", "coverage_90", "width_80", "width_90")}
        macro[arm]["pinball_by_quantile"] = {str(tau): sum(s["pinball_by_quantile"][str(tau)] for s in scores) / 3
                                              for tau in TAUS}
        for level in (80, 90):
            macro[arm][f"coverage_distance_{level}"] = sum(abs(s[f"coverage_{level}"] - level / 100) for s in scores) / 3
    ma, mb = macro["A"], macro["B"]
    for metric in ("qs", "mae"):
        key = "quantile_score" if metric == "qs" else "mae"
        macro[f"relative_{metric}_change"] = ratio(ma[key], mb[key]) if ma else None
    macro["coverage_distance_increase"] = {str(level): (mb[f"coverage_distance_{level}"] - ma[f"coverage_distance_{level}"])
                                           if ma else None for level in (80, 90)}
    bootstrap = bootstrap_loop(daily_a, daily_b, common)
    criteria = []

    def criterion(name, passed, category, **values):
        criteria.append({"name": name, "passed": bool(passed), "category": category, **values})

    for region, row in regional.items():
        criterion(f"{region}/b_covers_primary", row["b_unresolved_on_primary_days"] == 0, "data",
                  unresolved_days=row["b_unresolved_on_primary_days"], maximum=0)
        for field in ("source_coverage", "a_prediction_availability", "b_prediction_availability", "common_scoring_rate"):
            criterion(f"{region}/{field}", row[field] >= 0.95, "data", value=row[field], minimum=0.95)
    criterion("macro/qs_improvement", ma is not None and ma["quantile_score"] > 0
              and mb["quantile_score"] <= ma["quantile_score"] * 0.99, "effect",
              relative_change=macro["relative_qs_change"], maximum=-0.01)
    criterion("macro/mae_protection", ma is not None and mb["mae"] <= ma["mae"] * 1.01, "effect",
              relative_change=macro["relative_mae_change"], maximum=0.01)
    for region, row in regional.items():
        criterion(f"{region}/qs_protection", row["A"] is not None
                  and row["B"]["quantile_score"] <= 1.02 * row["A"]["quantile_score"], "effect",
                  relative_change=row["relative_qs_change"], maximum=0.02)
    for name, row in [("macro", macro), *regional.items()]:
        for level in (80, 90):
            increase = row["coverage_distance_increase"][str(level)]
            criterion(f"{name}/coverage_{level}_protection", increase is not None and increase <= 0.01,
                      "effect", distance_increase=increase, maximum=0.01)
    ci = bootstrap["relative_qs_ci95"]
    criterion("bootstrap/upper_bound_below_zero", ci is not None and ci[1] < 0, "inference",
              upper_bound=ci[1] if ci is not None else None)
    failed = [item["name"] for item in criteria if not item["passed"]]
    if any(not item["passed"] for item in criteria if item["category"] == "data"):
        decision = "DATA_OR_EXECUTION_BLOCKED"
    elif days < 7:
        decision = "INCONCLUSIVE"
    elif any(not item["passed"] for item in criteria if item["category"] == "effect"):
        decision = "NO_USEFUL_GAIN"
    elif not criteria[-1]["passed"]:
        decision = "INCONCLUSIVE"
    else:
        decision = "PASS_INFORMATION"
    return {"regional": regional, "macro": macro, "bootstrap": bootstrap, "criteria": criteria,
            "failed_criteria": failed, "decision": decision, "passed": decision == "PASS_INFORMATION",
            "primary_mask": primary.tolist(), "common_mask": common.tolist(),
            "scores_are_partial_diagnostics": bool(np.any(primary & ~fb))}


def verify(root: Path, archive: Path) -> dict:
    hashes, numeric_differences = {}, []
    exact_arrays = 0

    def remember(path):
        value = digest(path)
        hashes[str(path.resolve())] = value
        return value

    def read(path):
        remember(path)
        return json.loads(path.read_text())

    def match(actual, expected, where):
        if isinstance(expected, dict):
            for key, value in expected.items():
                match(actual[key], value, f"{where}/{key}")
        elif isinstance(expected, list):
            if len(actual) != len(expected):
                raise ValueError(f"length mismatch: {where}")
            for i, value in enumerate(expected):
                match(actual[i], value, f"{where}/{i}")
        elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
            difference = abs(float(actual) - expected)
            if not np.isfinite(difference) or difference > TOLERANCE:
                raise ValueError(f"numerical mismatch: {where}: {actual} versus {expected}")
            numeric_differences.append(float(difference))
        elif actual != expected:
            raise ValueError(f"mismatch: {where}: {actual} versus {expected}")

    def equal_array(left, right, where):
        nonlocal exact_arrays
        if not np.array_equal(left, right, equal_nan=True):
            raise ValueError(f"array mismatch: {where}")
        exact_arrays += 1

    prepared = read(root / "prepared.json")
    predicted = read(root / "predicted.json")
    reported = read(archive / "metrics.json")
    archived_prepared = read(archive / "prepared_manifest.json")
    if prepared != archived_prepared or predicted != reported["prediction_run"]:
        raise ValueError("archived execution/preparation records differ")
    match(predicted["execution_commit"], prepared["execution_commit"], "prediction source commit")
    match(reported["execution_commit"], prepared["execution_commit"], "scoring source commit")
    match(predicted["seed"], 2026, "frozen numerical seed")
    for name, expected in prepared["files"].items():
        if remember(root / name) != expected:
            raise ValueError(f"prepared hash mismatch: {name}")
    for name, expected in predicted["files"].items():
        if remember(root / "predictions" / name) != expected:
            raise ValueError(f"prediction hash mismatch: {name}")
    repository = Path(__file__).resolve().parents[1]
    for name, expected in prepared["source_sha256"].items():
        if remember(repository / name) != expected:
            raise ValueError(f"frozen source hash mismatch: {name}")

    dates = [(date(2023, 1, 1) + timedelta(days=i)).isoformat() for i in range(365)]
    first = int(datetime(2023, 1, 1, tzinfo=timezone(timedelta(hours=10))).timestamp())
    origins = first + np.arange(365, dtype=np.int64) * 86400
    match(prepared["dates"], dates, "planned dates")
    match(reported["planned_dates"], dates, "reported dates")
    match(prepared["regions"], list(REGIONS), "prepared regions")
    match(reported["regions"], list(REGIONS), "reported regions")
    equal_array(prepared["origins"], origins, "prepared origins")
    remember(archive / "paired_predictions.npz")
    with np.load(archive / "paired_predictions.npz", allow_pickle=False) as data:
        a, b, y, source = (np.array(data[key]) for key in ("A", "B", "actual", "source_complete"))
        equal_array(data["origin_unix"], origins, "paired origins")
    if a.shape != (365, 3, 24, 5) or b.shape != a.shape or y.shape != (365, 3, 24) or source.shape != (365, 3):
        raise ValueError("paired archive shape mismatch")
    if source.dtype != np.dtype(bool):
        raise ValueError("source completeness must be boolean")
    equal_array(source, prepared["source_complete"], "source completeness")
    origin_index = {int(origin): i for i, origin in enumerate(origins)}
    regional_origins = {}
    for r, region in enumerate(REGIONS):
        path = root / "predictions" / f"{region}.npz"
        remember(path)
        with np.load(path, allow_pickle=False) as data:
            saved_origins = data["origin_unix"]
            if len(set(saved_origins)) != len(saved_origins):
                raise ValueError(f"duplicate saved origins: {region}")
            input_records = read(root / "inputs" / f"{region}.json")
            equal_array(saved_origins, [record["origin_unix"] for record in input_records], f"{region} prepared inputs")
            match(len(saved_origins), prepared["input_counts"][region], f"{region} prepared count")
            regional_origins[region] = saved_origins.tolist()
            indices = [origin_index[int(origin)] for origin in saved_origins]
            equal_array(data["target_time_unix"], saved_origins[:, None] + np.arange(24) * 3600, f"{region} targets")
            equal_array(data["source_complete"], source[indices, r], f"{region} source")
            for arm, full in (("A", a), ("B", b)):
                reconstructed = np.full((365, 24, 5), np.nan)
                reconstructed[indices] = data[arm]
                equal_array(reconstructed, full[:, r], f"{region}/{arm} output-to-archive")
        path = root / "labels" / f"{region}.npz"
        remember(path)
        with np.load(path, allow_pickle=False) as data:
            if len(set(data["origin_unix"])) != len(data["origin_unix"]):
                raise ValueError(f"duplicate label origins: {region}")
            reconstructed = np.full((365, 24), np.nan)
            indices = [origin_index[int(origin)] for origin in data["origin_unix"]]
            reconstructed[indices] = data["actual"]
            equal_array(reconstructed, y[:, r], f"{region} label-to-archive")

    ledger_path = root / "predictions" / "ledger.jsonl"
    remember(ledger_path)
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    entries = {(entry["region"], entry["origin_unix"]): entry for entry in ledger}
    expected_keys = {(region, origin) for region, values in regional_origins.items() for origin in values}
    if len(entries) != len(ledger) or set(entries) != expected_keys:
        raise ValueError("ledger entries do not match saved region/origin units")
    attempted_b = {"predicted", "fallback_nonfinite_forecast", "fallback_model_error"}
    counts_a, counts_b = Counter(), Counter()
    fallback_count, attempts_b = 0, 0
    for entry in ledger:
        r, i = REGIONS.index(entry["region"]), origin_index[entry["origin_unix"]]
        sa, sb = entry["A_status"], entry["B_status"]
        counts_a[sa] += 1
        counts_b[sb] += 1
        a_finite = bool(np.isfinite(a[i, r]).all())
        if sa != ("predicted" if a_finite else "unavailable"):
            raise ValueError("A ledger status does not match native output")
        if not isinstance(entry["input_A_sha256"], str) or len(entry["input_A_sha256"]) != 64:
            raise ValueError("missing A input hash")
        if not a_finite:
            if sb != "unavailable" or entry["input_B_sha256"] is not None:
                raise ValueError("B was attempted without usable A")
            equal_array(b[i, r], a[i, r], "unavailable B preserves unavailable A")
            continue
        if not source[i, r]:
            if sb != "fallback_source_missing" or entry["input_B_sha256"] != entry["input_A_sha256"]:
                raise ValueError("source-missing fallback ledger mismatch")
        else:
            if sb not in attempted_b or not isinstance(entry["input_B_sha256"], str) or len(entry["input_B_sha256"]) != 64:
                raise ValueError("source-present B call/status/hash mismatch")
            attempts_b += 1
        if not np.isfinite(b[i, r]).all():
            raise ValueError("B has an unresolved gap despite usable A")
        if sb.startswith("fallback_"):
            equal_array(b[i, r], a[i, r], "fallback output must exactly equal A")
            fallback_count += 1
    attempts = {"A": len(ledger), "B": attempts_b}
    match(predicted["unit_attempts"], attempts, "attempt counts")
    if sum(attempts.values()) > 2190 or max(attempts.values()) > 1095:
        raise ValueError("numerical task budget exceeded")
    calls = {"A": 0, "B": 0}
    for region, values in regional_origins.items():
        batches = [values[:1]] + [values[i:i + 16] for i in range(1, len(values), 16)]
        for batch in batches:
            if batch:
                calls["A"] += 1
                calls["B"] += int(any(entries[(region, origin)]["B_status"] in attempted_b for origin in batch))
    match(predicted["pipeline_calls"], calls, "pipeline calls")
    for name, expected in (("device", "cpu"), ("cross_learning", False), ("training_calls", 0), ("llm_calls", 0)):
        match(predicted[name], expected, f"execution/{name}")
    recomputed = scientific_recalculation(a, b, y, source)
    match(reported, recomputed, "independent calculation")
    remember(archive / "regional_metrics.csv")
    return {
        "passed": True, "verdict": "PASS_EVIDENCE_MATCH",
        "scope": "Stored evidence and independent calculations match; not a claim of information or Agent benefit.",
        "execution_commit": predicted["execution_commit"], "scientific_decision": recomputed["decision"],
        "planned_days": 365, "regions": list(REGIONS), "regional_scores_checked": 6,
        "criteria_checked": len(recomputed["criteria"]), "numeric_fields_checked": len(numeric_differences),
        "max_absolute_numeric_difference": max(numeric_differences, default=0.0), "numeric_tolerance": TOLERANCE,
        "exact_array_checks": exact_arrays, "ledger_units": len(ledger),
        "ledger_status_counts": {"A": dict(counts_a), "B": dict(counts_b)},
        "fallback_outputs_checked": fallback_count, "unit_attempts": attempts, "pipeline_calls": calls,
        "bootstrap": recomputed["bootstrap"], "macro": recomputed["macro"],
        "failed_criteria": recomputed["failed_criteria"], "files_hashed": len(hashes), "sha256": hashes,
        "limitations": ["No model replay or raw source re-parsing; receipt-time and demand-semantics limitations remain.",
                        "Input file hashes and ledger hash presence are checked; model-input hashes are not reconstructed here."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start, cpu_start = time.monotonic(), time.process_time()
    try:
        report = verify(args.root.resolve(), args.archive.resolve())
        exit_code = 0
    except Exception as exc:
        report = {"passed": False, "verdict": "FAIL_EVIDENCE_MISMATCH", "error": f"{type(exc).__name__}: {exc}"}
        exit_code = 1
    report["verifier_sha256"] = digest(__file__)
    report["resources"] = {"wall_seconds": time.monotonic() - start,
                           "cpu_seconds": time.process_time() - cpu_start,
                           "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "verdict": report["verdict"], "output": str(args.output)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
