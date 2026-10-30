"""Independent, offline audit of the completed r9 matrix.

Only reads this run's published inputs/labels/predictions and local source hashes.
No production scoring/decision implementation is imported; no model is called.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/agent_forecasting/r9_development_20261030.json"
ACTIONS = ["KEEP", "SHIFT_UP", "SHIFT_DOWN", "WIDEN", "NARROW"]
LEVELS = np.array([.05, .10, .50, .90, .95])
FINAL_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "final_decision", "strict": True, "schema": {
        "type": "object", "properties": {
            "action": {"type": "string", "enum": ACTIONS}, "reason": {"type": "string"}},
        "required": ["action", "reason"], "additionalProperties": False}}}


def _check(ok, message):
    if not ok:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text())


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _verify_directory(path):
    inventory = _read(path / "COMMIT.json")["files"]
    actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}
    _check(actual == set(inventory) | {"COMMIT.json"}, f"file inventory: {path}")
    for name, item in inventory.items():
        file = path / name
        _check(file.stat().st_size == item["bytes"] and _sha(file) == item["sha256"],
               f"artifact hash: {file}")
    return _sha(path / "COMMIT.json")


def independent_scores(quantiles, actual):
    # Same published estimand: order the five raw quantiles before scoring.
    ordered = np.sort(np.asarray(quantiles, dtype=np.float64), axis=2)
    residual = np.asarray(actual, dtype=np.float64)[..., None] - ordered
    return {"mae": float(np.mean(np.abs(residual[..., 2]))),
            "quantile_score": float(2 * np.mean(residual * (LEVELS - (residual < 0))))}


def _unique_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs)


def _executed_tools(wires):
    """Independently count the valid tool batch executions visible on the wire."""
    executed = 0
    for wire in wires:
        if wire["error"] or wire["tag"] == "D-final":
            continue
        message = _unique_json(wire["raw_response_text"])["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        for call in calls:
            fn = call["function"]
            name, args = fn["name"], _unique_json(fn["arguments"])
            key = "days_back" if name == "hourly_profile" else "hours"
            expected = set() if name == "base_quantiles" else {key}
            allowed = (7, 14) if name == "hourly_profile" else (24, 48, 72)
            valid = (name in ("history_window", "demand_window", "hourly_profile", "base_quantiles")
                     and isinstance(args, dict) and set(args) == expected
                     and (not expected or (type(args[key]) is int and args[key] in allowed)))
            if not valid:
                return executed  # the entire invalid batch is rejected before execution
        executed += min(len(calls), 4 - executed)
    return executed


def audit(cfg):
    root = Path(cfg["output_root"])
    if not root.is_absolute():
        root = REPO / root
    seeds, regions, count = cfg["seeds"], cfg["regions"], cfg["origins_per_region"]
    commits = {name: _verify_directory(root / name) for name in ("inputs", "labels", "results")}
    manifest = _read(root / "inputs" / "manifest.json")
    _check(manifest["seeds"] == seeds, "prepared seeds differ from configuration")
    _check(set(manifest["sources"]) == set(cfg["source_files"]), "incomplete source fingerprint")
    for name, digest in manifest["sources"].items():
        _check(_sha(REPO / name) == digest, f"execution source changed: {name}")
    summary = _read(root / "results" / "summary.json")
    _check(summary["execution_commit"] == manifest["execution_commit"], "summary execution identity")
    _check(summary["final_test_evaluated"] is False, "unexpected final-test claim")
    published = {(r["seed"], r["region"]): r for r in summary["results"]}
    _check(len(published) == len(summary["results"]) == len(seeds) * len(regions), "summary matrix size")
    scores, stages, metric_errors = {}, [], []
    model_ids, failures, transport_errors, actions = Counter(), Counter(), Counter(), Counter()
    wire_tokens, decision_tokens = Counter(), Counter()
    llm_calls = missing_usage = missing_models = incomplete_execution_counts = 0
    origin_counts, decision_failures, per_arm_calls = Counter(), Counter(), Counter()

    def compare(value, expected, label):
        error = abs(float(value) - float(expected))
        metric_errors.append(error)
        _check(np.isclose(value, expected, atol=1e-9, rtol=1e-10), f"metric mismatch {label}: {value} != {expected}")

    for region in regions:
        labels = _arrays(root / "labels" / f"{region}.npz")
        records = _read(root / "inputs" / f"{region}.json")
        origins, targets, actual = (labels[k] for k in ("origin_unix", "target_time_unix", "actual"))
        _check(len(records) == len(origins) == count and len(set(origins.tolist())) == count, f"origin count: {region}")
        _check(actual.shape == (count, 24) and np.isfinite(actual).all(), f"actual shape: {region}")
        _check(np.array_equal(targets, origins[:, None] + np.arange(24) * 3600), f"target hours: {region}")
        _check(np.array_equal(origins, [r["origin_unix"] for r in records]), f"input origins: {region}")
        _check(np.array_equal(targets, [r["target_time_unix"] for r in records]), f"input targets: {region}")
        for record in records:
            _check(not {"actual", "labels", "target"}.intersection(record), "label field in forecast input")
            _check(len(record["history_prices"]) == len(record["history_demand"]) == 72, "history length")
            _check(all(datetime.fromisoformat(t).timestamp() <= record["origin_unix"]
                       for t in record["tool_history"]["available_at_aest"]), "unavailable tool history")
        for seed in seeds:
            baseline = None
            for arm in ("A", "B", "C", "D"):
                path = root / str(seed) / region / arm
                commits[str(path.relative_to(root))] = _verify_directory(path)
                prediction, run = _arrays(path / "predictions.npz"), _read(path / "run.json")
                _check((run["seed"], run["region"], run["arm"]) == (seed, region, arm), f"stage identity: {path}")
                _check(run["execution_commit"] == manifest["execution_commit"], "stage execution commit")
                _check(run["input_sha256"] == _sha(root / "inputs" / f"{region}.json"), "stage input hash")
                for key in ("origin_unix", "target_time_unix"):
                    _check(np.array_equal(prediction[key], labels[key]), f"prediction identity: {path}/{key}")
                values = prediction["quantiles"]
                _check(values.shape == (count, 24, 5) and np.isfinite(values).all(), f"forecast shape: {path}")
                if arm == "A":
                    baseline = values
                    _check(run["predict_calls"] == 1 and run["training_calls"] == 0, "base call count")
                    _check(run["checkpoint"] == cfg["checkpoint"], "base checkpoint identity")
                    exits = np.zeros(count, dtype=int)
                else:
                    _check(run["base_sha256"] == _sha(path.parent / "A" / "predictions.npz"), "shared base hash")
                    exits = prediction["exit_codes"]
                    _check(exits.shape == (count,), "decision count")
                    bad = exits != 0
                    _check(values.dtype == baseline.dtype and values[bad].tobytes() == baseline[bad].tobytes(),
                           "failed decisions must preserve exact A bytes")
                    rows, wires = _lines(path / "decisions.jsonl"), _lines(path / "wire.jsonl")
                    _check(len(rows) == count and [r["origin_unix"] for r in rows] == origins.tolist(), "decision row identities")
                    journal_path = root / "attempts" / f"{seed}_{region}_{arm}.jsonl"
                    _check(run["attempt_journal_sha256"] == _sha(journal_path), "attempt journal hash")
                    journal = _lines(journal_path)
                    _check([{k: v for k, v in e.items() if k != "event"} for e in journal if e["event"] == "llm"] == wires,
                           "published wire differs from attempted calls")
                    _check([e["origin_unix"] for e in journal if e["event"] == "origin_started"] == origins.tolist(), "attempted origins")
                    _check([{k: v for k, v in e.items() if k != "event"} for e in journal if e["event"] == "origin_completed"] == rows,
                           "published decisions differ from completion journal")
                    grouped = defaultdict(list)
                    for wire in wires:
                        _check((wire["seed"], wire["region"], wire["arm"]) == (seed, region, arm), "wire identity")
                        _check(wire["origin_unix"] in origins, "wire origin not planned")
                        request = _unique_json(wire["raw_request_json"])
                        _check(request["seed"] == seed and request["temperature"] == wire["temperature"] == 0.0, "request seed/temperature")
                        _check(request["model"] == wire["requested_model"] == "qwen3-30b3a", "requested model changed")
                        _check(wire["url"] == "http://127.0.0.1:18011/v1/chat/completions", "endpoint changed")
                        _check(request["max_tokens"] == (600 if arm == "C" else 700), "token limit changed")
                        if arm == "C" or wire["tag"] == "D-final":
                            _check(request.get("response_format") == FINAL_SCHEMA and not request.get("tools"), "final schema/tools")
                        try:
                            body = _unique_json(wire["raw_response_text"]) if wire["raw_response_text"] is not None else None
                        except (ValueError, TypeError):
                            body = None
                        if isinstance(body, dict) and (not wire["error"] or wire["error"]["reason_code"] != "HTTP_ERROR"):
                            _check(wire["response_model"] == body.get("model") and wire["usage"] == body.get("usage"), "response model/usage recording")
                        if wire["response_model"] is None:
                            missing_models += 1
                        else:
                            model_ids[wire["response_model"]] += 1
                        usage = wire["usage"]
                        if usage is None:
                            missing_usage += 1
                        else:
                            for key in ("prompt_tokens", "completion_tokens"):
                                if key in usage:
                                    wire_tokens[key] += int(usage[key])
                        if wire["error"]:
                            transport_errors[wire["error"]["reason_code"]] += 1
                        grouped[wire["origin_unix"]].append(wire)
                    for i, row in enumerate(rows):
                        _check((row["seed"], row["region"], row["arm"]) == (seed, region, arm), "decision identity")
                        _check(row["exit_code"] == exits[i] and row["fallback_to_A"] == bool(bad[i]), "failure status")
                        cell = grouped[row["origin_unix"]]
                        _check(row["llm_calls"] == len(cell) <= {"B": 0, "C": 1, "D": 3}[arm], "per-origin LLM budget/count")
                        _check(0 <= row["tool_calls"] <= (4 if arm == "D" else 0), "tool execution budget")
                        if arm == "D":
                            tags = [w["tag"] for w in cell]
                            _check(tags in ([], ["D1"], ["D1", "D2"], ["D1", "D-final"], ["D1", "D2", "D-final"]), "D turn order")
                            if not row.get("usage_incomplete", False):
                                _check(row["tool_calls"] == _executed_tools(cell), "tool count versus raw responses")
                        elif arm == "C":
                            _check(all(w["tag"] == "C" for w in cell), "C call tag")
                        if row.get("usage_incomplete", False):
                            incomplete_execution_counts += 1
                        if bad[i]:
                            _check(row["action"] == "" and bool(row["failure_reason"]), "failed cell presented as success")
                            failures[row["failure_reason"]] += 1
                        else:
                            _check(row["action"] in ACTIONS and not row["failure_reason"], "successful action")
                            if arm != "B":
                                _check(len(cell) >= 1, "successful LLM decision without a call")
                            actions[f"{arm}/{row['action']}"] += 1
                        llm_calls += len(cell)
                        per_arm_calls[arm] += len(cell)
                        decision_tokens.update(row["tokens"])
                    _check(run["n_origins"] == count and run["n_decision_failures"] == int(bad.sum()), "run decision totals")
                calculated = independent_scores(values, actual)
                scores[seed, region, arm] = calculated
                reported = published[seed, region]["arms"][arm]
                _check(reported["n_origins"] == count and reported["n_cells"] == count * 24, "full-grid scoring population")
                _check(reported["decision_failures"] == int(np.count_nonzero(exits)), "score failure count")
                for key, value in calculated.items():
                    compare(value, reported[key], f"{seed}/{region}/{arm}/{key}")
                origin_counts[arm] += count
                decision_failures[arm] += int(np.count_nonzero(exits))
                stages.append({"seed": seed, "region": region, "arm": arm, **{
                    key: run[key] for key in ("wall_seconds", "user_cpu_seconds", "system_cpu_seconds", "peak_rss_bytes")}})
            for other in ("A", "B", "C"):
                for key in ("mae", "quantile_score"):
                    compare(scores[seed, region, "D"][key] - scores[seed, region, other][key],
                            published[seed, region]["increments"][f"D-{other}"][key], "paired increment")
    for arm in ("A", "B", "C", "D"):
        for key in ("mae", "quantile_score"):
            macro = [float(np.mean([scores[s, r, arm][key] for r in regions])) for s in seeds]
            for seed, value in zip(seeds, macro):
                compare(value, summary["macro_by_seed"][str(seed)][arm][key], "region macro")
            compare(np.mean(macro), summary["across_seeds"][arm][key]["mean"], "seed mean")
            compare(np.std(macro, ddof=1), summary["across_seeds"][arm][key]["sample_std"], "seed sample std")
    return {"passed": True, "execution_commit": manifest["execution_commit"],
        "verified_stage_count": len(stages), "base_batches": len(seeds) * len(regions),
        "origins_by_arm": dict(origin_counts), "decision_failures_by_arm": dict(decision_failures),
        "failure_reasons": dict(failures), "successful_actions": dict(actions),
        "metric_comparisons": len(metric_errors), "maximum_absolute_metric_error": max(metric_errors),
        "metric_tolerance": {"atol": 1e-9, "rtol": 1e-10},
        "llm_calls": llm_calls, "llm_calls_by_arm": dict(per_arm_calls),
        "transport_failures": dict(transport_errors), "response_models": dict(model_ids),
        "response_model_missing_calls": missing_models, "wire_reported_tokens": dict(wire_tokens),
        "usage_missing_calls": missing_usage, "decision_reported_tokens": dict(decision_tokens),
        "execution_counts_incomplete_cells": incomplete_execution_counts,
        "resources": {"stages": stages, "summed_stage_wall_seconds": sum(s["wall_seconds"] for s in stages),
            "summed_user_cpu_seconds": sum(s["user_cpu_seconds"] for s in stages),
            "summed_system_cpu_seconds": sum(s["system_cpu_seconds"] for s in stages),
            "maximum_stage_peak_rss_bytes": max(s["peak_rss_bytes"] for s in stages)},
        "commit_manifest_sha256": commits, "seed_limit": "temperature=0; repeats are not independent datasets or trained models",
        "resource_limit": "stage wall times overlap; their sum is not elapsed matrix wall time; server resources unknown",
        "model_limit": "response IDs/aliases are recorded claims, not proof of server internals or predictive success",
        "audit_scope": "artifact and full-grid metric consistency; passing does not establish forecast improvement",
        "metrics_verified": ["mae", "quantile_score"], "models_called_by_auditor": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        result = audit(_read(args.config))
    except Exception as exc:
        result = {"passed": False, "error_type": type(exc).__name__, "error": str(exc)}
    result.update(auditor_sha256=_sha(__file__), config_sha256=_sha(args.config),
                  checked_at_utc=datetime.now(timezone.utc).isoformat())
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("resources", "commit_manifest_sha256")}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
