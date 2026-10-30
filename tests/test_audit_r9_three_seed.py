"""One synthetic matrix checks independent arithmetic and failure accounting."""
import hashlib
import json

import numpy as np
import pytest

from scripts import audit_r9_three_seed as checker


def test_independent_audit_counts_failures_and_rejects_rehashed_wrong_scores(tmp_path):
    def write_json(path, value):
        path.write_text(json.dumps(value))

    def seal(path):
        entries = {p.name: {"bytes": p.stat().st_size,
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in path.iterdir() if p.name != "COMMIT.json"}
        write_json(path / "COMMIT.json", {"files": entries})

    cfg = {"output_root": str(tmp_path), "seeds": [2026, 2027, 2028], "regions": ["NSW1"],
           "origins_per_region": 2, "source_files": [], "checkpoint": {"revision": "synthetic"}}
    for folder in ("inputs", "labels", "results", "attempts"):
        (tmp_path / folder).mkdir()
    origins = np.array([1640959200, 1641564000])
    identity = {"origin_unix": origins, "target_time_unix": origins[:, None] + np.arange(24) * 3600}
    records = [{"origin_unix": int(t), "target_time_unix": identity["target_time_unix"][i].tolist(),
                "history_prices": [-40, 40] * 36, "history_demand": [8000] * 72,
                "tool_history": {"available_at_aest": ["2021-12-31T12:00:00+00:00"]}}
               for i, t in enumerate(origins)]
    write_json(tmp_path / "inputs/NSW1.json", records)
    write_json(tmp_path / "inputs/manifest.json", {"execution_commit": "synthetic", "sources": {}, "seeds": cfg["seeds"]})
    np.savez(tmp_path / "labels/NSW1.npz", actual=np.zeros((2, 24)), **identity)
    seal(tmp_path / "inputs")
    seal(tmp_path / "labels")
    baseline = np.full((2, 24, 5), 10.0)
    results = []
    for seed in cfg["seeds"]:
        arm_scores = {}
        for arm in ("A", "B", "C", "D"):
            path = tmp_path / str(seed) / "NSW1" / arm
            path.mkdir(parents=True)
            values, exits = baseline.copy(), np.zeros(2, dtype=int)
            if arm == "C":
                exits[0] = 1
            if arm == "D":
                values[0] = 0  # std(last24)=40, SHIFT_DOWN moves the flat forecast by 10
                exits[1] = 1
            np.savez(path / "predictions.npz", quantiles=values, exit_codes=exits, **identity)
            run = {"seed": seed, "region": "NSW1", "arm": arm, "execution_commit": "synthetic",
                   "input_sha256": checker._sha(tmp_path / "inputs/NSW1.json"),
                   "wall_seconds": 1.0, "user_cpu_seconds": .1, "system_cpu_seconds": .05,
                   "peak_rss_bytes": 1024}
            if arm == "A":
                run.update(predict_calls=1, training_calls=0, checkpoint=cfg["checkpoint"])
            else:
                rows, wires, journal = [], [], []
                for i, origin in enumerate(origins):
                    failed = bool(exits[i])
                    action = "" if failed else "SHIFT_DOWN" if arm == "D" else "WIDEN" if arm == "B" else "KEEP"
                    failure = ("INVALID_JSON" if arm == "C" else "INVALID_ACTION") if failed else ""
                    row = {"seed": seed, "region": "NSW1", "arm": arm, "origin_unix": int(origin),
                           "exit_code": int(exits[i]), "fallback_to_A": failed, "action": action,
                           "failure_reason": failure, "llm_calls": int(arm != "B"), "tool_calls": 0,
                           "tokens": {"prompt": 0 if arm == "B" else 11, "completion": 0 if arm == "B" else 7}}
                    rows.append(row)
                    journal.append({"event": "origin_started", "origin_unix": int(origin), "seed": seed})
                    if arm != "B":
                        request = {"model": "qwen3-30b3a", "messages": [], "seed": seed,
                                   "temperature": 0.0, "max_tokens": 600 if arm == "C" else 700}
                        if arm == "C":
                            request["response_format"] = checker.FINAL_SCHEMA
                        content = "not JSON" if failed and arm == "C" else json.dumps({
                            "action": "UNKNOWN" if failed else action, "reason": "synthetic"})
                        body = {"model": "synthetic-server-alias", "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                                "choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
                        wire = {"seed": seed, "region": "NSW1", "arm": arm, "origin_unix": int(origin),
                                "temperature": 0.0, "tag": "C" if arm == "C" else "D1",
                                "requested_model": "qwen3-30b3a", "response_model": body["model"],
                                "url": "http://127.0.0.1:18011/v1/chat/completions", "error": None,
                                "usage": body["usage"], "raw_request_json": json.dumps(request),
                                "raw_response_text": json.dumps(body)}
                        wires.append(wire)
                        journal.append({"event": "llm", **wire})
                    journal.append({"event": "origin_completed", **row})
                for filename, entries in (("decisions.jsonl", rows), ("wire.jsonl", wires)):
                    (path / filename).write_text("".join(json.dumps(e) + "\n" for e in entries))
                journal_path = tmp_path / "attempts" / f"{seed}_NSW1_{arm}.jsonl"
                journal_path.write_text("".join(json.dumps(e) + "\n" for e in journal))
                run.update(base_sha256=checker._sha(path.parent / "A/predictions.npz"), n_origins=2,
                           n_decision_failures=int(exits.sum()), attempt_journal_sha256=checker._sha(journal_path))
            write_json(path / "run.json", run)
            seal(path)
            # Hand-derived for y=0 and all five quantiles equal to 10: MAE=QS=10.
            loss = 5.0 if arm == "D" else 10.0
            arm_scores[arm] = {"mae": loss, "quantile_score": loss, "n_origins": 2,
                               "n_cells": 48, "decision_failures": int(exits.sum())}
        results.append({"seed": seed, "region": "NSW1", "arms": arm_scores,
                        "increments": {f"D-{arm}": {"mae": -5.0, "quantile_score": -5.0} for arm in ("A", "B", "C")}})
    summary = {"execution_commit": "synthetic", "final_test_evaluated": False, "results": results,
               "macro_by_seed": {str(seed): {arm: {key: 5.0 if arm == "D" else 10.0 for key in ("mae", "quantile_score")}
                                            for arm in ("A", "B", "C", "D")} for seed in cfg["seeds"]},
               "across_seeds": {arm: {key: {"mean": 5.0 if arm == "D" else 10.0, "sample_std": 0.0}
                                      for key in ("mae", "quantile_score")} for arm in ("A", "B", "C", "D")}}
    write_json(tmp_path / "results/summary.json", summary)
    seal(tmp_path / "results")
    checked = checker.audit(cfg)
    assert checked["passed"] and checked["verified_stage_count"] == 12
    assert checked["origins_by_arm"] == {"A": 6, "B": 6, "C": 6, "D": 6}
    assert checked["decision_failures_by_arm"]["D"] == 3
    assert checked["llm_calls"] == 12
    assert checked["wire_reported_tokens"] == {"prompt_tokens": 132, "completion_tokens": 84}
    assert checked["response_models"] == {"synthetic-server-alias": 12}
    summary["results"][0]["arms"]["D"]["mae"] = 0  # erroneous success-only score
    write_json(tmp_path / "results/summary.json", summary)
    seal(tmp_path / "results")
    with pytest.raises(ValueError, match="metric mismatch"):
        checker.audit(cfg)
