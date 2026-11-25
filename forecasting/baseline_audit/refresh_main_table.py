"""Re-score current sources, validate Moirai replay, and refresh the main table.

Stage publication separately and retain the complete previous table. Never
rewrite training archives or silently substitute diagnostic ensemble outputs.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from forecasting.baseline_audit.recompute import digest, fingerprint, git, write
from scripts import update_latest_main_comparison as publisher

ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "4c5847ddee31fc7b901ae7889973f3bf8ee5beea"


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if git("status", "--porcelain"):
        raise RuntimeError("Commit the refresh implementation before execution")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    current = ROOT / "logs/forecasting/main_comparison_20261023"
    replay = ROOT / "outputs/forecasting/moirai_real_replay_20261125"
    completion = json.loads((replay / "completed.json").read_text())
    for path, expected in completion["output_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Moirai replay evidence changed: {path}")
    replay_result = json.loads((replay / "results.json").read_text())
    replay_manifest = json.loads((replay / "manifest.json").read_text())
    if not all(replay_result[k] for k in ("passed", "full_test_replay", "originals_and_source_unchanged")):
        raise ValueError("Moirai full replay incomplete or failed")
    if set(replay_result["regions"]) != set(publisher.REGIONS):
        raise ValueError("Moirai replay region grid incomplete")
    for region in replay_result["regions"].values():
        if region["n_origins"] != 17521 or not region["old_new_all_nine_native_quantiles"]["array_equal"]:
            raise ValueError("Moirai replay does not establish full numeric equivalence")
        checks = region["archived_seed_comparisons"]
        if set(checks) != {"2026", "2027", "2028"} or not all(v["array_equal"] for v in checks.values()):
            raise ValueError("Moirai replay does not match all archived seeds exactly")
    source_paths = [ROOT / p for p in git("ls-files", "forecasting", "scripts/update_latest_main_comparison.py").splitlines()
                    if Path(p).suffix in (".py", ".json", ".yaml")]
    source_hashes = fingerprint(source_paths)
    old_hashes = fingerprint(sorted(p for p in current.iterdir() if p.is_file()))
    old_audit = json.loads((current / "verification.json").read_text())
    original_artifacts = {v["path"]: v["sha256"] for v in old_audit["artifacts"]}
    if fingerprint([Path(p) for p in original_artifacts]) != original_artifacts:
        raise ValueError("An original prediction archive changed before refresh")
    refreshed_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    evidence = {"baseline_commit": BASE_COMMIT, "execution_commit": git("rev-parse", "HEAD"),
                "branch": git("branch", "--show-current"), "refreshed_at": refreshed_at,
                "source_sha256": source_hashes,
                "source_fingerprint": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
                "previous_table_sha256": old_hashes, "original_artifacts_sha256": original_artifacts,
                "moirai_replay_results_sha256": digest(replay / "results.json"),
                "moirai_replay_execution_commit": replay_manifest["execution_commit"],
                "scope": "Current 25-row/225-cell table plus 18 historical controls; no training, inference, selection changes or R27 ensemble insertion"}
    write(output / "manifest.json", evidence)
    shutil.copytree(current, output / "previous_table")
    publisher.OUTPUT = output / "staged_table"
    publisher.ARCHIVE = output / "staged_joint_archive"
    publisher.ARCHIVE.mkdir()
    publisher.main([])
    staged = publisher.OUTPUT
    audit = json.loads((staged / "verification.json").read_text())
    new_artifacts = {v["path"]: v["sha256"] for v in audit["artifacts"]}
    if new_artifacts != original_artifacts:
        raise ValueError("Refresh unexpectedly changed prediction source selection")
    for artifact in audit["artifacts"]:
        if artifact["model"] == "moirai2_zeroshot":
            if replay_manifest["input_sha256"].get(artifact["path"]) != artifact["sha256"]:
                raise ValueError("Published Moirai source is not covered by the real replay")
    previous, updated = rows(current / "main.csv"), rows(staged / "main.csv")
    if [r["model"] for r in previous] != [r["model"] for r in updated]:
        raise ValueError("Main-table model ordering unexpectedly changed")
    numeric_diffs = []
    for old, new in zip(previous, updated):
        for key in old:
            if key.endswith(("_mean", "_std")) and old[key] != new[key]:
                if not old[key] or not new[key]:
                    raise ValueError("Metric availability unexpectedly changed")
                numeric_diffs.append({"model": old["model"], "metric": key,
                                      "old": float(old[key]), "new": float(new[key]),
                                      "abs_diff": abs(float(old[key])-float(new[key]))})
    summary = {"outputs": audit["outputs"], "cells": audit["cells"], "source_cells": audit["source_cells"],
               "model_order_unchanged": True, "source_selection_unchanged": True,
               "numeric_differences": numeric_diffs,
               "max_abs_main_table_difference": max((r["abs_diff"] for r in numeric_diffs), default=0.),
               "moirai_full_replay_verified": True}
    audit["refresh_20261125"] = {
        **{k: v for k, v in evidence.items() if k not in
           ("source_sha256", "original_artifacts_sha256")},
        "execution_manifest": str(output / "manifest.json"), "comparison": summary}
    write(staged / "verification.json", audit)
    note = ("2026-11-25 修复后刷新：已从原预测重新评分全部243个来源格（主表225格＋历史18格），"
            "保持三地区等权、三个seed分别评分后汇总的口径。Moirai真实权重全测试集对照确认修前/修后预测"
            "及原九格归档逐元素一致，主表数值无需因该修复改变；本次不是重新训练全部模型。"
            "R25/R27更正结果单独归档，不将组合增益混入基线行。详见"
            "[修复及重算](../../../forecasting/baseline_audit/RESULTS.md)与"
            "[Moirai全量对照](../../../forecasting/baseline_audit/MOIRAI_REPLAY.md)。")
    for name in ("README.md", "VALIDATION.md"):
        path = staged / name
        content = path.read_text().replace("2026-10-24 更新", "2026-11-25 重新评分更新")
        if name == "VALIDATION.md":
            content = content.replace("（2026-10-24）", "（2026-11-25）")
        heading, rest = content.split("\n", 1)
        path.write_text(heading + "\n\n" + note + "\n" + rest)
    if (fingerprint(source_paths) != source_hashes or
            fingerprint([Path(p) for p in original_artifacts]) != original_artifacts or
            fingerprint([Path(p) for p in old_hashes]) != old_hashes):
        raise ValueError("Code, original predictions or published table changed during staging")
    write(output / "comparison.json", summary)
    for path in staged.iterdir():
        if path.is_file():
            shutil.copyfile(path, current / path.name)
    write(output / "completed.json", {"published_sha256": fingerprint(sorted(p for p in current.iterdir() if p.is_file())),
                                       "comparison": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
