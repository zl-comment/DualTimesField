"""Re-score frozen seed-2026 artifacts and add the five audited baselines.

Run from the repository root with:
    PYTHONPATH=. .venv/bin/python scripts/update_main_comparison_20261022.py
No training, checkpoint selection, or prediction files are modified.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from forecasting.aligned_scoring import (
    assert_comparable,
    load_prediction_artifact,
    score_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs/forecasting"
BASELINES = LOGS / "ddnn_jsu_unclipped_20261022_r1"
JOINT = LOGS / "v5_joint_history_20261020_final_seed2026_r1"
OLD_TABLES = LOGS / "aligned_baselines_20261013_extension/results"
OUTPUT = LOGS / "main_comparison_20261022"
REGIONS = ("NSW1", "QLD1", "TAS1")
NEW_MODELS = {"lightgbm_cp", "lightgbm_qra", "xgboost_cp", "ddnn_jsu", "nhits_qra"}
JOINT_ROLES = {
    "B_retrained": "retrained_base",
    "Full_joint": "full_model",
    "Anchor_joint": "same_checkpoint_intervention",
    "event_off_same_head": "same_checkpoint_intervention",
    "time_shift_off_same_head": "same_checkpoint_intervention",
}
METRICS = (
    "mae", "rmse_pooled", "quantile_score_proxy", "coverage_80", "coverage_90",
    "interval_width_80", "interval_width_90",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_old_rows(old_rows, new_rows, keys):
    indexed = {tuple(row[k] for k in keys): row for row in new_rows}
    for old in old_rows:
        new = indexed[tuple(old[k] for k in keys)]
        for key, value in old.items():
            if isinstance(new[key], (int, float)):
                require(float(value) == new[key], f"Historical value changed: {keys}, {key}")
            else:
                require(value == new[key], f"Historical label changed: {keys}, {key}")


def main():
    report_path = BASELINES / "combined_report.json"
    report = json.loads(report_path.read_text())
    require(report["comparison_available"], "Baseline comparison is incomplete")
    require(report["cross_run_alignment_validation"] == "passed_all_162_artifacts",
            "Baseline source audit is incomplete")
    require(report["progress"]["completed_cells"] == 153, "Expected 153 active cells")
    baseline_rows = read_csv(BASELINES / "combined_report_tables/per_seed.csv")
    require(len(baseline_rows) == 153, "Missing baseline CSV rows")
    grid = {(r["model"], r["region"], int(r["seed"])) for r in baseline_rows}
    models = set(report["model_source_selection"])
    require(len(models) == 17 and NEW_MODELS <= models, "Unexpected baseline model set")
    require(grid == {(m, r, s) for m in models for r in REGIONS for s in (2026, 2027, 2028)},
            "Incomplete baseline model/region/seed grid")
    baseline_rows = [r for r in baseline_rows if r["seed"] == "2026"]
    ddnn = Path(report["model_source_selection"]["ddnn_jsu"])
    require(ddnn.name == "ddnn_jsu_unclipped_20261022_r1", "Old clipped JSU selected")

    archive = json.loads((JOINT / "archive_provenance.json").read_text())
    for relative, expected in archive["sha256"].items():
        require(digest(JOINT / relative) == expected, f"Joint archive changed: {relative}")
    summary = json.loads((JOINT / "summary.json").read_text())
    freeze = json.loads((JOINT / "freeze.json").read_text())
    joint_root = Path(archive["source"])
    require(summary["seed"] == freeze["seed"] == 2026, "Unexpected joint seed")
    require(summary["test"] == "2023–2024", "Unexpected joint test period")

    regional, artifacts = [], []
    for region in REGIONS:
        reference_path = Path(report["model_source_selection"]["lightgbm"]) / "runs/lightgbm/seed2026" / region / "test.npz"
        reference = load_prediction_artifact(reference_path)
        manifest = json.loads((JOINT / region / "manifest.json").read_text())
        jobs = []
        for row in baseline_rows:
            if row["region"] != region:
                continue
            model = row["model"]
            source = Path(report["model_source_selection"][model])
            require(source.name == row["run_id"], "CSV and report source disagree")
            jobs.append(("baseline_" + model,
                         source / "runs" / model / "seed2026" / region / "test.npz",
                         row, "expanded_baseline" if model in NEW_MODELS else "original_baseline"))
        for model, role in JOINT_ROLES.items():
            jobs.append((model, joint_root / region / f"{model}.npz", summary["regions"][region][model], role))
        for model, path, recorded, role in jobs:
            artifact = load_prediction_artifact(path)
            assert_comparable(reference, artifact, allow_different_seed=False)
            # The common loader validates the exact five levels and their order.
            require(artifact.quantiles is not None, f"Missing quantile output: {model}/{region}")
            scores = score_predictions(artifact)
            require(scores["n_origins"] == 17521 and scores["horizon"] == 24,
                    f"Unexpected test shape: {model}/{region}")
            require(scores["point_definition"] == "rearranged_q50", "Unexpected point score")
            for metric in METRICS:
                require(scores[metric] == float(recorded[metric]),
                        f"Re-score mismatch: {model}/{region}/{metric}")
            sha = digest(path)
            if model in JOINT_ROLES:
                require(artifact.metadata["checkpoint_sha256"] == freeze["checkpoints"][region],
                        f"Joint checkpoint mismatch: {model}/{region}")
            elif model in manifest["baseline_artifacts"]:
                require(sha == manifest["baseline_artifacts"][model]["sha256"],
                        f"Historical baseline artifact changed: {model}/{region}")
            scope = ("public_inputs_plus_mature_history_and_errors"
                     if model in JOINT_ROLES and model != "B_retrained" else "public_inputs")
            regional.append(dict(model=model, seed=2026, test_period="2023-2024",
                                 input_scope=scope, role=role, region=region,
                                 n_origins=scores["n_origins"], horizon=scores["horizon"],
                                 **{metric: scores[metric] for metric in METRICS}))
            artifacts.append(dict(model=model, region=region, seed=2026, path=str(path), sha256=sha))
        print(f"{region}: verified and re-scored all 22 frozen outputs", flush=True)

    main_rows = []
    for model in sorted({row["model"] for row in regional}):
        rows = [next(r for r in regional if r["model"] == model and r["region"] == region)
                for region in REGIONS]
        row = {key: rows[0][key] for key in ("model", "seed", "test_period", "input_scope", "role")}
        row.update({r["region"] + "_mae": r["mae"] for r in rows})
        row.update(macro_mae=float(np.mean([r["mae"] for r in rows])),
                   macro_qs=float(np.mean([r["quantile_score_proxy"] for r in rows])),
                   mean_regional_rmse=float(np.mean([r["rmse_pooled"] for r in rows])))
        main_rows.append(row)
    main_rows.sort(key=lambda row: (row["macro_mae"], row["model"]))
    require(len(main_rows) == 22 and len(regional) == 66, "Unexpected combined table size")
    old_main = read_csv(OLD_TABLES / "main_seed2026.csv")
    old_regional = read_csv(OLD_TABLES / "regional_seed2026.csv")
    require(len(old_main) == 17 and len(old_regional) == 51, "Unexpected historical table size")
    verify_old_rows(old_main, main_rows, ("model",))
    verify_old_rows(old_regional, regional, ("model", "region"))
    order = {row["model"]: i for i, row in enumerate(main_rows)}
    regional.sort(key=lambda row: (order[row["model"]], REGIONS.index(row["region"])))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "main_seed2026.csv", main_rows)
    write_csv(OUTPUT / "regional_seed2026.csv", regional)
    input_paths = [report_path, BASELINES / "combined_report_tables/per_seed.csv",
                   JOINT / "summary.json", JOINT / "freeze.json", JOINT / "archive_provenance.json",
                   OLD_TABLES / "main_seed2026.csv", OLD_TABLES / "regional_seed2026.csv"]
    audit = dict(seed=2026, test_period="2023-2024", output_rows=22, regional_rows=66,
                 baseline_models=17, joint_outputs=5, rescored_artifacts=66,
                 same_samples_labels_preprocessing_and_scoring=True,
                 identical_information_scope=False, historical_17_rows_exactly_unchanged=True,
                 old_clipped_ddnn_excluded=True, anchor_development_results_excluded=True,
                 new_training=False, baseline_source_audit=report["cross_run_alignment_validation"],
                 metric_max_abs_difference=0.0,
                 input_sha256={str(p.relative_to(ROOT)): digest(p) for p in input_paths},
                 generator_sha256=digest(Path(__file__)), artifacts=artifacts,
                 output_sha256={p.name: digest(p) for p in (OUTPUT / "main_seed2026.csv", OUTPUT / "regional_seed2026.csv")})
    (OUTPUT / "verification.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 当前电价预测主比较（2026-10-22）", "",
             "zcode 基线扩展及 DDNN-JSU 修复已合入 `research/v5-anchor-calibration-fix`，合并提交 `f771da3`。", "",
             "## 包含 Full_joint 的 seed 2026 比较", "",
             "本表为 17 个基线输出 + 5 个联合方法输出，共 22 行、66 个地区结果。"
             "新增 LightGBM-CP、LightGBM-QRA、XGBoost-CP、DDNN-JSU、NHITS-QRA；DDNN 使用去钳位修复后的预测。", "",
             "所有行统一使用 seed 2026、2023—2024 测试样本、每地区 17,521 个起点、72 小时输入与 24 小时预测、"
             "原始价格空间及公共五分位评分。MAE 使用重排后的 q50，QS=2×五分位平均 pinball；"
             "RMSE 列是三个地区各自 pooled RMSE 的算术平均，不是将三地区误差拼接后再开方。", "",
             "Full_joint 额外使用因果可用的成熟历史与误差，测试期参数冻结、记忆更新；"
             "因此预测样本和评分一致，但历史信息使用范围不同。"
             "Anchor_joint、event_off_same_head、time_shift_off_same_head 是同一检查点的关闭干预，不能当作三个独立重训模型。", "",
             "本次只重算已有预测并补齐比较表，原 17 行数值完全不变。Full_joint 仍为 2026-10-20 冻结版本；"
             "2021/2022 的 balanced、location_only、identity_anchor、asymmetric_scale 开发确认结果尚未在本测试协议下评估，不纳入本表。", "",
             "| 输出 | NSW1 MAE | QLD1 MAE | TAS1 MAE | 宏 MAE | 宏 QS | 地区平均 RMSE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in main_rows:
        values = [row[r + "_mae"] for r in REGIONS] + [row[k] for k in ("macro_mae", "macro_qs", "mean_regional_rmse")]
        lines.append("| " + row["model"] + " | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", "- [22 行主表](main_seed2026.csv) · [66 行地区表](regional_seed2026.csv) · [逐产物重评分与来源核验](verification.json)",
              "- [Full_joint 原冻结报告](../v5_joint_history_20261020_final_seed2026_r1/report.md)", "",
              "## 三种子公平基线汇总", "",
              "静态基线汇总为 17 个模型 × 3 地区 × 3 seed（2026/2027/2028）=153 格。"
              "该表对各 seed 分别评分后汇总，不与上方单种子数值混排；其中 full 是原版静态 Full。", "",
              "- [17 模型宏平均主表](../ddnn_jsu_unclipped_20261022_r1/combined_report_tables/main.csv)",
              "- [51 行地区汇总](../ddnn_jsu_unclipped_20261022_r1/combined_report_tables/regional.csv) · [153 格逐 seed 表](../ddnn_jsu_unclipped_20261022_r1/combined_report_tables/per_seed.csv)",
              "- [修复与完整审计报告](../ddnn_jsu_unclipped_20261022_r1/README.md)", "",
              "这些结果属于固定配置、回顾性比较，不代表完成公平超参搜索，也不构成盲测或统计显著性结论。", "",
              "复算：在仓库根目录运行 `PYTHONPATH=. .venv/bin/python scripts/update_main_comparison_20261022.py`。"
              "脚本需要保留在 outputs 中的冻结预测，先核验并重算全部 66 份预测，全部通过后才写表。", ""]
    (OUTPUT / "README.md").write_text("\n".join(lines))
    print(f"Wrote {len(main_rows)} main rows and {len(regional)} regional rows to {OUTPUT}")


if __name__ == "__main__":
    main()
