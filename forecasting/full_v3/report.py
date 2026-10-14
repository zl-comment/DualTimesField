"""Audit the complete single-seed V3 matrix before publishing comparison tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from forecasting.aligned_scoring import load_prediction_artifact, score_predictions
from forecasting.aligned_experiment import verify_snapshot
from .priors import VARIANTS

REGIONS = ("NSW1", "QLD1", "TAS1")
SEED = 2026
SCORE_KEYS = ("mae", "rmse_pooled", "quantile_score_proxy")
NOTICE = ("单 seed 2026、回顾性测试、固定配置比较。历史库可使用查询时已经实现并发布的观测，"
          "包括更早的验证/测试年份；树参数保持静态。这增加了历史信息，不能称为与原始基线"
          "完全相同输入下的 SOTA。QS 为共同五分位数 pinball 均值的两倍，不是精确 CRPS。")


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _base_input_hash(metadata):
    value = metadata.get("base_input_hash")
    for key in ("input_provenance", "effective_input_provenance"):
        if value is None and isinstance(metadata.get(key), dict):
            value = metadata[key].get("base_input_hash")
    if not isinstance(value, str) or not value:
        raise ValueError("V3 prediction lacks canonical base_input_hash provenance")
    return value


def assert_reference_alignment(artifact, reference):
    """Require common samples and base inputs while allowing added V3 information."""
    for key in ("protocol_id", "split", "region", "seed", "sample_hash", "preprocessing_hash", "prediction_space"):
        if artifact.metadata[key] != reference.metadata[key]:
            raise ValueError(f"Reference alignment mismatch: {key}")
    if _base_input_hash(artifact.metadata) != reference.metadata["input_hash"]:
        raise ValueError("Reference alignment mismatch: canonical base input hash")
    if artifact.quantiles is None or reference.quantiles is None:
        raise ValueError("V3 and reference require the common five quantiles")
    for name in ("actual", "origin_unix", "target_time_unix"):
        if not np.array_equal(getattr(artifact, name), getattr(reference, name)):
            raise ValueError(f"Reference alignment mismatch: {name}")


def diagnostic_scores(artifact):
    """Conditional errors and point alarms; these do not replace proper scores.

    Onsets are unique delivery hours with actual >300 and a known contiguous
    predecessor <=300. Their forecast error pools all origin/horizon predictions
    of those hours, with the distinct onset count reported separately.
    """
    point = np.sort(artifact.quantiles, axis=-1)[..., 2]
    y, times = artifact.actual, artifact.target_time_unix
    unique, first, inverse = np.unique(times, return_index=True, return_inverse=True)
    observations = y.ravel()[first]
    if not np.array_equal(observations[inverse].reshape(y.shape), y):
        raise ValueError("Actual price conflicts between overlapping forecast windows")
    onset = np.zeros(len(unique), dtype=bool)
    onset[1:] = ((np.diff(unique) == 3600) & (observations[:-1] <= 300) & (observations[1:] > 300))
    onset_mask = onset[inverse].reshape(y.shape)
    masks = {"ordinary_0_300": (y >= 0) & (y <= 300), "spike_gt_300": y > 300,
             "negative_lt_0": y < 0, "spike_onset": onset_mask}
    slices = {}
    for name, mask in masks.items():
        errors = point[mask] - y[mask]
        slices[name] = dict(n_cells=int(mask.sum()), n_unique_delivery_hours=int(len(np.unique(times[mask]))),
            mae=float(np.abs(errors).mean()) if errors.size else None,
            signed_bias=float(errors.mean()) if errors.size else None)
    positive, alarm = y > 300, point > 300
    tp, fp = int((positive & alarm).sum()), int((~positive & alarm).sum())
    fn, tn = int((positive & ~alarm).sum()), int((~positive & ~alarm).sum())
    return dict(slices=slices, point_alarm_gt_300=dict(true_positive=tp, false_positive=fp,
        false_negative=fn, true_negative=tn, precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        false_alarm_rate=fp / (fp + tn) if fp + tn else None),
        onset_definition="actual >300 and immediately preceding delivery hour is observed and <=300",
        overlap_note="Forecast errors pool origin × horizon cells; unique delivery counts are not independent events.")


def _verify_metrics(saved, computed):
    for key in (*SCORE_KEYS, "n_origins", "n_cells", "horizon"):
        if key not in saved or not np.isclose(saved[key], computed[key], rtol=1e-10, atol=1e-10):
            raise ValueError(f"Saved metric differs from prediction artifact: {key}")
    for key in ("table", "point_definition"):
        if saved.get(key) != computed[key]:
            raise ValueError(f"Saved metric definition differs: {key}")


def _macro(regional):
    counts = np.asarray([row["n_cells"] for row in regional], dtype=np.float64)
    return dict(mae=float(np.mean([row["mae"] for row in regional])),
        quantile_score_proxy=float(np.mean([row["quantile_score_proxy"] for row in regional])),
        rmse_pooled=float(np.sqrt(np.average([row["rmse_pooled"] ** 2 for row in regional], weights=counts))),
        rmse_region_macro=float(np.mean([row["rmse_pooled"] for row in regional])),
        n_cells=int(counts.sum()), regions=len(regional), seed=SEED,
        aggregation="MAE/QS equal region means; rmse_pooled pools squared errors by cell count; rmse_region_macro averages regional RMSE")


def _delta(row, reference):
    return {f"delta_{key}_vs_lightgbm": row[key] - reference[key] for key in SCORE_KEYS}


def write_report(run_root) -> dict:
    root = Path(run_root).resolve()
    manifest = _read(root / "jobs.json")
    jobs = manifest["jobs"]
    expected = {(variant, region, SEED) for variant in VARIANTS for region in REGIONS}
    cells = [(job["variant"], job["region"], job["seed"]) for job in jobs]
    errors = []
    if "source_files" in manifest:
        try:
            verify_snapshot(root, manifest)
        except (ValueError, KeyError, FileNotFoundError, TypeError) as exc:
            errors.append(f"Frozen snapshot audit failed: {exc}")
    if len(cells) != len(expected) or set(cells) != expected:
        errors.append("Manifest must contain exactly five variants × three regions × seed 2026 (15 cells)")
    progress = []
    for job in jobs:
        directory = root / "runs" / job["variant"] / f"seed{job['seed']}" / job["region"]
        path = directory / "state.json"
        state = _read(path) if path.exists() else {"status": "pending"}
        progress.append(dict(variant=job["variant"], region=job["region"], seed=job["seed"],
                             status=state.get("status", "pending"), error=state.get("error")))
    counts = {status: sum(row["status"] == status for row in progress)
              for status in ("completed", "running", "failed", "pending")}
    summary = dict(experiment_id=manifest.get("experiment_id", root.name), comparison_available=False,
        expected_cells=15, declared_cells=len(cells), progress=counts, cells=progress,
        limitations=NOTICE, errors=errors, macro=[], regional=[], reference=None)
    csv_rows = []
    if not errors and counts["completed"] == 15:
        baseline_root = Path(manifest["baseline_root"])
        reference, references, reference_scores, reference_diagnostics = {}, {}, {}, {}
        try:
            if ("baseline_manifest_hash" in manifest
                    and _sha(baseline_root / "jobs.json") != manifest["baseline_manifest_hash"]):
                raise ValueError("Frozen baseline manifest changed")
            for region in REGIONS:
                path = baseline_root / "runs" / "lightgbm" / f"seed{SEED}" / region / "test.npz"
                status_path = path.parent / "status.json"
                if status_path.exists() or "baseline_manifest_hash" in manifest or "source_files" in manifest:
                    baseline_status = _read(status_path)
                    pinned_hash = baseline_status.get("artifact_hashes", {}).get("test.npz")
                    if baseline_status.get("state") != "completed" or not pinned_hash or _sha(path) != pinned_hash:
                        raise ValueError(f"Frozen LightGBM reference checksum/status mismatch: {region}")
                artifact = load_prediction_artifact(path)
                reference_model = artifact.metadata.get("model", artifact.metadata.get("source", "").split(":", 1)[0])
                if (artifact.metadata["region"] != region or artifact.metadata["seed"] != SEED
                        or artifact.metadata["split"] != "test" or reference_model != "lightgbm"):
                    raise ValueError("LightGBM reference cell identity mismatch")
                references[region] = artifact
                reference_scores[region] = score_predictions(artifact)
                reference_diagnostics[region] = diagnostic_scores(artifact)
                reference[region] = dict(path=str(path), sha256=_sha(path))
            reference_macro = _macro(list(reference_scores.values()))
            regional, macro = [], []
            for variant in VARIANTS:
                variant_rows = []
                for region in REGIONS:
                    directory = root / "runs" / variant / f"seed{SEED}" / region
                    state = _read(directory / "state.json")
                    for key in ("source_hash", "config_hash"):
                        if key in manifest and state.get(key) != manifest[key]:
                            raise ValueError(f"Completed cell {key} differs from frozen run")
                    checksums_path = directory / "output_hashes.json"
                    if checksums_path.exists():
                        checksums = _read(checksums_path)
                        required = {"metrics.json", *(f"{split}_predictions.npz" for split in ("selection", "calibration", "test"))}
                        if not required <= checksums.keys():
                            raise ValueError("Completed output checksum manifest is incomplete")
                        for name, digest in checksums.items():
                            path = (directory / name).resolve()
                            if path.parent != directory.resolve() or _sha(path) != digest:
                                raise ValueError(f"Completed output checksum mismatch: {name}")
                    elif "source_hash" in manifest:
                        raise ValueError("Completed cell lacks its output checksum manifest")
                    stored_metrics = _read(directory / "metrics.json")
                    artifact = None
                    # All requested split outputs must be present and truthful
                    # before a job enters the complete experimental table.
                    split_scores = {}
                    for split in ("selection", "calibration", "test"):
                        current = load_prediction_artifact(directory / f"{split}_predictions.npz")
                        if (current.metadata["region"] != region or current.metadata["seed"] != SEED
                                or current.metadata["split"] != split
                                or current.metadata.get("model", current.metadata.get("variant", current.metadata.get("source"))) != variant):
                            raise ValueError(f"Prediction cell identity mismatch: {variant}/{region}/{split}")
                        for key in ("source_hash", "config_hash"):
                            if key in manifest and current.metadata.get(key) != manifest[key]:
                                raise ValueError(f"Prediction {key} differs from frozen run")
                        split_scores[split] = score_predictions(current)
                        _verify_metrics(stored_metrics[split], split_scores[split])
                        if split == "test":
                            artifact = current
                    assert_reference_alignment(artifact, references[region])
                    row = dict(variant=variant, region=region, seed=SEED, **split_scores["test"],
                        **_delta(split_scores["test"], reference_scores[region]),
                        diagnostics=diagnostic_scores(artifact),
                        test_prediction_sha256=_sha(directory / "test_predictions.npz"))
                    regional.append(row)
                    variant_rows.append(row)
                score = _macro(variant_rows)
                macro.append(dict(variant=variant, **score, **_delta(score, reference_macro)))
            macro.sort(key=lambda row: row["mae"])
            summary.update(comparison_available=True, macro=macro, regional=regional,
                reference=dict(model="lightgbm", seed=SEED, artifacts=reference, macro=reference_macro,
                    regional=reference_scores, diagnostics=reference_diagnostics))
            for row in regional:
                csv_rows.append(dict(variant=row["variant"], region=row["region"], seed=SEED,
                    **{key: row[key] for key in ("n_cells", *SCORE_KEYS)}, **_delta(row, reference_scores[row["region"]])))
            for row in macro:
                csv_rows.append(dict(variant=row["variant"], region="ALL", seed=SEED,
                    **{key: row[key] for key in ("n_cells", *SCORE_KEYS, "rmse_region_macro")}, **_delta(row, reference_macro)))
            for region, score in reference_scores.items():
                csv_rows.append(dict(variant="lightgbm_reference", region=region, seed=SEED,
                    **{key: score[key] for key in ("n_cells", *SCORE_KEYS)}, **_delta(score, score)))
            csv_rows.append(dict(variant="lightgbm_reference", region="ALL", seed=SEED,
                **{key: reference_macro[key] for key in ("n_cells", *SCORE_KEYS, "rmse_region_macro")},
                **_delta(reference_macro, reference_macro)))
        except (ValueError, KeyError, FileNotFoundError, TypeError) as exc:
            errors.append(f"Completed-grid audit failed: {exc}")
    _json(root / "summary.json", summary)
    columns = ["variant", "region", "seed", "n_cells", *SCORE_KEYS, "rmse_region_macro",
               *(f"delta_{key}_vs_lightgbm" for key in SCORE_KEYS)]
    with (root / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = [f"# {summary['experiment_id']}", "", NOTICE, "",
             f"进度：完成 {counts['completed']}/15；运行 {counts['running']}；失败 {counts['failed']}；等待 {counts['pending']}。", ""]
    if not summary["comparison_available"]:
        lines += ["完整矩阵尚未通过核验，不发布宏平均、排名或最佳变体。", ""]
        lines += [f"- {error}" for error in errors]
    else:
        lines += ["MAE、QS 为三个地区等权平均；RMSE 合并所有地区的平方误差后开方。地区 RMSE 的算术平均另存于 CSV/JSON。", "",
            "| 模型 | MAE ↓ | pooled RMSE ↓ | QS ↓ | ΔMAE vs LightGBM |",
            "| --- | ---: | ---: | ---: | ---: |"]
        reference = summary["reference"]["macro"]
        lines.append(f"| LightGBM（seed 2026 参考） | {reference['mae']:.4f} | {reference['rmse_pooled']:.4f} | {reference['quantile_score_proxy']:.4f} | 0 |")
        for row in summary["macro"]:
            lines.append(f"| {row['variant']} | {row['mae']:.4f} | {row['rmse_pooled']:.4f} | {row['quantile_score_proxy']:.4f} | {row['delta_mae_vs_lightgbm']:+.4f} |")
        lines += ["", "分地区结果、0–300／>300／负价与尖峰起始小时误差、signed bias、点预测 >300 的误报率和召回率见 summary.json。",
                  "尖峰起始要求存在连续的前一交付小时且价格 ≤300；重叠 origin×horizon 误差不是独立事件样本。"]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    result = write_report(args.run_root)
    print(json.dumps({"comparison_available": result["comparison_available"],
                      "progress": result["progress"], "errors": result["errors"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
