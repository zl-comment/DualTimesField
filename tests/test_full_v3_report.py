"""Complete-grid reporting and raw-price comparisons with added historical inputs."""
import csv
import hashlib
import json

import numpy as np
import pytest

from forecasting.aligned_scoring import (
    QUANTILE_LEVELS, make_prediction_artifact, save_prediction_artifact, score_predictions,
)
from forecasting.full_v3.priors import VARIANTS
from forecasting.full_v3.report import REGIONS, diagnostic_scores, write_report


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def artifact(region="NSW1", split="test", variant="lightgbm", error=1.):
    actual = np.array([[100.], [500.], [-20.]])
    origins = np.array([1_650_000_000, 1_650_003_600, 1_650_007_200], dtype=np.int64)
    canonical = f"canonical-{region}-{split}"
    metadata = dict(protocol_id="common", split=split, region=region, seed=2026, source=variant,
        model=variant, input_hash=canonical if variant == "lightgbm" else f"effective-{variant}-{canonical}",
        base_input_hash=canonical, sample_hash=f"sample-{region}-{split}", preprocessing_hash=f"preproc-{region}",
        prediction_space="raw_rrp", output_type="quantile")
    quantiles = actual[..., None] + error + np.array([-4., -2., 0., 2., 4.])
    return make_prediction_artifact(actual=actual, origin_unix=origins, target_time_unix=origins[:, None],
        metadata=metadata, quantiles=quantiles, quantile_levels=QUANTILE_LEVELS)


@pytest.fixture
def complete_grid(tmp_path):
    root, baseline = tmp_path / "v3", tmp_path / "baseline"
    jobs = [dict(variant=variant, region=region, seed=2026) for variant in VARIANTS for region in REGIONS]
    dump(root / "jobs.json", dict(jobs=jobs, experiment_id="synthetic-v3", baseline_root=str(baseline)))
    for r, region in enumerate(REGIONS, 1):
        reference = artifact(region=region, error=r)
        save_prediction_artifact(baseline / "runs" / "lightgbm" / "seed2026" / region / "test.npz", reference)
    for v, variant in enumerate(VARIANTS, 1):
        for r, region in enumerate(REGIONS, 1):
            directory = root / "runs" / variant / "seed2026" / region
            metrics = {}
            for name in ("selection", "calibration", "test"):
                current = artifact(region, name, variant, error=r + v)
                save_prediction_artifact(directory / f"{name}_predictions.npz", current)
                metrics[name] = score_predictions(current)
            dump(directory / "metrics.json", metrics)
            dump(directory / "state.json", dict(status="completed"))
    return root


def test_complete_grid_reports_macro_deltas_and_distinguishes_pooled_rmse(complete_grid):
    result = write_report(complete_grid)
    assert result["comparison_available"] and not result["errors"]
    assert len(result["macro"]) == 5 and len(result["regional"]) == 15
    baseline = result["reference"]["macro"]
    assert baseline["mae"] == pytest.approx(2.)
    assert baseline["rmse_region_macro"] == pytest.approx(2.)
    assert baseline["rmse_pooled"] == pytest.approx(np.sqrt(14 / 3))
    for index, row in enumerate(result["macro"], 1):
        assert row["variant"] == VARIANTS[index - 1]
        assert row["mae"] == pytest.approx(2 + index)
        assert row["delta_mae_vs_lightgbm"] == pytest.approx(index)
    with (complete_grid / "results.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 24
    text = (complete_grid / "report.md").read_text()
    assert "单 seed 2026" in text and "增加了历史信息" in text and "不能称为" in text
    assert result["reference"]["artifacts"]["NSW1"]["sha256"]


def test_partial_matrix_does_not_publish_partial_best_or_macro(complete_grid):
    directory = complete_grid / "runs" / VARIANTS[-1] / "seed2026" / "TAS1"
    dump(directory / "state.json", dict(status="running"))
    result = write_report(complete_grid)
    assert not result["comparison_available"]
    assert result["progress"]["completed"] == 14
    assert result["macro"] == [] and result["regional"] == [] and result["reference"] is None
    with (complete_grid / "results.csv").open() as stream:
        assert list(csv.DictReader(stream)) == []
    assert "不发布宏平均" in (complete_grid / "report.md").read_text()


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_seed"])
def test_manifest_requires_exact_predeclared_fifteen_cells(complete_grid, change):
    path = complete_grid / "jobs.json"
    value = json.loads(path.read_text())
    if change == "missing":
        value["jobs"].pop()
    elif change == "duplicate":
        value["jobs"][-1] = value["jobs"][0]
    else:
        value["jobs"][-1]["seed"] = 2027
    dump(path, value)
    result = write_report(complete_grid)
    assert not result["comparison_available"] and result["errors"] and not result["macro"]


@pytest.mark.parametrize("damage", ["base_input", "sample_hash", "preprocessing", "actual", "timestamps", "metric", "missing_calibration"])
def test_completed_grid_fails_closed_on_alignment_and_metric_damage(complete_grid, damage):
    directory = complete_grid / "runs" / VARIANTS[0] / "seed2026" / "NSW1"
    prediction_path = directory / "test_predictions.npz"
    if damage == "missing_calibration":
        (directory / "calibration_predictions.npz").unlink()
    elif damage == "metric":
        metrics = json.loads((directory / "metrics.json").read_text())
        metrics["test"]["mae"] += 10
        dump(directory / "metrics.json", metrics)
    else:
        with np.load(prediction_path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        metadata = json.loads(arrays["metadata_json"].item())
        if damage == "base_input":
            metadata["base_input_hash"] = "different"
        elif damage == "sample_hash":
            metadata["sample_hash"] = "different"
        elif damage == "preprocessing":
            metadata["preprocessing_hash"] = "different"
        elif damage == "actual":
            arrays["actual"] += 1
        else:
            arrays["origin_unix"] += 3600
            arrays["target_time_unix"] += 3600
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))
        np.savez(prediction_path, **arrays)
    result = write_report(complete_grid)
    assert not result["comparison_available"] and result["errors"]
    assert result["macro"] == [] and result["regional"] == []


def test_failed_and_missing_status_are_progress_only(tmp_path):
    jobs = [dict(variant=variant, region=region, seed=2026) for variant in VARIANTS for region in REGIONS]
    dump(tmp_path / "jobs.json", dict(jobs=jobs, baseline_root="unused"))
    dump(tmp_path / "runs" / VARIANTS[0] / "seed2026" / "NSW1" / "state.json",
         dict(status="failed", error="synthetic training failure"))
    result = write_report(tmp_path)
    assert result["progress"] == dict(completed=0, running=0, failed=1, pending=14)
    assert not result["comparison_available"]


def test_source_only_variant_identity_matches_worker_schema(complete_grid):
    for path in complete_grid.glob("runs/*/seed2026/*/*_predictions.npz"):
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        metadata = json.loads(arrays["metadata_json"].item())
        metadata.pop("model")
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))
        np.savez(path, **arrays)
    assert write_report(complete_grid)["comparison_available"]


def test_output_checksum_manifest_must_match_completed_predictions(complete_grid):
    directory = complete_grid / "runs" / VARIANTS[0] / "seed2026" / "NSW1"
    files = ["metrics.json", *(f"{name}_predictions.npz" for name in ("selection", "calibration", "test"))]
    dump(directory / "output_hashes.json", {name: "wrong" for name in files})
    result = write_report(complete_grid)
    assert not result["comparison_available"]
    assert "checksum" in result["errors"][0]


@pytest.mark.parametrize("damage", ["source", "config"])
def test_frozen_source_and_config_audit_is_required_even_before_grid_complete(complete_grid, damage):
    root = complete_grid
    manifest = json.loads((root / "jobs.json").read_text())
    (root / "source").mkdir()
    (root / "source" / "model.py").write_text("original source\n")
    (root / "config.yaml").write_text("frozen: true\n")
    hashes = {"model.py": hashlib.sha256((root / "source" / "model.py").read_bytes()).hexdigest()}
    manifest.update(source_files=hashes,
        source_hash=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        config_hash=hashlib.sha256((root / "config.yaml").read_bytes()).hexdigest())
    dump(root / "jobs.json", manifest)
    dump(root / "runs" / VARIANTS[0] / "seed2026" / "NSW1" / "state.json", dict(status="running"))
    assert not write_report(root)["errors"]
    changed = root / "source" / "model.py" if damage == "source" else root / "config.yaml"
    changed.write_text("changed\n")
    result = write_report(root)
    assert not result["comparison_available"] and not result["macro"]
    assert "Frozen snapshot audit failed" in result["errors"][0]


def test_pinned_baseline_status_hash_rejects_a_replaced_reference(complete_grid):
    root = complete_grid
    manifest = json.loads((root / "jobs.json").read_text())
    from pathlib import Path
    baseline = Path(manifest["baseline_root"])
    dump(baseline / "jobs.json", {})
    manifest["baseline_manifest_hash"] = hashlib.sha256((baseline / "jobs.json").read_bytes()).hexdigest()
    dump(root / "jobs.json", manifest)
    for region in REGIONS:
        folder = baseline / "runs" / "lightgbm" / "seed2026" / region
        dump(folder / "status.json", dict(state="completed", artifact_hashes={
            "test.npz": hashlib.sha256((folder / "test.npz").read_bytes()).hexdigest()}))
    assert write_report(root)["comparison_available"]
    path = baseline / "runs" / "lightgbm" / "seed2026" / "NSW1" / "test.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays["quantiles"] += 1
    np.savez(path, **arrays)
    result = write_report(root)
    assert not result["comparison_available"] and not result["macro"]
    assert "reference checksum/status mismatch" in result["errors"][0]


def test_frozen_baseline_manifest_hash_is_required(complete_grid):
    root = complete_grid
    manifest = json.loads((root / "jobs.json").read_text())
    from pathlib import Path
    baseline = Path(manifest["baseline_root"])
    dump(baseline / "jobs.json", {})
    manifest["baseline_manifest_hash"] = "wrong"
    dump(root / "jobs.json", manifest)
    result = write_report(root)
    assert not result["comparison_available"] and "baseline manifest changed" in result["errors"][0]


def test_reference_source_only_identity_is_valid(complete_grid):
    manifest = json.loads((complete_grid / "jobs.json").read_text())
    from pathlib import Path
    for path in Path(manifest["baseline_root"]).glob("runs/lightgbm/seed2026/*/test.npz"):
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        metadata = json.loads(arrays["metadata_json"].item())
        metadata.pop("model")
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))
        np.savez(path, **arrays)
    assert write_report(complete_grid)["comparison_available"]


def test_onset_diagnostics_use_delivery_predecessor_and_retain_overlap_counts():
    origin = np.array([3600, 7200, 10800], dtype=np.int64)
    times = origin[:, None] + np.arange(2)[None, :] * 3600
    actual = np.array([[100., 500.], [500., 600.], [600., -10.]])
    point = np.array([[400., 200.], [550., 700.], [600., 20.]])
    source = artifact()
    test = make_prediction_artifact(actual=actual, origin_unix=origin, target_time_unix=times,
        metadata=source.metadata, quantiles=point[..., None] + np.array([-4., -2., 0., 2., 4.]),
        quantile_levels=QUANTILE_LEVELS)
    result = diagnostic_scores(test)
    assert result["slices"]["spike_onset"]["n_unique_delivery_hours"] == 1
    assert result["slices"]["spike_onset"]["n_cells"] == 2
    assert result["slices"]["spike_onset"]["mae"] == pytest.approx(175.)
    assert result["slices"]["spike_onset"]["signed_bias"] == pytest.approx(-125.)
    alarm = result["point_alarm_gt_300"]
    assert alarm["true_positive"] == 3 and alarm["false_positive"] == 1 and alarm["false_negative"] == 1
    assert alarm["recall"] == pytest.approx(.75)


def test_overlapping_label_inconsistency_cannot_create_fake_event_onset():
    source = artifact()
    actual = np.array([[100., 500.], [999., 600.]])
    origin = np.array([3600, 7200], dtype=np.int64)
    current = make_prediction_artifact(actual=actual, origin_unix=origin,
        target_time_unix=origin[:, None] + np.arange(2)[None, :] * 3600,
        metadata=source.metadata, quantiles=actual[..., None] + np.array([-4., -2., 0., 2., 4.]),
        quantile_levels=QUANTILE_LEVELS)
    with pytest.raises(ValueError, match="conflicts"):
        diagnostic_scores(current)
