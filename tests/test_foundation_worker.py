"""Worker mechanics for the foundation batch: resume, tamper, isolation, reuse.

Uses a deterministic mock adapter (tests/fixtures/mock_foundation_adapter.py)
on a synthetic mini run so no model package or GPU is required. The mock is a
pure function of the input blocks; any label leak or re-prediction on resume
changes recorded call logs and fails the test.
"""
from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

import forecasting.foundation_experiment as fe
from forecasting.aligned_experiment import sha
from forecasting.aligned_inputs import INPUT_KEYS, AlignedSplit
from forecasting.aligned_scoring import load_prediction_artifact
from forecasting.common.foundation_api import PredictionView

SEEDS = (2026, 2027, 2028)
N_ORIGINS = 40


def _mini_split(seed: int, name: str) -> AlignedSplit:
    rng = np.random.default_rng(seed)
    inputs = {
        "history_values": rng.normal(size=(N_ORIGINS, 72, 2)).astype(np.float32),
        "history_calendar": rng.normal(size=(N_ORIGINS, 72, 10)).astype(np.float32),
        "future_calendar": rng.normal(size=(N_ORIGINS, 24, 10)).astype(np.float32),
        "origin_context": rng.normal(size=(N_ORIGINS, 1)).astype(np.float32),
        "future_exogenous": rng.normal(size=(N_ORIGINS, 24, 11)).astype(np.float32),
        "ctf_exogenous": rng.normal(size=(N_ORIGINS, 24, 1)).astype(np.float32),
        "quantile_exogenous": rng.normal(size=(N_ORIGINS, 24, 3)).astype(np.float32),
    }
    actual = rng.normal(loc=60, scale=10, size=(N_ORIGINS, 24))
    origins = np.arange(N_ORIGINS, dtype=np.int64) * 3600 + 1_600_000_000
    targets = origins[:, None] + np.arange(1, 25, dtype=np.int64) * 3600
    metadata = {"protocol_id": "mini-v1", "region": "NSW1", "split": name,
                "input_hash": "ih", "sample_hash": "sh", "preprocessing_hash": "ph",
                "target_mean": 60.0, "target_std": 10.0}
    return AlignedSplit(inputs, actual, origins, targets, metadata,
                        (actual - 60.0) / 10.0)


def _save_mini_cache(region_root: Path, splits: dict[str, AlignedSplit]) -> None:
    files = {}
    for name, split in splits.items():
        directory = region_root / name
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {**{f"input__{k}": v for k, v in split.inputs.items()},
                  "actual": split.actual, "target": split.target,
                  "origin_unix": split.origin_unix, "target_time_unix": split.target_time_unix}
        for key, values in arrays.items():
            np.save(directory / f"{key}.npy", values, allow_pickle=False)
            files[str((directory / f"{key}.npy").relative_to(region_root))] = sha(directory / f"{key}.npy")
        (directory / "metadata.json").write_text(json.dumps(split.metadata))
        files[str((directory / "metadata.json").relative_to(region_root))] = sha(directory / "metadata.json")
    (region_root / "hashes.json").write_text(json.dumps(files, sort_keys=True))


@pytest.fixture()
def mini_run(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "forecasting" / "tabpfn_v3").mkdir(parents=True)
    (project / "forecasting" / "tabpfn_v3" / "requirements.lock.txt").write_text("mock\n")
    run_root = tmp_path / "run"
    run_root.mkdir()
    splits = {name: _mini_split(index, name) for index, name in
              enumerate(("train", "selection", "calibration", "test"))}
    _save_mini_cache(run_root / "data_cache" / "NSW1", splits)
    config = yaml.safe_load((Path(__file__).resolve().parents[1] /
                             "configs/experiments/aemo_forecast_foundation_20261023.yaml").read_text())
    config["experiment"]["models"] = ["tabpfn_v3"]
    config["experiment"]["execution_scope"] = "single_model"
    config["experiment"]["chunk_size"] = {"tabpfn_v3": 16}
    config["experiment"]["seed_reuse"] = {"mode": "verify_and_copy", "verification_origins": 8}
    config["project_root"] = str(project)
    (run_root / "config.yaml").write_text(yaml.safe_dump(config))
    source_root = run_root / "source"
    source_root.mkdir()
    (source_root / "forecasting").mkdir()
    (source_root / "forecasting" / "__init__.py").write_text("")
    (source_root / "src").mkdir()
    (source_root / "src" / "marker.txt").write_text("mini\n")
    files = {str(p.relative_to(source_root)): sha(p) for p in sorted(source_root.rglob("*")) if p.is_file()}
    import hashlib
    manifest = {
        "run_id": run_root.name, "project_root": str(project),
        "source_hash": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "source_files": files, "config_hash": sha(run_root / "config.yaml"),
        "cache_hashes": {"NSW1": sha(run_root / "data_cache" / "NSW1" / "hashes.json")},
        "cache_info": {"NSW1": {"test": {"origins": N_ORIGINS}}}, "data_hashes": {},
        "jobs": [{"id": f"tabpfn_v3/{seed}/NSW1", "model": "tabpfn_v3", "region": "NSW1",
                  "seed": seed, "output": f"runs/tabpfn_v3/seed{seed}/NSW1", "resource": "gpu"}
                 for seed in SEEDS],
        "models": ["tabpfn_v3"], "regions": ["NSW1"], "seeds": list(SEEDS),
        "foundation": {"schema": "foundation-expansion-20261023-v1", "batch": "second_20261023"},
        "interpreters": {"tabpfn_v3": "unused-in-tests"},
        "model_env_locks": {"tabpfn_v3": {
            "lock_file": "forecasting/tabpfn_v3/requirements.lock.txt",
            "sha256": sha(project / "forecasting" / "tabpfn_v3" / "requirements.lock.txt")}},
        "foundation_from": {"run_root": str(tmp_path / "parent"), "run_id": "parent",
                            "jobs_hash": "x", "source_hash": "x", "config_hash": "x",
                            "cache_hashes": {}},
    }
    fe.atomic_json(run_root / "jobs.json", manifest)

    # Point the registry entry at the mock adapter; tests never need model packages.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mock_foundation_adapter", Path(__file__).parent / "fixtures" / "mock_foundation_adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(fe, "verify_foundation", lambda root, m: Path(m["foundation_from"]["run_root"]))
    import forecasting.common.foundation_registry as registry
    monkeypatch.setattr(registry, "load_foundation_adapter",
                        lambda model_id: module.Adapter)
    yield run_root, manifest, splits, module
    shutil.rmtree(tmp_path, ignore_errors=True)


def _cell(run_root, seed=2026):
    return run_root / "runs" / "tabpfn_v3" / f"seed{seed}" / "NSW1"


def test_worker_completes_with_label_free_view_and_scope_metadata(mini_run):
    run_root, manifest, splits, module = mini_run
    fe.worker(run_root, "tabpfn_v3/2026/NSW1", "cpu")
    cell = _cell(run_root)
    assert (cell / "status.json").exists()
    status = json.loads((cell / "status.json").read_text())
    assert status["state"] == "completed"
    artifact = load_prediction_artifact(cell / "test.npz")
    assert artifact.metadata["smoke"] is False
    assert artifact.metadata["output_type"] == "quantile"
    assert artifact.metadata["model"] == "tabpfn_v3"
    assert artifact.metadata["source"] == f"tabpfn_v3:{manifest['source_hash']}"
    scope = artifact.metadata["model_metadata"]["foundation"]["declared_scope"]
    assert scope["weight_training"] is False and scope["fit_requires_train_labels"] is True
    assert artifact.quantiles.shape == (N_ORIGINS, 24, 5)
    budget = artifact.metadata["model_metadata"]["budget"]
    assert budget["n_origins"] == N_ORIGINS and "seconds_per_1000_origins" in budget
    chunks = sorted((cell / "predictions" / "chunks").glob("chunk_*.npz"))
    assert len(chunks) == int(np.ceil(N_ORIGINS / 16))
    assert (cell / "predictions" / "chunks" / "chunks_manifest.json").exists()


def test_prediction_view_carries_no_labels(mini_run):
    run_root, manifest, splits, module = mini_run
    view, actual, split = fe._load_test_view(run_root, manifest, "NSW1")
    assert isinstance(view, PredictionView)
    assert not hasattr(view, "actual") and not hasattr(view, "target")
    assert set(view.inputs) == set(INPUT_KEYS)
    np.testing.assert_array_equal(actual, np.asarray(splits["test"].actual))
    with pytest.raises(AttributeError):
        view.actual  # noqa: B018


def test_resume_repredicts_only_missing_chunks_bitwise(mini_run):
    run_root, manifest, splits, module = mini_run
    fe.worker(run_root, "tabpfn_v3/2026/NSW1", "cpu")
    cell = _cell(run_root)
    first = load_prediction_artifact(cell / "test.npz")
    calls_before = [json.loads(l) for l in (cell / "models" / "adapter_calls.jsonl").read_text().splitlines() if json.loads(l)["call"] == "predict"]
    # Simulate a crash mid-run: outputs vanish, one middle chunk is missing,
    # but the chunk ledger and fit state must survive the restart.
    (cell / "test.npz").unlink()
    (cell / "metrics.json").unlink()
    chunks_dir = cell / "predictions" / "chunks"
    (chunks_dir / "chunk_00001.npz").unlink()
    ledger = json.loads((chunks_dir / "chunks_manifest.json").read_text())
    del ledger["chunk_00001.npz"]
    (chunks_dir / "chunks_manifest.json").write_text(json.dumps(ledger))
    status = json.loads((cell / "status.json").read_text())
    status["state"] = "running"
    fe.atomic_json(cell / "status.json", status)
    fe.worker(run_root, "tabpfn_v3/2026/NSW1", "cpu")
    second = load_prediction_artifact(cell / "test.npz")
    np.testing.assert_array_equal(first.quantiles, second.quantiles)
    records = [json.loads(line) for line in
               (cell / "models" / "adapter_calls.jsonl").read_text().splitlines()]
    predicts = [r for r in records if r["call"] == "predict"]
    fits = [r for r in records if r["call"] == "fit"]
    assert len(predicts) == len(calls_before) + 1  # only the missing chunk reran
    assert len([r for r in fits if not r.get("skipped")]) == 1  # refit never happened
    fit_state = json.loads((cell / "models" / "fit_state.json").read_text())
    assert "fitted_at" in fit_state


def test_tampered_chunk_is_rejected_and_repredicted(mini_run):
    run_root, manifest, splits, module = mini_run
    fe.worker(run_root, "tabpfn_v3/2026/NSW1", "cpu")
    cell = _cell(run_root)
    first = load_prediction_artifact(cell / "test.npz")
    chunk = cell / "predictions" / "chunks" / "chunk_00002.npz"
    with np.load(chunk, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays["quantiles"] = arrays["quantiles"] + 5.0  # wellformed but wrong
    np.savez_compressed(chunk, **arrays)
    (cell / "test.npz").unlink()
    (cell / "metrics.json").unlink()
    status = json.loads((cell / "status.json").read_text())
    status["state"] = "running"
    fe.atomic_json(cell / "status.json", status)
    fe.worker(run_root, "tabpfn_v3/2026/NSW1", "cpu")
    second = load_prediction_artifact(cell / "test.npz")
    np.testing.assert_array_equal(first.quantiles, second.quantiles)


def test_cache_and_config_tamper_fails_worker(mini_run):
    run_root, manifest, splits, module = mini_run
    hashes = run_root / "data_cache" / "NSW1" / "hashes.json"
    original = hashes.read_text()
    hashes.write_text(json.dumps({"bogus": "0" * 64}))
    with pytest.raises(ValueError):
        fe.worker(run_root, "tabpfn_v3/2026/NSW1", "cpu")
    hashes.write_text(original)
    config_path = run_root / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["experiment"]["cpu_threads_per_job"] = 999
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        fe.worker(run_root, "tabpfn_v3/2026/NSW1", "cpu")


def test_label_changes_never_reach_the_adapter(mini_run):
    run_root, manifest, splits, module = mini_run
    fe.worker(run_root, "tabpfn_v3/2026/NSW1", "cpu")
    cell = _cell(run_root)
    first = load_prediction_artifact(cell / "test.npz")
    # Rewrite every label in the cache; the mock prediction must not move.
    splits2 = {name: _mini_split(index + 100, name) for index, name in
               enumerate(("train", "selection", "calibration", "test"))}
    region = run_root / "data_cache" / "NSW1"
    _save_mini_cache(region, splits2)
    manifest["cache_hashes"]["NSW1"] = sha(region / "hashes.json")
    fe.atomic_json(run_root / "jobs.json", manifest)
    status = json.loads((cell / "status.json").read_text())
    status["state"] = "running"
    status.pop("artifact_hashes", None)
    fe.atomic_json(cell / "status.json", status)
    (cell / "test.npz").unlink()
    (cell / "metrics.json").unlink()
    fe.worker(run_root, "tabpfn_v3/2026/NSW1", "cpu")
    second = load_prediction_artifact(cell / "test.npz")
    np.testing.assert_array_equal(first.quantiles, second.quantiles)
    assert not np.array_equal(first.actual, second.actual)  # scorer sees new labels


def test_deterministic_seed_reuse_copies_with_evidence(mini_run):
    run_root, manifest, splits, module = mini_run
    fe.worker(run_root, "tabpfn_v3/2026/NSW1", "cpu")
    reference = load_prediction_artifact(_cell(run_root, 2026) / "test.npz")
    fe.worker(run_root, "tabpfn_v3/2027/NSW1", "cpu")
    cell27 = _cell(run_root, 2027)
    reused = load_prediction_artifact(cell27 / "test.npz")
    np.testing.assert_array_equal(reference.quantiles, reused.quantiles)
    assert reused.metadata["seed"] == 2027
    budget = reused.metadata["model_metadata"]["budget"]
    assert budget["seed_reuse"]["reference_seed"] == 2026
    assert budget["seed_reuse"]["comparison"] == "bitwise_equal"
    chunk_meta = None
    for path in (cell27 / "predictions" / "chunks").glob("chunk_*.npz"):
        with np.load(path, allow_pickle=False) as archive:
            chunk_meta = json.loads(archive["meta_json"].item())
        break
    assert chunk_meta["reuse_of_seed"] == 2026
    # The verification subset ran once (8 origins), everything else was copied.
    calls = [json.loads(line) for line in
             (cell27 / "models" / "adapter_calls.jsonl").read_text().splitlines()]
    predicts = [call for call in calls if call["call"] == "predict"]
    assert len(predicts) == 1 and len(predicts[0]["indices"]) == 8
