import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from forecasting import exploration_events as events


def save_prediction(path, region, seed, probability, controls=False, actual=None):
    actual = np.full((3, 2), 50.) if actual is None else actual
    arrays = {"actual": actual, "origin_unix": np.arange(3) * 3600,
              "quantile_levels": events.LEVELS, "region": region, "seed": seed}
    for event in events.EVENTS:
        for method in (("logistic", "persistence") if controls else ("model",)):
            key = f"{method}_prob_{event}" if controls else f"prob_{event}"
            arrays[key] = np.full(actual.shape, probability)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def manifest_fixture(tmp_path):
    jobs = []
    for variant in ("event_joint_raw", "event_controls"):
        controls = variant == "event_controls"
        for region in events.REGIONS:
            for seed, probability in zip(events.SEEDS, (0.1, 0.5, 0.9)):
                output = tmp_path / variant / region / str(seed)
                for split in events.SPLITS:
                    save_prediction(output / f"{split}.npz", region, seed, probability, controls)
                job = {"variant": variant, "protocol": "static", "region": region, "seed": seed,
                       "output": str(output), "kind": "baseline" if controls else "dual"}
                if controls:
                    job["baseline_kind"] = "event_controls"
                jobs.append(job)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"regions": events.REGIONS, "seeds": events.SEEDS, "jobs": jobs}))
    return path, jobs


def test_event_scores_validate_probabilities_and_empty_events():
    result = events.event_scores(np.full((2, 3), 0.25), np.zeros((2, 3), dtype=bool))
    assert result["brier"] == 0.0625
    assert result["pr_auc"] is None
    assert result["roc_auc"] is None
    assert result["n_positive"] == 0
    json.dumps(result, allow_nan=False)
    for bad in (np.full((2, 3), 1.1), np.full((2, 3), np.nan)):
        with pytest.raises(ValueError, match="finite"):
            events.event_scores(bad, np.zeros((2, 3)))


def test_persistence_fits_training_conditionals_with_empty_group_fallback():
    labels = {"spike": np.array([[True, False], [True, True]]), "past_spike": np.array([False, True])}
    rates = events.fit_persistence(labels, "spike")
    assert rates == {"0": 0.5, "1": 1.0}
    np.testing.assert_equal(events.predict_persistence(labels, "spike", rates), [[0.5, 0.5], [1., 1.]])
    labels["past_spike"][:] = False
    assert events.fit_persistence(labels, "spike") == {"0": 0.75, "1": 0.75}


def test_event_report_scores_each_seed_without_ensembling(tmp_path):
    manifest, _ = manifest_fixture(tmp_path)
    result = events.summarize(manifest, protocols=["static"])
    models = result["protocols"]["static"]["test"]
    assert set(models) == {"event_joint_raw", "event_controls/logistic", "event_controls/persistence"}
    for model in models.values():
        value = model["region_seed_summary"]["NSW1"]["spike"]["brier"]
        assert value["mean"] == pytest.approx(np.mean([0.01, 0.25, 0.81]))
        assert value["std"] == pytest.approx(np.std([0.01, 0.25, 0.81], ddof=1))
        assert value["n_seeds"] == 3
        assert model["region_seed_summary"]["NSW1"]["spike"]["pr_auc"]["mean"] is None
    assert (tmp_path / "reports" / "events_report.json").exists()
    assert (tmp_path / "reports" / "events_report.md").exists()


def test_report_rejects_incomplete_seed_matrix(tmp_path):
    manifest, jobs = manifest_fixture(tmp_path)
    manifest.write_text(json.dumps({"jobs": jobs[:-1]}))
    with pytest.raises(ValueError, match="Incomplete"):
        events.summarize(manifest)


def test_report_rejects_actual_mismatch_across_seeds(tmp_path):
    manifest, jobs = manifest_fixture(tmp_path)
    job = jobs[0]
    save_prediction(Path(job["output"]) / "test.npz", job["region"], job["seed"], 0.1,
                    actual=np.full((3, 2), 60.))
    with pytest.raises(ValueError, match="actual prices differ"):
        events.summarize(manifest)


def test_event_archive_requires_correct_seed_and_full_hourly_grid(tmp_path):
    p = tmp_path / "event.npz"
    save_prediction(p, "NSW1", 2026, 0.2)
    with pytest.raises(ValueError, match="seed/region"):
        events.load_event_forecast(p, "NSW1", 2027)
    with np.load(p) as z:
        arrays = dict(z)
    arrays["origin_unix"][1] += 1
    np.savez(p, **arrays)
    with pytest.raises(ValueError, match="hourly grid"):
        events.load_event_forecast(p, "NSW1", 2026)


def tiny_dataset(origins):
    origins = np.asarray(origins)
    raw = np.resize(np.array([400., -20., 50., 90.], dtype=np.float32), 250)
    unix = np.arange(len(raw)) * 3600
    profiles = {int(unix[i]): np.array([[np.sin(i)], [np.sin(i + 1)]], dtype=np.float32) for i in origins}
    return SimpleNamespace(origin_indices=origins, target_values_raw=raw, output_hours=2,
                           delivery_unix_seconds=unix, future_exogenous_by_origin=profiles,
                           quantile_exogenous_by_origin=profiles, ctf_exogenous_by_origin=profiles,
                           calendar_values=np.arange(len(raw), dtype=float)[:, None] / 250,
                           origin_context_values=np.ones(len(raw), dtype=float) * 7)


def test_run_fits_training_only_and_exports_event_probabilities(tmp_path, monkeypatch):
    datasets = {"train": tiny_dataset(range(30, 70)), "validation": tiny_dataset(range(100, 110)),
                "test": tiny_dataset(range(150, 160))}
    monkeypatch.setattr(events, "build_region_datasets", lambda *_: datasets)
    monkeypatch.setattr(events, "load_forecast_config", lambda *_: {"loss": {}})
    result = events.run(tmp_path / "config.yaml", "NSW1", 2026, tmp_path / "run")
    assert result["logistic_parameters"]["fit_split"] == "train"
    assert result["models"]["spike"]["training_observations"] == 80
    labels = events._labels(datasets["train"], 300., 0.)
    features = events._features(datasets["train"], labels)
    assert np.all(features[:, -1] == 7)  # known origin gas context is included
    for split in events.SPLITS:
        path = tmp_path / "run" / f"{split}.npz"
        data = events.load_event_forecast(path, "NSW1", 2026, controls=True)
        assert set(data.probabilities) == {"logistic", "persistence"}
        with np.load(path) as archive:
            assert "point" not in archive.files and "quantile" not in archive.files
    json.loads((tmp_path / "run" / "events.json").read_text())
