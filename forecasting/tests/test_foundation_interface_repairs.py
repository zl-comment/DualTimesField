"""Checkpoint provenance and native diagnostic persistence, without real inference."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from forecasting.common.foundation_weights import verified_checkpoint
from forecasting.common.foundation_api import PredictionView
from forecasting import foundation_experiment as fe
from forecasting.foundation_audit import _audit_auxiliary
from forecasting.moirai2_zeroshot.adapter import Adapter as MoiraiAdapter
from forecasting.moirai2_zeroshot import model as moirai_model


# Reuse the existing synthetic worker fixture without changing historical tests.
_spec = importlib.util.spec_from_file_location(
    "foundation_worker_fixture", Path(__file__).parents[2] / "tests/test_foundation_worker.py")
_fixture_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture_module)
mini_run = _fixture_module.mini_run


def checkpoint_fixture(tmp_path):
    module = tmp_path / "project/forecasting/mock/model.py"
    module.parent.mkdir(parents=True)
    cache = tmp_path / "hf"
    revision = "a" * 40
    snapshot = cache / "hub/models--org--model/snapshots" / revision
    snapshot.mkdir(parents=True)
    data = b"audited weights"
    (snapshot / "model.safetensors").write_bytes(data)
    (snapshot / "config.json").write_bytes(b"{}")
    source = {"weights": {"repo": "org/model", "revision": revision,
                         "file": "model.safetensors", "size_bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest(),
                         "config_sha256": hashlib.sha256(b"{}").hexdigest()}}
    module.with_name("source.json").write_text(json.dumps(source))
    return module, cache, snapshot, source


def test_checkpoint_ignores_main_and_other_snapshots(tmp_path):
    module, cache, expected, _ = checkpoint_fixture(tmp_path)
    unrelated = expected.parent / ("b" * 40)
    unrelated.mkdir()
    refs = expected.parent.parent / "refs"
    refs.mkdir()
    (refs / "main").write_text(unrelated.name)
    snapshot, identity = verified_checkpoint(str(module), cache)
    assert snapshot == expected
    assert identity["revision"] == "a" * 40


@pytest.mark.parametrize("file", ["model.safetensors", "config.json"])
def test_checkpoint_rejects_changed_bytes(tmp_path, file):
    module, cache, snapshot, _ = checkpoint_fixture(tmp_path)
    (snapshot / file).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verified_checkpoint(str(module), cache)


def test_checkpoint_rejects_short_revision_and_missing_exact_snapshot(tmp_path):
    module, cache, snapshot, source = checkpoint_fixture(tmp_path)
    source["weights"]["revision"] = "a" * 12
    module.with_name("source.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="full commit"):
        verified_checkpoint(str(module), cache)
    source["weights"]["revision"] = "b" * 40
    module.with_name("source.json").write_text(json.dumps(source))
    with pytest.raises(FileNotFoundError):
        verified_checkpoint(str(module), cache)


@pytest.mark.parametrize('name', ['chronos2_zeroshot', 'moirai2_zeroshot', 'timesfm25_xreg'])
def test_native_loaders_receive_only_verified_local_snapshot(monkeypatch, tmp_path, name):
    from importlib import import_module
    native = import_module(f'forecasting.{name}.model')
    snapshot = tmp_path / 'exact-snapshot'
    checks, loads = [], []
    def verify(model_file, hf_home):
        checks.append((model_file, hf_home))
        return snapshot, {}
    monkeypatch.setattr(native, 'verified_checkpoint', verify)
    class Model:
        def to(self, *args, **kwargs): return self
        def eval(self): return self
    def load(*args, **kwargs):
        loads.append((args, kwargs))
        result = Model()
        result.model = Model()
        return result
    if name == 'chronos2_zeroshot':
        monkeypatch.setitem(sys.modules, 'chronos', SimpleNamespace(
            Chronos2Pipeline=SimpleNamespace(from_pretrained=load)))
        native.load_pipeline('cpu', hf_home='selected-cache')
    elif name == 'moirai2_zeroshot':
        monkeypatch.setitem(sys.modules, 'uni2ts', SimpleNamespace())
        monkeypatch.setitem(sys.modules, 'uni2ts.model', SimpleNamespace())
        monkeypatch.setitem(sys.modules, 'uni2ts.model.moirai2', SimpleNamespace(
            Moirai2Module=SimpleNamespace(from_pretrained=load),
            Moirai2Forecast=lambda **kwargs: Model()))
        native.load_forecast(prediction_length=24, target_dim=1, feat_dynamic_real_dim=10,
                             past_feat_dynamic_real_dim=1, context_length=72,
                             device='cpu', hf_home='selected-cache')
    else:
        monkeypatch.setitem(sys.modules, 'timesfm', SimpleNamespace(
            TimesFM_2p5_200M_torch=SimpleNamespace(from_pretrained=load)))
        native.load_model('cpu', hf_home='selected-cache')
    assert checks == [(native.__file__, 'selected-cache')]
    assert loads[0][0] == (str(snapshot),)
    assert loads[0][1]['local_files_only'] is True


def test_moirai_start_and_native_quantile_axes(monkeypatch):
    origin = np.array([1700000000, 1700003600])
    history = np.zeros((2, 72, 2))
    view = PredictionView({"history_values": history,
                           "history_calendar": np.zeros((2, 72, 10)),
                           "future_calendar": np.zeros((2, 24, 10))},
                          origin, origin[:, None] + np.arange(24) * 3600,
                          {"target_mean": 10., "target_std": 2.})
    seen = []
    def predict(predictor, entries):
        seen.extend(entries)
        return np.broadcast_to(np.arange(9)[None, :, None], (len(entries), 9, 24)).copy()
    monkeypatch.setattr(moirai_model, "predict_batch", predict)
    adapter = MoiraiAdapter.__new__(MoiraiAdapter)
    adapter.context = SimpleNamespace(test_view=view)
    adapter.predictor = object()
    adapter.batch_origins = 1
    result = fe._predict_indices(adapter, view, np.array([1, 0]), "mock")
    assert seen[0]["start"].timestamp() + 72 * 3600 == origin[1]
    assert seen[1]["start"].timestamp() + 72 * 3600 == origin[0]
    np.testing.assert_array_equal(seen[0]["target"], np.full(72, 10.))
    assert result["point"].shape == (2, 24)
    assert result["auxiliary_quantiles"].shape == (2, 24, 3)
    np.testing.assert_array_equal(result["auxiliary_quantiles"][0, 0], [0., 4., 8.])


def install_diagnostic_mock(monkeypatch, module):
    # Keep the fixture's model registration but attach deterministic native
    # diagnostics, making the test exercise full worker persistence and reuse.
    original = module.Adapter.predict_chunk
    def predict(self, indices):
        payload = original(self, indices)
        self._aux = payload["quantiles"][:, :, [0, 2, 4]]
        return payload
    monkeypatch.setattr(module.Adapter, "predict_chunk", predict)
    monkeypatch.setattr(module.Adapter, "auxiliary_levels", (0.1, 0.5, 0.9), raising=False)
    monkeypatch.setattr(module.Adapter, "auxiliary_quantiles", lambda self: self._aux, raising=False)


def test_native_diagnostics_survive_chunks_resume_and_seed_copy(mini_run, monkeypatch):
    root, manifest, _, module = mini_run
    install_diagnostic_mock(monkeypatch, module)
    fe.worker(root, "tabpfn_v3/2026/NSW1", "cpu")
    cell = root / "runs/tabpfn_v3/seed2026/NSW1"
    with np.load(cell / "auxiliary_quantiles.npz", allow_pickle=False) as data:
        original = data["quantiles"].copy()
        assert original.shape == (40, 24, 3)
        assert json.loads(data["levels_json"].item()) == [0.1, 0.5, 0.9]
    status = json.loads((cell / "status.json").read_text())
    assert "auxiliary_quantiles.npz" in status["artifact_hashes"]
    log = cell / "models/adapter_calls.jsonl"
    before = log.read_text()
    # Force assembly recovery, retaining all hash-verified prediction chunks.
    (cell / "auxiliary_quantiles.npz").unlink()
    with pytest.raises(ValueError, match="Completed artifact corrupt"):
        fe.worker(root, "tabpfn_v3/2026/NSW1", "cpu")
    # An interrupted (not successfully published) attempt may rebuild outputs.
    status['state'] = 'failed'
    (cell / "status.json").write_text(json.dumps(status))
    fe.worker(root, "tabpfn_v3/2026/NSW1", "cpu")
    assert sum('"call": "predict"' in line for line in log.read_text().splitlines()) == sum(
        '"call": "predict"' in line for line in before.splitlines())
    with np.load(cell / "auxiliary_quantiles.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["quantiles"], original)
    fe.worker(root, "tabpfn_v3/2027/NSW1", "cpu")
    copied = root / "runs/tabpfn_v3/seed2027/NSW1"
    with np.load(copied / "auxiliary_quantiles.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["quantiles"], original)
        assert json.loads(data["metadata_json"].item())["reuse"]["reference_seed"] == 2026
    for path in (copied / "predictions/chunks").glob("chunk_*.npz"):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["meta_json"].item())
            assert meta["reuse_of_seed"] == 2026
            assert "auxiliary_quantiles" in data
            assert fe._verify_chunk(path, meta["identity_hash"], data["indices"],
                                    require_auxiliary=True) is not None
    from forecasting.aligned_scoring import load_prediction_artifact
    artifact = load_prediction_artifact(copied / 'test.npz')
    identity = artifact.metadata['model_metadata']['adapter_identity']
    status = json.loads((copied / 'status.json').read_text())
    chunks, levels = [], []
    for path in sorted((copied / 'predictions/chunks').glob('chunk_*.npz')):
        with np.load(path, allow_pickle=False) as saved:
            chunks.append({'auxiliary_quantiles': saved['auxiliary_quantiles']})
            levels.append(json.loads(saved['meta_json'].item())['auxiliary_levels'])
    assert _audit_auxiliary(copied, artifact, identity, status, chunks, levels) == 'verified'
    chunks[0]['auxiliary_quantiles'][0, 0, 0] += 1
    with pytest.raises(ValueError, match='reassembly'):
        _audit_auxiliary(copied, artifact, identity, status, chunks, levels)


def test_seed_reuse_checks_diagnostics_not_just_point():
    point = np.ones((3, 24))
    left = {"point": point, "auxiliary_quantiles": np.zeros((3, 24, 3))}
    right = {"point": point, "auxiliary_quantiles": np.ones((3, 24, 3))}
    assert not fe._reuse_compare(left, right, 0.)[0]
    assert not fe._reuse_compare(left, {"point": point}, 0.)[0]


def test_auxiliary_audit_distinguishes_legacy_from_broken_new_artifact(tmp_path):
    assert _audit_auxiliary(tmp_path, None, {}, {}, [], []) == 'legacy_missing'
    with pytest.raises(ValueError, match='Missing declared'):
        _audit_auxiliary(tmp_path, None, {'auxiliary_schema': 'native_diagnostics_n_h_q_v1'},
                         {}, [], [])
