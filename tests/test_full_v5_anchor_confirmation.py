"""Confirmation isolation, exact replay, asymmetric tails, and acceptance gates."""
import copy
import fcntl
import json

import numpy as np
import pytest
import torch

from forecasting.full_v5_anchor_balance import confirmation_experiment as e
from forecasting.full_v5_anchor_balance.confirmation_data import DeviceHistoryStore
from forecasting.full_v5_anchor_balance.confirmation_model import VARIANTS, build_model
from forecasting.full_v5_anchor_balance.confirmation_report import acceptance
from test_full_v5_anchor_architecture_experiment import synthetic_store


def test_default_hook_preserves_exact_old_formula_and_asymmetric_initialization():
    torch.set_num_threads(2)
    reference = build_model('balanced', 2026)
    next_random = torch.rand(8)
    asymmetric = build_model('asymmetric_scale', 2026)
    torch.testing.assert_close(torch.rand(8), next_random, rtol=0, atol=0)
    for name, value in reference.state_dict().items():
        if not name.startswith('calibration.2.'):
            torch.testing.assert_close(asymmetric.state_dict()[name], value, rtol=0, atol=0)
    assert sum(p.numel() for p in asymmetric.parameters()) - sum(p.numel() for p in reference.parameters()) == 25
    batch = synthetic_store().batch([4, 7])
    torch.testing.assert_close(reference(batch)['prediction'], asymmetric(batch)['prediction'], rtol=0, atol=0)
    torch.nn.init.normal_(reference.calibration[-1].weight)
    features = torch.randn(2, 24, 32)
    baseline = batch['baseline']
    raw = reference.calibration(features)
    loc = 150 * raw[..., 0].tanh()
    scale = 1 + .5 * raw[..., 1].tanh()
    original = baseline[..., 2:3] + loc[..., None] + scale[..., None] * (baseline - baseline[..., 2:3])
    actual, actual_loc = reference.calibrate_anchor(baseline, features)
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    torch.testing.assert_close(actual_loc, loc, rtol=0, atol=0)


def test_asymmetric_tails_independent_positive_ordered_and_differentiable():
    model = build_model('asymmetric_scale')
    b = torch.tensor([-1000., -100., -10., 0., 15000.]).repeat(2, 24, 1)
    features = torch.randn(2, 24, 32)
    original, _ = model.calibrate_anchor(b, features)
    with torch.no_grad():
        model.calibration[-1].bias[1] = .7
    lower, _ = model.calibrate_anchor(b, features)
    torch.testing.assert_close(lower[..., 2:], original[..., 2:], rtol=0, atol=0)
    assert (lower[..., :2] < original[..., :2]).all()
    with torch.no_grad():
        model.calibration[-1].bias[2] = -.8
    both, _ = model.calibrate_anchor(b, features)
    torch.testing.assert_close(both[..., :3], lower[..., :3], rtol=0, atol=0)
    assert (both[..., 3:] < lower[..., 3:]).all()
    assert torch.all(both.diff(dim=-1) >= 0)
    both[..., 0].sum().backward()
    assert model.calibration[-1].bias.grad[1].abs() > 0
    assert model.calibration[-1].bias.grad[2] == 0
    for bias in (-100., 100.):
        with torch.no_grad():
            model.calibration[-1].bias.fill_(bias)
        for base in (b, torch.zeros_like(b), b.abs().sort(dim=-1).values):
            q, location = model.calibrate_anchor(base, features)
            assert torch.isfinite(q).all() and torch.all(q.diff(dim=-1) >= 0)
            torch.testing.assert_close(q[..., 2], base[..., 2] + location)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:7', marks=pytest.mark.skipif(
    torch.cuda.device_count() < 8, reason='GPU adapter smoke check requires local device 7'))])
def test_device_adapter_exact_no_current_targets_and_mature_memory(device):
    reference = synthetic_store()
    store = DeviceHistoryStore(reference, np.arange(12), device)
    assert store.verify(reference, np.arange(12))['exact']
    batch = {k: v.clone() for k, v in store.batch([8]).items()}
    reference.a['actual'][8] += 12345
    # CPU shares the backing array; update the device copy explicitly on CUDA.
    store.data['actual'][8] = torch.as_tensor(reference.a['actual'][8], device=device)
    new = store.batch([8])
    assert 'actual' not in new
    for key in batch:
        torch.testing.assert_close(new[key], batch[key], rtol=0, atol=0)
    assert (store.targets([8]) > 10000).all()
    unprepared = DeviceHistoryStore(synthetic_store(), [4], device)
    with pytest.raises(ValueError, match='Unprepared'):
        unprepared.batch([5])


def small_config():
    cfg = e.configuration(['cpu'] * 3)
    cfg.update(epochs=2, batch_size=2, prediction_batch_size=2)
    return cfg


def small_rows():
    return dict(train=np.array([4, 5]), selection=np.array([7, 8]), evaluation=np.array([10, 11]))


def test_four_fits_match_order_checkpoint_restore_and_evaluation_labels_unused(tmp_path):
    torch.set_num_threads(2)
    store = DeviceHistoryStore(synthetic_store(), np.arange(12))
    logs = []
    for variant in VARIANTS:
        directory = tmp_path / variant
        record = e.fit(store, small_rows(), variant, 2026, small_config(), directory, 'synthetic-run')
        assert record['epochs_completed'] == 2
        assert e.fit(store, small_rows(), variant, 2026, small_config(), directory, 'synthetic-run') == record
        logs.append([row['order_sha256'] for row in json.loads((directory / 'training.json').read_text())])
        model, selected = e.base.restore_checkpoint(directory / 'raw.pt', model_builder=build_model)
        assert selected == record['selection']['raw']
        out = model(store.batch([10]))
        assert torch.all(out['prediction'].diff(dim=-1) >= 0)
        if variant == 'identity_anchor':
            torch.testing.assert_close(out['anchor'], store.batch([10])['baseline'], rtol=0, atol=0)
    assert all(log == logs[0] for log in logs)
    other = DeviceHistoryStore(synthetic_store(), np.arange(12))
    other.data['actual'][[10, 11]] += 99999
    other.a['actual'][[10, 11]] += 99999
    changed = tmp_path / 'changed_eval'
    e.fit(other, small_rows(), 'balanced', 2026, small_config(), changed, 'synthetic-run')
    before, _ = e.base.restore_checkpoint(tmp_path / 'balanced' / 'raw.pt', model_builder=build_model)
    after, _ = e.base.restore_checkpoint(changed / 'raw.pt', model_builder=build_model)
    for key, value in before.state_dict().items():
        torch.testing.assert_close(value, after.state_dict()[key], rtol=0, atol=0)


def test_interruption_after_epoch_commit_replays_exactly(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    store = DeviceHistoryStore(synthetic_store(), np.arange(12))
    reference, resumed = tmp_path / 'reference', tmp_path / 'resumed'
    cfg = small_config()
    expected = e.fit(store, small_rows(), 'asymmetric_scale', 2027, cfg, reference, 'run')
    write_json = e.base.write_json
    def interrupted(path, value):
        if path == resumed / 'training.json' and len(value) == 1:
            raise RuntimeError('simulated interruption after atomic epoch commit')
        return write_json(path, value)
    monkeypatch.setattr(e.base, 'write_json', interrupted)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        e.fit(store, small_rows(), 'asymmetric_scale', 2027, cfg, resumed, 'run')
    monkeypatch.setattr(e.base, 'write_json', write_json)
    actual = e.fit(store, small_rows(), 'asymmetric_scale', 2027, cfg, resumed, 'run')
    assert actual['selection'] == expected['selection']
    for mode in ('raw', 'guarded'):
        a, _ = e.base.restore_checkpoint(reference / f'{mode}.pt', model_builder=build_model)
        b, _ = e.base.restore_checkpoint(resumed / f'{mode}.pt', model_builder=build_model)
        for name, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match='identity changed'):
        e.fit(store, small_rows(), 'asymmetric_scale', 2027, cfg, resumed, 'changed-run')


def test_fixed_configuration_and_all_72_global_barrier(tmp_path):
    cfg = e.configuration(['cpu'] * 3)
    assert cfg['epochs'] == 20 and cfg['seeds'] == [2026, 2027, 2028]
    assert len(e.expected_keys(cfg)) == 72
    path = tmp_path / 'checkpoint.pt'
    path.write_text('fixture')
    record = dict(epochs_completed=20, checkpoints={str(path): e.base.digest(path)})
    completed = {key: copy.deepcopy(record) for key in e.expected_keys(cfg)}
    e.verify_barrier(completed, cfg)
    key = next(iter(completed))
    incomplete = dict(completed)
    del incomplete[key]
    with pytest.raises(ValueError, match='All confirmation fits'):
        e.verify_barrier(incomplete, cfg)
    completed[key]['epochs_completed'] = 19
    with pytest.raises(ValueError, match='Incomplete epoch budget'):
        e.verify_barrier(completed, cfg)
    completed[key]['epochs_completed'] = 20
    path.write_text('tampered')
    with pytest.raises(ValueError, match='checkpoint changed'):
        e.verify_barrier(completed, cfg)


def test_shape_diagnostics_physical_units_and_spring_failure_not_hidden():
    b = np.broadcast_to(np.array([0., 20., 50., 80., 100.]), (2, 24, 5))
    a, f, y = b + 10, b + 3, np.full((2, 24), 40.)
    d = e.shape_diagnostics(y, b, a, f)
    assert d['mean_abs_delta'] == 10 and d['mean_abs_correction'] == 7
    assert d['cancelled_magnitude'] == 14
    assert d['cancellation_fraction'] == pytest.approx(14 / 17)
    assert 2 * np.mean(d['full']['pinball_by_quantile']) == pytest.approx(e.score(y, f)['qs'])
    cfg = e.configuration(['cpu'] * 3)
    results = {}
    for key in e.expected_keys(cfg):
        result = dict(B={g: dict(mae=10., qs=10.) for g in ('all', 'season/spring')})
        for mode in ('raw', 'guarded'):
            result[mode] = dict(metrics={g: dict(full=dict(mae=9., qs=9.)) for g in ('all', 'season/spring')})
        results[key] = result
    assert acceptance(results, cfg, 'asymmetric_scale', 'guarded', cfg['seeds'])['passed']
    results['TAS1/2021/2026/asymmetric_scale']['guarded']['metrics']['season/spring']['full']['mae'] = 20.
    gate = acceptance(results, cfg, 'asymmetric_scale', 'guarded', cfg['seeds'])
    assert not gate['passed']
    assert not gate['folds']['2021']['checks']['each_spring_noninferior_to_B']


def test_end_to_end_evaluation_waits_for_all_arms_and_generates_report(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    cfg = small_config()
    cfg.update(regions=['TAS1'], folds=['2021'], seeds=[2026], devices=['cpu'])
    store = synthetic_store()
    # Put the fixture in spring so the predeclared seasonal gate is exercised.
    store.origins[:] += 280 * e.base.DAY
    monkeypatch.setattr(e.base, 'load_store', lambda *a: (store, []))
    monkeypatch.setattr(e.base, 'split_rows', lambda *a: small_rows())
    output = tmp_path / 'run'
    def prepare(*args, **kwargs):
        output.mkdir()
        return dict(run_id='fixture-run')
    monkeypatch.setattr(e, 'prepare', prepare)
    monkeypatch.setattr(e, 'verify_manifest', lambda manifest: None)
    calls = []
    def pool(worker, specs):
        if worker is e.evaluate_region:
            assert calls == ['train']
            barrier = json.loads((output / 'selection_complete.json').read_text())
            assert len(barrier) == 4
            calls.append('evaluate')
        else:
            assert not (output / 'selection_complete.json').exists()
            calls.append('train')
        result = {}
        for spec in specs:
            result.update(worker(spec))
        return result
    monkeypatch.setattr(e, 'run_pool', pool)
    e.run(cfg, 'fixture-bundles', output)
    assert calls == ['train', 'evaluate']
    assert json.loads((output / 'progress.json').read_text())['stage'] == 'complete'
    assert json.loads((output / 'training_order_audit.json').read_text())['matched']
    assert (output / 'results.csv').read_text().count('\n') > 25
    assert (output / 'acceptance.json').exists()
    assert len(list(output.rglob('*.npz'))) == 8


def test_cli_rejects_duplicate_runner_without_touching_output(tmp_path):
    output = tmp_path / 'run'
    output.mkdir()
    argv = ['--development', '--output', str(output)]
    with (tmp_path / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='active runner'):
            e.main(argv + ['--resume'])
    with pytest.raises(FileExistsError, match='Output exists'):
        e.main(argv)
    assert not list(output.iterdir())
