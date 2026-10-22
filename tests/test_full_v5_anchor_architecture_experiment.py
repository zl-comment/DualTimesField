"""Matched architecture budgets, staged isolation, restoration and global barrier."""
from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from forecasting.full_v5_anchor_balance import architecture_experiment as e
from forecasting.full_v5_joint.data import HistoryStore
from forecasting.full_v5_joint.protocol import DAY, HOUR, timestamp


def synthetic_store():
    rng = np.random.default_rng(3)
    n = 12
    origins = timestamp('2019-01-01') + np.arange(n) * DAY
    arrays = dict(origin=origins, actual=np.full((n, 24), 80, dtype=np.float32),
                  baseline=np.broadcast_to(np.array([0, 20, 50, 80, 100], dtype=np.float32), (n, 24, 5)).copy(),
                  z=rng.normal(size=(n, 24, 16)).astype('f'),
                  history=rng.normal(size=(n, 72, 12)).astype('f'),
                  future=rng.normal(size=(n, 24, 31)).astype('f'),
                  lags=rng.normal(size=(n, 24, 3)).astype('f'),
                  trained_before=np.full(n, origins[0] - 2 * DAY),
                  selected_before=np.full(n, origins[0] - DAY))
    return HistoryStore(arrays, candidates=2)


def test_fixed_budget_and_stage_schedule():
    args = Namespace(profile='screen', epochs=6, seed=2026, tree_seed=2026,
                     workers=3, development=True, variants=list(e.VARIANTS))
    assert e.configuration(args)['epochs'] == 6
    assert [e.stage_for('staged', i) for i in range(1, 7)] == ['anchor'] * 3 + ['history'] * 3
    assert all(e.stage_for('identity_anchor', i) == 'joint' for i in range(1, 7))
    args.epochs = 3
    with pytest.raises(ValueError, match='exactly six'):
        e.configuration(args)


def test_six_variants_match_orders_restore_heads_and_staged_anchor_never_drifts(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    store = synthetic_store()
    rows = dict(train=np.array([4, 5]), selection=np.array([7, 8]), evaluation=np.array([10, 11]))
    cfg = dict(seed=2026, epochs=6, profile='screen', batch_size=2)
    logs = {}
    real_predict = e.base.predict
    selection_calls = []
    def observed(*args, **kwargs):
        selection_calls.append(1)
        return real_predict(*args, **kwargs)
    monkeypatch.setattr(e.base, 'predict', observed)
    for variant in e.VARIANTS:
        before = len(selection_calls)
        out = tmp_path / variant
        result = e.fit(store, rows, variant, cfg, out)
        logs[variant] = json.loads((out / 'training.json').read_text())
        assert len(logs[variant]) == 6
        assert len(selection_calls) - before == (3 if variant == 'staged' else 6)
        model, selection = e.restore_checkpoint(out / 'raw.pt')
        replay, same = e.base.restore_checkpoint(out / 'raw.pt', model_builder=e.checkpoint_model)
        assert selection == same
        with torch.no_grad():
            torch.testing.assert_close(model(store.batch([10]))['prediction'],
                                       replay(store.batch([10]))['prediction'], rtol=0, atol=0)
        if variant == 'staged':
            assert result['raw']['epoch'] >= 4 and result['guarded']['epoch'] >= 4
            assert not any(p.requires_grad for p in model.state.parameters())
            assert not any(p.requires_grad for p in model.calibration.parameters())
            assert all('raw' not in entry for entry in logs[variant][:3])
            for entry in logs[variant][3:]:
                assert entry['frozen_anchor']['identical'] is True
                assert entry['frozen_anchor']['max_probe_prediction_difference'] == 0
            pretrained, record = e.restore_checkpoint(out / 'anchor_pretrained.pt')
            assert record['epoch'] == 3 and record['stage'] == 'anchor'
            assert all(p.requires_grad for p in pretrained.state.parameters())
            assert not any(p.requires_grad for p in pretrained.gates.parameters())
        else:
            assert all('full_pinball' in row['terms'] for row in logs[variant])
    expected = [r['order_sha256'] for r in logs['balanced']]
    assert all([r['order_sha256'] for r in log] == expected for log in logs.values())


def test_frozen_anchor_audit_detects_weight_mutation():
    model = e.build_model('staged', 2026)
    batch = synthetic_store().batch([4])
    reference, probe = e.anchor_state(model), e.anchor_probe(model, batch)
    with torch.no_grad():
        next(model.calibration.parameters()).add_(.01)
    with pytest.raises(ValueError, match='Frozen Anchor changed'):
        e.verify_frozen_anchor(model, reference, batch, probe)


def test_all_36_fits_finish_before_evaluation(tmp_path, monkeypatch):
    origins = np.arange(timestamp('2017-01-01'), timestamp('2023-01-01'), HOUR)
    store = Namespace(origins=origins, a={k: np.full(len(origins), origins[0] - DAY)
                                        for k in ('trained_before', 'selected_before')})
    monkeypatch.setattr(e.base, 'load_store', lambda *a: (store, []))
    calls = []
    def train(spec):
        region, folds, cfg, root, output = spec
        result = {}
        assert not any(c == 'evaluate' for c in calls)
        for fold in folds:
            for variant in cfg['variants']:
                directory = Path(output) / region / fold / variant
                directory.mkdir(parents=True)
                hashes = {}
                for mode in ('raw', 'guarded'):
                    path = directory / f'{mode}.pt'
                    path.write_text('synthetic checkpoint')
                    hashes[str(path)] = e.base.digest(path)
                result[f'{region}/{fold}/{variant}'] = dict(selection={}, checkpoints=hashes)
                calls.append('fit')
        return result
    def evaluate(*args, **kwargs):
        assert calls.count('fit') == 36
        assert len(json.loads((tmp_path / 'run' / 'selection_complete.json').read_text())) == 36
        assert kwargs['model_builder'] is e.checkpoint_model
        calls.append('evaluate')
        return {}
    monkeypatch.setattr(e, 'train_region', train)
    monkeypatch.setattr(e.base, 'evaluate', evaluate)
    monkeypatch.setattr(e.base, 'report_markdown', lambda *a: 'synthetic report')
    argv = ['--development', '--workers', '1', '--output', str(tmp_path / 'run')]
    e.main(argv)
    assert calls == ['fit'] * 36 + ['evaluate'] * 36
    with pytest.raises(FileExistsError):
        e.main(argv)
