"""Runner contracts: rotation, purging, provenance, budget and checkpoint replay."""
from argparse import Namespace
import json

import numpy as np
import pytest
import torch

from forecasting.full_v5_anchor_balance import experiment as e
from forecasting.full_v5_anchor_balance.objective import build_model
from forecasting.full_v5_anchor_balance.selection import blend, select_candidate
from forecasting.full_v5_joint.protocol import DAY, HOUR, timestamp


def hourly(start='2017-01-01', end='2023-01-01'):
    return np.arange(timestamp(start), timestamp(end), HOUR)


@pytest.mark.parametrize('fold', ['2021', '2022'])
def test_rotating_screen_sampling_covers_months_and_all_hours(fold):
    origins = hourly()
    rows = e.split_rows(origins, fold, 'screen')
    for role, ids in rows.items():
        selected = origins[ids] + 10 * HOUR
        assert set((selected % DAY) // HOUR) == set(range(24))
        dates, counts = np.unique(selected // DAY, return_counts=True)
        assert len(dates) > 360
        assert counts.max() == (1 if role == 'train' else 4)
        assert counts.min() == counts.max()  # Purged final day is absent.
        metadata = e.sample_manifest(origins, ids)
        assert len(metadata['months']) >= 12
        assert sum(metadata['hours'].values()) == len(ids)
        assert metadata['origin_sha256'] == e.sample_manifest(origins, ids)['origin_sha256']
    assert origins[rows['train']].max() + DAY < origins[rows['selection']].min()
    assert origins[rows['selection']].max() + DAY < origins[rows['evaluation']].min()


def test_full_preserves_all_eligible_hours_and_strict_purge():
    origins = hourly()
    rows = e.split_rows(origins, '2022', 'full')
    edges = e.FOLDS['2022']
    for role, start, end in zip(rows, edges, edges[1:]):
        expected = np.flatnonzero((origins >= timestamp(start)) & (origins + DAY < timestamp(end)))
        np.testing.assert_array_equal(rows[role], expected)
        assert origins[rows[role]].max() == timestamp(end) - DAY - HOUR
    with pytest.raises(ValueError, match='>=2023'):
        e.split_rows(np.append(origins, timestamp('2023-01-01')), '2022', 'full')


def test_profile_budget_is_explicit():
    args = Namespace(profile='full', epochs=None, development=True, seed=2026, tree_seed=2026, variants=['legacy'])
    assert e.configuration(args)['epochs'] == 20
    args.epochs = 3
    with pytest.raises(ValueError, match='exactly 20'):
        e.configuration(args)
    args.profile = 'screen'
    assert e.configuration(args)['epochs'] == 3
    args.development = False
    with pytest.raises(ValueError, match='development'):
        e.configuration(args)


@pytest.mark.parametrize('variant', e.VARIANTS)
@pytest.mark.parametrize('fallback', [False, True])
def test_guard_checkpoint_persists_alpha_epoch_and_frozen_identity(tmp_path, variant, fallback):
    baseline = np.broadcast_to(np.array([-2, -1, 0, 1, 2], dtype=np.float32), (2, 24, 5)).copy()
    actual = np.ones((2, 24), np.float32)
    correction = -1 if fallback else .5
    candidate = select_candidate(actual, baseline, baseline + correction, baseline + correction, epoch=2)
    assert candidate['fallback'] == fallback
    model = build_model(variant)
    path = tmp_path / 'guard.pt'
    e.save_checkpoint(path, model, variant, 2026, candidate)
    recovered, decision = e.restore_checkpoint(path)
    assert decision == candidate
    for key, value in model.state_dict().items():
        torch.testing.assert_close(recovered.state_dict()[key], value, rtol=0, atol=0)
    assert all(p.requires_grad == (variant != 'identity_anchor') for p in recovered.calibration.parameters())
    np.testing.assert_array_equal(blend(baseline, baseline + correction, decision['alpha']),
                                  blend(baseline, baseline + correction, candidate['alpha']))


def metadata_tree(tmp_path, region='NSW1'):
    paths = []
    for year, half in [(y, False) for y in range(2017, 2023)] + [(2022, True)]:
        path = tmp_path / 'oof' / region / (f'{year}H2' if half else str(year)) / 'bundle.npz'
        path.parent.mkdir(parents=True)
        path.with_suffix('.json').write_text(json.dumps(dict(year=year, final_half=half, region=region, seed=2026)))
        paths.append(path)
    return paths


def test_loader_rejects_wrong_fold_and_seed_before_loading_arrays(tmp_path, monkeypatch):
    paths = metadata_tree(tmp_path)
    def unexpected(*args, **kwargs):
        pytest.fail('Invalid metadata must fail before HistoryStore loads data')
    monkeypatch.setattr(e.HistoryStore, 'from_paths', unexpected)
    with pytest.raises(ValueError, match='region/seed'):
        e.load_store(tmp_path, 'NSW1', 99)
    meta_path = paths[-1].with_suffix('.json')
    meta = json.loads(meta_path.read_text())
    meta['year'] = 2023
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='seven'):
        e.load_store(tmp_path, 'NSW1', 2026)


def test_main_finishes_all_selection_before_any_evaluation(tmp_path, monkeypatch):
    origins = hourly()
    store = Namespace(origins=origins, a={k: origins - DAY for k in ('trained_before', 'selected_before')})
    monkeypatch.setattr(e, 'load_store', lambda *a: (store, []))
    calls = []
    def fit(*args):
        assert (tmp_path / 'run' / 'manifest.json').exists()
        assert not any(c == 'evaluate' for c in calls)
        directory = args[-1]
        directory.mkdir(parents=True)
        for mode in ('raw', 'guarded'):
            (directory / f'{mode}.pt').write_text('synthetic checkpoint')
        calls.append('fit')
    def evaluate(*args):
        assert calls.count('fit') == 6
        assert (tmp_path / 'run' / 'selection_complete.json').exists()
        calls.append('evaluate')
        return {'synthetic': True}
    monkeypatch.setattr(e, 'fit', fit)
    monkeypatch.setattr(e, 'evaluate', evaluate)
    monkeypatch.setattr(e, 'report_markdown', lambda *a: 'synthetic report')
    argv = ['--region', 'NSW1', '--development', '--output', str(tmp_path / 'run')]
    e.main(argv)
    assert calls == ['fit'] * 6 + ['evaluate'] * 6
    manifest = json.loads((tmp_path / 'run' / 'manifest.json').read_text())
    assert len(manifest['samples']) == 2
    assert manifest['scope'] == 'retrospective_development_not_final_test'
    with pytest.raises(FileExistsError):
        e.main(argv)


def test_provenance_mapping_counts_cutoffs_and_rejects_noncausal_rows(tmp_path):
    origins = np.array([timestamp('2020-01-02'), timestamp('2020-08-03'), timestamp('2021-01-02')])
    train = np.array([timestamp('2019-07-01')] * 2 + [timestamp('2020-07-01')])
    select = np.array([timestamp('2019-12-31')] * 2 + [timestamp('2020-12-31')])
    store = Namespace(origins=origins, a=dict(trained_before=train, selected_before=select))
    paths = []
    for i, year in ((0, '2020'), (2, '2021')):
        path = tmp_path / year / 'bundle.npz'
        path.parent.mkdir()
        path.with_suffix('.json').write_text(json.dumps(dict(trained_through=int(train[i]), selected_through=int(select[i]))))
        paths.append(path)
    audit = e.provenance_manifest(store, np.arange(3), paths)
    assert audit['every_row_cutoffs_strictly_before_origin'] is True
    assert [g['n_origins'] for g in audit['cutoff_groups']] == [2, 1]
    assert [g['oof_folds'] for g in audit['cutoff_groups']] == [['2020'], ['2021']]
    assert audit['cutoff_groups'][0]['last_origin'] == int(origins[1])
    store.a['selected_before'][0] = origins[0]
    with pytest.raises(ValueError, match='must precede'):
        e.provenance_manifest(store, np.arange(3), paths)


def test_diagnostic_seasons_price_boundaries_and_cross_partition():
    origins = np.array([timestamp(f'2022-{m:02}-15') for m in range(1, 13)])
    means = np.array([-1, 0, 49, 50, 99, 100, 149, 150, 299, 300, 500, -2])
    baseline = np.broadcast_to(means[:, None, None], (12, 24, 5)).copy()
    groups = e.evaluation_groups(origins, baseline)
    np.testing.assert_array_equal(np.flatnonzero(groups['season/summer']), [0, 1, 11])
    np.testing.assert_array_equal(np.flatnonzero(groups['season/spring']), [8, 9, 10])
    np.testing.assert_array_equal(np.flatnonzero(groups['B_q50_mean/[0,50)']), [1, 2])
    np.testing.assert_array_equal(np.flatnonzero(groups['B_q50_mean/[300,inf)']), [9, 10])
    crosses = [v for k, v in groups.items() if '|' in k]
    assert len(crosses) == 24
    np.testing.assert_array_equal(np.stack(crosses).sum(axis=0), np.ones(12))
    assert groups['2022H1'].sum() == groups['2022H2'].sum() == 6
    assert all(groups[f'2022Q{q}'].sum() == 3 for q in range(1, 5))
    # A future 24h forecast mean, not its first hour, defines the band.
    changed = baseline.copy().astype(float)
    changed[0, 1:, 2] = 100
    assert e.evaluation_groups(origins, changed)['B_q50_mean/[50,100)'][0]


def test_evaluation_reports_all_group_counts_without_reselection(tmp_path, monkeypatch):
    origins = np.array([timestamp(f'2022-{m:02}-15') for m in (1, 4, 7, 10)])
    baseline = np.broadcast_to(np.array([0, 20, 50, 80, 100], dtype=np.float32), (4, 24, 5)).copy()
    actual = baseline[..., 2] + 1
    store = Namespace(origins=origins, a=dict(actual=actual, baseline=baseline))
    monkeypatch.setattr(e, 'restore_checkpoint', lambda p: (None, dict(alpha=.5, epoch=1, fallback=False)))
    monkeypatch.setattr(e, 'predict', lambda *a: dict(anchor=baseline + .5, full=baseline + 1))
    def forbidden(*a, **kw):
        pytest.fail('Evaluation cannot reselect alpha or epochs')
    monkeypatch.setattr(e, 'select_candidate', forbidden)
    result = e.evaluate(store, {'evaluation': np.arange(4)}, 'balanced', {'batch_size': 16}, tmp_path)
    assert result['groups']['all'] == dict(n_origins=4, n_cells=96)
    assert result['groups']['season/spring|B_q50_mean/[50,100)'] == dict(n_origins=1, n_cells=24)
    assert result['groups']['season/spring|B_q50_mean/[300,inf)'] == dict(n_origins=0, n_cells=0)
    for mode in ('raw', 'guarded'):
        assert result[mode]['metrics']['season/spring']['n_cells'] == 24
    assert result['B']['season/winter']['n_origins'] == 1
