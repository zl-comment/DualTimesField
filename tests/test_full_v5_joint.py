import copy

import numpy as np
import pytest
import torch

from forecasting.full_v5_joint.data import HistoryStore
from forecasting.full_v5_joint.model import JointHistoryModel, pinball
from forecasting.full_v5_joint.protocol import DAY, Fold, assert_mature, timestamp, segment_rows


def synthetic(n=100):
    rng = np.random.default_rng(5)
    origin = timestamp('2020-01-01') + np.arange(n) * DAY
    baseline = np.broadcast_to(np.array([0, 20, 50, 80, 100], np.float32), (n,24,5)).copy()
    actual = np.full((n,24), 80., np.float32)
    actual[:,12] = 900
    return dict(origin=origin, baseline=baseline, actual=actual,
                z=rng.normal(size=(n,24,16)).astype('f'), history=rng.normal(size=(n,72,12)).astype('f'),
                future=rng.normal(size=(n,24,31)).astype('f'), lags=rng.normal(size=(n,24,3)).astype('f')*10+70,
                trained_before=np.full(n,origin[0]-DAY*20), selected_before=np.full(n,origin[0]-DAY))


def test_future_label_invariance_and_maturity():
    a = synthetic()
    b = copy.deepcopy(a)
    # Query and every unmatured/future label must not change any prediction input.
    b['actual'][60:] += 90000
    s, t = HistoryStore(a), HistoryStore(b)
    left, right = s.batch([60]), t.batch([60])
    assert 'actual' not in left
    for key in left:
        torch.testing.assert_close(left[key], right[key])
    assert_mature([100*DAY], np.array([[99*DAY]]), np.array([[True]]))
    with pytest.raises(ValueError, match='not mature'):
        assert_mature([100*DAY], np.array([[99*DAY+1]]), np.array([[True]]))
    a['selected_before'][50] = a['origin'][50]
    with pytest.raises(ValueError, match='must precede'):
        HistoryStore(a)


def test_final_prediction_loss_reaches_all_modules():
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    store = HistoryStore(synthetic(), candidates=4)
    model = JointHistoryModel()
    output = model(store.batch([50,60]))
    # Deliberately exclude the reconstruction loss and gate regularizer.
    pinball(output['prediction'], store.targets([50,60])).backward()
    for name in ('state', 'calibration', 'dual', 'match', 'alignment_query', 'alignment_key', 'gates'):
        gradients = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
        assert gradients, name
        assert all(torch.isfinite(g).all() for g in gradients), name
        assert sum(float(g.abs().sum()) for g in gradients) > 0, name
    assert torch.isfinite(output['prediction']).all()
    assert (output['prediction'].diff(dim=-1) >= 0).all()
    assert output['correction'].abs().max() <= 150


def test_empty_memory_and_history_off():
    torch.set_num_threads(2)
    model = JointHistoryModel()
    store = HistoryStore(synthetic(), candidates=4)
    output = model(store.batch([0]))
    torch.testing.assert_close(output['prediction'], output['anchor'])
    output = model(store.batch([60]), history_off=True)
    torch.testing.assert_close(output['prediction'], output['anchor'])
    assert float(model(store.batch([60]))['correction'].abs().sum()) > 0


def test_time_contract_and_no_anniversary_filter():
    assert Fold(2022).prediction_end == '2022-07-01'
    assert Fold(2022, True).train_end == '2022-01-01'
    assert Fold(2022, True).prediction_start == '2022-07-01'
    rows = segment_rows(np.array([timestamp('2022-06-29'), timestamp('2022-06-30')]), '2022-01-01','2022-07-01')
    assert rows.tolist() == [0]
    store = HistoryStore(synthetic(), candidates=8)
    ids, _ = store.record(60)
    assert (ids >= 0).all()  # No365day memory is available, but causal memories are usable.
    assert (store.origins[ids] + DAY <= store.origins[60]).all()


def test_formal_loader_rejects_smoke_and_tampered_artifacts(tmp_path):
    import json
    from forecasting.full_v5_joint.data import digest
    path = tmp_path/'bundle.npz'
    np.savez_compressed(path, **synthetic())
    meta = dict(scope='smoke_archived2022', array_sha256=digest(path))
    path.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='only accepts chronological'):
        HistoryStore.from_paths([path])
    meta.update(scope='chronological_oof', array_sha256='incorrect')
    path.with_suffix('.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='hash mismatch'):
        HistoryStore.from_paths([path])


def test_targets_change_only_supervision_not_future_prediction():
    torch.set_num_threads(2)
    torch.manual_seed(4)
    a = synthetic()
    b = copy.deepcopy(a)
    b['actual'][60:] *= 100
    left, right = HistoryStore(a, candidates=4), HistoryStore(b, candidates=4)
    model = JointHistoryModel().eval()
    with torch.no_grad():
        torch.testing.assert_close(model(left.batch([60]))['prediction'], model(right.batch([60]))['prediction'])
    assert not torch.equal(left.targets([60]), right.targets([60]))


def test_purge_keeps_hashes_and_split_metadata_consistent():
    from forecasting.aligned_inputs import AlignedSplit, _array_hash
    from forecasting.full_v5_joint.prepare import purge
    origins = np.array([timestamp('2021-12-29'), timestamp('2021-12-30'), timestamp('2021-12-31')])
    actual = np.zeros((3,24))
    split = AlignedSplit({'example':np.zeros((3,2))}, actual, origins,
                         origins[:,None]+np.arange(24)[None]*3600, {}, actual.copy())
    result = purge(split, '2022-01-01')
    assert len(result) == 2
    assert result.origin_unix.max()+DAY < timestamp('2022-01-01')
    assert result.metadata['input_hash'] == _array_hash(result.inputs)
    assert result.metadata['sample_hash'] == _array_hash(dict(origin=result.origin_unix,target_time=result.target_time_unix,actual=result.actual))
