import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from forecasting.full_v5_anchor_balance import confirmation_experiment as original
from forecasting.full_v5_anchor_balance.confirmation_data import DeviceHistoryStore
from forecasting.full_v5_anchor_balance.selection import select_candidate
from forecasting.full_v5_calibration_robustness import experiment as e
from forecasting.full_v5_calibration_robustness.model import build_model
from forecasting.full_v5_calibration_robustness.selection import quarter_windows, select_windows
from test_full_v5_anchor_architecture_experiment import synthetic_store


def four_windows(n=8):
    return {f'2020Q{i+1}':np.arange(n)//(n//4)==i for i in range(4)}


def test_window_guard_rejects_aggregate_improvement_hiding_one_bad_quarter():
    b=np.broadcast_to(np.array([0.,20.,50.,80.,100.]),(8,24,5)).copy()
    y=np.full((8,24),60.);y[:2]=40.
    a=b+5
    assert not select_candidate(y,b,a,a,alphas=[0,1])['fallback']
    robust=select_windows(y,b,a,a,four_windows(),alphas=[0,1])
    assert robust['fallback'] and robust['alpha']==0
    y[:2]=60
    good=select_windows(y,b,a,a,four_windows(),alphas=[0,1])
    assert good['alpha']==1 and not good['fallback']
    assert good['worst_quarter_relative_qs']<0
    bad=four_windows();bad['2020Q4'][:]=True
    with pytest.raises(ValueError,match='partition'):
        select_windows(y,b,a,a,bad)


def test_quarter_windows_partition_without_adding_data():
    times=np.array([e.base.timestamp(f'2020-{m:02d}-01') for m in range(1,13)])
    masks=quarter_windows(times)
    assert all(type(name) is str for name in masks)
    assert list(masks)==['2020Q1','2020Q2','2020Q3','2020Q4']
    np.testing.assert_array_equal(np.sum(list(masks.values()),axis=0),np.ones(12))
    with pytest.raises(ValueError,match='Four quarters'):
        quarter_windows(times[:6])


def test_bounds_identity_initialization_ordering_and_separate_tails():
    torch.set_num_threads(2)
    reference,bounded=(build_model(v,2026) for v in ('asymmetric_scale','bounded_calibration'))
    for name,value in reference.state_dict().items():
        torch.testing.assert_close(value,bounded.state_dict()[name],rtol=0,atol=0)
    batch=synthetic_store().batch([4,7])
    torch.testing.assert_close(reference(batch)['prediction'],bounded(batch)['prediction'],rtol=0,atol=0)
    b=batch['baseline'];features=torch.randn(2,24,32)
    for sign in (-1,1):
        with torch.no_grad():bounded.calibration[-1].bias.fill_(sign*100)
        q,delta=bounded.calibrate_anchor(b,features)
        assert torch.all(delta.abs()<=30)
        scale=(q-b[...,2:3]-delta[...,None])/(b-b[...,2:3])
        assert torch.all(scale[...,[0,1,3,4]]>=.9-1e-6)
        assert torch.all(scale[...,[0,1,3,4]]<=1.1+1e-6)
        assert torch.all(q.diff(dim=-1)>=0)
        torch.testing.assert_close(q[...,2],b[...,2]+delta)
    with torch.no_grad():bounded.calibration[-1].bias.zero_()
    q,_=bounded.calibrate_anchor(b,features)
    q[...,0].sum().backward()
    assert bounded.calibration[-1].bias.grad[1].abs()>0
    assert bounded.calibration[-1].bias.grad[2]==0


def small_config():
    cfg=e.configuration(['cpu']*3)
    cfg.update(epochs=2,batch_size=2,prediction_batch_size=2)
    return cfg


def small_rows():
    return dict(train=np.array([4,5]),selection=np.array([6,7,8,9]),evaluation=np.array([10,11]))


def test_reference_replay_exact_and_eval_targets_not_used(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(e,'quarter_windows',lambda origins:four_windows(4))
    store=DeviceHistoryStore(synthetic_store(),np.arange(12))
    cfg,rows=small_config(),small_rows()
    original.fit(store,rows,'asymmetric_scale',2026,cfg,tmp_path/'old','old-run')
    a=e.fit(store,rows,'reference',2026,cfg,tmp_path/'new','run')
    assert e.verify_reference(tmp_path/'new',tmp_path/'old')['exact']
    # Future labels cannot affect training or either checkpoint-selection policy.
    other=DeviceHistoryStore(synthetic_store(),np.arange(12))
    other.a['actual'][[10,11]]+=99999
    other.data['actual'][[10,11]]+=99999
    b=e.fit(other,rows,'reference',2026,cfg,tmp_path/'changed','run')
    assert a['selection']==b['selection']
    for mode in ('raw','guarded','multiwindow'):
        am,_=e.base.restore_checkpoint(tmp_path/'new'/f'{mode}.pt',model_builder=build_model)
        bm,_=e.base.restore_checkpoint(tmp_path/'changed'/f'{mode}.pt',model_builder=build_model)
        for name,value in am.state_dict().items():
            torch.testing.assert_close(value,bm.state_dict()[name],rtol=0,atol=0)


def test_atomic_resume_preserves_both_selection_rules(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    calendar=synthetic_store()
    calendar.origins[6:10]=[e.base.timestamp(f'2020-{m:02d}-01') for m in (1,4,7,10)]
    calendar.origins[10:]=[e.base.timestamp('2021-01-01'),e.base.timestamp('2021-01-03')]
    store=DeviceHistoryStore(calendar,np.arange(12))
    cfg,rows=small_config(),small_rows()
    expected=e.fit(store,rows,'reference',2027,cfg,tmp_path/'expected','run')
    writer=e.base.write_json
    def interrupt(path,value):
        if path==tmp_path/'resumed'/'training.json' and len(value)==1:
            raise RuntimeError('simulated interruption')
        return writer(path,value)
    monkeypatch.setattr(e.base,'write_json',interrupt)
    with pytest.raises(RuntimeError,match='simulated'):
        e.fit(store,rows,'reference',2027,cfg,tmp_path/'resumed','run')
    monkeypatch.setattr(e.base,'write_json',writer)
    actual=e.fit(store,rows,'reference',2027,cfg,tmp_path/'resumed','run')
    assert expected['selection']==actual['selection']
    for mode in ('raw','guarded','multiwindow'):
        a,b=(torch.load(tmp_path/p/f'{mode}.pt',weights_only=True)['state_dict'] for p in ('expected','resumed'))
        for name in a:torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
    with pytest.raises(ValueError,match='identity changed'):
        e.fit(store,rows,'reference',2027,cfg,tmp_path/'resumed','different')


def test_barrier_complete_shared_trajectories_and_checkpoint_tampering(tmp_path):
    cfg=e.configuration(['cpu']*3)
    assert cfg['epochs']==20 and cfg['seeds']==[2026,2027,2028]
    assert len(e.expected_keys(cfg))==54
    path=tmp_path/'weights.pt';path.write_text('checkpoint')
    rows={}
    for key in e.expected_keys(cfg):
        r,f,s,arm=key.split('/')
        trajectory='reference' if arm in ('reference','multiwindow') else arm
        rows[key]=dict(fit_id=f'{r}/{f}/{s}/{trajectory}',epochs_completed=20,
            checkpoints={str(path):e.base.digest(path)},checkpoint_paths=dict(raw=str(path),guarded=str(path)))
    e.verify_barrier(rows,cfg)
    incomplete=dict(rows);incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError,match='All robustness'):e.verify_barrier(incomplete,cfg)
    changed=copy.deepcopy(rows);changed[next(iter(changed))]['epochs_completed']=19
    with pytest.raises(ValueError,match='epoch budget'):e.verify_barrier(changed,cfg)
    path.write_text('changed')
    with pytest.raises(ValueError,match='checkpoint changed'):e.verify_barrier(rows,cfg)


def test_diagnostic_width_intervention_keeps_mae_and_groups_reconcile():
    from scripts.diagnose_anchor_robustness import decompose
    b=np.broadcast_to(np.array([0.,20.,50.,80.,100.]),(8,24,5)).copy()
    y=np.full((8,24),60.)
    a=50+5+.8*(b-50);f=a+2
    origins=np.array([e.base.timestamp(f'2023-{m:02d}-01') for m in range(1,9)])
    rows=decompose(origins,y,b,a,f)
    all_rows={r['component']:r for r in rows if r['group']=='all'}
    assert all_rows['Anchor']['mae']==all_rows['Anchor_location_only']['mae']
    assert all_rows['Full']['mae']==all_rows['Full_tree_width']['mae']
    assert all_rows['B']['mae']==all_rows['Anchor_scale_only']['mae']
    buckets=[r for r in rows if r['component']=='Full' and r['group'].startswith('actual_')]
    assert sum(r['n_cells'] for r in buckets)==y.size
    assert sum(r['n_cells']*r['mae_minus_B'] for r in buckets)/y.size==pytest.approx(all_rows['Full']['mae_minus_B'])


def test_full_runner_freezes_three_logical_arms_before_evaluation(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    cfg=small_config()
    cfg.update(regions=['TAS1'],folds=['2021'],seeds=[2026],devices=['cpu'],old_reference=str(tmp_path/'old'))
    store=synthetic_store();store.origins[:]+=280*e.base.DAY
    rows=small_rows()
    adapted=DeviceHistoryStore(store,np.arange(12))
    oldpath=e.arm_path(cfg['old_reference'],'TAS1','2021',2026,'asymmetric_scale')
    record=original.fit(adapted,rows,'asymmetric_scale',2026,cfg,oldpath,'old-run')
    metrics=original.evaluate_fit(adapted,rows,'asymmetric_scale',cfg,oldpath,record)
    balanced=e.arm_path(cfg['old_reference'],'TAS1','2021',2026,'balanced')
    balanced.mkdir();e.base.write_json(balanced/'metrics.json',metrics)
    monkeypatch.setattr(e,'quarter_windows',lambda origins:four_windows(4))
    monkeypatch.setattr(e.base,'load_store',lambda *args:(store,[]))
    monkeypatch.setattr(e.base,'split_rows',lambda *args:rows)
    output=tmp_path/'run'
    def prepare(*args,**kwargs):
        output.mkdir()
        manifest=dict(run_id='run',config=cfg)
        e.base.write_json(output/'manifest.json',manifest)
        return manifest
    monkeypatch.setattr(e,'prepare',prepare)
    monkeypatch.setattr(e,'verify_manifest',lambda *args:None)
    calls=[]
    def pool(worker,specs):
        if worker is e.evaluate_region:
            assert calls==['training']
            barrier=json.loads((output/'selection_complete.json').read_text())
            assert len(barrier)==3 and len({r['fit_id'] for r in barrier.values()})==2
            calls.append('evaluation')
        else:
            assert not (output/'selection_complete.json').exists();calls.append('training')
        return {k:v for spec in specs for k,v in worker(spec).items()}
    monkeypatch.setattr(e.c,'run_pool',pool)
    e.run(cfg,Path('fixture'),output)
    assert calls==['training','evaluation']
    assert json.loads((output/'progress.json').read_text())['stage']=='complete'
    assert json.loads((output/'training_order_audit.json').read_text())['reference_replays_exact']
    assert len(list(output.rglob('*.npz')))==6
    assert (output/'acceptance.json').exists()
    from scripts.verify_calibration_robustness import score_audit
    audit=score_audit(output)
    assert audit['prediction_files']==6 and audit['historical_predictions_bitwise_exact']==2
    assert audit['multiwindow_selections_reconstructed']==1


def test_preparation_preserves_all_input_bytes(tmp_path,monkeypatch):
    cfg=small_config()
    cfg.update(regions=['TAS1'],folds=['2021'],seeds=[2026],devices=['cpu'],old_reference=str(tmp_path/'old'))
    bundle=tmp_path/'bundles'/'bundle.npz';bundle.parent.mkdir()
    bundle.write_bytes(b'original immutable NPZ input')
    bundle.with_suffix('.json').write_text(json.dumps(dict(model_files=[])))
    (bundle.parent/'config.json').write_text('{}')
    for variant in ('asymmetric_scale','balanced'):
        directory=e.arm_path(cfg['old_reference'],'TAS1','2021',2026,variant);directory.mkdir(parents=True)
        for name in ('training.json','selection.json','raw.pt','guarded.pt','last.pt','metrics.json'):
            (directory/name).write_bytes(b'original historical fixture')
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    store=synthetic_store()
    monkeypatch.setattr(e.base,'load_store',lambda *args:(store,[bundle]))
    monkeypatch.setattr(e.base,'split_rows',lambda *args:small_rows())
    monkeypatch.setattr(e,'quarter_windows',lambda origins:four_windows(4))
    output=tmp_path/'run'
    manifest=e.prepare(output,cfg,bundle.parent,False)
    e.verify_manifest(manifest)
    assert all(p.read_bytes()==value for p,value in before.items())
    assert json.loads((output/'manifest.json').read_text())==manifest
    assert e.prepare(output,cfg,bundle.parent,True)==manifest
    with pytest.raises(FileExistsError):e.prepare(output,cfg,bundle.parent,False)


def test_cli_refuses_duplicate_runner_and_requires_development(tmp_path):
    import fcntl
    output=tmp_path/'run';output.mkdir()
    with pytest.raises(ValueError,match='development required'):
        e.main(['--output',str(output)])
    with (tmp_path/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='active runner'):
            e.main(['--development','--output',str(output),'--resume'])
    with pytest.raises(FileExistsError):
        e.main(['--development','--output',str(output)])
    assert not list(output.iterdir())


def test_new_acceptance_keeps_single_seed_and_spring_failures_visible():
    from forecasting.full_v5_calibration_robustness.report import acceptance
    cfg=e.configuration(['cpu']*3);cfg.update(regions=['QLD1'],folds=['2021'])
    results={}
    for seed in cfg['seeds']:
        for arm in (*e.ARMS,'balanced'):
            value=9.5 if arm=='reference' else 9.6 if arm=='balanced' else (11. if seed==2026 else 8.)
            metrics={g:dict(full=dict(mae=value,qs=value),baseline=dict(mae=10.,qs=10.))
                     for g in ('all','season/spring')}
            results[f'QLD1/2021/{seed}/{arm}']=dict(B={g:dict(mae=10.,qs=10.) for g in metrics},
                                                 guarded=dict(metrics=metrics))
    assert acceptance(results,cfg,'multiwindow','guarded',cfg['seeds'])['passed']
    individual=acceptance(results,cfg,'multiwindow','guarded',[2026])
    assert not individual['passed']
    assert not individual['folds']['2021']['checks']['each_spring_noninferior_to_B']
    assert not individual['folds']['2021']['checks']['macro_noninferior_to_current_reference']


def collision_records(root,cfg):
    records={}
    for region in cfg['regions']:
        for fold in cfg['folds']:
            for seed in cfg['seeds']:
                for trajectory in e.TRAJECTORIES:
                    path=e.arm_path(root,region,fold,seed,trajectory);path.mkdir(parents=True)
                    modes=('raw','guarded','multiwindow') if trajectory=='reference' else ('raw','guarded')
                    for mode in modes:(path/f'{mode}.pt').write_text('frozen test checkpoint')
                    record=dict(fit_id=f'{fold}/{seed}/{trajectory}',trajectory=trajectory,
                        epochs_completed=cfg['epochs'],selection={mode:dict(epoch=1,alpha=0.) for mode in modes},
                        checkpoints={str(path/f'{mode}.pt'):e.base.digest(path/f'{mode}.pt') for mode in modes})
                    e.base.write_json(path/'selection.json',record)
                    for arm in (('reference','multiwindow') if trajectory=='reference' else (trajectory,)):
                        mode='multiwindow' if arm=='multiwindow' else 'guarded'
                        records[f'{region}/{fold}/{seed}/{arm}']=dict(
                            fit_id=record['fit_id'],trajectory=trajectory,epochs_completed=cfg['epochs'],
                            selection=dict(raw=record['selection']['raw'],guarded=record['selection'][mode]),
                            checkpoints=record['checkpoints'],
                            checkpoint_paths=dict(raw=str(path/'raw.pt'),guarded=str(path/f'{mode}.pt')))
    return records


def test_region_local_fit_id_collisions_are_not_duplicate_physical_fits(tmp_path):
    from forecasting.full_v5_calibration_robustness import runner
    cfg=e.configuration(['cpu']*3)
    records=collision_records(tmp_path,cfg)
    assert len({r['fit_id'] for r in records.values()})==12
    with pytest.raises(ValueError,match='physical trajectories'):e.verify_barrier(records,cfg)
    runner.verify_barrier(records,cfg)
    changed=copy.deepcopy(records)
    changed['NSW1/2021/2026/reference']['checkpoint_paths']=records['QLD1/2021/2026/reference']['checkpoint_paths']
    with pytest.raises(ValueError):runner.verify_barrier(changed,cfg)


def test_completed_record_collection_preserves_every_selection_and_file(tmp_path,monkeypatch):
    from forecasting.full_v5_calibration_robustness import runner
    cfg=e.configuration(['cpu']*3);cfg.update(folds=['2021'],seeds=[2026])
    records=collision_records(tmp_path,cfg)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.setattr(e,'verify_reference',lambda *args:dict(exact=True))
    assert runner.completed_records(tmp_path,cfg)==records
    assert all(Path(p).read_bytes()==data for p,data in before.items())
    next(tmp_path.glob('*/*/seed_*/*/selection.json')).unlink()
    assert runner.completed_records(tmp_path,cfg) is None


def test_completed_training_continuation_never_calls_training(tmp_path,monkeypatch):
    from forecasting.full_v5_calibration_robustness import runner,report
    cfg=e.configuration(['cpu']*3);cfg.update(folds=['2021'],seeds=[2026])
    records=collision_records(tmp_path,cfg)
    monkeypatch.setattr(e,'prepare',lambda *a,**k:dict(run_id='frozen-training'))
    monkeypatch.setattr(e,'verify_manifest',lambda *a:None)
    monkeypatch.setattr(runner,'freeze_orchestration',lambda *a:None)
    monkeypatch.setattr(runner,'verify_orchestration',lambda *a:None)
    monkeypatch.setattr(e,'verify_reference',lambda *a:dict(exact=True))
    (tmp_path/'orchestration.json').write_text('{}')
    called=[]
    def pool(worker,specs):
        assert worker is runner.evaluate_region
        barrier=json.loads((tmp_path/'selection_complete.json').read_text())
        assert barrier==records
        called.append('evaluation')
        return {key:{} for key in records}
    monkeypatch.setattr(e.c,'run_pool',pool)
    monkeypatch.setattr(report,'write_report',lambda *a:None)
    runner.run(cfg,Path('fixture'),tmp_path,resume=True)
    assert called==['evaluation']
    progress=json.loads((tmp_path/'progress.json').read_text())
    assert progress['physical_fits']==6 and progress['logical_fits']==9
    assert progress['reused_completed_training'] and progress['stage']=='complete'


def test_orchestration_freeze_keeps_original_manifest_and_weights(tmp_path):
    from forecasting.full_v5_calibration_robustness import runner
    cfg=e.configuration(['cpu']*3);cfg.update(folds=['2021'],seeds=[2026])
    collision_records(tmp_path,cfg)
    manifest=dict(run_id='frozen-training');e.base.write_json(tmp_path/'manifest.json',manifest)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    runner.freeze_orchestration(tmp_path,manifest)
    runner.verify_orchestration(tmp_path)
    assert all(Path(p).read_bytes()==data for p,data in before.items())
    next(tmp_path.glob('*/*/seed_*/*/*.pt')).write_text('tampered checkpoint')
    with pytest.raises(ValueError,match='evidence changed'):runner.verify_orchestration(tmp_path)
