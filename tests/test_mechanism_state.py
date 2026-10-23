import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from forecasting.full_v5_mechanism_state.policy import classify, fit_policy
from forecasting.full_v5_mechanism_state.model import build_model, VARIANTS
from forecasting.full_v5_mechanism_state import runner as r
from forecasting.full_v5_mechanism_state.data import MechanismStore
from forecasting.full_v5_mechanism_state.diagnostics import diagnose
from forecasting.full_v5_anchor_balance.confirmation_data import DeviceHistoryStore
from test_full_v5_anchor_architecture_experiment import synthetic_store


def observations():
    rng = np.random.default_rng(73)
    x = np.concatenate((rng.uniform(0, 100, (160, 24, 4)), np.ones((160, 24, 2))), axis=-1)
    b = np.broadcast_to(np.array([-30., -20., 0., 20., 30.]), (160, 24, 5)).copy()
    y = rng.normal(5, 12, (160, 24))
    t = r.base.timestamp('2018-01-01') + np.arange(160)*86400
    return x, y, b, t


def test_policy_training_only_separate_evidence_and_matched_constant():
    x, y, b, t = observations()
    policy = fit_policy(x, y, b, t)
    states = classify(x, policy['thresholds'])
    gates = np.asarray(policy['gates'])
    assert gates.shape == (6, 3) and np.all((gates>=0)&(gates<=1))
    np.testing.assert_allclose(gates[states].mean(axis=(0, 1)), policy['constant_gates'])
    assert np.all(gates[5] == 0)
    # The same physical state reverses the tree median residual in alternating blocks.
    reversed_y = y.copy()
    reversed_y[40:80] -= 10; reversed_y[120:] -= 10
    reversed_policy = fit_policy(x, reversed_y, b, t)
    assert reversed_policy['gates'][0][0] < policy['gates'][0][0]
    assert policy['thresholds'] == reversed_policy['thresholds']
    changed = y.copy(); changed[80:] = 1e8
    assert fit_policy(x[:80], y[:80], b[:80], t[:80]) == fit_policy(x[:80], changed[:80], b[:80], t[:80])
    policy = fit_policy(x[:4], y[:4], b[:4], t[:4])
    assert max(np.array(policy['gates']).ravel()) <= 1/31


def test_classifier_numpy_torch_boundary_and_uncovered_agree():
    x, y, b, t = observations()
    policy = fit_policy(x, y, b, t)
    model = build_model('state_evidence'); model.set_policy(policy)
    x[0, 0, 4] = 0
    x[1, :, :4] = [policy['thresholds'][0], policy['thresholds'][3], policy['thresholds'][4], policy['thresholds'][5]]
    state, gate = model.constraint(torch.tensor(x))
    np.testing.assert_array_equal(state.numpy(), classify(x, policy['thresholds']))
    assert not gate[0, 0].any()
    constant = build_model('constant_evidence'); constant.set_policy(policy)
    _, cg = constant.constraint(torch.tensor(x))
    assert not cg[0, 0].any()
    with pytest.raises(ValueError, match='binary'):
        x[0, 0, 4] = .2; model.constraint(torch.tensor(x))


def test_initialization_and_unit_gates_reproduce_entire_archived_forward():
    torch.set_num_threads(2)
    x, y, b, t = observations(); policy = fit_policy(x, y, b, t)
    policy['gates'] = [[1., 1., 1.]]*5 + [[0., 0., 0.]]
    reference, state = build_model('bounded_reference'), build_model('state_evidence')
    state.set_policy(policy)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, state.state_dict()[name], rtol=0, atol=0)
    # Nonzero location is essential to test the original history coupling.
    with torch.no_grad():
        reference.calibration[-1].bias.copy_(torch.tensor([.12, -.15, .2]))
        state.calibration[-1].bias.copy_(reference.calibration[-1].bias)
    batch = synthetic_store().batch([4, 7])
    batch['mechanism_inputs'] = torch.tensor(x[:2])
    for flags in ({}, {'event_off':True}, {'history_off':True}, {'time_shift_off':True}):
        a, c = reference(batch, **flags), state(batch, **flags)
        for key in a:
            torch.testing.assert_close(a[key], c[key], rtol=0, atol=0)


def test_bounds_gradient_zero_gate_identity_and_checkpoint(tmp_path):
    torch.set_num_threads(2)
    x,y,b,t = observations(); policy=fit_policy(x,y,b,t)
    model=build_model('state_evidence');model.set_policy(policy)
    batch=synthetic_store().batch([4,7]);batch['mechanism_inputs']=torch.tensor(x[:2])
    with torch.no_grad():model.calibration[-1].bias.copy_(torch.tensor([5.,-5.,5.]))
    output=model(batch)
    delta=output['anchor'][...,2]-batch['baseline'][...,2]
    assert torch.all(delta.abs() <= 30*output['constraint_gates'][...,0]+1e-5)
    assert torch.all(torch.diff(output['anchor'],dim=-1)>=0)
    output['prediction'].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    path=tmp_path/'saved.pt'
    torch.save(dict(variant='state_evidence',seed=2026,state_dict=model.state_dict(),selection={}),path)
    restored,_=r.base.restore_checkpoint(path,model_builder=build_model)
    torch.testing.assert_close(restored(batch)['prediction'],model(batch)['prediction'],rtol=0,atol=0)
    batch['mechanism_inputs'][...,4]=0
    output=model(batch)
    torch.testing.assert_close(output['anchor'],batch['baseline'],rtol=0,atol=0)
    assert not output['constraint_gates'].any()


def test_current_targets_cannot_enter_prediction():
    x,y,b,t=observations();policy=fit_policy(x,y,b,t)
    original=synthetic_store(); adapted=DeviceHistoryStore(original,np.arange(12))
    store=MechanismStore(adapted,x[:12])
    model=build_model('state_evidence');model.set_policy(policy)
    before=model(store.batch([7]))['prediction'].detach().clone()
    adapted.data['actual'][7:]=1e7
    torch.testing.assert_close(before,model(store.batch([7]))['prediction'],rtol=0,atol=0)


def test_complete_training_resume_and_region_barrier(tmp_path):
    torch.set_num_threads(2)
    x,y,b,t=observations();policy=fit_policy(x,y,b,t)
    source=synthetic_store();store=MechanismStore(DeviceHistoryStore(source,np.arange(12)),x[:12])
    rows=dict(train=np.array([4,5]),selection=np.array([7,8]),evaluation=np.array([10,11]))
    cfg=r.configuration(['cpu']*3,True);cfg.update(seeds=[2026],batch_size=2,prediction_batch_size=2)
    records={}
    for variant in VARIANTS:
        path=r.arm_path(tmp_path,'NSW1','2021',2026,variant)
        record=r.fit(store,rows,variant,2026,cfg,path,'test',policy)
        records[f'NSW1/2021/2026/{variant}']=record
        assert r.fit(store,rows,variant,2026,cfg,path,'test',policy)==record
        # Resume the committed final epoch without selection.json; no retraining.
        (path/'selection.json').unlink()
        assert r.fit(store,rows,variant,2026,cfg,path,'test',policy)==record
        result=r.evaluate_fit(store,rows,cfg,path,record,policy)
        assert result['guarded']['mechanism']['all']['cells']==48
        bad=copy.deepcopy(policy);bad['gates'][0][0]*=.5
        with pytest.raises(ValueError,match='identity changed'):
            r.fit(store,rows,variant,2026,cfg,path,'test',bad)
    r.verify_barrier(records,cfg,tmp_path)
    bad=copy.deepcopy(records);bad.pop(next(iter(bad)))
    with pytest.raises(ValueError,match='All selections'):r.verify_barrier(bad,cfg,tmp_path)
    bad=copy.deepcopy(records)
    one=next(iter(bad.values()));one['checkpoints']={k.replace('NSW1','QLD1'):v for k,v in one['checkpoints'].items()}
    with pytest.raises(ValueError,match='region/fold'):r.verify_barrier(bad,cfg,tmp_path)


def test_diagnostic_harm_benefit_decomposition():
    x,y,b,t=observations();policy=fit_policy(x,y,b,t)
    a=b+4
    d=diagnose(y,b,a,a,x,policy)
    assert d['all']['anchor_harm']-d['all']['anchor_benefit']==pytest.approx(d['all']['anchor_mae_delta'])
    assert sum(v['cells'] for k,v in d.items() if k.startswith('state/'))==y.size
