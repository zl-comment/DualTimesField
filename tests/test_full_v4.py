import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from forecasting.full_v4.fields import DualField
from forecasting.full_v4.frontend import Frontend, forecast_loss
from forecasting.full_v4.retrieval import sparsemax
from forecasting.full_v4.data import DeviceBank, prefix_rows, unix
from forecasting.full_v4.schema import tree_features, OUTPUT_KEYS
from forecasting.full_v4.experiment import validate_config


def batch(n=3, k=5):
    torch.manual_seed(42)
    return dict(history=torch.randn(n, 72, 12), future=torch.randn(n, 24, 26),
                candidate_flat=torch.randn(n, k, 1465), candidate_future=torch.randn(n, k, 24, 26),
                candidate_curve=torch.randn(n, k, 24), valid=torch.ones(n, k, dtype=torch.bool),
                ages=torch.ones(n, k))


def test_sparsemax_exact_null_and_gradient():
    x = torch.tensor([[0., -2., 3.], [.1, .2, .3]], requires_grad=True)
    p = sparsemax(x)
    torch.testing.assert_close(p[0], torch.tensor([0., 0., 1.]))
    torch.testing.assert_close(p.sum(-1), torch.ones(2))
    (p * torch.tensor([1., 2., 3.])).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad[1].abs().sum() > 0


def test_empty_library_returns_current_prediction():
    m = Frontend().eval(); b = batch(); b['valid'][:] = False
    out = m(b)
    torch.testing.assert_close(out['P'], out['P0'], rtol=0, atol=0)
    assert out['gC'].eq(0).all() and out['gD'].eq(0).all()


def test_force_null_has_no_history_bypass():
    m = Frontend().eval(); b = batch()
    a = m(b, force_null=True)
    for k in ('candidate_flat', 'candidate_future', 'candidate_curve'):
        b[k][:] = float('nan')
    c = m(b, force_null=True)
    for key in OUTPUT_KEYS:
        torch.testing.assert_close(a[key], c[key], rtol=0, atol=0)
    x = tree_features({k: a[k].detach().numpy() for k in OUTPUT_KEYS}, np.ones(3))
    assert x.shape == (3, 193) and np.isfinite(x).all()


def test_forced_history_and_empty_fallback():
    m = Frontend('forced').eval(); b = batch()
    b['valid'][0] = False
    out = m(b)
    assert out['gC'][0].eq(0).all()
    torch.testing.assert_close(out['gC'][1:], torch.ones(2, 24))
    torch.testing.assert_close(out['gD'][1:], torch.ones(2, 24))


def test_joint_objective_trains_historical_and_current_modules():
    torch.set_num_threads(1)
    m = Frontend('forced', memory_dropout=0.).train()
    out = m(batch()); loss = forecast_loss(out, torch.randn(3, 24))
    loss.backward()
    for module in (m.historical, m.fields, m.evidence, m.retrieval):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_fields_reconstruct_curve_shapes_and_partition_basis():
    field = DualField()
    torch.testing.assert_close(field.basis.sum(-1), torch.ones(24))
    c, d, a = field(torch.randn(7, 32))
    assert c.shape == d.shape == (7, 24) and a.shape == (7, 4)


def test_current_events_are_not_restricted_to_historical_events():
    m = Frontend('current').eval()
    with torch.no_grad():
        m.fields.events.bias[:4] = 1.
    out = m(batch())
    assert out['D0'].abs().sum() > 0
    torch.testing.assert_close(out['P'], out['P0'])


def test_prefix_boundary_checks_full_label_and_release():
    cutoff = unix('2020-01-01')
    mem = SimpleNamespace(origins=np.array([cutoff - 25*3600, cutoff - 24*3600, cutoff - 23*3600]),
        complete=np.ones(3, bool), max_availability=np.array([cutoff - 3600, cutoff, cutoff + 3600]))
    np.testing.assert_array_equal(prefix_rows(mem, '2020-01-01'), [0])


def test_device_bank_masks_unavailable_values_before_encoding():
    mem = SimpleNamespace(curves=np.ones((3,24)), levels=np.zeros(3), origins=np.arange(3)*3600)
    packed = dict(history=np.zeros((3,72,12), np.float32), future=np.zeros((3,24,26), np.float32),
                  flat=np.zeros((3,1465), np.float32))
    packed['flat'][0] = np.nan
    bank = DeviceBank(mem, packed, {'target_std': 1.}, np.full((3,2), -1), 'cpu')
    b = bank.batch(np.array([1,2]))
    assert torch.isfinite(b['candidate_flat']).all()


def test_checkpoint_reload_predictions_match(tmp_path):
    m = Frontend().eval(); b = batch()
    expected = m(b)['P']
    torch.save(m.state_dict(), tmp_path/'model.pt')
    other = Frontend().eval()
    other.load_state_dict(torch.load(tmp_path/'model.pt', weights_only=True))
    torch.testing.assert_close(expected, other(b)['P'], rtol=0, atol=0)


def test_prediction_ignores_unrecognized_query_label():
    m = Frontend().eval(); b = batch()
    expected = m(b)['P']
    b['query_actual'] = torch.full((3,24), float('nan'))
    torch.testing.assert_close(expected, m(b)['P'], rtol=0, atol=0)


def test_formal_config_forbids_truncation():
    import yaml
    cfg = yaml.safe_load(open('configs/full_v4/experiment.yaml'))
    validate_config(cfg)
    cfg['max_train_origins'] = 10
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_changing_query_outcome_cannot_enter_memory_prediction():
    torch.set_num_threads(1)
    rng = np.random.default_rng(2026)
    memory = SimpleNamespace(curves=rng.normal(size=(4,24)), levels=np.zeros(4), origins=np.arange(4)*86400)
    packed = {k:rng.normal(size=shape).astype(np.float32) for k,shape in
              dict(history=(4,72,12), future=(4,24,26), flat=(4,1465)).items()}
    candidates = np.array([[-1,-1], [0,-1], [0,1], [1,2]])
    bank = DeviceBank(memory, packed, {'target_std': 1.}, candidates, 'cpu')
    model = Frontend().eval()
    expected = model(bank.batch(np.array([3])))['P'].detach().clone()
    bank.curves[3] = float('nan')
    torch.testing.assert_close(expected, model(bank.batch(np.array([3])))['P'], rtol=0, atol=0)


def test_field_fusion_matches_explicit_formula():
    model = Frontend('forced').eval(); b = batch()
    with torch.no_grad():
        result = model(b)
        hc, hd, _ = model.reconstruct(b['candidate_flat'].flatten(0,1), b['candidate_curve'].flatten(0,1))
        hc, hd = hc.reshape(3,5,24), hd.reshape(3,5,24)
        real = result['weights'][:,:-1]
        expected_c = (real[...,0] * hc).sum(1)
        expected_d = (real[...,1] * hd).sum(1)
        torch.testing.assert_close(result['P'], expected_c + expected_d, atol=1e-6, rtol=1e-5)
