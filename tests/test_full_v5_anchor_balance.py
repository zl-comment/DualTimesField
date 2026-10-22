import copy
import io

import numpy as np
import pytest
import torch

from forecasting.full_v5_anchor_balance import (
    BalanceWeights, balance_loss, blend, build_model, score, select_candidate,
    selection_key,
)
from forecasting.full_v5_joint.data import HistoryStore
from forecasting.full_v5_joint.model import training_loss
from test_full_v5_joint import synthetic


@pytest.fixture
def store():
    torch.set_num_threads(2)
    return HistoryStore(synthetic(), candidates=4)


def test_zero_weights_preserve_legacy_loss_and_gradients(store):
    model = build_model('balanced')
    batch = store.batch([50, 60])
    output = model(batch)
    targets = store.targets([50, 60])
    reference = training_loss(output, targets, reconstruction_weight=.07, gate_penalty=.003)
    loss, _ = balance_loss(output, targets, batch['baseline'], BalanceWeights(0, 0, 0),
                           reconstruction_weight=.07, gate_penalty=.003)
    assert torch.equal(loss, reference)
    params = list(model.parameters())
    left = torch.autograd.grad(reference, params, retain_graph=True)
    right = torch.autograd.grad(loss, params)
    for a, b in zip(left, right):
        assert torch.equal(a, b)


@pytest.mark.parametrize('term', ['anchor_pinball', 'median_pinball', 'identity'])
def test_each_auxiliary_supervises_calibration_without_full_loss(store, term):
    model = build_model('balanced')
    with torch.no_grad():
        model.calibration[-1].bias[0] = .1
    batch = store.batch([50, 60])
    _, terms = balance_loss(model(batch), store.targets([50, 60]), batch['baseline'])
    terms[term].backward()
    gradient = model.calibration[-1].bias.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_identity_regularizer_units_and_median_pinball_factor():
    baseline = torch.tensor([[[-4., -2., 0., 2., 4.]]])
    output = dict(anchor=baseline + 30, prediction=baseline + 60,
                  gates=torch.zeros(1, 1, 2), reconstruction_loss=torch.tensor(0.))
    _, terms = balance_loss(output, torch.zeros(1, 1), baseline)
    assert terms['identity'].item() == pytest.approx(.01 * .1 ** 2)
    assert terms['median_pinball'].item() == pytest.approx(.5 * .5 * (30 + 60) / 300)


def test_identity_survives_training_and_checkpoint_reload(store):
    model = build_model('identity_anchor')
    batch, target = store.batch([50, 60]), store.targets([50, 60])
    # B50 + (Bq - B50) is not bit-exact identity for arbitrary float32 values.
    batch['baseline'] = torch.tensor([-.1, .2, 100., 200., 300.]).expand_as(batch['baseline']).clone()
    before = model.gates[-1].weight.detach().clone()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001)
    assert all(not p.requires_grad for p in model.calibration.parameters())
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        assert torch.equal(output['anchor'], batch['baseline'])
        training_loss(output, target).backward()
        optimizer.step()
    assert not torch.equal(model.gates[-1].weight, before)
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = build_model('identity_anchor')
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert all(not p.requires_grad for p in restored.calibration.parameters())
    assert torch.equal(restored(batch)['anchor'], batch['baseline'])


def test_future_labels_do_not_change_balanced_predictions():
    a = synthetic()
    changed = copy.deepcopy(a)
    changed['actual'][60:] += 100000
    model = build_model('balanced').eval()
    left, right = HistoryStore(a, candidates=4), HistoryStore(changed, candidates=4)
    with torch.no_grad():
        original = model(left.batch([60]))
        other = model(right.batch([60]))
    for key in ('anchor', 'prediction'):
        torch.testing.assert_close(original[key], other[key], rtol=0, atol=0)


def test_anchor_bounds_and_convex_quantile_order(store):
    model = build_model('balanced')
    batch = store.batch([50, 60])
    with torch.no_grad():
        model.calibration[-1].weight.zero_()
        model.calibration[-1].bias.copy_(torch.tensor([2., -2.]))
        output = model(batch)
    anchor, baseline = output['anchor'], batch['baseline']
    assert (anchor[..., 2] - baseline[..., 2]).abs().max() <= 150
    scale = (anchor[..., -1] - anchor[..., 0]) / (baseline[..., -1] - baseline[..., 0])
    assert ((scale >= .5) & (scale <= 1.5)).all()
    for alpha in (0, .25, .5, 1):
        result = blend(baseline.numpy(), output['prediction'].numpy(), alpha)
        assert (np.diff(result, axis=-1) >= 0).all()
        assert np.all(result >= np.minimum(baseline.numpy(), output['prediction'].numpy()) - 1e-6)
        assert np.all(result <= np.maximum(baseline.numpy(), output['prediction'].numpy()) + 1e-6)


def example():
    baseline = np.array([[-4., -2., 2., 6., 8.]], dtype=np.float32)
    return np.zeros(1), baseline, baseline - 2


def test_guard_accepts_improvement_and_uses_shared_alpha():
    actual, baseline, improved = example()
    choice = select_candidate(actual, baseline, improved, improved, epoch=3)
    assert not choice['fallback'] and choice['alpha'] == 1
    assert choice['epoch'] == 3
    assert choice['anchor'] == choice['full'] == score(actual, improved)
    assert choice['full']['mae'] == 0


def test_guard_rejects_bad_anchor_even_if_full_improves():
    actual, baseline, improved = example()
    choice = select_candidate(actual, baseline, baseline + 2, improved)
    assert choice['fallback'] and choice['alpha'] == 0
    assert choice['full'] == choice['anchor'] == score(actual, baseline)
    assert 'no_positive_alpha' in choice['reason']


def test_guard_requires_real_improvement_and_fallback_is_exact():
    actual, baseline, _ = example()
    choice = select_candidate(actual, baseline, baseline, baseline)
    assert choice['fallback']
    result = blend(baseline, baseline + 20, 0)
    np.testing.assert_array_equal(result, baseline)
    assert result.dtype == baseline.dtype
    assert result is not baseline


def test_guard_checks_qs_even_when_median_improves():
    actual, baseline, _ = example()
    # Correct median with implausibly wide tails has worse proper score.
    wide = np.array([[-1000., -900., 0., 900., 1000.]])
    assert score(actual, wide)['mae'] < score(actual, baseline)['mae']
    assert select_candidate(actual, baseline, wide, wide)['fallback']


def test_guard_tolerance_is_relative_and_not_a_performance_budget():
    actual, baseline, improved = example()
    worse_anchor = baseline.astype(np.float64) + 1e-8
    assert not select_candidate(actual, baseline, worse_anchor, improved)['fallback']
    assert select_candidate(actual, baseline, baseline + .001, improved)['fallback']
    near_equal = baseline.astype(np.float64) - 1e-9
    assert select_candidate(actual, baseline, near_equal, near_equal)['fallback']


def test_selection_ties_prefer_shrinkage_then_earlier_epoch():
    actual, baseline, improved = example()
    candidate = select_candidate(actual, baseline, improved, improved, epoch=2)
    smaller = dict(candidate, alpha=.25, epoch=4)
    earlier = dict(smaller, epoch=1)
    fallback = select_candidate(actual, baseline, baseline, baseline)
    assert min([candidate, smaller, earlier, fallback], key=selection_key) is earlier


@pytest.mark.parametrize('alpha', [-1, 2, float('nan'), float('inf')])
def test_invalid_alpha_rejected(alpha):
    _, baseline, improved = example()
    with pytest.raises(ValueError, match='Alpha'):
        blend(baseline, improved, alpha)


def test_invalid_quantiles_targets_and_weights_rejected():
    actual, baseline, improved = example()
    with pytest.raises(ValueError, match='ordered'):
        blend(baseline, improved[..., ::-1], 0)
    with pytest.raises(ValueError, match='finite'):
        score(np.array([np.nan]), baseline)
    with pytest.raises(ValueError, match='shape'):
        score(actual[:, None], baseline)
    with pytest.raises(ValueError, match='nonnegative'):
        BalanceWeights(identity=-1)
    with pytest.raises(ValueError, match='Unknown'):
        build_model('typo')
