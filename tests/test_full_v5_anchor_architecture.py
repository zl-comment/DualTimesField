import io

import pytest
import torch

from forecasting.full_v5_anchor_balance.architecture import (
    VARIANTS, anchor_pretrain_loss, build_model, configure_stage,
)
from forecasting.full_v5_anchor_balance.objective import balance_loss
from forecasting.full_v5_joint.data import HistoryStore
from forecasting.full_v5_joint.model import pinball
from test_full_v5_joint import synthetic


@pytest.fixture
def batch_and_target():
    torch.set_num_threads(2)
    store = HistoryStore(synthetic(), candidates=4)
    return store.batch([50, 60]), store.targets([50, 60])


@pytest.mark.parametrize('variant', VARIANTS)
def test_initial_anchor_is_identity_and_other_weights_match(variant, batch_and_target):
    reference = build_model('balanced', 2026)
    model = build_model(variant, 2026)
    original = reference.state_dict()
    for name, value in model.state_dict().items():
        if not name.startswith('calibration.'):
            assert torch.equal(value, original[name]), name
    batch, _ = batch_and_target
    output = model(batch)
    torch.testing.assert_close(output['anchor'], batch['baseline'], rtol=0, atol=0)
    torch.testing.assert_close(output['prediction'], reference(batch)['prediction'], rtol=0, atol=0)


def test_location_only_is_translation_and_scale_row_has_no_effect(batch_and_target):
    model = build_model('location_only')
    batch, target = batch_and_target
    with torch.no_grad():
        model.calibration[-1].bias[0] = .2
    original = model(batch)
    offset = original['anchor'] - batch['baseline']
    torch.testing.assert_close(offset, offset[..., :1].expand_as(offset))
    with torch.no_grad():
        model.calibration[-1].weight[1].fill_(10000)
        model.calibration[-1].bias[1] = 10000
    output = model(batch)
    torch.testing.assert_close(output['prediction'], original['prediction'], rtol=0, atol=0)
    raw = model.calibration(torch.randn(2, 24, 32))
    assert torch.equal(raw[..., 1], torch.zeros_like(raw[..., 1]))
    loss, _ = balance_loss(output, target, batch['baseline'])
    loss.backward()
    assert model.calibration[-1].weight.grad[1].count_nonzero() == 0
    assert model.calibration[-1].bias.grad[1] == 0


def test_separate_heads_have_independent_parameters_and_gradients():
    model = build_model('separate_heads')
    head = model.calibration
    location_ids = {id(p) for p in head.location.parameters()}
    assert location_ids.isdisjoint(id(p) for p in head.scale.parameters())
    features = torch.randn(2, 24, 32)
    with torch.no_grad():
        head.location[-1].bias.fill_(.25)
        head.scale[-1].bias.fill_(-.5)
    raw = head(features)
    torch.testing.assert_close(raw[..., 0], torch.full((2, 24), .25))
    torch.testing.assert_close(raw[..., 1], torch.full((2, 24), -.5))
    raw[..., 0].sum().backward()
    assert head.location[-1].bias.grad.abs().sum() > 0
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in head.scale.parameters())


def test_head_parameter_budgets_and_staged_architecture_match():
    expected = dict(balanced=842, identity_anchor=842, location_only=842,
                    separate_heads=818, staged=842, wide_head=1682)
    for variant, count in expected.items():
        model = build_model(variant)
        assert sum(p.numel() for p in model.calibration.parameters()) == count
    left, right = build_model('staged').state_dict(), build_model('balanced').state_dict()
    assert left.keys() == right.keys()
    assert all(torch.equal(left[name], right[name]) for name in left)


def test_replacement_heads_preserve_subsequent_cpu_random_draws():
    build_model('balanced', 17)
    expected = torch.rand(32)
    for variant in VARIANTS:
        build_model(variant, 17)
        assert torch.equal(torch.rand(32), expected), variant


def test_stages_control_real_gradients_and_clear_stale_gradients(batch_and_target):
    model = build_model('staged')
    batch, target = batch_and_target
    # Nonzero head weights make encoder gradients observable as well as the head.
    with torch.no_grad():
        model.calibration[-1].weight.fill_(.01)
    configure_stage(model, 'staged', 'anchor')
    loss, _ = anchor_pretrain_loss(model(batch), target, batch['baseline'])
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.state.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.calibration.parameters())
    for name, parameter in model.named_parameters():
        if not name.startswith(('state.', 'calibration.')):
            assert not parameter.requires_grad and parameter.grad is None
    configure_stage(model, 'staged', 'history')
    assert all(p.grad is None for p in model.parameters())
    loss, _ = balance_loss(model(batch), target, batch['baseline'])
    loss.backward()
    for name, parameter in model.named_parameters():
        if name.startswith(('state.', 'calibration.')):
            assert not parameter.requires_grad and parameter.grad is None
    for module in ('dual', 'match', 'alignment_query', 'alignment_key', 'gates'):
        gradients = [p.grad for p in getattr(model, module).parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


@pytest.mark.parametrize('variant', VARIANTS)
def test_checkpoint_restore_reapplies_variant_and_stage(variant, batch_and_target):
    stage = 'history' if variant == 'staged' else 'joint'
    model = configure_stage(build_model(variant), variant, stage)
    with torch.no_grad():
        model.gates[-1].bias.add_(.1)
    stream = io.BytesIO()
    torch.save(dict(variant=variant, stage=stage, state_dict=model.state_dict()), stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    restored = build_model(saved['variant'])
    restored.load_state_dict(saved['state_dict'])
    configure_stage(restored, saved['variant'], saved['stage'])
    assert [p.requires_grad for p in model.parameters()] == [p.requires_grad for p in restored.parameters()]
    batch, _ = batch_and_target
    torch.testing.assert_close(restored(batch)['prediction'], model(batch)['prediction'], rtol=0, atol=0)
    if variant == 'identity_anchor':
        assert all(not p.requires_grad for p in restored.calibration.parameters())


def test_anchor_pretrain_loss_ignores_history_and_has_declared_units():
    baseline = torch.tensor([[[-4., -2., 0., 2., 4.]]])
    anchor = (baseline + 30).requires_grad_()
    target = torch.zeros(1, 1)
    output = dict(anchor=anchor, prediction=torch.full_like(anchor, float('nan')),
                  reconstruction_loss=torch.tensor(float('nan')))
    loss, terms = anchor_pretrain_loss(output, target, baseline)
    expected = pinball(anchor, target) / 300 + .5 * 30 / 300 + .01 * .1 ** 2
    torch.testing.assert_close(loss, expected)
    assert terms['anchor_median'].item() == pytest.approx(.05)
    assert terms['identity'].item() == pytest.approx(.0001)
    loss.backward()
    assert anchor.grad is not None and torch.isfinite(anchor.grad).all()


def test_stage_and_variant_validation():
    with pytest.raises(ValueError, match='Unknown'):
        build_model('legacy')
    with pytest.raises(ValueError, match='Invalid training stage'):
        configure_stage(build_model('balanced'), 'balanced', 'history')
    with pytest.raises(ValueError, match='Invalid training stage'):
        configure_stage(build_model('staged'), 'staged', 'wrong')
