import json

import numpy as np
import pytest
import torch

from forecasting.full_v5.conditional_transfer_gate import (
    ConditionalTransferGate, GateConfig, FIT_START, FIT_CUTOFF,
    MAX_FIT_CUTOFF, fit_transfer_gate,
)

DAY = 86400


def inputs(n=4, d=3):
    features = np.zeros((n, 24, d), dtype=np.float64)
    features[..., 0] = 1.
    trend = np.full((n, 24), 20., dtype=np.float64)
    event = np.full((n, 24), 100., dtype=np.float64)
    anchor = np.broadcast_to(np.array([30., 50., 80., 110., 160.]), (n, 24, 5)).copy()
    return features, trend, event, anchor


def test_clamp_gates_support_exact_null_and_keep_anchor_untouched():
    model = ConditionalTransferGate(3)
    args = inputs()
    expected_anchor = args[-1].copy()
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.fill_(-1.)
    result = model.predict(*args)
    assert np.count_nonzero(result['gate_trend']) == 0
    assert np.count_nonzero(result['gate_event']) == 0
    assert np.count_nonzero(result['delta']) == 0
    np.testing.assert_array_equal(result['quantiles'], expected_anchor)
    np.testing.assert_array_equal(args[-1], expected_anchor)


def test_shared_shift_preserves_order_and_obeys_correction_bound():
    model = ConditionalTransferGate(3)
    features, trend, event, anchor = inputs()
    trend[:] = 150.
    event[:] = 150.
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.fill_(2.)
    result = model.predict(features, trend, event, anchor)
    np.testing.assert_array_equal(result['delta'], np.full((4, 24), 150.))
    assert np.all(np.diff(result['quantiles'], axis=-1) >= 0)
    np.testing.assert_allclose(np.diff(result['quantiles'], axis=-1), np.diff(anchor, axis=-1), atol=0, rtol=0)
    trend[:] = -150.
    result = model.predict(features, trend, event * 0, anchor)
    np.testing.assert_array_equal(result['delta'], np.full((4, 24), -150.))


def test_event_off_holds_both_gates_features_and_trend_fixed():
    model = ConditionalTransferGate(3)
    args = inputs()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[.2, .1, -.1], [.4, -.1, .1]], dtype=torch.float64))
    full, removed = model.predict(*args), model.predict(*args, event_off=True)
    np.testing.assert_array_equal(full['gate_trend'], removed['gate_trend'])
    np.testing.assert_array_equal(full['gate_event'], removed['gate_event'])
    np.testing.assert_array_equal(removed['delta'], full['gate_trend'] * args[1])
    assert np.all(full['delta'] > removed['delta'])
    # A separately trained trend-only control has no event contribution.
    trend_only = ConditionalTransferGate(3, mode='trend').predict(*args)
    assert np.count_nonzero(trend_only['gate_event']) == 0
    np.testing.assert_array_equal(trend_only['delta'], trend_only['gate_trend'] * args[1])


def test_fit_ignores_pre_window_future_and_boundary_labels_and_features():
    origins = np.array([FIT_START - DAY, FIT_START, FIT_START + DAY,
                        FIT_CUTOFF - DAY, FIT_CUTOFF], dtype=np.int64)
    features, trend, event, anchor = inputs(len(origins))
    actual = np.full((len(origins), 24), 150.)
    config = GateConfig(epochs=4)
    first, audit = fit_transfer_gate(origins, actual, anchor, features, trend, event, config=config)
    changed = [a.copy() for a in (actual, anchor, features, trend, event)]
    for value in changed:
        value[[0, 3, 4]] = np.nan
    second, repeated_audit = fit_transfer_gate(origins, *changed, config=config)
    assert audit == repeated_audit
    assert audit['fit_rows'] == 2
    assert audit['latest_fit_label_available'] < FIT_CUTOFF
    assert audit['feature_count'] == 3
    assert audit['trainable_parameters'] == 2 * (3 + 1)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)
    changed[0][1, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite fitting labels'):
        fit_transfer_gate(origins, *changed, config=config)


def test_fit_updates_gate_and_uses_final_fixed_epoch():
    origins = FIT_START + np.arange(4) * DAY
    features, trend, event, anchor = inputs()
    actual = np.full((4, 24), 150.)
    initial = ConditionalTransferGate(3).predict(features, trend, event, anchor)
    fitted, audit = fit_transfer_gate(origins, actual, anchor, features, trend, event,
                                      config=GateConfig(epochs=12))
    result = fitted.predict(features, trend, event, anchor)
    assert np.mean(result['delta']) > np.mean(initial['delta'])
    assert len(audit['trace']) == 12
    assert audit['trace'][-1]['epoch'] == 12
    assert audit['final_objective'] < audit['trace'][0]['loss']
    assert audit['selected_null'] is False
    assert audit['final_selected_objective'] == audit['final_trained_objective']
    assert audit['final_selected_objective'] < audit['null_objective']
    assert 'no epoch search or held-out selection' in audit['selection']


def test_json_checkpoint_exact_prediction_replay_and_shape_validation(tmp_path):
    origins = FIT_START + np.arange(4) * DAY
    features, trend, event, anchor = inputs()
    actual = np.full((4, 24), 150.)
    model, audit = fit_transfer_gate(origins, actual, anchor, features, trend, event,
                                     config=GateConfig(epochs=3))
    path = tmp_path / 'gate.json'
    model.save_checkpoint(path, audit)
    payload = json.loads(path.read_text())
    assert payload['format'] == 'conditional-transfer-gate-v1'
    restored, restored_audit = ConditionalTransferGate.load_checkpoint(path)
    assert restored_audit == audit
    for event_off in (False, True):
        original = model.predict(features, trend, event, anchor, event_off=event_off)
        replay = restored.predict(features, trend, event, anchor, event_off=event_off)
        for key in original:
            np.testing.assert_array_equal(original[key], replay[key])
    payload['weight'] = [[0.], [0.]]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='Malformed gate checkpoint'):
        ConditionalTransferGate.load_checkpoint(path)


def test_invalid_ranges_order_and_fit_cutoff_are_rejected():
    model = ConditionalTransferGate(3)
    features, trend, event, anchor = inputs()
    with pytest.raises(ValueError, match='event in'):
        model.predict(features, trend, -event, anchor)
    with pytest.raises(ValueError, match='event in'):
        model.predict(features, trend + 151, event, anchor)
    reversed_anchor = anchor[..., ::-1].copy()
    with pytest.raises(ValueError, match='ordered'):
        model.predict(features, trend, event, reversed_anchor)
    with pytest.raises(ValueError, match='Feature dimension'):
        model.predict(features[..., :2], trend, event, anchor)
    origins = FIT_START + np.arange(4) * DAY
    with pytest.raises(ValueError, match='October--November'):
        fit_transfer_gate(origins, trend, anchor, features, trend, event,
                          cutoff=MAX_FIT_CUTOFF + DAY)
    with pytest.raises(ValueError, match='No fully matured'):
        fit_transfer_gate(origins, trend, anchor, features, trend, event,
                          cutoff=FIT_START + DAY)


@pytest.mark.parametrize('proposal', ['flat', 'unhelpful'])
def test_training_only_null_fallback_rejects_flat_or_unhelpful_proposals(proposal):
    origins = FIT_START + np.arange(4) * DAY
    features, trend, event, anchor = inputs()
    if proposal == 'flat':
        trend[:] = 0.
        event[:] = 0.
        actual = anchor[..., 2].copy()
    else:
        # Both proposed corrections raise prices when all targets are below
        # every anchor quantile; the exact null is the safe prespecified choice.
        actual = np.full((4, 24), -100.)
    model, audit = fit_transfer_gate(origins, actual, anchor, features, trend, event)
    assert len(audit['trace']) == 80
    assert audit['selected_null'] is True
    assert audit['final_trained_objective'] >= audit['null_objective']
    assert audit['final_selected_objective'] == audit['null_objective']
    for parameter in model.parameters():
        assert torch.count_nonzero(parameter) == 0
    result = model.predict(features, trend, event, anchor)
    np.testing.assert_array_equal(result['quantiles'], anchor)
    assert not np.any(result['gate_trend'])
    assert not np.any(result['gate_event'])
