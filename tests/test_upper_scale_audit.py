"""No-training tests for one frozen change: restoring the upper scale to one."""
import inspect
import json

import numpy as np
import pytest
import torch

from forecasting.full_v5_upper_scale_audit.core import (
    common_history_audit, compute_comparison, remove_upper_scale,
)
from forecasting.full_v5_upper_scale_audit.audit import evaluate_decision


def decision_fixture():
    cfg = dict(regions=['NSW1', 'QLD1', 'TAS1'], seeds=[2026, 2027, 2028],
               decision_price_groups=['actual/0_to_300', 'actual/0_to_100', 'actual/1000_plus'])
    rows = []
    for region in cfg['regions']:
        for seed in cfg['seeds']:
            for group in ('all', *cfg['decision_price_groups']):
                rows.append(dict(role='primary', scope='formal', variant='Full_joint',
                                 mode='guarded', representation='selected', region=region,
                                 seed=seed, group=group, qs_delta=-.1, cells=100, empty=False,
                                 point_equal=True, lower_quantiles_equal=True,
                                 quantiles_ordered=True, common_history_passed=True))
    return cfg, rows


def test_upper_removal_preserves_three_quantiles_and_scores_even_for_negative_prices():
    baseline = np.array([[-40., -20., -5., 0., 10.], [5., 5., 5., 5., 5.]])
    full = np.array([[-48., -27., -7., 1., 20.], [8., 8., 8., 8., 8.]])
    off = remove_upper_scale(baseline, full)
    np.testing.assert_array_equal(off[..., :3], full[..., :3])
    np.testing.assert_array_equal(off[..., 3:]-off[..., 2:3], baseline[..., 3:]-baseline[..., 2:3])
    assert (np.diff(off, axis=-1) >= 0).all()
    actual = np.array([-12., 1000.])
    result = compute_comparison(actual, full, off)
    assert result['overall']['mae_delta'] == 0.
    assert result['overall']['pinball_q50_delta'] == 0.
    # A prediction transform has no access to outcomes, including through kwargs.
    assert list(inspect.signature(remove_upper_scale).parameters) == ['baseline', 'full']
    with pytest.raises(TypeError):
        remove_upper_scale(baseline, full, actual=actual)
    np.testing.assert_array_equal(remove_upper_scale(baseline, baseline), baseline)


@pytest.mark.parametrize('bounded', [False, True])
def test_real_model_upper_only_intervention_leaves_history_and_median_unchanged(bounded):
    from forecasting.full_v5_anchor_balance.confirmation_model import AsymmetricScaleModel
    from forecasting.full_v5_calibration_robustness.model import BoundedCalibrationModel
    from test_full_v5_anchor_architecture_experiment import synthetic_store
    torch.set_num_threads(2)
    torch.manual_seed(73)
    model = (BoundedCalibrationModel if bounded else AsymmetricScaleModel)().eval()
    batch = synthetic_store().batch([7, 8])
    with torch.no_grad():
        # Include nonzero input-dependent location and separate tail changes.
        model.calibration[-1].weight.normal_(0, .025)
        model.calibration[-1].bias.copy_(torch.tensor([.12, -.2, .3]))
        before = model(batch)
        model.calibration[-1].weight[2].zero_()
        model.calibration[-1].bias[2].zero_()
        after = model(batch)
    for key in ('correction', 'trend', 'event', 'gates', 'weights'):
        torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
    torch.testing.assert_close(after['anchor'][..., 2], before['anchor'][..., 2], rtol=0, atol=0)
    torch.testing.assert_close(after['prediction'][..., :3], before['prediction'][..., :3], rtol=0, atol=0)
    assert not torch.equal(after['prediction'][..., 3:], before['prediction'][..., 3:])
    baseline = batch['baseline'].numpy()
    original = before['prediction'].numpy()
    common_history_audit(baseline, before['anchor'].numpy(), original, 1.)
    transformed = remove_upper_scale(baseline, original)
    # Independently executing the original neural forward gives the same
    # counterfactual up to its FP32 addition/subtraction rounding.
    rounding = 8*np.finfo(np.float32).eps*(np.abs(transformed)+np.abs(baseline)+1)
    assert np.all(np.abs(transformed-after['prediction'].numpy()) <= rounding)


def test_common_history_tolerates_fp32_addition_but_rejects_a_real_tail_change():
    baseline = np.array([[-7000., -30., 1., 15000., 30000.]], dtype=np.float32)
    anchor = baseline + np.float32(.3137)
    full = anchor + np.float32(.1173)
    # Common FP32 addition is not an exactly common real-valued difference.
    assert np.ptp((full-anchor).astype(np.float64)) > 0
    common_history_audit(baseline, anchor, full, 1.)
    alpha = .25
    selected_anchor = (1-alpha)*baseline.astype(np.float64)+alpha*anchor.astype(np.float64)
    selected_full = (1-alpha)*baseline.astype(np.float64)+alpha*full.astype(np.float64)
    common_history_audit(baseline, selected_anchor, selected_full, alpha)
    changed = selected_full.copy()
    changed[..., 4] += 1.
    with pytest.raises((ValueError, AssertionError)):
        common_history_audit(baseline, selected_anchor, changed, alpha)
    common_history_audit(baseline, baseline, baseline, 0.)
    with pytest.raises((ValueError, AssertionError)):
        common_history_audit(baseline, selected_anchor, selected_full, 0.)


def test_decision_requires_regional_and_tail_protection_despite_macro_gain():
    cfg, rows = decision_fixture()
    assert evaluate_decision(rows, cfg)['passed']
    for row in rows:
        row['qs_delta'] = .01 if row['region'] == 'NSW1' else -.5
    assert not evaluate_decision(rows, cfg)['passed']
    cfg, rows = decision_fixture()
    for row in rows:
        if row['region'] == 'NSW1' and row['group'] == 'actual/1000_plus':
            row['qs_delta'] = .01
    assert not evaluate_decision(rows, cfg)['passed']


def test_empty_or_missing_primary_cells_cannot_be_replaced_by_successful_other_modes():
    cfg, rows = decision_fixture()
    for row in rows:
        if row['region'] == 'TAS1' and row['group'] == 'actual/1000_plus':
            row.update(cells=0, empty=True)
            row.pop('qs_delta')
    assert not evaluate_decision(rows, cfg)['passed']
    cfg, rows = decision_fixture()
    raw = [dict(row, role='sensitivity', variant='Full_joint_raw', mode='raw', qs_delta=-10.)
           for row in rows]
    primary = [row for row in rows if row['region'] != 'NSW1']
    assert not evaluate_decision(primary+raw, cfg)['passed']
    assert not evaluate_decision(raw, cfg)['passed']


def test_saved_case_passes_common_history_and_never_routes_on_changed_outcomes(tmp_path):
    from forecasting.full_v5_upper_scale_audit.audit import _save_case
    baseline = np.broadcast_to([-30., -20., 0., 20., 30.], (1, 24, 5)).copy()
    anchor = 2.+baseline*np.array([.9, .9, 1., 1.1, 1.1])
    full = anchor+3.
    origins = np.array([1672531200], dtype=np.int64)
    targets = origins[:, None]+np.arange(24)*3600
    actual = np.full((1, 24), 3.)
    identity = dict(case_id='formal/Full_joint/guarded/seed2026/NSW1',
                    role='primary', scope='formal', variant='Full_joint', mode='guarded',
                    representation='selected', region='NSW1', seed=2026, fold='2023-2024')
    selection = dict(epoch=7, alpha=1., checkpoint_sha256='unchanged-checkpoint')
    rows, artifact = _save_case(tmp_path/'original_labels', identity, actual, baseline,
                               full, origins, targets, selection, anchor, 1e-10)
    assert artifact['common_history']['passed'] is True
    assert all(row['common_history_passed'] is True for row in rows)
    tail = next(row for row in rows if row['group'] == 'actual/1000_plus')
    assert tail['cells'] == 0 and tail['empty'] is True and 'qs_delta' not in tail
    _, changed = _save_case(tmp_path/'changed_labels', identity, actual+100000., baseline,
                            full, origins, targets, selection, anchor, 1e-10)
    with np.load(artifact['path'], allow_pickle=False) as first, np.load(changed['path'], allow_pickle=False) as second:
        np.testing.assert_array_equal(first['upper_off'], second['upper_off'])
        metadata = json.loads(str(first['metadata_json']))
        other_metadata = json.loads(str(second['metadata_json']))
        assert metadata == other_metadata
        assert metadata['epoch'] == 7 and metadata['alpha'] == 1.
        assert not metadata['outcome_routing']


@pytest.mark.parametrize('alpha', [0., .25, .5, 1.])
def test_upper_removal_commutes_with_the_frozen_alpha_within_fp64_rounding(alpha):
    rng = np.random.default_rng(29)
    median = rng.uniform(-100, 15000, (2, 24, 1))
    gaps = np.array([-100., -40., 0., 60., 160.])
    baseline = (median+gaps).astype(np.float32).astype(np.float64)
    full = (median+13.25+gaps*np.array([.9, .9, 1., 1.1, 1.1])).astype(np.float32).astype(np.float64)
    selected = baseline if alpha == 0 else full if alpha == 1 else (1-alpha)*baseline+alpha*full
    after_selection = remove_upper_scale(baseline, selected)
    raw_off = remove_upper_scale(baseline, full)
    before_selection = baseline if alpha == 0 else raw_off if alpha == 1 else (1-alpha)*baseline+alpha*raw_off
    if alpha == 0:
        np.testing.assert_array_equal(after_selection, baseline)
    else:
        # Fixed conservative bound for the two subtraction/addition and affine
        # blend paths; it is set from precision/magnitudes, not observed error.
        bound = 16*np.finfo(np.float64).eps*(np.abs(baseline)+np.abs(full)+
                                            np.abs(baseline[..., 2:3])+np.abs(full[..., 2:3])+1)
        assert np.all(np.abs(after_selection-before_selection) <= bound)
    np.testing.assert_array_equal(after_selection[..., :3], selected[..., :3])
