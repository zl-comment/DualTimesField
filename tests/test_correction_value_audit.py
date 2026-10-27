"""No-training checks for frozen correction diagnostics and interventions."""
import hashlib
import json

import numpy as np
import pytest

from forecasting.full_v5_correction_audit.metrics import (
    anchor_stages, movement, score, score_and_gain,
)


def test_gain_keeps_harm_visible_and_distinguishes_crossing_from_overshoot():
    baseline = np.broadcast_to([-4., -2., 0., 2., 4.], (4, 5)).copy()
    actual = np.array([1., -1., 1., 0.])
    candidate = baseline + np.array([1.5, 1., 3., 2.])[:, None]
    result = score_and_gain(actual, baseline, candidate)
    # Crossing y=1 by 0.5 still improves MAE; crossing it by 2 does not.
    assert result['mae_delta'] == pytest.approx(.875)
    assert result['mae_benefit'] == pytest.approx(.125)
    assert result['mae_harm'] == pytest.approx(1.)
    assert result['wrong_direction_rate'] == .25
    assert result['target_crossing_rate'] == .5
    assert result['harmful_overshoot_rate'] == .25
    for name in ('mae', 'qs', 'pinball_q05', 'pinball_q10',
                 'pinball_q50', 'pinball_q90', 'pinball_q95'):
        assert result[f'{name}_harm'] - result[f'{name}_benefit'] == pytest.approx(
            result[f'{name}_delta'])
    for prefix in ('before', 'after'):
        assert result[f'{prefix}_pinball_q50'] == result[f'{prefix}_mae'] / 2


def test_tail_attribution_preserves_median_and_reconciles_without_dividing_by_width():
    baseline = np.array([[-4., -2., 0., 2., 4.], [5., 5., 5., 5., 5.]])
    anchor = np.array([[-3., 0., 1., 5., 8.], [7., 7., 7., 7., 7.]])
    actual = np.array([3., 4.])
    stages = anchor_stages(baseline, anchor)
    assert [name for name, _ in stages] == ['baseline', 'location', 'lower_scale', 'upper_scale']
    np.testing.assert_array_equal(stages[-1][1], anchor)
    for _, quantiles in stages[1:]:
        np.testing.assert_array_equal(quantiles[:, 2], anchor[:, 2])
        assert score(actual, quantiles)['mae'] == score(actual, anchor)['mae']
        assert np.isfinite(quantiles).all()
    changes = [score_and_gain(actual, left[1], right[1])
               for left, right in zip(stages[:-1], stages[1:])]
    total = score_and_gain(actual, baseline, anchor)
    for name in ('mae', 'qs', 'pinball_q05', 'pinball_q95'):
        assert sum(change[f'{name}_delta'] for change in changes) == pytest.approx(
            total[f'{name}_delta'])
    assert changes[1]['mae_delta'] == changes[2]['mae_delta'] == 0.


def test_movement_exposes_cancellation_instead_of_treating_components_as_net_change():
    baseline = np.broadcast_to([-4., -2., 0., 2., 4.], (2, 5)).copy()
    anchor = baseline + 8.
    full = anchor - 6.
    result = movement(baseline, anchor, full)
    assert result['mean_abs_total'] == 2.
    assert result['opposing_fraction'] == 1.
    assert result['cancelled_magnitude'] == 12.
    assert result['cancellation_fraction'] == pytest.approx(12 / 14)
    assert result['historical_common_shift_max_error'] == 0.


def test_scores_reject_crossed_or_nonfinite_inputs():
    baseline = np.array([[-4., -2., 0., 2., 4.]])
    with pytest.raises(ValueError, match='Crossed'):
        score(np.array([0.]), baseline[:, ::-1])
    with pytest.raises(ValueError, match='Nonfinite'):
        score(np.array([np.nan]), baseline)
    with pytest.raises(ValueError, match='nonempty'):
        score(np.empty(0), np.empty((0, 5)))


def test_frozen_audit_rejects_mutated_inputs_and_manifest(tmp_path):
    from scripts.audit_correction_value import digest, verify_frozen
    source = tmp_path / 'source.py'
    archive = tmp_path / 'selected_predictions.npz'
    source.write_bytes(b'no training')
    archive.write_bytes(b'frozen selected predictions')
    manifest = dict(sources={str(source): digest(source)},
                    protected_inputs={str(archive): digest(archive)},
                    selected_epoch=7, selected_alpha=.25)
    manifest['run_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    verify_frozen(manifest)
    tampered = dict(manifest, selected_alpha=1.)
    with pytest.raises(ValueError, match='fingerprint changed'):
        verify_frozen(tampered)
    archive.write_bytes(b'silently replaced predictions')
    with pytest.raises(ValueError, match='protected_inputs changed'):
        verify_frozen(manifest)


def test_completed_stage_cannot_recertify_changed_or_added_artifacts(tmp_path):
    from scripts.audit_correction_value import load_completed, save_completed
    stage = tmp_path / 'saved'
    stage.mkdir()
    artifact = stage / 'adverse_groups.csv'
    original = b'group,mae_delta\nweak,1.5\n'
    artifact.write_bytes(original)
    save_completed(tmp_path, 'saved', 'frozen-run', {'negative_groups': 1})
    assert load_completed(tmp_path, 'saved', 'frozen-run') == {'negative_groups': 1}
    artifact.write_bytes(b'group,mae_delta\nweak,-1.5\n')
    with pytest.raises(ValueError, match='artifacts changed'):
        load_completed(tmp_path, 'saved', 'frozen-run')
    artifact.write_bytes(original)
    (stage / 'posthoc.csv').write_bytes(b'new result')
    with pytest.raises(ValueError, match='artifacts changed'):
        load_completed(tmp_path, 'saved', 'frozen-run')


def test_origin_budget_matches_real_displacement_without_cross_origin_or_label_input():
    from forecasting.full_v5_correction_audit.replay import budget_controls
    baseline = np.broadcast_to([-4., -2., 0., 2., 4.], (3, 24, 5)).copy()
    candidate = baseline.copy()
    candidate[0, :12] += 10.
    candidate[0, 12:] += 1.
    candidate[1] -= 3.
    gates = np.ones((3, 24))
    gates[0, :12] = .1
    gates[1] = .2
    control = budget_controls(baseline, candidate, gates)
    # The average gate is .55, but its budget-matched multiplier is 2/11.
    assert control['applied_constant'][0] == pytest.approx(2 / 11)
    assert control['applied_constant'][0] != pytest.approx(gates[0].mean())
    assert control['zero_budget'].tolist() == [False, False, True]
    assert control['applied_constant'][2] == 0.
    for key in ('quantiles_final_state', 'quantiles_matched_uniform'):
        assert (np.diff(control[key], axis=-1) >= 0).all()
        assert np.isfinite(control[key]).all()
    np.testing.assert_allclose(control['state_budget'], control['uniform_budget'], rtol=0, atol=1e-12)
    # Another origin's predictions cannot change origin 0's decision.
    candidate[1] += 100000.
    gates[1] = 0.
    changed = budget_controls(baseline, candidate, gates)
    np.testing.assert_array_equal(changed['quantiles_matched_uniform'][0],
                                  control['quantiles_matched_uniform'][0])
    np.testing.assert_array_equal(control['quantiles_matched_uniform'][2], baseline[2])


def test_frozen_anchor_gate_preserves_scale_only_median_but_recomputes_history_for_location():
    import torch
    from forecasting.full_v5_calibration_robustness.model import BoundedCalibrationModel
    from forecasting.full_v5_correction_audit.replay import InterventionModel
    from test_full_v5_anchor_architecture_experiment import synthetic_store
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    model = InterventionModel().eval()
    with torch.no_grad():
        model.calibration[-1].bias.copy_(torch.tensor([.2, -.3, .4]))
        reference = BoundedCalibrationModel().eval()
        reference.load_state_dict(model.state_dict())
        batch = synthetic_store().batch([7])
        original = model(batch)
        expected = reference(batch)
        for name in ('prediction', 'anchor', 'trend', 'event', 'gates', 'weights'):
            torch.testing.assert_close(original[name], expected[name], rtol=0, atol=0)
        model.set_gate([1., 0., 0.])
        scales_off = model(batch)
        torch.testing.assert_close(scales_off['prediction'][..., 2], original['prediction'][..., 2], rtol=0, atol=0)
        torch.testing.assert_close(scales_off['correction'], original['correction'], rtol=0, atol=0)
        model.set_gate([0., 0., 0.])
        zero_anchor = model(batch)
        torch.testing.assert_close(zero_anchor['anchor'], batch['baseline'], rtol=0, atol=1e-5)
        assert zero_anchor['correction'].abs().max() > 0
        assert not torch.equal(zero_anchor['trend'], original['trend'])
        assert not torch.equal(zero_anchor['prediction'], batch['baseline'])
        with pytest.raises(ValueError, match='shape'):
            model.set_gate(torch.ones(24, 3))
        with pytest.raises(ValueError, match='Finite'):
            model.set_gate([1., -1., 1.])


def test_intervention_reloads_history_without_current_or_immature_labels():
    import torch
    from forecasting.full_v5_correction_audit.replay import InterventionModel, _assertions
    from test_full_v5_anchor_architecture_experiment import synthetic_store
    torch.set_num_threads(2)
    model = InterventionModel().eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    result = _assertions(model, synthetic_store(), 7)
    assert result['current_and_immature_label_invariance_exact']
    assert result['perturbed_immature_rows'] == 5
    assert result['unit_gate_exact'] and result['nonzero_location_exact']
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_covered_weak_evidence_keeps_uncovered_and_empty_strata_distinct():
    from forecasting.full_v5_correction_audit.saved import state_groups, _rows
    states = np.array([[0, 5]])
    gates = np.array([[[.8, .7, .6], [0., 0., 0.]]])
    groups = state_groups(states, gates)
    assert groups['weak_location'].sum() == 1
    assert groups['uncovered'].sum() == 1
    assert groups['covered_weak_location'].sum() == 0
    baseline = np.broadcast_to([-4., -2., 0., 2., 4.], (1, 2, 5)).copy()
    rows = _rows(np.array([[1., -1.]]), baseline, baseline+2, groups, {})
    indexed = {row['group']: row for row in rows}
    assert indexed['covered_weak_location']['empty']
    assert 'mae_delta' not in indexed['covered_weak_location']
    for metric in ('mae_delta', 'qs_delta'):
        weighted = sum(indexed[name][metric]*indexed[name]['cells']
                       for name in ('covered', 'uncovered')) / indexed['all']['cells']
        assert weighted == pytest.approx(indexed['all'][metric])


def test_intervention_cannot_mix_selected_checkpoints_or_alpha():
    from forecasting.full_v5_correction_audit.saved import validate_same_checkpoint
    selected = dict(checkpoint_sha256='guarded-epoch-7', epoch=7, alpha=.25)
    entries = dict(full=dict(selected), anchor=dict(selected), event_off=dict(selected),
                   raw=dict(checkpoint_sha256='raw-epoch-12', epoch=12, alpha=1.))
    validate_same_checkpoint(entries, ('full', 'anchor', 'event_off'))
    # Independently selected raw outputs cannot be relabeled as pre-alpha Full.
    with pytest.raises(ValueError, match='checkpoint, epoch and alpha'):
        validate_same_checkpoint(entries, ('full', 'raw'))
    for field, bad in (('checkpoint_sha256', 'other-weights'), ('epoch', 8), ('alpha', .5)):
        entries['event_off'] = dict(selected, **{field: bad})
        with pytest.raises(ValueError, match='checkpoint, epoch and alpha'):
            validate_same_checkpoint(entries, ('full', 'event_off'))


def test_actual_price_diagnostics_partition_threshold_boundaries():
    from forecasting.full_v5_correction_audit.saved import price_groups
    actual = np.array([[-1., 0., 99.99, 100., 299.99, 300., 999.99, 1000.]])
    groups = price_groups(actual)
    np.testing.assert_array_equal(np.sum(list(groups.values()), axis=0), np.ones(actual.shape))
    assert [int(mask.sum()) for mask in groups.values()] == [1, 2, 2, 2, 1]
