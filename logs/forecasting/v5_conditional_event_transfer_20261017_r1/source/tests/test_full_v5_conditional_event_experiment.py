"""Synthetic causal integration for the frozen conditional event experiment."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from forecasting.aligned_scoring import assert_comparable
from forecasting.full_v5 import diagnostics
from forecasting.full_v5.adaptive_dual_history import AdaptiveHistoricalDualField
from forecasting.full_v5.conditional_event_experiment import (
    MODELS, METRICS, REGIONS, check_shared_candidates, summarize,
)
from forecasting.full_v5.conditional_event_memory import build_transfer_memory
from forecasting.full_v5.conditional_transfer_gate import (
    FIT_START, FIT_CUTOFF, GateConfig, fit_transfer_gate,
)

DAY = 86400


@pytest.fixture(scope='module')
def world():
    torch.set_num_threads(2)
    n = 36
    origins = FIT_START + np.arange(n, dtype=np.int64) * DAY
    hours = np.arange(24, dtype=np.float64)
    peak = (5 - np.arange(n)) % 24
    actual = 100. + 800. * np.exp(-.5 * ((hours[None] - peak[:, None]) / .8) ** 2)
    anchor = np.broadcast_to(np.array([20., 40., 70., 100., 130.]), (n, 24, 5)).copy()
    z = np.zeros((n, 24, 16), dtype=np.float64)
    z[..., 0] = 1.
    phase = 2 * np.pi * (hours[None] - peak[:, None]) / 24
    z[..., 1], z[..., 2] = np.cos(phase), np.sin(phase)
    lags = np.full((n, 24, 3), 100.)
    encoder = AdaptiveHistoricalDualField()
    # Frozen non-default geometry, independent of these future labels.
    with torch.no_grad():
        encoder.geometry[-1].bias.copy_(torch.tensor([.12, -.2]))
    memories = {
        'full': build_transfer_memory(origins, actual, anchor, z, lags, encoder),
        'unshifted': build_transfer_memory(origins, actual, anchor, z, lags, encoder, shift=False),
        'fixed': build_transfer_memory(origins, actual, anchor, z, lags, encoder, fixed=True),
    }
    memory = memories['full']
    model, audit = fit_transfer_gate(origins, actual, anchor, memory['features'],
                                     memory['trend'], memory['event'], config=GateConfig(epochs=4))
    return dict(origins=origins, actual=actual, anchor=anchor, z=z, lags=lags,
                encoder=encoder, memories=memories, model=model, audit=audit)


def test_actual_memory_candidates_are_shared_unique_and_mature(world):
    memories = world['memories']
    check_shared_candidates(memories)
    full = memories['full']
    assert full['audit']['open_queries'] > 0
    assert full['audit']['nonzero_event_fraction'] > 0
    assert full['audit']['moved_event_fraction'] > 0
    origins = world['origins']
    for i, ids in enumerate(full['event_record_indices']):
        valid = ids >= 0
        ids = ids[valid]
        source_h = full['event_source_h'][i, valid]
        # A query cannot count the same historical delivery-hour event twice.
        source_times = origins[ids] + source_h * 3600
        assert len(np.unique(source_times)) == len(source_times)
        assert np.all(origins[ids] + DAY <= origins[i])
        if len(ids):
            np.testing.assert_allclose(full['event_weights'][i, valid].sum(), 1., atol=1e-12)
    for memory in memories.values():
        assert memory['audit']['maturity_violations'] == 0
    # The runner must reject a seemingly minor change to candidate weighting.
    tampered = deepcopy(memories)
    tampered['fixed']['event_weights'][10, 0] += .001
    with pytest.raises(ValueError, match='event_weights'):
        check_shared_candidates(tampered)
    tampered = deepcopy(memories)
    tampered['unshifted']['event_amplitudes'][10, 0] += 1.
    with pytest.raises(ValueError, match='more than center'):
        check_shared_candidates(tampered)


def test_fixed_head_event_and_timing_interventions_preserve_gates(world):
    full, unshifted = world['memories']['full'], world['memories']['unshifted']
    model, anchor = world['model'], world['anchor']
    prediction = model.predict(full['features'], full['trend'], full['event'], anchor)
    event_off = model.predict(full['features'], full['trend'], full['event'], anchor, event_off=True)
    timing_off = model.predict(full['features'], full['trend'], unshifted['event'], anchor)
    assert not world['audit']['selected_null']
    assert np.any(prediction['gate_event'] > 0)
    for intervention in (event_off, timing_off):
        for key in ('gate_trend', 'gate_event'):
            np.testing.assert_array_equal(prediction[key], intervention[key])
    assert np.any(prediction['quantiles'] != event_off['quantiles'])
    assert np.any(prediction['quantiles'] != timing_off['quantiles'])
    np.testing.assert_allclose(event_off['delta'],
                               np.clip(prediction['gate_trend'] * full['trend'], -150., 150.))
    assert world['audit']['latest_fit_label_available'] < FIT_CUTOFF
    assert world['audit']['last_fit_origin'] + DAY < FIT_CUTOFF


def test_fitted_pipeline_immune_to_current_and_future_assessment_labels(world):
    origins, actual = world['origins'], world['actual']
    query = 33  # November, strictly after all gate-fitting labels.
    assert origins[query] >= FIT_CUTOFF
    changed = actual.copy()
    changed[query:] += 1_000_000.
    altered = build_transfer_memory(origins, changed, world['anchor'], world['z'],
                                     world['lags'], world['encoder'])
    full = world['memories']['full']
    for key in ('features', 'trend', 'event', 'event_record_indices', 'trend_ids'):
        np.testing.assert_array_equal(full[key][:query + 1], altered[key][:query + 1])
    # Even rebuilding AND refitting the whole pipeline cannot consume these
    # labels in the October fit or in predictions made at/before the query.
    model, audit = fit_transfer_gate(origins, changed, world['anchor'], altered['features'],
                                     altered['trend'], altered['event'], config=GateConfig(epochs=4))
    assert audit == world['audit']
    before = world['model'].predict(full['features'], full['trend'], full['event'], world['anchor'])
    after = model.predict(altered['features'], altered['trend'], altered['event'], world['anchor'])
    for key in before:
        np.testing.assert_array_equal(before[key][:query + 1], after[key][:query + 1])


def test_shared_scorer_rejects_changed_origins_or_labels(world):
    origins = world['origins']
    rows = origins >= FIT_CUTOFF
    arrays = dict(origin_unix=origins[rows], actual=world['actual'][rows],
                  target_time_unix=origins[rows, None] + np.arange(24, dtype=np.int64)[None] * 3600)
    meta = dict(protocol_id='synthetic-transfer', region='NSW1', input_hash='shared-inputs',
                sample_hash='shared-samples', preprocessing_hash='shared-preprocessing')
    full = world['memories']['full']
    q = world['model'].predict(full['features'], full['trend'], full['event'], world['anchor'])['quantiles'][rows]
    anchor = diagnostics.build_artifact('C_roll', world['anchor'][rows], arrays, meta, 'november')
    proposed = diagnostics.build_artifact('TE_full', q, arrays, meta, 'november')
    assert_comparable(anchor['artifact'], proposed['artifact'], allow_different_seed=False)
    np.testing.assert_allclose(proposed['scores']['mae'], np.abs(q[..., 2] - arrays['actual']).mean())
    changed = dict(arrays, actual=arrays['actual'] + 1.)
    wrong = diagnostics.build_artifact('wrong-labels', q, changed, meta, 'november')
    with pytest.raises(ValueError, match='actual differs'):
        assert_comparable(proposed['artifact'], wrong['artifact'])
    changed = dict(arrays, origin_unix=arrays['origin_unix'] + 3600,
                   target_time_unix=arrays['target_time_unix'] + 3600)
    wrong = diagnostics.build_artifact('wrong-origins', q, changed, meta, 'november')
    with pytest.raises(ValueError, match='origin_unix differs'):
        assert_comparable(proposed['artifact'], wrong['artifact'])


def summary_cells():
    return [dict(region=region, scores={name: dict(metrics={metric: 10. for metric in METRICS})
                                      for name in MODELS},
                 activation={'TE_full': {'gate_event': {'positive_fraction': 1.}}})
            for region in REGIONS]


def test_complete_summary_requires_performance_not_merely_event_activation():
    cells = summary_cells()
    for invalid in (cells[:2], [cells[1], cells[0], cells[2]], [cells[0]] * 3):
        with pytest.raises(ValueError, match='Three ordered'):
            summarize(invalid)
    assert not summarize(cells)['incremental_value_pass']
    for cell in cells:
        for metric in METRICS:
            cell['scores']['TE_full']['metrics'][metric] = 9.
    assert summarize(cells)['incremental_value_pass']
    fixed_tie = deepcopy(cells)
    for cell in fixed_tie:
        cell['scores']['TE_fixed_pulses']['metrics'] = dict(cell['scores']['TE_full']['metrics'])
    assert not summarize(fixed_tie)['incremental_value_pass']
    bad_region = deepcopy(cells)
    for i, cell in enumerate(bad_region):
        for metric in METRICS:
            cell['scores']['TE_full']['metrics'][metric] = 11. if i == 0 else 8.
    summary = summarize(bad_region)
    assert summary['macro_wins_mae_and_qs']['C_roll']
    assert not summary['per_region_non_regression']
    assert not summary['incremental_value_pass']
