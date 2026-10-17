"""Small real-model integration checks for the adaptive historical experiment."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from forecasting.full_v5.adaptive_dual_experiment import (
    FixedGeometry, METRICS, MODELS, OCTOBER, REGIONS,
    assess_reconstruction, summarize,
)
from forecasting.full_v5.adaptive_dual_history import AdaptiveHistoricalDualField, FIT_CUTOFF
from forecasting.full_v5.dual_history_memory import build_corrections
from forecasting.full_v5.dual_history_model import HistoricalDualField

DAY = 24 * 3600


def synthetic(n):
    hour = np.arange(24, dtype=np.float32)
    actual = np.repeat((80 + 12 * np.cos(2 * np.pi * hour / 24))[None], n, axis=0)
    for i in range(n):
        actual[i] += (500 + 10 * i) * np.exp(-.5 * ((hour - (3 + i % 15)) / .8) ** 2)
    z = np.zeros((n, 24, 16), dtype=np.float32)
    z[:, :, 1] = np.arange(n, dtype=np.float32)[:, None] / max(n, 1)
    lags = np.full((n, 24, 3), 70., dtype=np.float32)
    quantiles = np.broadcast_to(np.array([30., 50., 80., 110., 160.]), (n, 24, 5)).copy()
    return actual, quantiles, z, lags


def test_real_memories_share_candidates_and_ignore_unreleased_targets():
    torch.set_num_threads(2)
    n = 12
    origins = OCTOBER + np.arange(n, dtype=np.int64) * DAY
    actual, quantiles, z, lags = synthetic(n)
    model = AdaptiveHistoricalDualField()
    # Distinct learned geometry, without costly fitting or using later labels.
    with torch.no_grad():
        model.geometry[-1].bias.copy_(torch.tensor([.3, -.4]))
    memories = [build_corrections(origins, actual, quantiles, z, lags, encoder)
                for encoder in (model, FixedGeometry(model))]
    adaptive, fixed = memories
    assert adaptive['audit']['open_queries'] > 0
    np.testing.assert_array_equal(adaptive['matched_indices'], fixed['matched_indices'])
    np.testing.assert_array_equal(adaptive['recent_indices'], fixed['recent_indices'])
    for control in ('rawmatched', 'recentraw', 'fixedsmooth'):
        np.testing.assert_array_equal(adaptive['corrections'][control], fixed['corrections'][control])
    # Every candidate's full 24h label window was released at that query.
    ids = adaptive['matched_indices']
    for i in range(n):
        chosen = ids[i][ids[i] >= 0]
        assert np.all(origins[chosen] + DAY <= origins[i])
    for memory in memories:
        assert memory['audit']['maturity_violations'] == 0
        encoded = memory['encoded_curves']
        np.testing.assert_allclose(encoded['reconstruction'].mean(1),
                                   actual[encoded['indices']].mean(1), rtol=1e-6, atol=3e-5)
        np.testing.assert_allclose(encoded['trend'] + encoded['event'],
                                   encoded['reconstruction'], rtol=1e-6, atol=3e-5)
    # At query 9, row 9 and later are outcomes that do not yet exist.
    query = 9
    changed = actual.copy()
    changed[query:] += 1_000_000.
    perturbed = build_corrections(origins, changed, quantiles, z, lags, model)
    for name in adaptive['corrections']:
        np.testing.assert_array_equal(adaptive['corrections'][name][:query + 1],
                                      perturbed['corrections'][name][:query + 1])
    np.testing.assert_array_equal(adaptive['matched_indices'], perturbed['matched_indices'])


class RecordingOld(HistoricalDualField):
    def __init__(self):
        super().__init__(fit_cutoff=FIT_CUTOFF)
        self.calls = []

    def forward(self, price, z, lag_prices, **kwargs):
        result = super().forward(price, z, lag_prices, **kwargs)
        self.calls.append((price.detach().clone(), result['hidden_mask'].clone()))
        return result


class RecordingAdaptive(AdaptiveHistoricalDualField):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, price, z, lag_prices, **kwargs):
        visible = kwargs.get('observed_mask')
        self.calls.append((price.detach().clone(), None if visible is None else visible.clone(),
                           kwargs.get('fixed', False)))
        return super().forward(price, z, lag_prices, **kwargs)


def test_september_assessment_uses_same_mask_and_excludes_late_targets():
    torch.set_num_threads(2)
    origins = np.array([FIT_CUTOFF - DAY, FIT_CUTOFF, FIT_CUTOFF + 4 * DAY,
                        OCTOBER - 2 * DAY, OCTOBER - DAY, OCTOBER], dtype=np.int64)
    actual, _, z, lags = synthetic(len(origins))
    old, adaptive = RecordingOld(), RecordingAdaptive()
    original = assess_reconstruction(origins, actual, z, lags, old, adaptive)
    assert original['rows'] == 3
    assert original['latest_label_available'] < OCTOBER
    assert original['selection_used'] is False
    assert all(len(call[0]) == 3 for call in old.calls + adaptive.calls)
    masked_old = old.calls[-1][1]
    masked_calls = [call for call in adaptive.calls if call[1] is not None]
    assert len(masked_calls) == 2
    assert {call[2] for call in masked_calls} == {False, True}
    for _, visible, _ in masked_calls:
        assert torch.equal(visible, ~masked_old)
    assert original['hidden_cells'] == int(masked_old.sum())
    # Neither pre-assessment labels nor not-yet-mature September/October rows
    # may reach either model or the reported assessment metric.
    changed = actual.copy()
    changed[[0, 4, 5]] = np.nan
    repeated = assess_reconstruction(origins, changed, z, lags, old, adaptive)
    assert repeated == original


def result_cells():
    return [dict(region=region,
                 scores={model: dict(metrics={metric: 10. for metric in METRICS})
                         for model in MODELS},
                 changes={'H_adaptive_full': {'fraction': 1., 'mean_abs': 100.}},
                 reconstruction={'event_mean_abs': 100.}) for region in REGIONS]


def test_summary_requires_complete_ordered_three_region_grid():
    cells = result_cells()
    for bad in (cells[:2], [cells[1], cells[0], cells[2]], [cells[0]] * 3):
        with pytest.raises(ValueError, match='three ordered region'):
            summarize(bad)


def test_active_fields_do_not_satisfy_incremental_value_gate():
    cells = result_cells()
    # Events and corrected predictions are nonzero, but performance is equal.
    summary = summarize(cells)
    assert summary['per_region_non_regression'] is True
    assert summary['incremental_value_pass'] is False
    for cell in cells:
        for metric in METRICS:
            cell['scores']['H_adaptive_full']['metrics'][metric] = 9.
    assert summarize(cells)['incremental_value_pass'] is True
    # Macro gains vs C_roll cannot excuse no gain over the fixed representation.
    fixed_tie = deepcopy(cells)
    for cell in fixed_tie:
        cell['scores']['H_fixed_pulses']['metrics'] = dict(cell['scores']['H_adaptive_full']['metrics'])
    assert summarize(fixed_tie)['incremental_value_pass'] is False
    # Nor can cross-region averaging hide degradation in an individual region.
    regression = deepcopy(cells)
    for i, cell in enumerate(regression):
        for metric in METRICS:
            cell['scores']['H_adaptive_full']['metrics'][metric] = 11. if i == 0 else 8.
    summary = summarize(regression)
    assert summary['macro']['H_adaptive_full']['mae'] < summary['macro']['C_roll']['mae']
    assert summary['per_region_non_regression'] is False
    assert summary['incremental_value_pass'] is False
