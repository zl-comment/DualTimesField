"""Independent availability, weighting and conditional-prior contracts."""
import copy

import numpy as np
import pytest

from forecasting.aligned_inputs import AlignedSplit, INPUT_KEYS
from forecasting.full_v3.fields import decompose_curves
from forecasting.full_v3.memory import HistoricalMemory
from forecasting.full_v3.priors import build_priors


NAMES = ('v3_level_lgb', 'v3_analog_features_lgb', 'v3_analog_prior_lgb',
         'v3_dual_shared_lgb', 'v3_dual_conditional_lgb')


def _split(origins, seed=52):
    rng = np.random.default_rng(seed)
    origins = np.asarray(origins, dtype=np.int64)
    n = len(origins)
    shapes = ((72, 2), (72, 10), (24, 10), (1,), (24, 11), (24, 1), (24, 3))
    inputs = {key: rng.normal(size=(n, *shape)).astype(np.float64)
              for key, shape in zip(INPUT_KEYS, shapes)}
    actual = 50 + rng.normal(size=(n, 24)) * 60
    actual[:, 11] += np.arange(n) * 70
    targets = origins[:, None] + np.arange(24)[None, :] * 3600
    metadata = {'region': 'NSW1', 'preprocessing_hash': 'same', 'target_mean': 50.,
                'target_std': 20., 'price_transform': 'identity', 'protocol_id': 'fixture',
                'sample_hash': 'fixture', 'input_hash': 'fixture', 'split': 'selection',
                'input_block_order': list(INPUT_KEYS)}
    return AlignedSplit(inputs, actual, origins, targets, metadata, (actual - 50) / 20)


def _fixture(*, late_candidate=True):
    start = 1577836800  # 2020-01-01 00:00 UTC, hourly aligned.
    library = _split(start + np.arange(12) * 7 * 86400)
    query = _split(library.origin_unix[[8, 9]], seed=937)
    required = (library.origin_unix[:, None] + np.arange(-72, 24)[None, :] * 3600).ravel()
    availability = {int(t): int(t + 3600) for t in required}
    if late_candidate:
        # A calendar-eligible previous-week curve is not fully released yet.
        unavailable_hour = int(library.target_time_unix[7, -1])
        availability[unavailable_hour] = int(query.origin_unix[-1] + 3600)
    memory_config = {'year_lags': [], 'day_offsets': [],
                     'weekly_lags_days': [7, 14, 21, 28, 35, 42]}
    memory = HistoricalMemory({'library': library}, availability, config=memory_config)
    fields = decompose_curves(memory.curves)
    return library, query, availability, memory_config, memory, fields


def _assert_same(left, right):
    for key in ('raw_features', 'field_features'):
        np.testing.assert_array_equal(left[key], right[key])
    for key in NAMES:
        np.testing.assert_array_equal(left['priors'][key], right['priors'][key])
    for key in ('indices', 'valid', 'max_availability', 'w_shared', 'w_background', 'w_event'):
        np.testing.assert_array_equal(left['audit'][key], right['audit'][key])


def test_query_label_poisoning_and_unavailable_library_values_cannot_change_predictions():
    library, query, availability, memory_config, memory, fields = _fixture()
    reference = build_priors(memory, query, fields)
    poisoned_query = copy.deepcopy(query)
    poisoned_query.actual[:] = np.nan
    poisoned_query.target[:] = np.inf
    _assert_same(reference, build_priors(memory, poisoned_query, fields))

    # Row 7 is historical but delayed; rows 10/11 are future to both queries.
    # Changing their outcomes, levels and retrieval inputs must have no effect.
    assert 7 not in reference['audit']['indices']
    poisoned_library = copy.deepcopy(library)
    for row in (7, 10, 11):
        poisoned_library.actual[row] = 1e9 + np.arange(24) * 1e7
        poisoned_library.target[row] = -1e8
        for value in poisoned_library.inputs.values():
            value[row] += 1e6
    poisoned_memory = HistoricalMemory({'library': poisoned_library}, availability, config=memory_config)
    poisoned_fields = decompose_curves(poisoned_memory.curves)
    _assert_same(reference, build_priors(poisoned_memory, query, poisoned_fields))


def test_weight_simplexes_availability_and_all_five_prior_equations():
    _, query, _, _, memory, fields = _fixture()
    result = build_priors(memory, query, fields)
    audit = result['audit']
    indices, valid = audit['indices'], audit['valid']
    assert indices.shape == valid.shape == (len(query), memory.max_candidates)
    assert valid.dtype == np.bool_
    np.testing.assert_array_equal(valid, indices >= 0)
    assert np.all(audit['max_availability'][valid] <=
                  np.broadcast_to(query.origin_unix[:, None], valid.shape)[valid])
    for key in ('w_shared', 'w_background'):
        weight = audit[key]
        assert weight.shape == indices.shape and np.isfinite(weight).all()
        assert (weight >= 0).all() and (weight[~valid] == 0).all()
        assert (np.count_nonzero(weight, axis=1) <= 4).all()
        np.testing.assert_allclose(weight.sum(1), 1, atol=1e-14)
    assert audit['w_event'].shape == (*indices.shape, 24)
    assert (audit['w_event'][~valid] == 0).all()
    np.testing.assert_allclose(audit['w_event'].sum(1), 1, atol=1e-14)
    assert (np.count_nonzero(audit['w_event'], axis=1) <= 4).all()

    level = np.median(query.inputs['history_values'][:, -24:, 0] * 20 + 50, axis=1)
    safe = np.maximum(indices, 0)
    candidate_levels = memory.levels[safe]
    centered = memory.curves[safe] - candidate_levels[..., None]
    c = fields['background'][safe] - candidate_levels[..., None]
    d = fields['event'][safe]
    ws, wc, wd = audit['w_shared'], audit['w_background'], audit['w_event']
    shared_c = np.einsum('nk,nkh->nh', ws, c)
    shared_d = np.einsum('nk,nkh->nh', ws, d)
    dual_c = np.einsum('nk,nkh->nh', wc, c)
    dual_d = np.einsum('nkh,nkh->nh', wd, d)
    expected = {
        'v3_level_lgb': np.broadcast_to(level[:, None], (len(query), 24)),
        'v3_analog_features_lgb': np.zeros((len(query), 24)),
        'v3_analog_prior_lgb': level[:, None] + np.einsum('nk,nkh->nh', ws, centered),
        'v3_dual_shared_lgb': level[:, None] + shared_c + shared_d,
        'v3_dual_conditional_lgb': level[:, None] + dual_c + dual_d,
    }
    assert set(result['priors']) == set(NAMES)
    for name in NAMES:
        np.testing.assert_allclose(result['priors'][name], expected[name], atol=1e-12)
    expected_fields = np.concatenate((shared_c, shared_d, dual_c, dual_d), axis=1) / 20
    assert result['field_features'].shape == (len(query), 96)
    np.testing.assert_allclose(result['field_features'], expected_fields, atol=1e-12)
    assert result['raw_features'].shape == (len(query), memory.max_candidates * 29)
    assert np.isfinite(result['raw_features']).all()


def test_no_available_candidates_has_explicit_level_fallback_and_zero_features():
    _, query, _, memory_config, memory, _ = _fixture()
    early = copy.deepcopy(query)
    early.origin_unix = np.array([memory.origins[0], memory.origins[0] + 3600])
    early.target_time_unix = early.origin_unix[:, None] + np.arange(24) * 3600
    fields = decompose_curves(memory.curves)
    result = build_priors(memory, early, fields)
    assert not result['audit']['valid'].any()
    assert (result['audit']['indices'] == -1).all()
    for key in ('w_shared', 'w_background', 'w_event'):
        assert np.count_nonzero(result['audit'][key]) == 0
    assert np.count_nonzero(result['field_features']) == 0
    assert np.isfinite(result['raw_features']).all()
    level = np.median(early.inputs['history_values'][:, -24:, 0] * 20 + 50, axis=1)
    for name in NAMES:
        expected = np.zeros((len(early), 24)) if name == 'v3_analog_features_lgb' else np.repeat(level[:, None], 24, axis=1)
        np.testing.assert_array_equal(result['priors'][name], expected)


def test_translation_changes_only_raw_price_priors_not_retrieval_or_features():
    library, query, availability, memory_config, memory, fields = _fixture()
    reference = build_priors(memory, query, fields)
    shift = 250.
    shifted_library, shifted_query = copy.deepcopy(library), copy.deepcopy(query)
    for split in (shifted_library, shifted_query):
        split.inputs['history_values'][..., 0] += shift / 20
        split.actual += shift
        split.target += shift / 20
    shifted_memory = HistoricalMemory({'library': shifted_library}, availability, config=memory_config)
    shifted = build_priors(shifted_memory, shifted_query, decompose_curves(shifted_memory.curves))
    for key in ('indices', 'valid', 'max_availability'):
        np.testing.assert_array_equal(shifted['audit'][key], reference['audit'][key])
    for key in ('w_shared', 'w_background', 'w_event'):
        np.testing.assert_allclose(shifted['audit'][key], reference['audit'][key], atol=1e-13)
    for key in ('raw_features', 'field_features'):
        np.testing.assert_allclose(shifted[key], reference[key], atol=1e-11)
    for name in NAMES:
        increment = 0 if name == 'v3_analog_features_lgb' else shift
        np.testing.assert_allclose(shifted['priors'][name], reference['priors'][name] + increment, atol=1e-10)


def test_query_batch_size_and_input_reordering_do_not_change_features_or_priors():
    _, query, _, _, memory, fields = _fixture()
    reference = build_priors(memory, query, fields, config={'batch_size': 1})
    batched = build_priors(memory, query, fields, config={'batch_size': 256})
    _assert_same(reference, batched)
    reversed_query = AlignedSplit({k: v[::-1].copy() for k, v in query.inputs.items()},
        query.actual[::-1].copy(), query.origin_unix[::-1].copy(), query.target_time_unix[::-1].copy(),
        dict(query.metadata), query.target[::-1].copy())
    reversed_result = build_priors(memory, reversed_query, fields)
    for key in ('raw_features', 'field_features'):
        np.testing.assert_array_equal(reversed_result[key], reference[key][::-1])
    for key in NAMES:
        np.testing.assert_array_equal(reversed_result['priors'][key], reference['priors'][key][::-1])


def test_top_four_ties_are_stable_in_calendar_candidate_order():
    library, query, availability, memory_config, _, _ = _fixture(late_candidate=False)
    for key in INPUT_KEYS:
        library.inputs[key][:] = 0
        query.inputs[key][:] = 0
    memory = HistoricalMemory({'library': library}, availability, config=memory_config)
    result = build_priors(memory, query, decompose_curves(memory.curves))
    expected = np.zeros((len(query), memory.max_candidates))
    expected[:, :4] = .25
    for key in ('w_shared', 'w_background'):
        np.testing.assert_array_equal(result['audit'][key], expected)
    np.testing.assert_array_equal(result['audit']['w_event'], np.repeat(expected[..., None], 24, axis=-1))


def test_distances_softmax_and_feature_column_order_match_independent_scalar_calculation():
    _, query, _, _, memory, fields = _fixture()
    result = build_priors(memory, query, fields,
                          config={'retrieval': {'top_k': 4, 'temperature': 1., 'batch_size': 1}})
    indices, valid = result['audit']['indices'], result['audit']['valid']
    n, k = indices.shape
    dc, dd = np.zeros((n, k)), np.zeros((n, k, 24))
    centered_curves, level_difference = np.zeros((n, k, 24)), np.zeros((n, k))
    event_columns = [0, 1, 2, 3, 4, 6, 7, 8]
    for row in range(n):
        hist = query.inputs['history_values'][row]
        query_level = np.median(hist[-24:, 0])
        for slot in range(k):
            if not valid[row, slot]:
                continue
            index = indices[row, slot]
            old_hist = memory.inputs['history_values'][index]
            old_level = np.median(old_hist[-24:, 0])
            dc[row, slot] = np.mean(((hist[:, 0] - query_level) -
                                     (old_hist[:, 0] - old_level)) ** 2)
            dc[row, slot] += .5 * np.mean((hist[-24:, 1] - old_hist[-24:, 1]) ** 2)
            dc[row, slot] += .5 * np.mean((query.inputs['future_calendar'][row, :, :8] -
                                           memory.inputs['future_calendar'][index, :, :8]) ** 2)
            for hour in range(24):
                dd[row, slot, hour] = np.mean((query.inputs['future_exogenous'][row, hour, event_columns] -
                                              memory.inputs['future_exogenous'][index, hour, event_columns]) ** 2)
            centered_curves[row, slot] = (memory.curves[index] - memory.levels[index]) / 20
            level_difference[row, slot] = (memory.levels[index] - (query_level * 20 + 50)) / 20

    def expected_weights(distances):
        output = np.zeros_like(distances)
        for row in range(n):
            available = np.flatnonzero(valid[row]).tolist()
            ordered = sorted(available, key=lambda candidate: (distances[row, candidate], candidate))[:4]
            exponentials = np.exp(-distances[row, ordered] + min(distances[row, ordered]))
            output[row, ordered] = exponentials / exponentials.sum()
        return output

    ws = expected_weights(dc + dd.mean(axis=2))
    wc = expected_weights(dc)
    wd = np.stack([expected_weights(dd[..., hour]) for hour in range(24)], axis=-1)
    for key, expected in [('w_shared', ws), ('w_background', wc), ('w_event', wd)]:
        np.testing.assert_allclose(result['audit'][key], expected, atol=2e-15, rtol=2e-14)
    assert not np.allclose(ws, wc)
    assert not np.allclose(wd[..., 0], wd[..., 1])
    expected_features = np.concatenate((centered_curves.reshape(n, -1), level_difference,
                                        valid, dc, dd.mean(2), ws), axis=1).astype(np.float32)
    np.testing.assert_allclose(result['raw_features'], expected_features, atol=1e-7, rtol=1e-7)


def test_unavailable_padded_row_nan_values_are_masked_before_priors_and_features():
    _, query, _, _, memory, fields = _fixture()
    query.origin_unix = np.array([memory.origins[0], memory.origins[0] + 3600])
    query.target_time_unix = query.origin_unix[:, None] + np.arange(24) * 3600
    reference = build_priors(memory, query, fields)
    # Padded -1 indices internally use safe row 0; its unknown content must not
    # contaminate the level fallback through zero times NaN arithmetic.
    memory.curves[0] = np.nan
    memory.levels[0] = np.nan
    for value in memory.inputs.values():
        value[0] = np.nan
    fields['background'][0] = np.nan
    fields['event'][0] = np.nan
    _assert_same(reference, build_priors(memory, query, fields))
