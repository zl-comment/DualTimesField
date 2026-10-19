import numpy as np
import pytest

from forecasting.full_v5.conditional_event_memory import build_transfer_memory, FEATURE_NAMES
from forecasting.full_v5.history_correction import DAY, HISTORY_START


class Encoder:
    def __init__(self, center_offset=0., amplitude=600.):
        self.calls = []
        self.offset, self.amplitude = center_offset, amplitude

    def encode_matured(self, origins, actual, z, lags, *, as_of, fixed=False):
        assert np.all(origins >= HISTORY_START)
        assert np.all(origins + DAY <= as_of)
        self.calls.append((origins.copy(), actual.copy(), as_of, fixed))
        peak = actual.argmax(axis=1)
        return dict(indices=np.arange(len(origins)), trend=np.full_like(actual, 100.),
                    centers=peak[:, None] + self.offset, widths=np.full((len(origins), 1), .5),
                    amplitude=np.full((len(origins), 1), self.amplitude))


def arrays(days=13, peak=3, target=17):
    origins = HISTORY_START + np.arange(days) * DAY
    y = np.full((days, 24), 100.)
    y[:, peak] = 500.
    q = np.broadcast_to(np.array([60., 80., 100., 120., 140.]), (days, 24, 5)).copy()
    z = np.zeros((days, 24, 16))
    z[:, peak, 1:10] = 1.
    z[-1, :, 1:10] = 0.
    z[-1, target, 1:10] = 1.
    lags = np.full((days, 24, 3), 100.)
    return origins, y, q, z, lags


def test_event_follows_known_query_state_not_unknown_price_and_timing_control_is_clean():
    o, y, q, z, lags = arrays()
    y[-1] = np.nan
    shifted = build_transfer_memory(o, y, q, z, lags, Encoder())
    unshifted = build_transfer_memory(o, y, q, z, lags, Encoder(), shift=False)
    assert shifted['event'][-1].argmax() == 17
    assert unshifted['event'][-1].argmax() == 3
    np.testing.assert_array_equal(shifted['event_record_indices'], unshifted['event_record_indices'])
    np.testing.assert_array_equal(shifted['event_weights'], unshifted['event_weights'])
    np.testing.assert_array_equal(shifted['event_proposed_target_h'], unshifted['event_proposed_target_h'])
    assert np.all(shifted['event_target_h'][-1] == 17)
    assert np.all(unshifted['event_target_h'][-1] == 3)
    y[-1] = np.arange(24) * 1e7
    changed = build_transfer_memory(o, y, q, z, lags, Encoder())
    np.testing.assert_array_equal(changed['features'], shifted['features'])
    assert shifted['feature_names'] == list(FEATURE_NAMES)
    assert shifted['features'].shape == (len(o), 24, 29)
    assert np.abs(shifted['features']).max() <= 1


def test_future_label_perturbation_does_not_change_earlier_inputs_or_outputs():
    o, y, q, z, lags = arrays(25)
    spy = Encoder()
    baseline = build_transfer_memory(o, y, q, z, lags, spy)
    cutoff = o[12]
    y[o + DAY > cutoff] = 1e7
    changed = build_transfer_memory(o, y, q, z, lags, Encoder())
    for key in ('trend', 'event', 'features', 'event_record_indices', 'trend_ids'):
        np.testing.assert_array_equal(baseline[key][:13], changed[key][:13])
    assert all(np.all(oo + DAY <= t) for oo, yy, t, fixed in spy.calls)
    assert baseline['audit']['maturity_violations'] == 0
    assert baseline['audit']['minimum_release_margin_seconds'] == 0


def test_minimum_pool_maturity_and_same_hour_nonoverlapping_records():
    o, y, q, z, lags = arrays(12)
    # Interleave hourly shifted windows; same-hour selection must exclude them.
    oo = np.sort(np.concatenate((o, o + 3600)))
    yy, qq, zz, ll = (np.repeat(a, 2, axis=0) for a in (y, q, z, lags))
    result = build_transfer_memory(oo, yy, qq, zz, ll, Encoder())
    assert np.all(result['event'][:16] == 0)
    assert np.any(result['event'][16] > 0)
    for row, t in enumerate(oo):
        for ids in (result['trend_ids'][row].ravel(), result['event_record_indices'][row]):
            ids = ids[ids >= 0]
            assert np.all(oo[ids] + DAY <= t)
            assert np.all(oo[ids] % DAY == t % DAY)
        ids = result['event_record_indices'][row]
        ids = ids[ids >= 0]
        assert len(ids) == len(np.unique(ids))
        assert np.all(np.diff(np.sort(oo[ids])) >= DAY)


def test_no_event_fallback_and_no_double_count_already_predicted_spike():
    o, y, q, z, lags = arrays()
    y[:] = 100
    result = build_transfer_memory(o, y, q, z, lags, Encoder())
    assert np.all(result['event'] == 0)
    assert np.all(result['event_record_indices'] == -1)
    assert np.isfinite(result['features']).all()
    o, y, q, z, lags = arrays()
    q[:, 3] += 500
    result = build_transfer_memory(o, y, q, z, lags, Encoder())
    assert np.all(result['event'] == 0)


def test_missed_amplitude_cap_and_no_wrap_at_target_boundary():
    o, y, q, z, lags = arrays(target=0)
    q[:, 3] += 350  # actual500 - q50=450 leaves only50 unmet
    result = build_transfer_memory(o, y, q, z, lags, Encoder())
    np.testing.assert_allclose(result['event_amplitudes'][-1], 50)
    np.testing.assert_allclose(result['event'][-1, 0], 50)
    assert result['event'][-1, -1] < 1e-100
    assert np.all((result['event'] >= 0) & (result['event'] <= 150))
    assert np.all(np.abs(result['trend']) <= 150)
    # A center shifted slightly outside the window is truncated, never wrapped.
    outside = build_transfer_memory(o, y, q, z, lags, Encoder(center_offset=-.4))
    assert np.all(outside['event_centers'][-1] < 0)
    assert outside['event'][-1, -1] < 1e-100


def test_pre_library_excluded_fixed_geometry_forwarded_and_release_edge_exact():
    o, y, q, z, lags = arrays(13)
    o = o - 2 * DAY
    spy = Encoder()
    result = build_transfer_memory(o, y, q, z, lags, spy, fixed=True)
    assert result['encoded_at'][0] == result['encoded_at'][1] == -1
    assert all(fixed for oo, yy, t, fixed in spy.calls)
    assert all(np.all(oo >= HISTORY_START) for oo, yy, t, fixed in spy.calls)
    o, y, q, z, lags = arrays(3)
    o[:] = [HISTORY_START, HISTORY_START + DAY - 1, HISTORY_START + DAY]
    edge = build_transfer_memory(o, y, q, z, lags, Encoder())
    np.testing.assert_array_equal(np.flatnonzero(edge['encoded_at'] >= 0), [0])
    assert edge['encoded_at'][0] == o[-1]


def test_nonfinite_matured_label_rejected_but_unknown_query_labels_allowed():
    o, y, q, z, lags = arrays()
    y[-1] = np.nan
    build_transfer_memory(o, y, q, z, lags, Encoder())
    y[0] = np.nan
    with pytest.raises(ValueError, match='matured'):
        build_transfer_memory(o, y, q, z, lags, Encoder())


def test_trend_is_weighted_historical_field_minus_smoothed_anchor():
    o, y, q, z, lags = arrays()
    q -= 20
    result = build_transfer_memory(o, y, q, z, lags, Encoder())
    np.testing.assert_allclose(result['trend'][8:], 20)
    assert np.all(result['trend'][:8] == 0)


def test_geometry_changes_payload_not_candidate_identity_and_trend_is_not_double_counted():
    o, y, q, z, lags = arrays()
    class GeometryControl(Encoder):
        def encode_matured(self, *args, fixed=False, **kwargs):
            out = super().encode_matured(*args, fixed=fixed, **kwargs)
            out['trend'][:] = 250.  # +150 trend absorbs part of missed400
            if fixed:
                out['amplitude'][:] = -10.  # no qualifying positive atom
            return out
    adaptive = build_transfer_memory(o, y, q, z, lags, GeometryControl())
    fixed = build_transfer_memory(o, y, q, z, lags, GeometryControl(), fixed=True)
    for key in ('event_record_indices', 'event_weights', 'event_distances',
                'event_target_h', 'event_source_h', 'trend_ids'):
        np.testing.assert_array_equal(adaptive[key], fixed[key])
    assert np.any(adaptive['event'] > 0)
    assert np.all(fixed['event'] == 0)
    np.testing.assert_allclose(adaptive['event_amplitudes'][-1], 250.)
    assert fixed['audit']['historical_event_records'] > 0
    assert fixed['audit']['positive_payload_event_records'] == 0


def test_offset_narrow_atom_cannot_exceed_remaining_miss_after_transport():
    o, y, q, z, lags = arrays(target=17)
    q[:, 3] += 350  # raw miss50, trend100 below smoothed anchor => remaining50
    class NarrowOffset(Encoder):
        def encode_matured(self, *args, **kwargs):
            fields = super().encode_matured(*args, **kwargs)
            fields['widths'][:] = .25
            return fields
    for shift in (False, True):
        result = build_transfer_memory(o, y, q, z, lags,
                                       NarrowOffset(center_offset=1., amplitude=1000.), shift=shift)
        np.testing.assert_allclose(result['event_amplitudes'][-1], 50.)
        assert np.max(result['event'][-1]) <= 50. + 1e-12
        assert result['event'][-1].argmax() == (18 if shift else 4)
