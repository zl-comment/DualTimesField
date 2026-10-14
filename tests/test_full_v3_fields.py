"""Scientific contracts for deterministic, per-curve dual-field bases."""
import copy

import numpy as np
import pytest

from forecasting.full_v3.fields import decompose_curves


def test_constants_have_only_background_and_do_not_modify_input_or_config():
    curves = np.repeat(np.array([-200., 0., 80., 15000.])[:, None], 24, axis=1)
    original = curves.copy()
    settings = {'widths': [0.75, 1.5, 3.], 'batch_size': 2}
    saved_settings = copy.deepcopy(settings)
    result = decompose_curves(curves, config=settings)
    np.testing.assert_array_equal(curves, original)
    assert settings == saved_settings
    np.testing.assert_array_equal(result['background'], curves)
    np.testing.assert_array_equal(result['event'], np.zeros_like(curves))
    np.testing.assert_array_equal(result['residual'], np.zeros_like(curves))
    np.testing.assert_array_equal(result['diagnostics']['objective'], np.zeros(len(curves)))
    np.testing.assert_array_equal(result['diagnostics']['local_scale'], np.ones(len(curves)))


def test_positive_and_negative_spikes_are_signed_events_and_reconstruct():
    curves = np.full((2, 24), 100.)
    curves[0, 12] += 500.
    curves[1, 12] -= 500.
    result = decompose_curves(curves)
    assert result['event'][0, 12] > 250
    assert result['event'][1, 12] < -250
    assert abs(result['background'][0, 12] - 100) < abs(result['event'][0, 12])
    np.testing.assert_allclose(result['event'][0], -result['event'][1], atol=1e-12)
    np.testing.assert_allclose(result['background'][0] - 100,
                               -(result['background'][1] - 100), atol=1e-12)
    np.testing.assert_allclose(sum(result[key] for key in ('background', 'event', 'residual')),
                               curves, atol=1e-12, rtol=1e-14)
    assert (result['diagnostics']['objective'] < result['diagnostics']['initial_objective']).all()


def test_translation_changes_only_the_background():
    rng = np.random.default_rng(936)
    curves = rng.normal(size=(7, 24)) * 70
    shifts = np.array([-500., -200., -10., 0., 15., 500., 10000.])[:, None]
    before = decompose_curves(curves)
    after = decompose_curves(curves + shifts)
    np.testing.assert_allclose(after['background'], before['background'] + shifts, atol=1e-10, rtol=1e-13)
    for key in ('event', 'residual'):
        np.testing.assert_allclose(after[key], before[key], atol=1e-10, rtol=1e-13)
    np.testing.assert_allclose(after['diagnostics']['objective'], before['diagnostics']['objective'],
                               atol=1e-12, rtol=1e-13)


def test_batch_boundaries_reordering_and_unrelated_curves_do_not_change_results():
    rng = np.random.default_rng(125)
    curves = rng.normal(size=(11, 24)) * 100
    reference = decompose_curves(curves, config={'batch_size': 1})
    for batch_size in (2, 7, 2048):
        other = decompose_curves(curves, config={'batch_size': batch_size})
        for key in ('background', 'event', 'residual'):
            np.testing.assert_array_equal(other[key], reference[key])
        for key in ('objective', 'background_coefficients', 'event_coefficients'):
            np.testing.assert_array_equal(other['diagnostics'][key], reference['diagnostics'][key])
    permutation = rng.permutation(len(curves))
    reordered = decompose_curves(curves[permutation])
    padded = decompose_curves(np.concatenate((curves, rng.normal(size=(3, 24)) * 1e7)))
    for key in ('background', 'event', 'residual'):
        np.testing.assert_array_equal(reordered[key], reference[key][permutation])
        np.testing.assert_array_equal(padded[key][:len(curves)], reference[key])


def test_objective_and_exact_background_update_match_the_documented_convex_problem():
    curves = np.random.default_rng(41).normal(size=(5, 24)) * 80 + 50
    result = decompose_curves(curves)
    info = result['diagnostics']
    mean, scale = info['local_mean'][:, None], info['local_scale'][:, None]
    z = (curves - mean) / scale
    c = (result['background'] - mean) / scale
    d = result['event'] / scale
    a, b = info['background_coefficients'], info['event_coefficients']
    frequency = np.arange(4)
    cosine = np.sqrt(2 / 24) * np.cos(np.pi * (np.arange(24)[:, None] + .5) * frequency / 24)
    cosine[:, 0] = 1 / np.sqrt(24)
    np.testing.assert_allclose(cosine.T @ cosine, np.eye(4), atol=1e-15)
    residual = z - c - d
    independent_objective = (.5 * np.sum(residual ** 2, axis=1)
                              + .005 * np.sum(frequency ** 4 * a ** 2, axis=1)
                              + .03 * np.sum(np.abs(b), axis=1))
    np.testing.assert_allclose(info['objective'], independent_objective, atol=1e-12)
    # The last exact ridge update must satisfy its normal equation, even when
    # the fixed number of sparse steps has not reached a joint optimum.
    np.testing.assert_allclose(residual @ cosine, .01 * frequency ** 4 * a, atol=1e-12)
    fewer = decompose_curves(curves, config={'iterations': 10})
    assert (info['objective'] <= fewer['diagnostics']['objective'] + 1e-12).all()


def test_strong_l1_removes_events_and_cosine_curvature_has_known_attenuation():
    hours = np.arange(24)
    curves = (40 + 100 * np.cos(np.pi * (hours + .5) / 24))[None, :]
    result = decompose_curves(curves, config={'l1': 1e6, 'ridge': .1})
    np.testing.assert_array_equal(result['event'], np.zeros_like(curves))
    expected = 40 + (curves - 40) / 1.1
    np.testing.assert_allclose(result['background'], expected, atol=1e-12)


@pytest.mark.parametrize('values', [np.ones((24,)), np.ones((2, 23)), np.ones((0, 24)),
                                  np.full((1, 24), np.nan), np.full((1, 24), np.inf),
                                  np.ones((1, 24), dtype=complex)])
def test_invalid_curves_are_rejected(values):
    with pytest.raises(ValueError):
        decompose_curves(values)


@pytest.mark.parametrize('config', [{'ridge': -1}, {'l1': np.nan}, {'iterations': 0},
                                  {'iterations': 1.5}, {'batch_size': True}, {'basis_count': 25},
                                  {'widths': []}, {'widths': [0]}, {'widths': [1, 1]},
                                  {'widths': [np.inf]}, {'learn_from_test': True}])
def test_invalid_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        decompose_curves(np.ones((1, 24)), config=config)
