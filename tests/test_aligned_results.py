import numpy as np

from forecasting.analysis.aligned_results import masks, slice_scores, macro_seed_stats
from forecasting.aligned_report import REGIONS, SEEDS


def test_price_slices_partition_boundaries_and_reconstruct_mae():
    actual = np.array([[-1., 0., 100.], [101., 300., 301.]])
    median = actual + np.array([[2., -3., 4.], [-5., 6., -7.]])
    quantiles = median[..., None] + np.array([-10., -5., 0., 5., 10.])
    # Repeated timestamps represent overlapping origins, not independent events.
    times = np.array([[0, 1, 2], [1, 2, 3]])
    parts = masks(actual)
    np.testing.assert_array_equal(sum(parts.values()), np.ones_like(actual))
    scores = [slice_scores(actual, quantiles, mask, times) for mask in parts.values()]
    assert [s['n_cells'] for s in scores] == [1, 2, 2, 1]
    np.testing.assert_allclose(sum(s['share'] * s['mae'] for s in scores), 4.5)
    all_scores = slice_scores(actual, quantiles, np.ones_like(actual, dtype=bool), times)
    assert all_scores['n_unique_delivery_hours'] == 4
    np.testing.assert_allclose(sum(s['share'] * s['quantile_score_proxy'] for s in scores),
                               all_scores['quantile_score_proxy'])


def test_seed_dispersion_is_not_region_or_prediction_dispersion():
    regions = {region: {'scores': {'per_seed': {str(seed): {'mae': offset + index}
                                               for index, seed in enumerate(SEEDS)}}}
               for region, offset in zip(REGIONS, [0, 100, 200])}
    mean, sd = macro_seed_stats(regions, 'mae')
    assert mean == 101
    assert sd == 1
