import json

import numpy as np
import pytest

from forecasting.aligned_scoring import (
    QUANTILE_LEVELS, aggregate_seed_scores, assert_comparable,
    load_prediction_artifact, make_prediction_artifact,
    save_prediction_artifact, score_predictions,
)


def arguments(**overrides):
    actual = np.array([[0., 4.], [2., 6.]])
    kwargs = dict(
        actual=actual,
        origin_unix=np.array([100, 200], dtype=np.int64),
        target_time_unix=np.array([[100, 110], [200, 210]], dtype=np.int64),
        metadata=dict(protocol_id='core_v1', split='validation', region='NSW1',
                      seed=2026, source='model_a', input_hash='inputs',
                      sample_hash='samples', preprocessing_hash='train_only_scaler',
                      prediction_space='raw_rrp', output_type='quantile'),
        quantiles=actual[..., None] + np.array([-3., -2., 1., 2., 3.]),
        quantile_levels=QUANTILE_LEVELS,
    )
    kwargs.update(overrides)
    return kwargs


def test_raw_contract_round_trip_preserves_crossing_and_native_head(tmp_path):
    args = arguments()
    args['quantiles'] = args['quantiles'][..., ::-1]
    args['point'] = args['actual'].copy()
    artifact = make_prediction_artifact(**args)
    path = tmp_path / 'one_seed.npz'
    save_prediction_artifact(path, artifact)
    loaded = load_prediction_artifact(path)
    np.testing.assert_array_equal(loaded.quantiles, args['quantiles'])
    assert_comparable(artifact, loaded)
    score = score_predictions(loaded)
    assert score['crossing_rate'] == 1.
    assert score['mae'] == 1.
    assert score['native_point_diagnostic']['mae'] == 0.
    assert score['point_definition'] == 'rearranged_q50'
    assert score['coverage_80'] == score['coverage_90'] == 1.
    assert score['interval_width_80'] == 4.
    assert score['interval_width_90'] == 6.
    expected_pinball = [0.15, 0.2, 0.5, 0.2, 0.15]
    assert list(score['pinball_by_quantile'].values()) == pytest.approx(expected_pinball)
    assert score['quantile_score_proxy'] == pytest.approx(2 * np.mean(expected_pinball))
    assert not any('crps' in key for key in score)
    json.dumps(score, allow_nan=False)


def test_pooled_rmse_uses_all_cells_and_q50_not_native_mean():
    args = arguments()
    args['quantiles'] = args['actual'][..., None] + np.array([-10., -9., 0., 9., 10.])
    args['quantiles'][1, :, 2] += 4
    args['point'] = args['actual']  # Perfect mean/native head cannot replace q50.
    score = score_predictions(make_prediction_artifact(**args))
    assert score['mae'] == 2.
    assert score['rmse_pooled'] == pytest.approx(np.sqrt(8))
    assert score['native_point_diagnostic']['rmse_pooled'] == 0.


@pytest.mark.parametrize('key', ['actual', 'origin_unix', 'target_time_unix'])
def test_same_metadata_cannot_hide_changed_labels_or_times(key):
    left_args, right_args = arguments(), arguments()
    right_args[key] = right_args[key] + (-1 if key == 'origin_unix' else 1)
    with pytest.raises(ValueError, match=key):
        assert_comparable(make_prediction_artifact(**left_args), make_prediction_artifact(**right_args))


@pytest.mark.parametrize('key', ['protocol_id', 'region', 'split', 'input_hash', 'sample_hash', 'preprocessing_hash'])
def test_metadata_mismatches_block_comparison(key):
    left_args, right_args = arguments(), arguments()
    right_args['metadata'][key] = 'different'
    with pytest.raises(ValueError, match=key):
        assert_comparable(make_prediction_artifact(**left_args), make_prediction_artifact(**right_args))


def test_comparison_allows_distinct_models_and_seeds():
    left_args, right_args = arguments(), arguments()
    right_args['metadata'].update(source='model_b', seed=2027)
    assert_comparable(make_prediction_artifact(**left_args), make_prediction_artifact(**right_args))


@pytest.mark.parametrize('bad', ['nan_actual', 'inf_quantile', 'nan_native', 'empty', 'shape',
                                  'target_shape', 'float_times', 'duplicate_origin', 'target_order',
                                  'past_target', 'wrong_levels', 'missing_hash', 'transformed'])
def test_invalid_predictions_rejected(bad):
    args = arguments()
    if bad == 'nan_actual':
        args['actual'][0, 0] = np.nan
    elif bad == 'inf_quantile':
        args['quantiles'][0, 0, 0] = np.inf
    elif bad == 'nan_native':
        args['point'] = np.full((2, 2), np.nan)
    elif bad == 'empty':
        args['actual'] = np.empty((0, 2))
    elif bad == 'shape':
        args['quantiles'] = np.zeros((2, 2, 4))
    elif bad == 'target_shape':
        args['target_time_unix'] = np.array([100, 200])
    elif bad == 'float_times':
        args['origin_unix'] = args['origin_unix'].astype(float)
    elif bad == 'duplicate_origin':
        args['origin_unix'][:] = 100
    elif bad == 'target_order':
        args['target_time_unix'][0] = [110, 100]
    elif bad == 'past_target':
        args['target_time_unix'][0, 0] = 99
    elif bad == 'wrong_levels':
        args['quantile_levels'] = [0.01, 0.1, 0.5, 0.9, 0.99]
    elif bad == 'missing_hash':
        del args['metadata']['preprocessing_hash']
    else:
        args['metadata']['prediction_space'] = 'standardized'
    with pytest.raises(ValueError):
        make_prediction_artifact(**args)


def test_point_only_is_separate_and_never_gets_probability_metrics():
    args = arguments()
    args.update(quantiles=None, quantile_levels=None, point=args['actual'] + 2)
    args['metadata']['output_type'] = 'point_only'
    artifact = make_prediction_artifact(**args)
    score = score_predictions(artifact)
    assert score['mae'] == score['rmse_pooled'] == 2.
    assert score['table'] == 'P'
    assert 'quantile_score_proxy' not in score
    assert 'coverage_80' not in score
    with pytest.raises(ValueError, match='separate tables'):
        assert_comparable(artifact, make_prediction_artifact(**arguments()))
    args['quantiles'] = np.repeat(args['point'][..., None], 5, axis=-1)
    with pytest.raises(ValueError, match='forbids'):
        make_prediction_artifact(**args)


def test_explicit_degenerate_distribution_allowed_but_not_fabricated_implicitly():
    args = arguments()
    args['quantiles'] = np.repeat(args['actual'][..., None], 5, axis=-1)
    with pytest.raises(ValueError, match='explicitly declared'):
        make_prediction_artifact(**args)
    args['metadata']['output_type'] = 'degenerate_quantile'
    assert score_predictions(make_prediction_artifact(**args))['quantile_score_proxy'] == 0.
    args['quantiles'][0, 0, 0] += 1
    with pytest.raises(ValueError, match='constant'):
        make_prediction_artifact(**args)


def test_missing_quantiles_not_backfilled_from_point():
    args = arguments(quantiles=None, point=np.zeros((2, 2)))
    with pytest.raises(ValueError, match='require quantiles'):
        make_prediction_artifact(**args)


def test_seed_summary_averages_losses_not_predictions():
    artifacts = []
    for seed, error in ((2026, -10), (2027, 10)):
        args = arguments()
        args['metadata']['seed'] = seed
        args['quantiles'] = args['actual'][..., None] + error + np.array([-2., -1., 0., 1., 2.])
        artifacts.append(make_prediction_artifact(**args))
    summary = aggregate_seed_scores(artifacts)
    assert summary['seed_summary']['mae'] == dict(mean=10., std=0., n=2)
    assert summary['seed_summary']['rmse_pooled']['mean'] == 10.
    assert summary['aggregation'] == 'mean_of_seed_metrics'
    assert np.abs(np.mean([a.quantiles[..., 2] for a in artifacts], axis=0) - artifacts[0].actual).mean() == 0.
    with pytest.raises(ValueError, match='distinct'):
        aggregate_seed_scores([artifacts[0], artifacts[0]])
    artifacts[1].metadata['source'] = 'other_model'
    with pytest.raises(ValueError, match='same model source'):
        aggregate_seed_scores(artifacts)


def test_legacy_archive_without_contract_is_rejected(tmp_path):
    path = tmp_path / 'legacy.npz'
    np.savez(path, actual=np.zeros((2, 2)), point=np.zeros((2, 2)))
    with pytest.raises(ValueError, match='schema'):
        load_prediction_artifact(path)
