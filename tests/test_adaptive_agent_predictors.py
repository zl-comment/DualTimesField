"""Prediction contracts, effective input interventions, and causal diagnostics."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from forecasting.api_agent.workflow import Snapshot
from forecasting.adaptive_agent.contracts import ForecastContext
from forecasting.adaptive_agent.registry import ForecastRegistry
from forecasting.adaptive_agent.pooling import pool_quantiles
from forecasting.adaptive_agent.diagnostics import evaluate_episodes, compare_candidates, evaluate_recent_history
from forecasting.adaptive_agent.predictors.chronos import FrozenChronosAdapter, make_input


def context(origin=1_700_000_000, shift=0, watermark=None, model_inputs=None, metadata=None):
    snapshot = Snapshot('NSW1', origin, origin + 3600 * np.arange(24),
                        np.arange(72.) + shift, np.arange(72.) / 10,
                        np.arange(720.).reshape(72, 10), np.ones((24, 10)), 'test-preprocessing')
    return ForecastContext(snapshot, model_inputs or {}, metadata or {},
                           memory_watermark=watermark)


class FakeAdapter:
    identity = {'sha256': 'test-model', 'train_end_unix': 100, 'selection_end_unix': 200}

    def __init__(self):
        self.calls = []

    def predict(self, ctx, profile_id, window_hours):
        self.calls.append((ctx.identity(), profile_id, window_hours))
        return np.tile(np.arange(5.), (24, 1)) + ctx.snapshot.price[-window_hours:].mean()


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.adapter = FakeAdapter()
        self.registry = ForecastRegistry({'chronos': self.adapter}, 'test-release')

    def test_on_demand_cache_binds_inputs_models_and_release(self):
        first = self.registry.forecast(context(), 'chronos:price_only:24')
        again = self.registry.forecast(context(), 'chronos:price_only:24')
        self.assertIs(first, again)
        changed = self.registry.forecast(context(shift=100), 'chronos:price_only:24')
        self.assertNotEqual(changed.input_hash, first.input_hash)
        self.assertGreater(changed.quantiles.mean(), first.quantiles.mean())
        other = ForecastRegistry({'chronos': FakeAdapter()}, 'other-release')
        self.assertNotEqual(other.forecast(context(), 'chronos:price_only:24').input_hash, first.input_hash)
        self.assertEqual(len(self.adapter.calls), 2)
        with self.assertRaises(ValueError):
            first.quantiles[0, 0] = 1

    def test_cap_is_per_context_and_unknown_model_rejected(self):
        ids = list(self.registry.specs)
        for candidate in ids[:5]:
            self.registry.forecast(context(), candidate)
        with self.assertRaises(ValueError):
            self.registry.forecast(context(), ids[5])
        self.registry.forecast(context(origin=1_700_100_000), ids[5])
        with self.assertRaises(ValueError):
            self.registry.forecast(context(), 'chronos:arbitrary:13')

    def test_adapter_identity_cannot_change_after_registration(self):
        self.adapter.identity = {'sha256': 'mutated'}
        with self.assertRaises(ValueError):
            self.registry.forecast(context(), 'chronos:base:72')

    def test_chronos_windows_are_suffixes_and_profiles_change_covariates(self):
        ctx = context()
        for window in (24, 48, 72):
            row = make_input(ctx, 'price_only', window)
            np.testing.assert_array_equal(row['target'], ctx.snapshot.price[-window:])
            self.assertEqual(set(row), {'target'})
        base = make_input(ctx, 'base', 24)
        cal = make_input(ctx, 'calendar', 24)
        self.assertIn('demand_std', base['past_covariates'])
        self.assertNotIn('demand_std', cal['past_covariates'])
        np.testing.assert_array_equal(base['future_covariates']['cal_0'], ctx.snapshot.calendar_future[:, 0])
        with self.assertRaises(ValueError):
            make_input(ctx, 'price_only', 96)

    def test_pipeline_receives_window_and_isolated_cross_learning(self):
        pipeline = Mock()
        def predict(rows, **kwargs):
            value = rows[0]['target'].mean()
            return [np.ones((1, 24, 5)) * value], None
        pipeline.predict_quantiles.side_effect = predict
        adapter = FrozenChronosAdapter(pipeline, {'sha256': 'verified'})
        short = adapter.predict(context(), 'price_only', 24)
        long = adapter.predict(context(), 'price_only', 72)
        self.assertFalse(np.array_equal(short, long))
        self.assertFalse(pipeline.predict_quantiles.call_args.kwargs['cross_learning'])
        self.assertEqual(pipeline.predict_quantiles.call_args.kwargs['prediction_length'], 24)

    def test_quantile_pool_changes_output_and_rejects_cross_context(self):
        a = self.registry.forecast(context(), 'chronos:price_only:24')
        b = self.registry.forecast(context(), 'chronos:price_only:72')
        result = pool_quantiles(a, b, .25)
        np.testing.assert_allclose(result.quantiles, .25 * a.quantiles + .75 * b.quantiles)
        self.assertFalse(result.provenance['exact_mixture_quantiles'])
        with self.assertRaises(ValueError):
            pool_quantiles(a, b, .3)
        future = self.registry.forecast(context(origin=1_700_100_000), 'chronos:price_only:72')
        with self.assertRaises(ValueError):
            pool_quantiles(a, future)

    def episode(self, role='holdout'):
        historical = context(origin=1_699_000_000)
        return {'episode_id': 'past', 'context': historical, 'actual': np.ones(24) * 50,
                'outcome_available_at': historical.origin_unix + 86400, 'data_role': role}

    def test_historical_diagnostics_use_complete_mature_contexts_and_tag_overlap(self):
        result = evaluate_episodes(context(), ['chronos:base:72'], [self.episode()], self.registry)
        row = result['candidates']['chronos:base:72']
        self.assertEqual(row['sample_count'], 1)
        self.assertEqual(row['holdout_sample_count'], 1)
        self.assertEqual(result['rows'][0]['diagnostic_kind'], 'retrospective_frozen_recompute')
        result = evaluate_episodes(context(), ['chronos:base:72'], [self.episode('selection')], self.registry)
        self.assertEqual(result['candidates']['chronos:base:72']['holdout_sample_count'], 0)
        self.assertIn('model_selection_exposed', result['candidates']['chronos:base:72']['data_roles'])

    def test_unmature_outcomes_and_watermark_are_rejected_before_inference(self):
        ep = self.episode()
        ep['outcome_available_at'] = context().cutoff + 1
        with self.assertRaises(ValueError):
            evaluate_episodes(context(), ['chronos:base:72'], [ep], self.registry)
        ep = self.episode()
        with self.assertRaises(ValueError):
            evaluate_episodes(context(watermark=ep['outcome_available_at'] - 1),
                              ['chronos:base:72'], [ep], self.registry)
        self.assertEqual(len(self.adapter.calls), 0)

    def test_unknown_model_overlap_is_not_unbiased(self):
        adapter = FakeAdapter()
        adapter.identity = {'sha256': 'opaque-pretraining'}
        reg = ForecastRegistry({'chronos': adapter}, 'test')
        result = evaluate_episodes(context(), ['chronos:base:72'], [self.episode()], reg)
        row = result['candidates']['chronos:base:72']
        self.assertEqual(row['holdout_sample_count'], 0)
        self.assertEqual(row['data_roles'], ['training_or_selection_overlap_unknown'])

    def test_diagnostic_budget_and_duplicates_rejected_before_inference(self):
        with self.assertRaises(ValueError):
            evaluate_episodes(context(), ['chronos:base:72'], [self.episode()], self.registry, max_units=0)
        with self.assertRaises(ValueError):
            evaluate_episodes(context(), ['chronos:base:72'], [self.episode(), self.episode()], self.registry)
        self.assertEqual(len(self.adapter.calls), 0)

    def test_disagreement_has_no_fake_accuracy(self):
        predictions = [self.registry.forecast(context(), f'chronos:price_only:{w}') for w in (24, 72)]
        report = compare_candidates(predictions)
        self.assertFalse(report['future_actuals_used'])
        self.assertEqual(report['sample_count'], 0)
        self.assertGreater(report['pairs'][0]['median_absolute_difference'], 0)

    def test_recent_hindcast_uses_only_preceding_observed_context(self):
        pipeline = Mock()
        pipeline.predict_quantiles.return_value = ([np.ones((1, 24, 5))], None)
        registry = ForecastRegistry({'chronos': FrozenChronosAdapter(pipeline, {'sha256': 'test'})}, 'test')
        result = evaluate_recent_history(context(), ['chronos:base:72', 'chronos:price_only:24'], registry)
        first, second = pipeline.predict_quantiles.call_args_list
        np.testing.assert_array_equal(first.args[0][0]['target'], context().snapshot.price[:48])
        np.testing.assert_array_equal(second.args[0][0]['target'], context().snapshot.price[24:48])
        np.testing.assert_array_equal(first.args[0][0]['future_covariates']['cal_0'],
                                      context().snapshot.calendar_past[48:, 0])
        self.assertEqual(result['logical_inference_units'], 2)
        self.assertTrue(result['candidates']['chronos:base:72']['context_length_differs'])
        self.assertFalse(result['candidates']['chronos:price_only:24']['context_length_differs'])
        with self.assertRaises(ValueError):
            evaluate_recent_history(context(), ['chronos:base:72'], registry, max_units=0)

    def test_lightgbm_adapter_preserves_exact_flatten_and_target_transform(self):
        from forecasting.adaptive_agent.predictors.lightgbm import FrozenLightGBMAdapter
        from forecasting.lightgbm.predictor import _LEGACY_SHAPES
        from forecasting.aligned_inputs import INPUT_KEYS
        inputs = {key: np.full(shape, i, dtype=np.float64)
                  for i, (key, shape) in enumerate(_LEGACY_SHAPES.items())}
        metadata = {'region': 'NSW1', 'preprocessing_hash': 'test-preprocessing',
                    'protocol_id': 'test-protocol', 'target_mean': 10., 'target_std': 2.,
                    'input_block_order': list(INPUT_KEYS)}
        ctx = context(model_inputs=inputs, metadata=metadata)
        adapter = FrozenLightGBMAdapter.__new__(FrozenLightGBMAdapter)
        adapter.anchor = SimpleNamespace(input_shapes=_LEGACY_SHAPES,
                                        manifest={'train_metadata': metadata}, target_mean=10.,
                                        target_std=2., horizon=24, selected_rounds=[3] * 24, threads=1)
        captured = []
        def predict(matrix, **kwargs):
            captured.append(matrix.copy())
            return np.array([2.])
        adapter._boosters = {(h, q): Mock(predict=predict) for h in range(24) for q in range(5)}
        np.testing.assert_array_equal(adapter.predict(ctx), np.full((24, 5), 14.))
        self.assertEqual(captured[0].shape, (1, 1465))
        expected = np.concatenate([inputs[key].ravel() for key in INPUT_KEYS])
        np.testing.assert_array_equal(captured[0][0], expected)
        with self.assertRaises(ValueError):
            adapter.predict(ctx, 'price_only', 72)


if __name__ == '__main__':
    unittest.main()
