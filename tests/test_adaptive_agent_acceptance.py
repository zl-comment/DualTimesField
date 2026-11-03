"""R18 T022/T026/T027 acceptance against the existing shared scorer.

Tests only: no frozen implementation, skill or model configuration is changed.
"""
from types import SimpleNamespace
import unittest

import numpy as np

from forecasting.api_agent.workflow import Snapshot
from forecasting.aligned_scoring import make_prediction_artifact, score_predictions
from forecasting.adaptive_agent.contracts import ForecastContext
from forecasting.adaptive_agent.diagnostics import evaluate_episodes
from forecasting.adaptive_agent.pooling import pool_quantiles
from forecasting.adaptive_agent.registry import ForecastRegistry
from forecasting.adaptive_agent.review import validate_artifact


def context(origin):
    return ForecastContext(Snapshot(
        'NSW1', origin, origin + np.arange(24) * 3600, np.arange(72.) * 1.5,
        np.linspace(-1, 1, 72), np.ones((72, 10)), np.ones((24, 10)), 'acceptance-preprocessing'))


class CrossedAdapter:
    identity = {'sha256': 'synthetic-frozen', 'train_end_unix': 100, 'selection_end_unix': 200}

    def __init__(self):
        self.calls = 0

    @staticmethod
    def raw(ctx, window):
        # Deliberately crossed quantiles, negative prices, and changing horizons.
        offset = (ctx.origin_unix // 86400) % 11
        return (np.arange(24.)[:, None] * 2.3 + np.array([50., -20., 12., 35., 0.])[None]
                + window / 3 + offset)

    def predict(self, ctx, profile, window):
        self.calls += 1
        return self.raw(ctx, window)


def shared_score(contexts, actual, raw_quantiles):
    origins = np.array([ctx.origin_unix for ctx in contexts])
    return score_predictions(make_prediction_artifact(
        actual=np.asarray(actual), origin_unix=origins,
        target_time_unix=np.stack([ctx.snapshot.target_time_unix for ctx in contexts]),
        quantiles=np.asarray(raw_quantiles), quantile_levels=[.05, .1, .5, .9, .95],
        metadata=dict(protocol_id='acceptance', split='selection', region='NSW1', seed=2026,
                      source='r18-acceptance', input_hash='fixture', sample_hash='fixture',
                      preprocessing_hash='acceptance-preprocessing', output_type='quantile',
                      prediction_space='raw_rrp')))


class R18AcceptanceTests(unittest.TestCase):
    def test_T022_historical_metrics_match_shared_scorer_and_charge_budget(self):
        adapter = CrossedAdapter()
        registry = ForecastRegistry({'chronos': adapter}, 'acceptance')
        current = context(1_710_000_000)
        past = [context(current.origin_unix - days * 86400) for days in (3, 2)]
        actual = [np.linspace(-50, 150, 24), np.linspace(120, -40, 24)]
        episodes = [{'episode_id': f'ep-{i}', 'context': ctx, 'actual': y,
                     'outcome_available_at': ctx.origin_unix + 86400, 'data_role': 'holdout'}
                    for i, (ctx, y) in enumerate(zip(past, actual))]
        ids = ['chronos:price_only:24', 'chronos:price_only:72']
        diagnostic = evaluate_episodes(current, ids, episodes, registry, max_units=4)
        self.assertEqual(diagnostic['logical_inference_units'], 4)
        self.assertEqual(diagnostic['numeric_inferences'], 4)
        self.assertEqual(adapter.calls, 4)
        self.assertEqual(diagnostic['failures'], [])
        mapping = {'qs': 'quantile_score_proxy', 'mae': 'mae', 'coverage90': 'coverage_90',
                   'width90': 'interval_width_90'}
        for candidate in ids:
            window = int(candidate.rsplit(':', 1)[1])
            reference = shared_score(past, actual, [adapter.raw(ctx, window) for ctx in past])
            result = diagnostic['candidates'][candidate]
            self.assertEqual(result['sample_count'], 2)
            self.assertGreater(result['metrics']['qs'], 0)
            for diagnostic_key, main_key in mapping.items():
                self.assertAlmostEqual(result['metrics'][diagnostic_key], reference[main_key], places=12)
            for row in (r for r in diagnostic['rows'] if r['candidate_id'] == candidate):
                i = int(row['episode_id'].split('-')[1])
                expected = shared_score([past[i]], [actual[i]], [adapter.raw(past[i], window)])
                self.assertAlmostEqual(row['metrics']['qs'], expected['quantile_score_proxy'], places=12)
                self.assertEqual(row['diagnostic_kind'], 'retrospective_frozen_recompute')
        with self.assertRaisesRegex(ValueError, 'budget'):
            evaluate_episodes(current, ids, episodes, registry, max_units=3)
        self.assertEqual(adapter.calls, 4, 'Budget rejection must happen before numerical inference')

    def test_T026_review_rejects_forged_nan_axis_context_and_missing_source(self):
        ctx = context(1_710_000_000)
        artifact = ForecastRegistry({'chronos': CrossedAdapter()}, 'acceptance').forecast(ctx, 'chronos:base:72')
        baseline = {name: getattr(artifact, name) for name in
                    ('candidate_id', 'context_id', 'model_id', 'input_hash', 'quantiles', 'target_time_unix')}
        invalid = []
        q = artifact.quantiles.copy()
        q[7, 2] = np.nan
        invalid.append(('nan', {'quantiles': q}))
        invalid.append(('axis', {'target_time_unix': artifact.target_time_unix + 3600}))
        invalid.append(('context', {'context_id': context(ctx.origin_unix - 86400).identity()}))
        invalid.append(('source', {'model_id': ''}))
        for name, changes in invalid:
            with self.subTest(case=name), self.assertRaises(ValueError):
                # Bypass dataclass construction to verify the final submit guard,
                # not just the earlier object constructor's schema validation.
                validate_artifact(ctx, SimpleNamespace(**{**baseline, **changes}))
        self.assertIs(validate_artifact(ctx, artifact), artifact)
        self.assertTrue(np.isnan(q[7, 2]), 'Submission guard must not repair invalid values')

    def test_T027_registry_rearrangement_and_pool_follow_public_scoring_rules(self):
        ctx = context(1_710_000_000)
        adapter = CrossedAdapter()
        registry = ForecastRegistry({'chronos': adapter}, 'acceptance')
        a = registry.forecast(ctx, 'chronos:price_only:24')
        b = registry.forecast(ctx, 'chronos:price_only:72')
        self.assertTrue((np.diff(adapter.raw(ctx, 24), axis=1) < 0).any())
        np.testing.assert_array_equal(a.quantiles, np.sort(adapter.raw(ctx, 24), axis=-1))
        actual = np.linspace(-30, 130, 24)
        for weight in (.25, .5, .75):
            pooled = pool_quantiles(a, b, weight)
            expected = weight * np.sort(adapter.raw(ctx, 24), axis=-1) + (1-weight) * np.sort(adapter.raw(ctx, 72), axis=-1)
            np.testing.assert_allclose(pooled.quantiles, expected, rtol=0, atol=1e-12)
            self.assertFalse(pooled.provenance['exact_mixture_quantiles'])
            self.assertIs(validate_artifact(ctx, pooled), pooled)
            expected_scores = shared_score([ctx], [actual], [expected])
            actual_scores = shared_score([ctx], [actual], [pooled.quantiles])
            self.assertEqual(actual_scores['quantile_score_proxy'], expected_scores['quantile_score_proxy'])
        with self.assertRaises(ValueError):
            pool_quantiles(a, b, .4)


if __name__ == '__main__':
    unittest.main()
