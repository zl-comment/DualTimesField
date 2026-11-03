"""No-network replay persistence, maturity, integrity, and failure tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from forecasting.api_agent.workflow import Snapshot
from forecasting.adaptive_agent.contracts import ForecastContext, BudgetLimits
from forecasting.adaptive_agent.registry import ForecastRegistry
from forecasting.adaptive_agent.evidence import EvidenceStore
from forecasting.adaptive_agent.replay import (validate_manifest, run_replay, stable_service_identity,
                                               aggregate_scores, main, preserve_run_git_provenance)


class Repository:
    def __init__(self, region='NSW1'):
        self.region = region
        self.origins = 1_700_000_000 + np.arange(3) * 86400
        self.outcome_reads = []
        self.evaluation_reads = []

    def context(self, index):
        origin = int(self.origins[index])
        return ForecastContext(Snapshot(self.region, origin, origin + np.arange(24) * 3600,
                                        np.arange(72.), np.ones(72), np.ones((72, 10)),
                                        np.ones((24, 10)), 'test-preprocess'))

    def mature_actual(self, index, cutoff):
        if int(self.origins[index]) + 86400 > cutoff:
            raise AssertionError('Future outcome was requested')
        self.outcome_reads.append((index, cutoff))
        return np.full(24, 40.)

    def episodes(self, context):
        return []

    def evaluation_actual(self, indices):
        self.evaluation_reads.append(list(indices))
        return np.full((len(indices), 24), 40.)


class Adapter:
    identity = {'sha256': 'fixed-test'}
    def predict(self, context, profile_id, window_hours):
        return np.tile(np.array([20., 30., 40., 50., 60.]), (24, 1))


class Host:
    def __init__(self, registry, evidence, analogs, feedback, **kwargs):
        self.registry, self.feedback = registry, feedback
        self.audit = []
        self.saved = {}
    def artifact(self, context, candidate_id):
        return self.saved[candidate_id]


def runner(context, host, client, limits):
    artifact = host.registry.forecast(context, host.registry.default_id)
    host.saved[artifact.candidate_id] = artifact
    for row in host.feedback.snapshot(context):
        if row['outcome_available_at'] > context.cutoff:
            raise AssertionError('Future feedback')
    trace = {'selected_forecast_id': artifact.identity(), 'selected_candidate_id': artifact.candidate_id,
             'costs': {'api_calls': 1}, 'tools': [], 'fallback': None}
    return artifact, trace


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / 'run'
        self.manifest = {'configuration': {'origins_per_region': '3'}, 'release': 'test'}
        self.repo = Repository()
        self.registries = {'NSW1': ForecastRegistry({'chronos': Adapter()}, 'test')}
    def tearDown(self):
        self.temp.cleanup()
    def replay(self, agent_runner=runner):
        return run_replay(self.output, self.manifest, {'NSW1': self.repo}, self.registries,
                          None, EvidenceStore(), BudgetLimits(), agent_runner=agent_runner,
                          host_factory=Host)

    def test_complete_replay_scores_after_commit_and_resume_does_not_rerun(self):
        validate_manifest(self.output, self.manifest)
        report = self.replay()
        self.assertEqual(report['origins'], 3)
        self.assertEqual(report['coverage_fraction'], 1)
        self.assertEqual(report['regions']['NSW1']['BASE']['mae'], 0)
        self.assertEqual(self.repo.evaluation_reads, [[0, 1, 2]])
        self.assertEqual([x[0] for x in self.repo.outcome_reads], [0, 1])
        validate_manifest(self.output, self.manifest, resume=True)
        def forbidden(*args):
            raise AssertionError('Committed origin reran')
        self.assertEqual(self.replay(forbidden)['regions'], report['regions'])

    def test_crash_before_second_commit_restarts_without_evaluation_leak(self):
        validate_manifest(self.output, self.manifest)
        def interrupted(context, host, client, limits):
            if context.origin_unix == int(self.repo.origins[1]):
                raise RuntimeError('Injected crash')
            return runner(context, host, client, limits)
        with self.assertRaises(RuntimeError):
            self.replay(interrupted)
        self.assertEqual(self.repo.evaluation_reads, [])
        self.assertFalse((self.output / 'predictions_complete.json').exists())
        self.assertEqual(len(list((self.output / 'origins').iterdir())), 1)
        validate_manifest(self.output, self.manifest, resume=True)
        report = self.replay()
        self.assertEqual(report['origins'], 3)

    def test_manifest_changes_and_existing_path_are_refused(self):
        validate_manifest(self.output, self.manifest)
        with self.assertRaises(ValueError):
            validate_manifest(self.output, self.manifest)
        with self.assertRaises(ValueError):
            validate_manifest(self.output, {**self.manifest, 'release': 'other'}, resume=True)

    def test_resume_preserves_original_git_but_still_rejects_source_change(self):
        original = {**self.manifest, 'git': {'base_commit': 'original'}, 'source': {'sha256': 'frozen'}}
        validate_manifest(self.output, original)
        current = {**original, 'git': {'base_commit': 'docs-only-commit'}}
        restored = preserve_run_git_provenance(self.output, current)
        self.assertEqual(restored, original)
        validate_manifest(self.output, restored, resume=True)
        self.assertEqual(current['git']['base_commit'], 'docs-only-commit')
        with self.assertRaisesRegex(ValueError, 'outside Git'):
            preserve_run_git_provenance(self.output, {**current, 'source': {'sha256': 'changed'}})

    def test_crash_after_bundle_commit_repairs_missing_ledger_forecast(self):
        validate_manifest(self.output, self.manifest)
        with patch('forecasting.adaptive_agent.memory.FeedbackLedger.append_forecast',
                   side_effect=RuntimeError('Crash after publication')):
            with self.assertRaises(RuntimeError):
                self.replay()
        self.assertEqual(len(list((self.output / 'origins').iterdir())), 1)
        self.assertEqual(self.repo.evaluation_reads, [])
        self.assertEqual(self.replay()['origins'], 3)

    def test_schedule_is_global_chronological_and_feedback_stays_in_region(self):
        validate_manifest(self.output, self.manifest)
        repositories = {'NSW1': self.repo, 'QLD1': Repository('QLD1')}
        registries = {region: ForecastRegistry({'chronos': Adapter()}, 'test') for region in repositories}
        visited = []
        def observe(context, host, client, limits):
            visited.append((context.origin_unix, context.region))
            self.assertTrue(all(row['region'] == context.region for row in host.feedback.snapshot(context)))
            return runner(context, host, client, limits)
        result = run_replay(self.output, self.manifest, repositories, registries, None, EvidenceStore(),
                            BudgetLimits(), agent_runner=observe, host_factory=Host)
        self.assertEqual(visited, sorted(visited))
        self.assertEqual(result['origins'], 6)

    def test_tampered_trace_refused_before_evaluation(self):
        validate_manifest(self.output, self.manifest)
        self.replay()
        first = sorted((self.output / 'origins').iterdir())[0]
        (first / 'trace.json').write_text('{}')
        self.repo.evaluation_reads.clear()
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.replay()
        self.assertEqual(self.repo.evaluation_reads, [])

    def test_nonprefix_results_fail_closed(self):
        validate_manifest(self.output, self.manifest)
        self.replay()
        import shutil
        shutil.rmtree(sorted((self.output / 'origins').iterdir())[0])
        with self.assertRaisesRegex(ValueError, 'prefix'):
            self.replay()

    def test_service_counts_are_ignored_but_configuration_changes_are_not(self):
        one = {'models': [{'name': 'kev', 'dtype': 'bf16', 'prefix_cache': {'size': 2, 'hits': 1},
                           'batches': {'count': 1}}]}
        two = json.loads(json.dumps(one))
        two['models'][0]['prefix_cache']['hits'] = 200
        two['models'][0]['batches']['count'] = 800
        self.assertEqual(stable_service_identity(one), stable_service_identity(two))
        two['models'][0]['dtype'] = 'fp32'
        self.assertNotEqual(stable_service_identity(one), stable_service_identity(two))

    def test_cli_without_execute_is_plan_only(self):
        self.assertEqual(main(['--output', str(self.output)]), 0)
        self.assertFalse(self.output.exists())

    def test_scoring_uses_five_quantiles_and_explicit_axes(self):
        result = aggregate_scores(np.zeros((2, 24)), np.zeros((2, 24, 5)))
        self.assertEqual(result['qs'], 0)
        self.assertEqual(result['coverage90'], 1)
        with self.assertRaises(ValueError):
            aggregate_scores(np.zeros((2, 24)), np.zeros((2, 24, 4)))

    def test_evaluation_failure_invalidates_stale_success_report(self):
        validate_manifest(self.output, self.manifest)
        self.replay()
        self.repo.evaluation_actual = lambda indices: (_ for _ in ()).throw(ValueError('outcomes corrupted'))
        with self.assertRaisesRegex(ValueError, 'outcomes corrupted'):
            self.replay()
        self.assertEqual(json.loads((self.output / 'progress.json').read_text())['status'], 'EVALUATION_FAILED')
        report = json.loads((self.output / 'report.json').read_text())
        self.assertFalse(report['metrics_valid'])
        self.assertNotIn('regions', report)

    def test_scores_match_shared_main_table_scorer_on_nontrivial_crossed_quantiles(self):
        from forecasting.aligned_scoring import make_prediction_artifact, score_predictions
        rng = np.random.default_rng(17)
        actual = rng.normal(40, 30, (2, 24))
        quantiles = rng.normal(30, 40, (2, 24, 5))
        origins = np.array([1_700_000_000, 1_700_086_400])
        meta = dict(protocol_id='test', split='selection', region='NSW1', seed=2026,
                    source='r18', input_hash='input', sample_hash='sample',
                    preprocessing_hash='pre', prediction_space='raw_rrp', output_type='quantile')
        reference = score_predictions(make_prediction_artifact(
            actual=actual, origin_unix=origins, target_time_unix=origins[:, None] + 3600 * np.arange(24),
            metadata=meta, quantiles=quantiles, quantile_levels=[.05, .1, .5, .9, .95]))
        result = aggregate_scores(actual, quantiles)
        for own, shared in [('qs', 'quantile_score_proxy'), ('mae', 'mae'), ('rmse', 'rmse_pooled'),
                            ('coverage90', 'coverage_90'), ('width90', 'interval_width_90')]:
            self.assertEqual(result[own], reference[shared])


if __name__ == '__main__':
    unittest.main()
