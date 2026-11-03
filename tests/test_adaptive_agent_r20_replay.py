"""R20 replay integration and compatibility; no live API or model inference."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from forecasting.adaptive_agent.contracts import BudgetLimits
from forecasting.adaptive_agent.evidence import EvidenceStore
from forecasting.adaptive_agent.registry import ForecastRegistry
from forecasting.adaptive_agent.replay import (build_schedule, main, policy_host_factory,
    policy_module, preserve_run_git_provenance, run_replay, source_identity, validate_manifest)
from test_adaptive_agent_replay import Repository, Adapter, Host, runner


class IterativeHostFixture(Host):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.artifacts = self.saved


def iterative_runner(context, host, client, limits):
    artifact, trace = runner(context, host, client, limits)
    alternative = host.registry.forecast(context, 'chronos:price_only:24')
    host.artifacts[alternative.candidate_id] = alternative
    trace['iterations'] = [
        {'index': 0, 'candidate_ids': [artifact.candidate_id], 'incremental_historical_units': 1},
        {'index': 1, 'candidate_ids': [artifact.candidate_id, alternative.candidate_id], 'incremental_historical_units': 1}]
    trace['revisions'] = [{'action': 'forecast:' + alternative.candidate_id,
                           'new_candidate_id': alternative.candidate_id, 'execution_mode': 'llm_selected'}]
    trace['optional_evidence_actions'] = ['inspect_state']
    trace['costs'].update(tool_calls=3, candidates=2, historical_evaluations=2)
    return artifact, trace


class R20ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / 'run'
        self.manifest = {'configuration': {'origins_per_region': '3', 'policy': 'r20_iterative_input'},
                         'release': 'r20-test'}
        self.repo = Repository()
        self.registries = {'NSW1': ForecastRegistry({'chronos': Adapter()}, 'r20-test')}

    def tearDown(self):
        self.temp.cleanup()

    def replay(self, **kwargs):
        return run_replay(self.output, self.manifest, {'NSW1': self.repo}, self.registries,
                          None, EvidenceStore(), BudgetLimits(), **kwargs)

    def test_plan_only_r20_has_expected_args_and_no_output_side_effect(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = main(['--policy', 'r20_iterative_input', '--origins', '181', '--seed', '2026',
                         '--regions', 'NSW1', 'QLD1', 'TAS1', '--output', str(self.output)])
        plan = json.loads(stream.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(plan['status'], 'PLAN_ONLY')
        self.assertEqual(plan['configuration']['policy'], 'r20_iterative_input')
        self.assertEqual(plan['configuration']['origins_per_region'], '181')
        self.assertEqual(plan['configuration']['seed'], 2026)
        self.assertEqual(plan['configuration']['regions'], ['NSW1', 'QLD1', 'TAS1'])
        self.assertFalse(self.output.exists())

    def test_r20_defaults_route_to_iterative_policy_and_host_and_report_revisions(self):
        validate_manifest(self.output, self.manifest)
        def resolve(module, package):
            if module == '.iterative_policy':
                return SimpleNamespace(run=iterative_runner)
            if module == '.iterative_runtime':
                return SimpleNamespace(IterativeToolHost=IterativeHostFixture)
            raise AssertionError('Unexpected dynamic routing: ' + module)
        with patch('forecasting.adaptive_agent.replay.importlib.import_module', side_effect=resolve) as route:
            report = self.replay()
        self.assertEqual([call.args[0] for call in route.call_args_list], ['.iterative_policy', '.iterative_runtime'])
        self.assertEqual(report['iteration_count'], 6)
        self.assertEqual(report['revision_count'], 3)
        self.assertEqual(report['origins_with_revision'], 3)
        self.assertEqual(report['optional_evidence_action_count'], 3)
        self.assertEqual(len(report['origin_trace_costs']), 3)
        for row in report['origin_trace_costs']:
            trace = json.loads((self.output / row['trace_path']).read_text())
            self.assertEqual(row['costs'], trace['costs'])
            self.assertEqual(row['iteration_count'], len(trace['iterations']))
            self.assertEqual(row['selected_forecast_id'], trace['selected_forecast_id'])
        self.assertEqual(report['policy'], 'r20_iterative_input')

    def test_r20_archive_preserves_materialized_candidates_and_resumes(self):
        validate_manifest(self.output, self.manifest)
        original = self.replay(agent_runner=iterative_runner, host_factory=IterativeHostFixture)
        for directory in (self.output / 'origins').iterdir():
            record = json.loads((directory / 'record.json').read_text())
            self.assertIn('materialized_candidates.npz', record['files'])
            with np.load(directory / 'materialized_candidates.npz', allow_pickle=False) as values:
                self.assertEqual(values['quantiles'].shape, (2, 24, 5))
                self.assertEqual(set(values['candidate_ids'].tolist()), {'chronos:base:72', 'chronos:price_only:24'})
                self.assertEqual(values['forecast_ids'].shape, (2,))
        def forbidden(*args):
            raise AssertionError('Committed origin must not rerun')
        resumed = self.replay(agent_runner=forbidden, host_factory=IterativeHostFixture)
        self.assertEqual(resumed['regions'], original['regions'])
        self.assertEqual(resumed['origin_trace_costs'], original['origin_trace_costs'])
        directory = sorted((self.output / 'origins').iterdir())[0]
        with (directory / 'materialized_candidates.npz').open('ab') as handle:
            handle.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.replay(agent_runner=forbidden, host_factory=IterativeHostFixture)

    def test_r19_host_and_three_file_bundles_remain_unchanged(self):
        self.manifest['configuration']['policy'] = 'r19_evidence_first'
        validate_manifest(self.output, self.manifest)
        report = self.replay(agent_runner=runner, host_factory=Host)
        self.assertEqual(report['iteration_count'], 0)
        self.assertEqual(report['revision_count'], 0)
        for directory in (self.output / 'origins').iterdir():
            files = json.loads((directory / 'record.json').read_text())['files']
            self.assertEqual(set(files), {'predictions.npz', 'trace.json', 'audit.json'})
        sentinel = object()
        with patch('forecasting.adaptive_agent.replay.importlib.import_module',
                   return_value=SimpleNamespace(ToolHost=sentinel)) as route:
            self.assertIs(policy_host_factory('r19_evidence_first'), sentinel)
            route.assert_called_once_with('.runtime', package='forecasting.adaptive_agent')

    def test_r20_policy_and_skill_changes_refuse_resume(self):
        original = {**self.manifest, 'source': {'sha256': 'frozen'}}
        validate_manifest(self.output, original)
        changed_policy = {**original, 'configuration': {**original['configuration'], 'policy': 'r19_evidence_first'}}
        with self.assertRaises(ValueError):
            preserve_run_git_provenance(self.output, changed_policy)
        with self.assertRaises(ValueError):
            preserve_run_git_provenance(self.output, {**original, 'source': {'sha256': 'changed-skill'}})
        root = Path(self.temp.name) / 'source'
        directory = root / 'forecasting/adaptive_agent/skills_r20/plan_forecast'
        directory.mkdir(parents=True)
        skill = directory / 'instructions.txt'
        skill.write_text('r20 original')
        first = source_identity(root)
        skill.write_text('r20 revision')
        self.assertNotEqual(first['sha256'], source_identity(root)['sha256'])
        self.assertIn('forecasting/adaptive_agent/skills_r20/plan_forecast/instructions.txt', first['files'])

    def test_both_policies_share_exact_three_region_24_and_181_origin_schedule(self):
        repositories = {region: SimpleNamespace(origins=1_640_959_200 + np.arange(4321) * 3600)
                        for region in ('NSW1', 'QLD1', 'TAS1')}
        for count in ('24', '181'):
            # Policy is intentionally absent from scheduling: both experimental
            # arms receive the exact same selected integer origin indices.
            r19 = build_schedule(repositories, count)
            r20 = build_schedule(repositories, count)
            self.assertEqual(r19, r20)
            self.assertEqual(len(r20), 3 * int(count))
            self.assertEqual(r20, sorted(r20))
            for region in repositories:
                self.assertEqual(sum(row[1] == region for row in r20), int(count))
        with self.assertRaises(ValueError):
            policy_module('r20_unregistered')
        with self.assertRaises(ValueError):
            policy_host_factory('r20_unregistered')


if __name__ == '__main__':
    unittest.main()
