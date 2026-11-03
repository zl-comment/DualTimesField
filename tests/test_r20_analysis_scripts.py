"""Post-run R20 analysis validates real trace schemas without reading new labels."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_r20_replay import audit_origin
from forecasting.adaptive_agent.contracts import BudgetLimits
from forecasting.adaptive_agent.iterative_policy import run
from forecasting.adaptive_agent.replay import save_bundle
from test_adaptive_agent_iterative_policy import iterative_host, REVISION
from test_adaptive_agent_runtime import context, ALTERNATIVE, DEFAULT


class RecordedChoices:
    def __init__(self, choices):
        self.choices = iter(choices)
        self.timeout = 30
    def choose(self, state, instructions, criteria):
        choice = next(self.choices)
        return choice, {'request': {'model': 'kev-latest', 'state': state,
            'questions': {'action': {'type': 'choice', 'instructions': instructions, 'criteria': criteria}}},
            'elapsed_seconds': .01, 'selected_probability': 1.}


class R20AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        (self.output / 'origins').mkdir()
        self.manifest = {'configuration': {'policy': 'r20_iterative_input'}, 'budgets': asdict(BudgetLimits())}
    def tearDown(self):
        self.temp.cleanup()
    def row(self, failure=False):
        host = iterative_host()
        if failure:
            original = host.execute
            def broken(ctx, action, state):
                if action.startswith('forecast:'):
                    raise RuntimeError('Injected no-result model failure')
                return original(ctx, action, state)
            host.execute = broken
        agent, trace = run(context(), host, RecordedChoices([
            'retrieve_analogs:state', 'forecast:' + ALTERNATIVE,
            'forecast:' + REVISION, 'commit:' + REVISION]))
        base = host.artifact(context(), DEFAULT)
        directory = save_bundle(self.output, context(), 0, base, agent, trace,
            {'tool_audit': host.audit}, 'test-manifest', materialized=host.artifacts)
        return {'record': json.loads((directory / 'record.json').read_text()),
                'trace': trace, 'directory': directory,
                'arrays': {'base': base.quantiles, 'agent': agent.quantiles,
                           'target_time_unix': agent.target_time_unix}}

    def test_real_iterative_trace_passes_with_cache_reuse_and_selected_array(self):
        result = audit_origin(self.manifest, self.row())
        self.assertEqual(result['violations'], [])
        self.assertEqual(result['revisions'], 1)
        self.assertEqual(result['verified_same_cohort_revisions'], 1)
        self.assertEqual(result['retrospective_logical_units'], 9)
        self.assertEqual(result['retrospective_numeric_inferences'], 9)
        self.assertEqual(result['cache_hits'], 6)
        self.assertEqual(result['cache_misses'], 9)
        self.assertEqual(result['materialized_candidate_count'], 3)
        self.assertLessEqual(result['max_api_request_bytes'], 24000)

    def test_revision_without_following_same_cohort_diagnostic_is_rejected(self):
        row = self.row()
        position = next(i for i, t in enumerate(row['trace']['tools']) if t['action'] == 'forecast:' + REVISION)
        row['trace']['tools'][position+1]['result']['observation']['cohort_id'] = 'different-cohort'
        result = audit_origin(self.manifest, row)
        self.assertIn('revision_requires_same_cohort_diagnosis_or_default_fallback',
                      [value['rule'] for value in result['violations']])

    def test_oversized_request_and_budget_violation_are_reported(self):
        row = self.row()
        row['trace']['calls'][0]['record']['request']['questions']['action']['instructions'] = 'x' * 25000
        row['trace']['costs']['api_calls'] = 100
        result = audit_origin(self.manifest, row)
        rules = {value['rule'] for value in result['violations']}
        self.assertIn('api_request_bytes', rules)
        self.assertIn('budget:api_calls', rules)

    def test_unknown_tool_failure_cost_is_a_bound_not_observed_inference(self):
        result = audit_origin(self.manifest, self.row(failure=True))
        self.assertEqual(result['violations'], [])
        self.assertEqual(result['unknown_actual_cost_events'], 1)
        self.assertEqual(result['estimated_upper_bound_costs']['candidates'], 1)
        self.assertEqual(result.get('retrospective_numeric_inferences', 0), 0)
        self.assertIsNotNone(result['fallback'])


if __name__ == '__main__':
    unittest.main()
