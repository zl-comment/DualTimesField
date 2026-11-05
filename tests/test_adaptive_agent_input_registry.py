import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from forecasting.api_agent.workflow import Snapshot
from forecasting.adaptive_agent.contracts import ForecastArtifact, ForecastContext
from forecasting.adaptive_agent.input_operations import OPERATIONS
from forecasting.adaptive_agent.input_registry import InputForecastRegistry, load_operator_admission


def context():
    target = np.arange(24, dtype=np.int64) * 3600 + 100000
    snapshot = Snapshot('NSW1', 100000, target, np.ones(72), np.ones(72),
                        np.ones((72, 10)), np.ones((24, 10)), 'preprocessing')
    return ForecastContext(snapshot, {}, {'region': 'NSW1'})


class Adapter:
    admitted_operation_ids = ('identity', 'origin_gas_mean')
    identity = {'model': 'one-frozen-lightgbm'}

    def __init__(self):
        self.calls = []

    def predict(self, ctx, operation_id):
        self.calls.append(operation_id)
        value = 1 if operation_id == 'identity' else 2
        return ForecastArtifact(
            'lightgbm-input:' + operation_id, ctx.identity(),
            np.full((24, 5), value, dtype=float), ctx.snapshot.target_time_unix,
            'lightgbm', 'raw-' + operation_id,
            {'input_operation': {'operation': {'operation_id': operation_id}}})


class RegistryTests(unittest.TestCase):
    def test_same_model_identity_and_distinct_operation_candidates(self):
        adapter = Adapter()
        registry = InputForecastRegistry(adapter, 'r21', max_current_candidates=2)
        self.assertEqual(set(registry.adapters), {'lightgbm'})
        self.assertEqual(registry.default_id, 'lightgbm-input:identity')
        first = registry.forecast(context(), registry.default_id)
        second = registry.forecast(context(), 'lightgbm-input:origin_gas_mean')
        self.assertEqual(first.model_id, second.model_id)
        self.assertNotEqual(first.input_hash, second.input_hash)
        self.assertNotEqual(first.identity(), second.identity())
        self.assertEqual(adapter.calls, ['identity', 'origin_gas_mean'])

    def test_cache_and_budget_are_per_context_candidate(self):
        adapter = Adapter()
        registry = InputForecastRegistry(adapter, 'r21', max_current_candidates=1)
        first = registry.forecast(context(), registry.default_id)
        self.assertIs(first, registry.forecast(context(), registry.default_id))
        self.assertEqual(registry.stats, {'logical_requests': 2, 'numeric_inferences': 1, 'cache_hits': 1})
        with self.assertRaises(ValueError):
            registry.forecast(context(), 'lightgbm-input:origin_gas_mean')

    def test_admission_loader_reproduces_fixed_design_ranking(self):
        rows = {}
        for key in ('identity', 'origin_gas_mean'):
            rows[key] = {'spec': json.loads(json.dumps(OPERATIONS[key].__dict__)), 'admission': {
                'passes_rule_before_top_four_cap': key != 'identity',
                'design_conditional_oracle_qs_gain': 1.0 if key != 'identity' else 0.0}}
        record = {
            'status': 'ADMITTED', 'split': 'calibration', 'seed': 2026,
            'admitted_operation_ids': ['identity', 'origin_gas_mean'],
            'selection_rule_fixed_before_scores': {
                'validation': 'reported after selection and never changes admission'},
            'semantics': 'training mean', 'source': {'commit': 'abc'}, 'operations': rows}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'admission.json'
            path.write_text(json.dumps(record))
            loaded = load_operator_admission(path, seed=2026)
            self.assertEqual(loaded['admitted_operation_ids'], ('identity', 'origin_gas_mean'))
            record['operations']['origin_gas_mean']['spec']['hypothesis'] = 'changed'
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                load_operator_admission(path, seed=2026)


if __name__ == '__main__':
    unittest.main()
