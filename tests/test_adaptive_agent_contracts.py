"""Fail closed on ambiguous temporal and resource contracts."""
import unittest
import numpy as np
from forecasting.api_agent.workflow import Snapshot
from forecasting.adaptive_agent.contracts import ForecastContext, BudgetLimits


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = Snapshot('NSW1', 1700000000, 1700000000 + 3600 * np.arange(24),
                                 np.ones(72), np.ones(72), np.ones((72, 10)),
                                 np.ones((24, 10)), 'fixture')

    def test_timestamps_cannot_be_silently_rounded(self):
        for value in (1700000000.5, '1700000000', True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ForecastContext(self.snapshot, cutoff=value)
        with self.assertRaises(ValueError):
            ForecastContext(self.snapshot, memory_watermark=1700000001)

    def test_count_budgets_cannot_be_fractional(self):
        for value in (1.5, True, '8', float('nan'), -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BudgetLimits(api_calls=value)
        self.assertEqual(BudgetLimits(origin_deadline_seconds=.5).origin_deadline_seconds, .5)

    def test_context_copies_and_freezes_nested_inputs(self):
        inputs, meta = {'history': np.ones(3)}, {'nested': {'version': 1}}
        context = ForecastContext(self.snapshot, inputs, meta)
        identity = context.identity()
        inputs['history'][0], meta['nested']['version'] = 99, 99
        self.assertEqual(identity, context.identity())
        with self.assertRaises(ValueError):
            context.model_inputs['history'][0] = 3
        with self.assertRaises(TypeError):
            context.metadata['nested']['version'] = 2


if __name__ == '__main__':
    unittest.main()
