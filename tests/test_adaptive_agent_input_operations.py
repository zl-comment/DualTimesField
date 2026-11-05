import unittest

import numpy as np

from forecasting.adaptive_agent.input_operations import (
    EXPECTED_SHAPES, OPERATIONS, apply_operation, validate_input_blocks,
)
from forecasting.adaptive_agent.predictors.lightgbm import FrozenLightGBMAdapter


def inputs():
    return {name: np.ones(shape, dtype=np.float32) for name, shape in EXPECTED_SHAPES.items()}


class InputOperationTests(unittest.TestCase):
    def test_identity_is_exact_and_copies(self):
        source = inputs()
        transformed, provenance = apply_operation(source, 'identity')
        self.assertEqual(provenance['source_input_sha256'], provenance['transformed_input_sha256'])
        for name in source:
            self.assertTrue(np.array_equal(source[name], transformed[name]))
            self.assertIsNot(source[name], transformed[name])

    def test_every_intervention_changes_only_registered_slice(self):
        source = inputs()
        for operation_id, operation in OPERATIONS.items():
            if operation_id == 'identity':
                continue
            with self.subTest(operation_id=operation_id):
                transformed, provenance = apply_operation(source, operation_id)
                self.assertNotEqual(provenance['source_input_sha256'], provenance['transformed_input_sha256'])
                expected = inputs()
                values = expected[operation.block]
                if values.ndim == 1:
                    values[operation.column_start:operation.column_stop] = 0
                else:
                    values[operation.row_start:operation.row_stop,
                           operation.column_start:operation.column_stop] = 0
                for name in expected:
                    self.assertTrue(np.array_equal(expected[name], transformed[name]))

    def test_source_is_immutable(self):
        source = inputs()
        before = {name: value.copy() for name, value in source.items()}
        apply_operation(source, 'history_all_last24_mean')
        for name in source:
            self.assertTrue(np.array_equal(source[name], before[name]))

    def test_reject_unknown_nonfinite_or_wrong_shape(self):
        with self.assertRaises(ValueError):
            apply_operation(inputs(), 'unknown')
        bad = inputs(); bad['history_values'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_input_blocks(bad)

    def test_frozen_matrix_predictor_rejects_wrong_schema_before_boosters(self):
        adapter = object.__new__(FrozenLightGBMAdapter)
        adapter.anchor = type('Anchor', (), {'input_shapes': dict(EXPECTED_SHAPES)})()
        with self.assertRaises(ValueError):
            adapter.predict_matrix(np.ones((2, 10)))
        bad = inputs(); bad['history_values'] = bad['history_values'][-24:]
        with self.assertRaises(ValueError):
            validate_input_blocks(bad)


if __name__ == '__main__':
    unittest.main()
