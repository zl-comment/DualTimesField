"""Audited, shape-preserving input interventions for the frozen LightGBM.

The cached numerical inputs are already transformed with training-only
standardizers.  Replacing a standardized value by zero therefore asks the
frozen tree model for a counterfactual at the training normalization mean.  It
is an explicit intervention, not native missing-value handling or column
deletion.  Calendar values are deliberately excluded because zero is not a
jointly valid neutral calendar state for sine/cosine pairs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np

from .contracts import digest


EXPECTED_SHAPES = MappingProxyType({
    'history_values': (72, 2),
    'history_calendar': (72, 10),
    'future_calendar': (24, 10),
    'origin_context': (1,),
    'future_exogenous': (24, 11),
    'ctf_exogenous': (24, 1),
    'quantile_exogenous': (24, 3),
})


@dataclass(frozen=True)
class InputOperation:
    operation_id: str
    block: str | None
    row_start: int | None
    row_stop: int | None
    column_start: int | None
    column_stop: int | None
    replacement: str
    hypothesis: str

    def __post_init__(self):
        if not self.operation_id or self.replacement not in ('identity', 'training_normalization_mean'):
            raise ValueError('Invalid input operation identity or replacement')
        if self.replacement == 'identity':
            if any(value is not None for value in (
                    self.block, self.row_start, self.row_stop,
                    self.column_start, self.column_stop)):
                raise ValueError('Identity operation cannot select input cells')
            return
        if self.block not in EXPECTED_SHAPES or self.block in ('history_calendar', 'future_calendar'):
            raise ValueError('Operation must target a registered standardized non-calendar block')
        shape = EXPECTED_SHAPES[self.block]
        rows = shape[0] if len(shape) == 2 else 1
        columns = shape[1] if len(shape) == 2 else shape[0]
        if not (isinstance(self.row_start, int) and isinstance(self.row_stop, int)
                and 0 <= self.row_start < self.row_stop <= rows
                and isinstance(self.column_start, int) and isinstance(self.column_stop, int)
                and 0 <= self.column_start < self.column_stop <= columns):
            raise ValueError('Input operation slice is outside the fixed schema')

    def identity(self):
        return digest(asdict(self))


def _op(operation_id, block, rows, columns, hypothesis):
    return InputOperation(operation_id, block, rows[0], rows[1], columns[0], columns[1],
                          'training_normalization_mean', hypothesis)


OPERATIONS = MappingProxyType({item.operation_id: item for item in (
    InputOperation('identity', None, None, None, None, None, 'identity',
                   'Use the complete canonical input unchanged.'),
    _op('history_all_last48_mean', 'history_values', (0, 24), (0, 2),
        'Test whether the oldest day of price and demand history adds harmful regime information.'),
    _op('history_all_last24_mean', 'history_values', (0, 48), (0, 2),
        'Test whether only the most recent day of price and demand history should remain active.'),
    _op('history_price_last48_mean', 'history_values', (0, 24), (0, 1),
        'Test whether the oldest day of price history adds harmful regime information.'),
    _op('history_price_last24_mean', 'history_values', (0, 48), (0, 1),
        'Test whether only the most recent day of price history should remain active.'),
    _op('history_demand_last48_mean', 'history_values', (0, 24), (1, 2),
        'Test whether the oldest day of realized demand history adds harmful information.'),
    _op('history_demand_last24_mean', 'history_values', (0, 48), (1, 2),
        'Test whether only the most recent day of realized demand history should remain active.'),
    _op('origin_gas_mean', 'origin_context', (0, 1), (0, 1),
        'Test whether the trailing gas-price context is unhelpful in the current regime.'),
    _op('future_capacity_mean', 'future_exogenous', (0, 24), (0, 1),
        'Test whether the available max-spare-capacity forecast is unhelpful.'),
    _op('future_predispatch_mean', 'future_exogenous', (0, 24), (1, 6),
        'Test whether AEMO predispatch price, demand, generation, interchange and coverage are unhelpful.'),
    _op('future_interconnector_mean', 'future_exogenous', (0, 24), (6, 11),
        'Test whether interconnector forecast features are unhelpful.'),
    _op('future_netload_mean', 'ctf_exogenous', (0, 24), (0, 1),
        'Test whether the PD-PASA net-load forecast is unhelpful.'),
    _op('future_scarcity_mean', 'quantile_exogenous', (0, 24), (0, 3),
        'Test whether max-spare-capacity scarcity shortfall features are unhelpful.'),
)})


def validate_input_blocks(inputs: Mapping[str, np.ndarray]) -> None:
    if set(inputs) != set(EXPECTED_SHAPES):
        raise ValueError('LightGBM input blocks differ from the fixed seven-block contract')
    for name, shape in EXPECTED_SHAPES.items():
        values = np.asarray(inputs[name])
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f'Invalid LightGBM input block {name}: {values.shape}')


def apply_operation(inputs: Mapping[str, np.ndarray], operation_id: str):
    """Return independent blocks and exact intervention provenance.

    A zero is meaningful only because every admitted target block is standardized
    using training-only statistics.  This routine never edits the source context.
    """
    validate_input_blocks(inputs)
    if operation_id not in OPERATIONS:
        raise ValueError('Unregistered LightGBM input operation: ' + str(operation_id))
    operation = OPERATIONS[operation_id]
    transformed = {name: np.array(value, copy=True) for name, value in inputs.items()}
    changed = 0
    if operation.replacement != 'identity':
        values = transformed[operation.block]
        if values.ndim == 1:
            selected = values[operation.column_start:operation.column_stop]
        else:
            selected = values[operation.row_start:operation.row_stop,
                              operation.column_start:operation.column_stop]
        changed = int(np.count_nonzero(selected))
        selected[...] = 0.0
    validate_input_blocks(transformed)
    provenance = {
        'operation': asdict(operation),
        'operation_sha256': operation.identity(),
        'semantics': ('canonical_identity' if operation_id == 'identity' else
                      'counterfactual_training_mean_neutralization_not_missingness'),
        'changed_nonzero_cells': changed,
        'source_input_sha256': digest(inputs),
        'transformed_input_sha256': digest(transformed),
    }
    if operation_id == 'identity' and provenance['source_input_sha256'] != provenance['transformed_input_sha256']:
        raise AssertionError('Identity operation changed the numerical input')
    # A registered intervention can legitimately be a no-op for one origin when
    # every selected standardized cell already equals the training mean.  The
    # operation remains auditable through its id, slice, and both hashes.
    return transformed, provenance
