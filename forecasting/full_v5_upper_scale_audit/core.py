"""Label-free upper-scale removal and one-pass frozen-output diagnostics.

The intervention is defined on saved outputs in FP64. It is mathematically
equivalent to setting the upper Anchor scale to one while keeping location and
history fixed, including a fixed scalar alpha. A new FP32 forward may differ by
rounding; bitwise equivalence to that forward is deliberately not claimed.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

LEVELS = np.asarray([.05, .10, .50, .90, .95], dtype=np.float64)
NAMES = ('q05', 'q10', 'q50', 'q90', 'q95')
HARM_RATE_THRESHOLD = 1e-7  # Same descriptive threshold as the previous audit.
ROUNDING_RULE = (
    'IEEE round-to-nearest: one FP32 addition per raw Full quantile; '
    'at alpha in (0,1), gamma(3,float64) bounds each archived affine blend; '
    'gamma(3,float64) also bounds computing the common-shift contrast. '
    'Use saved magnitudes to upper-bound alpha*abs(raw values), without dividing '
    'by alpha; include FP32 subnormal absolute rounding. alpha=0 requires exact B.'
)


def _quantiles(values, name):
    q = np.asarray(values, dtype=np.float64)
    if not q.size or q.ndim < 1 or q.shape[-1] != 5:
        raise ValueError(f'{name}: expected nonempty five-quantile array')
    if not np.isfinite(q).all():
        raise ValueError(f'{name}: nonfinite quantiles')
    if np.any(q[..., 1:] < q[..., :-1]):
        raise ValueError(f'{name}: crossed quantiles')
    return q


def remove_upper_scale(baseline, full):
    """Keep q05/q10/q50 exactly; restore B's upper gaps around Full q50.

    q'_tau = B_tau + (Full_50 - B_50) for tau in {.90,.95}.
    No outcomes, Anchor, gate, or post hoc tuning are consumed. The zero-delta
    case returns the original B upper quantiles exactly, including tied gaps.
    """
    b, f = _quantiles(baseline, 'baseline'), _quantiles(full, 'full')
    if b.shape != f.shape:
        raise ValueError('Baseline and Full grids differ')
    q = f.copy()
    with np.errstate(over='raise', invalid='raise'):
        delta = f[..., 2] - b[..., 2]
        q[..., 3:] = b[..., 3:] + delta[..., None]
    _quantiles(q, 'upper-scale-removed output')
    if not np.array_equal(q[..., :3], f[..., :3]):
        raise AssertionError('The lower quantiles or q50 changed')
    if np.array_equal(b, f) and not np.array_equal(q, b):
        raise AssertionError('An exact baseline fallback was changed')
    return q


def _gamma(operations, dtype):
    unit_roundoff = np.finfo(dtype).eps / 2
    return operations * unit_roundoff / (1 - operations * unit_roundoff)


def common_history_audit(baseline, anchor, full, alpha):
    """Verify Full-Anchor is a common shift within a *predeclared* bound.

    Let raw Full_j = fl32(raw Anchor_j + H). Its deviation from a common H
    across quantiles j and 50 is at most
        gamma(1,32) * (|raw Full_j| + |raw Full_50|)
    plus one smallest FP32 subnormal. A saved guarded value X is an affine
    FP64 blend beta*B + alpha*raw X, beta=1-alpha. Set g=gamma(3,64). Then
        P_X = (|X| + (1+g)*beta*|B|)/(1-g)
    upper-bounds alpha*|raw X| and
        R_X = g*(beta*|B| + P_X)
    bounds the blend error. Summing R for Full/Anchor at j/50 and the contrast's
    own FP64 error gives the array bound below. At alpha=1 no blend was executed
    by the archived blend function, so P_X=|X| and R_X=0.

    The bound depends on arithmetic precision/magnitudes, never on observed
    discrepancies. Float64 stored guarded arrays still require the FP32 term.
    Passing this check supports the saved common-shift premise; it does not
    claim that remove_upper_scale bitwise reproduces a new FP32 forward.
    """
    b, a, f = (_quantiles(v, name) for v, name in
               ((baseline, 'baseline'), (anchor, 'anchor'), (full, 'full')))
    if b.shape != a.shape or b.shape != f.shape:
        raise ValueError('Baseline, Anchor and Full grids differ')
    alpha_array = np.asarray(alpha)
    if alpha_array.ndim != 0:
        raise ValueError('One frozen scalar alpha is required')
    alpha = float(alpha_array)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Alpha must be finite and in [0,1]')
    if alpha == 0:
        if not np.array_equal(a, b) or not np.array_equal(f, b):
            raise AssertionError('alpha=0 requires Anchor and Full to equal B exactly')
        if not np.array_equal(remove_upper_scale(b, f), b):
            raise AssertionError('Upper-scale removal changed the exact alpha=0 fallback')
        return dict(alpha=alpha, passed=True, within_rounding_bound=True, alpha_zero_exact=True,
                    common_shift_max_abs_error=0., rounding_bound_max=0.,
                    max_error_to_bound_ratio=0., rounding_rule=ROUNDING_RULE)
    g32, g64 = _gamma(1, np.float32), _gamma(3, np.float64)
    with np.errstate(over='raise', invalid='raise'):
        if alpha == 1:
            p_f, p_a = np.abs(f), np.abs(a)
            r_f, r_a = np.zeros_like(f), np.zeros_like(a)
        else:
            base_magnitude = (1-alpha)*np.abs(b)
            p_f = (np.abs(f)+(1+g64)*base_magnitude)/(1-g64)
            p_a = (np.abs(a)+(1+g64)*base_magnitude)/(1-g64)
            r_f, r_a = g64*(base_magnitude+p_f), g64*(base_magnitude+p_a)
        shift = f-a
        error = np.abs(shift-shift[..., 2:3])
        bound = g32*(p_f+p_f[..., 2:3])
        bound += r_f+r_a+r_f[..., 2:3]+r_a[..., 2:3]
        bound += g64*(np.abs(f)+np.abs(a)+np.abs(f[..., 2:3])+np.abs(a[..., 2:3]))
        bound += alpha*float(np.nextafter(np.float32(0), np.float32(1)))
    if not np.isfinite(bound).all():
        raise AssertionError('Nonfinite common-history rounding bound')
    ratio = np.divide(error, bound, out=np.zeros_like(error), where=bound > 0)
    result = dict(alpha=alpha, passed=bool(np.all(error <= bound)), within_rounding_bound=bool(np.all(error <= bound)),
        alpha_zero_exact=None, common_shift_max_abs_error=float(error.max()),
        rounding_bound_max=float(bound.max()), max_error_to_bound_ratio=float(ratio.max()),
        rounding_rule=ROUNDING_RULE)
    if not result['within_rounding_bound']:
        raise AssertionError(f'Full-Anchor is not a common shift within the predefined rounding bound: {result}')
    return result


def _prediction_cells(y, q):
    """Compute all loss/coverage primitives once for one prediction array."""
    error = y[..., None]-q
    pinball = np.maximum(LEVELS*error, (LEVELS-1)*error)
    median_error = error[..., 2]
    columns = dict(mae=np.abs(median_error), qs=2*pinball.mean(axis=-1),
        rmse=np.square(median_error),  # Square root only AFTER group averaging.
        bias=-median_error,
        coverage80=(y >= q[..., 1]) & (y <= q[..., 3]),
        coverage90=(y >= q[..., 0]) & (y <= q[..., 4]),
        width80=q[..., 3]-q[..., 1], width90=q[..., 4]-q[..., 0],
        under_q05=y < q[..., 0], over_q95=y > q[..., 4])
    columns.update({f'pinball_{name}': pinball[..., i] for i, name in enumerate(NAMES)})
    return columns, pinball


def compute_comparison(actual, before, after, masks=None):
    """Compute a pair's primitives once, then reduce the supplied cell masks.

    Return {'overall': flat, 'groups': {name: flat}}. Flat keys are compatible
    with the prior score_and_gain audit, plus cells/empty and upper/lower-tail
    positive loss parts. Empty groups are retained as {'cells':0,'empty':True}.
    Each tail QS is twice the mean of its TWO pinball losses, so its contribution
    to the five-quantile QS is 2/5 of that tail score (not the full tail score).
    Masks must be boolean and exactly match actual's cell grid: no implicit
    origin-to-horizon broadcasting or outcome-dependent routing is performed.
    """
    y = np.asarray(actual, dtype=np.float64)
    b, a = _quantiles(before, 'before'), _quantiles(after, 'after')
    if not y.size or b.shape != (*y.shape, 5) or a.shape != b.shape:
        raise ValueError('Expected nonempty matching actual and quantile grids')
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite actuals')
    if masks is None:
        masks = {}
    if not isinstance(masks, Mapping):
        raise ValueError('masks must be a mapping of names to boolean cell arrays')
    checked_masks = {}
    for name, mask in masks.items():
        value = np.asarray(mask)
        if not isinstance(name, str) or value.dtype != np.dtype(bool) or value.shape != y.shape:
            raise ValueError('Each named mask must be boolean and match the actual cell grid')
        checked_masks[name] = value.reshape(-1)
    with np.errstate(over='raise', invalid='raise'):
        before_cells, pb = _prediction_cells(y, b)
        after_cells, pa = _prediction_cells(y, a)
        columns = {f'before_{key}': value for key, value in before_cells.items()}
        columns.update({f'after_{key}': value for key, value in after_cells.items()})
        for name in ('mae', 'qs'):
            change = after_cells[name]-before_cells[name]
            columns.update({f'{name}_delta': change, f'{name}_benefit': np.maximum(-change, 0),
                f'{name}_harm': np.maximum(change, 0), f'{name}_harm_rate': change > HARM_RATE_THRESHOLD})
        delta_pb = pa-pb
        for i, name in enumerate(NAMES):
            change = delta_pb[..., i]
            columns.update({f'pinball_{name}_delta': change,
                f'pinball_{name}_benefit': np.maximum(-change, 0),
                f'pinball_{name}_harm': np.maximum(change, 0)})
        for name, index in (('lower', slice(None, 2)), ('upper', slice(3, None))):
            change = 2*delta_pb[..., index].mean(axis=-1)
            columns.update({f'{name}_tail_qs_delta': change,
                f'{name}_tail_qs_benefit': np.maximum(-change, 0),
                f'{name}_tail_qs_harm': np.maximum(change, 0)})
        residual, delta = y-b[..., 2], a[..., 2]-b[..., 2]
        direction = delta*residual
        columns.update(wrong_direction_rate=direction < 0,
            target_crossing_rate=(direction > 0) & (np.abs(delta) > np.abs(residual)),
            harmful_overshoot_rate=(direction > 0) & (np.abs(delta) > 2*np.abs(residual)),
            correction_from_exact_baseline_rate=(residual == 0) & (delta != 0),
            zero_correction_rate=delta == 0, mean_abs_median_delta=np.abs(delta),
            mean_abs_quantile_delta=np.abs(a-b).mean(axis=-1), mean_signed_median_delta=delta)
    # A single numeric matrix lets each group select once, rather than rerunning
    # five-quantile losses/coverage or repeatedly masking dozens of arrays.
    names = tuple(columns)
    matrix = np.column_stack([np.asarray(columns[name], dtype=np.float64).reshape(-1) for name in names])
    if not np.isfinite(matrix).all():
        raise AssertionError('Nonfinite metric primitive')
    del columns, before_cells, after_cells, pb, pa, delta_pb

    def aggregate(mask=None):
        count = len(matrix) if mask is None else int(mask.sum())
        if count == 0:
            return dict(cells=0, empty=True)
        selected = matrix if mask is None else matrix[mask]
        means = selected.mean(axis=0)
        result = {name: float(value) for name, value in zip(names, means)}
        for key in ('before_rmse', 'after_rmse'):
            result[key] = math.sqrt(result[key])
        result.update(cells=count, empty=False)
        return result

    return dict(overall=aggregate(), groups={name: aggregate(mask) for name, mask in checked_masks.items()})
