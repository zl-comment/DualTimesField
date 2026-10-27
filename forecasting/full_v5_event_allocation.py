"""Allocate a frozen model's already saturated event increment to its upper tail.

The inputs must be Full and event_off from the SAME checkpoint, query batch,
baseline and selected scalar alpha. In JointHistoryModel.forward, event_off
zeros only the gated event correction, after computing the unchanged gates,
weights and trend. Thus, before finite-precision rounding,

    Q_T = Anchor + 150*tanh(T/150)
    e = 150*(tanh((T+E)/150) - tanh(T/150)),  E >= 0
    Q_F = Q_T + e,                         0 <= e <= 300.

The selected outputs use the original common blend B+alpha*(Q-B), so their
event increment is alpha*e. Copying q05/q10/q50 from selected Q_T and q90/q95
from selected Q_F retains this saturation and alpha without a new forward or
any outcome-dependent routing. It does not replace e by E or by tanh(E/150).
Array checks cannot establish checkpoint identity; the runner must verify the
saved metadata, selection record and checkpoint/source hashes separately.
"""
from __future__ import annotations

import math

import numpy as np


ROUNDING_RULE = (
    'IEEE round-to-nearest: Full and event_off each have one final FP32 '
    'Anchor-plus-correction addition. A common-shift contrast therefore uses '
    'gamma(1,float32) times the absolute raw Full/event_off magnitudes at '
    'both compared quantiles, plus two FP32 smallest subnormals. At alpha '
    'in (0,1), gamma(3,float64) bounds each archived affine blend; '
    'gamma(3,float64) also bounds the computed contrast. Stored FP64 outputs '
    'retain the original FP32 bound. Exact Full>=event_off is required; '
    'alpha=0 requires both arrays to equal B exactly. No empirical tolerance.'
)


def _quantiles(values, name):
    result = np.asarray(values, dtype=np.float64)
    if not result.size or result.ndim < 1 or result.shape[-1] != 5:
        raise ValueError(f'{name}: expected a nonempty five-quantile array')
    if not np.isfinite(result).all():
        raise ValueError(f'{name}: nonfinite quantiles')
    if np.any(result[..., 1:] < result[..., :-1]):
        raise ValueError(f'{name}: crossed quantiles')
    return result


def _pair(full, trend):
    f, t = _quantiles(full, 'Full'), _quantiles(trend, 'same-head event_off')
    if f.shape != t.shape:
        raise ValueError('Full and same-head event_off grids differ')
    # Positive E, monotone tanh, a common Anchor and nonnegative common alpha
    # preserve this ordering even under the original round-to-nearest steps.
    if np.any(f < t):
        raise AssertionError('Full-event_off contains a negative event increment')
    return f, t


def event_upper_only(full, trend):
    """Return FP64 [Trend05, Trend10, Trend50, Full90, Full95] exactly.

    ``trend`` means the same frozen model's event_off prediction, not another
    fitted Trend_only model. This operation consumes no labels, does no sorting
    or clipping, and leaves both inputs unchanged. Since Trend50 <= Trend90
    <= Full90, valid nonnegative event pairs cannot cross at the copied seam.
    Use audit_event_pair as well to check the common-shift/model premise.
    """
    f, t = _pair(full, trend)
    result = t.copy()
    result[..., 3:] = f[..., 3:]
    _quantiles(result, 'event-upper-only output')
    if not np.array_equal(result[..., :3], t[..., :3]):
        raise AssertionError('The copied same-head event_off lower quantiles changed')
    if not np.array_equal(result[..., 3:], f[..., 3:]):
        raise AssertionError('The copied Full upper quantiles changed')
    return result


def _gamma(operations, dtype):
    unit_roundoff = np.finfo(dtype).eps / 2
    return operations*unit_roundoff/(1-operations*unit_roundoff)


def audit_event_pair(baseline, full, trend, alpha):
    """Check a selected Full/event_off pair with a predeclared rounding bound.

    Let raw F_j=fl32(A_j+H_F) and raw T_j=fl32(A_j+H_T). The contrast
    (F_j-T_j)-(F_50-T_50) has zero mathematical value. Its raw rounding bound
    is gamma(1,32)*(|F_j|+|T_j|+|F_50|+|T_50|), plus two smallest FP32
    subnormals. Both final additions count; this is larger than the bound for
    a Full-minus-Anchor contrast, which contains only one rounded addition.

    For a saved blend X=fl64(beta*B+alpha*raw X), let g=gamma(3,64),
    P_X=(|X|+(1+g)*beta*|B|)/(1-g) and R_X=g*(beta*|B|+P_X).
    P_X bounds alpha*|raw X| without dividing by alpha; R_X bounds blend
    error. Add the four R_X terms and the FP64 contrast-rounding bound. The
    per-quantile event range [0,300*alpha] has the analogous two-value bound
    at its upper endpoint. Nonnegativity itself is exact, not tolerance based.

    Passing supports the saved common positive-shift premise, not a claim
    that these arrays independently prove the internal T/E decomposition.
    The copied output is exact on the saved arrays; no equality to the bits
    of a hypothetical newly implemented FP32 forward is claimed.
    """
    b = _quantiles(baseline, 'baseline')
    f, t = _pair(full, trend)
    if b.shape != f.shape:
        raise ValueError('Baseline and event prediction grids differ')
    alpha_value = np.asarray(alpha)
    if alpha_value.ndim != 0:
        raise ValueError('One frozen scalar alpha is required')
    alpha = float(alpha_value)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Alpha must be finite and in [0,1]')
    if alpha == 0:
        if not np.array_equal(f, b) or not np.array_equal(t, b):
            raise AssertionError('alpha=0 requires Full and event_off to equal B exactly')
        if not np.array_equal(event_upper_only(f, t), b):
            raise AssertionError('The exact alpha=0 baseline fallback changed')
        return dict(alpha=alpha, passed=True, common_shift_passed=True,
            nonnegative_passed=True, event_bound_passed=True, alpha_zero_exact=True,
            common_shift_max_abs_error=0., rounding_bound_max=0.,
            max_error_to_bound_ratio=0., event_increment_min=0.,
            event_increment_max=0., median_event_increment_mean=0.,
            positive_event_fraction=0., theoretical_event_upper_bound=0.,
            event_range_rounding_bound_max=0., rounding_rule=ROUNDING_RULE)

    g32, g64 = _gamma(1, np.float32), _gamma(3, np.float64)
    subnormal32 = float(np.nextafter(np.float32(0), np.float32(1)))
    with np.errstate(over='raise', invalid='raise'):
        if alpha == 1:
            p_f, p_t = np.abs(f), np.abs(t)
            r_f, r_t = np.zeros_like(f), np.zeros_like(t)
        else:
            base_magnitude = (1-alpha)*np.abs(b)
            p_f = (np.abs(f)+(1+g64)*base_magnitude)/(1-g64)
            p_t = (np.abs(t)+(1+g64)*base_magnitude)/(1-g64)
            r_f, r_t = g64*(base_magnitude+p_f), g64*(base_magnitude+p_t)
        increment = f-t
        error = np.abs(increment-increment[..., 2:3])
        bound = g32*(p_f+p_t+p_f[..., 2:3]+p_t[..., 2:3])
        bound += r_f+r_t+r_f[..., 2:3]+r_t[..., 2:3]
        bound += g64*(np.abs(f)+np.abs(t)+np.abs(f[..., 2:3])+np.abs(t[..., 2:3]))
        bound += 2*alpha*subnormal32
        range_bound = g32*(p_f+p_t)+r_f+r_t
        range_bound += g64*(np.abs(f)+np.abs(t)+300*alpha)+alpha*subnormal32
    if not np.isfinite(bound).all() or not np.isfinite(range_bound).all():
        raise AssertionError('Nonfinite event rounding bound')
    common_passed = bool(np.all(error <= bound))
    range_passed = bool(np.all(increment <= 300*alpha+range_bound))
    ratio = np.divide(error, bound, out=np.zeros_like(error), where=bound > 0)
    result = dict(alpha=alpha, passed=common_passed and range_passed,
        common_shift_passed=common_passed, nonnegative_passed=True,
        event_bound_passed=range_passed, alpha_zero_exact=None,
        common_shift_max_abs_error=float(error.max()), rounding_bound_max=float(bound.max()),
        max_error_to_bound_ratio=float(ratio.max()), event_increment_min=float(increment.min()),
        event_increment_max=float(increment.max()),
        median_event_increment_mean=float(increment[..., 2].mean()),
        positive_event_fraction=float(np.mean(increment[..., 2] > 0)),
        theoretical_event_upper_bound=300*alpha,
        event_range_rounding_bound_max=float(range_bound.max()), rounding_rule=ROUNDING_RULE)
    if not common_passed:
        raise AssertionError(f'Full-event_off is not a common shift within the predefined rounding bound: {result}')
    if not range_passed:
        raise AssertionError(f'Event increment exceeds the fixed saturated range and rounding bound: {result}')
    return result
