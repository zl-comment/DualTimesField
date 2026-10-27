"""Pure FP64 diagnostics for frozen quantile predictions; no model fitting."""
import numpy as np

LEVELS = np.asarray([.05, .10, .50, .90, .95], dtype=np.float64)
NAMES = ('q05', 'q10', 'q50', 'q90', 'q95')


def validate(actual, quantiles):
    y, q = np.asarray(actual, dtype=np.float64), np.asarray(quantiles, dtype=np.float64)
    if not y.size or q.shape != (*y.shape, 5):
        raise ValueError('Expected nonempty actuals and matching five quantiles')
    if not np.isfinite(y).all() or not np.isfinite(q).all():
        raise ValueError('Nonfinite prediction or outcome')
    if np.any(np.diff(q, axis=-1) < -1e-8):
        raise ValueError('Crossed quantiles')
    return y, q


def cell_losses(actual, quantiles):
    """MAE, QS and five pinball losses, retaining all observation dimensions."""
    y, q = validate(actual, quantiles)
    error = y[..., None] - q
    pb = np.maximum(LEVELS * error, (LEVELS - 1) * error)
    return np.abs(error[..., 2]), 2 * pb.mean(axis=-1), pb


def score(actual, quantiles):
    y, q = validate(actual, quantiles)
    mae, qs, pb = cell_losses(y, q)
    result = dict(mae=float(mae.mean()), qs=float(qs.mean()),
        rmse=float(np.sqrt(np.square(y - q[..., 2]).mean())),
        bias=float((q[..., 2] - y).mean()),
        coverage80=float(((y >= q[..., 1]) & (y <= q[..., 3])).mean()),
        coverage90=float(((y >= q[..., 0]) & (y <= q[..., 4])).mean()),
        width80=float((q[..., 3] - q[..., 1]).mean()),
        width90=float((q[..., 4] - q[..., 0]).mean()),
        under_q05=float((y < q[..., 0]).mean()), over_q95=float((y > q[..., 4]).mean()))
    result.update({f'pinball_{name}': float(pb[..., i].mean()) for i, name in enumerate(NAMES)})
    return result


def score_and_gain(actual, before, after):
    """Delta = after loss minus before; benefit/harm are positive parts.

    Direction and overshoot are q50 diagnostics over ALL supplied cells. Target
    crossing alone need not harm MAE; harmful overshoot exceeds twice |residual|.
    """
    y, b = validate(actual, before)
    _, a = validate(y, after)
    mb, qb, pb = cell_losses(y, b)
    ma, qa, pa = cell_losses(y, a)
    result = {f'before_{k}': v for k, v in score(y, b).items()}
    result.update({f'after_{k}': v for k, v in score(y, a).items()})
    for label, excess in (('mae', ma-mb), ('qs', qa-qb)):
        result.update({f'{label}_delta': float(excess.mean()),
                       f'{label}_benefit': float(np.maximum(-excess, 0).mean()),
                       f'{label}_harm': float(np.maximum(excess, 0).mean()),
                       f'{label}_harm_rate': float((excess > 1e-7).mean())})
    for i, name in enumerate(NAMES):
        excess = pa[..., i] - pb[..., i]
        result.update({f'pinball_{name}_delta': float(excess.mean()),
                       f'pinball_{name}_benefit': float(np.maximum(-excess, 0).mean()),
                       f'pinball_{name}_harm': float(np.maximum(excess, 0).mean())})
    result['lower_tail_qs_delta'] = float(2*(pa[..., :2]-pb[..., :2]).mean())
    result['upper_tail_qs_delta'] = float(2*(pa[..., 3:]-pb[..., 3:]).mean())
    residual, delta = y-b[..., 2], a[..., 2]-b[..., 2]
    result.update(wrong_direction_rate=float((delta * residual < 0).mean()),
        target_crossing_rate=float(((delta * residual > 0) & (np.abs(delta) > np.abs(residual))).mean()),
        harmful_overshoot_rate=float(((delta * residual > 0) & (np.abs(delta) > 2*np.abs(residual))).mean()),
        correction_from_exact_baseline_rate=float(((residual == 0) & (delta != 0)).mean()),
        zero_correction_rate=float((delta == 0).mean()),
        mean_abs_median_delta=float(np.abs(delta).mean()),
        mean_abs_quantile_delta=float(np.abs(a-b).mean()),
        mean_signed_median_delta=float(delta.mean()))
    return result


def anchor_stages(baseline, anchor):
    """Exact additive output decomposition, with no ratios at zero tree gaps.

    This does not rerun the coupled historical branch. Sequential attribution
    is order dependent and must not be described as retraining ablation.
    """
    b, a = np.asarray(baseline, dtype=np.float64), np.asarray(anchor, dtype=np.float64)
    if b.shape != a.shape or b.shape[-1] != 5:
        raise ValueError('Anchor/baseline shape mismatch')
    location = b + (a[..., 2]-b[..., 2])[..., None]
    lower = location.copy()
    lower[..., :2] = a[..., :2]
    upper = lower.copy()
    upper[..., 3:] = a[..., 3:]
    return [('baseline', b), ('location', location), ('lower_scale', lower), ('upper_scale', upper)]


def movement(baseline, anchor, full):
    b, a, f = (np.asarray(v, dtype=np.float64) for v in (baseline, anchor, full))
    location, history = a[..., 2]-b[..., 2], f[..., 2]-a[..., 2]
    denominator = np.abs(location)+np.abs(history)
    cancelled = denominator-np.abs(location+history)
    return dict(mean_abs_anchor_location=float(np.abs(location).mean()),
        mean_abs_history=float(np.abs(history).mean()), mean_abs_total=float(np.abs(f[..., 2]-b[..., 2]).mean()),
        opposing_fraction=float((location*history < 0).mean()),
        cancelled_magnitude=float(cancelled.mean()),
        cancellation_fraction=float(cancelled.sum()/denominator.sum()) if denominator.sum() else 0.,
        historical_common_shift_max_error=float(np.max(np.abs((f-a)-(f-a)[..., 2:3]))))
