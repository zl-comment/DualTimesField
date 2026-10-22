"""Pure selection-period safety guard; no evaluation labels or files are read."""
import math

import numpy as np

from ..full_v5_joint.model import LEVELS

RTOL = 1e-7


def _quantiles(prediction):
    q = np.asarray(prediction)
    if q.ndim < 2 or q.shape[-1] != len(LEVELS) or not q.size:
        raise ValueError('Expected nonempty five-quantile predictions')
    if not np.isfinite(q).all() or (np.diff(q, axis=-1) < 0).any():
        raise ValueError('Quantiles must be finite and ordered')
    return q


def score(actual, prediction):
    q = _quantiles(prediction).astype(np.float64, copy=False)
    y = np.asarray(actual, dtype=np.float64)
    if y.shape != q.shape[:-1] or not np.isfinite(y).all():
        raise ValueError('Actuals must be finite and match prediction shape')
    error = y[..., None] - q
    levels = np.asarray(LEVELS)
    return dict(mae=float(np.abs(error[..., 2]).mean()),
                qs=float(2 * np.maximum(levels * error, (levels - 1) * error).mean()),
                coverage_80=float(((y >= q[..., 1]) & (y <= q[..., 3])).mean()),
                coverage_90=float(((y >= q[..., 0]) & (y <= q[..., 4])).mean()),
                bias=float((-error[..., 2]).mean()))


def blend(baseline, prediction, alpha):
    b, q = _quantiles(baseline), _quantiles(prediction)
    if b.shape != q.shape:
        raise ValueError('Baseline and prediction shapes differ')
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Alpha must be finite and between zero and one')
    if alpha == 0:
        return b.copy()  # Exact fallback, including the baseline dtype.
    if alpha == 1:
        return q.copy()
    # Convert before arithmetic to avoid overflow in narrower input dtypes.
    result = (1 - alpha) * b.astype(np.float64) + alpha * q.astype(np.float64)
    return _quantiles(result)


def selection_key(candidate):
    """Sort feasible positive-alpha candidates before fallbacks across epochs.

    Within feasible candidates: Full QS, Full MAE, smaller alpha, earlier epoch.
    Fallbacks compete only when no epoch supplies an eligible positive alpha.
    """
    return (bool(candidate['fallback']), candidate['full']['qs'],
            candidate['full']['mae'], candidate['alpha'], candidate['epoch'])


def select_candidate(actual, baseline, anchor, full, alphas=(0, .25, .5, 1), *, epoch=0):
    """Select one shared alpha for Anchor and Full, using only supplied labels.

    Four aggregate MAE/QS guards allow 1e-7 relative numerical tolerance.
    Full must strictly improve MAE or QS beyond that tolerance. The returned
    dictionary is JSON-serializable and stores scores, not prediction arrays.
    """
    if not isinstance(epoch, (int, np.integer)) or epoch < 0:
        raise ValueError('Epoch must be a nonnegative integer')
    bscore = score(actual, baseline)
    # Validate even when the only requested alpha is zero.
    score(actual, anchor)
    score(actual, full)
    candidates = []
    for alpha in alphas:
        a = float(alpha)
        ascore = score(actual, blend(baseline, anchor, a))
        fscore = score(actual, blend(baseline, full, a))
        if a == 0:
            continue
        safe = all(s[m] <= bscore[m] + RTOL * abs(bscore[m])
                   for s in (ascore, fscore) for m in ('mae', 'qs'))
        improves = any(fscore[m] < bscore[m] - RTOL * abs(bscore[m])
                       for m in ('mae', 'qs'))
        if safe and improves:
            candidates.append(dict(alpha=a, epoch=int(epoch), fallback=False,
                                   reason='positive_alpha_passes_four_guards_and_improves_full',
                                   baseline=bscore, anchor=ascore, full=fscore))
    if candidates:
        return min(candidates, key=selection_key)
    return dict(alpha=0., epoch=int(epoch), fallback=True,
                reason='no_positive_alpha_passes_four_guards_and_improves_full',
                baseline=bscore, anchor=dict(bscore), full=dict(bscore))
