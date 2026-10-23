"""Outcome-based mechanism diagnostics: never used by the prediction gate."""
import numpy as np
from .policy import STATES, classify


def diagnose(actual, baseline, anchor, full, inputs, policy):
    y, b, a, f = (np.asarray(v, dtype=np.float64) for v in (actual, baseline, anchor, full))
    states = classify(inputs, policy['thresholds'])
    gate = np.asarray(policy['gates'])[states]
    residual, delta = y - b[..., 2], a[..., 2] - b[..., 2]
    excess = np.abs(y - a[..., 2]) - np.abs(residual)
    masks = {'all': np.ones(y.shape, bool), 'weak_location': gate[..., 0] <= .25,
             'stronger_location': gate[..., 0] > .25}
    masks.update({f'state/{name}': states == i for i, name in enumerate(STATES)})
    result = {}
    for name, m in masks.items():
        if not m.any():
            result[name] = dict(cells=0, origins=0, empty=True)
            continue
        result[name] = dict(cells=int(m.sum()), origins=int(m.any(axis=1).sum()), empty=False,
            mean_location_gate=float(gate[..., 0][m].mean()),
            anchor_harm_rate=float((excess[m] > 1e-7).mean()),
            anchor_harm=float(np.maximum(excess[m], 0).mean()),
            anchor_benefit=float(np.maximum(-excess[m], 0).mean()),
            anchor_mae_delta=float(excess[m].mean()),
            full_mae_delta=float((np.abs(y - f[..., 2]) - np.abs(residual))[m].mean()),
            mean_abs_location=float(np.abs(delta[m]).mean()),
            wrong_direction_rate=float((delta[m] * residual[m] < 0).mean()),
            target_crossing_rate=float(((delta[m] * residual[m] > 0) & (np.abs(delta[m]) > np.abs(residual[m]))).mean()),
            baseline_width80=float((b[..., 3] - b[..., 1])[m].mean()),
            anchor_width80=float((a[..., 3] - a[..., 1])[m].mean()),
            baseline_coverage80=float(((y >= b[..., 1]) & (y <= b[..., 3]))[m].mean()),
            anchor_coverage80=float(((y >= a[..., 1]) & (y <= a[..., 3]))[m].mean()),
            full_coverage80=float(((y >= f[..., 1]) & (y <= f[..., 3]))[m].mean()))
    return result
