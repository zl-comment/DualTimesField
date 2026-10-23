"""Fixed physical-state partition; chronological evidence is not a causal claim.

Only fit_policy sees labels, and only the explicitly passed training labels.
Classification at inference consumes six already shared, available inputs.
"""
import numpy as np

STATES = ('ordinary', 'tight', 'tight_import_limited', 'surplus',
          'surplus_export_limited', 'uncovered')
FEATURES = ('max_spare_capacity_mw', 'net_load_mw', 'ic_import_room_mw',
            'ic_export_room_mw', 'predispatch_covered', 'ic_covered')
HEADS = ('location', 'lower_scale', 'upper_scale')
BLOCKS = 4
SUPPORT_DAYS = 30.


def validate_inputs(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (24, 6) or not np.isfinite(x).all():
        raise ValueError('Expected finite [origin,24,6] mechanism inputs')
    if not np.isin(x[..., 4:], [0., 1.]).all():
        raise ValueError('Coverage flags must be raw binary values')
    return x


def classify(x, thresholds):
    x = validate_inputs(x)
    spare20, spare50, load20, load50, import20, export20 = thresholds
    spare, load, imp, exp = (x[..., i] for i in range(4))
    state = np.zeros(x.shape[:2], dtype=np.int64)
    tight = (spare < spare20) & (load >= load50)
    surplus = (load < load20) & (spare >= spare50)
    state[tight] = 1
    state[tight & (imp < import20)] = 2
    state[surplus] = 3
    state[surplus & (exp < export20)] = 4
    state[(x[..., 4] < .5) | (x[..., 5] < .5)] = 5
    return state


def fit_policy(x, actual, baseline, origins):
    """Four chronological training blocks, with one support count per local day.

    Location evidence is the median OOF tree residual. Tail evidence is the
    signed coverage error after removing that block/state median residual.
    Evidence only controls magnitude, never the sign of a neural correction.
    """
    x = validate_inputs(x)
    y, b, t = np.asarray(actual, dtype=np.float64), np.asarray(baseline, dtype=np.float64), np.asarray(origins)
    if y.shape != x.shape[:2] or b.shape != (*y.shape, 5) or t.shape != (len(x),):
        raise ValueError('Training grid mismatch')
    if len(x) < BLOCKS or np.any(np.diff(t) <= 0) or not np.isfinite(y).all() or not np.isfinite(b).all():
        raise ValueError('Finite chronological training observations required')
    if np.any(np.diff(b, axis=-1) < 0):
        raise ValueError('Crossed baseline quantiles')
    valid = (x[..., 4:] == 1).all(axis=-1)
    if not valid.any():
        raise ValueError('No covered training states')
    v = x[valid]
    thresholds = [*np.quantile(v[:, 0], [.2, .5]), *np.quantile(v[:, 1], [.2, .5]),
                  np.quantile(v[:, 2], .2), np.quantile(v[:, 3], .2)]
    states = classify(x, thresholds)
    days = (t + 10 * 3600) // 86400
    evidence = np.zeros((6, BLOCKS, 3), dtype=np.float64)
    counts = np.zeros((6, BLOCKS), dtype=np.int64)
    blocks = []
    # Split unique days, not 24 overlapping target cells into pseudo-independent blocks.
    for j, selected_days in enumerate(np.array_split(np.unique(days), BLOCKS)):
        if not len(selected_days):
            raise ValueError('At least four distinct training days required')
        row_ids = np.flatnonzero(np.isin(days, selected_days))
        blocks.append(dict(first_origin=int(t[row_ids[0]]), last_origin=int(t[row_ids[-1]])))
        for state in range(5):
            mask = states[row_ids] == state
            counts[state, j] = len(np.unique(days[row_ids][mask.any(axis=1)]))
            if not mask.any():
                continue
            yy, bb = y[row_ids][mask], b[row_ids][mask]
            shift = np.median(yy - bb[:, 2])
            centered = yy - shift
            evidence[state, j] = [shift,
                .5 * (np.mean(centered < bb[:, 0]) - .05 + np.mean(centered < bb[:, 1]) - .1),
                .5 * (np.mean(centered > bb[:, 3]) - .1 + np.mean(centered > bb[:, 4]) - .05)]
    minimum_days = counts.min(axis=1)
    support = minimum_days / (minimum_days + SUPPORT_DAYS)
    denominator = np.abs(evidence).sum(axis=1)
    coherence = np.divide(np.abs(evidence.sum(axis=1)), denominator,
                          out=np.zeros_like(denominator), where=denominator > 0)
    gates = support[:, None] * coherence
    gates[5] = 0.
    frequencies = np.bincount(states.ravel(), minlength=6)
    constant = np.average(gates[:5], weights=frequencies[:5], axis=0)
    return dict(version=1, features=list(FEATURES), states=list(STATES), heads=list(HEADS),
        thresholds=np.asarray(thresholds).tolist(), gates=gates.tolist(), constant_gates=constant.tolist(),
        support_days=SUPPORT_DAYS, blocks=blocks, day_counts=counts.tolist(),
        block_signals=evidence.tolist(), support=support.tolist(), coherence=coherence.tolist(),
        cell_counts=frequencies.tolist(), training_first_origin=int(t[0]), training_last_origin=int(t[-1]),
        meaning='descriptive historical evidence; not a confidence probability or causal identification')
