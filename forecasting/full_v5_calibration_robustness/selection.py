"""Four chronological windows inside the original selection year only."""
import numpy as np

from ..full_v5_anchor_balance.experiment import local_dates
from ..full_v5_anchor_balance.selection import RTOL, blend, score


def quarter_windows(origins):
    origins = np.asarray(origins)
    if origins.ndim != 1 or not len(origins) or np.any(np.diff(origins) <= 0):
        raise ValueError('Selection origins must be nonempty and ordered')
    months = local_dates(origins).astype('datetime64[M]').astype(str)
    labels = np.array([m[:4] + 'Q' + str((int(m[-2:])-1)//3+1) for m in months])
    names = np.unique(labels)
    if len(names) != 4 or len({name[:4] for name in names}) != 1:
        raise ValueError('Four quarters in one selection year are required')
    # Checkpoint metadata must contain Python primitives, not numpy.str_ keys:
    # weights_only=True deliberately rejects NumPy scalar pickle constructors.
    return {str(name): labels == name for name in names}


def window_key(candidate):
    return (bool(candidate['fallback']), candidate['worst_quarter_relative_qs'],
            candidate['mean_quarter_relative_qs'], candidate['full']['mae'],
            candidate['alpha'], candidate['epoch'])


def select_windows(actual, baseline, anchor, full, windows, alphas=(0, .25, .5, 1), *, epoch=0):
    masks = list(windows.values())
    if len(masks) != 4 or any(np.asarray(m).dtype != bool or np.asarray(m).shape != (len(actual),)
                              or not np.any(m) for m in masks):
        raise ValueError('Four nonempty selection window masks required')
    if not np.all(np.sum(masks, axis=0) == 1):
        raise ValueError('Selection windows must partition the original selection origins')
    bs = score(actual, baseline)
    bq = {name: score(actual[mask], baseline[mask]) for name, mask in windows.items()}
    score(actual, anchor)
    score(actual, full)
    candidates, audit = [], []
    for alpha in alphas:
        a, f = blend(baseline, anchor, alpha), blend(baseline, full, alpha)
        if alpha == 0:
            continue
        aggregate = dict(anchor=score(actual, a), full=score(actual, f))
        quarters = {name: dict(baseline=bq[name], anchor=score(actual[mask], a[mask]),
            full=score(actual[mask], f[mask])) for name, mask in windows.items()}
        safe = all(value[component][metric] <= value['baseline'][metric] + RTOL*abs(value['baseline'][metric])
            for value in [dict(baseline=bs, **aggregate), *quarters.values()]
            for component in ('anchor', 'full') for metric in ('mae', 'qs'))
        improves = any(aggregate['full'][m] < bs[m]-RTOL*abs(bs[m]) for m in ('mae', 'qs'))
        audit.append(dict(alpha=float(alpha), safe=bool(safe), improves=bool(improves),
                          aggregate=aggregate, quarters=quarters))
        if safe and improves:
            relative = [(v['full']['qs']-v['baseline']['qs'])/max(abs(v['baseline']['qs']), 1e-12)
                        for v in quarters.values()]
            candidates.append(dict(epoch=epoch, alpha=float(alpha), fallback=False, baseline=bs, **aggregate,
                quarters=quarters, worst_quarter_relative_qs=max(relative),
                mean_quarter_relative_qs=float(np.mean(relative)),
                reason='all_four_quarters_and_original_aggregate_guards_pass'))
    if candidates:
        return dict(min(candidates, key=window_key), alpha_audit=audit)
    return dict(epoch=epoch, alpha=0., fallback=True, baseline=bs, anchor=dict(bs), full=dict(bs),
        quarters={name: dict(baseline=v, anchor=dict(v), full=dict(v)) for name, v in bq.items()},
        worst_quarter_relative_qs=0., mean_quarter_relative_qs=0., alpha_audit=audit,
        reason='no_positive_alpha_passes_all_four_quarters_and_aggregate_guards')
