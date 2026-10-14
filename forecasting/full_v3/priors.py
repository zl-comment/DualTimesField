"""Five predeclared contrasts; no query labels participate in feature construction."""
from __future__ import annotations

import numpy as np

VARIANTS = ('v3_level_lgb', 'v3_analog_features_lgb', 'v3_analog_prior_lgb',
            'v3_dual_shared_lgb', 'v3_dual_conditional_lgb')
EVENT_COLUMNS = (0, 1, 2, 3, 4, 6, 7, 8)


def nearest_weights(distance, valid, top_k=4, temperature=1.0):
    """Stable nearest-k softmax along candidate axis 1, including empty rows."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive')
    d = np.asarray(distance, dtype=np.float64)
    mask = np.broadcast_to(valid if d.ndim == 2 else valid[..., None], d.shape)
    masked = np.where(mask, d, np.inf)
    order = np.argsort(masked, axis=1, kind='stable')
    keep = np.zeros_like(mask, dtype=bool)
    np.put_along_axis(keep, order[:, :top_k], True, axis=1)
    keep &= mask
    floor = np.min(masked, axis=1, keepdims=True)
    floor = np.where(np.isfinite(floor), floor, 0.)
    weights = np.exp(-np.where(keep, d - floor, np.inf) / temperature)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)


def build_priors(memory, split, fields, *, config=None):
    """Raw-unit priors plus standardized features and a complete retrieval audit.

    C and D distances use canonical frozen standardized inputs. Prior levels and
    decomposed curves use raw prices. Calibration/test labels at the query are
    never read; previously published library outcomes are legitimate inputs.
    """
    cfg = dict((config or {}).get('retrieval', config or {}))
    top_k, temperature = cfg.get('top_k', 4), cfg.get('temperature', 1.)
    batch = cfg.get('batch_size', 256)
    if not isinstance(batch, int) or batch < 1:
        raise ValueError('batch_size must be positive')
    if (split.metadata['target_std'] != memory.target_std
            or split.metadata['target_mean'] != memory.target_mean
            or split.metadata['region'] != memory.region
            or split.metadata.get('preprocessing_hash') != memory.preprocessing_hash):
        raise ValueError('Query/library scale or region mismatch')
    n, h = len(split.origin_unix), 24
    indices = memory.candidate_indices(split.origin_unix)
    valid = indices >= 0
    k = indices.shape[1]
    audits = dict(indices=indices, valid=valid,
                  max_availability=memory.candidate_max_availability(indices),
                  w_shared=np.zeros((n, k)), w_background=np.zeros((n, k)),
                  w_event=np.zeros((n, k, h)))
    raw = np.empty((n, k * (h + 5)), dtype=np.float32)
    field = np.empty((n, 4 * h), dtype=np.float32)
    priors = {v: np.zeros((n, h), dtype=np.float64) for v in VARIANTS}
    std = memory.target_std
    for start in range(0, n, batch):
        sl = slice(start, min(start + batch, n))
        ix, ok = np.maximum(indices[sl], 0), valid[sl]
        hist = np.asarray(split.inputs['history_values'][sl], dtype=np.float64)
        level_z = np.median(hist[:, -24:, 0], axis=1)
        level = level_z * std + memory.target_mean
        cand_hist = memory.inputs['history_values'][ix].astype(np.float64)
        cand_level_z = np.median(cand_hist[:, :, -24:, 0], axis=2)
        centered = hist[:, None, :, 0] - level_z[:, None, None]
        cand_centered = cand_hist[..., 0] - cand_level_z[..., None]
        dc = np.mean((centered - cand_centered) ** 2, axis=2)
        dc += .5 * np.mean((hist[:, None, -24:, 1] - cand_hist[:, :, -24:, 1]) ** 2, axis=2)
        cal = split.inputs['future_calendar'][sl, :, :8].astype(np.float64)
        cand_cal = memory.inputs['future_calendar'][ix, :, :8].astype(np.float64)
        dc += .5 * np.mean((cal[:, None] - cand_cal) ** 2, axis=(2, 3))
        event = split.inputs['future_exogenous'][sl][..., EVENT_COLUMNS].astype(np.float64)
        cand_event = memory.inputs['future_exogenous'][ix][..., EVENT_COLUMNS].astype(np.float64)
        dd = np.mean((event[:, None] - cand_event) ** 2, axis=3)
        ds = dc + dd.mean(axis=2)
        ws = nearest_weights(ds, ok, top_k, temperature)
        wc = nearest_weights(dc, ok, top_k, temperature)
        wd = nearest_weights(dd, ok, top_k, temperature)
        audits['w_shared'][sl], audits['w_background'][sl], audits['w_event'][sl] = ws, wc, wd
        # Mask before arithmetic: unavailable rows may contain poisoned/NaN values.
        lj = np.where(ok, memory.levels[ix], 0.)
        curves = np.where(ok[..., None], memory.curves[ix], 0.)
        c = np.where(ok[..., None], fields['background'][ix], 0.)
        d = np.where(ok[..., None], fields['event'][ix], 0.)
        centered_curves = curves - lj[..., None]
        centered_c = c - lj[..., None]
        b_shared = np.sum(ws[..., None] * centered_c, axis=1)
        e_shared = np.sum(ws[..., None] * d, axis=1)
        b_dual = np.sum(wc[..., None] * centered_c, axis=1)
        e_dual = np.sum(wd * d, axis=1)
        priors[VARIANTS[0]][sl] = level[:, None]
        priors[VARIANTS[2]][sl] = level[:, None] + np.sum(ws[..., None] * centered_curves, axis=1)
        priors[VARIANTS[3]][sl] = level[:, None] + b_shared + e_shared
        priors[VARIANTS[4]][sl] = level[:, None] + b_dual + e_dual
        raw[sl] = np.concatenate((centered_curves.reshape(len(ix), -1) / std,
                                 np.where(ok, (lj - level[:, None]) / std, 0.), ok,
                                 np.where(ok, dc, 0.), np.where(ok, dd.mean(axis=2), 0.), ws), axis=1)
        field[sl] = np.concatenate((b_shared, e_shared, b_dual, e_dual), axis=1) / std
    if not all(np.isfinite(v).all() for v in [raw, field, *priors.values()]):
        raise ValueError('Nonfinite eligible inputs or priors')
    if np.any(audits['max_availability'][valid] > np.broadcast_to(split.origin_unix[:, None], valid.shape)[valid]):
        raise AssertionError('Historical retrieval contains unpublished values')
    return dict(raw_features=raw, field_features=field, priors=priors, audit=audits)


def variant_features(variant, raw_features, field_features):
    if variant not in VARIANTS:
        raise ValueError('Unknown V3 contrast')
    if variant == VARIANTS[0]:
        return np.empty((len(raw_features), 0), dtype=np.float32)
    if variant in VARIANTS[1:3]:
        return raw_features
    return np.concatenate((raw_features, field_features), axis=1)
