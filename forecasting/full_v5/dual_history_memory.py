"""Causal historical-curve memory for a frozen dual-field encoder.

The encoder reconstructs *past* curves only after their complete outcome is
available. It never sees the target curve of the current query. All matched
representations share exactly the same neighbor IDs; only the representation
of each historical curve changes. Nothing is trained in this module.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import schema
from .history_correction import DAY, HISTORY_START, HOUR, MAX_CORRECTION, _arrays


VARIANTS = ("rawmatched", "recentraw", "fixedsmooth", "dualtrend", "dualfull")


def fixed_smooth(curves):
    """DC plus first harmonic, the encoder's fixed trend-prior comparator."""
    values = np.asarray(curves, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != schema.HORIZONS:
        raise ValueError("Expected curves [N,24]")
    spectrum = np.fft.rfft(values, axis=1)
    spectrum[:, 2:] = 0
    return np.fft.irfft(spectrum, n=schema.HORIZONS, axis=1)


def _encoded_values(encoded, size):
    """Verify local encoder indices, then map outputs back to supplied order."""
    ids = np.asarray(encoded["indices"])
    if (ids.shape != (size,) or not np.issubdtype(ids.dtype, np.integer)
            or not np.array_equal(np.sort(ids), np.arange(size))):
        raise ValueError("Encoder indices must map every supplied matured row exactly once")
    order = np.argsort(ids)
    fields = {}
    for name in ("trend", "event", "reconstruction"):
        value = np.asarray(encoded[name], dtype=np.float64)
        if value.shape != (size, schema.HORIZONS) or not np.isfinite(value).all():
            raise ValueError(f"Encoder {name} must be finite [M,24]")
        fields[name] = value[order]
    if not np.allclose(fields["trend"] + fields["event"], fields["reconstruction"],
                       rtol=1e-5, atol=1e-4):
        raise ValueError("Encoder reconstruction must equal trend plus event")
    return fields


def build_corrections(origins, actual, frozen_quantiles, z, lag_prices, encoder, *,
                      history_start=None):
    """Return five [N,24] corrections, causal audits, and encoded past curves.

    Full-window labels become available at ``origin + 24h`` (inclusive).
    Only October-onward records are admitted because C_roll was fitted before
    October. Matching uses the original six current-state coordinates, the
    same origin hour and lead, a 365-day window, K=16 and minimum pool size 8.

    ``encoder.encode_matured`` receives *only* newly released rows, with local
    indices mapped explicitly back to original input indices. Neither future
    targets nor pre-library records are passed to its forward method.

    Returned ``matched_indices`` are the single shared neighbor-ID tensor for
    rawmatched/fixedsmooth/dualtrend/dualfull; -1 marks unopened slots. The
    recentraw control selects the latest K rows from the same eligible pool.
    The correction is median(represented past curve - past C_roll median),
    shrunk by K_actual/16 and clipped to +/-150, with no fitted parameters.
    """
    origins, actual, quantiles = _arrays(origins, actual, frozen_quantiles)
    n = len(origins)
    start = HISTORY_START if history_start is None else int(history_start)
    if start < HISTORY_START:
        raise ValueError("History cannot predate the frozen C_roll fitting cutoff")
    z = np.asarray(z, dtype=np.float64)
    lag_prices = np.asarray(lag_prices, dtype=np.float64)
    if z.shape != (n, schema.HORIZONS, schema.NZ) or not np.isfinite(z).all():
        raise ValueError("Expected finite z [N,24,16]")
    if lag_prices.shape != (n, schema.HORIZONS, 3) or not np.isfinite(lag_prices).all():
        raise ValueError("Expected finite lag_prices [N,24,3]")

    shape = (n, schema.HORIZONS)
    corrections = {name: np.zeros(shape, dtype=np.float64) for name in VARIANTS}
    represented = {name: np.full(shape, np.nan) for name in
                   ("rawmatched", "fixedsmooth", "dualtrend", "dualfull")}
    events = np.full(shape, np.nan)
    encoded_at = np.full(n, -1, dtype=np.int64)
    matched_ids = np.full((n, schema.HORIZONS, schema.MATCH_K), -1, dtype=np.int64)
    recent_ids = np.full((n, schema.MATCH_K), -1, dtype=np.int64)
    states = z[:, :, list(schema.MATCH_Z_INDICES)]
    hours = (origins // HOUR + 10) % 24
    first = int(np.searchsorted(origins, start, side="left"))
    released = first
    open_queries, calls = 0, 0
    pool_sizes, selection_samples, release_samples = [], [], []

    for i, value in enumerate(origins):
        t = int(value)
        hi = int(np.searchsorted(origins, t - schema.HORIZONS * HOUR, side="right"))
        # Bounds are based on publication time before any label slice or FFT.
        lo = max(first, int(np.searchsorted(origins, t - schema.LOOKBACK_DAYS * DAY,
                                           side="right")))
        new = np.arange(max(released, lo), hi, dtype=np.int64)
        if len(new):
            result = encoder.encode_matured(origins[new], actual[new], z[new],
                                            lag_prices[new], as_of=t)
            fields = _encoded_values(result, len(new))
            represented["rawmatched"][new] = actual[new]
            represented["fixedsmooth"][new] = fixed_smooth(actual[new])
            represented["dualtrend"][new] = fields["trend"]
            represented["dualfull"][new] = fields["reconstruction"]
            events[new] = fields["event"]
            encoded_at[new] = t
            calls += 1
            if len(release_samples) < 16:
                release_samples.append(dict(as_of=t, global_indices=new.tolist(),
                    latest_label_available=int(origins[new[-1]] + schema.HORIZONS * HOUR)))
        released = max(released, hi)
        pool = np.arange(lo, hi, dtype=np.int64)
        pool = pool[hours[pool] == hours[i]]
        pool_sizes.append(len(pool))
        if len(pool) < schema.MATCH_MIN_CANDIDATES:
            continue
        if np.any(encoded_at[pool] < 0) or np.any(encoded_at[pool] > t):
            raise RuntimeError("Candidate history was not causally encoded")
        open_queries += 1
        k = min(schema.MATCH_K, len(pool))
        eta = k / schema.MATCH_K
        recent = pool[-k:]
        recent_ids[i, :k] = recent
        corrections["recentraw"][i] = np.clip(eta * np.median(
            represented["rawmatched"][recent] - quantiles[recent, :, 2], axis=0),
            -MAX_CORRECTION, MAX_CORRECTION)
        for h in range(schema.HORIZONS):
            distances = np.sum((states[pool, h] - states[i, h]) ** 2, axis=1)
            chosen = pool[np.argsort(distances, kind="stable")[:k]]
            matched_ids[i, h, :k] = chosen
            for name, curves in represented.items():
                residuals = curves[chosen, h] - quantiles[chosen, h, 2]
                corrections[name][i, h] = np.clip(eta * np.median(residuals),
                                                 -MAX_CORRECTION, MAX_CORRECTION)
            if h == 0 and len(selection_samples) < 16 and open_queries % 96 == 1:
                selection_samples.append(dict(query_origin=t, horizon=h,
                    selected_indices=chosen.tolist(), selected_origins=origins[chosen].tolist(),
                    recent_indices=recent.tolist()))

    encoded_ids = np.flatnonzero(encoded_at >= 0)
    availability = origins[encoded_ids] + schema.HORIZONS * HOUR
    release_margin = encoded_at[encoded_ids] - availability
    digest = hashlib.sha256(np.ascontiguousarray(matched_ids).tobytes()).hexdigest()
    audit = dict(history_start=start, queries=n, open_queries=open_queries,
        open_fraction=open_queries / max(n, 1), encoder_calls=calls,
        encoded_records=len(encoded_ids), lookback_days=schema.LOOKBACK_DAYS,
        match_k=schema.MATCH_K, match_min_candidates=schema.MATCH_MIN_CANDIDATES,
        pool_size_max=max(pool_sizes, default=0), max_abs_correction=MAX_CORRECTION,
        residual_definition="represented mature past price - frozen_C_roll_q50",
        matching_uses="current and historical six z coordinates only; no query target",
        matched_variants_share_ids=True,
        matched_variant_candidate_sha256={name: digest for name in represented},
        latest_encoded_availability=int(availability.max()) if len(encoded_ids) else None,
        minimum_release_margin_seconds=int(release_margin.min()) if len(encoded_ids) else None,
        maturity_violations=int(np.sum(release_margin < 0)),
        selection_samples=selection_samples, release_samples=release_samples)
    return dict(corrections=corrections, audit=audit, matched_indices=matched_ids,
        recent_indices=recent_ids, encoded_curves=dict(indices=encoded_ids,
            origins=origins[encoded_ids], available_at=availability,
            encoded_at=encoded_at[encoded_ids], trend=represented["dualtrend"][encoded_ids],
            event=events[encoded_ids], reconstruction=represented["dualfull"][encoded_ids],
            fixedsmooth=represented["fixedsmooth"][encoded_ids]))
