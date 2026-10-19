"""Causal conditional transport of mature historical trend and event fields.

Prices of the current query never construct its features or corrections. One
event per mature same-origin-hour historical curve prevents selecting several
overlapping windows for the same historical day within a query.
"""
from __future__ import annotations

import hashlib
import numpy as np

from . import schema
from .dual_history_memory import fixed_smooth
from .history_correction import DAY, HISTORY_START, MAX_CORRECTION

EVENT_K = 4
STATE_IDS = np.arange(1, 10)
FEATURE_NAMES = tuple(c[0] for c in schema.Z_COLUMNS) + (
    "lag72_mean", "lag72_std", "lag_last_price", "lag24_mean",
    "trend_correction", "event_correction", "trend_match_distance",
    "recent28_median_error", "event_best_distance", "event_support",
    "event_age", "eligible_pool_count", "event_movement",
)


def _weights(distance, age):
    log_weight = -np.asarray(distance) / .1 - np.asarray(age) / 28.
    exp_weight = np.exp(log_weight - log_weight.max(axis=0, keepdims=True))
    return exp_weight / exp_weight.sum(axis=0, keepdims=True)


def build_transfer_memory(origins, actual, anchor, z, lags, encoder, *, fixed=False, shift=True):
    """Build bounded corrections and query features using released labels only.

    ``lags[N,24,3]`` contains the three consecutive past days in columns,
    oldest first. Atom amplitudes returned by the encoder are in price units.
    Fixed geometry and unshifted controls use the same output feature schema.
    """
    origins = np.asarray(origins, dtype=np.int64)
    actual, anchor = np.asarray(actual, dtype=np.float64), np.asarray(anchor, dtype=np.float64)
    n = len(origins)
    if origins.ndim != 1 or np.any(np.diff(origins) <= 0):
        raise ValueError("Origins must be strictly increasing")
    if actual.shape != (n, 24) or anchor.shape != (n, 24, 5):
        raise ValueError("Expected actual[N,24] and anchor[N,24,5]")
    if not np.isfinite(anchor).all() or np.any(np.diff(anchor, axis=-1) < 0):
        raise ValueError("Expected finite ordered anchor quantiles")
    z, lags = np.asarray(z, dtype=np.float64), np.asarray(lags, dtype=np.float64)
    if z.shape != (n, 24, schema.NZ) or lags.shape != (n, 24, 3):
        raise ValueError("Expected z[N,24,16] and lags[N,24,3]")
    if not np.isfinite(z).all() or not np.isfinite(lags).all() or np.any(np.abs(z) > 1 + 1e-6):
        raise ValueError("Expected finite lags and bounded z")
    trend, event = np.zeros((n, 24)), np.zeros((n, 24))
    features = np.zeros((n, 24, len(FEATURE_NAMES)))
    features[..., :16] = np.clip(z, -1., 1.)
    lag_summaries = np.column_stack((lags.mean(axis=(1, 2)), lags.std(axis=(1, 2)),
                                    lags[:, -1, -1], lags[:, :, -1].mean(axis=1)))
    features[..., 16:20] = np.clip(lag_summaries[:, None, :] / 300., -1., 1.)
    features[..., 24] = 1.  # no event: maximum distance/age, zero support
    features[..., 26] = 1.
    trend_ids = np.full((n, 24, schema.MATCH_K), -1, dtype=np.int64)
    event_ids = np.full((n, EVENT_K), -1, dtype=np.int64)
    source_h, target_h = np.full_like(event_ids, -1), np.full_like(event_ids, -1)
    proposed_h = np.full_like(event_ids, -1)
    event_distance, event_weight = np.zeros((n, EVENT_K)), np.zeros((n, EVENT_K))
    selected_centers, selected_widths, selected_amplitudes = (np.zeros((n, EVENT_K)) for _ in range(3))
    encoded_at = np.full(n, -1, dtype=np.int64)
    historical_trend = np.full((n, 24), np.nan)
    historical_baseline = np.full((n, 24), np.nan)
    peak_h = np.full(n, -1, dtype=np.int64)
    atom_center, atom_width, atom_amplitude = (np.zeros(n) for _ in range(3))
    event_valid = np.zeros(n, dtype=bool)
    hours = (origins // 3600 + 10) % 24
    first = int(np.searchsorted(origins, HISTORY_START))
    released = first
    calls, open_queries, peak_candidates = 0, 0, 0
    selection_samples = []
    for i, origin in enumerate(origins):
        t = int(origin)
        hi = int(np.searchsorted(origins, t - DAY, side="right"))
        lo = max(first, int(np.searchsorted(origins, t - schema.LOOKBACK_DAYS * DAY, side="right")))
        new = np.arange(max(released, lo), hi)
        if len(new):
            if not np.isfinite(actual[new]).all():
                raise ValueError("Nonfinite matured historical prices")
            fields = encoder.encode_matured(origins[new], actual[new], z[new], lags[new], as_of=t, fixed=fixed)
            ids = np.asarray(fields["indices"])
            if ids.shape != (len(new),) or not np.issubdtype(ids.dtype, np.integer) or not np.array_equal(np.sort(ids), np.arange(len(new))):
                raise ValueError("Encoder must return each supplied mature row exactly once")
            order = np.argsort(ids)
            ft = np.asarray(fields["trend"], dtype=np.float64)[order]
            centers, widths, amplitudes = (np.asarray(fields[k], dtype=np.float64)[order]
                                            for k in ("centers", "widths", "amplitude"))
            if (ft.shape != (len(new), 24) or centers.ndim != 2 or centers.shape[0] != len(new)
                    or centers.shape != widths.shape or centers.shape != amplitudes.shape
                    or centers.shape[1] == 0 or np.any(widths <= 0)
                    or not all(np.isfinite(a).all() for a in (ft, centers, widths, amplitudes))):
                raise ValueError("Invalid mature encoder fields or geometry")
            historical_trend[new] = ft
            historical_baseline[new] = fixed_smooth(anchor[new, :, 2])
            encoded_at[new] = t
            # This is the only place actual curves enter peak selection.
            peaks = actual[new].argmax(axis=1)
            for j, row in enumerate(new):
                h = int(peaks[j])
                missed = actual[row, h] - anchor[row, h, 2]
                if actual[row, h] <= 300 or missed <= 0:
                    continue
                # Candidate identity depends only on released raw prices and
                # frozen forecasts, never on which geometry happens to fit.
                peak_h[row], atom_center[row], atom_width[row] = h, h, .5
                event_valid[row] = True
                peak_candidates += 1
                distance = np.where(amplitudes[j] > 0, np.abs(centers[j] - h), np.inf)
                atom = int(distance.argmin())
                if distance[atom] > 2:
                    continue
                trend_error = historical_trend[row, h] - historical_baseline[row, h]
                remaining = max(0., missed - max(trend_error, 0.))
                # Bound the entire transported pulse, not only its value at
                # the original observed peak. A center offset must not allow
                # division by a small Gaussian value to inflate the peak.
                amp = min(amplitudes[j, atom], remaining)
                peak_h[row], atom_center[row], atom_width[row], atom_amplitude[row] = h, centers[j, atom], widths[j, atom], amp
            calls += 1
        released = max(released, hi)
        pool = np.arange(lo, hi)
        pool = pool[hours[pool] == hours[i]]
        features[i, :, 27] = min(len(pool) / 365., 1.)
        recent = pool[origins[pool] >= t - 28 * DAY]
        if len(recent):
            features[i, :, 23] = np.clip(np.median(actual[recent] - anchor[recent, :, 2], axis=0) / 150., -1., 1.)
        if len(pool) < schema.MATCH_MIN_CANDIDATES:
            continue
        if np.any(encoded_at[pool] < 0) or np.any(encoded_at[pool] > t):
            raise RuntimeError("Historical feature consumed before encoding")
        open_queries += 1
        age = (t - origins[pool]) / DAY
        state_distance = np.mean((z[pool][:, :, schema.MATCH_Z_INDICES] - z[i][:, schema.MATCH_Z_INDICES]) ** 2, axis=-1)
        k = min(schema.MATCH_K, len(pool))
        chosen_local = np.argsort(state_distance, axis=0, kind="stable")[:k]
        leads = np.arange(24)[None, :]
        chosen = pool[chosen_local]
        distances = state_distance[chosen_local, leads]
        weights = _weights(distances, age[chosen_local])
        trend[i] = np.clip(np.sum(weights * (historical_trend[chosen, leads] - historical_baseline[chosen, leads]), axis=0), -MAX_CORRECTION, MAX_CORRECTION)
        trend_ids[i, :, :k] = chosen.T
        features[i, :, 22] = np.clip(np.sum(weights * distances, axis=0) / 4., 0., 1.)
        ep = pool[event_valid[pool]]
        if len(ep):
            # Distances use current known state at all target leads, no target prices.
            state = z[ep, peak_h[ep]][:, STATE_IDS]
            all_distance = np.mean((state[:, None, :] - z[i][:, STATE_IDS][None, :, :]) ** 2, axis=-1)
            best_h = all_distance.argmin(axis=1)
            best_d = all_distance[np.arange(len(ep)), best_h]
            selected = np.argsort(best_d, kind="stable")[:EVENT_K]
            records, proposal, distance = ep[selected], best_h[selected], best_d[selected]
            target = proposal if shift else peak_h[records]
            ages = (t - origins[records]) / DAY
            weights = _weights(distance, ages)
            centers = atom_center[records] + target - peak_h[records]
            pulses = atom_amplitude[records, None] * np.exp(-.5 * ((np.arange(24)[None] - centers[:, None]) / atom_width[records, None]) ** 2)
            event[i] = np.clip(np.sum(weights[:, None] * pulses, axis=0), 0., MAX_CORRECTION)
            count = len(records)
            event_ids[i, :count], source_h[i, :count], target_h[i, :count] = records, peak_h[records], target
            proposed_h[i, :count] = proposal
            event_distance[i, :count], event_weight[i, :count] = distance, weights
            selected_centers[i, :count], selected_widths[i, :count], selected_amplitudes[i, :count] = centers, atom_width[records], atom_amplitude[records]
            features[i, :, 24] = min(float(distance.min()) / 4., 1.)
            features[i, :, 25] = count / EVENT_K
            features[i, :, 26] = min(float(np.sum(weights * ages)) / 365., 1.)
            features[i, :, 28] = min(float(np.sum(weights * np.abs(target - peak_h[records]))) / 23., 1.)
            if len(selection_samples) < 16:
                selection_samples.append(dict(query_origin=t, event_origins=origins[records].tolist(),
                    source_h=peak_h[records].tolist(), target_h=target.tolist()))
    features[..., 20], features[..., 21] = trend / 150., event / 150.
    valid = event_ids >= 0
    encoded = np.flatnonzero(encoded_at >= 0)
    margins = encoded_at[encoded] - origins[encoded] - DAY
    event_margins = np.broadcast_to(origins[:, None], event_ids.shape)[valid] - origins[event_ids[valid]] - DAY
    selected_trend = trend_ids >= 0
    trend_margins = np.broadcast_to(origins[:, None, None], trend_ids.shape)[selected_trend] - origins[trend_ids[selected_trend]] - DAY
    violations = int(np.sum(margins < 0) + np.sum(event_margins < 0) + np.sum(trend_margins < 0))
    if violations:
        raise RuntimeError("History release violation")
    return dict(trend=trend, event=event, features=features, feature_names=list(FEATURE_NAMES),
        event_record_indices=event_ids, event_source_h=source_h, event_target_h=target_h,
        event_proposed_target_h=proposed_h,
        event_distances=event_distance, event_weights=event_weight, event_centers=selected_centers,
        event_widths=selected_widths, event_amplitudes=selected_amplitudes, trend_ids=trend_ids,
        encoded_at=encoded_at,
        audit=dict(queries=n, history_start=HISTORY_START, fixed_geometry=bool(fixed), shift=bool(shift),
            encoded_records=len(encoded), encoder_calls=calls, open_queries=open_queries,
            historical_event_records=peak_candidates, no_event_queries=int(np.sum(~valid.any(axis=1))),
            positive_payload_event_records=int(np.sum(atom_amplitude > 0)),
            nonzero_event_fraction=float(np.mean(event > 1e-9)) if n else 0.,
            moved_event_fraction=float(np.mean(source_h[valid] != target_h[valid])) if valid.any() else 0.,
            maturity_violations=violations, max_release_violation_seconds=0,
            minimum_release_margin_seconds=int(margins.min()) if len(margins) else None,
            selected_event_references=int(valid.sum()), selected_trend_references=int(selected_trend.sum()),
            trend_candidate_sha256=hashlib.sha256(trend_ids.tobytes()).hexdigest(),
            event_selection_samples=selection_samples,
            event_eligibility="released actual peak>300 and actual peak minus frozen anchor q50>0; independent of learned geometry",
            event_amplitude_cap="entire pulse amplitude <= missed peak minus positive historical trend error; nonmatching atom yields zero payload, candidate retained",
            residual_definition="encoded mature trend minus smoothed historical anchor q50; missed positive atom only",
            feature_scaling="z bounded; prices/300; corrections/150; distances/4; age/365; counts/365 or4; movement/23"))
