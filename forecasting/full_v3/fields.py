"""Deterministic cosine-background and sparse Gaussian-event decomposition.

This is a convex fixed-basis realization of the dual-field idea. It is NOT the
old Full's learned neural INR or its learned Gabor model. Gaussian atoms here
are zero-carrier Gabor atoms; their signed coefficients can describe positive
and negative events. All operations act independently on individual curves.

For each raw 24-hour curve ``y``, let ``m = mean(y)``,
``s = max(sqrt(mean((y - m)**2)), 1)`` and ``z = (y - m) / s``. The exact
normalized-space objective is

    0.5 * ||z - B a - G b||_2**2
    + 0.5 * ridge * sum(k**4 * a[k]**2)
    + l1 * ||b||_1.

``B`` consists of orthonormal DCT-II cosine columns, including the constant
column k=0. ``G`` consists of unit-L2 Gaussian columns centered at each hour,
with the configured widths measured in hours. Thus the constant coefficient is
not penalized, and curvature penalizes progressively higher cosine frequencies.
The factor pi**4 of a continuous curvature integral is absorbed into ``ridge``.

One fixed iteration exactly minimizes over ``a`` given ``b``, then takes one
proximal-gradient step over ``b`` with step 1 / ||G||_2**2. A final exact ``a``
update follows the configured iterations. There is no convergence-based early
exit, curve-dependent iteration count, learned dictionary, or pooled statistic.
The finite-iteration output is an approximation to the convex minimizer, not a
claim of exact global convergence. Returned raw fields are
``background = m + s * B a``, ``event = s * G b`` and
``residual = y - background - event``.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


DEFAULT_CONFIG = {
    "basis_count": 4,
    "widths": (0.75, 1.5, 3.0),
    "ridge": 0.01,
    "l1": 0.03,
    "iterations": 40,
    "batch_size": 2048,
}


def _configuration(config):
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("field config must be a mapping")
    supplied = dict(config or {})
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown field configuration keys: {sorted(unknown)}")
    settings = dict(DEFAULT_CONFIG, **supplied)
    for key in ("basis_count", "iterations", "batch_size"):
        value = settings[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        settings[key] = int(value)
    if settings["basis_count"] > 24:
        raise ValueError("basis_count cannot exceed the 24-hour horizon")
    for key in ("ridge", "l1"):
        value = settings[key]
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            raise ValueError(f"{key} must be finite and nonnegative")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be finite and nonnegative") from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        settings[key] = value
    try:
        widths = np.asarray(settings["widths"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("widths must be a nonempty sequence of finite positive hours") from exc
    if widths.ndim != 1 or not len(widths) or not np.isfinite(widths).all() or (widths <= 0).any():
        raise ValueError("widths must be a nonempty sequence of finite positive hours")
    if len(np.unique(widths)) != len(widths):
        raise ValueError("widths must not contain duplicate atoms")
    settings["widths"] = tuple(float(value) for value in widths)
    return settings


def _fixed_bases(horizon, basis_count, widths):
    hours = np.arange(horizon, dtype=np.float64)
    frequency = np.arange(basis_count, dtype=np.float64)
    background = np.sqrt(2.0 / horizon) * np.cos(
        np.pi * (hours[:, None] + 0.5) * frequency[None, :] / horizon)
    background[:, 0] = 1.0 / np.sqrt(horizon)
    # Center-major column ordering: all widths for hour 0, then hour 1, etc.
    centers = np.repeat(hours, len(widths))
    atom_widths = np.tile(np.asarray(widths), horizon)
    events = np.exp(-0.5 * ((hours[:, None] - centers[None, :]) / atom_widths[None, :]) ** 2)
    events /= np.sqrt(np.sum(events ** 2, axis=0, keepdims=True))
    lipschitz = float(np.linalg.eigvalsh(events @ events.T)[-1])
    return background, events, frequency ** 4, lipschitz


def _synthesize(coefficients, basis):
    # Explicit contraction order is independent of batch shape. BLAS GEMM can
    # select a different reduction kernel for differently sized row batches.
    return np.einsum("nk,hk->nh", coefficients, basis, optimize=False)


def _project(values, basis):
    return np.einsum("nh,hk->nk", values, basis, optimize=False)


def decompose_curves(values, *, config=None):
    """Decompose nonempty raw-price ``[N, 24]`` curves without modifying inputs.

    Returns float64 ``background``, ``event`` and ``residual`` arrays, plus a
    diagnostics mapping. The diagnostics objective is in normalized units;
    ``local_mean`` and ``local_scale`` permit auditing the raw-space mapping.
    The same curve produces identical results alone, reordered or in batches.
    """
    settings = _configuration(config)
    source = np.asarray(values)
    if source.ndim != 2 or source.shape[1] != 24 or source.shape[0] == 0:
        raise ValueError("values must have nonempty shape [N, 24]")
    if source.dtype.kind not in "iuf" or not np.isfinite(source).all():
        raise ValueError("values must contain finite real prices")
    y = np.array(source, dtype=np.float64, copy=True)
    basis, dictionary, curvature, lipschitz = _fixed_bases(24, settings["basis_count"], settings["widths"])
    denominator = 1.0 + settings["ridge"] * curvature
    step = 1.0 / lipschitz
    threshold = settings["l1"] * step
    background, event = np.empty_like(y), np.empty_like(y)
    means, scales = np.empty(len(y)), np.empty(len(y))
    objectives, initial_objectives, smooth_cost, sparse_cost, data_cost = (
        np.empty(len(y)) for _ in range(5))
    active_counts = np.empty(len(y), dtype=np.int64)
    coefficients_c = np.empty((len(y), settings["basis_count"]))
    coefficients_d = np.empty((len(y), dictionary.shape[1]))
    for start in range(0, len(y), settings["batch_size"]):
        stop = min(start + settings["batch_size"], len(y))
        local = y[start:stop]
        mean = local.mean(axis=1, keepdims=True)
        centered = local - mean
        # Price-domain inputs are far from overflow; reject overflow rather than
        # silently creating zeros or nonfinite fields for unsupported magnitudes.
        with np.errstate(over="ignore", invalid="ignore"):
            scale = np.maximum(np.sqrt(np.mean(centered ** 2, axis=1, keepdims=True)), 1.0)
        if not np.isfinite(scale).all():
            raise ValueError("prices exceed finite float64 normalization range")
        z = centered / scale
        b = np.zeros((stop - start, dictionary.shape[1]), dtype=np.float64)
        d = np.zeros_like(z)
        initial_objectives[start:stop] = 0.5 * np.sum(z ** 2, axis=1)
        for _ in range(settings["iterations"]):
            a = _project(z - d, basis) / denominator
            c = _synthesize(a, basis)
            proposal = b + step * _project(z - c - d, dictionary)
            b = np.sign(proposal) * np.maximum(np.abs(proposal) - threshold, 0.0)
            d = _synthesize(b, dictionary)
        a = _project(z - d, basis) / denominator
        c = _synthesize(a, basis)
        background[start:stop] = mean + scale * c
        event[start:stop] = scale * d
        means[start:stop], scales[start:stop] = mean[:, 0], scale[:, 0]
        coefficients_c[start:stop], coefficients_d[start:stop] = a, b
        data_cost[start:stop] = 0.5 * np.sum((z - c - d) ** 2, axis=1)
        smooth_cost[start:stop] = 0.5 * settings["ridge"] * np.sum(curvature * a ** 2, axis=1)
        sparse_cost[start:stop] = settings["l1"] * np.sum(np.abs(b), axis=1)
        objectives[start:stop] = data_cost[start:stop] + smooth_cost[start:stop] + sparse_cost[start:stop]
        active_counts[start:stop] = np.count_nonzero(b, axis=1)
    residual = y - background - event
    diagnostics = {
        "objective": objectives, "initial_objective": initial_objectives,
        "data_cost": data_cost, "curvature_cost": smooth_cost, "l1_cost": sparse_cost,
        "local_mean": means, "local_scale": scales,
        "background_coefficients": coefficients_c, "event_coefficients": coefficients_d,
        "active_event_atoms": active_counts,
        "reconstruction_max_abs": np.max(np.abs(y - (background + event + residual)), axis=1),
        "iterations": settings["iterations"], "lipschitz": lipschitz,
        "basis_count": settings["basis_count"], "widths": settings["widths"],
        "ridge": settings["ridge"], "l1": settings["l1"],
        "event_atom_order": "center_major_then_width", "normalization_floor": 1.0,
    }
    if not all(np.isfinite(part).all() for part in (background, event, residual, objectives)):
        raise FloatingPointError("Nonfinite curve decomposition")
    return {"background": background, "event": event, "residual": residual, "diagnostics": diagnostics}
