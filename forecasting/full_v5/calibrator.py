"""Bounded affine location/scale calibrator solved as sparse linear programs.

Spec section 5: delta = 150 a^T x, s = 1 + c^T u, Q = B_q50 + delta +
s (B_q - B_q50), with ||a_g||_1 <= 1 and ||c_g||_1 <= 0.5 per horizon group.
The pinball objective with auxiliary residuals is an LP; each (region, group)
is solved independently by SciPy HiGHS.  Zero parameters reproduce the frozen
baseline exactly.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from . import schema

QUANTILES = np.asarray(schema.QUANTILE_LEVELS, dtype=np.float64)
WEIGHTS = np.asarray(schema.LOSS_WEIGHTS, dtype=np.float64)


def apply_params(a: np.ndarray, c: np.ndarray, x: np.ndarray, u: np.ndarray,
                 baseline_q_sorted: np.ndarray) -> dict:
    """Apply one horizon-group's parameters to rows [n, 5] of sorted baseline
    quantiles with features x [n, nx], u [n, nu].  Returns Q and the parts."""
    delta = schema.DELTA_SCALE * x @ a
    s = 1.0 + u @ c
    bq = np.asarray(baseline_q_sorted, dtype=np.float64)
    q = bq[:, 2:3] + delta[:, None] + s[:, None] * (bq - bq[:, 2:3])
    if not np.isfinite(q).all():
        raise FloatingPointError("Non-finite calibrated quantiles")
    return dict(q=q, delta=delta, s=s)


class CalibratorModel:
    """Per-region parameters: {group -> (a, c)} plus solver diagnostics."""

    def __init__(self, model_id: str, spec: dict, groups: dict, params: dict, diagnostics: dict):
        self.model_id = model_id
        self.spec = spec
        self.groups = groups          # horizon -> group index
        self.params = params          # group -> (a, c)
        self.diagnostics = diagnostics

    def predict(self, features: np.ndarray, baseline_q_sorted: np.ndarray) -> dict:
        """features: [N, 24, 26] full block; baseline: [N, 24, 5] sorted."""
        features = np.asarray(features, dtype=np.float64)
        bq = np.asarray(baseline_q_sorted, dtype=np.float64)
        n = len(features)
        q = np.empty((n, schema.HORIZONS, 5), dtype=np.float64)
        delta = np.empty((n, schema.HORIZONS), dtype=np.float64)
        s = np.empty((n, schema.HORIZONS), dtype=np.float64)
        x_idx = list(self.spec["x"])
        u_idx = list(self.spec["u"])
        for h in range(schema.HORIZONS):
            a, c = self.params[self.groups[h]]
            out = apply_params(a, c, features[:, h][:, x_idx], features[:, h][:, u_idx], bq[:, h])
            q[:, h], delta[:, h], s[:, h] = out["q"], out["delta"], out["s"]
        return dict(quantiles=q, delta=delta, s=s)

    def to_json(self) -> dict:
        return dict(model_id=self.model_id,
                    params={str(g): [a.tolist(), c.tolist()] for g, (a, c) in self.params.items()},
                    diagnostics=self.diagnostics)

    @classmethod
    def from_json(cls, payload: dict, specs: dict):
        spec = specs[payload["model_id"]]
        groups = {h: schema.horizon_group_of(h) for h in range(schema.HORIZONS)}
        params = {int(g): (np.asarray(v[0]), np.asarray(v[1])) for g, v in payload["params"].items()}
        return cls(payload["model_id"], spec, groups, params, payload["diagnostics"])


def zero_objective(baseline_q_sorted: np.ndarray, actual: np.ndarray, rows: np.ndarray) -> float:
    """Objective value of the zero-parameter feasible point (Q = B)."""
    bq = np.asarray(baseline_q_sorted, dtype=np.float64)[rows]
    y = np.asarray(actual, dtype=np.float64)[rows]
    residual = (y[:, :, None] - bq) / schema.PRICE_SCALE
    pinball = np.maximum(QUANTILES * residual, (QUANTILES - 1.0) * residual)
    return float(np.sum(WEIGHTS * pinball))


def _pinball_objective(q: np.ndarray, y: np.ndarray) -> float:
    residual = (np.asarray(y, dtype=np.float64)[:, None] - q) / schema.PRICE_SCALE
    pinball = np.maximum(QUANTILES * residual, (QUANTILES - 1.0) * residual)
    return float(np.sum(WEIGHTS * pinball))


def solve_group_lp(x: np.ndarray, u: np.ndarray, baseline: np.ndarray, y: np.ndarray,
                   n_rows_total: int) -> dict:
    """Solve one horizon group's LP.

    Variables: [a+(nx), a-(nx), c+(nu), c-(nu), e+(n,5), e-(n,5)] >= 0.
    Equalities per (row, q): 150 x a + (B_q - B_q50) u c + 300 (e+ - e-) = Y - B_q.
    Inequalities: |a|_1 <= 1, |c|_1 <= 0.5.  Objective: weighted pinball of the
    scaled residuals plus (n_rows_total * 1e-3 / 4) * (|a|_1 + |c|_1), i.e. the
    spec's mean loss plus (1e-3/4) sum_g(...) scaled by the total row count.
    """
    n, nx = x.shape
    nu = u.shape[1]
    bq = np.asarray(baseline, dtype=np.float64)
    spread = bq - bq[:, 2:3]                      # [n, 5]
    reg = n_rows_total * schema.LP_REG / 4.0

    nvar = 2 * nx + 2 * nu + 10 * n
    off_ap, off_am = 0, nx
    off_cp, off_cm = 2 * nx, 2 * nx + nu
    off_ep = 2 * nx + 2 * nu
    off_em = off_ep + 5 * n

    rows_i, cols, vals = [], [], []
    rhs = np.empty(5 * n)
    for q in range(5):
        block = slice(q * n, (q + 1) * n)
        idx = np.arange(n)
        for d in range(nx):
            rows_i.append(idx + block.start); cols.append(np.full(n, off_ap + d)); vals.append(schema.DELTA_SCALE * x[:, d])
            rows_i.append(idx + block.start); cols.append(np.full(n, off_am + d)); vals.append(-schema.DELTA_SCALE * x[:, d])
        for d in range(nu):
            coef = spread[:, q] * u[:, d]
            rows_i.append(idx + block.start); cols.append(np.full(n, off_cp + d)); vals.append(coef)
            rows_i.append(idx + block.start); cols.append(np.full(n, off_cm + d)); vals.append(-coef)
        rows_i.append(idx + block.start); cols.append(off_ep + q * n + idx); vals.append(np.full(n, 300.0))
        rows_i.append(idx + block.start); cols.append(off_em + q * n + idx); vals.append(np.full(n, -300.0))
        rhs[block] = y - bq[:, q]

    a_eq = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows_i), np.concatenate(cols))),
                         shape=(5 * n, nvar))
    a_ub = sp.lil_matrix((2, nvar))
    a_ub[0, off_ap:off_cp] = 1.0
    a_ub[1, off_cp:off_ep] = 1.0
    b_ub = np.array([schema.A_L1_MAX, schema.C_L1_MAX])

    c_obj = np.zeros(nvar)
    c_obj[off_ap:off_ep] = reg
    for q in range(5):
        c_obj[off_ep + q * n: off_ep + (q + 1) * n] = WEIGHTS[q] * QUANTILES[q]
        c_obj[off_em + q * n: off_em + (q + 1) * n] = WEIGHTS[q] * (1.0 - QUANTILES[q])

    result = linprog(c_obj, A_ub=a_ub.tocsr(), b_ub=b_ub, A_eq=a_eq, b_eq=rhs,
                     bounds=(0.0, None), method="highs")
    if result.status != 0:
        return dict(status=result.status, message=str(result.message), numerical_failure=True,
                    solver="scipy.optimize.linprog[highs]")
    theta = result.x
    a = theta[off_ap:off_ap + nx] - theta[off_am:off_am + nx]
    c = theta[off_cp:off_cp + nu] - theta[off_cm:off_cm + nu]
    l1_a, l1_c = float(np.abs(a).sum()), float(np.abs(c).sum())
    eq_residual = float(np.max(np.abs(a_eq @ theta - rhs))) if n else 0.0
    obj_zero = _pinball_objective(bq, y)
    failure = (eq_residual > schema.CONSTRAINT_TOLERANCE
               or l1_a > schema.A_L1_MAX + schema.CONSTRAINT_TOLERANCE
               or l1_c > schema.C_L1_MAX + schema.CONSTRAINT_TOLERANCE
               or result.fun > obj_zero + schema.CONSTRAINT_TOLERANCE * max(1.0, abs(obj_zero)))
    return dict(status=0, message=str(result.message), a=a, c=c, objective=float(result.fun),
                objective_zero_parameters=obj_zero, l1_a=l1_a, l1_c=l1_c,
                max_equality_residual=eq_residual, numerical_failure=bool(failure),
                solver="scipy.optimize.linprog[highs]")


def fit_calibrator(model_id: str, specs: dict, features: np.ndarray, baseline_q_sorted: np.ndarray,
                   actual: np.ndarray, fit_rows: np.ndarray) -> CalibratorModel:
    """Fit one control over its Dfit (or H2) rows.  fit_rows: [n_fit] origin
    indices; the loss is the mean over all (t, h) fit rows of the whole run."""
    spec = specs[model_id]
    x_idx, u_idx = list(spec["x"]), list(spec["u"])
    features = np.asarray(features, dtype=np.float64)
    bq = np.asarray(baseline_q_sorted, dtype=np.float64)
    y = np.asarray(actual, dtype=np.float64)
    n_rows_total = int(np.sum(fit_rows)) * schema.HORIZONS
    params, diagnostics = {}, {}
    for g, (lo, hi) in enumerate(schema.HORIZON_GROUPS):
        hs = list(range(lo, hi + 1))
        # Rows are (origin, h) pairs of the fit set for this group.
        origins = np.flatnonzero(fit_rows)
        x = features[np.ix_(origins, hs)][:, :, x_idx].reshape(len(origins) * len(hs), len(x_idx))
        u = features[np.ix_(origins, hs)][:, :, u_idx].reshape(len(origins) * len(hs), len(u_idx))
        bg = bq[np.ix_(origins, hs)].reshape(len(origins) * len(hs), 5)
        yg = y[np.ix_(origins, hs)].reshape(len(origins) * len(hs))
        solution = solve_group_lp(x, u, bg, yg, n_rows_total)
        if solution.get("numerical_failure") or solution["status"] != 0:
            raise RuntimeError(f"{model_id} group {g} LP failed: {solution}")
        params[g] = (solution.pop("a"), solution.pop("c"))
        diagnostics[f"group_{g}"] = solution
    groups = {h: schema.horizon_group_of(h) for h in range(schema.HORIZONS)}
    diagnostics["n_fit_rows_total"] = n_rows_total
    return CalibratorModel(model_id, spec, groups, params, diagnostics)


class StackerModel:
    """A_stack: per (region, horizon-group) single convex weight on V4 Full."""

    def __init__(self, weights: dict, diagnostics: dict):
        self.weights = weights       # group -> w in [0, 1]
        self.diagnostics = diagnostics

    def predict(self, baseline_q_sorted: np.ndarray, v4_q_sorted: np.ndarray) -> np.ndarray:
        bq = np.asarray(baseline_q_sorted, dtype=np.float64)
        vq = np.asarray(v4_q_sorted, dtype=np.float64)
        out = np.empty_like(bq)
        for h in range(schema.HORIZONS):
            w = self.weights[schema.horizon_group_of(h)]
            out[:, h] = (1.0 - w) * bq[:, h] + w * vq[:, h]
        return out

    def to_json(self) -> dict:
        return dict(weights={str(g): float(w) for g, w in self.weights.items()},
                    diagnostics=self.diagnostics)

    @classmethod
    def from_json(cls, payload: dict):
        return cls({int(g): float(w) for g, w in payload["weights"].items()}, payload["diagnostics"])


def fit_stacker(baseline_q_sorted: np.ndarray, v4_q_sorted: np.ndarray, actual: np.ndarray,
                fit_rows: np.ndarray) -> StackerModel:
    bq = np.asarray(baseline_q_sorted, dtype=np.float64)
    vq = np.asarray(v4_q_sorted, dtype=np.float64)
    y = np.asarray(actual, dtype=np.float64)
    n_rows_total = int(np.sum(fit_rows)) * schema.HORIZONS
    reg = n_rows_total * schema.STACKER_REG / 4.0
    weights, diagnostics = {}, {}
    origins = np.flatnonzero(fit_rows)
    for g, (lo, hi) in enumerate(schema.HORIZON_GROUPS):
        hs = list(range(lo, hi + 1))
        bg = bq[np.ix_(origins, hs)].reshape(-1, 5)
        vg = vq[np.ix_(origins, hs)].reshape(-1, 5)
        yg = y[np.ix_(origins, hs)].reshape(-1)
        n = len(yg)
        diff = vg - bg
        # Variable layout: [w, e+(n,5), e-(n,5)].
        rows_i, cols, vals = [], [], []
        rhs = np.empty(5 * n)
        idx = np.arange(n)
        for q in range(5):
            block = idx + q * n
            rows_i.append(block); cols.append(np.zeros(n, dtype=int)); vals.append(diff[:, q])
            rows_i.append(block); cols.append(1 + q * n + idx); vals.append(np.full(n, 300.0))
            rows_i.append(block); cols.append(1 + 5 * n + q * n + idx); vals.append(np.full(n, -300.0))
            rhs[q * n:(q + 1) * n] = yg - bg[:, q]
        a_eq = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows_i), np.concatenate(cols))),
                             shape=(5 * n, 1 + 10 * n))
        c_obj = np.zeros(1 + 10 * n)
        c_obj[0] = reg
        for q in range(5):
            c_obj[1 + q * n: 1 + (q + 1) * n] = WEIGHTS[q] * QUANTILES[q]
            c_obj[1 + 5 * n + q * n: 1 + 5 * n + (q + 1) * n] = WEIGHTS[q] * (1.0 - QUANTILES[q])
        result = linprog(c_obj, A_eq=a_eq, b_eq=rhs, bounds=[(0.0, 1.0)] + [(0.0, None)] * (10 * n),
                         method="highs")
        if result.status != 0:
            raise RuntimeError(f"A_stack group {g} LP failed: {result.message}")
        w = float(result.x[0])
        residual = (yg[:, None] - ((1.0 - w) * bg + w * vg)) / schema.PRICE_SCALE
        pinball = np.maximum(QUANTILES * residual, (QUANTILES - 1.0) * residual)
        obj_zero = float(np.sum(WEIGHTS * np.maximum(
            QUANTILES * (yg[:, None] - bg) / schema.PRICE_SCALE,
            (QUANTILES - 1.0) * (yg[:, None] - bg) / schema.PRICE_SCALE)))
        if result.fun > obj_zero + schema.CONSTRAINT_TOLERANCE * max(1.0, abs(obj_zero)) or not 0.0 <= w <= 1.0:
            raise RuntimeError("A_stack numerical failure")
        weights[g] = w
        diagnostics[f"group_{g}"] = dict(objective=float(result.fun), objective_at_w_zero=obj_zero,
                                         weight=w, solver="scipy.optimize.linprog[highs]")
    return StackerModel(weights, diagnostics)
