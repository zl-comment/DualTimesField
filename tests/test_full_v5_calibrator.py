"""Calibrator LP tests: zero-param fallback, bounds, solver constraints, and
hand-verified scoring identities (spec section 9 items 6, 9)."""
import numpy as np
import pytest

from forecasting.full_v5 import calibrator, controls, diagnostics, schema


def test_zero_parameters_reproduce_baseline_exactly():
    rng = np.random.default_rng(0)
    n = 50
    bq = np.sort(rng.normal(100, 30, size=(n, 5)), axis=-1)
    x = rng.uniform(-1, 1, size=(n, 4))
    u = rng.uniform(-1, 1, size=(n, 3))
    out = calibrator.apply_params(np.zeros(4), np.zeros(3), x, u, bq)
    assert np.array_equal(out["q"], bq)
    assert np.all(out["delta"] == 0.0) and np.all(out["s"] == 1.0)


def test_bounds_and_quantile_monotonicity():
    rng = np.random.default_rng(1)
    n = 500
    x = rng.uniform(-1, 1, size=(n, 26))
    u = rng.uniform(-1, 1, size=(n, 5))
    bq = np.sort(rng.normal(80, 40, size=(n, 5)), axis=-1)
    a = np.zeros(26); a[3] = 0.6; a[7] = -0.4          # ||a||_1 == 1
    c = np.zeros(5); c[1] = 0.3; c[4] = -0.2            # ||c||_1 == 0.5
    out = calibrator.apply_params(a, c, x, u, bq)
    assert np.all(np.abs(out["delta"]) <= 150.0 + 1e-9)
    assert np.all(out["s"] >= 0.5 - 1e-9) and np.all(out["s"] <= 1.5 + 1e-9)
    assert np.all(np.diff(out["q"], axis=-1) >= -1e-9)  # positive scale keeps order


def test_lp_solver_constraints_and_zero_param_guarantee():
    rng = np.random.default_rng(2)
    n = 120
    bq = np.sort(rng.normal(90, 35, size=(n, 5)), axis=-1)
    x = rng.uniform(-1, 1, size=(n, 6))
    u = rng.uniform(-1, 1, size=(n, 4))
    y = bq[:, 2] + 25 * x[:, 0] + rng.normal(0, 5, size=n)
    solution = calibrator.solve_group_lp(x, u, bq, y, n_rows_total=n)
    assert solution["status"] == 0 and not solution["numerical_failure"]
    assert solution["max_equality_residual"] <= schema.CONSTRAINT_TOLERANCE
    assert solution["l1_a"] <= schema.A_L1_MAX + 1e-9
    assert solution["l1_c"] <= schema.C_L1_MAX + 1e-9
    assert solution["objective"] <= solution["objective_zero_parameters"] + 1e-6


def test_lp_matches_bruteforce_on_hand_example():
    # x = 1, u = 1, degenerate (zero-width) baseline: pure constant shift, the
    # scale variable has no effect, so the LP is one-dimensional in delta.
    y = np.array([10.0, 12.0, -4.0, 30.0, 5.0, 6.0, 7.0, -2.0, 11.0, 3.0])
    bq = np.repeat(y[:, None], 5, axis=1)  # zero spread: c drops out
    x = np.ones((10, 1)); u = np.ones((10, 1))
    solution = calibrator.solve_group_lp(x, u, bq, y, n_rows_total=10)
    delta = float(solution["a"][0]) * schema.DELTA_SCALE
    weights = np.asarray(schema.LOSS_WEIGHTS)
    quantiles = np.asarray(schema.QUANTILE_LEVELS)
    reg = 10 * schema.LP_REG / 4.0
    grid = np.linspace(-150, 150, 30001)
    values = []
    for shift in grid:
        residual = (y[:, None] - (bq[:, 2:3] + shift + np.zeros_like(bq))) / schema.PRICE_SCALE
        pinball = np.maximum(quantiles * residual, (quantiles - 1.0) * residual)
        values.append(float(np.sum(weights * pinball)) + reg * abs(shift) / schema.DELTA_SCALE)
    best = grid[int(np.argmin(values))]
    assert abs(delta - best) < 3 * 150 / 30000  # within three grid steps
    assert abs(solution["c"]).max() < 1e-6      # scale must stay at zero effect


def test_calibrator_fit_recovers_known_state_bias():
    rng = np.random.default_rng(4)
    n = 400
    bq = np.sort(rng.normal(80, 20, size=(n, 24, 5)), axis=-1)
    z = rng.uniform(-1, 1, size=(n, 24, schema.NZ)); z[:, :, 0] = 1.0
    bias = 40.0 * z[:, :, 6]
    actual = bq[:, :, 2] + bias
    features = np.concatenate([z, np.zeros((n, 24, schema.NM + schema.NV))], axis=2)
    rows = np.ones(n, dtype=bool)
    model = calibrator.fit_calibrator("C_state", schema.CONTROL_SPECS, features, bq, actual, rows)
    prediction = model.predict(features, bq)
    assert np.abs(prediction["delta"] - bias).max() < 2.0
    assert float(np.abs(np.sort(prediction["quantiles"], axis=-1)[..., 2] - actual).mean()) < 2.0
    assert model.diagnostics["group_0"]["l1_a"] <= 1.0 + 1e-6


def test_lp_nontrivial_with_spread_matches_bruteforce():
    # Non-degenerate baseline spread and two active parameter blocks: the LP
    # must reach the global optimum of the true (shift, slope, scale) problem,
    # which a zero-solution check cannot detect.
    rng = np.random.default_rng(23)
    n = 80
    bq = np.sort(rng.normal(90, 25, size=(n, 5)), axis=-1)      # real spread
    x = np.stack([np.ones(n), rng.uniform(-1, 1, n)], axis=1)   # shift + slope
    u = rng.uniform(0.2, 1.0, size=(n, 1))
    y = (bq[:, 2] + 30.0 * x[:, 1] + 8.0 * (bq[:, 3] - bq[:, 2]) * u[:, 0]
         + rng.normal(0, 4, n))
    solution = calibrator.solve_group_lp(x, u, bq, y, n_rows_total=n)
    assert solution["status"] == 0 and not solution["numerical_failure"]
    assert solution["objective"] < solution["objective_zero_parameters"] - 1e-3

    weights = np.asarray(schema.LOSS_WEIGHTS)
    quantiles = np.asarray(schema.QUANTILE_LEVELS)
    reg = n * schema.LP_REG / 4.0

    def objective(a0, a1, c):
        delta = schema.DELTA_SCALE * (a0 * x[:, 0] + a1 * x[:, 1])
        s = 1.0 + c * u[:, 0]
        q = bq[:, 2:3] + delta[:, None] + s[:, None] * (bq - bq[:, 2:3])
        residual = (y[:, None] - q) / schema.PRICE_SCALE
        pinball = np.maximum(quantiles * residual, (quantiles - 1.0) * residual)
        return float(np.sum(weights * pinball)) + reg * (abs(a0) + abs(a1) + abs(c))

    grid = np.linspace(-1.0, 1.0, 41)
    c_grid = np.linspace(-0.5, 0.5, 21)
    best = np.inf
    for a0 in grid:
        for a1 in grid:
            if abs(a0) + abs(a1) > 1.0 + 1e-12:
                continue
            for c in c_grid:
                best = min(best, objective(a0, a1, c))
    assert solution["objective"] <= best + 1e-6          # LP at least as good as grid
    assert best - solution["objective"] <= 0.02 * max(1.0, abs(best))  # grid near-optimal


def test_control_parameter_counts_match_spec():
    counts = {name: 4 * (len(spec["x"]) + len(spec["u"]))
              for name, spec in schema.CONTROL_SPECS.items()}
    assert counts == {"C_const": 8, "C_state": 76, "C_roll": 112, "C_recent": 124, "V5_history": 124}
    assert len(schema.Z_COLUMNS) == 16 and len(schema.M_COLUMNS) == 8 and len(schema.V_COLUMNS) == 2


def test_stacker_bounds_and_zero_weight_is_baseline():
    rng = np.random.default_rng(6)
    n = 200
    bq = np.sort(rng.normal(80, 20, size=(n, 24, 5)), axis=-1)
    v4 = np.sort(rng.normal(80, 20, size=(n, 24, 5)), axis=-1)
    actual = bq[:, :, 2] + 0.5 * (v4[:, :, 2] - bq[:, :, 2]) + rng.normal(0, 3, size=(n, 24))
    model = calibrator.fit_stacker(bq, v4, actual, np.ones(n, dtype=bool))
    mixed = model.predict(bq, v4)
    zero = calibrator.StackerModel({g: 0.0 for g in range(4)}, {})
    assert np.array_equal(zero.predict(bq, v4), bq)
    assert all(0.0 <= w <= 1.0 for w in model.weights.values())
    q50_mixed = np.sort(mixed, axis=-1)[..., 2]
    mae_mixed = float(np.abs(q50_mixed - actual).mean())
    mae_b = float(np.abs(bq[..., 2] - actual).mean())
    assert mae_mixed < mae_b


def test_band_decomposition_and_rmse_hand_example():
    actual = np.array([[10.0, 320.0], [-5.0, 100.0]])
    q50 = np.array([[12.0, 300.0], [1.0, 90.0]])
    bands = diagnostics.band_decomposition(q50, actual)
    assert abs(bands["mae_total"] - (2 + 20 + 6 + 10) / 4) < 1e-12
    assert abs(bands["negative"]["conditional_mae"] - 6.0) < 1e-12
    assert abs(bands["above_300"]["conditional_mae"] - 20.0) < 1e-12
    assert abs(bands["range_0_300"]["conditional_mae"] - (2 + 10) / 2) < 1e-12
    assert abs(bands["contribution_sum"] - bands["mae_total"]) < 1e-12
    # RMSE through the real implementations (not a re-derivation): horizon
    # groups, per-region macro accumulator and the pooled accumulator.
    # Quantile rows are built monotone (the implementations sort internally).
    def q_block(q50_24):
        return np.stack([q50_24 - 6.0, q50_24 - 3.0, q50_24, q50_24 + 3.0, q50_24 + 6.0], axis=-1)[None]

    y24 = np.zeros(24)
    q50_a = np.zeros(24)
    q50_a[0:6] = 3.0              # group 0 errors: +3
    q50_a[6:12] = -3.0            # group 1 errors: -3
    q_a = q_block(q50_a)
    groups = diagnostics.horizon_group_metrics(q_a, y24[None])
    assert abs(groups["group_0_h0_5"]["rmse"] - 3.0) < 1e-12
    assert abs(groups["group_1_h6_11"]["rmse"] - 3.0) < 1e-12
    assert abs(groups["group_2_h12_17"]["rmse"] - 0.0) < 1e-12
    q_b = q_block(np.full(24, 4.0))   # every error +4
    accumulator = {}
    rmse_a = diagnostics.macro_and_pooled_rmse(q_a, y24[None], "A", accumulator)
    rmse_b = diagnostics.macro_and_pooled_rmse(q_b, y24[None], "B", accumulator)
    assert abs(rmse_a["rmse_region"] - np.sqrt(4.5)) < 1e-12
    assert abs(rmse_b["rmse_region"] - 4.0) < 1e-12
    assert abs(diagnostics.pooled_rmse(accumulator) - np.sqrt((108.0 + 384.0) / 48)) < 1e-12
    macro = float(np.mean([rmse_a["rmse_region"], rmse_b["rmse_region"]]))
    assert abs(macro - (np.sqrt(4.5) + 4.0) / 2) < 1e-12


def test_coverage_hand_example():
    actual = np.array([[0.5, 1.0, 2.0], [0.0, 1.0, 5.0]])
    q = np.stack([np.array([-1.0, -0.5, 0.0, 0.5, 1.0])] * 6).reshape(2, 3, 5)
    out = diagnostics.coverage_diagnostics(q, actual)
    inside80 = ((actual >= q[..., 1]) & (actual <= q[..., 3]))
    assert abs(out["coverage_80"] - inside80.mean()) < 1e-12
    assert abs(out["interval_width_80"] - 1.0) < 1e-12


def test_holm_adjustment_hand_example():
    adjusted = diagnostics.holm_adjust([0.01, 0.02, 0.03, 0.20])
    # Hand-computed Holm step-down: m=4; 4*0.01=0.04; 3*0.02=0.06;
    # 2*0.03=0.06 (running max); 1*0.20=0.20.
    assert adjusted == [0.04, 0.06, 0.06, 0.20]
    # A family with monotonicity enforcement across equal ranks.
    assert diagnostics.holm_adjust([0.5, 0.001]) == [0.5, 0.002]


def test_paired_bootstrap_constant_difference_is_exact():
    # Round-4 M3: a constant ref-cand difference must reproduce exactly in
    # every draw, including a 1-origin tail block (169 origins, 168h blocks).
    origins = np.arange(0, 169 * 3600, 3600, dtype=np.int64)
    losses_ref = {"R": np.stack([np.full(169, 20.0), np.full(169, 30.0)], axis=1)}
    losses_cand = {"R": np.stack([np.full(169, 10.0), np.full(169, 20.0)], axis=1)}
    out = diagnostics.paired_block_bootstrap({"R": origins}, losses_ref, losses_cand,
                                             block_hours=168, n_boot=500, seed=7)
    assert out["n_blocks"] == 2                       # 168 + 1 tail
    assert out["point_difference"]["mae"] == 10.0     # positive = candidate better
    assert out["ci95"]["mae"] == [10.0, 10.0]
    assert out["ci95"]["qs"] == [10.0, 10.0]
    assert out["p_value"]["mae"] == 0.0


def test_paired_bootstrap_tail_renormalisation_matches_direct_draws():
    # Short tail block: per-draw normalisation by the sampled block counts.
    origins = np.arange(0, 169 * 3600, 3600, dtype=np.int64)
    diff = np.concatenate([np.full(168, 2.0), np.full(1, 100.0)])
    losses_ref = {"R": np.stack([diff, diff], axis=1)}
    losses_cand = {"R": np.zeros((169, 2))}
    out = diagnostics.paired_block_bootstrap({"R": origins}, losses_ref, losses_cand,
                                             block_hours=168, n_boot=400, seed=2026)
    # Independent replication of the intended semantics.
    block_means = np.array([2.0, 100.0])
    block_counts = np.array([168.0, 1.0])
    rng = np.random.default_rng(2026)
    draws = rng.integers(0, 2, size=(400, 2))
    counts = block_counts[draws]
    weights = counts / counts.sum(axis=1, keepdims=True)
    means = (block_means[draws] * weights).sum(axis=1)
    expected_ci = np.quantile(means, [0.025, 0.975])
    assert abs(out["point_difference"]["mae"] - (168 * 2.0 + 100.0) / 169) < 1e-12
    assert abs(out["ci95"]["mae"][0] - expected_ci[0]) < 1e-12
    assert abs(out["ci95"]["mae"][1] - expected_ci[1]) < 1e-12


def test_tree_feature_matrix_layout():
    rng = np.random.default_rng(8)
    n = 7
    flat = rng.normal(size=(n, 30)).astype(np.float32)
    features = rng.normal(size=(n, 24, schema.NF)).astype(np.float32)
    matrix = controls.tree_feature_matrix(flat, features, with_v=True)
    assert matrix.shape == (n * 24, 30 + schema.NF + 1 + 4)
    assert np.allclose(matrix[0, :30], flat[0])
    assert np.allclose(matrix[0, 30:30 + schema.NF], features[0, 0])
    assert matrix[0, -5] == 0.0                      # h/23 for h = 0
    assert matrix[n, -5] == pytest.approx(1 / 23)    # h = 1 starts at row n
    assert matrix[0, -4] == 1.0 and matrix[n, -4] == 1.0  # group one-hot
    assert matrix[18 * n, -1] == 1.0                 # h = 18 -> group 3
    # All four one-hot columns are binary and every row sums to exactly one.
    onehot = matrix[:, -4:]
    assert np.isin(onehot, (0.0, 1.0)).all()
    assert np.array_equal(onehot.sum(axis=1), np.ones(n * 24))
    for h in range(24):
        assert onehot[h * n, schema.horizon_group_of(h)] == 1.0


def test_tree_onehot_columns_fully_initialised_poisoned_memory(monkeypatch):
    """Round-4 H1 regression: one-hot columns must be deterministically zeroed.

    np.empty is poisoned with a sentinel so any uninitialised cell survives
    into the output; with_v True and False both checked.
    """
    real_empty = np.empty

    def poisoned_empty(*args, **kwargs):
        array = real_empty(*args, **kwargs)
        try:
            array.fill(37.0)
        except (TypeError, ValueError):
            pass
        return array

    monkeypatch.setattr(np, "empty", poisoned_empty)
    rng = np.random.default_rng(21)
    n = 11
    flat = rng.normal(size=(n, 5)).astype(np.float32)
    features = rng.normal(size=(n, 24, schema.NF)).astype(np.float32)
    for with_v in (True, False):
        matrix = controls.tree_feature_matrix(flat, features, with_v=with_v)
        n_feat = schema.NZ + schema.NM + (schema.NV if with_v else 0)
        onehot = matrix[:, flat.shape[1] + n_feat + 1:]
        assert onehot.shape == (n * 24, 4)
        assert np.isfinite(matrix).all()
        assert np.isin(onehot, (0.0, 1.0)).all(), "non-binary one-hot cell (memory leak)"
        assert np.array_equal(onehot.sum(axis=1), np.ones(n * 24))
        # exactly the active group column is set per horizon row
        for h in range(schema.HORIZONS):
            row = h * n
            active = schema.horizon_group_of(h)
            assert onehot[row, active] == 1.0
            assert onehot[row].sum() == 1.0
