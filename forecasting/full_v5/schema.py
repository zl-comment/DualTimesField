"""Frozen schema, segments and constants for V5 Calibrated History.

Every value here mirrors the round-2 specification; they are implementation
identity, not tunables.  Changing this file changes SCHEMA_HASH and makes any
existing run root refuse to resume.
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd

TIMEZONE = "Australia/Brisbane"
HORIZONS = 24
QUANTILE_LEVELS = (0.05, 0.10, 0.50, 0.90, 0.95)
HORIZON_GROUPS = ((0, 5), (6, 11), (12, 17), (18, 23))

# Query/fit windows must lie fully inside these AEST calendar segments.
SEGMENTS = {
    "dfit": ("2022-07-01", "2022-10-01"),
    "v1": ("2022-10-01", "2022-12-01"),
    "v2": ("2022-12-01", "2023-01-01"),
    "h2": ("2022-07-01", "2023-01-01"),
    "test": ("2023-01-01", "2025-01-01"),
}
# Labels consumed by fitting must be available strictly before these cut-offs.
FIT_CUTOFFS = {"dfit": "2022-10-01", "h2": "2023-01-01"}

PRICE_SCALE = 300.0
EXOGENOUS_CLIP = 3.0       # standardised exogenous columns: clip [-3,3] then /3
MATCH_K = 16               # neighbours for the state-matched history branch
MATCH_MIN_CANDIDATES = 8   # fewer -> history branch closed
LOOKBACK_DAYS = 365
WINDOW_SHORT_DAYS = 7
WINDOW_LONG_DAYS = 28
WINDOW_MIN_RECORDS = 24    # fewer mature records in a window -> statistics zeroed
COUNT_NORMALISER = {WINDOW_SHORT_DAYS: 168, WINDOW_LONG_DAYS: 672}
AVAILABILITY_LAG_HOURS = 1  # every hourly observation available delivery+1h

DELTA_SCALE = 150.0
S_MIN, S_MAX = 0.5, 1.5
A_L1_MAX = 1.0
C_L1_MAX = 0.5
LP_REG = 1e-3              # (1e-3 / 4) * sum_g (||a_g||_1 + ||c_g||_1)
EXTRA_Q50_WEIGHT = 0.2
CONSTRAINT_TOLERANCE = 1e-6
# (1/5) * sum_q pinball + 0.2 * extra q50 pinball -> per-quantile weights.
LOSS_WEIGHTS = (0.2, 0.2, 0.2 + EXTRA_Q50_WEIGHT, 0.2, 0.2)

STACKER_REG = 1e-3         # + 1e-3 * mean_g w_g shrinkage towards pure baseline

TREE_PARAMS = dict(
    rounds=200, learning_rate=0.05, num_leaves=15, max_depth=4,
    min_child_samples=200, max_bin=63, feature_fraction=1.0,
    bagging_fraction=1.0, seed=2026, threads=4,
)

# --- feature layout -------------------------------------------------------
# z: 16 columns, spec 4.1.  Sources: `baseline` columns come from the sorted
# frozen baseline quantiles; `future_exogenous`/`ctf_exogenous`/
# `quantile_exogenous` use the frozen training-standardised cache values
# (clip [-3,3] then /3); calendar columns derive from the target timestamp.
Z_COLUMNS = (
    ("const_one", "constant", "1"),
    ("baseline_q50_scaled", "baseline", "clip(B_q50/300, -1, 1)"),
    ("baseline_width80_scaled", "baseline", "clip((B_q90-B_q10)/300, 0, 1)"),
    ("predispatch_price", "future_exogenous", "clip([-3,3])/3 col predispatch_rrp"),
    ("predispatch_demand", "future_exogenous", "clip([-3,3])/3 col predispatch_total_demand_mw"),
    ("available_generation", "future_exogenous", "clip([-3,3])/3 col predispatch_available_generation_mw"),
    ("spare_capacity", "future_exogenous", "clip([-3,3])/3 col max_spare_capacity_mw"),
    ("net_load", "ctf_exogenous", "clip([-3,3])/3 col net_load_mw"),
    ("import_room", "future_exogenous", "clip([-3,3])/3 col ic_import_room_mw"),
    ("shortfall_q05", "quantile_exogenous", "clip([-3,3])/3 col max_spare_capacity_mw_shortfall_q0.05"),
    ("predispatch_coverage", "future_exogenous", "clip([-3,3])/3 col predispatch_covered"),
    ("interconnector_coverage", "future_exogenous", "clip([-3,3])/3 col ic_covered"),
    ("target_hour_sin", "calendar", "sin(2*pi*hour/24) of target delivery (AEST)"),
    ("target_hour_cos", "calendar", "cos(2*pi*hour/24)"),
    ("target_weekend", "calendar", "1 if AEST day-of-week >= 5"),
    ("target_dow_sin", "calendar", "sin(2*pi*dow/7)"),
)
NZ = len(Z_COLUMNS)

# m: 8 columns, spec 4.2 (same region, same lead, matured library records).
M_COLUMNS = (
    "resid_median_7d", "resid_median_28d", "abs_resid_median_7d",
    "abs_resid_median_28d", "coverage80_28d_minus_p8", "coverage90_28d_minus_p9",
    "count_7d_over_168", "count_28d_over_672",
)
NM = len(M_COLUMNS)

# v: 2 columns, spec 4.3 (matched-median minus 28d median, shrunk by eta).
V_COLUMNS = ("v_loc", "v_abs")
NV = len(V_COLUMNS)

# Distance variables for state matching: z column indices (spec 4.3):
# B_q50, 80% width, predispatch price, spare capacity, net load, import room.
MATCH_Z_INDICES = (1, 2, 3, 6, 7, 8)

# Full feature matrix layout F = [z (16) | m (8) | v (2)] -> column indices.
F_Z = 0
F_M = NZ
F_V = NZ + NM
NF = NZ + NM + NV

# Scale-vector u entries reference F columns: [1, z_width, z_spare, m_abs28, v_abs].
U_FULL = (F_Z + 0, F_Z + 2, F_Z + 6, F_M + 3, F_V + 1)

# Control feature sets (spec section 6).  Controls use the same Dfit fit rows,
# the same loss and the same bounded-affine machinery; they differ only in the
# feature columns they may consume.  `v_source` selects which history branch
# fills the v slot ("match", "recent" or None).  Parameter counts per region:
# C_const 8, C_state 76, C_roll 112, C_recent 124, V5_history 124 (4 groups).
CONTROL_SPECS = {
    "C_const": dict(x=(F_Z + 0,), u=(U_FULL[0],), v_source=None),
    "C_state": dict(x=tuple(F_Z + i for i in range(NZ)), u=(U_FULL[0], U_FULL[1], U_FULL[2]), v_source=None),
    "C_roll": dict(x=tuple(F_Z + i for i in range(NZ)) + tuple(F_M + i for i in range(NM)),
                   u=U_FULL[:4], v_source=None),
    "C_recent": dict(x=tuple(range(NF)), u=U_FULL, v_source="recent"),
    "V5_history": dict(x=tuple(range(NF)), u=U_FULL, v_source="match"),
}
CALIBRATOR_CONTROLS = tuple(CONTROL_SPECS)
TREE_CONTROLS = ("T_roll", "T_history")
STACKER = "A_stack"
BASELINE = "B"
ALL_MODELS = (BASELINE,) + CALIBRATOR_CONTROLS + TREE_CONTROLS + (STACKER,)

BANDS = (("negative", "actual < 0"), ("range_0_300", "0 <= actual <= 300"), ("above_300", "actual > 300"))

# Pre-declared primary comparisons for Holm-corrected inference (spec section 7).
PRIMARY_COMPARISONS = (
    ("V5_history", "C_roll"), ("V5_history", "C_recent"),
)
PRIMARY_METRICS = ("mae", "qs")


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def unix_timestamp(value: str) -> int:
    return int(pd.Timestamp(value, tz=TIMEZONE).timestamp())


def segment_bounds() -> dict[str, tuple[int, int]]:
    return {name: (unix_timestamp(a), unix_timestamp(b)) for name, (a, b) in SEGMENTS.items()}


def fit_cutoffs() -> dict[str, int]:
    return {name: unix_timestamp(value) for name, value in FIT_CUTOFFS.items()}


def horizon_group_of(h: int) -> int:
    for index, (lo, hi) in enumerate(HORIZON_GROUPS):
        if lo <= h <= hi:
            return index
    raise ValueError(f"Horizon {h} outside groups")


def validate_config(config: dict) -> None:
    exp = config["v5_calibrated_history"]
    if exp.get("protocol_id") != "v5-calibrated-history-20261015-r1":
        raise ValueError("Unexpected V5 protocol id")
    if int(exp["seed"]) != 2026:
        raise ValueError("V5 is locked to seed 2026")
    if tuple(exp["regions"]) not in (("NSW1", "QLD1", "TAS1"), ("NSW1",)):
        raise ValueError("Regions must be the frozen three-region set or the single-region smoke set")
    features = exp["features"]
    frozen = dict(price_scale=PRICE_SCALE, match_k=MATCH_K, match_min_candidates=MATCH_MIN_CANDIDATES,
                  lookback_days=LOOKBACK_DAYS, window_short_days=WINDOW_SHORT_DAYS,
                  window_long_days=WINDOW_LONG_DAYS, window_min_records=WINDOW_MIN_RECORDS,
                  exogenous_clip=EXOGENOUS_CLIP)
    for key, value in frozen.items():
        if float(features[key]) != float(value):
            raise ValueError(f"Feature constant {key} must stay at the frozen value {value}")
    cal = exp["calibrator"]
    frozen_cal = dict(delta_scale=DELTA_SCALE, s_min=S_MIN, s_max=S_MAX, a_l1_max=A_L1_MAX,
                      c_l1_max=C_L1_MAX, reg=LP_REG, extra_q50_weight=EXTRA_Q50_WEIGHT,
                      constraint_tolerance=CONSTRAINT_TOLERANCE)
    for key, value in frozen_cal.items():
        if float(cal[key]) != float(value):
            raise ValueError(f"Calibrator constant {key} must stay at the frozen value {value}")
    if [list(g) for g in exp["features"]["horizon_groups"]] != [list(g) for g in HORIZON_GROUPS]:
        raise ValueError("Horizon groups must stay at the frozen four-group split")
    gates = exp["gates"]
    if (float(gates["g1_macro_improvement"]) != 0.01 or float(gates["g1_region_mae_max_worsening"]) != 0.01
            or float(gates["g1_region_band0300_max_worsening"]) != 0.005
            or float(gates["g2_macro_improvement"]) != 0.005):
        raise ValueError("Gate thresholds must stay at the pre-declared values")
    if exp["baseline"]["mode"] not in ("frozen_lgbm", "precomputed"):
        raise ValueError("Unknown baseline mode")
    if exp["v4"]["mode"] not in ("frozen_artifact", "precomputed"):
        raise ValueError("Unknown V4 mode")
    trees = exp["trees"]
    for key, value in TREE_PARAMS.items():
        if float(trees[key]) != float(value):
            raise ValueError(f"Tree constant {key} must stay at the frozen value {value}")
    if int(exp["bootstrap"]["n"]) != 9999 or int(exp["bootstrap"]["block_hours"]) != 168:
        raise ValueError("Bootstrap must stay at 9999 draws of 168h blocks")


def _schema_payload() -> dict:
    return dict(
        segments={k: list(v) for k, v in SEGMENTS.items()},
        fit_cutoffs=dict(FIT_CUTOFFS),
        horizon_groups=[list(g) for g in HORIZON_GROUPS],
        quantile_levels=list(QUANTILE_LEVELS),
        z_columns=[list(c) for c in Z_COLUMNS],
        m_columns=list(M_COLUMNS),
        v_columns=list(V_COLUMNS),
        match_z_indices=list(MATCH_Z_INDICES),
        u_full=list(U_FULL),
        control_specs={k: {kk: list(vv) if isinstance(vv, tuple) else vv for kk, vv in v.items()}
                       for k, v in CONTROL_SPECS.items()},
        constants=dict(price_scale=PRICE_SCALE, exogenous_clip=EXOGENOUS_CLIP, match_k=MATCH_K,
                       match_min_candidates=MATCH_MIN_CANDIDATES, lookback_days=LOOKBACK_DAYS,
                       window_short_days=WINDOW_SHORT_DAYS, window_long_days=WINDOW_LONG_DAYS,
                       window_min_records=WINDOW_MIN_RECORDS, availability_lag_hours=AVAILABILITY_LAG_HOURS,
                       delta_scale=DELTA_SCALE, s_min=S_MIN, s_max=S_MAX, a_l1_max=A_L1_MAX,
                       c_l1_max=C_L1_MAX, lp_reg=LP_REG, extra_q50_weight=EXTRA_Q50_WEIGHT,
                       loss_weights=list(LOSS_WEIGHTS), stacker_reg=STACKER_REG),
        tree_params=dict(TREE_PARAMS),
        primary_comparisons=[list(c) for c in PRIMARY_COMPARISONS],
    )


SCHEMA_HASH = hashlib.sha256(_canonical(_schema_payload()).encode()).hexdigest()
