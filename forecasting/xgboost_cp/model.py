"""xgboost_cp: symmetric conformal intervals on the frozen aligned XGBoost.

Identical public calibration protocol as lightgbm_cp with the frozen XGBoost
base; both call the same shared tools, so the CP formula exists exactly once.
"""
from __future__ import annotations

from ..common.frozen_base import conformal_expansion

EXPECTED_PARAMS = {"calibration_formula": "absolute_residual_type7_per_horizon",
                   "centre": "frozen_base_rearranged_q50_unchanged"}


def run_expansion(context):
    return conformal_expansion(context, base_model="xgboost", expected_params=EXPECTED_PARAMS)
