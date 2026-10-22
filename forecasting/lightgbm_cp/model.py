"""lightgbm_cp: symmetric conformal intervals on the frozen aligned LightGBM.

Reuses the parent run's frozen boosters under the bitwise reproduction gate
and calibrates per-horizon interval widths on 2022H2 out-of-sample absolute
residuals. The centre stays exactly the frozen tree's rearranged q50. The
shared driver lives in ``forecasting/common/frozen_base.py``; the CP formula
itself is the already-audited ``common/calibration.absolute_residual_cp``.
"""
from __future__ import annotations

from ..common.frozen_base import conformal_expansion

EXPECTED_PARAMS = {"calibration_formula": "absolute_residual_type7_per_horizon",
                   "centre": "frozen_base_rearranged_q50_unchanged"}


def run_expansion(context):
    return conformal_expansion(context, base_model="lightgbm", expected_params=EXPECTED_PARAMS)
