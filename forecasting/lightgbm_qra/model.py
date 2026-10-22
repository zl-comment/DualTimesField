"""lightgbm_qra: quantile regression averaging on the frozen aligned LightGBM.

Features are the frozen tree's sorted five-quantile forecasts per horizon;
unregularized exact-LP linear quantile regression is fitted on 2022H2 labels
only and applied unchanged to the test period. Unlike lightgbm_cp this can
move the centre prediction. Shared driver: ``common/frozen_base.qra_expansion``.
"""
from __future__ import annotations

from ..common.frozen_base import qra_expansion

EXPECTED_PARAMS = {"qra_alpha": 0.0, "qra_solver": "highs",
                   "feature_set": "intercept_plus_sorted_base_five_quantiles"}


def run_expansion(context):
    return qra_expansion(context, base_model="lightgbm", expected_params=EXPECTED_PARAMS)
