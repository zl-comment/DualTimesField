"""lightgbm_qra: quantile regression averaging on the frozen aligned LightGBM.

Features are the frozen tree's sorted five-quantile forecasts per horizon;
unregularized exact-LP linear quantile regression is fitted on 2022H2 labels
only and applied unchanged to the test period. Unlike lightgbm_cp this can
move the centre prediction. Shared driver: ``common/frozen_base.qra_expansion``.

论文链接与引用 / References
--------------------------
本地 LightGBM+QRA 组合：QRA 特征为单个冻结基座的五分位预测与截距；原 QRA 使用多个模型的点预测。J34 为相关比较，不是本实现的完整设计来源。

BibTeX（可直接复制）: 本目录 CITATIONS.bib；逐项用途与核对来源: source.json。

[ke2017lightgbm] LightGBM 基座原论文
Ke, Guolin; Meng, Qi; Finley, Thomas; Wang, Taifeng; Chen, Wei; Ma, Weidong; Ye, Qiwei; Liu, Tie-Yan
(2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information
Processing Systems 30.
https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

[nowotarski2015qra] QRA 原论文
Nowotarski, Jakub; Weron, Rafał (2015). Computing electricity spot price prediction intervals using
quantile regression and forecast averaging. Computational Statistics, 30(3), 791–803.
https://doi.org/10.1007/s00180-014-0523-0

[lebedev2026uncertainty] 相关电价比较论文（J34），非完整复现
Lebedev, Andreas; Das, Abhinav; Pappert, Sven; Schlüter, Stephan (2026). Analyzing Uncertainty
Quantification in Statistical and Deep Learning Models for Probabilistic Electricity Price
Forecasting. IEEE Access, 14, 52162–52189.
https://doi.org/10.1109/ACCESS.2026.3672716
"""
from __future__ import annotations

from ..common.frozen_base import qra_expansion

EXPECTED_PARAMS = {"qra_alpha": 0.0, "qra_solver": "highs",
                   "feature_set": "intercept_plus_sorted_base_five_quantiles"}


def run_expansion(context):
    return qra_expansion(context, base_model="lightgbm", expected_params=EXPECTED_PARAMS)
