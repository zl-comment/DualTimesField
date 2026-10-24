"""xgboost_cp: symmetric conformal intervals on the frozen aligned XGBoost.

Identical public calibration protocol as lightgbm_cp with the frozen XGBoost
base; both call the same shared tools, so the CP formula exists exactly once.

论文链接与引用 / References
--------------------------
冻结 XGBoost 基座上的本地组合。本地逐 horizon 对称绝对残差校准，使用 type-7 分位插值；Lei 等仅提供分割共形预测的方法背景，具体公式参考 PostForecasts.jl。插值规则与时间序列非交换性意味着这里不主张该文的有限样本覆盖保证。

BibTeX（可直接复制）: 本目录 CITATIONS.bib；逐项用途与核对来源: source.json。

[chen2016xgboost] XGBoost 基座原论文
Chen, Tianqi; Guestrin, Carlos (2016). XGBoost: A Scalable Tree Boosting System. KDD, 785–794.
https://doi.org/10.1145/2939672.2939785

[lei2018distributionfree] 分割共形预测的方法背景
Lei, Jing; G'Sell, Max; Rinaldo, Alessandro; Tibshirani, Ryan J.; Wasserman, Larry (2018).
Distribution-Free Predictive Inference for Regression. Journal of the American Statistical
Association, 113(523), 1094–1111.
https://doi.org/10.1080/01621459.2017.1307116

[lipiecki2025postforecasts] 绝对残差 CP 公式的软件来源
Lipiecki, Arkadiusz; Weron, Rafał (2025). PostForecasts.jl: A Julia package for probabilistic
forecasting by postprocessing point predictions. SoftwareX, 31, 102200.
https://doi.org/10.1016/j.softx.2025.102200

[lebedev2026uncertainty] 相关电价比较论文（J34），非完整复现
Lebedev, Andreas; Das, Abhinav; Pappert, Sven; Schlüter, Stephan (2026). Analyzing Uncertainty
Quantification in Statistical and Deep Learning Models for Probabilistic Electricity Price
Forecasting. IEEE Access, 14, 52162–52189.
https://doi.org/10.1109/ACCESS.2026.3672716
"""
from __future__ import annotations

from ..common.frozen_base import conformal_expansion

EXPECTED_PARAMS = {"calibration_formula": "absolute_residual_type7_per_horizon",
                   "centre": "frozen_base_rearranged_q50_unchanged"}


def run_expansion(context):
    return conformal_expansion(context, base_model="xgboost", expected_params=EXPECTED_PARAMS)
