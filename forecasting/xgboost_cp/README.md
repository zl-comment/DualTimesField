# xgboost_cp

实现类型：**本地适配（冻结基座复用 + 公共 CP 工具）**。

与 `lightgbm_cp` 完全相同的公共校准协议，基座换成冻结 aligned XGBoost。两者共用 `forecasting/common/frozen_base.py::conformal_expansion` 与 `common/calibration.py` 的 CP 公式；各自的适配器、配置、来源与结果独立保存。

- 输入块映射：无新增输入；冻结 booster 再生成 + 逐位复现门槛。
- 输出：原始 RRP 五分位；q50 == 冻结 XGBoost q50。
- 限制：同 lightgbm_cp（覆盖保证、阻断行为）。

配置：[configs/xgboost_cp/aligned.yaml](../../configs/xgboost_cp/aligned.yaml)。实验模型 ID：`xgboost_cp`。来源与差异：[source.json](source.json)。
