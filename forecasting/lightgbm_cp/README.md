# lightgbm_cp

实现类型：**本地适配（冻结基座复用 + 公共 CP 工具）**。

在冻结的 aligned LightGBM（父运行 `aligned_baselines_20261012`，只读复用，不重训）之上做逐 horizon 对称保形区间：中心为该树排序后 q50 且保持完全一致，宽度取 2022H2 样本外绝对残差的 type-7 分位。回答的问题是"复杂 Anchor 是否优于便宜的不确定性校准"，它不修正中心偏差。

- 输入块映射：无新增输入；校准/测试预测由冻结 booster 再生成，且每格运行时以逐位复现父运行 test.npz 为硬门槛。
- 输出：原始 RRP 五分位；q50 == 冻结基座 q50（浮点精确）。
- 公式来源：`forecasting/common/calibration.py::absolute_residual_cp`（PostForecasts.jl 589f940；绝对残差 + type-7），与 `xgboost_cp` 共用，不复制公式。
- 限制：时间重叠样本非交换，不承诺有限样本覆盖保证；基座缺失时明确报阻断，不以重训冒充复用。

配置：[configs/lightgbm_cp/aligned.yaml](../../configs/lightgbm_cp/aligned.yaml)。实验模型 ID：`lightgbm_cp`。来源与差异：[source.json](source.json)。
