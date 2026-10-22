# lightgbm_qra

实现类型：**本地适配（冻结基座复用 + 线性分位回归平均）**。

以冻结 aligned LightGBM 的排序后五分位预测为逐 horizon 特征，在 2022H2 标签上拟合无正则精确 LP 线性分位回归（scikit-learn QuantileRegressor，solver=highs，alpha=0，确定性），测试期系数冻结。与 lightgbm_cp 不同，QRA 可以改变中心预测。

- 输入块映射：无新增输入；特征即冻结基座的五分位输出 + 截距，特征顺序固定为水平升序。
- 输出：原始 RRP 五分位；点预测为 QRA 输出重排后 q50；全部系数、拟合范围与求解器状态存档于 `models/qra_fit.npz`。
- 限制：明确的本地 QRA 适配，不复现任何论文完整设计；与 lightgbm_cp 一起构成"便宜树后校准 vs 复杂 Anchor"的机制归因对照。

配置：[configs/lightgbm_qra/aligned.yaml](../../configs/lightgbm_qra/aligned.yaml)。实验模型 ID：`lightgbm_qra`。来源与差异：[source.json](source.json)。
