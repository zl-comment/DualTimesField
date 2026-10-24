# lightgbm_qra

实现类型：**本地适配（冻结基座复用 + 线性分位回归平均）**。

以冻结 aligned LightGBM 的排序后五分位预测为逐 horizon 特征，在 2022H2 标签上拟合无正则精确 LP 线性分位回归（scikit-learn QuantileRegressor，solver=highs，alpha=0，确定性），测试期系数冻结。与 lightgbm_cp 不同，QRA 可以改变中心预测。

- 输入块映射：无新增输入；特征即冻结基座的五分位输出 + 截距，特征顺序固定为水平升序。
- 输出：原始 RRP 五分位；点预测为 QRA 输出重排后 q50；全部系数、拟合范围与求解器状态存档于 `models/qra_fit.npz`。
- 限制：明确的本地 QRA 适配，不复现任何论文完整设计；与 lightgbm_cp 一起构成"便宜树后校准 vs 复杂 Anchor"的机制归因对照。

配置：[configs/lightgbm_qra/aligned.yaml](../../configs/lightgbm_qra/aligned.yaml)。实验模型 ID：`lightgbm_qra`。来源与差异：[source.json](source.json)。

## 论文链接与引用

本地 LightGBM+QRA 组合：QRA 特征为单个冻结基座的五分位预测与截距；原 QRA 使用多个模型的点预测。J34 为相关比较，不是本实现的完整设计来源。

可复制的引用格式：[BibTeX](CITATIONS.bib)。作者、年份、出处及核对来源见 [source.json](source.json)，模型代码开头也列出引用。

- **LightGBM 基座原论文**：Ke, Guolin; Meng, Qi; Finley, Thomas; Wang, Taifeng; Chen, Wei; Ma, Weidong; Ye, Qiwei; Liu, Tie-Yan (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information Processing Systems 30. [论文链接](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) · BibTeX key：`ke2017lightgbm`。
- **QRA 原论文**：Nowotarski, Jakub; Weron, Rafał (2015). Computing electricity spot price prediction intervals using quantile regression and forecast averaging. Computational Statistics, 30(3), 791–803. [论文链接](https://doi.org/10.1007/s00180-014-0523-0) · BibTeX key：`nowotarski2015qra`。
- **相关电价比较论文（J34），非完整复现**：Lebedev, Andreas; Das, Abhinav; Pappert, Sven; Schlüter, Stephan (2026). Analyzing Uncertainty Quantification in Statistical and Deep Learning Models for Probabilistic Electricity Price Forecasting. IEEE Access, 14, 52162–52189. [论文链接](https://doi.org/10.1109/ACCESS.2026.3672716) · BibTeX key：`lebedev2026uncertainty`。

引用信息核对日期：2026-10-23。引用方法或组件不代表复现了原论文的完整实验。
