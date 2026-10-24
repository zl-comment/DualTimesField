# lightgbm_cp

实现类型：**本地适配（冻结基座复用 + 公共 CP 工具）**。

在冻结的 aligned LightGBM（父运行 `aligned_baselines_20261012`，只读复用，不重训）之上做逐 horizon 对称保形区间：中心为该树排序后 q50 且保持完全一致，宽度取 2022H2 样本外绝对残差的 type-7 分位。回答的问题是"复杂 Anchor 是否优于便宜的不确定性校准"，它不修正中心偏差。

- 输入块映射：无新增输入；校准/测试预测由冻结 booster 再生成，且每格运行时以逐位复现父运行 test.npz 为硬门槛。
- 输出：原始 RRP 五分位；q50 == 冻结基座 q50（浮点精确）。
- 公式来源：`forecasting/common/calibration.py::absolute_residual_cp`（PostForecasts.jl 589f940；绝对残差 + type-7），与 `xgboost_cp` 共用，不复制公式。
- 限制：时间重叠样本非交换，不承诺有限样本覆盖保证；基座缺失时明确报阻断，不以重训冒充复用。

配置：[configs/lightgbm_cp/aligned.yaml](../../configs/lightgbm_cp/aligned.yaml)。实验模型 ID：`lightgbm_cp`。来源与差异：[source.json](source.json)。

## 论文链接与引用

冻结 LightGBM 基座上的本地组合。本地逐 horizon 对称绝对残差校准，使用 type-7 分位插值；Lei 等仅提供分割共形预测的方法背景，具体公式参考 PostForecasts.jl。插值规则与时间序列非交换性意味着这里不主张该文的有限样本覆盖保证。

可复制的引用格式：[BibTeX](CITATIONS.bib)。作者、年份、出处及核对来源见 [source.json](source.json)，模型代码开头也列出引用。

- **LightGBM 基座原论文**：Ke, Guolin; Meng, Qi; Finley, Thomas; Wang, Taifeng; Chen, Wei; Ma, Weidong; Ye, Qiwei; Liu, Tie-Yan (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information Processing Systems 30. [论文链接](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) · BibTeX key：`ke2017lightgbm`。
- **分割共形预测的方法背景**：Lei, Jing; G'Sell, Max; Rinaldo, Alessandro; Tibshirani, Ryan J.; Wasserman, Larry (2018). Distribution-Free Predictive Inference for Regression. Journal of the American Statistical Association, 113(523), 1094–1111. [论文链接](https://doi.org/10.1080/01621459.2017.1307116) · BibTeX key：`lei2018distributionfree`。
- **绝对残差 CP 公式的软件来源**：Lipiecki, Arkadiusz; Weron, Rafał (2025). PostForecasts.jl: A Julia package for probabilistic forecasting by postprocessing point predictions. SoftwareX, 31, 102200. [论文链接](https://doi.org/10.1016/j.softx.2025.102200) · BibTeX key：`lipiecki2025postforecasts`。
- **相关电价比较论文（J34），非完整复现**：Lebedev, Andreas; Das, Abhinav; Pappert, Sven; Schlüter, Stephan (2026). Analyzing Uncertainty Quantification in Statistical and Deep Learning Models for Probabilistic Electricity Price Forecasting. IEEE Access, 14, 52162–52189. [论文链接](https://doi.org/10.1109/ACCESS.2026.3672716) · BibTeX key：`lebedev2026uncertainty`。

引用信息核对日期：2026-10-23。引用方法或组件不代表复现了原论文的完整实验。
