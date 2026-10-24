# xgboost_cp

实现类型：**本地适配（冻结基座复用 + 公共 CP 工具）**。

与 `lightgbm_cp` 完全相同的公共校准协议，基座换成冻结 aligned XGBoost。两者共用 `forecasting/common/frozen_base.py::conformal_expansion` 与 `common/calibration.py` 的 CP 公式；各自的适配器、配置、来源与结果独立保存。

- 输入块映射：无新增输入；冻结 booster 再生成 + 逐位复现门槛。
- 输出：原始 RRP 五分位；q50 == 冻结 XGBoost q50。
- 限制：同 lightgbm_cp（覆盖保证、阻断行为）。

配置：[configs/xgboost_cp/aligned.yaml](../../configs/xgboost_cp/aligned.yaml)。实验模型 ID：`xgboost_cp`。来源与差异：[source.json](source.json)。

## 论文链接与引用

冻结 XGBoost 基座上的本地组合。本地逐 horizon 对称绝对残差校准，使用 type-7 分位插值；Lei 等仅提供分割共形预测的方法背景，具体公式参考 PostForecasts.jl。插值规则与时间序列非交换性意味着这里不主张该文的有限样本覆盖保证。

可复制的引用格式：[BibTeX](CITATIONS.bib)。作者、年份、出处及核对来源见 [source.json](source.json)，模型代码开头也列出引用。

- **XGBoost 基座原论文**：Chen, Tianqi; Guestrin, Carlos (2016). XGBoost: A Scalable Tree Boosting System. KDD, 785–794. [论文链接](https://doi.org/10.1145/2939672.2939785) · BibTeX key：`chen2016xgboost`。
- **分割共形预测的方法背景**：Lei, Jing; G'Sell, Max; Rinaldo, Alessandro; Tibshirani, Ryan J.; Wasserman, Larry (2018). Distribution-Free Predictive Inference for Regression. Journal of the American Statistical Association, 113(523), 1094–1111. [论文链接](https://doi.org/10.1080/01621459.2017.1307116) · BibTeX key：`lei2018distributionfree`。
- **绝对残差 CP 公式的软件来源**：Lipiecki, Arkadiusz; Weron, Rafał (2025). PostForecasts.jl: A Julia package for probabilistic forecasting by postprocessing point predictions. SoftwareX, 31, 102200. [论文链接](https://doi.org/10.1016/j.softx.2025.102200) · BibTeX key：`lipiecki2025postforecasts`。
- **相关电价比较论文（J34），非完整复现**：Lebedev, Andreas; Das, Abhinav; Pappert, Sven; Schlüter, Stephan (2026). Analyzing Uncertainty Quantification in Statistical and Deep Learning Models for Probabilistic Electricity Price Forecasting. IEEE Access, 14, 52162–52189. [论文链接](https://doi.org/10.1109/ACCESS.2026.3672716) · BibTeX key：`lebedev2026uncertainty`。

引用信息核对日期：2026-10-23。引用方法或组件不代表复现了原论文的完整实验。
