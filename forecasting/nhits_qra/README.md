# nhits_qra

实现类型：**官方源码移植 + 本地训练适配**（vendor，Apache-2.0，来源与逐条修改见 [vendor/PROVENANCE.md](vendor/PROVENANCE.md)）。

NHITS 多尺度骨干（MaxPool 层级池化 + 谐波插值基 + 残差回投 + naive1 水平初始化，均保留上游结构）以 MAE 点头训练；冻结后做固定种子 MC-dropout（K=20，仅 dropout 层进入 train 模式，不更新任何训练状态），MC 样本经验分位作为逐 horizon 特征，在 2022H2 标签上拟合 QRA（alpha=0 精确 LP），测试期系数冻结。

- 输入块映射：价格历史→insample_y；需求+历史日历→hist_exog；未来日历+三组未来外生→futr_exog（预测期长度，PROVENANCE 修改1）；origin_context→stat_exog。七块全部接入，无丢弃。
- 选择指标：选择期原始价格 MAE（eval 点预测，配置预声明）；MC/QRA 阶段严格在选择之后。
- 成本：MC 采样开销计入任务时间；固定样本数与种子规则写入 metadata。

配置：[configs/nhits_qra/aligned.yaml](../../configs/nhits_qra/aligned.yaml)。实验模型 ID：`nhits_qra`。来源与差异：[source.json](source.json)。

## 论文链接与引用

官方 NHITS 骨干移植，加固定 MC-dropout 与本地 QRA。QRA 输入为 MC 样本经验分位；采样、训练、特征与选择按公共框架执行，不是 P08/P16 的完整复现。

可复制的引用格式：[BibTeX](CITATIONS.bib)。作者、年份、出处及核对来源见 [source.json](source.json)，模型代码开头也列出引用。

- **NHITS 骨干原论文**：Challu, Cristian; Olivares, Kin G.; Oreshkin, Boris N.; Garza Ramirez, Federico; Mergenthaler Canseco, Max; Dubrawski, Artur (2023). NHITS: Neural Hierarchical Interpolation for Time Series Forecasting. Proceedings of the AAAI Conference on Artificial Intelligence, 37(6), 6989–6997. [论文链接](https://doi.org/10.1609/aaai.v37i6.25854) · BibTeX key：`challu2023nhits`。
- **MC-dropout 方法论文**：Gal, Yarin; Ghahramani, Zoubin (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. ICML, PMLR 48, 1050–1059. [论文链接](https://proceedings.mlr.press/v48/gal16.html) · BibTeX key：`gal2016dropout`。
- **QRA 原论文**：Nowotarski, Jakub; Weron, Rafał (2015). Computing electricity spot price prediction intervals using quantile regression and forecast averaging. Computational Statistics, 30(3), 791–803. [论文链接](https://doi.org/10.1007/s00180-014-0523-0) · BibTeX key：`nowotarski2015qra`。
- **NHITS+QRA 电价比较论文（P08）**：Lettner, Jan Niklas; El Ashhab, Hadeer; Hagenmeyer, Veit; Schäfer, Benjamin (2026). Assessing the Performance-Efficiency Trade-off of Foundation Models in Probabilistic Electricity Price Forecasting. arXiv:2604.14739. [论文链接](https://arxiv.org/abs/2604.14739) · BibTeX key：`lettner2026performance`。
- **相关校准讨论（P16）**：Lettner, Jan Niklas; El Ashhab, Hadeer; Schäfer, Benjamin (2026). Investigating Calibration Challenges in Probabilistic Electricity Price Forecasting. Proceedings of the 2026 ACM Sustainability Week, 565–566. [论文链接](https://doi.org/10.1145/3765611.3815392) · BibTeX key：`lettner2026calibration`。

引用信息核对日期：2026-10-23。引用方法或组件不代表复现了原论文的完整实验。
