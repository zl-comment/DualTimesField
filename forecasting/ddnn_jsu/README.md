# ddnn_jsu

实现类型：**方法重实现**（依据论文公式；此前核对的 J34 配套仓库 `lebeand/uq-epf` 仅提供 Normal 头，此说明不适用于更早 DDNN 原论文的作者仓库）。

分布深度网络 + Johnson SU 参数头：MLP 输出每 horizon 的 (μ_w, σ_w, ξ, λ)，σ_w、λ 经 softplus+0.1 保证为正。令 u=asinh((x−ξ)/λ)，z=(u−μ_w)/σ_w，NLL 为 −log φ(z)+log σ_w+log λ+log cosh(u)。这等价于经典参数 γ=−μ_w/σ_w、δ=1/σ_w。

以精确 JSU NLL 在共享标准化价格空间训练；五个公共分位为 ξ+λ sinh(μ_w+σ_w Φ⁻¹(q))，用 float64 计算解析逆 CDF，不裁剪参数或预测，非有限结果明确报错。预测再逆变换回原始 RRP。检查点按选择期**原始价格公共五分位 QS** 选择（与原文 NLL 选择方式的差异已在 source.json 与运行 metadata 中声明）。

- 输入块映射：七块公共输入按 INPUT_KEYS 展平（与树模型同展平），无丢弃块。
- 网络形态参照官方 ddnn_normal.py：Kaiming 隐层/Xavier 输出、可选 BN/dropout（第一版 hidden [512,512]、无 BN、dropout 0）、Adam。
- 限制：位置参数 ξ 不声称等于中位数；点指标用真实分布中位数；未复现原文超参搜索。
- 学习率实际读取 `training.learning_rate`：原 20261022 正式矩阵为 3e-4，当前单模型默认也对齐到 3e-4（旧单模型默认与声明曾为 1e-3）。此次去钳位修复复跑固定为原矩阵实际的 3e-4，未重新调参。
- 原 `aligned_expansion_20261022_r1` 的 DDNN 预测曾对 sinh 自变量钳位至 ±12，正式上尾确有触发；该旧行保留作审计，不作为标准 JSU 修复结果。

配置：[configs/ddnn_jsu/aligned.yaml](../../configs/ddnn_jsu/aligned.yaml)。实验模型 ID：`ddnn_jsu`。来源与差异：[source.json](source.json)。

## 论文链接与引用

DDNN-JSU 的方法重实现。J34 配套仓库的 Normal 网络为网络形态参考；JSU 参数化在本地实现。Marcjasz 等为更早的电价 DDNN/JSU 来源；未复现两文完整数据与调参流程。

可复制的引用格式：[BibTeX](CITATIONS.bib)。作者、年份、出处及核对来源见 [source.json](source.json)，模型代码开头也列出引用。

- **电价 DDNN/Johnson SU 原论文**：Marcjasz, Grzegorz; Narajewski, Michał; Weron, Rafał; Ziel, Florian (2023). Distributional neural networks for electricity price forecasting. Energy Economics, 125, 106843. [论文链接](https://doi.org/10.1016/j.eneco.2023.106843) · BibTeX key：`marcjasz2023distributional`。
- **J34 比较论文及本地网络形态参考**：Lebedev, Andreas; Das, Abhinav; Pappert, Sven; Schlüter, Stephan (2026). Analyzing Uncertainty Quantification in Statistical and Deep Learning Models for Probabilistic Electricity Price Forecasting. IEEE Access, 14, 52162–52189. [论文链接](https://doi.org/10.1109/ACCESS.2026.3672716) · BibTeX key：`lebedev2026uncertainty`。

引用信息核对日期：2026-10-23。引用方法或组件不代表复现了原论文的完整实验。
