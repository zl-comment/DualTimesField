# pricefm

实现类型：**本地移植（官方 TensorFlow → PyTorch）**。

来源：[https://github.com/runyao-yu/PriceFM](https://github.com/runyao-yu/PriceFM)。

固定 2026 MoE 版本；单地区、单节点、从头训练。未复现多地区预训练或空间迁移；官方 TensorFlow 等价测试受主机运行环境阻碍，未通过验证。

配置：[configs/pricefm/aligned.yaml](../../configs/pricefm/aligned.yaml)。实验模型 ID：`pricefm`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。

原始代码与固定提交记录位于 [vendor/PROVENANCE.md](vendor/PROVENANCE.md)；`vendor` 中的原始源码快照保留校验值。

## 论文链接与引用

原工作首次提交于 2025 年，引用元数据按 2026 年修订版核对；代码独立固定 March 2026 MoE 提交。单区域单节点、从头训练的 PyTorch 移植，未复现原文多区域预训练与迁移。

可复制的引用格式：[BibTeX](CITATIONS.bib)。作者、年份、出处及核对来源见 [source.json](source.json)，模型代码开头也列出引用。

- **PriceFM 原始工作（预印本）**：Yu, Runyao; Gu, Chenhui; Stiasny, Jochen; Wen, Qingsong; Dilov, Wasim Sarwar; Qi, Lianlian; Cremer, Jochen L. (2025). PriceFM: Foundation Model for Probabilistic Electricity Price Forecasting. arXiv:2508.04875 (v4 revised 2026). [论文链接](https://arxiv.org/abs/2508.04875) · BibTeX key：`yu2025pricefm`。

引用信息核对日期：2026-10-23。引用方法或组件不代表复现了原论文的完整实验。
