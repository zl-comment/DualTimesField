# gatets

实现类型：**官方核心源码 + 本地适配**。

来源：[https://github.com/ruiite/MOE_time_series](https://github.com/ruiite/MOE_time_series)。

官方 new_moe.py 按固定提交保存；本地修改输出维度并增加线性外生 adapter。并非原论文数据集分数复现。

配置：[configs/gatets/aligned.yaml](../../configs/gatets/aligned.yaml)。实验模型 ID：`gatets_adapter`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。

原始代码与固定提交记录位于 [vendor/PROVENANCE.md](vendor/PROVENANCE.md)；`vendor` 中的原始源码快照保留校验值。

## 论文链接与引用

固定官方骨干，增加外生变量 adapter 与五分位输出；主表 ID 为 gatets_adapter，不是原论文数据集分数复现。

可复制的引用格式：[BibTeX](CITATIONS.bib)。作者、年份、出处及核对来源见 [source.json](source.json)，模型代码开头也列出引用。

- **GateTS 原论文**：Yemets, Kyrylo; Lukashchuk, Mykola; Izonin, Ivan (2025). GateTS: Versatile and Efficient Forecasting via Attention-Inspired routed Mixture-of-Experts. arXiv:2508.17515. [论文链接](https://arxiv.org/abs/2508.17515) · BibTeX key：`yemets2025gatets`。

引用信息核对日期：2026-10-23。引用方法或组件不代表复现了原论文的完整实验。
