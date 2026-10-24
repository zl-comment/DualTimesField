# mlp

实现类型：**本地实现**。

来源：**本地实现，无外部模型仓库**。

共享输入展平后送入两层 MLP；没有对应的作者仓库。

配置：[configs/mlp/aligned.yaml](../../configs/mlp/aligned.yaml)。实验模型 ID：`mlp`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。

## 论文链接与引用

本地通用 MLP 对照，没有对应本结构的单一原论文。所列文献仅为反向传播的经典背景，不是该电价模型或其两层/GELU/分位头设计的出处。

可复制的引用格式：[BibTeX](CITATIONS.bib)。作者、年份、出处及核对来源见 [source.json](source.json)，模型代码开头也列出引用。

- **前馈网络反向传播的经典背景文献**：Rumelhart, David E.; Hinton, Geoffrey E.; Williams, Ronald J. (1986). Learning representations by back-propagating errors. Nature, 323, 533–536. [论文链接](https://doi.org/10.1038/323533a0) · BibTeX key：`rumelhart1986learning`。

引用信息核对日期：2026-10-23。引用方法或组件不代表复现了原论文的完整实验。
