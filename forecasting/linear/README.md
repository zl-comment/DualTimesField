# linear

实现类型：**本地实现**。

来源：**本地实现，无外部模型仓库**。

不进行趋势分解的 DLinear 对照；包含已知外生输入线性 adapter。不是单独论文的官方实现。

配置：[configs/linear/aligned.yaml](../../configs/linear/aligned.yaml)。实验模型 ID：`linear`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。

## 论文链接与引用

本地无趋势分解的线性预测对照，含已知外生 adapter；引用说明其设计背景，不将本地完整模型标为单独论文的官方实现。

可复制的引用格式：[BibTeX](CITATIONS.bib)。作者、年份、出处及核对来源见 [source.json](source.json)，模型代码开头也列出引用。

- **Linear/DLinear 对照设计的相关论文**：Zeng, Ailing; Chen, Muxi; Zhang, Lei; Xu, Qiang (2023). Are Transformers Effective for Time Series Forecasting?. Proceedings of the AAAI Conference on Artificial Intelligence, 37(9), 11121–11128. [论文链接](https://doi.org/10.1609/aaai.v37i9.26317) · BibTeX key：`zeng2023transformers`。

引用信息核对日期：2026-10-23。引用方法或组件不代表复现了原论文的完整实验。
