# dlinear

实现类型：**本地实现（参考模型）**。

来源：[https://github.com/cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)。

移动平均分解与线性预测，本地增加分位数头及外生输入 adapter。未建立官方权重映射等价验证。

配置：[configs/dlinear/aligned.yaml](../../configs/dlinear/aligned.yaml)。实验模型 ID：`dlinear`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。

## 论文链接与引用

移动平均分解与线性预测的本地参考实现，增加外生 adapter 与五分位头；未通过官方权重映射等价验证。

可复制的引用格式：[BibTeX](CITATIONS.bib)。作者、年份、出处及核对来源见 [source.json](source.json)，模型代码开头也列出引用。

- **DLinear 原论文**：Zeng, Ailing; Chen, Muxi; Zhang, Lei; Xu, Qiang (2023). Are Transformers Effective for Time Series Forecasting?. Proceedings of the AAAI Conference on Artificial Intelligence, 37(9), 11121–11128. [论文链接](https://doi.org/10.1609/aaai.v37i9.26317) · BibTeX key：`zeng2023transformers`。

引用信息核对日期：2026-10-23。引用方法或组件不代表复现了原论文的完整实验。
