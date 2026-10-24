# patchtst

实现类型：**本地实现（参考模型）**。

来源：[https://github.com/yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)。

本地重写 RevIN、patch、Transformer 与预测头；外生数据经线性 adapter。尚未完成与官方代码的权重映射等价验证。

配置：[configs/patchtst/aligned.yaml](../../configs/patchtst/aligned.yaml)。实验模型 ID：`patchtst`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。

## 论文链接与引用

PatchTST/RevIN 的本地参考实现，增加外生变量 adapter 和五分位输出；未通过与官方权重映射的等价验证。

可复制的引用格式：[BibTeX](CITATIONS.bib)。作者、年份、出处及核对来源见 [source.json](source.json)，模型代码开头也列出引用。

- **PatchTST 原论文**：Nie, Yuqi; Nguyen, Nam H.; Sinthong, Phanwadee; Kalagnanam, Jayant (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR. [论文链接](https://arxiv.org/abs/2211.14730) · BibTeX key：`nie2023time`。
- **RevIN 组件原论文**：Kim, Taesung; Kim, Jinhee; Tae, Yunwon; Park, Cheonbok; Choi, Jang-Ho; Choo, Jaegul (2022). Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift. ICLR. [论文链接](https://openreview.net/forum?id=cGDAkQo1C0p) · BibTeX key：`kim2022reversible`。

引用信息核对日期：2026-10-23。引用方法或组件不代表复现了原论文的完整实验。
