# lightgbm

实现类型：**官方库调用 + 本地训练适配**。

来源：[https://github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM)。

核心算法来自安装的 lightgbm 包；逐 horizon 训练五个分位数，五分位数联合选择该 horizon 的轮数。

配置：[configs/lightgbm/aligned.yaml](../../configs/lightgbm/aligned.yaml)。实验模型 ID：`lightgbm`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。

`predictor.py` 仅加载与校验已保存的 LightGBM 基线，供诊断使用，不包含 V2 的交叉拟合或训练逻辑。

## 论文链接与引用

官方库调用；逐 horizon、五分位训练与选择流程为本地适配。

可复制的引用格式：[BibTeX](CITATIONS.bib)。作者、年份、出处及核对来源见 [source.json](source.json)，模型代码开头也列出引用。

- **LightGBM 原论文**：Ke, Guolin; Meng, Qi; Finley, Thomas; Wang, Taifeng; Chen, Wei; Ma, Weidong; Ye, Qiwei; Liu, Tie-Yan (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information Processing Systems 30. [论文链接](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) · BibTeX key：`ke2017lightgbm`。

引用信息核对日期：2026-10-23。引用方法或组件不代表复现了原论文的完整实验。
