# xgboost

实现类型：**官方库调用 + 本地训练适配**。

来源：[https://github.com/dmlc/xgboost](https://github.com/dmlc/xgboost)。

核心算法来自安装的 xgboost 包；本地实现共同验证指标和逐 horizon 的训练选择。

配置：[configs/xgboost/aligned.yaml](../../configs/xgboost/aligned.yaml)。实验模型 ID：`xgboost`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。

## 论文链接与引用

官方库调用；逐 horizon、五分位输出和公共选择流程为本地适配。

可复制的引用格式：[BibTeX](CITATIONS.bib)。作者、年份、出处及核对来源见 [source.json](source.json)，模型代码开头也列出引用。

- **XGBoost 原论文**：Chen, Tianqi; Guestrin, Carlos (2016). XGBoost: A Scalable Tree Boosting System. KDD, 785–794. [论文链接](https://doi.org/10.1145/2939672.2939785) · BibTeX key：`chen2016xgboost`。

引用信息核对日期：2026-10-23。引用方法或组件不代表复现了原论文的完整实验。
