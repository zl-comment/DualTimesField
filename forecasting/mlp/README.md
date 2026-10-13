# mlp

实现类型：**本地实现**。

来源：**本地实现，无外部模型仓库**。

共享输入展平后送入两层 MLP；没有对应的作者仓库。

配置：[configs/mlp/aligned.yaml](../../configs/mlp/aligned.yaml)。实验模型 ID：`mlp`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。
