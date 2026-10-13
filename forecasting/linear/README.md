# linear

实现类型：**本地实现**。

来源：**本地实现，无外部模型仓库**。

不进行趋势分解的 DLinear 对照；包含已知外生输入线性 adapter。不是单独论文的官方实现。

配置：[configs/linear/aligned.yaml](../../configs/linear/aligned.yaml)。实验模型 ID：`linear`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。
