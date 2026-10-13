# xgboost

实现类型：**官方库调用 + 本地训练适配**。

来源：[https://github.com/dmlc/xgboost](https://github.com/dmlc/xgboost)。

核心算法来自安装的 xgboost 包；本地实现共同验证指标和逐 horizon 的训练选择。

配置：[configs/xgboost/aligned.yaml](../../configs/xgboost/aligned.yaml)。实验模型 ID：`xgboost`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。
