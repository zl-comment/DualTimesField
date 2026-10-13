# dlinear

实现类型：**本地实现（参考模型）**。

来源：[https://github.com/cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)。

移动平均分解与线性预测，本地增加分位数头及外生输入 adapter。未建立官方权重映射等价验证。

配置：[configs/dlinear/aligned.yaml](../../configs/dlinear/aligned.yaml)。实验模型 ID：`dlinear`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。
