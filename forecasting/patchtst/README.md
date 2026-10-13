# patchtst

实现类型：**本地实现（参考模型）**。

来源：[https://github.com/yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)。

本地重写 RevIN、patch、Transformer 与预测头；外生数据经线性 adapter。尚未完成与官方代码的权重映射等价验证。

配置：[configs/patchtst/aligned.yaml](../../configs/patchtst/aligned.yaml)。实验模型 ID：`patchtst`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。
