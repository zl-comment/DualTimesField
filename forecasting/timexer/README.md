# timexer

实现类型：**官方核心源码 + 本地适配**。

来源：[https://github.com/thuml/TimeXer](https://github.com/thuml/TimeXer)。

保留官方主干；加入未来已知变量 token，输出改为五分位数。历史路径有同权重数值比较测试。

配置：[configs/timexer/aligned.yaml](../../configs/timexer/aligned.yaml)。实验模型 ID：`timexer`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。

原始代码与固定提交记录位于 [vendor/PROVENANCE.md](vendor/PROVENANCE.md)；`vendor` 中的原始源码快照保留校验值。
