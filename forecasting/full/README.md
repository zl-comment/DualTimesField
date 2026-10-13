# full

实现类型：**本地实现**。

来源：[https://github.com/zl-comment/DualTimesField](https://github.com/zl-comment/DualTimesField)。

本项目原始 Full；底层双场算子仍位于 src/dualfield/core.py。保留原训练目标与结构。

配置：[configs/full/aligned.yaml](../../configs/full/aligned.yaml)。实验模型 ID：`full`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。

`loss.py` 保存原始 Full 损失。Full V2 已移除；此处保留的是已完成基线实验中的原始 Full。
