# no_fields

实现类型：**本地实现**。

来源：[https://github.com/zl-comment/DualTimesField](https://github.com/zl-comment/DualTimesField)。

Full 的 field_forecast=False 对照；仍包含历史分解计算和损失，不能称为完全删除双场计算。

配置：[configs/no_fields/aligned.yaml](../../configs/no_fields/aligned.yaml)。实验模型 ID：`no_fields`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。
