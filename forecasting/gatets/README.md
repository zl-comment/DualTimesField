# gatets

实现类型：**官方核心源码 + 本地适配**。

来源：[https://github.com/ruiite/MOE_time_series](https://github.com/ruiite/MOE_time_series)。

官方 new_moe.py 按固定提交保存；本地修改输出维度并增加线性外生 adapter。并非原论文数据集分数复现。

配置：[configs/gatets/aligned.yaml](../../configs/gatets/aligned.yaml)。实验模型 ID：`gatets_adapter`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。

原始代码与固定提交记录位于 [vendor/PROVENANCE.md](vendor/PROVENANCE.md)；`vendor` 中的原始源码快照保留校验值。
