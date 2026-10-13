# pricefm

实现类型：**本地移植（官方 TensorFlow → PyTorch）**。

来源：[https://github.com/runyao-yu/PriceFM](https://github.com/runyao-yu/PriceFM)。

固定 2026 MoE 版本；单地区、单节点、从头训练。未复现多地区预训练或空间迁移；官方 TensorFlow 等价测试受主机运行环境阻碍，未通过验证。

配置：[configs/pricefm/aligned.yaml](../../configs/pricefm/aligned.yaml)。实验模型 ID：`pricefm`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。

原始代码与固定提交记录位于 [vendor/PROVENANCE.md](vendor/PROVENANCE.md)；`vendor` 中的原始源码快照保留校验值。
