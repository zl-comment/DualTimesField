# lightgbm

实现类型：**官方库调用 + 本地训练适配**。

来源：[https://github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM)。

核心算法来自安装的 lightgbm 包；逐 horizon 训练五个分位数，五分位数联合选择该 horizon 的轮数。

配置：[configs/lightgbm/aligned.yaml](../../configs/lightgbm/aligned.yaml)。实验模型 ID：`lightgbm`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。

`predictor.py` 仅加载与校验已保存的 LightGBM 基线，供诊断使用，不包含 V2 的交叉拟合或训练逻辑。
