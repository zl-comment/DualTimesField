# NHITS vendor provenance

- 来源：https://github.com/Nixtla/neuralforecast
- 上游提交：`6c11accfbcfdd62a40ef69a7c81e3d9c7f272682`（2026-10-14，main 分支浅克隆）
- 上游文件：`neuralforecast/models/nhits.py`
- 许可证：Apache-2.0（上游 LICENSE 全文见 https://github.com/Nixtla/neuralforecast/blob/main/LICENSE ）
- 移植日期：2026-10-22

## 保留（与上游逐行一致）

- `_IdentityBasis`：谐波插值基（linear/nearest/cubic F.interpolate），theta 拆分为 backcast/knots。
- `NHITSBlock`：多尺度池化（MaxPool1d/AvgPool1d，ceil_mode）、hist/futr/stat 拼接顺序、MLP 层/激活/dropout 排布、输出层。
- 模型级残差结构（`nhits_qra/model.py`）：`residuals = insample_y.flip(-1)` 初始化、逐块 `residuals = (residuals - backcast) * mask`、`forecast = insample_y[:, -1:, None]` naive1 水平初始化、逐块累加。

## 修改（全部列出）

1. `NHITSBlock.__init__` 新增显式参数 `futr_length`：上游以 `input_size + h` 推 futr 池化长度；本地适配器只提供预测期 [t+1, t+H] 的已知未来通道（共享七块输入边界），历史侧日历/外生进入 `hist_exog`。池化公式本身不变（`ceil(futr_length / pool)`）。
2. 去除 `neuralforecast.common._base_model.BaseModel`/pytorch-lightning 依赖：训练循环、优化器、检查点选择由仓库自有 `forecasting/common/native_trainer.py` 提供（共享预算：epochs/batch/patience 沿用公共 training 节）。
3. 上游默认损失 MAE 保留为标准化空间的逐 horizon MAE（`outputsize_multiplier=1` 点头）。
4. 未移植：官方 DataFrame 协议、windows 采样器、scaler_type、学习率衰减、category embedding、decompose_forecast。这些属于官方训练管线而非骨干；适配版的等价物在 `model.py` 中明确声明。

## 与官方库的关系声明

本目录是"官方源码移植 + 本地训练适配"，不是 pip 安装的官方库运行结果，也不是从头重实现。骨干结构与上游提交一致（除上述修改）。
