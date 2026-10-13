# 配置目录

- `full/`、`no_fields/`、`mlp/`、`linear/`、`dlinear/`、`patchtst/`、`gatets/`、`timexer/`、`pricefm/`、`lightgbm/`、`xgboost/`、`lasso/`：各模型独立的 `aligned.yaml`，包含完整公共数据协议与该模型训练参数，可以直接加载；模型 ID 保持不变。
- `experiments/`：原始 90 格矩阵、18 格扩展及 smoke 协议；迁移时保留文件原内容。
- `legacy/`：旧探索/滚动/消融配置，保留原文件名。这些配置不代表当前统一基线。
- `fair_sota/`：协议草案和审查记录。

各模型 `aligned.yaml` 使用原来的三地区、三 seed、时间划分、预处理和训练预算。单模型配置显式声明 `execution_scope: single_model`，生成 9 个任务，不得冒充完整基线榜单。配置使用完整内容以便冻结追溯，不在运行时隐式继承其他文件。

运行入口（不会因整理自动启动）：

```bash
.venv/bin/python -m forecasting.aligned_experiment prepare --config configs/lightgbm/aligned.yaml --run-root outputs/forecasting/NEW_DATE_lightgbm
.venv/bin/python -m forecasting.aligned_experiment run --run-root outputs/forecasting/NEW_DATE_lightgbm
```

所有命令在项目根目录执行，运行目录必须为新目录。已完成实验仍以其 `outputs/.../config.yaml` 和冻结源码为准。

非 Full 配置不包含 Full 专属结构或辅助损失字段。`model.quantiles` 是公共输出约定；神经模型的结构参数在 `model_baselines.<模型 ID>`，由运行器实际读取。树模型与 Lasso 参数在 `experiment.baseline_parameters`。
