# DDNN-JSU 修复完成：9 格复跑与 153 格主表

统一比较入口见[当前主表](../main_comparison_20261022/README.md)，其中另列包含 Full_joint 与五个新增基线的 22 行 seed 2026 比较；本目录保留完整三 seed 静态基线汇总。

修复代码提交 `2cb9edf`，显式替换报告器提交 `e473877`。
原 45 格运行保持原样；本轮只从头重训/选模 DDNN 的 9 格，全部完成、零失败。

## 修复与验证

- 删除逆 CDF 自变量 ±12 钳位；Phi⁻¹ 和 sinh 使用 float64，有限的大尾部如实导出，非有限结果报错。
- 架构、初始化、NLL、scale floor、优化器与实际训练预算未变。学习率保持旧正式矩阵实际生效的 3e-4，同时纠正此前 1e-3 的文档/配置声明。
- 9 个检查点逐位复现其保存的测试预测，并独立通过 SciPy PPF 校验。
- 新旧每轮训练损失差异为 0，9 个最终选中 epoch 均相同；选择分数仍用修复公式逐轮重新计算，没有直接复用旧检查点。
- 修复前的选择期也触发过钳位。此次结果证明选中 epoch 恰好相同，不能将“直接导出旧模型”当作普遍正确的修复流程。
- 相关回归 139 项通过，包含大正负尾/NLL 等价、非有限失败、阶段隔离、替换来源、旧产物审计与不完整运行禁止发榜。
- 合入 Anchor 工作区后，基线与 Full/Anchor 联合回归 **215 项全部通过**。

## 真实分数

三地区等权、三种子分别评分后汇总。QS=2×五分位平均 pinball，原始 RRP 单位。

| DDNN 版本 | 宏 MAE | 宏 QS | 90%覆盖 | 90%平均宽度 |
|---|---:|---:|---:|---:|
| 旧钳位版本，仅作审计 | 57.2755 | 40.8275 | 50.4265% | 148.9972 |
| 修复后未截断 JSU | 57.2755 | 140.8608 | 50.4265% | 4034.6532 |

原钳位掩盖了极少数上尾预测产生的大误差，尤其是 NSW1。修复后分数变差仍必须保留；
不能为维持旧分数重新加截断。这说明当前固定配置的 DDNN 尾部表现很差，不代表 JSU 方法普遍无效。
本轮只修复方法一致性，不进行基于已见测试结果的参数优化。

## 当前主表与历史记录

当前主表仍有 **17 模型 / 153 格**：原90格 + 原扩展18格 + 第一批新增中保留36格 + 修复DDNN9格。
全部 **162 份来源产物**参与审计，其中旧DDNN9格明确标记为 `superseded`，不参与当前主表。
另外16个模型行的数值及来源与旧合并报告逐字段相同。

- `combined_report.json`：当前总报告，记录新来源与 supersedes。
- `combined_report_tables/main.csv`：17 行宏平均。
- `combined_report_tables/regional.csv`：51 行分地区。
- `combined_report_tables/per_seed.csv`：153 行逐种子。
- `ddnn_before_after.csv`：9 格新旧对照；`before_after_summary.json`：宏平均对照。
- `independent_verification.json`：检查点复现、SciPy、选择期与训练轨迹审计。
- `integration_verification.json`：替换数量及其他16模型完全不变的检查。
- `config.yaml`、`jobs.json.gz`、`launch.json`、`progress.json`、`training_logs.json.gz`：冻结来源与运行记录。

17模型表的 `full` 仍是原静态 aligned 模型，不是 2021/2022 Anchor 确认实验的新模型。
两个任务的评估年份不同，绝对误差不能直接用于跨表排名。

## 复算

在保留实验文件的工作区运行（模型推理复现需使用与训练相同的 CUDA 后端）：

```bash
.venv/bin/python -m forecasting.ddnn_jsu.audit \
  --original outputs/forecasting/aligned_expansion_20261022_r1 \
  --repaired outputs/forecasting/ddnn_jsu_unclipped_20261022_r1 \
  --device cuda:6 --output /tmp/ddnn_jsu_reverification.json
.venv/bin/python -m forecasting.aligned_expansion_report \
  --runs outputs/forecasting/aligned_baselines_20261012 \
         outputs/forecasting/aligned_baselines_20261013_extension \
         outputs/forecasting/aligned_expansion_20261022_r1 \
         outputs/forecasting/ddnn_jsu_unclipped_20261022_r1 \
  --replace-model ddnn_jsu=outputs/forecasting/ddnn_jsu_unclipped_20261022_r1 \
  --output /tmp/ddnn_jsu_combined_reverification.json
```

原始运行文件保留在 `/root/workspace/Code/DualTimesField-fair-baselines/outputs/forecasting/`。
主工作区已建立这两个实验的独立运行目录链接，冻结 manifest 中的来源路径不重写。
