# Anchor 确认实验：72 次拟合完成

协议和代码提交：`c1c546e`；启动记录提交：`00872ae`。
全部 72 次拟合各完成 20 epochs；全部选模与 alpha 冻结后才进行评估。

独立核验检查了源代码/协议/输入/检查点哈希、每轮训练顺序、144 份预测、
相同地区年份的起点/标签/树预测、alpha 混合、汇总及验收。
预测文件的 MAE/QS/覆盖率/偏差重算与报告最大差异为 **0**。

## 主要结果

四组均未通过跨两个年份的全部门槛，raw 与 guarded 都不能宣称稳定成功。
种子均值层面，asymmetric_scale guarded 在 2021 通过，但 2022 未通过与 balanced 的宏平均比较；
location_only guarded 在 2022 通过，但 2021 仍存在地区退化等失败项。
其他具体失败条件见 `acceptance.json`。不可用跨地区均值覆盖单地区退化。

| 方案，guarded | 2021 MAE | 2021 QS | 2022 MAE | 2022 QS |
|---|---:|---:|---:|---:|
| balanced | 28.0511 | 20.0240 | 69.6292 | 42.2681 |
| location_only | 28.0782 | 19.9010 | 68.1300 | 41.8017 |
| identity_anchor | 28.1401 | 19.9510 | 68.7495 | 42.3569 |
| asymmetric_scale | 28.0493 | 19.9835 | 69.6511 | 42.1369 |

数值为三地区等权宏平均、再平均三个神经种子；标准差、原始输出、fallback、
选择轮数与分组明细保存在报告中。树种子固定，未检验统计显著性。
这是 2021/2022 回顾性开发确认；不能与 2023/2024 aligned 基线主榜直接按绝对数值排名。

## 文件

- `report.md`：完整概要与验收说明。
- `results.csv`：地区/年份/种子/模型/输出模式/分组结果表。
  归档副本只将 CRLF 换行为 LF，逐字段一致；原运行文件保持不变。
- `summary.json`、`acceptance.json`：均值/种子波动及四项门槛。
- `report.json.gz`：全部分组和分位/宽度/位移抵消诊断。
- `selection_complete.json.gz`：评估前冻结的选择与检查点哈希。
- `training_logs.json.gz`：72 次训练的完整逐轮日志。
- `manifest.json`、`launch.json`：原始预注册来源、样本、输入哈希与启动记录。
- `independent_verification.json`：独立核验结果。
- `training_order_audit.json`、`progress.json`：顺序一致性及完成状态。

检查点、预测 NPZ 和源码快照保留于
`outputs/forecasting/v5_anchor_confirmation_20261022_r1`，不覆盖原六轮初筛结果。
`launch.json` 的 launched 状态描述启动时事实，当前完成状态以 `progress.json` 为准。
