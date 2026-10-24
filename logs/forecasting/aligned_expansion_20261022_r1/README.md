# 第一批公平基线扩展：正式矩阵与合并报告（2026-10-22）

后续验收发现本页旧 DDNN-JSU 结果包含逆 CDF 钳位，已完成去钳位修复和9格复跑。
本页及原结果保留作历史记录；当前主表与验证见[最新统一主表](../main_comparison_20261023/README.md)，DDNN 修复过程见[修复报告](../ddnn_jsu_unclipped_20261022_r1/README.md)。

45/45 格完成（5 模型 × NSW1/QLD1/TAS1 × seed 2026/2027/2028），失败 0 格，约 30 分钟。
运行根：`outputs/forecasting/aligned_expansion_20261022_r1`（本仓库 outputs 忽略跟踪；轻量报告与合并表在本目录）。

## 可复制命令（与实现后的 CLI 一致）

```bash
.venv/bin/python -m forecasting.aligned_expansion prepare \
  --config configs/experiments/aemo_forecast_aligned_expansion_20261022.yaml \
  --run-root outputs/forecasting/aligned_expansion_20261022_r1 \
  --extend-from outputs/forecasting/aligned_baselines_20261012
.venv/bin/python -m forecasting.aligned_expansion run \
  --run-root outputs/forecasting/aligned_expansion_20261022_r1
.venv/bin/python -m forecasting.aligned_expansion_report \
  --runs outputs/forecasting/aligned_baselines_20261012 \
         outputs/forecasting/aligned_baselines_20261013_extension \
         outputs/forecasting/aligned_expansion_20261022_r1 \
  --output logs/forecasting/aligned_expansion_20261022_r1/combined_report.json
```

单模型 smoke（真实父数据、smoke=true、独立目录，不进正式榜）：

```bash
.venv/bin/python -m forecasting.aligned_expansion_smoke --models lightgbm_cp lightgbm_qra xgboost_cp ddnn_jsu nhits_qra
```

## 协议与来源

- 复用冻结 90 格父运行（`aligned_baselines_20261012`）的只读 data_cache（符号链接）与冻结 LightGBM/XGBoost 检查点；`protocol_id: aligned-baselines-20261012-static-v1` 不变；公共七块输入、时间切分、17,521 测试起点与评分公式与父运行逐项一致。
- 每个树校准格在运行时以**逐位复现父运行 test.npz（偏差 0.0）**为硬门槛后才消费校准段；booster/manifest 哈希写入各格 manifest。
- 模型身份：lightgbm_cp / xgboost_cp / lightgbm_qra 为本地适配；ddnn_jsu 为方法重实现（J34 公开仓库无 JSU 头，按论文公式本地实现，scipy johnsonsu 等价性有测试）；nhits_qra 为官方源码移植（neuralforecast @ 6c11accf，Apache-2.0，vendor/PROVENANCE.md 列全部差异）。见各 `forecasting/<model>/source.json`。
- 回顾性测试、无超参搜索、seed 重复非独立时间序列证据；显著性未计算。

## 新增五模型（三 seed 宏平均，原始价格空间）

| 模型 | 宏MAE | 宏QS | 90%覆盖 | 90%宽度 |
|---|---:|---:|---:|---:|
| lightgbm_cp | 37.101 | 30.973 | 0.987 | 278.7 |
| xgboost_cp | 37.930 | 31.325 | 0.986 | 279.3 |
| lightgbm_qra | 41.577 | 27.161 | 0.938 | 164.9 |
| ddnn_jsu | 57.275 | 40.828 | 0.504 | 149.0 |
| nhits_qra | 60.746 | 36.498 | 0.923 | 220.3 |

机制结论（与主研究问题的对照）：
- CP 中心与冻结基座逐位一致（37.101/37.930 精确等于树），仅校准区间；绝对残差 CP 在重叠小时样本上过度覆盖（98.7% vs 名义 90%），QS 明显劣于树自身分位。便宜的 CP 不能解释复杂 Anchor 的位置校正。
- QRA 在 2022H2 学到的中心位移不迁移到 2023–24（MAE +4.5）——与 Full_joint 的 Anchor 同类失败模式，强化"高价期拟合的中心校正不可直接外推"的归因。
- ddnn_jsu 严重欠覆盖（50%），与 J34 关于 DDNN 需校准的结论一致；nhits_qra 区间合理（92%）但骨干点预测弱。

## 合并总表（153 格 / 17 模型 Q 表，前五名）

| 模型 | 宏MAE | 宏QS |
|---|---:|---:|
| lightgbm | 37.101 | 24.191 |
| lightgbm_cp | 37.101 | 30.973 |
| xgboost | 37.930 | 25.622 |
| xgboost_cp | 37.930 | 31.325 |
| lightgbm_qra | 41.577 | 27.161 |

完整表：[combined_report.json](combined_report.json)（含逐来源 provenance 与审计状态）、
[combined_report_tables/main.csv](combined_report_tables/main.csv)（17 行宏）、
[regional.csv](combined_report_tables/regional.csv)（51 行分地区）、
[per_seed.csv](combined_report_tables/per_seed.csv)（153 行逐 seed）。
跨 run 对齐校验：`passed_all_153_artifacts`（每地区逐 artifact assert_comparable）。

## 产物与测试

- 运行内每格：`test.npz`、`metrics.json`、`status.json`（含 artifact 哈希）、`models/`（CP 残差/QRA 系数/检查点）、`progress.json`；控制器 `progress.json`/`controller_launch.log`；`report.json` 由公共报告器生成。
- 契约与模型测试：`tests/test_expansion_{model_layout,calibration,frozen_base,native_models}.py`（22 项）；旧 118 项 aligned/layout 测试全部保持通过（共享接口唯一变更：`validate_config` 提取 `_validate_common_protocol` + single_model 白名单纳入注册扩展模型，均向后兼容）。
- 已知边界：nhits_qra 的 futr 通道仅覆盖预测期长度（vendor PROVENANCE 修改1）；ddnn_jsu 与原文的预算/选择差异已在 source.json 声明；无 2023–24 盲测声明。
