# 第二批基础模型基线归档（foundation_baselines_20261023）

分支 `research/foundation-baseline-expansion-20261023`（基点 19ba700），
worktree `/root/workspace/Code/DualTimesField-foundation-baselines`。
正式运行 `outputs/forecasting/foundation_baselines_20261023_r1`（36 格 =
4 模型 × 3 地区 × 3 seed；只读复用父运行 aligned_baselines_20261012 缓存）。

## 内容

- `capability_table.json` —— 阶段0机器可读能力表（权重 revision/SHA、输入映射、
  原生分位、上下文限制、预训练来源、成本探测、分组）。
- `verify/<model>_seed2026/verification.json` —— 阶段2真实权重验证：无标签视图、
  逆变换、批次/同伴不变性、seed 重放、映射敏感性、负价行为。
- `smoke/`、`smoke_summary.json` —— 真实权重开发数据 smoke（标 smoke，不入正式表）。
- `candidate/` —— 候选对比（main_candidate.csv、regional.csv、macro_per_seed.csv、
  per_seed.csv、provenance.json）；主表目录只读，不覆盖。
- `README.md`（本文件）、`progress_snapshot.json`（正式运行完成态快照）。

## 状态与入口

- 阶段0审计：[docs/foundation_baselines_stage0_20261023.md](../../../docs/foundation_baselines_stage0_20261023.md)
- 统一入口：`scripts/foundation_baselines.sh {prepare|run|smoke|verify|audit|report|tests}`
- 每格状态在 run 目录 `runs/<model>/seed<seed>/<region>/status.json`；
  分块断点与 SHA 台账在 `predictions/chunks/`；预算记录在每格 test.npz 的
  `model_metadata.budget`。

## 关键事实（完整依据见 stage0 文档与各模型 source.json）

- TabPFN 固定 V3 回归检查点（默认构造器会取 v3.5）；全量 1465 列公共输入、
  全量 train 上下文；五分位为原生分布分位 → Q 表。seed 只控制集成预处理 RNG。
- Chronos-2 检查点原生训练 21 级分位含全部五级 → Q 表；输入受限
  （价格+需求 past+日历 past/future），不做等输入比较。
- Moirai-2 原生分位恰为 0.1–0.9 → P 表（点=原生 q50），q10/q90 仅辅助。
- TimesFM-2.5 官方 XReg **按 origin 逐个调用**：官方批量接口会把一批 origin 混入
  同一最小二乘设计矩阵（实测批次组成可移动预测 14–36 AUD/MWh，属跨起点信息
  共享）；逐 origin 调用后同伴交换不变性为 0.0。无 q05/q95 → P 表。
- 四个检查点均发布于测试期之后，预训练与 NEM 重叠未审计 → 候选表全部带
  `pretraining_overlap=unaudited` 标注，不称严格无污染盲测。
