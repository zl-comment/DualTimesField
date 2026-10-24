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


## 正式结果（36/36 格完成，零失败；独立复算四模型全部通过）

宏平均 = 三地区等权、先逐 seed 评分后取均值 ± 样本标准差。Q 表点指标用重排 q50；
P 表无概率指标（缺失，不填零）。对照行见 candidate/（主表 21 输出只读并入）。

| 模型 | 表 | 宏 MAE（mean±SD） | 宏 QS | 地区均 RMSE | 80% 覆盖 | 80% 宽度 | 交叉率 |
|---|---|---:|---:|---:|---:|---:|---:|
| tabpfn_v3（全 1465 列） | Q | 44.13 ± 0.18 | 39.48 | 275.38 | 0.888 | 319.8 | 0.0000 |
| chronos2_zeroshot（受限输入） | Q | 48.53 ± 0.00 | 32.50 | 352.03 | 0.747 | 146.2 | 0.0001 |
| moirai2_zeroshot（受限输入） | P | 55.75 ± 0.00 | — | 291.78 | — | — | — |
| timesfm25_xreg（受限输入+XReg） | P | 95.89 ± 0.00 | — | 424.26 | — | — | — |

参考（当前主表）：LightGBM 37.10/24.19；PatchTST 44.76/27.48；Full_joint 37.78/24.39。
完整候选表：[candidate/main_candidate.csv](candidate/main_candidate.csv) 及
regional/macro_per_seed/per_seed/provenance。

读法要点：
- 无预训练基础模型在合法输入下超过 LightGBM 或 Full_joint；tabpfn_v3 点误差
  优于 patchtst/timexer/linear/dlinear/mlp/pricefm，但五分位 QS 明显偏弱
  （区间宽 320、覆盖 0.89）。
- chronos2 五分位为其 21 级原生头精确输出，QS 32.5 尚可但 80% 覆盖 0.747 偏低。
- timesfm 点 MAE 95.9 全部由 2024-05 类危机日尖峰外推爆炸驱动（中位误差 37.8 正常），
  无任何裁剪，如实报告。
- chronos/moirai 的 2027/2028 seed 复用检查因 GPU 核抖动（>1e-4 绝对）回退为完全
  独立推理；timesfm 6 格中 4 格逐位复用（reuse_of=2026 记录）、2 格回退独立。
- 独立复算：每模型逐格重评分+分块逐位重组通过；重放——零样本模型分层 96+极端 20
  起点相对差 ≤1e-3；tabpfn 生产形状整块（4096 起点）重放**逐位 0.0**，散样本
  0.0075 相对差为已记录的批量尺寸核伪影（同伴交换 0.0）。

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
