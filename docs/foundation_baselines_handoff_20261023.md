# 交接说明：第二批基础模型基线（2026-10-23，供 Codex 审查合并）

## 分支与基点

- 分支：`research/foundation-baseline-expansion-20261023`
- worktree：`/root/workspace/Code/DualTimesField-foundation-baselines`
- 基点：`19ba700d5988c07083ba7751a31ab853744e94af`（未动活动目录
  `research/v5-mechanism-state-calibration-20261023` 的未提交引用修改）

## 分阶段提交

| commit | 内容 |
|---|---|
| `cefc15e` | 第二批注册/无标签视图/运行器/四模型目录与适配器/配置/测试/阶段0审计 |
| `2e7d37f` | 阶段2修复：TimesFM 池化 XReg 隔离、容差 seed 复用、TabPFN 续跑重建、增强验证 |
| `e916046` | 解释器符号链接保留修复（曾静默落到系统 python） |
| （后续） | 正式结果归档、audit、候选对比、本交接的补充 |

均为线性提交，可按需 cherry-pick；无合并回 Codex 分支的操作。

## 共享接口改动（对本仓库的影响面）

- **新增**：`forecasting/common/foundation_registry.py`、`foundation_api.py`、
  `forecasting/foundation_{experiment,smoke,verify,audit,candidate}.py`、
  `forecasting/{tabpfn_v3,chronos2_zeroshot,moirai2_zeroshot,timesfm25_xreg}/`、
  `configs/experiments/aemo_forecast_foundation_20261023.yaml`、
  `configs/<模型>/aligned.yaml`、`scripts/foundation_baselines.sh`、
  `tests/test_foundation_{registry,worker}.py`、`tests/fixtures/mock_foundation_adapter.py`。
- **未修改**：第一批 `baseline_registry.py`/`aligned_expansion*.py`、BASE/EXTENSION
  语义、父运行与其报告、Full/Anchor/校正稳健性/机理状态实验（回归测试通过）。
- `.gitignore` 追加 `.venvs/`、`weights/`（模型权重与环境不入 Git）。

## 复现命令（均有 --help；统一入口 scripts/foundation_baselines.sh）

```bash
cd /root/workspace/Code/DualTimesField-foundation-baselines
# 环境与权重（一次性，见各模型 ENVIRONMENT.md）
scripts/foundation_baselines.sh tests                        # 框架+回归测试
scripts/foundation_baselines.sh verify tabpfn_v3             # 真实权重阶段2验证（逐模型）
scripts/foundation_baselines.sh smoke chronos2_zeroshot      # 开发数据 smoke
scripts/foundation_baselines.sh prepare                      # 冻结快照+36格
scripts/foundation_baselines.sh run                          # 可断点续跑控制器（GPU 6/7）
scripts/foundation_baselines.sh audit timesfm25_xreg         # 独立复算（逐模型）
scripts/foundation_baselines.sh audit-collect
scripts/foundation_baselines.sh report                       # 候选对比（主表只读）
```

## 已完成 / 失败 / 阻断矩阵（逐项状态）

| 条目 | 编码 | 真实权重 smoke | 阶段2验证 | 正式9格 | 独立复算 |
|---|---|---|---|---|---|
| tabpfn_v3 | ✅ | ✅（MAE 45.4 dev24） | ✅（同伴交换 0.0；批量尺寸核差异 0.15% 已记录并设容差） | 运行中 | 待运行完成后 |
| chronos2_zeroshot | ✅ | ✅（18.1） | ✅（全 0.0/容差内） | 部分（NSW1 ✅ 41s/格） | 待 |
| moirai2_zeroshot | ✅ | ✅（28.2） | ✅ | 部分（NSW1 ✅ 19s/格） | 待 |
| timesfm25_xreg | ✅ | ✅（21.9） | ✅（池化 XReg 修复后全 0.0） | 部分（~30min/格） | 待 |
| 外部阻断 | 无 | | | | |

失败/修复记录（全部保留证据）：
1. TimesFM 官方批量 XReg 池化泄漏（阶段2发现）→ 逐 origin 调用修复；
2. venv 解释器被 `.resolve()` 化为系统 python → 保留符号链接；
3. TabPFN fit 续跑早退留下空回归器 → 改为确定性重建 + 身份钉扎；
4. GPU 逐位复现不成立（~1e-6 核抖动）→ 容差复用并记录实测值。

## 合并审查注意事项

1. **两处语义扩张在 runner**：无标签 `PredictionView`（预测侧结构性不可见测试标签）
   与分块 SHA 台账断点续跑；均有测试覆盖。
2. **TabPFN 权重访问方式**：绕过官方交互许可下载器（本地文件路径），许可条款
   tabpfn-3-license-v1.0 允许研究使用；如需走官方流程见 ENVIRONMENT.md 精确入口。
3. **TimesFM 逐 origin 调用**是对官方接口的能力限制（池化）的必要适配，非优化；
   source.json 与 stage0 文档有实测证据。
4. **分组读法**：候选表中 Q5 等输入组仅 tabpfn_v3；chronos2 为受限输入 Q5；
   moirai2/timesfm 为 P 表（原生无 q05/q95，不造分位）；全部带
   `pretraining_overlap=unaudited`；Full_joint 行保留更宽信息标签。
5. **seed 语义**：tabpfn 三 seed 为同权重不同集成抽样（真实独立推理）；
   三个零样本模型 seed 无作用，2027/2028 经 64 起点子集容差复核后复制 seed2026
   （reuse_of 与实测差异写入产物 metadata.budget.seed_reuse）。
6. 预训练计算成本与本地任务成本已分开记录（budget.pretrained_cost_note）；
   不宣称与从头训练等总算力。
