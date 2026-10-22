# zcode 第一批基线交付：Codex 独立验收

后续状态：已取消 DDNN-JSU 分位钳位，按原实际预算重跑9格并完成独立验证。
修复结果与当前153格总表见
[DDNN-JSU 修复报告](../logs/forecasting/ddnn_jsu_unclipped_20261022_r1/README.md)。
以下保留修复前的验收记录，原钳位结果只作历史审计。

审查分支 `research/fair-baseline-expansion-20261022`，提交 `b364c89`。
结论：45 格新增实验确实全部完成；153 格归档分数可独立复算。
但 DDNN-JSU 的概率结果受未如实披露的逆 CDF 钳位影响，暂不能将整批交付无条件验收为标准方法对照。
本次保留 zcode 的分支、检查点和原始结果，尚未合并到 Anchor 分支。

## 已核验

- 五模型均有独立实现及配置目录，合并表包含 17 个模型、3 地区、3 种子。
- 独立调用 `build_multi_run_report`，153 格全部完成、哈希/评分/跨运行对齐检查通过，重算 tables 与归档完全相同。
- 在 zcode 独立环境重跑 124 项相关测试，全部通过（扩展模型、校准、冻结树、公共输入、评分、调度、扩展、报告、目录布局）。
- 检查点仅由训练/选择阶段决定；校准系数拟合与评分流程有阶段隔离测试。
- 此处的重算一致证明归档与实现一致，不等于实现已经满足论文定义和全部公平约束。

## 阻断项：JSU 正式分位被钳位

`forecasting/ddnn_jsu/model.py:124` 在解析分位中执行
`argument = torch.clamp(argument, -ARG_CLAMP, ARG_CLAMP)`，其中 `ARG_CLAMP=12`。
NLL 对应未截断 Johnson SU，但输出使用被钳位的 sinh 参数；两者不再是同一分布。
代码称收敛后不会触发，本轮实际检查点否定了这一假设。

用九个正式检查点检查 selection/test 两阶段，共 18 组：

- NSW1 三个种子的 test 分别触发 49、38、43 个分位值，共 130 个；均位于上尾，q50 没有触发。
- NSW1 seed 2027 的 selection 还触发 1 个分位值，因此不能只改最终导出而继续沿用未经复核的选模。
- QLD1/TAS1 的这些检查点未触发。此检查不涵盖所有历史 epoch。

对 NSW1 三个种子以完整测试 batch 重新推理：保留钳位时预测与归档逐位相同，最大误差 0。
仅取消钳位、保持模型权重和评分公式不变，结果如下（原始价格空间）：

| 种子 | 归档/钳位 QS | 同检查点未钳位 QS | test 钳位分位数 |
|---|---:|---:|---:|
| 2026 | 45.850548 | 651.496444 | 49 |
| 2027 | 44.150681 | 141.383095 | 38 |
| 2028 | 47.086936 | 244.506730 | 43 |

未钳位预测仍为有限值，不是必须通过裁剪才能表示的浮点溢出。
这些数值仅用于定位实现影响，**不是修复后重新选模的正式基线成绩**。
此前的 DDNN 宏 QS=40.8275 应标记为带钳位实现的旧成绩，不能继续写作标准解析 JSU。

建议修复边界：

1. 标准 JSU 分位采用未截断公式，必要时用 float64 计算尾部；遇到非有限值明确报错，不静默截断。
2. 增加参数越过 ±12 时与 SciPy `johnsonsu.ppf` 的等价测试、NLL 与 logpdf 等价测试。
3. 同步修正文档：`source.json` 中旧 NLL 的 delta/lambda 符号和 cosh 自变量有误，README 也未更新正态变换参数化。
4. 保持原数据、预算、候选和选择规则，另建运行重新训练/选择 DDNN 的 9 格；不能使用本次测试分数调优参数。
5. 完整保留旧 45 格及冻结来源，合并报告以明确的新来源替换旧 DDNN 行，保留其余 36 格。

## 需要收紧的结果解释

- “新增模型未超过 LightGBM”只适用于当前宏 MAE/QS。LightGBM-QRA 的 pooled RMSE 为 269.4659，低于原 LightGBM 的 277.0644；应报告指标权衡。
- QRA 和 Anchor 都出现校正后退化，不足以证明根因相同。尚需对校准期/评估期的偏差、价格分组、中心与宽度贡献做分解。
- 17 模型表中的 `full` 来自历史静态 aligned 运行，不是新一轮 Full_joint/Anchor 确认实验。
- 本轮 Anchor 确认使用 2021/2022 回顾性开发区间；基线主表使用 2023/2024。不能把两张表的绝对误差直接排名。

## 证据

- `logs/forecasting/baseline_expansion_codex_review_20261022/review.json`
- `logs/forecasting/baseline_expansion_codex_review_20261022/jsu_clamp_counts.json`
- `logs/forecasting/baseline_expansion_codex_review_20261022/jsu_clamp_impact.json`

zcode 原交付和实现位于独立 worktree `/root/workspace/Code/DualTimesField-fair-baselines`。
原运行与冻结代码保持不变，本审查只记录事实与验收要求。
