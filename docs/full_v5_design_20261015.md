# V5 Calibrated History — Design (frozen 2026-10-15)

**Name**: V5 Calibrated History — 冻结 LightGBM + 受限位置/尺度校正 + 状态匹配成熟残差记忆。

本文件是 round-2 冻结规范（`reports/round2_codex_review.md`）的实现设计。两个且只有两个核心假设：

- **H_cal**：低容量校正能改善冻结 LightGBM。
- **H_hist**：状态匹配历史能超过同一校正器使用的简单成熟残差统计，以及同容量的近期日历历史对照。

H_cal 成立不代表 H_hist 成立；两者都不等价于 V4 双场机制成立。V5 不借 DualTimesField 重建成绩背书电价预测，不自称"完整 DualTimesField 复现"。

## 1. 数据边界与协议

三地区（NSW1/QLD1/TAS1）、72h 输入、逐小时 origin、24h 输出、五分位 (0.05, 0.10, 0.50, 0.90, 0.95)。全部时间边界使用 `Australia/Brisbane`（无夏令时），按 Unix 秒计算。

| 时间段 | 用途 |
|---|---|
| 2015–2021 | 冻结 LightGBM 拟合期；不重训 |
| 2022H1 | 原早停/选择期；仅作来源记录 |
| Dfit 2022-07-01 → 2022-10-01 | 新校正器与拟合型对照的开发拟合段 |
| V1 2022-10-01 → 2022-12-01 | 开发验证 1 |
| V2 2022-12-01 → 2023-01-01 | 开发验证 2 |
| 2023–2024 | 已污染的最终回顾性评价；冻结后一次运行 |

完整 24h 标签窗口必须落在段内（origin ≥ start 且 origin+23h < end）。拟合行还要求标签在拟合截止点前可用：数据校验强制每条小时观测 `available_at = delivery_start + 1h`，故 origin j 的可用时刻 `a_j = j + 24h`；Dfit 拟合行要求 `a_j < 2022-10-01`，H2 最终重拟合行要求 `a_j < 2023-01-01`。被截止点排除的行数写入 dev 报告。V1、V2 使用同一个仅在 Dfit 拟合的模型。通过开发门禁后，所有拟合型对照按同一规则在完整 2022H2 重拟合（最终拟合，不是再选结构）。2022H2 正式降级为开发数据。

**查询时间规则**（所有历史特征）：查询 t 只能使用 `j < t` 且 `a_j ≤ t` 的库记录。该规则同时覆盖开发段与最终评价：最终 test 期间参数冻结，库按既定规则顺序吸收已成熟的较早标签（固定算法的顺序历史更新，不是重训）。实现上 FeatureEngine 对每个查询用 `origin ≤ t − 24h` 的上界切片，结构上保证"先预测后吸收"的信息流；预测分块持久化。

**严格 OOF 规则**：本轮不新建前缀 OOF、不重训任何 LightGBM。合法性来自冻结基线未用 2022H2 标签拟合或选择，校正器使用其后续样本外预测（"后续开发段校准"）。禁止进入残差库的预测：原基线训练内预测、被早停使用的 2022H1 预测、任何使用当前查询或未来标签生成的统计。残差库从 2022-07-01 开始。

## 2. 特征

### 2.1 当前状态 z（16 列，固定）

1. 常数 1；
2. B_q50/300（截断 [-1,1]）；
3. (B_q90−B_q10)/300（截断 [0,1]）；
4–12. predispatch price、predispatch demand、available generation、spare capacity、net load、import room、5% shortfall、predispatch coverage、interconnector coverage——直接使用冻结缓存的训练期标准化值，截断 [-3,3] 后除以 3（future_exogenous 列序经审计冻结：max_spare, predispatch_rrp, predispatch_total_demand_mw, predispatch_available_generation_mw, predispatch_net_interchange_mw, predispatch_covered, ic_net_import_mw, ic_import_room_mw, ic_export_room_mw, ic_main_import_room_mw, ic_covered；net load 取 ctf_exogenous 的 net_load_mw；shortfall 取 quantile_exogenous 的 q0.05 列）；
13–16. 目标交付小时 sin/cos、目标交付日 weekend、星期 sin（由 target timestamp 按 AEST 计算）。

### 2.2 简单成熟残差统计 m（8 列，固定）

同地区、同 lead，按历史 origin 最近 7/28 天：残差中位数（7d/28d）、绝对残差中位数（7d/28d）、28 天 80%/90% 区间覆盖率减名义值、有效数量/168、/672。价格统计除以 300 并截断；窗口少于 24 条成熟记录时统计置零、数量列保留实际比例。固定 lead 后每个交付小时只对应一条记录（origin 唯一性保证，不重复计数）。

### 2.3 状态匹配历史 v（2 列）

候选池：同地区、同 lead、origin 的 AEST 小时与查询相同、过去 365 天内、完整 24h 标签已成熟、来自合法样本外残差库。距离只使用六个预测时已知变量（B_q50、80% 宽度、predispatch price、spare capacity、net load、import room，全部用上述有界变换），六维等权平方欧氏距离，取最近 K=16，距离相同按 origin 升序。少于 8 个候选或 28 天统计不足时历史分支关闭（两列严格为 0，预测严格回到无历史校正器）。

v_loc = η·clip((median_N r − median_28d r)/300, −1, 1)；v_abs = η·clip((median_N |r| − median_28d |r|)/300, −1, 1)；η = k/16。

C_recent：同一候选池、同容量，仅把"按距离取 16"换成"取最近 16 个 origin"，其余规则（η、关闭条件）完全一致。

## 3. 模型与训练目标

horizon 四组（0–5、6–11、12–17、18–23），组内共享参数，每地区独立拟合。

x = [z, m, v]（26 列），u = [1, z_width, z_spare, m_abs28, v_abs]（5 列）。

δ = 150·aᵀx，s = 1 + cᵀu，Q_q = B_q50 + δ + s·(B_q − B_q50)。约束 ‖a‖₁ ≤ 1、‖c‖₁ ≤ 0.5 ⇒ |δ| ≤ 150、0.5 ≤ s ≤ 1.5，正尺度保证分位单调。每地区最多 124 参数（4 组 × (26+5)）。零参数精确回到 B。

目标（per 地区，均值 over 全部 (t,h) 拟合行）：

L = mean[(1/5)Σ_q ρ_q((Y−Q_q)/300) + 0.2·ρ_.5((Y−Q_.5)/300)] + (1e-3/4)·Σ_g(‖a_g‖₁+‖c_g‖₁)

等价于五分位 pinball 权重 (0.2, 0.2, 0.4, 0.2, 0.2)（q50 项合并）。问题按 (地区, horizon 组) 拆成稀疏 LP（pinball 辅助变量 + 参数正部分解），SciPy HiGHS 求解；保存求解器版本、状态、目标值、约束残差。要求约束误差 ≤ 1e-6，目标不得劣于零参数可行解超过数值容差，否则标记数值失败并以失败处理（不得换解）。对照的特征子集（u 依 x 可用特征同步收窄，为规范未显式定义处的最小一致解释）：

| ID | x | u | 参数/地区 |
|---|---|---|---|
| C_const | [1] | [1] | 8 |
| C_state | z | [1, z_width, z_spare] | 76 |
| C_roll | z+m | [1, z_width, z_spare, m_abs28] | 112 |
| C_recent | z+m+v_recent | 全 5 | 124 |
| V5_history | z+m+v | 全 5 | 124 |

A_stack：每地区每 horizon 组一个 w_g ∈ [0,1]，Q = (1−w)B + w·V4_full（五分位共享），同一开发目标加固定 1e-3·mean_g w_g 收缩；同样 LP 求解。不加门控、不加第二套权重。

T_roll / T_history：每地区 5 个 pooled-horizon quantile booster（输入 1465 公共列 + z + m (+ v) + h/23 + 四组 one-hot，目标原始价格 Y），200 rounds、lr 0.05、num_leaves 15、max_depth 4、min_child_samples 200、max_bin 63、feature_fraction 1、bagging_fraction 1、seed 2026、4 线程、不早停、不搜索。两对照每阶段 30 booster，是固定预算强对照而非调优上限。

## 4. 诊断与评价

所有模型经公共评分器（`forecasting/aligned_scoring.py`，只读）评分：MAE（rearranged q50）、QS（两倍五分位平均 pinball，非精确 CRPS）、coverage/宽度。另报真正 pooled RMSE 与地区宏 RMSE（分别命名）。负价 / 0–300 / >300 条件误差、偏差与整体 MAE 贡献分解（三段贡献之和恰为 MAE）；δ、s 与边界触及比例；空库/不足样本回退比例；历史候选数量、年龄、距离、选中 ID（抽样保存）与查询对应关系；z/m/v 在 Dfit/V1/V2 的分布分位数；开发段组合 oracle（段内拟合，仅描述上限）；q50>300 召回/精确率/误报（仅描述）。V4 force_null 前端诊断若未运行，在未完成项中如实列出。

统计分析：先对同一 origin 的 24 个 horizon 损失取平均，三地区联合抽取相同 168h 时间块，9,999 次配对 bootstrap（24h 为敏感性）；四个预声明主比较（V5−C_roll、V5−C_recent × MAE、QS）作 Holm 校正。短开发期统计功效有限；"不显著"不证明等效，显著性不恢复盲测资格。

## 5. 门禁与停止条件

- **G0**：身份与正确性——原模型/数据/缓存/特征顺序/发布时间映射核验 + 指定测试全部通过。
- **G1**（固定 Dfit 模型，V1+V2 合并）：V5_history 相对 B 宏 MAE、QS 均改善 ≥1%；任一地区 MAE 恶化 ≤1%；各地区 0–300 条件 MAE 恶化 ≤0.5%。
- **G2**（同验证集）：相对 C_roll、C_recent 宏 MAE、QS 均改善 ≥0.5%；V1、V2 分别比较时两指标方向不反转（对每个对照、每个分段、每个指标要求 ref−V5 ≥ 0）；完整报告与 T_roll/T_history/A_stack 的关系。
- **G3**：通过后所有方法在完整 H2 重拟合、冻结整张矩阵、一次性运行三地区 2023–2024。test 结果只判定冻结假设是否获回顾性支持，不触发任何换候选/换权重/改邻居/新训练。

解释规则（预声明）：C_roll 同效 → 无状态匹配增量证据；C_recent 同效 → 无匹配优于近期/日历历史证据；T_history 同样获益 → 支持历史信息但不能宣称专门结构必要；A_stack 获益 → 支持产物互补而非历史机制；单一地区改善不替代三地区判断。失败结论只能写为"在本次冻结配置、数据权限及评价时期下，未发现满足门槛的收益"。

预算：seed 2026；总并发 ≤4 CPU 线程；GPU 0；新前端训练 0；前缀 LightGBM 重训 0；超参搜索 0；开发阶段 ≤24 CPU 核时；最终重拟合与评价另 ≤24 CPU 核时。超预算则保存进度并报告"未完成/预算不足"。

## 6. 工程边界与输出

新增文件仅限 `forecasting/full_v5/`、`configs/full_v5/`、`tests/test_full_v5_*.py`、`docs/full_v5_*.md`。不改 `forecasting/full_v3`、`full_v4`、公共评分器或既有校验；复用一律只读（FrozenLightGBMPredictor、公共缓存、V4 预测产物、公共评分契约）。

输出根：`outputs/forecasting/v5_calibrated_history_20261015_r1/`（config.yaml、design.md、source/、manifest.json、identity/、baseline_predictions/、history_audit/、development/、final_models/、freeze.json、final_test/、reports/、state.json、budget.json、commands.jsonl）。归档日志：`logs/forecasting/v5_calibrated_history_20261015_r1/`。旧目录完全只读；新目录已存在且身份不匹配时直接拒绝。

身份冻结：prepare 记录源码/配置/schema/基线/V4/缓存/外部数据哈希；dev/freeze/final-test/report 每步复核，任何改动即拒绝。freeze 后 final-test 校验 freeze.json（身份摘要 + 全部模型文件哈希）；已完成最终评价不得再次运行。开发阶段禁止打开 test：数据层 `require_stage('final-test')` 在文件打开前抛错，测试以真实访问验证。测试结果以 `development/tests_passed.json`（操作者记录）作为 G0 输入。

## 7. Round-5 修正（2026-10-16，Codex round-4 审计阻断项）

按 round-4 只读审计修复以下工程缺陷；不改候选、门槛、K、特征、数据权限或任何冻结常量：

1. **树对照 one-hot 初始化（H1）**：`controls.tree_feature_matrix` 改为 `np.zeros` 全量确定性零初始化，激活组列置 1 前四个组列全部为 0；新增内存投毒回归测试（四列均为二值、每行和恰为 1）。r1 的 T_roll/T_history 证据因此作废，r2 重跑。
2. **身份与恢复绑定（H2）**：identity 增加三个 V5 测试文件、复用依赖源（aligned_inputs/aligned_scoring/lightgbm.predictor）、prepared baseline 文件哈希与 origin/target/quantile/actual 网格哈希；dev done marker 输出清单纳入该地区全部模型与预测文件；恢复/完成态重入时逐哈希复核，任何不一致直接拒绝（不再静默重算）。
3. **test 禁读与最终输入身份（M1）**：`V4Source.artifact/quantiles/file_digest` 与 `load_prepared_baseline` 对 test split 在任何文件句柄创建前执行 `require_stage('final-test')`；freeze 冻结 test 侧来源哈希（缓存 test 条目、基线 test.npz、V4 test 预测），final-test 先验哈希再逐元素校验 origin/target-time/actual 网格。
4. **顺序最终评价契约（M2）**：final-test 改为显式 `MaturedQueue` 顺序执行——逐块释放已成熟标签（available_at ≤ 块内最后查询时刻，engine 逐查询成熟掩码保证块内精确性）、先预测并原子持久化当前块再前进；原子运行锁（live-pid 拒绝并发、死 pid 自愈）；断点续跑只接受同身份且哈希一致的已持久化块，篡改即拒绝；已预测 origin 对其后才成熟的标签改动免疫（合成数据测试）。
5. **bootstrap 归一化与效应方向（M3）**：每次抽样按实际抽到的块计数重新归一化（部分/尾部块），恒定差恰还原；效应符号统一为 ref−V5（正值=V5 更优），报告标签与实现一致。
6. **预算执行（M4）**：dev/final 各阶段读取 `budget.dev_core_hours/final_core_hours` 并在阶段内检查；异常/中断路径同样累计 CPU 用量、记录 commands.jsonl 并把 state 置为 interrupted。
7. **分段特征分布（M5）**：Dfit/V1/V2 三段 × z/m/v 三组分别保存分布摘要（修复原先以 26 列全矩阵冒充 z 组的问题）。
8. **数值容差与措辞（L1/L2）**：G2 方向判定采用 1e-9 容差并保留原始差值；"逐位相同"改为"数值精度内等价"并报告 max|Δ|；最终报告污染声明按实际执行状态渲染（未运行则明确声明未运行）；`full_v5_baseline_identity.md` 修正 QLD1 缓存清单哈希。
