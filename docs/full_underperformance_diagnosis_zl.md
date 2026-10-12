# Full 落后 XGBoost 的证据驱动根因诊断

- 作者：zl
- 分支：`diag/full-underperformance-zl`（起点 `exp/fair-sota-plan-zl` @ `6f669b5f49b767c95ed5dc00902998f0785ce5fc`，本地与远端一致，工作区干净）
- 本报告保留初次诊断的汇总分解和历史证据；当前研究顺序、资源状态与证据等级以[修订计划](fair_sota_plan_zl.md)为准，不再以本报告的历史建议覆盖计划。
- 状态：**本报告只使用仓库内已有产物（冻结运行 `exploration_20261011` 的汇总报告、早期已入库的训练曲线与 metrics）和代码审计。没有重新训练任何模型。**

## 修订状态：旧产物复核已更新部分判断

2026-10-12 在当前工作区找到了冻结预测与曲线，完成只读检查，未重新训练。新增证据和复核命令集中在[修订计划 §2、§12](fair_sota_plan_zl.md#2-证据更新与旧判断修正)：

- 36 个 Full/XGBoost 预测文件哈希、来源、两模型 origin/actual 对齐通过，18 个测试 MAE 复现旧报告；九个最佳 checkpoint 哈希与元数据一致。原“缺口 1/2”是初次诊断机器的限制，不代表当前工作区。
- Full 测试宏 MAE：点头 38.922885、q50 42.089025；验证宏 MAE：点头 76.267012、q50 80.895386。直接改用 q50 未改善均值；尚未补全切片/配对区间，不能据此排除重新训练 L1 等干预。
- 冻结 Full 已有 `calibrator_hidden=32` 和学习型 router；完整模型不是纯线性头。尾部加权同时作用于全部分位数，旧 `no_tail` 的点头 MAE 不能排除其对 q50 的影响。
- 冻结曲线已可读取，CRPS~ 最佳 epoch 分别为 NSW1 2/1/2、QLD1 20/19/23、TAS1 6/4/8；点头 MAE 最佳为 1/0/1、1/5/1、0/0/0（均从 0 计数）。需要分离选点规则与优化机制，不能笼统声称所有地区都很早收敛。

以下 Q1/Q2 的旧点头误差分解继续有效；Q3/Q4 的原始分级、Q5 和 §6/§8 的建议记录为历史分析，存在冲突时采用修订计划。特别是“文件缺失”“q50 尚无法计算”“首次添加非线性头”的行动前提已不成立。

## 0. 初次诊断的范围、输入与限制（历史环境记录）

| 项目 | 事实 |
|---|---|
| 分析对象 | 静态协议（训练 2015–2021，验证 2022，测试 2023–2024），NSW1/QLD1/TAS1 × seed 2026/2027/2028，`full` vs `xgboost`；滚动、capped650 协议只用于交叉验证结论 |
| 主输入 | `logs/forecasting/exploration_20261011/reports/report.json.gz`（sha256 `07ade862cf1a556ea9b1f89681d39065e41254b578fb5bdd782515c7fca215f9`） |
| 辅助输入 | `outputs/forecasting/*/*/training_history.csv`、`metrics.json`（39 个早期已入库运行；**不是**冻结运行本身） |
| **缺口 1** | 冻结运行的逐 origin 预测文件（`test.npz`、`validation.npz`，含 `point` 与 `quantile`）和训练曲线只在训练机器上（README：“model/prediction outputs remain local”），**不在仓库、也不在本机**。因此“用统一 q50 重算 Full 的 MAE”“Full point head 与 q50 各自结果”“冻结运行各 seed 的收敛曲线/最佳 epoch”**无法完成**。这是本报告最大的证据缺口。 |
| **缺口 2** | 数据文件（`../electricity-price-forecasting-research/data/...`、`data/aemo_exogenous/*.npz`）不在本机；本机无 torch/xgboost/pandas（系统 Python 3.9.6）。因此无法做单地区 smoke 训练。该部分按任务要求记录为后续需求并停止，没有伪造 smoke。 |
| 评分口径 | CRPS~ = 5 个分位 pinball 均值的 2 倍（近似，不是精确 CRPS 积分）。数值取 3 seed 均值；宏平均 = 三地区等权。 |
| 差值符号 | 一律 `Full − XGBoost`，**正 = Full 更差**。 |
| 价格区间 | 按**事后实现价格**分段（negative <0，0–100，100–300，spike >300）。这是恒等式分解（各段观测数之和等于总数，已验证精确到 1e-14），不是可提前识别的状态；不能据此说“模型在 spike 前就知道”。 |

复现命令（只读已有产物，不覆盖结果；仅标准库）：

```bash
cd /Volumes/AI_WORKSPACE/01_Chief/DualTimesField
python3 forecasting/analysis/full_vs_xgboost_gap.py          # 写 logs/forecasting/full_diagnosis_20261012/{gap_decomposition.json,gap_decomposition.md}
python3 -m unittest tests/test_full_vs_xgboost_gap.py        # 6 tests
```

重复运行两次，`gap_decomposition.json` 与 `.md` 的 sha256 完全相同（输出确定）。

## 1. 直接回答五个核心问题

### Q1. 宏平均 MAE / CRPS~ 落后的直接来源

| 指标 | Full | XGBoost | 宏差距 | NSW1 | QLD1 | TAS1 |
|---|---:|---:|---:|---:|---:|---:|
| MAE | 38.923 | 37.968 | **+0.955** | +0.522（18%） | +0.054（2%） | **+2.289（80%）** |
| CRPS~ | 25.936 | 25.527 | **+0.409** | **+0.956（78%）** | −0.219（−18%） | +0.491（40%） |

（括号内为占宏差距的比例；比例之和为 100%。）这些宏平均与 `report_static.md` 的表一致（测试已断言）。

直接来源——**不是尾部，而是普通价格区间的精度**：

| 价格区间（事后） | MAE 贡献 | 占宏 MAE 差距 | CRPS~ 贡献 | 占宏 CRPS~ 差距 |
|---|---:|---:|---:|---:|
| negative | +0.027 | +3% | −0.398 | −97% |
| 0–100 | +0.830 | +87% | +0.845 | +207% |
| 100–300 | +0.800 | +84% | +0.518 | +127% |
| spike（>300） | **−0.702** | **−73%** | **−0.557** | **−136%** |
| 合计 | +0.955 | 100% | +0.409 | 100% |

即：Full 在 0–300 价格区间**落后 1.63（MAE）和 1.36（CRPS~）**，被 negative 与 spike 区间的**领先**部分抵消。Full 在 spike 与 negative 上优于 XGBoost，这与 README 的“尾部互有胜负”一致，但现在有了定量分解。

交叉验证（独立于上表）：
- capped650 协议（价格封顶 650，削弱 spike 的抵消作用）下，宏 MAE 差距**扩大**到 +1.547（CRPS~ +0.190）；滚动季度重训协议宏 MAE +1.636、CRPS~ +1.643（Full 36.868/24.749，XGBoost 35.232/23.106）。
- 说明静态协议里 +0.955 的 MAE 差距被 spike 区间“压低”过，真实的普通状态差距更大。
- 平方误差口径与 MAE 相反：Full 的窗口 RMSE 在 NSW1（96.64 vs 98.06）、QLD1（84.58 vs 86.38）**优于** XGBoost，宏 RMSE 差 −0.412；pooled RMSE 在 NSW1/QLD1 也更低。Full 的误差更少出现大偏差，但**中心（绝对误差）更差**。

### Q2. 差距集中在哪里

- **地区**：MAE 差距 80% 来自 TAS1；CRPS~ 差距 78% 来自 NSW1；QLD1 的 MAE 无差异、CRPS~ Full 更好。配对 block bootstrap（报告内，19,999 次）：MAE 宏差 −0.955（XGBoost 相对 Full），CI [−1.317, −0.596]，Holm p=0.049（168h block）/0.012（24h block）；TAS1 MAE、TAS1 CRPS~ 通过 Holm；**宏 CRPS~ 的 168h CI 为 [−0.803, +0.044]，含 0，Holm p=1**（24h block CI [−0.721, −0.039]，Holm 后仍不显著）。因此“MAE 落后成立；CRPS~ 落后证据不足”。
- **时段**：并非少数异常月份。MAE 在 72 个 region×月中 51 个 Full 更差（CRPS~ 56/72）；前 8 个正贡献月份只占宏 MAE 差距的 47%。按半年：MAE 差距 2023H1 +0.290、2023H2 +0.111、2024H1 +0.204、2024H2 +0.350，**四个半年都为正**；CRPS~ 为 +0.259、+0.181、+0.046、−0.077。结论：系统性、普遍性，不是事件驱动。
- **horizon**：72 个 region×horizon 中 58 个 MAE Full 更差（CRPS~ 57/72）。宏 MAE 贡献按 1–6/7–12/13–18/19–24 小时块为 +0.315/+0.253/+0.267/+0.120（贡献随 horizon 增大而减小）；horizon 1 的单点差距最大（MAE +2.235、CRPS~ +1.140，约为平均的 2.3 倍），horizon 22–24 接近 0。这与“差距不在长 horizon 的外推能力”一致（反而是近端最差）。
- **价格状态**：见 Q1 表。分地区看 MAE 差距最大的单个切片是 TAS1 0–100（+0.463，占 48.5%）、NSW1 100–300（+0.340）、TAS1 100–300（+0.265）、QLD1 0–100（+0.235）。CRPS~ 最大切片：NSW1 0–100（+0.505，宏差距的 123%）、NSW1 100–300（+0.438，107%）；spike/negative 的负贡献（Full 更好）部分抵消。
- **预测分量**：
  - 区间宽度/覆盖（描述性）：NSW1 的 Full 90% 区间宽 183.6 vs 151.8，覆盖 0.870 vs 0.836；在 NSW1 的 0–100 区间 Full 宽 101.2 vs 89.8，覆盖几乎相同（0.893 vs 0.892）——更宽但没有更高覆盖，对应 NSW1 0–100 的 CRPS~ 差 +2.6（10.17 vs 7.58）。这指向 Full 的**中间分位在普通小时上更宽、信息量更低**，而不是校准崩溃。
  - 整体校准 Full 优于 XGBoost（覆盖 NSW1/QLD1/TAS1：0.870/0.915/0.866 vs 0.836/0.912/0.805，名义 90%）。分位交叉率（重排前）两者均为 0。
  - point head 与 q50 的分别结果：**无法计算**（缺口 1）。

### Q3. 候选原因逐项评级

下表“Full 自身”指该问题属于 Full，而非 XGBoost 协议优势或其他基线实现问题。

| # | 候选原因 | 归属 | 等级 | 要点 |
|---|---|---|---|---|
| A | 点预测定义、训练目标与 checkpoint 选择错位 | **Full 自身** | **Confirmed（存在性）/ Hypothesis（对差距的贡献量）** | 见 §2-F2 |
| B | 差距在普通价格区间而非尾部 | Full 自身（定位） | **Confirmed** | 见 Q1，精确分解 + capped650 交叉验证 |
| C | 线性头 + 年份趋势特征在样本外年份外推 | **Full 自身** | **Hypothesis**（有代码依据） | 见 §2-F4 |
| D | 优化/验证不稳定：验证指标逐 epoch 剧烈波动、最佳 epoch 很早 | Full 自身 | **Strong evidence（对同族早期运行）；对冻结运行为 Hypothesis** | 见 §2-F3 |
| E | 容量/归纳偏置：线性头对 PD 价格等外生量响应欠灵活 | Full 自身 | **Strong evidence（定位）/ Hypothesis（机制）** | Full 追回线性→XGBoost MAE 差的 74%，剩余差在普通区间；见 §2-F5 |
| F | 输入信息利用不足（字段列表相同≠利用方式相同） | Full 自身 | Hypothesis | `no_pd` 令 MAE +3.95，说明 PD 被用到；XGBoost 是否更充分无直接证据 |
| G | 损失与主指标错位（tail 加权、decomposition 占比） | Full 自身 | 见 §2-F6：**tail 加权不是差距根因（反证）**；decomposition 稀释 = Hypothesis | |
| H | 校准/后处理 | Full 自身 | **证据不支持作为根因** | `full+floor` 相对 `full`：MAE +0.045、CRPS~ +0.065（更差）；Full 整体覆盖已接近名义值 |
| I | 实现缺陷（数据/切分/缩放/泄漏） | — | **审计未发现缺陷（Confirmed：所审路径无问题）** | 见 §3 |
| J | XGBoost 协议优势：24 个 horizon 分别拟合并分别早停；直接优化原始价格 pinball，q50 即点预测 | **XGBoost 协议优势** | **Confirmed（代码）**，贡献量未测 | 见 §4 |
| K | PatchTST 的 RevIN/目标空间缺陷、神经基线固定 lr/30 epoch | **其他神经基线实现问题** | Confirmed（README+代码），**与 Full 根因无关** | 见 §4 |

### Q4. 哪些已证实，哪些必须新实验

- **已由代码与已有产物证实**：B（位置）、I（审计无缺陷）、H（后处理不是原因）、A 的“设计错位”事实、J 的协议差异、G 中 tail 加权的净作用、稳健性（不是 seed 运气、不是少数月份）。
- **必须新实验才能确认**：A 的贡献量（需冻结运行的 `test.npz` 或重训）、C（年份特征截断对照）、D（冻结运行的曲线/最佳 epoch、EMA/多 checkpoint 平均）、E/F（非线性头对照）、J 的贡献量（XGBoost 改为 joint-horizon 对照）。见 §6。

### Q5. 历史最小修改建议（顺序已由修订计划替代）

1. **点预测口径统一 + checkpoint 选择与 MAE 对齐**（改善 MAE，次要改善 CRPS~）。依据：Confirmed 的错位事实；MAE 差距集中在普通区间；RMSE 与 MAE 反向（Full 的点头更像“平方误差型”估计）；零训练即可先验证（用 q50 重算）。**限制**：早期 39 个运行中，按验证 MAE 选 checkpoint 相比按验证 total 选，只有 11/39 在测试 MAE 上更好，平均绝对变化 0.581（最大 6.154），与宏差距 0.955 同量级。所以“单纯换选择指标”不保证有效，需配合平均（EMA/checkpoint 集成）降低噪声。
2. **point 头损失改为 raw-price L1/分位 0.5 pinball（或直接以 q50 作点预测）**（MAE）。依据同上；对应 XGBoost 的点预测就是原始价格的 q50。
3. **对 `year_standardized`（及其他连续输入）按训练范围截断**（MAE 与 CRPS~，主要影响 2023–2024 样本外年份）。依据：代码事实 + 验证/测试差距不稳定（见 §2-F4）；单因素、成本最低的训练对照。
4. **降低优化噪声：更小学习率/EMA/权重平均，patience 取验证 MAE+CRPS~ 综合**（MAE/CRPS~）。依据：§2-F3。
5. **调 `tail_weight`（0.5→0.25 等，仅用验证集）**（CRPS~）：tail 加权把 CRPS~ 从普通区间挪到尾部，净收益为正（0.329），但在普通区间有成本；最优值未知。
6. 非线性小头（对每小时 PD/备用/净负荷的小 MLP 或分段线性）（MAE）：支持证据最弱（Hypothesis），放最后。

## 2. Full 自身问题的证据链

### F1（Confirmed）MAE 缺口是“普通区间精度”缺口，且分布广

证据：Q1/Q2 表；18 个切片分解的残差最大 1.1e-14；capped650 下 MAE 缺口扩大到 +1.547；三地区中 NSW1/TAS1 的 3 个 Full seed 全部劣于 XGBoost 的 MAE 均值（QLD1 为 3 个 seed 中 1 个更好），Full 的 seed 极差（TAS1 30.377–32.076）小于缺口量级，所以不是 seed 运气。
反证可能：区间按事后价格划分，spike 的 −0.70 抵消属于恒等式，不能说明“Full 擅长预测 spike”；它说明 Full 在已发生的 spike 小时误差较小。
结论：必须以“普通状态点/中间分位精度”为主要整改对象，而不是继续堆尾部机制。

### F2（Confirmed 设计错位；贡献量未测）点预测、训练目标与选择指标三者不一致

代码事实（逐文件）：
- `forecasting/losses.py`：`point_loss_type` 默认 `mse`，`mse_loss = F.mse_loss(point_forecast, target_price)`，`target_price` 是 asinh 变换并标准化后的价格（`datasets.py`：`price_transform.forward` 后减均值除标准差）。配置 `loss.point_weight: 1.0`。
- `forecasting/train.py::evaluate`：`point = dataset.denormalize_target(outputs["point_forecast"])`，MAE/RMSE 由该 point 头计算；`crps_quantile_approx` 由分位头（`quantile_target: price`，在**原始价格**标准化空间做 pinball）计算。即 MAE 来自“asinh 空间 MSE 头”，CRPS~ 来自“原始价格 pinball 头”。
- 冻结配置：`training.checkpoint_metrics: [crps]`、`primary_selection: crps`：checkpoint 按**验证 CRPS~**选择，不按验证 MAE。
- XGBoost（`exploration_baselines.py::run_tree`）：`objective="reg:quantileerror"`，目标是原始价格，点预测为 `q[..., 2]`（q50）。

含义：Full 报告的 MAE 来自一个既不是为原始价格 L1 训练、也不是为 MAE 选择 checkpoint 的头；XGBoost 的 MAE 来自直接优化原始价格中位数的模型。asinh 空间的均值/MSE 最优点不等于原始价格中位数，且 MSE 对 spike 的压缩/放大方式与 MAE 不同。

支持性数据（不是因果证明）：Full 在 NSW1/QLD1 的 RMSE 更好、MAE 更差（上述）；MAE 缺口集中在普通区间。
反证可能：若 Full 的 q50 本身 MAE 与点头相当，则缺口来自表示能力而非口径；这正是 §6-E1 要测的。
**数据不足：无法在本机用 q50 重算（缺口 1）。**

### F3（早期同族运行 Strong evidence；冻结运行 Hypothesis）验证指标不稳定且最佳 epoch 很早

在 9 个已入库同族运行 × 3 地区 = 27 条曲线（`pdpasa_dgf`、`gas_price_ctf`、`residual_skip_path` 各 3 seed）上：验证 MAE 最优 epoch ≤ 5 的有 20/27（中位 4），而训练总损失持续下降；验证 MAE 在 epoch 间的取值范围平均 13.0（last−min 平均 9.06）；按验证 MAE 与按验证 total 选出的最佳 epoch 在 20/27 条曲线上不同；39 个运行上两种选择规则的测试 MAE 平均相差 0.581。
解释：不是“没训练好”，而是训练集损失下降与验证 MAE 脱钩（过拟合 / 2022 验证期与训练期分布不同）并伴随 checkpoint 抽样噪声。
限制：这些是**早期配置**（选择规则为 total/mae，无 tail 项、无 `scarcity_router`），**不是冻结运行**；冻结运行的曲线不在仓库。故对冻结 Full 只能列为 Hypothesis。

### F4（Hypothesis，有代码依据）线性头 + `year_standardized` 在样本外年份外推

事实：`calendar_features` 含 `year_standardized`；`_calendar_feature_map` 以训练段的年份均值/标准差标准化（只用训练段，无泄漏）。训练为 2015–2021，验证 2022，测试 2023–2024，所以测试期该特征落在训练范围之外（测试年份 2023–2024 完全落在训练年份 2015–2021 的取值范围之外（由配置的切分边界得出，未逐行验证））。Full 的各个头是 `nn.Linear`（`models.py`：`ctf_point_head`、`dgf_point_head`、`base_point_head` 等），对该特征线性外推；XGBoost 对越界输入饱和。早期 `qld1_diagnosis`（`raw_ours_vs_xgboost.md`/提交 `94d9a4a`）已记录同族模型在 asinh 空间外推产生过度负价预测（<−100 AUD/MWh 预测占 0.23% 小时，实际 0.01%）。
旁证：Full 在验证期（2022）相对 XGBoost 的差距远大于测试期——NSW1 MAE +11.21 vs +0.52，QLD1 +4.31 vs +0.05，CRPS~ NSW1 +4.51 vs +0.96，五个单元里 Full 验证更差（TAS1 的 MAE 为 −0.33）。两者在验证与测试上的排序不稳定，说明 Full 对分布偏移更敏感；但这也可能只是 2022 能源危机期的价格水平（验证 MAE 约为测试的 1.4–2.4 倍：NSW1 64–75 vs 45，QLD1 93–97 vs 40，TAS1 56 vs 29–31）。
反证可能：若截断年份特征后差距不变，则本项排除。

### F5（Strong evidence 定位 / Hypothesis 机制）剩余差距是非线性条件均值精度

- Linear 41.676 → Full 38.923 → XGBoost 37.968（宏 MAE）：Full 追回线性到 XGBoost 差距的 **74%**。`no_fields` 40.018（+1.095，Holm 显著）、`raw_history` 39.583（+0.660，显著）说明 dual-field 重构确实有用，不是负迁移。
- 所有以线性头为主的模型（linear/DLinear/GateTS-adapter）聚在 41.7–41.8；Full 在其上，XGBoost 再上。差距集中在普通区间，这与“线性响应在普通区间不足、树能捕捉交互”一致。
- `no_pd` 令 MAE 增加 3.95（CI [3.496, 4.450]），说明 PD 预调度信息是最关键输入，且 Full 确实在使用它；XGBoost 是否更充分，没有直接证据（需要 XGBoost 的 no_pd 对照，仓库没有）。
反证可能：若把 Full 的点预测换成 q50 后差距消失，则机制是口径（F2），不是容量。

### F6（部分 Confirmed）损失加权与 decomposition 占比

- **tail 加权不是差距根因（Confirmed，反证）**：`no_tail` 相对 XGBoost：CRPS~ 差 +0.738（比 Full 的 +0.409 更差），MAE 差 +0.969（与 Full 的 +0.955 相同）。分段看，`no_tail` 在 0–100/100–300 的 CRPS~ 贡献更小（+0.536/+0.115，Full 为 +0.845/+0.518），但在 spike/negative 更差（+0.327/−0.239 vs −0.557/−0.398）。即 tail 加权**把 CRPS~ 从普通区间挪向尾部，净收益为正**（配对 bootstrap 报告 no_tail−full 的 CRPS~ = +0.329，CI [0.006, 0.715]，Holm 后不显著）。尾部权重是否最优，未知（Hypothesis）。
- **decomposition 稀释（Hypothesis，Full 自身）**：同族早期运行最终训练损失（加权后）中，point 约 54–61%、quantile 仅 8–17%、decomposition 29–34%。但各头是独立线性层，各自只受自己的损失驱动，只有共享的 dual-field 受混合梯度影响，所以“损失量级大 = 稀释”不能直接成立。无冻结运行的分项曲线。

## 3. 代码路径与口径审计（A 部分）

| 审计点 | 结论 | 依据 |
|---|---|---|
| origins、actual、切分、外生特征是否相同 | **相同（Confirmed）** | `configs/aemo_forecast_exploration_20261011_static_{full,xgboost}.yaml` 的 `diff` 仅 `output_directory` 一行不同；两者都经 `build_region_datasets` 构建；报告 `check_sources/check_alignment` 对 origin 网格和 actual 逐元素断言相同 |
| 缩放/asinh 仅在训练段拟合 | **是（Confirmed）** | `datasets.py`：`train_mask = split == "train"`；`_fit_price_transform`、`_fit_standardizer`、`quantile_standardizer`、日历年份标准化、外生标准化（`train_rows`）和 `clip_to_train_quantiles` 均只用训练段 |
| 未来信息泄漏 | 未发现 | `_load_forecast_exogenous` 对 `source_run_unix/stpasa/last_changed > origin` 直接抛异常；但 README 已声明只是元数据代理，未审计真实发布延迟（遗留风险） |
| 分位数排序 | 正确 | `torch.sort(..., dim=-1)`；重排前交叉率 Full/XGBoost 三地区均为 0.0（来自报告） |
| clipping/floor | 无害 | floor 按 seed/region 在训练分位上选、验证选择；`full+floor` 不改善（见 H） |
| checkpoint 选择与最终主指标 | **不一致（见 F2）** | 冻结配置 `primary_selection: crps`，MAE 来自 point 头 |
| 点预测口径 | **不一致（见 F2）** | Full：asinh 标准化空间 MSE 头；XGBoost：原始价格 q50 |
| XGBoost 额外自由度 | **有（Confirmed）** | 24 个 horizon 各自 `fit`、各自 `early_stopping_rounds=50`，早停用 2022 验证集；每个 horizon 的最佳轮数只出现在训练机器的 metrics，不在仓库 |
| XGBoost seed 效应 | 小 | seed 极差 MAE ≤0.07（Full 最大 1.70） |
| Full 的共享参数/联合多步输出/router | 未发现欠拟合证据，但无法排除 | 早期同族曲线显示训练损失持续下降；NSW1 seed 2026 的 dgf 门控均值在 negative/normal/spike 小时分别为 0.27/0.38/0.48（报告 routing 诊断，描述性，非因果） |
| 外生变量进入方式 | **不同（Confirmed）** | XGBoost 把整段 24h 已知外生/分位外生（`fair_tree_features`）作为共享特征并带当前 horizon 的值；Full 的 point 路径：`base_point_head`（线性，吃全部）+ CTF/DGF 专家（线性）+ 小 MLP calibrator。有效容量差异见 F5 |

## 4. 严格区分三类问题

1. **XGBoost 协议优势（不能算作 Full 的缺陷）**：24 个 horizon 分别拟合/早停；原始价格 pinball 直接优化；q50 作点预测；对越界输入饱和。
2. **其他神经基线自己的实现问题（与 Full 无关）**：PatchTST 的 RevIN 用 asinh 历史统计反归一化，监督信号在标准化原始价格空间（README 与 `general_baselines.py`）；PatchTST/GateTS 外生信息只经线性 adapter；神经基线固定 lr 与 30 epoch，GateTS-adapter TAS1 三次均在第 30 个 epoch 取最优，未收敛。**这些不能作为 Full 落后的根因，也不能因此降低 Full 与 XGBoost 差距的严肃性**（Full 对 PatchTST 63.171、GateTS 48.548 的领先，不能被解读为 Full 本身稳健，因为这两个基线有已知缺陷）。
3. **Full 自身问题**：F1–F6。

## 5. 已有产物之外的“已入库曲线与 metrics”：数据与局限

- `forecasting/analysis/full_vs_xgboost_gap.py` 对 27 条同族曲线、39 个 metrics 做的分析（`history_runs`、`selection_rule_sensitivity`）只说明该架构族的行为，**不代表冻结运行**。
- 报告内容里“Full vs XGBoost 相对 2022 验证期的差距”使用报告 `floor_choices` 中的“无 clipping 候选”验证 MAE/CRPS~，是每个 seed 导出 checkpoint 的验证分数，Full 的 checkpoint 已按验证 CRPS~选择，对 Full 乐观，却仍明显更差。

## 6. 初次诊断提出的最小实验（历史计划；当前依赖见修订计划）

| 编号 | 实验 | 一次只改 | 判据 | 预期改善 |
|---|---|---|---|---|
| E1（零训练，优先） | 用冻结运行 `test.npz` / `validation.npz`：分别用 `point` 与 `quantile[...,2]`（q50）重算 Full 的 MAE/CRPS~，并按本报告的价格区间/地区/horizon 分解 | 无（仅重评分） | 若 q50 的 MAE 比 point 低 ≥0.5 → F2 成为主因；若无变化 → F2 排除，转向 F4/F5 | MAE |
| E2 | Full 单地区(TAS1)、单 seed、缩小 origin：checkpoint 按验证 MAE（q50）选择 | 选择指标 | 与 CRPS 选择比较测试 MAE | MAE |
| E3 | 同上，point 头改为原始价格 L1（或 q50 作点） | point 损失 | 验证/测试 MAE | MAE |
| E4 | 同上，`year_standardized` 截断到训练范围（或移除） | 年份特征 | 验证与测试 gap 是否缩小；TAS1/NSW1 最敏感 | MAE、CRPS~ |
| E5 | 同上，EMA/最后 K 个 epoch 平均，用 3 seed | 平均策略 | seed 方差与测试 MAE | MAE、CRPS~ |
| E6 | `tail_weight` ∈ {0, 0.25, 0.5}，仅用验证集选 | tail 权重 | CRPS~ 与普通区间 CRPS~ | CRPS~ |
| E7 | XGBoost 改为 joint-horizon（horizon 作特征、单次早停） | XGBoost 协议 | 差距缩小量=协议优势量 | 判定 J 的贡献量 |
| E8 | 非线性小头（PD/备用的每小时 MLP）加到 `base_point_head` | 头容量 | 普通区间 MAE | MAE |

这些都只用于排除/支持根因，**不能用 smoke 结果宣称 SOTA**。

## 7. 本次变更与验证

| 项目 | 内容 |
|---|---|
| 新增文件 | `docs/full_underperformance_diagnosis_zl.md`（本文）、`forecasting/analysis/full_vs_xgboost_gap.py`、`tests/test_full_vs_xgboost_gap.py`、`logs/forecasting/full_diagnosis_20261012/{gap_decomposition.json,gap_decomposition.md,run_stdout.txt}` |
| 修改文件 | `README.md`（链接）、`docs/fair_sota_plan_zl.md`（§1.3 指向本诊断并更正“仅有假设”的表述） |
| 未触碰 | 任何已有结果、模型、预测文件；未联网下载数据；未访问实验室服务器；未调用 Claude 相关工具 |
| 单元测试 | `python3 -m unittest tests/test_full_vs_xgboost_gap.py`：6 tests OK（分解恒等式、符号约定、与已发布 `report_static.md` 宏均值一致、与 bootstrap 点估计一致、输出确定性） |
| 重复性 | 两次运行输出 sha256 相同；`max |Σ切片贡献 − 地区差距| = 1.1e-14`（18 项） |
| 限制 | 缺口 1/2；冻结运行训练曲线与最佳 epoch 缺失；E1 之前 F2 的贡献量为 Hypothesis；测试期 2023–2024 已被反复查看，不是盲测 |

## 8. 历史后续修复顺序（已由修订计划替代）

1. E1（零训练重评分，需要冻结运行预测文件）——决定 F2 是否是主因。
2. E4（年份特征截断）与 E2/E3（MAE 对齐的选择/损失），单因素、单地区、单 seed。
3. E5（平均/EMA）降低 checkpoint 噪声，再做 3 seed。
4. E6（tail 权重，验证集选）。
5. E7（XGBoost joint-horizon 对照）量化协议优势，用于公平比较，不是修 Full。
6. E8 非线性头，放最后。
