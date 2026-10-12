# 电价预测探索实验计划（审查修订版）

> 原始意图：检验结构化事件、稀缺性路由和尾部预测是否有独立价值。
> 审查日期：2026-10-11。本文是执行协议，不是模型获胜的结论。
> 范围：三个地区、三个 seed；完整静态矩阵，以及完整版本和关键对照的八季度滚动复验。不能把实现完成或少量试跑称为实验完成。

## 1. 审查结论与已知证据

原计划的方向可以研究，但“显式事件必然更好”“互联线必然增益”和预期排名 `F > E > D > C > B > A` 没有充分支持。现有代码提交只能证明某功能已实现；应以同协议实验结果判断收益。

| 已有观察 | 正确解释 | 可核对记录 |
|---|---|---|
| detected-event DGF 改善过旧静态 trunk | 历史事件分解有过收益，不等于未来事件概率预测有效 | [实验记录：Detected-event DGF](../logs/forecasting/EXPERIMENT_RESULTS.md#detected-event-dgf) |
| future event field 的 stage A、A' 和 detected-trunk 版本未稳定胜过 logistic；DGF state 可劣于无 state | 未来事件头必须与 logistic、persistence、raw/no-state 对照，不得重复宣称已验证 | [未来事件计划及结果状态](../logs/forecasting/PLAN_FUTURE_EVENT_FIELD.md)、[完整实验记录](../logs/forecasting/EXPERIMENT_RESULTS.md) |
| quantile-specific gate 已完成三个 seed 的实验，旧 trunk 的 CRPS~ 从28.95到28.83 | 它是已有模块，需测新增路由的边际价值，不能再次作为新贡献 | [Quantile-specific fusion gate](../logs/forecasting/EXPERIMENT_RESULTS.md#quantile-specific-fusion-gate) |
| rolling raw：IC + floor MAE 36.68，PD-only + floor 36.64；capped650为25.48和25.29 | 互联线的整体新增收益尚未成立，应保留无 IC 对照 | [Predispatch interconnector inputs](../logs/forecasting/EXPERIMENT_RESULTS.md#predispatch-interconnector-inputs) |
| 同协议 XGBoost 经常优于 dual-field；PD calibrator + 验证集 floor 已是更强的旧方案 | 不能只与早期 trunk 比较；必须重新训练同输入强对照 | [预调度、校准和rolling结果](../logs/forecasting/EXPERIMENT_RESULTS.md)、[基线定义](../logs/forecasting/BASELINES.md) |

另一个远程分支 `origin/feature/tail-decision-value` 已记录后续 ensemble 和2025年结果。本分支没有自动合并这些实现；审查通过 `git show` 读取其 `logs/forecasting/PLAN_ENSEMBLE_HOLDOUT.md`。该记录报告：2025年1月至7月，加入无 fields 神经成员的 ensemble MAE 38.18，加入有 fields 成员为38.20；field contrast +0.016，区间[-0.08,+0.11]。这不支持 fields 的独立增益。部分树模型只有一个 seed，不能充当本轮三个 seed 的完成结果。

**2023–2024已被反复用于研究，2025年已有结果也已经查看。二者都不能在本轮重新称为未见 holdout。** 本轮结论限定为历史基准上的探索性复验；不声称具有全新盲测证据。2025年7月后 PD PASA 可比性问题见该远程分支的 `logs/forecasting/holdout_data/README.md`。

## 2. 可证伪假设

- H1：在相同数据、输入、训练预算和参数规模对照下，结构化路由优于直接拼接未来外生变量。失败时保留负结果。
- H2：稀缺性、predispatch、IC 对 router 有不同边际价值；不能从“价格形成机制合理”推出“可观测特征有新增预测信息”。
- H3：独立 quantile 路由改善未加权概率评分，同时不过度扩大区间或增加虚警。
- H4：成本敏感尾部辅助 loss 改善指定尾部指标，但可能损害整体分位数校准；必须共同报告这项代价。
- H5：fields 的贡献超过 raw-history/no-fields 对照，而不只是网络容量、已知未来协变量或模型集成的收益。

不预先规定地区排名，不把 TAS1 的负结果解释成已证明“自动选择专家”。需要实际 gate、条件表现和消融证据。

## 3. 数据与执行协议

### 3.1 固定基本设置

| 项目 | 冻结值 |
|---|---|
| 地区 | NSW1、QLD1、TAS1 |
| seed | 2026、2027、2028；每个训练变体和新增基线均完整覆盖 |
| 输入、预测、stride | 72小时历史、24小时预测、逐小时 origin |
| 市场时间 | Australia/Brisbane，沿用已有 AEST 小时定义 |
| 主价格任务 | 原始 RRP；point 使用现有 asinh 目标空间，quantile 主路径保留 price-space |
| 静态训练 | 2015-01-01至2021-12-31 |
| 静态验证 | 2022-01-01至2022-12-31 |
| 静态探索测试 | 2023-01-01至2024-12-31；完整24小时目标窗口，17,521个 origins/地区 |
| 尾部阈值 | negative：实际价格<0；spike：实际价格>300 AUD/MWh |
| 分位数 | 0.05、0.10、0.50、0.90、0.95 |

完整静态矩阵首先固定配置再运行。新方法、消融和同输入基线使用相同 splits 和 origins。raw 为主任务；如果执行 capped650 敏感性实验，所有被比较方法都须按该处理重新训练和评分，并单独成表，不能把 raw-trained capped-scoring 与 capped-trained 混为同一设置。任何补充价格处理都不能替代 raw 结果。

### 3.2 必须完成的滚动复验

完整模型及关键对照还须完成2023Q1至2024Q4共八季度 rolling，每模型为8季度×3地区×3 seeds=72次拟合。至少包括完整模型、同输入旧架构、no-router、no-fields、no-tail-loss，以及同输入最强树基线。沿用 [rolling.py](../forecasting/rolling.py)：每季度开始前，用2015年以来截至验证季度之前的数据训练，之前3个月作验证，其后季度作测试；拼接后严格核对与静态测试相同的17,521个 origins。

静态和 rolling 分表并分别比较。已有 rolling 数字只作历史背景，不能与本轮 static 数字排名。旧产物仅在代码/配置/seed/数据hash/预测origins/目标均核实一致时复用；不满足时重新训练。完整模型若不胜基线，也须完成预先规定的关键 rolling 对照并报告失败，不能事后只复验获胜版本。

### 3.3 时间泄漏与可追溯性

- 每个 origin 只能读取当时已可用的历史、日历和预测 vintage；forecast run、last_changed、publication 时间均不得晚于 origin。沿用并测试 [datasets.py](../forecasting/datasets.py) 的检查。
- scaler、asinh参数、裁剪阈值、hinge knot、事件初始化率只用当前拟合的训练数据确定。未来实际 demand、备用容量或价格不得作为已知未来特征。
- checkpoint、floor、校准、ensemble权重、超参数只由对应验证集选择；不能按测试地区表现修改配置后仍宣称同一冻结实验。
- 对每次拟合保存代码版本、配置hash、数据hash、seed、时间区间、输入特征、checkpoint选择规则、设备、运行时间和预测 origins。跨 seed、跨方法核对 actual 与 origins 完全一致。
- 测试结果在配置冻结后集中生成；任何结果驱动的新变体另开实验版本，明确标为探索。

## 4. 模型与消融

模型保留 CTF 趋势路径、DGF 历史事件路径、读取已知未来外生变量的 router，以及 point/quantile 输出。router 输出专家权重；必须另行检验直接拼接相同特征的等输入对照。已存在的 quantile gate 和 future event head 不得重复列为新实现。

完整矩阵的准确配置名、数量和预算由执行 manifest 固定；启动正式拟合前应补齐，不可用此处概念表替代运行清单。以下是最低覆盖要求：

| 对照或消融 | 检验内容 |
|---|---|
| 现有同输入架构、完整结构化router | 新路由相对成熟架构的独立收益 |
| 完整版本减 router；等输入直接拼接 | 外生信息收益与路由机制收益分离 |
| 完整版本分别减 IC、predispatch、scarcity | 三类未来信号的边际贡献 |
| 完整版本减 quantile gate、减 tail loss | 分位路由和代价敏感训练的独立贡献 |
| no-fields / raw-history、no-event branch | fields 与事件路径是否必要 |
| gabor / adaptive / detected | 历史事件表示的选择；所有地区与seed |
| 若使用 future event head：dgf/raw/none state | 未来事件概率是否依赖有用的事件状态 |
| 若主张冻结训练优点：joint/frozen 对照 | 训练方式而非模型定义的收益 |

逐项增加特征的 A–F 链可作辅助展示，不能替代从完整模型移除组件的消融。固定训练上限和搜索预算，记录参数量与拟合成本；关键容量差异应提供相近规模的普通网络对照。不得在测试集上选择一条有利的加法顺序。

## 5. 损失、选择规则与校准

基础目标保留 point、未加权 quantile、decomposition、smoothness和sparsity；future-event NLL/anchor 仅在该模块开启时使用。新增 tail loss 为**成本敏感辅助项**，不是原始分布的严格恰当分位数评分：按真实标签给 tail 加权会改变最优分位数，可能造成区间过宽和虚警增加。

- 主 checkpoint 按**未加权验证 CRPS~**选择，即五个固定分位数的平均 pinball×2；同分时按较早 epoch。不得跨变体比较尺度不同的总训练 loss 来选模型。
- tail 权重、router正则和可选事件校准权重在训练前固定或用预先规定的等预算验证搜索确定；报告完整选择规则。
- 未加权验证指标、验证MAE、90% AIS/coverage必须全部保留。不能仅因负价或spike hit rate提高就宣称整体更好。
- 验证选择的 floor 如启用，须在对照中统一可用，并报告无 floor 版本。point与quantile的floor分别按验证MAE与未加权CRPS~选择。
- 路由稀疏率0.3只是旧正则起点，不是已证实事件概率；不强迫所有地区都有同样的真实事件频率。

## 6. 新增复现比较

主对照包括同数据同输入的 XGBoost（含合适的 L1/profile 强配置）、LightGBM（可用时）、linear/DLinear、PatchTST，以及成熟 dual-field 与 no-fields 神经方案。对新官方模型进行来源审计：论文、官方仓库、固定commit、许可证、依赖、输入协议和预测输出必须记录，优先审查 GateTS 等与路由假设直接相关且有可运行官方代码的模型。

官方骨干加本地外生或quantile头应明确标为适配复现，并说明改动；不可冒充原论文完整复现。所有新训练对照均为三个地区三个seed，使用同 split/rolling 协议；未完成的模型记为未完成，不能用单seed或原论文数字替代。若某骨干不能接收外生输入，另外报告其信息集合，不把输入不足导致的差距全部归因于新架构。

未来事件概率的直接对照为 logistic 和 persistence；没有概率输出的分位数模型不能伪造 Brier/PR-AUC。RE-Price 等外部论文数值单独引用，不与不同数据处理、输入或指标定义的本地结果排名。参考 [PROTOCOL_ALIGNMENT.md](../logs/forecasting/PROTOCOL_ALIGNMENT.md)，其中对原论文价格处理的推断也只能称为推断，不能当作已证实事实。

## 7. 评价、统计与判定

主表：每个地区每个seed的 MAE、window RMSE、CRPS~、90% AIS、coverage、width；报告seed均值与标准差。每seed独立评分后汇总，**seed ensemble先平均预测再评分必须另表**，不能将其CRPS与单seed平均MAE拼成一个“平均模型”。普通RMSE与window RMSE明确区分。CRPS~只是固定五分位近似，不冒称完整分布CRPS。

指定次要指标：negative/spike条件CRPS~、q05-negative与q95-spike pinball、quantile crossing率、hit/false alarm；如确有事件概率则加 Brier、PR-AUC和可靠性曲线。按地区、月、horizon输出，并记录尾部origin-horizon观测数和独立实际小时数。案例按验证阶段固定规则或测试中的客观事件阈值选择，不能只展示成功案例。

统计规则：

1. 使用对齐的完整小时 origin 时间轴，三地区同步重采样；主区间用168小时 moving-block bootstrap，24小时block为敏感性检查，固定bootstrap seed和重复次数。
2. 尾部指标必须在每个重采样块内累加事件loss与事件计数，再取比值；不能先删无事件origins后把剩余时间当连续序列做HAC。空事件重采样与零方差情形要明确处理，不能输出伪显著结果。
3. 报告配对差异、95%区间和原始p值；预先冻结的“完整模型对各关键对照×地区/总体×主指标”比较族用Holm校正。次要尾部比较单独标探索性并进行对应校正。
4. 不把三个seed当作三次独立市场实现，也不把24小时重叠预测视为独立观测。DM可作为补充，须说明lag与依赖假设。
5. 只有完整模型相对同输入强基线在预定主概率指标上改善且区间/校正检验支持，才能称概率预测增益；MAE、校准和虚警的代价同时报告。若收益仅在某尾部或地区成立，应限定结论，不能写“全面最好”。

## 8. 完成条件与产物

### 本轮冻结执行清单（2026-10-11）

准确清单见 [manifest.json](../logs/forecasting/exploration_20261011/manifest.json)，由
`forecasting.exploration_configs` 生成127份配置、1,143个独立训练任务：
静态35组×9=315，滚动11组×8季度×9=792，capped650四组×9=36。
每个九格固定为 NSW1/QLD1/TAS1 × seed 2026/2027/2028。
所有神经价格模型固定30 epochs，冻结事件头40 epochs；XGBoost每horizon最多2000轮、验证早停50轮。
没有根据本轮测试结果追加调参或删除失败配置。

静态清单包含完整模型、去尾部loss、旧gate/固定gate、去独立quantile gate、去事件、
raw-history、no-fields、去IC/PD/稀缺性、gabor/adaptive/detected、echo、A–D输入累加链、
PD校正器和无fields对照、joint/frozen事件头的dgf/raw/none状态及raw去近期极值。
新增对比为全输入XGBoost、linear、DLinear、PatchTST、官方GateTS价格历史版及全输入适配版；
logistic/persistence只进入事件概率表。滚动固定full/no_tail/concat/no_ic/no_fields/
pd_calibrator/pd_nofields/XGBoost/DLinear/PatchTST/GateTS适配版；capped650固定full/no_tail/concat/XGBoost。

完整模型设 `scarcity_router=learned`、`quantile_gate=true`，额外尾部loss系数0.5，
负价/尖峰辅助权重均2，因而总pinball权重为普通小时1、尾部小时2。
外生源移除通过标准化后整块置零实现，保留维度与参数量；稀缺性移除同时遮挡备用容量、
net load及所有shortfall派生量。`no_fields`保留用于历史重构的fields计算，仅去除其预测贡献，
不能解释为节省字段计算；它的有效短缺输入路径也退出，严格结构归因还须参考raw_history。
frozen事件模型按验证事件损失选优、继承full价格预测；joint按价格CRPS~选优，二者属于
不同训练与选择策略，不能把frozen价格预测当作新预测增益。

GateTS使用官方commit `a3b821fff6fda3bde8124a79564a8d33df9732f1`，其核心源码原样保留；
72小时输入、24×5分位输出和外生adapter是明确标注的AEMO适配，非原论文数据集成绩复现。
来源与许可见 [GateTS provenance](../forecasting/vendor/gatets/PROVENANCE.md)。
FAME公开代码存在proxy专家，且任务不同，本轮不冒称其完整版复现；未定位到可核验官方代码
的xTime/AME-TS仅保留为相关工作。

正式运行与检查命令见 [运行说明](../logs/forecasting/exploration_20261011/README.md)。
单epoch程序验证只检查可运行性，不计入这1,143个正式实验或结果表。

- 审查修正落地，配置manifest冻结；数据时间边界、vintage和指标的必要测试通过。
- static 全矩阵，以及完整模型和关键control的八季rolling均完成三个地区三个seed；失败任务及原因明确记录，不能用启动日志代替完成结果。
- 保存逐seed预测、模型选择记录、主表、消融表、独立ensemble表、完整时间轴bootstrap/Holm结果、成本表和地区/月/horizon分析。
- 发布负结果与新增复现的来源和限制；结果不足时直接写“不支持假设”。

## 9. 引用与远程证据位置

原文数字引用存在错位和不存在的[18][19]，已删除；Fourier features与SIREN不能作为互联线预测收益的证据。本文的实证判断以以上仓库记录为依据；新增论文只有核验原文及官方代码后才纳入manifest，不用未经核实的标题支持结论。

本地核心文件：[models.py](../forecasting/models.py)、[losses.py](../forecasting/losses.py)、[datasets.py](../forecasting/datasets.py)、[train.py](../forecasting/train.py)、[tail_metrics.py](../forecasting/tail_metrics.py)、[significance.py](../forecasting/significance.py)。

以下文件存在于审查时的 `origin/feature/tail-decision-value`，不保证本分支工作区存在；用 `git show <ref>:<path>` 查看，不自动merge：

- `logs/forecasting/PLAN_ENSEMBLE_HOLDOUT.md`
- `logs/forecasting/members/ensemble_dev/ensemble_report.json`
- `logs/forecasting/members/ensemble_dev_final/ensemble_report.json`
- `logs/forecasting/members_hold/ensemble/holdout_report.json`
- `logs/forecasting/tail_value/tail_value_dev.json`、`tail_value_hold.json`
- `logs/forecasting/holdout_data/README.md`
- `forecasting/ensemble.py`、`forecasting/tail_value.py`
