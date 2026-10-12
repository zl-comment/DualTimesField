# 公平对比与 SOTA 检验实验计划（fair-sota-plan）

- 作者：zl
- 分支：`exp/fair-sota-plan-zl`
- 基线：`origin/zl/forecasting-exploration-plan` @ `ca70146`（仓库中最新的目标分支，见 §0）
- 状态：**计划文档，尚未执行任何新实验。** 文中凡未标明“已有证据”的数字、排名、收益均不是结果。
- 目标：**争取并检验**在 NSW1/QLD1/TAS1 上能否得到可靠的 SOTA。不预设、不承诺会达到 SOTA；若证据不支持，则如实报告“不支持”。

## 0. 仓库与分支核验记录

| 项目 | 核验结果（2026-10-12 实测） |
|---|---|
| 本地路径 | `/Volumes/AI_WORKSPACE/01_Chief/DualTimesField`（`/Volumes/AI_WORKSPACE` 已挂载，apfs） |
| origin | `https://github.com/zl-comment/DualTimesField` |
| `git fetch origin` | 成功（本次未出现 SSL_ERROR_SYSCALL） |
| `git ls-remote` | `main` = `5ff4853`；`zl/forecasting-exploration-plan` = `ca70146` |
| 最新分支 | 按提交时间，`origin/zl/forecasting-exploration-plan`（2026-10-12）最新；`origin/main` 是它的祖先（`merge-base --is-ancestor` 为真），落后 2 个提交（`697d063` 计划、`ca70146` 探索套件与报告） |
| 本地 `main` | 与 `origin/main` 同为 `5ff4853`，工作区干净 |
| 本分支起点 | `ca70146`（即 `origin/zl/forecasting-exploration-plan`），新分支未覆盖或复用该分支，仅从其出发 |

结论：本地 `main` 与远程 `main` 一致，但**并不是最新**；最新工作在 `zl/forecasting-exploration-plan`，本计划据此出发。

## 1. 问题陈述与当前证据

### 1.1 已有证据（来源：`logs/forecasting/exploration_20261011/reports/report_static.md`、`report.md`、该目录 `README.md`）

静态协议（2023–2024 测试，三地区×三 seed，宏平均，均值±样本标准差）：

| 方法 | MAE | CRPS~ | 说明 |
|---|---:|---:|---|
| xgboost | 37.968 ± 0.019 | 25.527 ± 0.014 | MAE 与 CRPS~ 均为最低 |
| full | 38.923 ± 0.296 | 25.936 ± 0.017 | 未超过 XGBoost |
| gatets_adapter | 41.705 ± 0.043 | 26.310 ± 0.045 | |
| dlinear | 41.831 ± 0.335 | 27.017 ± 0.202 | |
| patchtst | 63.171 ± 1.386 | 37.066 ± 0.583 | 已知实现缺陷（§1.2 D6），不可作为模型质量证据 |

按地区看并非 XGBoost 通吃：static 中 TAS1 的 MAE XGBoost 为 28.919，full 为 31.207；NSW1 的 MAE XGBoost 45.050 对 full 45.572；但 QLD1 的 CRPS~ full 26.737 低于 XGBoost 26.957，而 NSW1 的 q05 负价 pinball full 为 10.565，远低于 XGBoost 的 18.198。尾部指标上 full 与 XGBoost 互有胜负。因此“XGBoost 最优”应写作“XGBoost 在宏平均 MAE/CRPS~ 上领先”，不是所有指标/地区。

配对差异（static，168 小时 block bootstrap）：xgboost 相对 full 的 MAE 差 -0.955（区间 [-1.317, -0.596]），CRPS~ 差 -0.409（区间 [-0.803, 0.044]，Holm 校正后不显著）。即 MAE 上 XGBoost 优势显著，CRPS~ 上证据不足。

README 已自述的限制（不再重复隐瞒）：测试期 2023–2024 与 2025 均已被反复查看，不是盲测；PatchTST 的 RevIN 与目标空间不一致；神经基线固定学习率、固定 30 epoch、无系统验证调参；PatchTST/GateTS 外生输入仅线性 adapter；full 的点预测与基线点预测定义不同；XGBoost 对 24 个 horizon 分别拟合并分别早停；时间戳可用性用 archive 元数据作代理，未审计真实发布延迟。

### 1.2 缺陷诊断：问题 → 修复 → 验证

下表区分“已被代码/报告证实”与“怀疑，待审计”。每项都映射到修复步骤和验证方式。

| # | 问题 | 状态 | 类别 | 修复 | 验证 |
|---|---|---|---|---|---|
| D1 | PatchTST：RevIN 用 asinh 价格历史的统计量反归一化，但监督信号是标准化的原始价格空间（README、`general_baselines.py` 的 `RevIN`） | 已证实（作者自述+代码） | 归一化/任务定义 | 让 PatchTST 在与监督目标相同的空间内做 RevIN（或对目标也做 asinh 并在评分前反变换）；加单元测试：输入 x 的尺度变化时输出等变 | 小规模 smoke：训练损失下降且验证 CRPS~ 不劣于 DLinear 的 2 倍；之后才进入正式矩阵 |
| D2 | 神经基线固定 lr、固定 30 epoch，无验证集调参；TAS1 GateTS-adapter 三次均在第 30 epoch 取得最优，说明未收敛 | 已证实 | 调参预算/early stopping | 给每个神经模型同等预算的验证集调参（§4.3）；训练到验证 CRPS~ 连续 patience 个 epoch 不改善；记录“是否触及 epoch 上限” | 报告每个拟合最佳 epoch 与上限；触上限者须重跑或标注 |
| D3 | XGBoost 与神经模型调参预算不对等：XGBoost 固定超参（lr 0.05、depth 6）加 2000 轮早停 50；神经没有对应预算。同时 XGBoost 并未调参，这意味着其强是“默认超参已强”，不等于已被公平压到极限 | 已证实（`exploration_baselines.py`） | 调参预算 | 所有方法统一“验证集上等 trial 数”的搜索（§4.3），包括 XGBoost/LightGBM | 记录每个模型的 trial 数、最优配置、搜索耗时 |
| D4 | seed 实际不对 XGBoost 起作用：三个 seed 的 MAE 标准差仅 0.019；full 为 0.296。seed 方差在方法间量级差 10 倍以上，“三 seed 均值±SD”比较不对称 | 观察到，原因待查（subsample/colsample 只有 0.8，GPU hist 近似确定性） | seed | 区分“训练随机性 seed”和“数据划分”；对树模型同时扰动 subsample/colsample 种子；报告 seed 间差异的来源 | 每个模型输出 seed 间预测的最大逐点差，确认 seed 确实改变了训练 |
| D5 | 点预测定义不一致：full 另有变换空间 MSE 点头，XGBoost/神经基线以 q50 为点预测（README） | 已证实 | 任务定义/指标 | 点预测主表统一使用“分位数中位数 q50”；full 的 MSE 点头另列辅表 | 同一 `npz` schema 中 `point_q50` 与 `point_head` 分字段保存 |
| D6 | PatchTST/GateTS 的外生信息只经线性 adapter 进入，非线性主干看不到；“信息集合相同”≠“利用方式相同” | 已证实 | 特征可用性/容量 | 增加“外生进入非线性主干”的适配版（如把已知未来特征作为每步 token/协变量 patch）；保留线性 adapter 版并标注 | 消融：adapter-only vs 非线性融合；参数量对齐 |
| D7 | XGBoost 对 24 个 horizon 各自早停、各自选择树数，而神经模型联合训练 24 horizon；horizon 级选择自由度更多 | 已证实 | early stopping | 保留 per-horizon 与 joint（horizon 作特征）两个版本，明确是两个赛道；早停只在验证集 | 报告每 horizon 的最佳轮数与验证曲线 |
| D8 | 可用性时间戳用 run/LASTCHANGED 元数据近似，真实发布时延未审计 | 已证实（README） | 数据泄漏/外生可用时点 | 对 PD PASA/predispatch/IC 每个来源，列出“origin 时刻能看到的最新 vintage”规则，并用独立方式（原始文件的发布时间）抽样核对 | 泄漏审计清单 §6，抽样数与通过率入库 |
| D9 | 2023–2024（及已查看的 2025）反复用于研究，不是盲测；多次探索使测试集选择偏差不可控 | 已证实 | 数据泄漏（选择性）/时间切分 | 冻结本轮“开发窗口”为训练+验证内部滚动；测试期仅用于最终一次评分；新增的未触碰窗口（如将来数据）单列 | 所有选择规则只读验证集；测试集文件在冻结前不生成 |
| D10 | 评估为逐小时 origin、24h 预测、72h 输入，而论文（arXiv:2609.00089）是日前、每日一个 origin、约 1449 天上下文；任务与上下文长度不同，不能把论文的“TabPFN 显著优于基准”直接当作本数据上的预期 | 已证实（论文 §4.3） | 预测窗口/任务定义/训练窗口 | 增加“日前固定 origin”赛道（见 §3.2）使其与论文口径对齐；保留逐小时赛道作为现有口径 | 两赛道分表，不混合排名 |
| D11 | 价格尺度：训练用 asinh/标准化，评分用原始价格；capped650 是另一设置 | 已证实 | 归一化/指标聚合 | 统一规则：所有方法在同一变换下训练，评分一律原始价格；scaler 只用训练段拟合 | 单元测试：scaler 仅见训练数据 |
| D12 | 指标聚合：宏平均对 seed 先在地区内平均再算 SD；TAS1 价格水平低，会让宏平均偏向高价地区 | 已证实 | 指标聚合 | 同时报告：每地区、宏平均（地区等权）、微平均（按 origin 数加权）、以及按地区的相对改善（相对 XGBoost 的比值） | 主表模板 §10 |
| D13 | 未测 LEAR/Lasso 与 TabPFN、Chronos 等论文中的强基线；已有强基线只有 XGBoost | 已证实 | 基线覆盖 | 按 §5 引入 | smoke 通过后进入榜单 |
| D14 | 概率后处理：已有 floor/PD calibrator，但无共形（CP）、IDR 等规范的点→概率后处理对照 | 已证实 | 概率预测后处理 | 加 LEAR+CP、XGBoost+CP（滚动一年校准窗口，与论文一致）；用本数据的验证段做校准 | 覆盖率、区间宽度、CRPS~ |
| D15 | “full”的收益来源不清：no_fields 的 MAE 与 full 差 1.095（显著），但 raw_history、no_tail 差别很小；加 ensemble 后 fields 无增益（`feature/ensemble-holdout-plan` 记录，非本分支，见 §11） | 部分证实 | 模型容量/预训练适配 | 容量对齐的普通网络对照；参数量与耗时入表 | 见 §8 |

### 1.3 根因诊断（取代上表中“仅为怀疑/待审计”的表述）

上表 D1–D15 是缺陷与怀疑清单，**不是**对 Full 落后 XGBoost 的根因结论。根因诊断见 [`docs/full_underperformance_diagnosis_zl.md`](full_underperformance_diagnosis_zl.md)（分支 `diag/full-underperformance-zl`）：基于已有产物的误差分解给出落后的位置（普通价格区间的精度；TAS1 贡献 80% 的 MAE 差距、NSW1 贡献 78% 的 CRPS~ 差距），并把 Full 自身问题、XGBoost 协议优势、其他神经基线的实现缺陷分开，逐项标注 Confirmed / Strong evidence / Hypothesis。该报告同时列出无法在本机完成的诊断及所需最小实验（如 `test.npz` 重评分）。下文各步骤中涉及 Full 的优先级应以该诊断为准。

注意：D1–D3 修复后，PatchTST/GateTS/DLinear 的成绩可能显著变化，也可能 XGBoost 依然领先。**修复后的结果未知**，不得预设。

## 2. 实验矩阵：三地区 × 三 seed

| 维度 | 取值 |
|---|---|
| 地区 | NSW1、QLD1、TAS1 |
| seed | 2026、2027、2028（沿用现有，固定，不得更换或挑选） |
| 矩阵单元 | 每个模型 × 每个赛道 × 9 格（3 地区 × 3 seed）；缺一格则该模型不进榜 |
| 重复层级 | 层 1：seed（训练随机性，含初始化、数据顺序、subsample/colsample 种子）；层 2：地区（不同市场，视为不同实验单元，不合并伪重复）；层 3：时间 origin（用于 bootstrap，见 §7） |
| 汇总 | 先在每个（地区,seed）上独立评分；再对 seed 求均值±SD（地区内）；再给出宏平均（地区等权）与微平均（origin 加权）；seed 平均预测后再评分（ensemble）另表，不与单 seed 均值混合 |
| 确定性模型 | LEAR/Lasso、TabPFN（推理）等无训练随机性的模型：仍跑 3 个 seed（影响采样/ensemble 配置），并如实报告 seed 方差为 0 或极小，不把它当作“更稳健” |

## 3. 公平对比协议（冻结前必须逐项确认）

### 3.1 通用项

| 项目 | 规则 |
|---|---|
| 数据版本 | 使用 `forecasting/datasets.py` 加载的同一数据；记录每地区数据文件 hash、行数、起止时间；所有模型读取同一 hash，否则作废 |
| 时间切分 | 静态：训练 2015-01-01–2021-12-31，验证 2022，测试 2023-01-01–2024-12-31（沿用）。滚动：沿用 `forecasting/rolling.py`，8 个季度，验证为测试季度前 3 个月 |
| 验证/测试隔离 | 超参、早停、floor、校准、集成权重、模型选择只读验证集；测试集预测在配置冻结（git tag）后一次性生成 |
| 目标空间 | 点预测与分位数均以原始 RRP 评分；训练空间（asinh/标准化）可因模型而异，但 scaler/asinh 参数仅从训练段拟合 |
| forecast horizon | 24 小时 |
| 输入窗口 | 逐小时赛道：72 小时历史（沿用）；日前赛道：见 §3.2；并加敏感性：168/336 小时 |
| 外生信息可用时点 | 对每个 origin，仅使用 publication/last_changed ≤ origin 的 vintage；未来实际需求/备用/价格不得作为特征；日历特征可用 |
| 特征集 | 所有方法使用同一“信息集合”；若某模型因结构无法使用某类特征，必须在表中单列“信息集合差异”，不得把差距全部归因于模型 |
| 调参空间与预算 | 见 §3.3 |
| early stopping | 一律基于验证集 CRPS~（五分位未加权 pinball×2）；patience 统一（神经 10 epoch，树 50 轮）；记录触顶情况 |
| 训练资源/超时 | 每模型每拟合设同一墙钟上限（计划值，执行前按冒烟测试耗时标定）；超时/失败记为失败，不补 seed、不挑选 |
| 失败运行 | 全部记录（原因、日志路径）；失败格不得用其他 seed 替补；失败模型不进榜 |
| 重训规则 | 静态赛道训练一次；滚动赛道每季度重训；日前赛道每日重训或每日增量（见 §3.2，需先估算成本） |
| 预训练模型模式 | zero-shot：不更新权重，仅提供上下文；few-shot：上下文中包含验证期同分布样本，权重不更新；full-shot / 微调：在训练段上更新权重（单独赛道，且须声明微调的预算）。三种模式分榜，不混合 |

### 3.2 两条评估赛道

| 赛道 | 定义 | 目的 |
|---|---|---|
| T1 逐小时 origin（沿用） | 每小时一个 origin，预测未来 24 小时，72 小时输入，17,521 个 origin/地区 | 与已有结果可比 |
| T2 日前固定 origin | 每日一个固定时刻 origin（如市场当日 12:30 之前信息），预测次日 24 小时；与 arXiv:2609.00089 口径靠拢（日前、24 小时一并预测）；该赛道需先审计 AEMO 数据在固定时刻的可用性，并与 T1 的 origin 子集（同时刻）对齐 | 检验论文设定下 TabPFN/Chronos 等是否有优势 |

T1 与 T2 分表分榜；T2 的 origin 是 T1 的子集时，须核对逐点一致。T2 的上下文长度敏感性（72 小时、1 年、3 年）单独列为扩展。

### 3.3 调参协议（所有可调模型一致）

- 搜索空间：每个模型在 `configs/fair_sota/search_spaces.yaml`（草案，见 §12）中预先登记，冻结后不得改。
- 预算：每个（模型×地区）固定 N 个 trial（建议 N=30，冒烟后按成本确认），搜索 seed 固定为 `search_seed`，三个训练 seed 共享同一最优配置（避免用 seed 做第二层调参）。
- 选择：验证集 CRPS~ 最低；平局取较小参数量。
- 记录：trial 表、最优配置、耗时；保存为 `outputs/fair_sota/tuning/{model}/{region}.json`。
- “全局单模型”与“地区专用”两条赛道：
  - G 赛道：一个模型联合三地区（地区 embedding/ID 特征），共享超参；
  - R 赛道：每地区独立训练和调参。
  两赛道分榜，各自给出每地区排名。不允许从两个赛道里为每个地区挑最优后拼成一个“组合模型”宣称 SOTA。

## 4. 模型清单

### 4.1 保留的现有模型（沿用 exploration_20261011 的配置）

XGBoost、full 及其消融、linear、DLinear、PatchTST（修复后）、GateTS（官方价格历史版和适配版）、pd_calibrator、no_fields 等。保留旧结果作历史背景；本轮结果只与本轮重训结果比较。

### 4.2 新增模型的取舍与集成成本

| 模型 | 来源与定位 | 集成方式 | Julia↔Python 成本 | 决定 |
|---|---|---|---|---|
| TabPFN（2 / 3 / TS） | 论文中唯一在三个市场显著胜出的族；Python 包 | 直接 Python 集成，用 `tabpfn`；输出离散化 PDF，取 5 个分位数 | 无 | **纳入**；注意 TabPFN-3 / TabPFN-TS-3 论文标注“不允许商用免费使用”（†），需核对许可再用；本仓库为学术研究，仍须在运行前检查许可文本并记录 |
| Chronos-2（及 small） | 论文强 FM，支持协变量，输出分位数 | Python（`chronos-forecasting` 等官方包），固定 commit | 无 | **纳入** zero-shot；Chronos-2-synth 仅在资源允许时 |
| TimesFM-2.5 + XReg | 论文中成绩偏弱，需 XReg | Python | 无 | **可选，低优先级** |
| Moirai-2 | 论文中不支持协变量、成绩偏弱；任务要求列出 | Python（`uni2ts`/官方），不含外生变量；明示信息集合不足 | 无 | **纳入为信息受限的参照**，不与全信息模型直接比较“结构优劣” |
| Mitra + CP | 表格 FM，需共形后处理 | Python | 无 | 可选 |
| LEAR（Lasso + 多窗口平均）+ CP | 论文基准，EPF 强基线 | **Python 重实现**：4 个训练窗口（56/84/728/1092 天），asinh，7 折 CV 选 λ；每日重训。窗口长度按本数据可用的训练长度调整并明确标注为“适配” | 若直接用 Julia 实现，则需 Julia 运行时与数据交换 | **纳入**，Python 重实现；不依赖论文官方 Julia 代码 |
| DDNN-JSU | 论文第二基准，Optuna+TPE 500 trial×4，每日重训 | 计算量大；可缩小预算复现，并明示“缩减版” | 无 | **可选，低优先级**；若做，必须声明预算缩减 |
| PostForecasts.jl（IDR/CP/QRA 等后处理） | MIT；Julia 包，点预测→概率预测后处理 | **仅复用思路**；用 Python 重实现 CP 与 IDR 的最小版本（CP 简单，IDR 可用 `scipy.optimize.isotonic_regression` 实现按排序点预测的条件分布） | 若直接调用，需 Julia + PyJulia/文件交换，数据量大时 I/O 昂贵；不推荐 | **复用思路**，不直接集成 |
| TestForecasts.jl（DM、GW、MCS、Kupiec） | 论文统计检验的作者实现；“Package is under construction”（README） | **仅复用口径**：DM/CPA 的实现用 Python（`significance.py` 已有 Newey-West DM；新增多变量 CPA 日聚合）；Kupiec 用 Python | 同上 | **复用思路**；可在少数样例上与 Julia 实现交叉核对一致性（需安装 Julia，视环境而定，不是必须） |
| energy-analytics（Lipiecki 课程 notebooks） | Jupyter 教学 notebooks（toy model、UQ、DM） | **仅参考**，不作为实验依赖 | 无 | 仅参考 |
| thief（时间层级协调） | 论文 [27] 对应的是不同论文（THieF），与 arXiv:2609.00089 本身的实验没有直接对应关系 | 不作为本轮依赖 | R 包/不同语言 | **不纳入**，仅在 §11 备注 |
| 合理集成 | 验证集选权的 XGBoost+full+TabPFN 等；权重只用验证集；seed 集成另表 | Python | 无 | 在单模型榜完成之后做；必须声明为集成赛道 |

不会为了凑数量加入不适配的模型（例如无协变量的 TimesFM 原生模式）；加入时按其信息集合分表。

### 4.3 模型准入标准

所有新增模型必须依次通过：(a) 单元测试（输出 shape、分位数单调、无 NaN）；(b) 单地区单 seed 的小规模 smoke（几百个 origin）并核对 origin/actual 与基准一致；(c) 泄漏审计；(d) 一个地区的完整验证段拟合。**不能通过的模型不得进入正式榜单**，只作为失败记录保存。

## 5. 地区差异与排名

- 排名在每个地区分别给出（按主指标），并另给出宏平均、微平均、以及“跨地区最差排名”。
- 报告“每个地区胜者是否一致”；不一致时结论写成“地区依赖”，不得以一个地区或宏平均掩盖另一个地区的劣势。
- 已知差异线索（来自现有结果）：QLD1 的 CRPS~ 与 NSW1 的负价 pinball 上 full 对 XGBoost 有优势，而 TAS1 上 XGBoost 明显更好；只能作为需要检验的假设，不是结论。
- 两条赛道：G（全局单模型）与 R（地区专用）如 §3.3，所有榜单带赛道标签。

## 6. 泄漏与公平性审计清单（执行前逐项签字，记录于 `logs/fair_sota/audit.md`）

1. 每个外生来源（PD PASA、predispatch、IC、备用）：origin 时刻可见的最新 vintage 规则及抽样核对（建议每地区每来源 ≥ 200 个 origin，含边界时刻）。
2. scaler/asinh/clip/hinge knot/事件阈值只用训练段拟合（单元测试）。
3. 滚动重训中，训练段不含验证季度之后的数据；验证段为测试季度之前 3 个月。
4. 日历、节假日特征不含未来信息。
5. 特征中没有以未来实际价格/需求/备用计算的派生量；shortfall 等派生量用预测而非实际。
6. 所有方法的 origin 序列、actual 数组逐点一致（hash 比对）。
7. 超参、早停、floor、校准、集成权重的选择文件时间戳早于测试预测文件。
8. 缓存文件中包含 code commit、config hash、data hash、seed、设备、运行时间；缺项者不进榜。
9. 预训练模型：核对其预训练语料是否可能包含本测试期（对 Chronos/TabPFN/Moirai 等，记录发布日期与训练数据声明；无法证明不含时，标注“可能存在预训练污染”，并不据此宣称零泄漏）。
10. 2023–2024、2025 测试窗口已被查看：所有结论必须标注“探索性复验，非盲测”。

## 7. 评价指标与统计检验

### 7.1 指标

- 点预测：MAE、RMSE（逐点 RMSE 与 window RMSE 明确区分）、点预测取 q50。
- 概率预测：CRPS~（五个分位数，沿用，仅作近似）；另加 99 分位 pinball 平均（用于与论文口径靠拢的 T2 赛道；需模型能输出更多分位数，不能的则仅用 5 分位并标注）、90% 区间 coverage、interval width、AIS、分位数交叉率、Kupiec 覆盖检验。
- 尾部：negative / spike（> 300 AUD/MWh）条件 CRPS~、q05-negative、q95-spike pinball，同时报告事件 origin-horizon 数与独立实际小时数。

### 7.2 检验

| 检验 | 做法 | 说明 |
|---|---|---|
| 主检验 | 配对 moving-block bootstrap，block=168 小时（主）、24 小时（敏感性），固定 bootstrap seed，19,999 次（沿用） | 单位：对齐的小时 origin，三地区同步重采样；不把 seed 当独立市场实现，不把重叠 24h 窗口当独立观测 |
| DM（补充） | Diebold–Mariano，Newey-West 长程方差（沿用 `significance.py`）；T2 赛道用按日聚合的损失 | 报告 lag 选择与依赖假设 |
| CPA / Giacomini–White | 多元（24 维）条件预测能力检验，日聚合 MAE/MSE/CRPS，与论文同口径；T2 赛道为主 | 实现可参考 `TestForecasts.jl` 的 `giacominiwhite` 接口，Python 重实现并以已知小例对拍 |
| 多重比较 | 预先冻结比较族：“候选 SOTA 模型 × 每个强基线 × （各地区 + 总体）× 主指标”，用 Holm 校正；次要尾部指标另行标探索性并校正；可选 MCS（`TestForecasts.jl` 的 mcs 思路）作为补充 | |
| 空事件 | 重采样块内无事件或方差为 0 时显式输出 NA，不得输出显著性 | |

## 8. 消融实验

### 8.1 最小必要消融（SOTA 声明的前提）

在 full 上逐项移除：独立 quantile gate、tail loss、事件分支（fields）、router（改为等输入 concat）、IC、PD、scarcity；raw_history；每项 3 地区×3 seed，静态协议。目的是确认“最终候选”的收益来自哪个组件而不是容量。

### 8.2 扩展消融

- 特征组（价格历史/日历/PD/IC/scarcity）逐组加入与移除；
- 地区共享 vs 独立训练（G vs R）；
- 预训练/微调策略：zero-shot / few-shot / full-shot；
- 输入窗口 72/168/336 小时；
- 损失函数：未加权 pinball vs 带 tail 权重；
- 后处理：floor、共形（CP）、IDR；
- 集成：seed 集成、模型间集成（验证集选权）；
- seed 敏感性：九格全部汇报，不仅均值。

另有“容量对齐”对照：参数量相近的普通 MLP/Transformer，防止收益来自规模。

## 9. 统一缓存与预测文件 schema

每个（赛道, 模型, 地区, seed, split）一个 `npz`，路径：`outputs/fair_sota/<track>/<model>/<region>/seed<seed>/<split>.npz`。

| 字段 | 形状/类型 | 含义 |
|---|---|---|
| `origin_unix` | [N] int64 | origin 时间（UTC 秒，标注 AEST 规则） |
| `actual` | [N,24] float | 原始 RRP 实际值 |
| `quantile` | [N,24,5] 或 [N,24,99] | 非递减分位数预测；`quantile_levels` 一并保存 |
| `point_q50` | [N,24] | q50 点预测（主点预测） |
| `point_head` | [N,24]，可选 | 模型自带点头（辅表） |
| `meta.json` | | code_commit、config_hash、data_hash、seed、region、split、起止时间、特征列表、checkpoint 选择规则、设备、墙钟时间、参数量、trial 数、是否触及 epoch 上限、失败信息 |

日志放 `logs/fair_sota/`；表格由脚本只读上述文件生成，不手写。

## 10. 结果表格模板（均为空模板，未填任何数字）

T-A 主结果（每赛道、每协议各一张）：

| 模型 | 赛道(G/R) | 模式(zero/few/full) | 地区 | MAE (均值±SD) | RMSE | CRPS~ | AIS90 | Coverage90 | 宽度 | 地区内排名 |
|---|---|---|---|---|---|---|---|---|---|---|

T-B 地区汇总：宏平均、微平均、跨地区最差排名、相对 XGBoost 的百分比变化。

T-C 显著性：

| 比较对 | 指标 | 地区/总体 | 差值 | 95% 区间 | 原始 p | Holm p | block |
|---|---|---|---|---|---|---|---|

T-D 消融：移除项 × 地区 × 主指标，附相对 full 的配对差与区间。

T-E 资源/效率：参数量、训练/推理墙钟、GPU/CPU、trial 数、峰值内存。

T-F 失败记录：模型、地区、seed、阶段、错误摘要、日志路径、处理决定。

T-G 尾部指标与事件计数表。

## 11. 外部来源与限制

- 论文 arXiv:2609.00089（Lipiecki & Weron）：九种基础模型变体零样本，对比 LEAR+CP 与 DDNN-JSU，德国/波兰/西班牙，2021–2025，日前市场，上下文 1449 天（表格模型）或更长（Moirai-2 上下文 8736 小时）；结论为只有 TabPFN 族在三个市场上全面显著胜出。**这些结论对欧洲日前市场成立，不能外推到 AEMO 5 分钟/30 分钟实时市场的小时聚合数据上。**
- 作者仓库 PostForecasts.jl（MIT）、TestForecasts.jl（“under construction”）、energy-analytics 为读取公开资料，**未克隆或运行**；上表“复用思路”均指此意。
- 论文引用的 THieF（`lipiecki/thief`）与该论文实验不是一一对应，本计划不依赖它。
- `origin/feature/ensemble-holdout-plan`、`feature/tail-decision-value` 中有 ensemble/2025 holdout 记录（读取方式：`git show <ref>:<path>`），本分支未合并；其数字不能当作本轮结果。
- 本计划没有运行任何实验或测试新模型；上表中任何“预期”都只是待检验假设。

## 12. 执行阶段、资源与判定

### 12.1 阶段

| 阶段 | 内容 | 退出标准 | 优先级 | 估计资源（待冒烟标定，非实测） |
|---|---|---|---|---|
| P0 协议冻结 | 本文档评审；确定 N trial、超时、patience；建 `configs/fair_sota/`；打 tag `fair-sota-protocol-v1` | 审计清单 §6 全部有负责人 | 最高 | 人工 |
| P1 缺陷修复 | D1（PatchTST）、D5、D6、D7；统一 schema；单元测试 | 测试通过；PatchTST 输出等变性测试通过 | 最高 | CPU 少量 |
| P2 冒烟 | 每模型 1 地区 1 seed，几百 origin；核对 origin/actual hash | 通过的模型进入 P3；不通过的写失败记录 | 高 | 小 |
| P3 验证段调参 | §3.3 预算；仅验证集 | 每模型每地区一份最优配置 | 高 | 取决于 trial 数；需先用 P2 估单 trial 耗时 |
| P4 静态完整矩阵 | 全部准入模型 × 3 地区 × 3 seed | 九格齐全；失败格已记录 | 高 | 参考上一轮 315 次静态拟合的耗时（README 称运行在 GPU 6/7 上，本环境不使用实验室服务器，需在授权的本机/云资源上重估） |
| P5 滚动与 T2 赛道 | 关键模型的 8 季度滚动；T2 日前赛道 | 与静态相同 origins 校验 | 中 | 大（72 次拟合/模型/赛道） |
| P6 消融 | §8 | | 中 | 中 |
| P7 统计与报告 | bootstrap/Holm/DM/CPA；全部表格 | 表格由脚本生成 | 高 | 小 |

**资源约束提示**：上一轮计划使用的 GPU 在实验室服务器上；本任务边界禁止访问实验室科研服务器，因此 P3 以后的算力来源需由用户另行指定，否则只能在本机小规模运行。这是执行前的待决事项。

### 12.2 停止/晋级标准

- 模型晋级：通过 §4.3 全部准入标准；
- 终止：关键组件泄漏审计不通过；或同一设置下两次修复仍不能收敛；
- 不因结果不好而删除失败配置（失败记录保留）。

### 12.3 SOTA 判定标准（预先冻结）

1. 主指标与主测试集预先指定：**宏平均 CRPS~ 与 MAE 同时作为主指标，主检验取 CRPS~**，其余为次要；测试集为静态 2023–2024（探索性）。
2. 候选模型须在**三个地区的每个地区**上不劣于最强基线（Holm 校正后差异的 95% 区间不整体落在劣势一侧），且**宏平均**上优于最强基线，主检验的 Holm 校正 p < 0.05。
3. 三个 seed 的方向一致：九格中至少 8/9 与结论方向一致；不得挑选 seed、地区或指标。
4. 在滚动协议（8 季度）上方向一致（探索性）。
5. 与任一强基线相比，MAE 或尾部指标的显著劣化需与 CRPS~ 的收益一起报告；若仅在某尾部或某地区成立，则结论限定为该范围，写“部分优势”，不写“SOTA”。
6. 若 XGBoost（含公平调参后）仍最强，则结论写“不支持 SOTA”。
7. 因 2023–2024 与 2025 已被查看，任何结论均须标“探索性，非盲测”。

## 13. 命令模板（计划中，尚未提供对应脚本）

```bash
# 例：配置草案位于 configs/fair_sota/，脚本入口待 P1 实现
python -m forecasting.fair_sota.run --track T1 --model xgboost --region NSW1 --seed 2026 \
  --config configs/fair_sota/protocol_v1.yaml --output outputs/fair_sota
python -m forecasting.fair_sota.tune --model patchtst --region TAS1 --trials 30 --search-seed 7
python -m forecasting.fair_sota.report --root outputs/fair_sota --out logs/fair_sota/report.md
```

以上命令尚不可运行；P1 实现后才会成为正式命令。矩阵循环：`for r in NSW1 QLD1 TAS1; for s in 2026 2027 2028`。

## 14. 与现有文件的关系

- 沿用：`forecasting/datasets.py`、`rolling.py`、`significance.py`、`exploration_*`；
- 将新增（计划）：`forecasting/fair_sota/`、`configs/fair_sota/`、`logs/fair_sota/`；
- 配置草案：`configs/fair_sota/protocol_v1.draft.yaml`（本分支已提交，仅草案，尚未被任何代码读取）。
