# Full 诊断、修正与公平比较计划

本轮以用户明确的边界为准：统一预测样本/标签、模型入口的数据及公共预处理、输出评分；模型内部变换、结构、融合方式和原生拟合策略由各模型自行决定。完成共同接口验收后建立核心基线的位置并筛查 Full 的互补价值；独立预测与集成增量是两条研究目标，最终结论允许是局部优势、预算内未改善或不支持领先。

- 修订依据：分支 `diag/full-underperformance-zl`，起点 `af546fa3b1fd75852075460015810f745fd2fd74`；2026-10-12 在 `/root/workspace/Code/DualTimesField` 复核。
- 状态：共同输入/评分代码及 LightGBM、Lasso-ARX72+CP、独立 MLP 已接入新路径，见 §8.6；全量固定配置 90 格已完成并整理结果，作为后续比较起点，见 §8.7。此前结果仅归档，不进入新主表。公平调参草案尚未执行，基础模型仍待接入。
- 本计划由已查看的历史结果形成，属于事后研究计划；冻结只能约束后续选择，不能把 2023–2024 或已研究的 2025 变回盲测。
- 数值配置见 [协议草案](../configs/fair_sota/protocol_v1.draft.yaml)。草案目前没有执行器读取；文中实现项、预算和晋级阈值均不是已完成实验。
- 历史参考：[冻结实验及限制](../logs/forecasting/exploration_20261011/README.md)、[原诊断报告](full_underperformance_diagnosis_zl.md)、[历次实验](../logs/forecasting/EXPERIMENT_RESULTS.md)。原诊断的本机缺文件、纯线性头等判断由本计划替代。

## 1. 要回答的研究问题

| 问题 | 能支持的结论 | 不能据此推出 |
|---|---|---|
| 同一 Full 的点预测、概率目标与 checkpoint 选择是否互相牵制？ | 特定训练/选择干预的效果及代价 | 仅凭损失量级认定梯度冲突，或凭早期最佳 epoch 认定训练失败 |
| Full 在普通价格区间的误差能否下降？ | 开发集复验后，冻结候选在回顾性评分中的效果 | 改善一个地区就宣称三地区通用 |
| 双场相比同输入、相近容量的普通模型有什么增量？ | 独立模型、消融或固定集成中的增量 | 双场内部消融有效就意味着胜过强基线 |
| 修复基线和调参预算后谁更好？ | 本数据、信息集、协议与预算内的比较 | 普遍 SOTA，或跨市场直接继承外部论文排名 |

## 2. 证据更新与旧判断修正

### 2.1 冻结实验中仍成立的事实

静态协议为训练 2015–2021、验证 2022、评分 2023–2024；NSW1/QLD1/TAS1、seed 2026/2027/2028。差值统一为 Full − XGBoost，正值表示 Full 更差。

| 指标 | Full | XGBoost | 差值与解释 |
|---|---:|---:|---|
| 宏 MAE | 38.922885 | 37.967935 | +0.954950；旧报告 168h 配对 bootstrap 的 MAE 优势支持 XGBoost |
| 宏 CRPS~ | 25.936 | 25.527 | +0.409；旧报告主区间含 0，不能断言 Full 概率预测显著更差 |
| 普通区间 MAE 差距贡献 | — | — | 0–100 为 +0.830，100–300 为 +0.800；两者合计约 +1.63 |
| 尖峰区间 MAE 差距贡献 | — | — | >300 为 −0.702；负价区间为 +0.027，不能概括为所有尾部指标均更好 |
| 地区定位 | — | — | TAS1 贡献约 80% 的宏 MAE 差距；NSW1 贡献约 78% 的宏 CRPS~ 差距 |

以上区间是按事后实际价格划分的误差分解；不能把实际价格区间作为预测时的路由输入。贡献百分比存在抵消，不等同于事件占比。差距跨月份与步长存在，但未因此得到唯一机制解释。

### 2.2 已完成的旧预测重评分

本机存在冻结预测。已核对 Full/XGBoost × 3 地区 × 3 seed × validation/test 共 36 个文件的 SHA-256 与运行元数据一致；来源哈希一致；两模型的 origin/actual 逐元素对齐；18 个测试 MAE 与冻结汇总一致。

- 报告：`logs/forecasting/exploration_20261011/reports/report.json.gz`。
- 报告 SHA-256：`07ade862cf1a556ea9b1f89681d39065e41254b578fb5bdd782515c7fca215f9`。
- 训练来源哈希：`33e2346fb81bbc68a31b8819f0adcb7ebbc7001ddd7c556d87890dcf684b3923`。
- 预测路径：`outputs/forecasting/exploration_20261011/runs/static/{full,xgboost}/seed{seed}/{region}/{validation,test}.npz`。

| 地区 | 测试 Full 点头 MAE | 测试 Full q50 MAE | 测试 XGBoost MAE | q50 − 点头 |
|---|---:|---:|---:|---:|
| NSW1 | 45.571992 | 49.821073 | 45.049645 | +4.249081 |
| QLD1 | 39.989565 | 44.728054 | 39.935591 | +4.738489 |
| TAS1 | 31.207096 | 31.717947 | 28.918570 | +0.510851 |
| 宏平均 | 38.922885 | 42.089025 | 37.967935 | +3.166140 |

验证集宏 MAE 同样从点头的 76.267012 上升到 q50 的 80.895386，差 +4.628374。q50 按 `quantile_levels` 定位，从排序后的分位数提取；所有评分均在原始价格空间。

**结论仅为：冻结 checkpoint 上直接改用 q50，没有改善平均 MAE。** 不支持将此操作作为修复；不排除重新训练原始价格 L1、改变尾部权重或 checkpoint 选择有效。此次未计算 q50 差值的 bootstrap 区间、完整切片或损失因果效应；不能标注“全部 E1 根因验证完成”。

### 2.3 需要重新定义的机制假设

| 旧判断 | 核验事实与修订 |
|---|---|
| Full 缺少非线性外生头 | 冻结配置已有 `calibrator_hidden: 32`、`scarcity_router: learned`；`models.py` 中校正器和路由均含 MLP。不能把再加 MLP 当作首次引入非线性 |
| Full 所有头对年份纯线性外推 | 线性路径存在，但完整网络也有校正器、门控等非线性路径；只能说年份输入超出训练范围，实际影响需逐路径审计和干预 |
| Full q50 就是通常意义上的条件中位数 | `losses.py` 对实际负价/尖峰样本额外加权，所有分位数都受影响；q50 是尾部加权目标下的估计，不保证是未加权价格分布的中位数 |
| tail 已被排除为根因 | 旧 `no_tail` 的点头 MAE 几乎未变，不能排除 tail 对 q50 的影响；需分别比较点头、q50 和概率得分 |
| 验证最佳 epoch 普遍很早 | 必须区分指标和地区；见下表。早期最优不等于提前停止，也不单独证明优化噪声 |
| XGBoost seed 方差小就代表 seed 失效 | 小方差不是实现缺陷。检查种子传递与预测差异；确定性允许，不人为扩大随机性 |
| 固定 30 epoch 的最优点在上限说明未收敛 | 只说明训练充分性未建立；记录曲线与延长预算的验证结果后再判断 |
| XGBoost 的 per-horizon 训练不公平 | 这是正常建模选择。保留强 per-horizon 基线，joint 版本只作结构/成本对照；两者差值不能归因于早停单一因素 |

冻结 Full 九条历史曲线均有 30 行；九个当前最佳 checkpoint 哈希与运行记录一致。以下 epoch 为文件原值，**从 0 开始计数**：

| 地区 | 按 CRPS~ 选中的 epoch（2026/2027/2028） | 曲线中点头 MAE 最佳 epoch |
|---|---|---|
| NSW1 | 2 / 1 / 2 | 1 / 0 / 1 |
| QLD1 | 20 / 19 / 23 | 1 / 5 / 1 |
| TAS1 | 6 / 4 / 8 | 0 / 0 / 0 |

这支持优先分离“选择指标不同”的问题。历史曲线能比较验证损失，不能重建已未保存的每个 epoch 权重或其测试预测。曲线内容应在审计阶段补存哈希；目前未把它们视为已经具备独立历史哈希的产物。

当前配置 `quantile_weight=1` 时，分位损失等于 `E[w(Y) * pinball_tau(Y,q)]`，其中 `w(Y)=1+tail_weight*(2*I[Y<0]+2*I[Y>300])`。在理想无约束情形，它对应按 w 重加权的条件分布的分位数；这是由损失形式得到的解释，不是对实际网络误差的因果证明。

### 2.4 与之前实验衔接

| 已做方向 | 本轮处理 |
|---|---|
| [非线性 predispatch 校正](../logs/forecasting/EXPERIMENT_RESULTS.md#nonlinear-predispatch-correction) | 旧滚动实验 MLP/hinge 有限改善但未追平 XGBoost；新的结构实验必须说明相对已有校正器的唯一差异 |
| fields、raw_history、no_pd 消融 | 保留结构和输入价值的已有证据；不能把不同协议的收益相加，也不能由字段名称相同推断信息流完全相同 |
| floor、共形校准、tail、量化门控 | 校准及尾部研究已有历史，不再写成“从未测过”；本轮先解决目标与信息流，校准作为后续独立问题 |
| 集成与 2025 研究分支 | `origin/feature/ensemble-holdout-plan` 与 `origin/feature/tail-decision-value` 已有 LightGBM、普通网络及集成产物；先登记来源、协议和选择窗口，再决定复用，不能只列 XGBoost 为全部已有强基线 |
| 2025 以后数据 | 该分支 `logs/forecasting/holdout_data/README.md` 记录 2025-08 起 PD PASA 类型及 UIGF 口径变化；不得直接把更晚数据接入旧协议或宣称新盲测 |

### 2.5 集成已有证据与下一步优先级

`origin/feature/ensemble-holdout-plan:logs/forecasting/members_hold/ensemble/holdout_report.json` 记录的旧 2025 评分协议中，宏 MAE 为：LightGBM 38.560220；验证选权 XGBoost+LightGBM 38.556809；加入 `nn_dual_b` 为 38.164669；加入 `nn_c_mae_v1` 为 38.117065。旧开发期集成报告中后两种加权集成分别为 34.570682、34.555669。这里只引用均值，不重新确认显著性；2025 已经被查看，不是本轮盲测。

该证据支持研究互补性，也要求普通网络对照；不能把“加入神经成员改善”归因于双场独特机制。旧成员、输入、选择窗口及协议与当前冻结 Full 不同，分数不能和 §2.1 静态分数混排。

**共同输入/评分接口的验收已在 §8.6 完成；下一步冻结正式开发预算并补齐基础模型接入，再进入 C0：核心基线定位 + 互补性筛查。** 核心集合为 XGBoost、LightGBM、Full 和普通网络，并新增 Lasso-ARX72+CP、TabPFN-2/3、Chronos-2、Moirai-2 对照。先通过的模型可形成明确标注覆盖范围的阶段表；新增必需对照未完成时，不把阶段表称作完整强基线结论。

每个基线 B 比较 B、B+Full、B+普通网络；另比较 XGBoost+LightGBM 及其加入 Full/普通网络的版本。先做等权和一个全局非负凸组合权重，权重仅在选择段拟合、在开发筛选段评价；相同信息与相同集成容量，必须保留零权重。概率输出先用所有分位共享同一权重以保持单调，不将分位数平均称作混合分布分位数。

先报整体和固定尾部切片，确认收益有无普通时段代价。只有简单组合在开发折显示可复验增量后，再立项分位特异权重或事前风险路由；不得用真实未来 spike 标签决定何时采用 Full。事后尾部误差较低并不证明能提前识别事件。

如果 Full 对强基线有稳定增量，优先研究互补机制和同预算替代对照；如果普通网络同样有效，结论限定为模型互补；如果没有增量，再由误差诊断决定是否投入 E2/E3/E6 等修正。不要求 Full 先单独超过基线，才有资格参与集成。

## 3. 任务和评分口径

保留原逐小时任务：AEST（`Australia/Brisbane`）、72h 历史、24h 目标、1h 步长。同一比较中保持信息集合、目标时间戳、样本、实际值和评分空间一致，允许模型使用不同训练变换、结构和原生训练方法。

| 输出/任务 | 主指标与选择 | 强制附报 |
|---|---|---|
| P：点预测 | 原始价格 MAE；概率模型统一取排序后 q50（包括 Full），checkpoint 按选择集 q50 MAE；仅有 L1 点预测的模型进入单独点预测表 | pooled RMSE、旧 window-mean RMSE、各地区及普通/尾部切片 |
| Q：概率预测 | 未加权 `CRPS~ = 2 * mean(pinball)`；五分位 `[.05,.10,.50,.90,.95]`；checkpoint 按选择集 CRPS~ | 每分位 pinball、q50 MAE、80/90% 覆盖与宽度、区间得分、重排前交叉率 |
| 历史对照 H | 旧 Full 的 CRPS checkpoint + 原点头评分 | 用于与旧实验对应，不替代 P/Q 独立选择后的结果 |

P 和 Q 是两个已声明的预测任务，可以对应不同 checkpoint。训练、调参、推理总成本必须包含全部候选 checkpoint 的开销。禁止把 P 的最佳点头和 Q 的最佳分位数拼成一个未经声明的“单模型”结果。本次修订把概率模型的 q50 统一表作为主点预测比较。历史 Full 点头仅保留在 H/目标函数诊断中，不因其旧 MAE 更低就替代 q50。纯点模型须声明目标函数，不能将重复的点预测列伪装成概率预测。

MAE 不要求所有模型内部都用同一种损失；必须公开目标差异并提供针对 MAE 的强对照。训练变换仅在训练段拟合。CRPS~ 是五分位得分代理，不是精确 CRPS；新模型提供更多分位数时，主表只提取已验证语义的同五分位。缺少 q05/q95 的模型不得静默夹断或外推后入表；密集分位评价另表，不能把五分位插值包装为论文 99 分位 CRPS。

## 4. 时间隔离与开发协议

### 4.1 历史复验与开发集分开

| 协议 | 拟合段 | checkpoint/校准选择段 | 配置筛选或评分段 | 用途 |
|---|---|---|---|---|
| H：冻结历史 | 2015–2021 | 2022 | 2023–2024 | 只读诊断，保持旧输出不变 |
| D1：开发折 | 2015–2019 | 2020 | 2021 | 配置筛选、机制对照 |
| D2：开发折 | 2015–2020 | 2021 | 2022 | 配置筛选、跨时期复验 |
| F：最终静态复验 | 2015–2021 | 2022 | 2023–2024 | 候选配置冻结后，每配置一次回顾性评分 |
| R：季度滚动复验 | 沿用 `rolling.py` 的扩展训练段 | 每季度前 3 个月 | 2023–2024 八季度 | 锁定更新规则后的在线模拟，和 F 分表 |

D1/D2 是本轮新增的开发协议，不修改 H/F 时间定义。每折 scaler/变换仅在该折拟合段训练。每个预测窗口全部目标都须位于对应段内，跨界窗口排除；历史上下文可以来自更早时段，但须当时已可见。D1/D2 重叠，不当作两个独立统计样本。

D1/D2 的选择段只选 checkpoint；筛选段只评价候选，不早停、不调 checkpoint。P/Q 各有固定指标。最终 F 的配置、损失、输出定义、早停规则由开发阶段冻结，2022 只按既定规则选 checkpoint。历史数据已经被看过，开发折和最终 F 都不宣称全新确认性检验。

开发路径必须不加载/生成 2023–2024 评分输出。现有 `train.py` 会在训练结束后评估 test，因此必须先增加开发专用路径，或显式映射并标注开发折筛选段；不能直接沿用旧静态配置开展搜索。评分器与调参器分别写产物，调参器不得读 F 的目录。

R 允许后续季度使用已到达的早期季度观测，这是预先冻结的更新规则；不允许研究者根据季度得分手动改模型。R 的后续训练数据与静态 F 不同，不混合为同一榜单。

### 4.2 延后事项

固定每日 origin、长上下文、全地区联合模型、基础模型微调、在线再校准均为扩展，当前阶段不形成任务维度的笛卡尔积。每日定时预测次日自然日时，目标可能不是该 origin 后连续 24h，不能直接宣称是逐小时任务的子集；以后单独定义目标时间与可用信息。

任何未来盲测须先审计数据口径、实际未触碰范围和基础模型可能的预训练覆盖。更换时段并不自动消除研究选择偏差或基础模型数据污染。

## 5. 数据与产物审计（A 阶段）

| 检查 | 验收条件 | 未通过时 |
|---|---|---|
| 旧预测身份 | 文件哈希、地区、seed、split、来源哈希、origin/actual、报告指标对应；本轮核心 36 文件已过 | 不用对应文件推导新结论 |
| 配置与来源归档 | 核对 manifest、source snapshot、实际配置和代码差异；输入数据另存哈希 | 不把仅“文件存在”当成可重复训练 |
| 曲线与 checkpoint | 按 seed/地区定位；保存现有曲线哈希；核对 checkpoint 内 seed/config/epoch；缺失 epoch 权重明确列出 | 不声称已完成更换 checkpoint 的重评分 |
| 真实数据完整性 | 逐来源覆盖率、缺失行、时间连续性和输入哈希；同一协议所有模型 origin/actual 完全相同 | 不填零、不换 seed，不静默删掉难样本 |
| 外生时间可用性 | 追踪 publication、archive run、LASTCHANGED 的语义；分年、来源、危机/平常期抽样核对；检查修订 vintage | 仅有代理时间戳则保留“可用性未完全确认”的限制；不能宣称实盘可用 |
| 特征路径 | 将 PD、IC、scarcity、日历、gas、历史分别映射到 base/fields/calibrator/router/树输入；检查值、尺度、维度 | 字段同名不足以证明相同利用方式 |
| 分位输出 | 保存重排前/后的分位数；按 levels 找 q50；验证原始价格逆变换 | 无效 shape、NaN 或误变换属于实现失败 |

当前 `.venv/bin/python` 可用，torch 2.5.1+cu124、numpy 2.4.6、pandas 3.0.6、xgboost 3.2.0；冻结配置中的 19 项输入路径引用均存在。**这只证明可读取环境与文件存在，尚未完成数据哈希复核、运行时资源盘点或训练容量标定。** 不再沿用另一台机器“缺 torch、数据、预测”的阻塞描述。

本次只修订计划，不启动训练。后续在当前已授权工作区内盘点设备和活跃任务；可用设备、并发数、墙钟上限写入冻结配置。不得据旧日志的 GPU 6/7 自动占用设备，不访问额外远程主机。

## 6. 根因实验：先分离目标，再改结构

下面 E 编号延续旧计划便于追踪，含义以本版为准。除 E1 的均值重评分和曲线清点外均未执行。每个对照从同一参考配置分叉，不能把上一实验胜者串成基线再声称单因素因果。

| 编号/顺序 | 问题与唯一干预 | 固定条件、输出与解释 |
|---|---|---|
| E1：已有预测审计 | 点头与 q50 的旧输出比较 | 均值结果见 §2.2；补齐分位/价格区间/步长/seed 分解和配对区间。允许复用既有 no_tail 的验证输出诊断，但不以历史测试改善选配置 |
| E2：选择规则 | 相同训练轨迹分别保存点头 MAE、q50 MAE、CRPS~ 最佳 checkpoint | 第一轮固定 30 epoch、关闭早停，保持批次/优化轨迹相同，只比较选择规则；不能把改变训练时长同时算作 E2 |
| E3a：point 目标 | asinh-MSE → 原始价格 L1；保留点头参数化 | 可微逆变换后按训练段原始价格标准差归一化计算 L1；固定 point 权重，报告梯度和 RMSE 代价。它检验目标替换的整体效果，不声称尺度完全等价 |
| E3b：条件扩展 | 仅 E3a 暴露逆变换数值/尺度问题时，单独比较直接 raw-space 点头 | 作为新参数化实验，与 E3a 分开；不能同时改变架构、损失和选点规则而只归因于 L1 |
| E6a：前移 | `tail_weight` 从 0.5 改为 0，其余不变 | 重点检验 q50 偏移、各分位 pinball 和普通/尾部权衡；旧点头消融不能替代。每条训练轨迹同时保存 P/Q checkpoint |
| E4：年份 | 原始年份 vs 截断到训练范围；移除年份是另一个臂 | 先做已存模型的开发/验证推理敏感性，再用完整开发流程复验；推理替换只说明敏感性。禁止同时截断其他连续输入 |
| E5a：学习率 | 原学习率 vs 其 1/3 | 固定训练时长、权重、选择规则；比较曲线、梯度范数、非有限值、seed 方差及开发得分，不由训练 loss 持续下降判定成功 |
| E5b：EMA | 同一学习率下无 EMA vs EMA | 作为独立对照；配置衰减、开始时刻，并保存 `model_epoch` 等调度状态。checkpoint 平均若做，另开实验，不与降学习率混为一臂 |
| E6b：后置权衡 | 在 0 与 0.5 之间仅加 0.25 | 只有 E6a 显示概率收益/尾部代价值得追踪时执行；按 Q 主指标选择，禁止看 F 的尾部评分调权 |
| E7：基线结构 | XGBoost per-horizon vs joint-horizon | 固定信息集并记录样本展开、训练轮数、参数/耗时；建模共享与早停同时变化，是整体结构对照，不是早停优势的因果估计 |
| E8：信息流/容量 | 完成路径审计后，至多增加一个明确缺失的交互路径 | 已有 calibrator/router 保留为参考；配原校正器开/关、相近容量普通网络。必须解释与旧 MLP/hinge 的差别，否则不立项 |
| E9：结构增量 | 最终候选 vs no_fields/raw_history/普通网络 | 对齐可用输入、目标、选择及调参预算；同时报告容量和成本。若无增量则缩小双场机制主张，不继续无上限叠模块 |

E3a 与 E6a 各自完成后，如有开发集证据，可用四格 `point loss {MSE,L1} × tail {0,0.5}` 检验交互；已有格复用，只补缺格。单因素效应不能直接相加预测组合收益。

首次训练筛选固定 TAS1（点误差）与 NSW1（概率问题）、seed 2026，两个开发折；QLD1 在三地区复验时作为迁移检查。每一臂都报告两个地区的 P/Q 得分，不能按地区只报胜出的指标。

梯度冲突仍是假设：先记录共享参数上的 point/quantile/reconstruction 梯度范数与夹角；损失数值占比不是证据。只有该诊断支持继续时，才登记 reconstruction 权重对照，并占用既定候选预算。

## 7. 晋级、停止与预算

所有晋级由 D1/D2 筛选段决定。阈值是控制资源的草案规则，不是机制成立的数学边界，冻结后不得看 F 的结果修改。

| 层级 | 规则 |
|---|---|
| 冒烟 | 每模型真实数据、1 地区、1 seed、小样本；检查数值、反变换、输入和预测对齐。不得用“必须胜过 DLinear”作实现正确性的标准 |
| 首轮筛选 | 对本任务主指标，两个开发折的平均相对损失改善 >0，且任一折恶化不超过 1%；保留所有失败/无改善记录。单 seed 只筛选 |
| 三 seed 复验 | 至多 2 个候选，3 地区 × 3 seed × 2 折；各地区对 seed 先平均，再对地区/折等权平均相对损失；主指标改善至少 0.5%，两个折方向都改善，至少 2/3 seed 的宏方向一致 |
| 副作用 | 候选的另一任务指标宏平均恶化 >1%，或某地区主指标平均恶化 >1%，不标“全面修复”；可保留为任务/地区限定候选并完整报告 |
| 冻结 | P 至多 1 个 Full 候选、Q 至多 1 个；可以是同一训练配置的两个 checkpoint。候选与基线配置、输出、比较族一次锁定 |
| 未过门 | 写“在预定预算和干预范围内未获得稳定改善”；不能写成因果机制已被证明不存在，也不因此阻止公平基线比较 |

相对损失为 `L_candidate / L_reference - 1`；分母是同折、地区、seed 的参考损失。先逐格计算再平均。零分母时输出未定义并停止该相对规则，不临时换分母。均值门槛用于工程筛选，显著性留给冻结后的比较，二者不可互换。

| 成本项 | 初始上限（草案） |
|---|---|
| 只读审计/重评分 | 不训练；曲线与现有预测优先复用 |
| 根因首轮 | 8 个新增训练配置，含组合/条件扩展；每个 2 地区 × 1 seed × 2 折，共至多 32 次；参考 Full/XGBoost 另计 8 次开发拟合 |
| 根因复验 | 最多 2 候选各 18 格，共 36 格；减去已完成的相同格；必要参考补齐另计 |
| C0 简单集成 | 每任务、成员集合预先固定等权及全局凸组合两种规则；选择段拟合权重，开发段选规则；不增加模型训练，需补齐的核心模型计入公平基线预算 |
| 公平调参 | 每模型、地区、任务最多 12 trial；每 trial 为两个开发折、固定 seed 2026；P/Q 共享轨迹时记一次训练并记录两种选择 |
| 校正预算 | 训练上限触顶可按预定规则扩展一次，对所有同类触顶运行一致适用；该耗时计入预算，不免费增加 trial |
| 资源标定 | 冒烟后才冻结设备、并发、CPU/GPU 小时及单 trial 超时；未填满资源字段不启动批量矩阵 |

各模型相同 trial 数不等于相同计算成本。必须同时公开搜索空间、总训练时间、失败数及已使用的历史人工搜索；不得声称抹平了此前 Full 的研发预算。跨任务独立搜索最多 24 trial/模型/地区，不能把两次预算藏在一次“公平搜索”中。

## 8. 基线代码审计、新增对照与准入

### 8.1 当前结论与可复核证据

**旧实验的样本对齐通过，其代码与选择流程不作为新公平榜单的已完成成员。新路径的共同输入/评分实现见 §8.6；接口通过不等于调参和统计比较完成。** [审计清单](../configs/fair_sota/baseline_audit_v1.yaml) 固定了本地源文件 SHA256、8 个外部项目的完整 commit/关键文件哈希、逐模型阻塞项和新增比较对。它是审计记录，不是可执行训练配置；初次审计快照和随后已实施的 aligned_path 状态分别保留。

[产物审计结果](../logs/forecasting/baseline_audit_20261012.json)覆盖 static 的 full/xgboost/linear/dlinear/patchtst/gatets/gatets_adapter 共 126 份预测：历史哈希、地区/seed 身份、样本时间和 actual 一致，数组通过现有严格加载器检查。验证段每格 8,737 origins，测试段 17,521；这不验证特征发布时间或统计独立性。

```bash
.venv/bin/python -m forecasting.analysis.baseline_consistency_audit \
  --output /tmp/dualtimesfield-baseline-artifact-audit.json
```

XGBoost 每地区三个 seed 的点预测哈希均不同，测试 MAE 的 seed 样本 SD：NSW1 0.030071、QLD1 0.008230、TAS1 0.036989。因此“seed 未生效”应撤回；小波动不是公平性缺陷。Full 的 18 份 point 均不等于排序后 q50，其余六模型的 108 份全部相等，旧点预测表确有定义差异。

### 8.2 现有模型代码审计

| 模型 | 实际代码与问题 | 新比较前必须完成 |
|---|---|---|
| Full | `train.py`/`losses.py`；asinh-MSE 点头与 raw-price 分位头分离，分位损失有事件加权，按 CRPS~ 选 checkpoint | q50 主点表与历史点头分表；未加权概率评价；与其他模型对齐开发选择预算；无 tail 作为显式对照 |
| XGBoost | `exploration_baselines.py:run_tree`；raw 五分位、24 horizon 分别拟合/早停，固定超参 | 保留原生 per-horizon 强基线，登记 24 次选择和总成本；与神经模型使用相同开发边界、trial 上限和评价目标 |
| Linear/DLinear | `general_baselines.py` 本地改写；asinh 历史到 raw 监督可作为回归映射，但不是官方实现的数值等价证明 | 标记 AEMO adaptation，核对分解、初始化、head；搜索学习率/正则；外生融合方式明确 |
| PatchTST | 本地改写；`FairWindows` 保留 asinh 历史，而 RevIN 反归一化后直接匹配 raw 标准化目标，空间冲突已确认 | 历史价格与监督统一 raw 标准化，或完整实现匹配的可微变换；与固定官方 commit 做骨干/RevIN 数值检查；quantile head 和 adapter 单列为适配 |
| GateTS-adapter | 骨干已固定官方 commit；五分位输出是本地适配，外生信息只经过线性 adapter；没有 PatchTST 的同一 RevIN 问题 | 保留官方骨干来源，检验 quantile head、同空间输出、外生路径及调参预算；线性融合能力限制不能写成时间泄漏 |
| GateTS history-only | 不消费外生变量 | 只进入价格历史信息集/消融表，不当全信息强基线 |
| 旧 LightGBM/普通网络 | 在集成研究分支，含 rolling、168h 特征、seed 先集成、不同选点/后处理等差异 | 移植后按新协议重训；原 LightGBM 是 L1 点模型，重复 point 不是分位数；旧 `nn_c` 仍依赖 Full 管线，需独立普通 MLP 对照 |

`general_baselines.py` 的模块说明描述的是旧 asinh 训练流程，不能据此判断 exploration 的 raw 目标；必须沿 `FairWindows → forward → target_quantile → denormalize_quantiles` 逐段核对。PatchTST 官方来源为 [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)，本地实现尚未完成权重映射数值对拍，不能标“官方复现已通过”。

### 8.3 作者代码的复用边界

论文为 [Lipiecki & Weron，arXiv:2609.00089v1](https://arxiv.org/abs/2609.00089v1)。本轮检查的论文/仓库没有提供可确认的本文全套统一 runner；不能由此断言作者其他位置一定没有发布。论文方法与库实现仍须逐项对应，AEMO 逐小时适配也不等于欧洲日前实验复现。

| 来源 | 已核实的实现 | 本项目处理 |
|---|---|---|
| [PostForecasts.jl](https://github.com/lipiecki/PostForecasts.jl/tree/589f9407d375e2272fb731873443f691c0060d85) | `CP(abs=true)` 使用排序后的绝对残差；分位为点值加/减 `quantile(abs_error, abs(2q-1))`，`alpha=beta=1` 线性插值；q50 等于点值。`abs=false` 是另一种有符号残差方案 | CP 数值参考；固定实现和残差来源，先做 Julia/Python 对拍。不得把任意 residual quantile 都称为相同算法，也不把该插值实现直接等同于有限样本秩修正保证 |
| [TestForecasts.jl](https://github.com/lipiecki/TestForecasts.jl/tree/7681474b77d5f02526925b5334ed7cd95a6cb92e) | README 标记 under construction；DM 使用样本 SD 而非 HAC，GW 固定 lag=1；CRPS 默认等距概率网格 | 作为作者实现参考；T1 不直接采用其 DM/GW 为主检验。五个非等距分位不能传成 `nlevels=5`；标量 CRPS/APS 还有赋值断言及索引疑点，未运行 Julia，不宣称运行验证完成 |
| [energy-analytics](https://github.com/lipiecki/energy-analytics/tree/8073e6d805aea303ad9cff25d80aeeb35d137764) | 教学 notebooks，Shrinkage 用 scipy 优化展示 Lasso/Ridge | 参考特征/集成思路，不作为本文专用 LEAR runner |
| [thief](https://github.com/lipiecki/thief) | 另一项时间层次预测研究 | 不计作本论文复现代码 |

PostForecasts 为 MIT；本次快照的 TestForecasts/energy-analytics 未发现仓库级许可证，先链接和核对，不直接拷贝源码。未安装 Julia，尚未执行跨语言对拍。库的 CP 公式可以复用，校准时间隔离仍由本项目负责：残差必须来自样本外预测且目标已发布，不能从训练拟合残差或尚未兑现的 24h 预测产生。

### 8.4 新增必需基线对照

| 新增模型 | 代码来源/版本选择 | 同口径接入要求 |
|---|---|---|
| LightGBM | 旧研究分支实现只作参考 | 对齐 72h 信息；L1 点预测和概率版本分开命名，Q 需真实分位训练或独立 CP |
| Lasso-ARX72+CP | [epftoolbox](https://github.com/jeslago/epftoolbox) + PostForecasts 公式 | T1 的 72h 适配，明确称 LEAR-inspired；不偷偷增加第 7 天价格 |
| LEAR+CP | 同上，另设 T2 | 原 LEAR 用第 1/2/3/7 天滞后；论文正则选择与 epftoolbox 默认 LARS 不同。固定日 origin、长滞后与 rolling 更新单列，不能并入 T1 静态榜 |
| TabPFN-2、TabPFN-3 | [PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN)，两个独立模型 ID | 显式选择版本/权重，不能用最新版默认代替；默认 `predict` 是 mean，须取明确 quantiles/q50。监督上下文标签只能来自协议允许的训练段，不能把测试早期样本当免费训练上下文 |
| Chronos-2 | [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)，`amazon/chronos-2` | 使用已发布的未来协变量；跨 origin 的 cross-learning 关闭，避免批次跨时信息交换。`predict_quantiles` 第二项虽名 mean，源码实际取 q50；检查权重原生分位范围，禁止边界夹断冒充 q05/q95 |
| Moirai-2 | [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts)，`Salesforce/moirai-2.0-R-small` | 源码有历史/未来协变量接口，不能因 README 示例维数为 0 就断言模型不支持；需实际接通。模块默认仅 q10…q90，必须检查权重配置；未解决 q05/q95 时只允许 q50 点表，不进入五分位 Q 表 |

8 个外部仓库的源码 commit 已固定；**基础模型权重 revision、文件哈希、权重许可和真实数据 smoke 均未完成**，基础模型条目仍标记待接入；LightGBM/Lasso+CP 的本地接入另见 §8.6，未把 source-only 检查算作成功复现。基础模型还要披露预训练数据与历史评分期的潜在重合；zero-shot 不意味着没有上下文/版本选择成本。

新增比较对：Full 分别对 XGBoost、LightGBM、独立 MLP、Lasso-ARX72+CP、TabPFN-2/3、Chronos-2、Moirai-2；另保留 Full vs no-fields，以及树模型+Full vs 树模型+普通 MLP 的同成员数比较。CP 效果须比较同一个点模型加 CP 前后，不能把更换点模型的增益归因于 CP。

当前核心集合不覆盖整篇论文：DDNN-JSU、Mitra+CP、TimesFM-2.5、Chronos-2-synth/small、TabPFN-TS-3 尚未接入。尤其 DDNN-JSU 是论文的另一条主要市场专用基线；未补齐前只称扩展核心比较，不称论文完整复现。BESS QB/UB 回测属于后续经济评价，不替代本次预测基线核验。

### 8.5 准入与执行顺序

1. 完成现有实现修正：先 PatchTST 空间及官方骨干核验，再统一输出定义、P/Q 选择、时间隔离和 trial/算力记录。保持旧产物只读。
2. 接入 LightGBM、独立 MLP、Lasso-ARX72+CP，并并列准备 TabPFN-2/3、Chronos-2、Moirai-2；核验权重及协变量映射，逐个做真实数据 smoke，不启动完整训练矩阵。
3. 每模型记录 source、输入、变换、输出、选择单位、预测哈希与失败。通过数值/逆变换、时间可用性、分位语义和完整开发段检查后，才允许进入同一阶段表。
4. 冻结搜索空间和资源上限，再运行 D1/D2 对齐比较；低分的有效模型保留，失败模型不冒充低分基线。然后研究 Full 的独立质量与集成增量。

相同 trial 数不等于相同总成本；不强迫 epoch 和树轮数相等，也不为“公平”人为弱化 XGBoost。确定性模型只拟合一次并记录预测哈希，不复制为独立 seed；随机模型固定 2026/2027/2028。全局与地区专用模型、T1 与 T2、静态与 rolling、短与长上下文分别排名。

### 8.6 已实施的共同输入/评分路径

本节落实最新边界：相同数据在进入模型前完成相同处理；模型自己怎样使用这些输入不作强制统一。**外生变量经线性 adapter、RevIN、不同损失或逐 horizon 建模，都不再被列作“输入不一致”的阻塞项。** 模型是否真正使用某块输入由原生实现决定，记录而不强行改造。目标空间错误、样本错位、未来信息泄漏仍须拒绝。

| 边界 | 已实施行为 |
|---|---|
| 样本/标签 | `aligned_inputs.py` 一次构造地区和阶段的数据；保存 origin、24h target times、raw actual，统一完整窗口过滤 |
| 公共处理 | 相同训练段拟合 price/demand zscore、日历及外生特征处理；价格采用 identity+训练 zscore。历史价格与监督直接引用同一标准化序列，消除旧 PatchTST 路径的空间冲突 |
| 输入字段 | 固定七块：历史价格/负荷、历史日历、未来日历、origin gas context、future exogenous、CTF、quantile exogenous；后两类派生特征对所有模型同值；树仅展平成同一组 1465 个特征 |
| 模型内部 | Full 保留原模型/原生 loss；PatchTST 保留 RevIN；GateTS/树保留自身方法。已有神经接口不使用历史日历，`native_unused_blocks` 明确记录；不能写成每个模型每块信息都参与运算 |
| 输出评分 | `aligned_scoring.py` 接收原始 RRP；同样本/标签/时间和三类hash才可比较。概率模型统一排序 q50 点评分、MAE/pooled RMSE/逐分位 pinball/覆盖率；原生点头只作诊断。`quantile_score_proxy` 不是精确 CRPS |
| 选择隔离 | 训练、选择、CP校准、评分分段；早段最后目标须已交付完成（target+1h）才进入后段；模型 forward 只收到输入白名单，评分标签不参与选择 |
| 新增代码 | XGBoost/LightGBM 真正分位训练，Lasso-ARX72+独立 OOS CP，独立普通 MLP；CP 按绝对残差/type-7 实现，未宣称 Julia 数值对拍已完成 |
| 可追踪性 | 输入/样本/公共预处理 hash，模型名+源码指纹、实际参数、库版本、选择轨迹和原始分位输出保留；源码在运行中变化则拒绝验收 |

实际入口是 `forecasting.aligned_runner`，使用 [独立 smoke 配置](../configs/aemo_forecast_aligned_core_v1.yaml)。为独立 CP 校准，集成验证用 train 2015–2019 / selection 2020 / calibration 2021 / screening 2022；此划分只验证接口，**不替代 §4 的正式 D1/D2 研究协议**。最大每阶段 64 origins，只用最早共同样本；公共处理参数仍拟合完整训练段。模型训练 2 epoch 或 8 树轮，Lasso 仅两档 alpha；所得分数不进入研究榜单。

```bash
.venv/bin/python -m forecasting.aligned_runner \
  --config configs/aemo_forecast_aligned_core_v1.yaml \
  --output outputs/forecasting/aligned_smoke_new_run \
  --regions NSW1 QLD1 TAS1 --max-origins 64 --epochs 2
```

输出目录必须全新，旧 `exploration_20261011` 不覆写。最终本轮验证证据汇总在 [aligned boundary 验证记录](../logs/forecasting/aligned_boundary_validation_20261012.json)。外生源目前仍以 archive run/LASTCHANGED 作可用性代理；共同预处理一致不自动证明真实发布审计完成。TabPFN-2/3、Chronos-2、Moirai-2 尚未加入运行器，调用基线 API 会报告具体未接入原因。

完成步骤：先并行实现输入、评分、新基线，再交叉审查；修复 scaler 身份假阳性、预测空间命名、交叉分位诊断、Lasso 参数、完整目标兑现时间及配置参数传递。所有必要修复后再次执行测试和三地区 smoke，以最终源码指纹验收。

### 8.7 2026-10-12 全量固定配置重跑

后续比较起点固定为本轮 `aligned_baselines_20261012`。此前实验和本文旧诊断只作归档，不进入新主表，也不用于证明新版改进；本规则优先于前文的历史结果对照安排。修改版 Full、新增基线及消融沿用本轮样本、标签、公共输入处理与评分，三地区 × 三 seed 完整后比较；候选、超参及集成权重只依据验证段确定，新版本另存冻结批次。主表组织为基线性能比较与最终 Full 消融，消融以最终版本为起点，保留固定配置预算及回顾性评估限制。

已完成 `aligned_baselines_20261012`：Full、no_fields、MLP、Linear、DLinear、PatchTST、GateTS adapter、XGBoost、LightGBM、Lasso-ARX72+CP，共 10 模型 × NSW1/QLD1/TAS1 × 2026/2027/2028 三 seed，90 格，零失败。运行配置为 [aemo_forecast_aligned_20261012.yaml](../configs/aemo_forecast_aligned_20261012.yaml)，入口为 `forecasting.aligned_experiment`。此处日期是运行批次日期，不延长数据到当前日期。2026-10-13 完成结果整理、重新校验与探索性统计，见 [结果解读](../logs/forecasting/aligned_baselines_20261012/README.md) 和 [完整比较表](../logs/forecasting/aligned_baselines_20261012/results/report.md)。

此轮独立协议使用 train 2015–2021、selection 2022H1、CP calibration 2022H2、test 2023–2024；每地区分别为 61273 / 4321 / 4393 / 17521 个完整窗口。各模型复用同地区冻结缓存，公共处理只拟合训练段。2022 拆分用于避免 CP 校准与超参/早停选择共用残差；此轮不等于 §4 的 D1/D2 开发实验，也不等于完整公平 HPO。

神经模型上限 100 epoch、patience 15；树上限 2000 轮、patience 50，以实际训练过程中验证分数早停；模型学习率及其他参数预先写入配置。Lasso 使用预设四档 alpha 和独立 CP 校准，三 seed 重复按确定性结果识别，不当作三个独立随机证据。神经模型和树的预算不是等价计算预算，不能据此宣称公平调参完成。

运行目录保存冻结源码（含 `forecasting` 与 `src`）、配置、输入缓存及哈希、90 格清单、checkpoint、逐格进度和预测；使用 GPU 5/6/7 与一个 CPU worker。队列支持跳过已校验完成格，失败格需显式重试；未完成模型不得进入最终比较表。源码 SHA256 为 `6f67f928f9689fb48e9943a1f0ddc4db797a324d1fa8cbd4aaf5c1cc48c650dd`，配置 SHA256 为 `59acc7cda472bb556e34475d2b77d91007ddaffde03371476bf6b5fd4871fb86`。

启动前 120 个测试和 3 个子测试通过，Full 与 XGBoost 的 GPU 小样本训练、预测及模型保存验证通过。第一批 Full、XGBoost、Linear、LightGBM 已实际训练；启动证据见 [launch.json](../logs/forecasting/aligned_baselines_20261012/launch.json)。实时状态见运行目录 `progress.json`；此处不记录未完成的性能结论。

尚待补充：基础模型适配与权重（TabPFN-2/3、Chronos-2、Moirai-2）、原版 LEAR+CP 的日前赛道、公平调参、验证集决定的组合，以及完整统计协议。结果整理已完成探索性的配对 block bootstrap 和 Holm 校正，DM/GW 尚未执行。后续新增实验采用独立冻结批次，不改写当前队列身份。此前已查看的 2023–2024 仍是回顾性评分，外生数据真实发布时间审计仍未完成。

## 9. 后处理、集成与尾部价值

简单集成在 C0 即可开展，先锁定成员集合、协议和选权规则，无须等待 Full 完成全部修正。复杂后处理在核心比较后单独立项。优先回答“加入 fields 是否比同成本替代成员带来增量”，而不是仅判断集成是否胜过单模型。

| 方向 | 约束 |
|---|---|
| 模型/seed 集成 | 成员和权重只从开发数据选；报告成员数和总成本；比较有/无 Full 的同预算集成，不引用 2025 得分调权 |
| CP/概率校准 | 另设严格早于被预测样本的校准窗口；残差由未拟合该样本的模型或逐步前推预测产生。不能在 checkpoint 选择用过的同批残差上声称独立校准保证 |
| 点到分布 | 普通区间共形校准不自动提供完整分布或精确 CRPS；若需五分位须明确定义算法并验证，不用插值补出未定义概率语义 |
| IDR 等复杂后处理 | 单独立项核对算法和实现；不把一次普通保序回归等同于分布回归 |
| 尾部价值 | 按事前可用预测作决策；真实 spike/negative 只用于评分切片，所有阈值/效用参数在开发期固定 |

尾部校准和经济价值不能代替点/概率榜单，2025 旧结果仍是回顾性证据。后处理失败不回头调整已冻结主模型。

## 10. 最终评价与统计

### 10.1 聚合

每个（协议、模型、任务、地区、seed）先评分。地区表为 seed 均值和样本 SD；宏表每个 seed 先对地区等权求均值，再跨 seed 计算均值/SD。微平均按实际观测数加权，分表；seed 预测先平均再评分属于集成。

主 MAE/CRPS~ 先对每个 origin 的 24h 损失求均值，再对 origin 求均值。保留各地区原始单位结果及相对固定基线改善，不能把宏平均称作价格尺度中性的比较。

固定切片：地区、年份/季度、0–100/100–300/negative/spike、horizon 1–6/7–12/13–18/19–24，另列 h1 和 h22–24；同时报告事件 origin-horizon 数与独立实际小时数。切片不用于事后换主指标。

### 10.2 配对检验

| 项目 | 规则 |
|---|---|
| 原假设与差值 | 候选 − 比较基线；负为改善。P/Q 分属两族，若声称联合优势须同时控制两族 |
| 主重采样 | 对齐 origin 的 moving-block bootstrap，168h、19,999 次、固定 seed；24h 为敏感性 |
| 依赖结构 | 先平均 seed 的逐 origin 损失，再形成差值；三地区同步抽相同时间块；seed 不当独立市场、horizon 不当独立观测 |
| 置信区间与 p 值 | 区间明确 basic/percentile 方法；p 值由在零假设下中心化的块重采样分布求得；未中心化的自助差值正负比例不能直接充当检验 p 值 |
| 多重比较 | 冻结每个任务的“候选 × 必须基线 × 3 地区及宏平均”比较族；Holm 校正原始 p 值；区间若未校正须明确写点态区间 |
| 其他检验 | DM/HAC 只作补充并公开 lag；不默认增加 CPA/MCS/多变量检验，避免实现和比较族膨胀 |
| 稀疏事件 | 无事件切片为 NA，报告有效样本/重采样数量；不删去无事件块后伪装成原总体检验 |

方法、随机种子、比较族和停止规则在最终评分前冻结；bootstrap 程序需验证零假设中心化和时间对齐。报告统计差异不能消除历史测试已经参与研究设计的选择偏差。

### 10.3 允许的结论

| 结果 | 写法 |
|---|---|
| 宏主指标通过校正后的优势检验、地区有取舍 | “本协议宏平均改善，存在地区/指标权衡”，列出退化 |
| 要声称各地区非劣 | 预先定义地区/指标容许差值，按单侧区间上界证明低于该界并控制比较族；本轮默认不作此声明 |
| 未显著不同 | “证据不足”，不写“等价/非劣/已追平” |
| 最终候选未胜过强基线 | “不支持在本协议下领先”；记录机制验证结果，允许结束该方向 |
| 在纳入的基线中领先 | “在列明数据、预算与基线集合内的回顾性领先”；未触碰数据及充分基线覆盖缺失时，不作普遍 SOTA 声明 |

三 seed 一致性只是稳定性描述，不能用“8/9 格同向”替代统计检验或非劣检验。

## 11. 产物、执行阶段与验收

### 11.1 新运行身份与最小 schema

旧 `exploration_20261011` 目录只读。新目录采用独立 `run_id`；D1/D2/F/R、任务、配置 hash、地区、seed 分层，防止缓存串用。旧文件不迁移、不覆写。

| 类别 | 必填内容 |
|---|---|
| 预测 | `origin_unix`、`target_time_unix`、`actual`、声明的 `point`、可用时的 `point_head`/`point_q50`、`quantile`、`quantile_levels`；原始 RRP 单位 |
| 身份 | 代码 commit/工作区补丁 hash、source/config/data hash、环境、协议、fold、地区、seed、任务、checkpoint 选择和 epoch、输出定义 |
| 训练/资源 | 各 epoch 分项损失与开发指标、早停原因、触顶、参数量、耗时/内存/设备、所有 trial 与失败 |
| 选择账本 | 候选清单、各折/seed 指标、晋级原因、预定阈值、冻结版本；最终评分不回写选择账本 |
| 报告 | 原始/修正 Full 对比、P/Q 主表、历史点头附表、地区/切片、统计、消融、成本、失败和研究限制 |

大型 checkpoint/预测保留本地或既定存储，仅将可复现的小型结果、配置和哈希清单纳入版本控制；依现有仓库规则处理，不自动上传。

### 11.2 执行顺序

| 阶段 | 工作 | 退出标准 |
|---|---|---|
| A | 完成 §5/§8 代码与协议审计、修正准入阻塞，登记新增基线来源 | 样本一致不冒充实验公平；未通过项明确 |
| B | 实现开发/最终评分隔离、P/Q 选择、schema、目标空间测试；修复 PatchTST 可在此独立进行 | 真实数据 smoke；开发路径不触碰最终评分段 |
| C0 | 同协议核心模型定位、简单组合及 Full/普通网络的增量对照 | 得到核心水平和互补性证据，再确定修正方向；权重不读取最终评分段 |
| C | 根据 C0 定位选择 E2/E3a/E6a，再考虑 E4/E5/E8；预算内三 seed 复验 | 候选冻结或记录预算内无修复；不要求单模型胜出才研究组合 |
| D | 扩展公平基线开发与调参；核心集合在 C0 已比较，新模型逐个准入，训练按统一资源预算排队 | 所有纳入模型完成开发；有效低分模型不删除；失败单列 |
| E | F 九格静态复验，计算预定统计与消融 | 三地区/seed 完整，未回流调参 |
| F | 对原 Full、最终候选、最强开发基线等预定子集做 R；条件允许再做集成/校准 | 在线规则和 origin 对齐，分表报告 |
| G | 评价证据边界和是否继续研究 | 给出限定结论；未达目标也结束本轮预算 |

### 11.3 正式实验的后续实现位置（共同接口已见 §8.6）

| 现有位置 | 后续职责 |
|---|---|
| `forecasting/analysis/full_vs_xgboost_gap.py` | 扩展为可选的原始预测重评分/曲线审计，保留旧汇总重现能力 |
| `forecasting/train.py`、`losses.py` | P/Q checkpoint、开发专用执行、raw L1、梯度诊断；保持旧配置重现 |
| `forecasting/datasets.py` | 开发折、时间戳检查、年份干预、目标空间信息；旧数据不改写 |
| `forecasting/exploration_baselines.py`、`general_baselines.py` | 修复基线、记录原生训练/早停与输入方式 |
| `forecasting/exploration_runner.py`、`exploration_report.py` | 扩展运行身份、阶段隔离与报告；不直接对冻结 manifest 增补任务 |
| `tests/` 中对应测试 | 时间边界、损失空间、指标/输出、身份、统计和最终评分隔离 |

优先扩展现有模块；只有职责无法容纳时才新增模块。删除旧计划中尚不存在的 `forecasting.fair_sota.*` 命令示例，正式入口经实现、测试和 smoke 后再写入执行文档。

## 12. 旧预测重评分的复核命令

以下命令在仓库根目录执行，只读旧文件，复核 §2.2 的均值；不训练、不写报告、不计算显著性，不代替 §5 的全部审计。

```bash
.venv/bin/python - <<'PY'
import gzip, hashlib, json
from pathlib import Path
import numpy as np

root = Path('outputs/forecasting/exploration_20261011/runs/static')
report_path = Path('logs/forecasting/exploration_20261011/reports/report.json.gz')
assert hashlib.sha256(report_path.read_bytes()).hexdigest() == '07ade862cf1a556ea9b1f89681d39065e41254b578fb5bdd782515c7fca215f9'
with gzip.open(report_path, 'rt') as f:
    report = json.load(f)
rows = []
for region in report['regions']:
    for seed in report['seeds']:
        for split in ('validation', 'test'):
            arrays = {}
            for model in ('full', 'xgboost'):
                directory = root / model / f'seed{seed}' / region
                meta = json.loads((directory / 'run_metadata.json').read_text())
                path = directory / f'{split}.npz'
                assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['artifact_hashes'][path.name]
                assert meta['source_hash'] == report['training_source_hash']
                with np.load(path, allow_pickle=False) as archive:
                    arrays[model] = {key: archive[key] for key in archive.files}
                assert int(arrays[model]['seed']) == seed
                assert str(arrays[model]['region']) == region
            full, tree = arrays['full'], arrays['xgboost']
            for key in ('origin_unix', 'actual'):
                assert np.array_equal(full[key], tree[key])
            y = full['actual'].astype(float)
            qi = np.flatnonzero(np.isclose(full['quantile_levels'], .5)).item()
            q50 = np.sort(full['quantile'].astype(float), axis=-1)[..., qi]
            scores = [np.abs(pred.astype(float) - y).mean()
                      for pred in (full['point'], q50, tree['point'])]
            if split == 'test':
                for model, score in (('full', scores[0]), ('xgboost', scores[2])):
                    expected = report['protocols']['static']['summaries'][model]['per_seed'][region][str(seed)]['overall']['mae']
                    assert abs(score - expected) < 1e-7
            rows.append((region, split, scores))
for split in ('validation', 'test'):
    for region in [*report['regions'], 'macro']:
        selected = [scores for r, s, scores in rows
                    if s == split and (region == 'macro' or r == region)]
        print(split, region, 'point/q50/xgb MAE:', np.mean(selected, axis=0))
PY
```
