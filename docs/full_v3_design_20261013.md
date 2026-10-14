# Full V3：条件历史双场先验与 LightGBM 残差预测

状态：2026-10-13 固定方案；先实现与审查，再冻结运行。分支 `research/full-v3-conditional-analogs-20261013`。

## 1. 要回答的问题

原 LightGBM 诊断显示，2022 年已知高压力条件下，模型往往识别出风险，却明显低估价格水平或幅度。这个证据支持检验“条件水平偏差”，不能证明所有误差都是尖峰、LightGBM 必然有结构缺陷，或旧双场能预测尚未发生的尖峰。

本版假设：历史相似市场状态的已实现价格轨迹，能够提供当前价格水平之上的形状先验；背景与事件依赖的相似性不同，因此分别检索可能优于一个共享相似度。LightGBM 用当前七块输入及检索特征，学习该先验的条件误差。

不把去年同一天当作今年的答案。去年、前年、三年前的同季邻近日，以及近期同星期候选，构成可审计的历史案例库。全部检索权重由预测时刻可用输入计算。没有未来真实价格参与查询相似度或选择权重。

## 2. 不变的比较边界

- 保留原基线代码、配置、72 小时输入、标签、预测样本与已有结果。读取并校验 `outputs/forecasting/aligned_baselines_20261012` 的冻结缓存；不重建或覆盖它。
- 每小时一个 origin，输入前 72 小时，目标为 origin 开始的未来 24 个小时；NSW1、QLD1、TAS1，seed 2026，共 15 个正式格。
- 训练 2015–2021；2022 上半年共同选择树迭代轮数；2022 下半年输出诊断结果但不拟合校准；2023–2024 回顾性评价。以上时期都已查看，不称盲测。
- 树参数固定，不做额外 HPO。每个 horizon 的五个分位数共同按验证 pinball 选一个迭代轮数，最大 2000、patience 50；其余参数与原 LightGBM 一致。
- 原始电价空间评分，点预测为排序后的 q50；MAE、RMSE、五分位数 QS proxy、覆盖率统一复用原评分器。QS proxy 不冒称精确 CRPS。
- 新方法额外取得历史已实现价格轨迹。相较原 LightGBM，这是信息范围扩展；原基线输入按用户要求不变，不能将该比较称为所有模型输入信息完全相同的纯结构消融。
- 单 seed 用于机制筛查，不宣称三 seed SOTA，不由测试结果自动调参或改权重。未完整覆盖 15 格不形成完整排名。

## 3. 真实时间边界与候选库

一个历史案例 j 包括：原始七块输入 x_j、此前 24 小时价格中位数 L_j、其后 24 小时已经实现的曲线 y_j。仅使用缓存中有完整预测样本的 origins；跨划分的缺口不补标签，不删当前查询样本。

对查询 t 枚举以下本地时间候选（Australia/Brisbane）：

1. 年份滞后 1、2、3 年，各自日期偏移 -7…+7 天，同小时，共 45 个；闰日减年映射到非闰年的 2 月 28 日。
2. 滞后 7、14、21、28 天，同小时，共 4 个。

候选必须精确命中缓存 origin；72 小时历史及 24 小时结果都须有连续小时记录，目标时间戳严格为 j+[0…23]h。逐条使用原 CSV `available_at_aest`，所有需要的小时最大发布时间 ≤ t，且 j+24h ≤ t。去重后保持日历枚举顺序，未满 49 槽填 -1，并输出 mask。缺失、迟发候选不参与检索。

这是**动态 as-of 历史库、静态树参数**：2023 年较早已公布的价格可用于之后的 2023 年查询，2022 数据也可供之后查询检索；不将这些值用于重新训练树。预计算可以读到整份档案，但每条查询的候选必须独立满足以上条件。单条曲线分解没有跨曲线拟合，因此以后曲线的分解不会影响过去先验。测试通过污染查询标签及不可用档案验证此边界。

需要区分：CSV 的 available_at 是当前档案记录的发布时间，原缓存标记仍为 `archive_timestamp_proxies_only`。这版审计不能把原外生预测档案的发布代理升级成已完成真实实时回放审计。

## 4. 双场的精确定义

这版是固定基函数的双场实现，**不是恢复旧神经 INR/Gabor，也不继承旧模型有效性的假设**。

每条历史曲线独立中心化、以 RMS（下限 1）缩放为 z_j。设 B 为常数及前三阶正交 DCT-II 基；G 为每个小时中心、宽度 0.75/1.5/3 小时的单位 L2 高斯原子，允许正负系数（Gabor 零载频特例）。求解：

\[
\min_{a,b}\ \tfrac12\|z_j-Ba-Gb\|_2^2
 +\tfrac{0.01}{2}\sum_k k^4 a_k^2+0.03\|b\|_1.
\]

固定 40 次：精确 ridge 更新 a，按 Lipschitz 步长对 b 做软阈值近端更新，末尾再精确更新 a。固定计算预算，不宣称到达精确最优解。恢复原始单位得到平滑背景 C_j 与带符号事件 D_j，保留重建残差 r_j=y_j-C_j-D_j。中心常数属于 C，不重复加到 D。分解未使用全档案的均值或尺度。

背景适合平滑水平/形状；事件原子可表示局部正负偏离；这些是表示偏置，不是因果识别。事件场不命名为“市场情绪概率”；现有数据没有直接情绪测量。旧历史拟合能力不能推出未来预测能力。

## 5. 条件检索与先验公式

所有距离使用冻结的训练期标准化输入，原始七块数据不再二次拟合 scaler。

背景距离 dC(t,j)：72 小时价格各自减前 24 小时中位数后的均方差，加 0.5×最近 24 小时负荷均方差，加 0.5×未来 24 小时日历前八个周期编码均方差。日历排除 year_standardized 和 is_weekend 两列。

事件距离 dD(t,j,h)：第 h 小时 future_exogenous 的列 [0,1,2,3,4,6,7,8] 的均方差，依次为备用容量、预调度价格/负荷/可用发电/净交换，以及互联线净进口/进口空间/出口空间。排除覆盖标记和重复主互联线空间。其值继承共同缓存的档案缺失处理；这版未新增外部新闻/情绪/实时供应商数据。

共享距离 dS=dC+mean_h(dD)。分别对 dC、dS、每小时 dD 取最小四个有效候选，按 softmax(-d/1.0) 归一；等距离保持原候选顺序。全无候选则权重全零。

定义当前原始价格水平 L_t=median(最近 24 小时价格)：

\[
B_S=L_t+\sum_jw_S(C_j-L_j),\quad E_S=\sum_jw_SD_j,
\]
\[
B_C=L_t+\sum_jw_C(C_j-L_j),\quad E_{D,h}=\sum_jw_{D,j,h}D_{j,h}.
\]

不同年份的整体价格水平先对齐到当前 L_t，再迁移形状。这避免直接照搬去年绝对价格，但仍可能受结构变化、无相似事件、峰宽/位置变化影响。逐小时事件权重没有额外平滑约束，可能产生不连续先验；残差树是否修正由实验检验。

## 6. 五个正式对照

| ID | 附加树特征 | 原始单位先验 P | 主要检验 |
|---|---|---|---|
| v3_level_lgb | 无 | L_t | 仅近期水平重心化 |
| v3_analog_features_lgb | R | 0 | 历史信息输入树即可产生多少收益 |
| v3_analog_prior_lgb | 同一份 R | L_t+ΣwS(y_j-L_j) | 显式历史曲线先验与残差学习 |
| v3_dual_shared_lgb | 同一份 R+F | B_S+E_S | 双场历史表示与共享先验 |
| v3_dual_conditional_lgb | 同一份 R+F | B_C+E_D | 背景与事件分别匹配的先验 |

R 固定按以下顺序拼接：49×24 个 `(y_j-L_j)/train_std`；49 个 `(L_j-L_t)/train_std`；49 个有效标记；49 个 dC；49 个 mean_h(dD)；49 个共享权重。共 1421 列，无效槽全部为零。F 是 `(B_S-L_t), E_S, (B_C-L_t), E_D` 各 24 维，除以 train_std，共 96 列。

第一组额外列数 0；第二、三组 1421；第四、五组 1517。所有树始终包含原来完整七块输入。P2=0；其他组无候选时都退化到 L_t，全部查询样本保留。

解释范围：P4 与 P5 都能看到两套场特征，因此它们只隔离**先验组合方式**，不能说 P4 完全没接触条件检索信息。P3→P4 同时增加场特征并采用有残差的分解重建，若有收益，不能仅归因“双场分解”一个因素。此轮是五个机制对照，不是完整因素消融；有明确收益后再扩展消融，不能从单轮排名反向编故事。

## 7. 残差 LightGBM 与分位数

为每个 horizon 和 q∈{.05,.1,.5,.9,.95} 训练新的 LightGBM：

\[
u_{t,h}=(y_{t,h}-P_{t,h})/\sigma_{train},\qquad
\hat Q_q(y|x,M)=P+\sigma_{train}\hat Q_q(u|x,M).
\]

先验是查询输入和已知档案的确定函数，条件分位数对这种平移与正缩放满足上述等式；不额外加训练价格均值。五个分位数维持原比较协议，不是双场先验必须输出五个曲线。双场本身只生成一个先验，树学习其残差的五个条件分位数。

每地区 24×5=120 个树模型。默认 lr=.05、31 leaves、min_child_samples=50、subsample=.8、colsample_bytree=.8、4 CPU threads，五组相同。模型原生实现复用现有 LightGBM fit_horizon。无预训练 LightGBM 锚、无 V2 OOF 蒸馏、无原 Full 训练。

## 8. 工程对应与运行

| 数学/实验职责 | 文件 |
|---|---|
| 真实发布时间与日历候选 | forecasting/full_v3/memory.py |
| C/D 分解与重建残差 | forecasting/full_v3/fields.py |
| 距离、权重、五个 P、R/F | forecasting/full_v3/priors.py |
| 新树训练、raw residual 解码、校验重载 | forecasting/full_v3/model.py |
| 冻结、准备、15 格 CPU 调度 | forecasting/full_v3/experiment.py |
| 完整性与共同评分汇总 | forecasting/full_v3/report.py |
| 唯一正式参数 | configs/full_v3/experiment.yaml |

运行目录 `outputs/forecasting/full_v3_20261013`：config.yaml、design.md、source/ 为冻结版本；jobs.json 记录代码/配置/原始缓存/派生特征 SHA256；features/{region}/{split}/ 保存特征、五个先验及逐行候选索引/实际最大发布时间/权重；runs/{variant}/seed2026/{region}/ 保存 120 树、轮数、进度、三段原始预测、共同指标及文件哈希。summary.json、results.csv、report.md 为汇总。

预测元数据保留 base_input_hash，input_hash 则记录额外特征和先验后的真实有效输入哈希，不冒用原基线哈希。报告核对 base input、样本、标签、时间和 preprocessing 一致，允许方法的历史扩展不同。

```
.venv/bin/python -m forecasting.full_v3.experiment prepare --run-root outputs/forecasting/full_v3_20261013
.venv/bin/python -m forecasting.full_v3.experiment launch --run-root outputs/forecasting/full_v3_20261013
.venv/bin/python -m forecasting.full_v3.experiment report --run-root outputs/forecasting/full_v3_20261013
```

六个 CPU 作业并行，每作业四线程；使用冻结源码子进程。中断的部分树目录保留后重训该格，已完成完整模型可重载预测，完成格不覆盖。准备失败需新目录或明确归档故障目录，不混用不完整缓存。

## 9. 运行前检查与判断准则

必须通过：未来/查询标签污染隔离；完整 96 小时时间边界；缺失/迟发/闰日；空候选回退；所有先验独立公式复算；正负尖峰/常数/单条分解不受其他曲线影响；单位逆变换；保存重载一致；输入/模型哈希篡改拒绝；完整矩阵与报告缺格拒绝。真实数据小样本只检验管线，独立 smoke 目录、极小轮数，不进入正式表。

主表报告三地区每组 MAE、RMSE、QS proxy，以及完整三地区宏平均。补充固定 >300、0…300、<0 分组的 MAE、偏差和 >300 点预测命中/误报，防止只降低普通区间误差却声称提高尖峰。单 seed 不做跨 seed 一致性结论。不同分组胜负与全局收益都照实报告，不能以挑地区/seed/阈值替代预设评价。

先对比第一组与原 LightGBM，排除仅水平重心化就已解释多数收益；再看历史输入与先验；最后比较同输入 P4/P5。若条件双场未改善，不继续宣称其有效；检查相似度、案例覆盖、分解误差和偏差来源，再另开预先说明的新版本。严禁用测试结果回写本次冻结参数。

## 10. 来源及创新边界

- [LightGBM 官方实现](https://github.com/microsoft/LightGBM)：树训练调用官方库，本项目实现统一训练/输出适配。
- [RAFT: Retrieval-Augmented Forecasting](https://arxiv.org/abs/2505.04163)、[官方代码](https://github.com/archon159/RAFT)：历史案例及后续已知曲线的检索预测已有先例；本版未复制其实现，也不将“使用历史检索”本身宣称新颖。
- [Extrapolating the long-term seasonal component in electricity price forecasting](https://arxiv.org/abs/2503.02518)：季节/长期水平是相关背景，但其任务和证据不能直接证明本版每小时滚动 24 小时预测有效。
- [AEMO pre-dispatch](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/market-management-system-mms-data/pre-dispatch)：现有外生预测的数据含义来源。

本轮本地实现的研究假设是：**价格水平对齐的历史双场，按背景与市场压力分别匹配，再由条件残差树修正**。能否成为论文贡献取决于机制对照、稳健结果和更充分文献核对，当前先写成待验证假设。
