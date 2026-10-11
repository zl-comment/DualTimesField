# 电价预测探索实验计划（zl）

> 作者：zl
> 目标：基于仓库中最近的 `forecasting/` 提交，系统判断下一步最值得推进的实验路线，并给出可以直接执行的实验设计、模型结构草图、损失函数设计、消融清单和两周推进节奏。

## 0. 结论先行

**我建议把当前工作主线收敛为：**

> **结构化事件预测 + 稀缺性路由 + 互联线/未来外生驱动 + 尾部指标优先**

具体来说，最值得继续推进的不是“再做一个更大的通用预测器”，而是把现有 DualTimesField 的 forecasting 线继续向下列方向加固：

1. **显式事件建模**：已证明比把事件埋进连续趋势里更有效。[3]
2. **稀缺性 / PD PASA / 最大备用容量驱动**：这是目前最强的未来外生信号之一，能稳定改善 MAE 与尾部指标。[4][5]
3. **预调度 + 互联线输入**：互联线约束是价格形成的重要外生结构，应该成为路由与事件触发的重要条件，而不只是“额外特征”。[5][17][18][19]
4. **尾部优先评估**：负价小时、尖峰小时、90% 区间覆盖和 CRPS~ 应该与 MAE 并列，甚至作为主优化目标。[6][7][8][10][11]

---

## 1. 为什么这是下一步最优方向

### 1.1 现有仓库已经给出非常清晰的信号
仓库 `forecasting/` 的近期提交链已经在做三件事：

- 用 **显式检测事件** 替代隐式 DGF：`e2cd5f8`。[3]
- 用 **PD PASA 稀缺性** 驱动 DGF 与 gate：`d767bf3`。[4]
- 把 **predispatch / interconnector** 作为 point-in-time 外生输入：`67df468`。[5]
- 用 **tail tests、case study、cost table** 强化尾部和结构分析：`0ccfb14`。[6]

这意味着仓库本身已经从“通用预测”转向了“**价格结构 + 尾部风险 + 稀缺性触发**”的方向。[3][4][5][6]

### 1.2 论文趋势也支持这个方向
下面这些文献与当前仓库方向高度一致：

- 电力价格受到互联线与跨区耦合显著影响。[17][18]
- 波动期里，**spike regularization / variance stabilization** 对预测提升明显。[7][10]
- 极端尾部和负价小时，**平均误差不够**，需要 tail-risk 视角。[8][11]
- 面向极端事件，**hierarchical expert fusion / MoE / routing** 比单一 backbone 更自然。[12][14][15][16]

因此，下一步最应该推进的是：

> **把“趋势 / 事件 / 稀缺性 / 互联线”做成一个结构化专家系统，而不是单一路径的黑箱回归。**

---

## 2. 本次实验的核心假设

### H1：显式事件分支优于隐式事件残差
如果把尖峰、负价和剧烈波动显式建模为事件分支，而不是只让 CTF 去拟合剩余残差，那么：
- spike hour 的 pinball / CRPS 会更稳；
- 负价小时召回会提升；
- 在 regime shift / 稀缺性场景下更鲁棒。[3][6][7][11]

### H2：稀缺性未来外生变量应成为“路由条件”
PD PASA 最大备用容量、短缺量、预调度、互联线约束，不应只是普通输入；它们应该决定：
- 哪个专家激活；
- 事件头幅度；
- quantile head 是否偏向尾部。[4][5][17][18]

### H3：尾部指标必须成为一等公民
如果模型只看 MAE / RMSE，而不看：
- 负价小时 pinball
- spike hour pinball
- CRPS~
- 90% coverage / AIS

那么优化会继续偏向“平滑主趋势”，而不是“真正有经济价值的尾部风险”。[6][8][11]

### H4：结构化 MoE 是更合适的归纳偏置
电价在不同 regime 下本来就不是同一种规律。趋势、稀缺性、负价、互联约束对应不同专家更合理。GateTS、AME-TS、FAME、xTime 都在强调结构感知路由 / 多专家融合的价值。[12][14][15][16]

---

## 3. 推荐路线：四层结构化 forecasting 模型

### 3.1 总体架构
我建议把 forecasting 线重构为四层：

1. **CTF 趋势专家**
   - 学习平滑主趋势
   - 读取历史价格 + 常规 calendar + 常规外生变量

2. **DGF / Event 专家**
   - 学习突发尖峰、负价、短时异常
   - 读取显式检测事件或稀缺性相关特征

3. **Scarcity / Interconnector 路由器**
   - 决定何时更相信事件专家
   - 读取 PD PASA / predispatch / interconnector / 备用容量 / net load / shortfall

4. **Tail-aware quantile head**
   - 为 point、quantile 和 tail event 输出不同但相关的预测
   - 尾部 quantile 允许比 point 更依赖事件专家

### 3.2 模型草图

```text
历史价格 + 常规协变量
          │
          ├──────────────► CTF Trend Expert ─────────┐
          │                                          │
          ├──────────────► Event / DGF Expert ───────┤
          │                                          │
          ├──────────────► Scarcity / IC Router ─────┤
          │                                          │
          └──────────────► Tail-aware Fusion Gate ───┤
                                                     ▼
                                 Point / Quantile / Tail Predictions
```

### 3.3 设计原则
- **趋势分支**只负责“主形状”。
- **事件分支**只负责“非平稳尖峰与稀缺”。
- **路由器**不直接输出预测，而是输出“专家权重”。
- **尾部 head**允许与 point head 不同的权重配置。

---

## 4. 实验路线图

### Phase 1：确认当前最强信号
目标：复现并固化已经验证过的收益。

#### 1.1 复现现有最强基线
建议对照以下版本：
- `main` / trunk
- `feature/event-field-detected`
- `feature/pdpasa-dgf`
- `feature/predispatch-inputs`
- `feature/interconnector-inputs`
- `feature/price-space-quantiles`

#### 1.2 先看四组指标
- **总体**：MAE / RMSE
- **尾部**：`crps_negative`, `q05_negative`, `crps_spike`, `q95_spike`
- **事件检测**：spike hit rate / false alarm
- **经济意义**：负价小时与 spike 小时的误差

---

### Phase 2：把“显式事件”做深
目标：验证显式事件是否比当前 detected-event DGF 更强。

#### 2.1 事件分支候选
1. **DetectedEventField**（已存在）[3]
2. **AdaptiveEventField**（位置可学习）[models.py 现有定义]
3. **FutureEventField**（输出未来 spike / trough mixture）[models.py 现有定义]

#### 2.2 关注问题
- 事件定位是否更稳？
- 事件分支是否真的只解释残差？
- 事件分支是否会吞掉主趋势？

#### 2.3 关键消融
- `event_field = false / true`
- `dgf_type = gabor / adaptive / detected`
- `event_state_source = dgf / raw / none`
- `event_recent_extremes = false / true`
- `event_echo = false / true`

---

### Phase 3：把稀缺性做成路由条件
目标：确认稀缺性变量是不是应该负责“开门”而不只是“加特征”。

#### 3.1 已有可用稀缺性信号
- `MAXSPARECAPACITY`
- `DEMAND10/50/90`
- `UIGF`
- `AGGREGATECAPACITYAVAILABLE`
- `net_load = DEMAND50 - UIGF`
- `predispatch` price / load / interchange
- `interconnector` import/export room

这些已经在 `build_pdpasa_exogenous.py` 和 `build_predispatch_exogenous.py` 的配置链中出现。[4][5]

#### 3.2 路由建议
把这些信号分别输入：
- CTF head
- DGF head
- fusion gate
- quantile gate
- event head

而不是只拼在同一个向量里。

#### 3.3 目标
- 事件头在 scarcity 高时更活跃；
- quantile 的上/下尾在 scaricity 和 interconnector 压力下更敏感；
- 普通时段不应该过度激活事件专家。

---

### Phase 4：尾部与校准优先
目标：让模型在尾部真的比 baseline 稳。

#### 4.1 评估指标
- `MAE`, `RMSE`
- `crps_quantile_approx`
- `crps_by_band`
- `pinball_q95_spike_hours`
- `pinball_q05_negative_hours`
- `spike_hit_rate_q95`
- `spike_false_alarm_rate_q95`
- `negative_hit_rate_q05`
- `negative_false_alarm_rate_q05`
- `coverage_90`

这些指标与 `forecasting/tail_metrics.py` 的定义一致。[6]

#### 4.2 预期现象
- 如果模型真的学到事件规律，spike / negative 的尾部指标应该明显优于 trunk。
- 如果模型只是在扩展自由度，尾部指标可能好一点但 false alarm 也会升。

---

## 5. Loss 设计建议

当前 `forecasting/losses.py` 已经有一个很好的起点：

- point loss
- quantile loss
- decomposition loss
- smoothness loss
- sparsity loss
- event_nll
- event_anchor[losses.py]

我建议下一版继续强化为下面这种形式：

$$
\mathcal{L} = \lambda_{p}\mathcal{L}_{point} + \lambda_{q}\mathcal{L}_{quantile} + \lambda_{d}\mathcal{L}_{decomp} + \lambda_{s}\mathcal{L}_{smooth} + \lambda_{z}\mathcal{L}_{sparse} + \lambda_{e}\mathcal{L}_{event\_nll} + \lambda_{a}\mathcal{L}_{event\_anchor} + \lambda_{t}\mathcal{L}_{tail}
$$

### 5.1 各项含义
- `L_point`：主点预测误差
- `L_quantile`：五分位 pinball loss
- `L_decomp`：CTF + DGF 重构历史
- `L_smooth`：限制 CTF 过度震荡
- `L_sparse`：限制事件门过密
- `L_event_nll`：未来事件 mixture likelihood
- `L_event_anchor`：让 spike / trough 概率与真实事件标签对齐
- `L_tail`：对负价小时和 spike 小时加权

### 5.2 推荐 tail loss
对每个 origin 单独加权：

- negative hours：`q05_negative + crps_negative`
- spike hours：`q95_spike + crps_spike`

权重建议：
- normal hours: 1.0
- negative hours: 2.0 ~ 4.0
- spike hours: 2.0 ~ 4.0

### 5.3 稀疏性与平衡
- `event_gate.mean()` 不要太高
- 如果 gate 太稀，事件模块学不到；太密则退化为平滑大模型
- 建议保留 target sparsity = 0.3 左右作为起点，再按验证集调

---

## 6. 与当前代码的对应关系

### 6.1 `forecasting/models.py`
这里已经有：
- `CausalResidualBlock`
- `TemporalConvForecastHead`
- `AdaptiveEventField`
- `DetectedEventField`
- `FutureEventField`
- `DualFieldLinearForecaster`

这说明模型已经足够接近“结构化事件系统”，下一步主要是：
- 让路由更明确；
- 让 tail head 更独立；
- 让稀缺性输入更像门控信号，而不是普通拼接特征。

### 6.2 `forecasting/losses.py`
已有：
- `DualFieldForecastLoss`

下一步建议继续强化：
- tail-aware loss
- route alignment loss
- event calibration loss

### 6.3 `forecasting/datasets.py`
已有：
- `price_transform = asinh`
- `future_exogenous`
- `ctf_exogenous`
- `quantile_exogenous`
- `predispatch_exogenous`
- `interconnector_exogenous`
- `origin_context`

这意味着数据层已经支持“多路输入、多种未来外生变量”，足够承载结构化路由实验。

### 6.4 `forecasting/train_event_field.py`
说明 event field 可以被冻结 trunk 后单独训练。[train_event_field.py]

这很适合做一个重要实验：
- 先训练 trunk；
- 再单独训练 event field；
- 看事件分支是否真能独立学到稀缺/尾部结构。

### 6.5 `forecasting/tail_metrics.py`
已经把 spike / negative 尾部指标定义出来。[tail_metrics.py]
这直接支持我们把尾部作为第一优先级。

---

## 7. 详细消融矩阵

| 版本 | CTF | DGF / Event | PD PASA | Predispatch | Interconnector | Quantile gate | Tail loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| B | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| C | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| D | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| E | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| F | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 7.1 解释
- A：当前 trunk 基线
- B：加稀缺性
- C：加 predispatch
- D：加 interconnector
- E：尾部分位 gate
- F：完整版本

### 7.2 预期排名
我预计最终最有希望的顺序大致是：

F > E > D > C > B > A

但在某些 region（例如更平滑的 TAS1），可能会出现：
- D / C 提升有限；
- B 的收益已经接近天花板；
- 说明该区域更适合趋势而不是事件。

这并不坏，反而能证明模型真的学会了“何时用事件专家”。

---

## 8. 模型结构草图（建议写进 md 的图）

```text
                ┌──────────────────────────────────────┐
                │          Historical Window          │
                │   price, demand, gas, calendar      │
                └──────────────────────────────────────┘
                              │
         ┌────────────────────┼─────────────────────┐
         │                    │                     │
         ▼                    ▼                     ▼
   CTF Trend Expert     Event / DGF Expert   Scarcity / IC Router
   (smooth trend)       (spike / trough)     (PD PASA, IC, predispatch)
         │                    │                     │
         └──────────────┬─────┴───────────┬─────────┘
                        ▼                 ▼
                Tail-aware Fusion Gate   Future Event Head
                        │                 │
                        └─────────┬───────┘
                                  ▼
                      Point + Quantile Forecasts
```

### 图的表达重点
- CTF 负责平滑；
- Event / DGF 负责尖峰与 trough；
- Router 用 future exogenous / scarcity / interconnector 决策；
- Tail head 单独照顾 spike / negative hours。

---

## 9. 推荐实验顺序

### 第 1 周
1. 复现 trunk 与当前最优 forecasting 版本。
2. 固化 `tail_metrics.py` 的评估表。
3. 先跑 `event_field` 相关的冻结训练。
4. 记录每个 region 的 tail gap。

### 第 2 周
1. 把 PD PASA / predispatch / interconnector 输入做成路由条件。
2. 加 `quantile_gate` 与 tail loss。
3. 跑 3 seed 消融。
4. 整理成主表 + 消融表 + 案例图。

### 第 3 周（如果要继续）
1. 做 region-specific 分析。
2. 做 error-by-price-band。
3. 做负价小时和 spike 小时案例图。
4. 将最强版本固化成正式实验分支。

---

## 10. 参考论文与代码链接

> 说明：下面尽量给出“论文链接 + 代码链接”双链接。若公开代码未找到，则标注“未公开代码链接”。

| # | 论文 | 代码 |
|---|---|---|
| 1 | [DualTimesField repository](https://github.com/zl-comment/DualTimesField) | [forecasting 目录](https://github.com/zl-comment/DualTimesField/tree/main/forecasting) |
| 2 | [Replace the DGF with explicitly detected price events](https://github.com/zl-comment/DualTimesField/commit/e2cd5f8436fd41dd96274d37472a38d4e5e9b8d5) | [same commit diff](https://github.com/zl-comment/DualTimesField/commit/e2cd5f8436fd41dd96274d37472a38d4e5e9b8d5) |
| 3 | [Add PD PASA scarcity DGF experiment](https://github.com/zl-comment/DualTimesField/commit/d767bf3f97b5a44c0f19a6f2cff8069ae1195e13) | [same commit diff](https://github.com/zl-comment/DualTimesField/commit/d767bf3f97b5a44c0f19a6f2cff8069ae1195e13) |
| 4 | [Add point-in-time predispatch interconnector inputs](https://github.com/zl-comment/DualTimesField/commit/67df468aa1388597a710bf59f5132e4ef7738bab) | [same commit diff](https://github.com/zl-comment/DualTimesField/commit/67df468aa1388597a710bf59f5132e4ef7738bab) |
| 5 | [Add tail tests, case studies, cost table, and a framework figure](https://github.com/zl-comment/DualTimesField/commit/0ccfb14d2aaa178d72e7d4d766fd18cc61683f57) | [forecasting/significance.py](https://github.com/zl-comment/DualTimesField/blob/main/forecasting/significance.py) |
| 6 | [Day-Ahead Electricity Price Forecasting for Volatile Markets Using Foundation Models with Regularization Strategy](https://arxiv.org/html/2602.05430) | 未公开代码链接 |
| 7 | [Probabilistic Mid- and Long-Term Electricity Price Forecasting](https://arxiv.org/abs/1703.10806) | [论文页提到 code repository](https://arxiv.org/abs/1703.10806) / 公开仓库未在本次检索中定位到 |
| 8 | [Forecasting Day-Ahead Electricity Prices in the Integrated Single Electricity Market](https://arxiv.org/abs/2408.05628) | 未公开代码链接 |
| 9 | [Variance Stabilizing Transformations for Electricity Price Forecasting in Periods of Increased Volatility](https://arxiv.org/html/2511.13603v2) | 未公开代码链接 |
| 10 | [Extreme Event Prediction with Hierarchical Knowledge Distillation and Expert Fusion (xTime)](https://arxiv.org/html/2510.20651v1) | 未公开代码链接 |
| 11 | [GateTS: Versatile and Efficient Forecasting via Attention-Inspired routed Mixture-of-Experts](https://arxiv.org/html/2508.17515v1) | 未公开代码链接 |
| 12 | [AME-TS: Anchored Mixture-of-Experts for Time Series Forecasting](https://arxiv.org/html/2605.25166v1) | 未公开代码链接 |
| 13 | [FAME: Forecastability-Aware Mixture of Experts for Heterogeneous Time Series Forecasting](https://arxiv.org/html/2606.08896v1) | [github.com/hit636/FAME](https://github.com/hit636/FAME) |
| 14 | [Analysing drivers and interdependencies in European electricity markets using XAI](https://arxiv.org/html/2606.19118v1) | 未公开代码链接 |
| 15 | [Electricity Price Forecasting in the Irish Balancing Market](https://arxiv.org/html/2402.06714v1) | [论文页提到 dataset / code](https://arxiv.org/html/2402.06714v1) / 公开仓库未在本次检索中定位到 |
| 16 | [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://github.com/tancik/fourier-feature-networks) | [GitHub repo](https://github.com/tancik/fourier-feature-networks) |
| 17 | [Implicit Neural Representations with Periodic Activation Functions](https://github.com/vsitzmann/siren) | [GitHub repo](https://github.com/vsitzmann/siren) |

---

## 11. 建议的文件级实现落点

如果下一步要落代码，我建议优先动这些文件：

- `forecasting/models.py`
  - 增加 / 强化 router
  - 区分 trend expert 与 event expert
  - 给 tail head 单独的 gating

- `forecasting/losses.py`
  - 加 `tail_loss`
  - 加 `router_regularization`
  - 加 `event_calibration_loss`

- `forecasting/datasets.py`
  - 把 `predispatch` / `interconnector` / `PDPASA` 的特征分层输出
  - 给不同 head 提供专属输入

- `forecasting/train.py`
  - 在训练日志里单独记录 tail 指标
  - 记录 event hit rate / false alarm rate

- `forecasting/tail_metrics.py`
  - 增加按 region、按 month、按 horizon 的分桶输出

---

## 12. 最终推荐的一句话

> **下一步最值得推进的，是“结构化事件预测 + 稀缺性路由 + 互联线驱动 + 尾部优先”的 DualTimesField forecasting 路线。**

这条路最符合你现有提交积累，也最容易形成真正可发表、可解释、可复现实验结论。
