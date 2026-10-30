# r6 真实 2022 三地区开发集评估（A/B/C/D）

- 轮次：`agentic-m2m3-dev-eval-c2-20261030-r6`；分支：`research/agentic-m2m3-dev-eval-c2-20261030-r6`
- 地区：NSW1 / QLD1 / TAS1；起报网格：2022 年每 7 天 00:00 AEST（UTC+10，与数据时间轴一致），共 53 个起报/地区；72h 历史 → 24h 预测；分位数 0.05/0.10/0.50/0.90/0.95（q50 为点预测）；seed 2026。
- A=冻结数值流（Chronos-2 zero-shot，CPU，离线，权重指纹固定）；B=固定规则；C=固定单次 LLM（qwen3-30b3a，temperature 0）；D=白名单有界 Agent（≤3 轮、≤4 次工具调用）。
- 成功/失败：{"B": [53, 53, 53], "C": [53, 53, 53], "D": [22, 20, 20]}（每地区）

## 全年（FULL）宏观指标

| Arm | MAE | QS | 覆盖80% | 覆盖90% |
|---|---|---|---|---|
| A | 52.07 | 30.33 | 74.4% | 87.2% |
| B | 53.15 | 32.14 | 80.1% | 91.5% |
| C | 62.75 | 36.73 | 66.8% | 81.4% |
| D | 75.16 | 44.48 | 66.8% | 82.0% |

## 增量（负值=左侧 Arm 更优）

| 对比 | MAE Δ(宏观) | QS Δ(宏观) |
|---|---|---|
| D-C | -0.536 | -0.430 |
| D-B | +12.096 | +6.240 |
| C-B | +9.604 | +4.591 |

## 分地区全年 MAE / QS

| 地区 | A | B | C | D |
|---|---|---|---|---|
| NSW1 | 46.92/25.70 | 48.62/27.09 | 54.57/29.10 | 66.63/35.58 |
| QLD1 | 52.33/30.40 | 52.96/33.19 | 69.37/41.15 | 88.21/54.47 |
| TAS1 | 56.96/34.90 | 57.85/36.16 | 64.31/39.96 | 70.64/43.40 |

## 失败、延迟与用量

- NSW1/B: 失败原因 {}；平均延迟 0ms；LLM 调用 0；工具调用 0；tokens 0+0；成本 0.0 AUD（本地端点）
- NSW1/C: 失败原因 {}；平均延迟 554ms；LLM 调用 53；工具调用 0；tokens 29931+1748；成本 0.0 AUD（本地端点）
- NSW1/D: 失败原因 {"BUDGET_EXHAUSTED_TOO_MANY_TOOL_CALLS": 20, "INVALID_JSON": 7, "BUDGET_EXHAUSTED_NO_DECISION": 4}；平均延迟 2283ms；LLM 调用 116；工具调用 212；tokens 254598+7036；成本 0.0 AUD（本地端点）
- QLD1/B: 失败原因 {}；平均延迟 0ms；LLM 调用 0；工具调用 0；tokens 0+0；成本 0.0 AUD（本地端点）
- QLD1/C: 失败原因 {}；平均延迟 561ms；LLM 调用 53；工具调用 0；tokens 29967+1771；成本 0.0 AUD（本地端点）
- QLD1/D: 失败原因 {"BUDGET_EXHAUSTED_TOO_MANY_TOOL_CALLS": 24, "BUDGET_EXHAUSTED_NO_DECISION": 4, "INVALID_JSON": 5}；平均延迟 2237ms；LLM 调用 118；工具调用 212；tokens 261012+6861；成本 0.0 AUD（本地端点）
- TAS1/B: 失败原因 {}；平均延迟 0ms；LLM 调用 0；工具调用 0；tokens 0+0；成本 0.0 AUD（本地端点）
- TAS1/C: 失败原因 {}；平均延迟 555ms；LLM 调用 53；工具调用 0；tokens 29854+1756；成本 0.0 AUD（本地端点）
- TAS1/D: 失败原因 {"BUDGET_EXHAUSTED_TOO_MANY_TOOL_CALLS": 18, "INVALID_JSON": 12, "BUDGET_EXHAUSTED_NO_DECISION": 3}；平均延迟 2577ms；LLM 调用 113；工具调用 212；tokens 245760+8072；成本 0.0 AUD（本地端点）

## H1 / H2

- H1: NSW1/A=39.34, NSW1/B=42.36, NSW1/C=46.17, NSW1/D=58.49, QLD1/A=51.46, QLD1/B=52.04, QLD1/C=76.15, QLD1/D=127.08, TAS1/A=46.88, TAS1/B=48.02, TAS1/C=53.26, TAS1/D=76.55
- H2: NSW1/A=54.21, NSW1/B=54.65, NSW1/C=62.66, NSW1/D=73.42, QLD1/A=53.17, QLD1/B=53.85, QLD1/C=62.84, QLD1/D=62.30, TAS1/A=66.67, TAS1/B=67.32, TAS1/C=74.95, TAS1/D=65.81

## 身份与守卫

- 标签 CSV SHA-256（NSW1）：`bb11dfb7d33af1996fca13359f90d0231c4968f24a0e4b57dcb0560ffe50befb`；每个 npz 的 SHA-256 见 results/summary.json。
- 严格 as-of：历史/工具仅访问 available_at ≤ origin 的行；标签窗口 24 行必须全部属于 2022 validation split（2023–2024 test 行被硬性排除）。
- 结果身份：四个 Arm 的 origin/target/label 数组逐元素一致；增量仅在双方成功的共享单元上计算。
- 真实退出码：每次 LLM/工具调用失败按原因记录且不得计为成功；进程退出码真实透传。
- 本轮无任何训练、拟合、调参、模型下载或 GPU 使用。
