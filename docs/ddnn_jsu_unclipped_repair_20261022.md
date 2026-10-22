# DDNN-JSU 去钳位修复与复跑协议

本轮只修正已证实的实现问题，不根据 2023/2024 的已见测试成绩重新调参。
原 45 格运行 `aligned_expansion_20261022_r1`、模型、冻结源代码和全部预测保持不变。

## 修复

- 删除解析逆 CDF 的 ±12 自变量钳位。Phi⁻¹、sinh 和分位合成统一在 float64 计算，不把尾部转回 float32。
- NLL、MLP 架构、初始化、scale floor=0.1、优化器和损失均不变。
- 非有限或非法输入明确报错；不得用输出裁剪、截断或伪造分位维持运行成功。
- 选择与测试使用同一未截断公式。补充大正负尾、float32 溢出而 float64 有限的案例、SciPy 分位/NLL 等价性及非有限失败测试。
- 文档修正为实际的正态变换参数化，并正确写出 NLL 的 Jacobian。

## 固定复跑

仅 DDNN-JSU，NSW1/QLD1/TAS1 × seeds2026/2027/2028，共 9 格。
沿用原七块输入、冻结缓存及标准化、原时间切分、完整 17,521 测试起点、原评分器。
从头训练和选择 checkpoint，禁止仅替换旧检查点的预测导出。

使用原矩阵**实际生效**的训练参数：Adam、lr=3e-4、weight_decay=1e-4、batch256、
最大100 epochs、patience15，选择期五分位 pinball（与 QS=2×pinball 同序）。
旧文档及 `experiment.neural.learning_rates` 写过 1e-3，但 adapter 实际读取
`training.learning_rate`，旧正式矩阵该值为 3e-4，运行 metadata 亦可验证。
本次将当前矩阵/单模型配置的声明统一为 3e-4，不改旧冻结配置或任何历史成绩。

新配置：`configs/experiments/aemo_forecast_ddnn_jsu_unclipped_20261022.yaml`。
其 training 与旧正式矩阵逐字段一致；资源使用 GPU6/7，减少对其他任务的影响。
输出：`outputs/forecasting/ddnn_jsu_unclipped_20261022_r1`。
复跑仍是回顾性固定配置实验，不构成未见测试或超参优化结果。

## 替换与验收

新 9 格全部完成、所有共享输入和评分/哈希校验通过后，合并表显式指定 DDNN 的新来源。
其余新增36格、原90格、TimeXer/PriceFM18格保持原预测与来源。主表仍为153格/17模型。
原 DDNN9格保留为 superseded 记录，不能与修复9格双计数，不能静默覆盖来源。
报告默认继续拒绝重复模型；只有显式指定且覆盖完全相同地区/种子集合的替换才允许。
被替换的旧结果也保持审计，不删除；任一运行不完整则不发布完整主榜。

修复验收标准是方法/数值/协议正确，**不要求分数优于旧钳位版本或其他基线**。
如果未截断结果很差或有限尾部很大，应如实报告，不能再次按测试效果添加钳位。

```bash
.venv/bin/python -m forecasting.aligned_expansion prepare \
  --config configs/experiments/aemo_forecast_ddnn_jsu_unclipped_20261022.yaml \
  --run-root outputs/forecasting/ddnn_jsu_unclipped_20261022_r1 \
  --extend-from outputs/forecasting/aligned_baselines_20261012
.venv/bin/python -m forecasting.aligned_expansion run \
  --run-root outputs/forecasting/ddnn_jsu_unclipped_20261022_r1
```
