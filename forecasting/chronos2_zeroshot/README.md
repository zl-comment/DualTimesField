# chronos2_zeroshot — Chronos-2 零样本时序基础模型基线（第二批 2026-10-23）

冻结 amazon/chronos-2 权重的零样本对照（非 Chronos-T5/Bolt/synth 变体）。
权威记录见 [source.json](source.json)。

- 五个公共分位全部是检查点原生训练级别（21 级 [.01–.99]），无插值、无端点
  复制 → 进入 Q 表。
- 输入映射受限：价格历史（逆标准化一次，输出即原价 RRP）+ 需求 past 协变量
  + 日历 past/future 协变量；origin_context 与仅未来外生块因官方接口要求
  历史段而无法合法映射，已在 registry/产物中声明，不作等输入比较。
- 每起点独立预测（cross_learning=False），显式 context_length=72、
  prediction_length=24。
- 环境：[ENVIRONMENT.md](ENVIRONMENT.md)、[requirements.lock.txt](requirements.lock.txt)。
