# moirai2_zeroshot — Moirai-2 零样本时序基础模型基线（第二批 2026-10-23）

冻结 Salesforce/moirai-2.0-R-small（uni2ts 固定 commit cfd46d4）的零样本对照。
权威记录见 [source.json](source.json)。

- 原生分位恰为 [0.1–0.9]（检查点 config.json 核实），无 q05/q95、无采样路径：
  不外推、不复制端点 → **P 表**（点 = 原生 q50），原生 q10/q90 另存辅助诊断。
- 输入映射受限：价格历史（逆标准化一次）为 past_target；需求为 past 协变量；
  日历合并为 feat_dynamic_real（96×10）。origin_context 与仅未来外生块无合法
  槽位，已在 registry/产物声明。
- 环境：[ENVIRONMENT.md](ENVIRONMENT.md)、[requirements.lock.txt](requirements.lock.txt)。
