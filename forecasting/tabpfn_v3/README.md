# tabpfn_v3 — TabPFN-3 表格基础模型基线（第二批 2026-10-23）

冻结的第三代 TabPFN 本地回归权重，作为表格预训练基础模型对照。定位、
数据边界与能力声明的权威记录见 [source.json](source.json)；本 README 只做导览。

- 原生实现：`tabpfn` 9.2.0 包 + `Prior-Labs/tabpfn_3` 的
  `tabpfn-v3-regressor-v3_default.ckpt`（revision 24a16a89）。包版本 ≠ 模型代际；
  默认构造器会解析到 v3.5，本适配器显式固定 V3 检查点本地路径。
- 论文与引用：见 [CITATIONS.bib](CITATIONS.bib)（V3 技术报告为主引用，
  Nature v2 论文仅作谱系）。不宣称复现论文基准成绩。
- 输入：公共七块按 `INPUT_KEYS` 顺序共享展平为 1465 列，无增删特征；
  按 horizon 建 24 个独立上下文回归任务。
- 阶段：train 标签仅作 in-context 支持集（label context fit，不训练权重，
  不回避这一标签接触）；selection/calibration 不使用；test 侧为无标签
  `PredictionView`。种子只控制集成预处理 RNG。
- 输出：原生分布分位 [.05,.10,.50,.90,.95]，原价 RRP 空间，进入 Q 表；
  点评分用重排 q50。
- 环境：见 [ENVIRONMENT.md](ENVIRONMENT.md) 与 [requirements.lock.txt](requirements.lock.txt)。

相关工作文件：适配器 `adapter.py`（七块映射与阶段边界）、`model.py`（官方库
加载与预测）。正式运行由 `forecasting.foundation_experiment` 调度。
