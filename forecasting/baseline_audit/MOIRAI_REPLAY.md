# Moirai 真实权重全测试集对照（2026-11-25）

## 结论

已完成真实权重、真实冻结输入、原生产批次大小下的修复前后全量推理对照，不是模拟测试，也不只是抽样。

- NSW1、QLD1、TAS1 各 17,521 个预测起点，共 52,563 个起点、1,261,512 个起点×预测步。
- 修前/修后全部九个原生分位数，共 11,353,608 个数值，逐元素完全相同，最大绝对差为 **0**。
- 修后 q50 与原 seed 2026/2027/2028 的九格点预测归档逐元素完全相同，最大绝对差为 **0**。
- 所有原生预测日期由错误的 `origin + 72h` 修正为 `origin`。原标准预测归档的时间轴由共享输入重建，本来就与真实标签对齐；不能将此问题描述为主表标签错移了72小时。
- 因此，本次起点修复不要求进一步重训或重跑 Moirai，主表的 Moirai 分数无需更改。主表未重新发布，也没有覆盖旧归档。

## 实测结果

| 地区 | 完整测试起点 | 修前/修后九分位最大差 | 修后/旧九格中对应三seed点预测最大差 | 修后 MAE | 修后 RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| NSW1 | 17,521 | 0 | 0 | 66.5517290393 | 415.7187812770 |
| QLD1 | 17,521 | 0 | 0 | 64.0873691124 | 293.1191201978 |
| TAS1 | 17,521 | 0 | 0 | 36.6223057019 | 166.5133344492 |

地区等权宏 MAE 为 **55.753801284510324**，地区平均 RMSE 为 **291.7837453079747**，保持主表显示值不变。主表 CSV 的 RMSE 最末位与本次汇总约相差 `6e-14`，属于浮点归约舍入；原始预测数组完全一致。
Moirai 仍属于 P 表，不因核验九个原生分位数而虚构缺少的 q05/q95 或增加五分位 QS。

## 对照设计与真实执行

1. 验证原归档 adapter 源码与 `jobs.json` 的 SHA 一致，读取该真实旧实现；不是手写一个假想旧版本。
2. 从原 Moirai 专用环境加载真实权重，核对其 SHA 与九格原预测的权重身份完全一致；离线执行，无下载。
3. 修前和修后 adapter 使用同一个冻结网络与官方 predictor，只隔离起点/辅助输出布局变化；各批分别执行真实网络前向，不用旧预测替代“修前推理”。
4. 保持原生产分块和批次为 512，首批通过后覆盖全部剩余测试起点，包含末尾不足512的一批。三个 seed 归档均做比较，但未把重复的确定性输出当成三次独立训练证据。
5. 逐批截取官方输出的九分位数组与原生预测日期；同时检查 q50、q10/q50/q90 辅助数组及日期关系。辅助数组统一到 `[N,24,3]` 后比较。
6. 原输入、权重、源码和安装包源码在执行前后指纹一致。完整推理与结果整理约 **140.1 秒**。

官方实现的 `_generate_time_id` 从窗口内相对序列位置生成时间索引，日历协变量由共享输入明确提供；本次修复不改变这些数值。这解释了为什么原生日期会变化，而在本次完整测试集上预测值完全不变。

本次比较归档保存完整修前/修后点预测及辅助分位。它是诊断产物，不伪装为旧实验当时已保存的辅助文件，也没有替换原 worker 的正式结果目录。

## 可追溯证据

- 基准提交：`e16f9b396f9143f88f42da2031792495170e3be6`。
- 实验分支：`research/moirai-real-inference-check-20261125`。
- 执行提交：`32daf8b400d06db7d8151944f1a8420674acffff`。
- 源码指纹：`544d3edfab7623f2834181f88a09470f66f6a8d319370e5b252bf6a27201389f`。
- 权重 revision：`30f43ff08c8494f4943ae1521e9d4e94a0fbb389`；SHA256：`fb5652a3db8ea572606221b7cb1e77bb8962b168e4d4cc752cf31ceb04074669`。
- 环境：原 Moirai Python 环境，uni2ts 固定提交 `cfd46d4510ed`，torch 2.4.1，NumPy 1.26.4，GluonTS 0.14.4；NVIDIA L40，物理 GPU 7。
- [执行清单及72份输入散列](../../outputs/forecasting/moirai_real_replay_20261125/manifest.json)、[逐地区/seed对照结果](../../outputs/forecasting/moirai_real_replay_20261125/results.json)、[产物散列](../../outputs/forecasting/moirai_real_replay_20261125/completed.json)。各地区完整比较数组位于同目录的 `*_comparison.npz`。
- [对照入口](moirai_replay.py)；新增5项比较/时间换算单元测试，`forecasting/tests` 合计 **55 passed**。真实模型证据来自本次 GPU 推理，不来自 mock 用例。

复现时使用新输出目录，入口拒绝覆盖已有结果；须先提交代码并保持工作区干净：

```bash
PYTHONDONTWRITEBYTECODE=1 \
HF_HOME=/root/workspace/Code/DualTimesField-foundation-baselines/weights/hf_home \
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7 \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
/root/workspace/Code/DualTimesField-foundation-baselines/.venvs/moirai2/bin/python \
  -m forecasting.baseline_audit.moirai_replay \
  --run-root /root/workspace/Code/DualTimesField-foundation-baselines/outputs/forecasting/foundation_baselines_20261023_r1 \
  --output-dir outputs/forecasting/moirai_real_replay_NEW
```

本结论限定于当前归档权重、实现、生产批次及2023—2024测试集；不外推为任意批次、硬件、模型版本或未来数据都逐位一致。本次也未解决预训练重叠、实时公开可得性等原有研究边界。
