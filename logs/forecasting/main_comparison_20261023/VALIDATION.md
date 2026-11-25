# 当前主表验证（2026-11-25）

2026-11-25 修复后刷新：已从原预测重新评分全部243个来源格（主表225格＋历史18格），保持三地区等权、三个seed分别评分后汇总的口径。Moirai真实权重全测试集对照确认修前/修后预测及原九格归档逐元素一致，主表数值无需因该修复改变；本次不是重新训练全部模型。R25/R27更正结果单独归档，不将组合增益混入基线行。详见[修复及重算](../../../forecasting/baseline_audit/RESULTS.md)与[Moirai全量对照](../../../forecasting/baseline_audit/MOIRAI_REPLAY.md)。

25 个输出、225 格；另保留 18 格历史结构对照。所有 243 个来源预测均重新评分并对齐原测试样本。每个模型覆盖三个地区和三个 seed；确定性 seed 重复不代表独立重复实验。

更新命令：`PYTHONPATH=. .venv/bin/python scripts/update_latest_main_comparison.py`；可用 `--foundation-run` 指定基础模型原始运行目录。缺失或失败审计、缺格、来源哈希或评分不一致均拒绝发布。

详见 [机器可读核验](verification.json)、[主表](README.md)和[基础模型审查边界](../foundation_baselines_20261023/README.md)。
