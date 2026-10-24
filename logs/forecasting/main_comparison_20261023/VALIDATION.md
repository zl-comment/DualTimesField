# 当前主表验证（2026-10-24）

25 个输出、225 格；另保留 18 格历史结构对照。所有 243 个来源预测均重新评分并对齐原测试样本。每个模型覆盖三个地区和三个 seed；确定性 seed 重复不代表独立重复实验。

更新命令：`PYTHONPATH=. .venv/bin/python scripts/update_latest_main_comparison.py`；可用 `--foundation-run` 指定基础模型原始运行目录。缺失或失败审计、缺格、来源哈希或评分不一致均拒绝发布。

详见 [机器可读核验](verification.json)、[主表](README.md)和[基础模型审查边界](../foundation_baselines_20261023/README.md)。
