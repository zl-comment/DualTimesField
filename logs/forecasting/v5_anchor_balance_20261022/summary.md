# Anchor 平衡修复：首轮开发筛选

三地区等权平均；screen 采样、3 epoch、神经 seed=2026、树 seed=2026。不是全量收敛或显著性结论。

| 评估年 | 方案 | 选择方式 | 宏MAE | 宏QS | 回退地区数 |
|---|---|---|---:|---:|---:|
| 2021 | B | baseline | 28.2080 | 19.9207 | 0 |
| 2021 | legacy | raw | 28.2376 | 20.2714 | 0 |
| 2021 | legacy | guarded | 28.0880 | 20.2096 | 0 |
| 2021 | balanced | raw | 28.0999 | 20.0363 | 0 |
| 2021 | balanced | guarded | 28.0553 | 20.0131 | 1 |
| 2021 | identity_anchor | raw | 27.9837 | 19.8587 | 0 |
| 2021 | identity_anchor | guarded | 28.0044 | 19.8724 | 1 |
| 2022 | B | baseline | 71.6513 | 43.2991 | 0 |
| 2022 | legacy | raw | 70.0133 | 41.9823 | 0 |
| 2022 | legacy | guarded | 71.0892 | 42.9601 | 1 |
| 2022 | balanced | raw | 69.5507 | 42.2619 | 0 |
| 2022 | balanced | guarded | 70.8229 | 42.9101 | 1 |
| 2022 | identity_anchor | raw | 69.3141 | 42.5739 | 0 |
| 2022 | identity_anchor | guarded | 69.7222 | 42.6589 | 0 |

预注册筛选标准：未通过。

各地区、季度、春季与预测价格带诊断见 report.json；逐地区主表见 report.md。
每个 raw/guarded 的选中 epoch 与 alpha 都保留。回退只保证选择期，不保证后续年份。
本轮未读取2023–2024最终测试。没有根据本轮评估重新调整权重或收缩候选。

具体未通过项：
- 2021: macro_noninferior_to_B_and_legacy, each_region_noninferior_to_B, each_spring_noninferior_to_B
- 2022: 全部满足
