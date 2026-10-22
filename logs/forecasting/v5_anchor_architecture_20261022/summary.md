# Anchor 结构与训练耦合：开发筛选结果

三地区等权；screen 采样、6 轮（staged 为 3+3）、神经和树 seed=2026。
本轮所有历史/联合训练统一平衡损失，不能直接与首轮三轮实验作结构归因。

| 年份 | 方案 | 选择 | Anchor MAE | Full MAE | Full QS | 回退地区数 |
|---|---|---|---:|---:|---:|---:|
| 2021 | B | baseline | 28.2080 | 28.2080 | 19.9207 | 0 |
| 2021 | balanced | raw | 28.2373 | 28.0916 | 20.0508 | 0 |
| 2021 | balanced | guarded | 28.1173 | 27.9864 | 20.0043 | 0 |
| 2021 | identity_anchor | raw | 28.2080 | 27.9911 | 19.8224 | 0 |
| 2021 | identity_anchor | guarded | 28.2080 | 28.0355 | 19.8562 | 0 |
| 2021 | location_only | raw | 28.2566 | 28.1198 | 19.7822 | 0 |
| 2021 | location_only | guarded | 28.1272 | 27.9831 | 19.7507 | 0 |
| 2021 | separate_heads | raw | 28.3286 | 28.1495 | 20.1390 | 0 |
| 2021 | separate_heads | guarded | 28.1337 | 27.9772 | 20.0794 | 0 |
| 2021 | staged | raw | 28.5355 | 28.3525 | 20.1030 | 0 |
| 2021 | staged | guarded | 28.1871 | 28.1070 | 20.0203 | 1 |
| 2021 | wide_head | raw | 28.2381 | 28.0783 | 20.0950 | 0 |
| 2021 | wide_head | guarded | 28.2508 | 28.1090 | 20.0673 | 1 |
| 2022 | B | baseline | 71.6513 | 71.6513 | 43.2991 | 0 |
| 2022 | balanced | raw | 70.1279 | 68.2324 | 41.4909 | 0 |
| 2022 | balanced | guarded | 70.4010 | 69.8167 | 42.3091 | 1 |
| 2022 | identity_anchor | raw | 71.6513 | 67.4281 | 41.9851 | 0 |
| 2022 | identity_anchor | guarded | 71.6513 | 67.9517 | 42.0644 | 0 |
| 2022 | location_only | raw | 70.0669 | 67.8423 | 41.8068 | 0 |
| 2022 | location_only | guarded | 69.9144 | 68.2492 | 41.9440 | 0 |
| 2022 | separate_heads | raw | 70.5981 | 68.7560 | 41.4676 | 0 |
| 2022 | separate_heads | guarded | 70.6216 | 69.9407 | 42.3309 | 1 |
| 2022 | staged | raw | 71.3320 | 70.1755 | 41.9441 | 0 |
| 2022 | staged | guarded | 71.4829 | 70.9352 | 42.9836 | 1 |
| 2022 | wide_head | raw | 70.0224 | 68.1846 | 41.2381 | 0 |
| 2022 | wide_head | guarded | 70.3267 | 69.6912 | 42.2681 | 1 |

## 预先固定的筛选标准

- identity_anchor: 未通过；2021: macro_noninferior_to_B_and_balanced, each_spring_noninferior_to_B；2022: each_spring_noninferior_to_B
- location_only: 未通过；2021: each_region_noninferior_to_B
- separate_heads: 未通过；2021: macro_noninferior_to_B_and_balanced, each_region_noninferior_to_B, each_spring_noninferior_to_B；2022: macro_noninferior_to_B_and_balanced
- staged: 未通过；2021: macro_noninferior_to_B_and_balanced, each_region_noninferior_to_B, each_spring_noninferior_to_B；2022: macro_noninferior_to_B_and_balanced
- wide_head: 未通过；2021: macro_noninferior_to_B_and_balanced, each_region_noninferior_to_B, each_spring_noninferior_to_B

## raw 跨年机制线索

要求两个年度宏 MAE/QS 均不差于 balanced，且每年至少一项严格改善；不是因果确证。

- identity_anchor: 不符合（{'2021': True, '2022': False}）
- location_only: 不符合（{'2021': False, '2022': False}）
- separate_heads: 不符合（{'2021': False, '2022': False}）
- staged: 不符合（{'2021': False, '2022': False}）
- wide_head: 不符合（{'2021': False, '2022': True}）

全部检查点和 alpha 在评估前确定。未读取 2023–2024；未按本轮评估重新调参。
六轮单种子筛选不能证明收敛或显著性；回退 B 不构成学习收益。
详细地区/季节/价格带在 report.json；校正幅度在 calibration.json；逐轮记录在 training/。
