# 两项校正稳健性开发实验

三个逻辑方案：原 asymmetric_scale 对照、仅改四季度选择、仅加校正幅度边界。
36 次物理拟合、54 个逻辑方案；每条轨迹 20 epochs，三个地区、两个开发年、三个神经 seed。
multiwindow 与 reference 使用完全相同的训练轨迹，raw 诊断重复同一输出，不是独立训练证据。
旧 72 次确认失败原样保留；本轮 2021/2022 为回顾性开发，未更新正式主表。

| 方案 | 输出 | 评估年 | 宏 MAE ± SD | 宏 QS ± SD | 回退/9 | 末两轮选中/9 |
|---|---|---|---:|---:|---:|---:|
| reference | guarded | 2021 | 28.0493 ± 0.1066 | 19.9835 ± 0.2884 | 3 | 0 |
| reference | guarded | 2022 | 69.6511 ± 0.6045 | 42.1369 ± 0.1162 | 2 | 0 |
| reference | raw | 2021 | 28.2936 ± 0.0722 | 20.2565 ± 0.2195 | 0 | 0 |
| reference | raw | 2022 | 69.0190 ± 1.1446 | 41.1401 ± 0.2218 | 0 | 0 |
| multiwindow | guarded | 2021 | 28.0328 ± 0.0596 | 19.9114 ± 0.1352 | 3 | 0 |
| multiwindow | guarded | 2022 | 70.5791 ± 0.1015 | 42.7516 ± 0.0147 | 6 | 0 |
| multiwindow | raw | 2021 | 28.2936 ± 0.0722 | 20.2565 ± 0.2195 | 0 | 0 |
| multiwindow | raw | 2022 | 69.0190 ± 1.1446 | 41.1401 ± 0.2218 | 0 | 0 |
| bounded_calibration | guarded | 2021 | 28.0391 ± 0.0548 | 19.8878 ± 0.0445 | 3 | 0 |
| bounded_calibration | guarded | 2022 | 68.8435 ± 1.3340 | 42.0012 ± 0.3934 | 0 | 1 |
| bounded_calibration | raw | 2021 | 28.3210 ± 0.0525 | 20.0326 ± 0.0738 | 0 | 0 |
| bounded_calibration | raw | 2022 | 68.8491 ± 1.0113 | 41.6958 ± 0.3771 | 0 | 0 |

| 方案 | 输出 | 种子均值全部通过 | 每 seed 全部通过 | 稳定通过 |
|---|---|---|---|---|
| reference | raw | False | False | False |
| reference | guarded | False | False | False |
| multiwindow | raw | False | False | False |
| multiwindow | guarded | False | False | False |
| bounded_calibration | raw | False | False | False |
| bounded_calibration | guarded | True | False | False |

旧四门槛全部保留：宏 MAE/QS 不劣于 B/旧 balanced、至少一项严格优于 B、每地区不劣于 B、每地区春季不劣于 B；新增宏 MAE/QS 不劣于当前 reference。
两年与三个 seed 均需通过。回退、单年通过、个别 seed 通过不视为稳定成功。
参考轨迹的每轮损失/训练顺序/原选模和 raw/guarded/最终参数均已与旧档精确核对。
均值为每 seed 三地区等权后再汇总，未做预测集成或显著性检验。树 seed 固定为 2026。
所有参数、窗口和边界均在本轮训练前冻结；不根据本轮评估修改这些设置。
