# Anchor 三种方案确认实验与非对称尺度头

范围：回顾性开发实验；2021/2022 已用于前期诊断，不能称为未见测试集。
同一七折树 OOF、全小时起点、24 小时目标、三地区、三个神经随机种子、每组固定 20 轮。
以下为地区等权宏平均，再对三个种子求均值和样本标准差；不是三模型预测集成。
QS = 2 × 五分位点平均 pinball，MAE 使用 q50，均在原始价格单位计算。

| 方案 | 输出 | 年份 | Full MAE（均值 ± SD） | Full QS（均值 ± SD） | 回退数/9 | 最后两轮选中数/9 |
|---|---|---|---:|---:|---:|---:|
| balanced | raw | 2021 | 28.3147 ± 0.1198 | 20.2574 ± 0.2398 | 0 | 0 |
| balanced | raw | 2022 | 69.3341 ± 0.4859 | 40.9298 ± 0.3857 | 0 | 0 |
| balanced | guarded | 2021 | 28.0511 ± 0.1076 | 20.0240 ± 0.2632 | 3 | 0 |
| balanced | guarded | 2022 | 69.6292 ± 0.3471 | 42.2681 ± 0.2490 | 3 | 0 |
| location_only | raw | 2021 | 28.2678 ± 0.0569 | 19.9496 ± 0.0350 | 0 | 0 |
| location_only | raw | 2022 | 68.8733 ± 0.0481 | 42.0650 ± 0.0924 | 0 | 0 |
| location_only | guarded | 2021 | 28.0782 ± 0.1651 | 19.9010 ± 0.0617 | 3 | 0 |
| location_only | guarded | 2022 | 68.1300 ± 0.4509 | 41.8017 ± 0.1724 | 0 | 0 |
| identity_anchor | raw | 2021 | 28.2370 ± 0.1188 | 20.0572 ± 0.1315 | 0 | 0 |
| identity_anchor | raw | 2022 | 67.0853 ± 0.2979 | 41.9721 ± 0.0469 | 0 | 0 |
| identity_anchor | guarded | 2021 | 28.1401 ± 0.0778 | 19.9510 ± 0.0611 | 1 | 0 |
| identity_anchor | guarded | 2022 | 68.7495 ± 0.3515 | 42.3569 ± 0.2079 | 0 | 0 |
| asymmetric_scale | raw | 2021 | 28.2936 ± 0.0722 | 20.2565 ± 0.2195 | 0 | 0 |
| asymmetric_scale | raw | 2022 | 69.0190 ± 1.1446 | 41.1401 ± 0.2218 | 0 | 0 |
| asymmetric_scale | guarded | 2021 | 28.0493 ± 0.1066 | 19.9835 ± 0.2884 | 3 | 0 |
| asymmetric_scale | guarded | 2022 | 69.6511 ± 0.6045 | 42.1369 ± 0.1162 | 2 | 0 |

| 方案 | 输出 | 种子均值通过全部门槛 | 三个种子分别通过全部门槛 | 稳定通过 |
|---|---|---|---|---|
| balanced | raw | False | False | False |
| balanced | guarded | False | False | False |
| location_only | raw | False | False | False |
| location_only | guarded | False | False | False |
| identity_anchor | raw | False | False | False |
| identity_anchor | guarded | False | False | False |
| asymmetric_scale | raw | False | False | False |
| asymmetric_scale | guarded | False | False | False |

验收保留原四条件：宏平均 MAE/QS 不劣于 B 和同种子 balanced；至少一个宏指标严格优于 B；每地区不劣于 B；每地区春季不劣于 B。2021/2022 均需通过。数值容差沿用 RTOL=1e-7。
稳定通过还要求三个种子各自通过；均值通过而部分种子失败仅记为不稳定候选。
raw 与 guarded 分别验收；guarded 的成功不能自动解释为原始网络结构成功。
最后两轮仍被选中的次数用于提示预算边界，不能仅因完成 20 轮便断言已经收敛。
三个种子只改变神经初始化与训练顺序，树种子固定；未计算市场总体置信区间或显著性。
所有分组、尾部 pinball、覆盖率、区间宽度、Anchor 位移与历史校正抵消量均为事后诊断，不能用于重选 checkpoint/alpha。
NPZ 保留原始/守护预测、起点、目标时间和实际值；results.csv 包含地区/年份/种子/季节/价格分组的完整结果。
数据可用性审计沿用归档时间戳代理，不能据此宣称已核实真实发布延迟。
