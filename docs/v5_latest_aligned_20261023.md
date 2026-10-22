# 最新 Full_joint：三神经种子同协议主表更新

用户要求将当前方法与相关对照替换为最新结果。本轮在训练和评估前冻结以下设置。
采用最新实现的 `asymmetric_scale`（平衡损失＋非对称 Anchor 尺度头），
不是从四方案的测试分数中选择赢家；此前四方案均未通过全部开发验收，仍如实保留该结论。

## 训练与选择

- 地区 NSW1、QLD1、TAS1；神经种子 2026、2027、2028，共 9 次拟合。
- 树固定 seed 2026，沿用已审计的七个 chronological OOF bundles；
  三种子波动仅反映神经初始化/训练顺序，不是整套树＋神经模型的独立重训。
- 恢复正式时间边界：神经训练 2018-01-01 至 2022-01-01，选择 2022 上半年；
  使用已有严格成熟度/purging。2022 下半年仅用于成熟历史记忆，不参与模型或 alpha 选择。
- 与最近确认实验相同：全小时起点、20 epochs、batch 128、预测 batch 256、
  AdamW lr 0.001、weight decay 0.0001、clip 5，FP32 神经/FP64 历史求解，关闭 TF32。
- 原 `balance_loss`、非对称头、候选数 16、730 天记忆窗口保持不变。
- raw checkpoint：选择期 Full QS 最小，相同取最早。
- guarded checkpoint/alpha：既有四守护规则，alpha∈{0,0.25,0.5,1}；
  Anchor/Full 以及同检查点关闭干预共用此 alpha，不为干预另行选参。
- 9 次拟合和全部 checkpoint/alpha 冻结后才可加载 2023—2024 测试数据。
  独立运行目录、冻结源码/输入哈希、逐 epoch 原子恢复、运行锁。

## 同协议评估与表格

- 精确复用原公平基线测试缓存：每地区 17,521 起点、72h 公共输入、24h 目标、AEST、原始 RRP。
- 最终树仍训练 2015—2021、选择 2022H1；测试期所有参数冻结，仅因果更新成熟记忆与误差统计。
  联合方法的信息范围大于纯静态基线，表中单列说明，不能声称完全相同历史信息量。
- 使用公共 `aligned_scoring`：排序 q50 MAE、五分位 QS=2×平均 pinball、RMSE/覆盖/宽度。
- 主方法 `Full_joint` 明确表示本轮新版 guarded 输出；另报 `Full_joint_raw`，其 checkpoint 独立按 QS 选择。
- `B_retrained`、`Anchor_joint`、`event_off_same_head`、`time_shift_off_same_head` 均更新至本轮；
  关闭干预不是重训消融。alpha=0 的回退不能解释为神经结构成功。
- 汇总为每 seed 独立评分，三地区等权后计算三个 seed 的均值与样本标准差；不先平均预测。
  三 seed 结果全部完成、来源和样本一致并重新评分后才替换当前入口。
- 17 个静态基线采用 DDNN 修复后的现有 153 格，保留数值与真实来源；
  其中旧静态 `full`、`no_fields` 作为历史结构对照显式标识。
- 此前 2026-10-20 单 seed Full_joint 与 2021/2022 开发结果保留历史档案，不能改写为新版结果。
- 本轮是已看过相关年份后的回顾性评估，不以测试结果调参、改 alpha 或挑选更有利输出作为主方法。

运行：

```bash
.venv/bin/python -m forecasting.full_v5_anchor_balance.aligned_final \
  --output outputs/forecasting/v5_latest_aligned_20261023_r1 \
  --devices cuda:3 cuda:6 cuda:7
```

恢复使用相同参数并添加 `--resume`。
