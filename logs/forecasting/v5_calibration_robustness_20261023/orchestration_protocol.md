# 训练完成后的计数核验修复

原 r2 已完成全部 36 次物理训练和 20 epochs；18 次 reference 的逐轮损失、原选择和
模型参数全部精确复现。原完成计数将 fit_id 当成全局唯一标识，但不同地区的训练/选择
行索引相同，旧 fit_id 又没有地区字段，因此 36 个真实独立目录只得到 12 个不同字符串。
原屏障拒绝继续，未产生全局 selection_complete、未开始评估。

本次仅修复流程核验：以带地区/年份/seed/轨迹的 raw checkpoint 路径识别物理训练，
要求 36 条独立路径和 54 个完整逻辑方案，并验证路径确实匹配对应地区、年份和 seed。
reference 与 multiwindow 必须共用同一路径；全部选择记录和 checkpoint 哈希仍需验证。

保留原 experiment.py 和 manifest/source 快照的全部字节，避免改写冻结训练的来源。
新增 runner.py 作为正式入口，它复用原训练、数据、模型、选择和评估函数；对现有 r2
直接读取全部已完成的选择记录，不重新拟合、不重新选择、不修改任何权重。
新增 orchestration.json 在评估前冻结流程代码、原 manifest、全部现有权重/选择文件的哈希。
原失败信息保留为 failure_before_orchestration_fix.json。训练源提交与流程修复提交分别记录。
季度规则、限幅强度、预算、数据、三个神经 seed、验收标准均维持原预注册协议。

继续现有完整训练：

```bash
.venv/bin/python -m forecasting.full_v5_calibration_robustness.runner \
  --development --devices cuda:3 cuda:6 cuda:7 \
  --output outputs/forecasting/v5_calibration_robustness_20261023_r2 --resume
```

未来新运行也使用 runner 入口并指定新的目录，省略 --resume。
原 experiment 入口留作冻结训练来源，不再作为推荐执行入口。
