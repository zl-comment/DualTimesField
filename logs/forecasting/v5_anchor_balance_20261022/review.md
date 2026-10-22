# 实现复核

2026-10-22，Codex 分工完成目标函数/选择逻辑、实验执行器和独立只读复核。

验证命令：

```bash
.venv/bin/python -m pytest -q \
  tests/test_full_v5_anchor_balance.py \
  tests/test_full_v5_anchor_balance_experiment.py \
  tests/test_full_v5_joint.py \
  tests/test_full_v5_joint_final.py
```

结果：43 passed。未修改旧 `full_v5_joint` 源码和既有检查点。

覆盖的主要行为：零附加权重恢复原损失及梯度；新增监督能够传入校正网络；
identity 在优化与保存/恢复后保持恒等；未来标签不改变当前预测输入；
时间边界 purge、全月/全小时采样覆盖、拒绝2023年数据；共用 alpha 的
四项守门与明确回退；神经/树种子分离；OOF映射、季节和预测价带诊断。

独立复核未发现需要中止 screen 的标签泄漏、评估期选择或恢复错误。
解释时必须保留以下限制：

- 2022 的基础树沿原 OOF 日程切换；神经检查点和 alpha 在评估前固定。
- balanced 同时改变监督与正则，不能单凭结果区分各新增项的因果贡献。
- raw 和 guarded 可能选择不同 epoch，二者差值并非仅为固定模型的收缩效果。
- source 是审计副本，实际从工作区执行，结束训练时核验源码哈希；不是隔离执行环境。
- 两折是开发回测，2021 同时用于第一折评估与第二折选择，不是独立盲测。
- 筛选裁决由 `summarize_run.py` 按评估前文档执行，不按最好单行结果裁决。
