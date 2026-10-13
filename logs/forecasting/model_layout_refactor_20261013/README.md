# 模型目录重构验证（2026-10-13）

12 个已完成基线的实际实现已按模型分目录，原始 Full 保留，Full V2 的活动源码、配置、设计文档、专属测试已移除。7 个暂停的 V2 进程已核对命令与进程组后终止。未启动正式实验。

`prediction_parity.json` 记录重构前后同种子、同合成输入的 12 模型预测比较与 9 个神经模型短训练权重比较；结果完全一致。树模型另外覆盖实际产生非恒定预测的逐 horizon / shared 两种选择方式。TimeXer 的短训练检查使用较小宽度，不能视为完整实验重跑。比较训练元数据时只排除了 checkpoint 保存位置。

`refactor_parity.py` 与 `refactor_tree_parity.py` 是验证脚本，需分别从重构前、后的代码根目录运行，并传入独立输出目录。`source_hashes.json` 记录本次参考实现与迁移后源文件哈希。

`validation.json` 记录全套测试：232 passed、1 skipped，另有 3 个子测试通过。跳过的是可选官方 TensorFlow PriceFM 等价测试；本次不改变其尚未验证的状态。

371 份旧配置及 9 份 vendor Python 源文件按原字节迁移；两个已完成基线的冻结源码与配置哈希校验通过。原有结果未修改。被废弃的 V2 运行与设计记录只保留于 `outputs/forecasting/full_v2_20261013/` 历史目录，状态为 `retired_by_user`。
