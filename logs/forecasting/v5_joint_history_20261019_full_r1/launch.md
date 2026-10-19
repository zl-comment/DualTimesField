# 三地区完整训练启动记录

分支：research/v5-joint-history-20261019。Seed：2026。

完整协议：21个LightGBM时间前推训练折（3地区×7折），随后各地区一个联合模型训练。联合训练2018—2021，选择2022H1，预测评估2022H2；2017提供初始历史记忆。没有小样本截取，联合模型训练20 epoch；整体检查点按选择期QS选取。2022H2属于已查看的开发数据，不能称为盲测。

最多同时运行6个树训练任务；各地区7折均完成后自动进入联合训练、五组同检查点干预预测和评分。任何历史折失败会阻止该地区下游训练，其他地区继续。联合训练每100个batch更新进度，并在选择指标改善时保存best.pt。

输出目录：/root/workspace/Code/DualTimesField/outputs/forecasting/v5_joint_history_20261019_full_r1

- launch.json：配置、代码快照哈希、13个输入文件哈希、冻结检查。
- source/：本次执行的冻结代码和配置，后续工作区编辑不会影响已启动任务。
- status.json：最新任务状态、PID、日志位置、退出码。
- oof/地区/折/progress.json：树训练horizon与轮数。
- joint/地区/progress.json：联合训练epoch与已处理origin数。
- joint/地区/：检查点、逐样本预测、指标。
- summary.json、report.md：任务结束时自动生成三地区总表。

本次启动前15项测试通过，144个旧冻结文件未改变。启动时补齐了代码快照遗漏的src依赖；首次失败发生在派发任务前，日志保留于supervisor_initial_import_failure.log，随后成功重新启动。

记录写入时总状态：running。精确实时进度请读取输出目录下status.json及各任务progress.json。
