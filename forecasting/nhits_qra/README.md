# nhits_qra

实现类型：**官方源码移植 + 本地训练适配**（vendor，Apache-2.0，来源与逐条修改见 [vendor/PROVENANCE.md](vendor/PROVENANCE.md)）。

NHITS 多尺度骨干（MaxPool 层级池化 + 谐波插值基 + 残差回投 + naive1 水平初始化，均保留上游结构）以 MAE 点头训练；冻结后做固定种子 MC-dropout（K=20，仅 dropout 层进入 train 模式，不更新任何训练状态），MC 样本经验分位作为逐 horizon 特征，在 2022H2 标签上拟合 QRA（alpha=0 精确 LP），测试期系数冻结。

- 输入块映射：价格历史→insample_y；需求+历史日历→hist_exog；未来日历+三组未来外生→futr_exog（预测期长度，PROVENANCE 修改1）；origin_context→stat_exog。七块全部接入，无丢弃。
- 选择指标：选择期原始价格 MAE（eval 点预测，配置预声明）；MC/QRA 阶段严格在选择之后。
- 成本：MC 采样开销计入任务时间；固定样本数与种子规则写入 metadata。

配置：[configs/nhits_qra/aligned.yaml](../../configs/nhits_qra/aligned.yaml)。实验模型 ID：`nhits_qra`。来源与差异：[source.json](source.json)。
