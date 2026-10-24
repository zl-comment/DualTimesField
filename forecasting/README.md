# 模型与公共实验流程

各模型实现、来源说明分别位于以下独立目录；配置位于对应的 `configs/<目录>/aligned.yaml`。

当前 19 个基线（15 个静态基线及 4 个基础模型）的[论文链接与引用总索引](CITATIONS.md)提供代码入口及各目录可复制的 `CITATIONS.bib`；引用用途和本地适配范围见对应 `source.json`。

当前结果见[主比较入口](../logs/forecasting/main_comparison_20261023/README.md)：新版 Full_joint 已完成三神经种子同协议评估，当前主表为 25 个输出、225 格，包含 15 个静态基线、4 个基础模型和联合方法的 6 个输出。旧静态 Full/no_fields 的 18 格移入历史附表。原 17 模型、153 格静态基线档案保留不变。

当前 Full_joint 使用[平衡损失与非对称 Anchor 尺度头](full_v5_anchor_balance/README.md)，通过 `forecasting.full_v5_anchor_balance.aligned_final` 训练/评估；下方 `full/` 为历史静态结构，不是当前完整方法。

后续[校正稳健性实验](full_v5_calibration_robustness/README.md)已完成两项独立改动、36 次训练及全部复核，[结果与验收](../logs/forecasting/v5_calibration_robustness_20261023/README.md)显示限幅均值改善但尚未稳定通过。原开发折和三神经种子预算保持不变，当前 Full_joint 主表保留已发布结果。

[机理状态约束](full_v5_mechanism_state/README.md)独立实现训练期证据门与固定平均门对照，重点检验校正可靠性，不把短流程指标当作有效性结果。

| 目录 | 实验模型 ID | 实现类型 |
| --- | --- | --- |
| [full/](full/README.md) | `full` | 本地实现，历史结构对照 |
| [no_fields/](no_fields/README.md) | `no_fields` | 本地实现，历史结构对照 |
| [mlp/](mlp/README.md) | `mlp` | 本地实现 |
| [linear/](linear/README.md) | `linear` | 本地实现 |
| [dlinear/](dlinear/README.md) | `dlinear` | 本地实现（参考模型） |
| [patchtst/](patchtst/README.md) | `patchtst` | 本地实现（参考模型） |
| [gatets/](gatets/README.md) | `gatets_adapter` | 官方核心源码 + 本地适配 |
| [timexer/](timexer/README.md) | `timexer` | 官方核心源码 + 本地适配 |
| [pricefm/](pricefm/README.md) | `pricefm` | 本地移植（官方 TensorFlow → PyTorch） |
| [lightgbm/](lightgbm/README.md) | `lightgbm` | 官方库调用 + 本地训练适配 |
| [xgboost/](xgboost/README.md) | `xgboost` | 官方库调用 + 本地训练适配 |
| [lasso/](lasso/README.md) | `lasso_arx72_cp` | 官方库调用 + 本地模型/CP 适配 |
| [lightgbm_cp/](lightgbm_cp/README.md) | `lightgbm_cp` | 冻结基座复用 + 本地 CP 适配 |
| [lightgbm_qra/](lightgbm_qra/README.md) | `lightgbm_qra` | 冻结基座复用 + 本地 QRA 适配 |
| [xgboost_cp/](xgboost_cp/README.md) | `xgboost_cp` | 冻结基座复用 + 本地 CP 适配 |
| [ddnn_jsu/](ddnn_jsu/README.md) | `ddnn_jsu` | 方法重实现，float64 未截断 JSU 逆 CDF |
| [nhits_qra/](nhits_qra/README.md) | `nhits_qra` | 官方源码移植 + 本地训练、MC-dropout/QRA 适配 |
| [tabpfn_v3/](tabpfn_v3/README.md) | `tabpfn_v3` | 官方冻结权重 + 本地输入适配；Q/公共输入 |
| [chronos2_zeroshot/](chronos2_zeroshot/README.md) | `chronos2_zeroshot` | 官方冻结权重 + 本地输入适配；Q/受限输入 |
| [moirai2_zeroshot/](moirai2_zeroshot/README.md) | `moirai2_zeroshot` | 官方冻结权重 + 本地输入适配；P/受限输入 |
| [timesfm25_xreg/](timesfm25_xreg/README.md) | `timesfm25_xreg` | 官方冻结权重 + 本地输入适配；P/受限输入、逐起点 XReg |

`itransformer/` 和 `informer/` 保留历史本地实现，不在当前完成的基线表中。TabPFN-3、Chronos-2、Moirai-2、TimesFM-2.5 + XReg 已完成 36 格评估并纳入主表；输出能力、输入限制与预训练边界单独标注。

`common/` 保存共享神经网络 adapter 与单 split 缓存读取。`aligned_inputs.py` 负责公共样本与预处理，`aligned_runner.py` 负责神经训练，`aligned_baselines.py` 负责树/Lasso 的协议校验与调度，`aligned_scoring.py` 负责共同评分，`aligned_experiment.py` 负责冻结与任务执行。`general_baselines.py` 只保留模型工厂与兼容导出。

五个新增基线通过 `aligned_expansion.py` 调度，复用公共缓存与评分；`common/` 内的 registry、frozen_base、calibration、native_trainer 分别提供注册、冻结基座、校准及原生训练支持。`aligned_expansion_report.py` 合并来源并显式替换修复结果，旧 DDNN 的 9 格只保留审计，不重复计入当前 153 格。

原 `models.py`、`losses.py`、`*_baseline.py` 和少量旧 `vendor` 模块保留轻量兼容导出，支持旧导入/对象加载；模型实现不在这些文件中重复存放。官方原始快照位于各模型自己的 `vendor/`，不能将本地重写或移植标成未经修改的官方复现。

Full V2 的活动实现、配置、专属测试和设计文档已删除，暂停进程已终止。被废弃运行的记录仅保留在原 outputs 历史目录，不能作为可恢复的活动任务。已完成的 108 格基线结果及冻结源码不做修改。
