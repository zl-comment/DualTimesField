# TimeXer / PriceFM 同协议扩展（2026-10-13）

本次新增两个模型、三个地区 NSW1/QLD1/TAS1、三个 seed 2026/2027/2028，共 18 格；与 `aligned_baselines_20261012` 原有 90 格联合后为 12 个模型、108 格。比较的是共同任务下的适配版本，不是原论文数据集上的分数复现，也不是完成超参搜索后的 SOTA 认证。

## 最新比较：联合历史双场 V5，seed 2026（2026-10-20）

本表统一使用原协议 **2023—2024 测试集**，每地区 17,521 个预测起点、未来 24 小时，原始价格空间、排序后 q50 的 MAE，以及共同五分位 QS。原有 12 个基线仅取 seed 2026 重新评分，加入本轮 5 个输出，共 17 行、51 个地区结果。下方原三 seed 档案保持不变，不能将本表与三 seed 均值混排。

参数与检查点在最终测试前冻结。联合方法额外使用已成熟的历史记录和近期误差，测试期间仅因果更新记忆/统计，不更新参数；因此是**相同预测样本与评分、不同历史信息使用方式**的比较。它不是严格相同信息量的基线扩展，也不是盲测或显著性/SOTA 认证。

| 模型 | NSW1 MAE | QLD1 MAE | TAS1 MAE | 宏MAE | 宏QS | 地区平均RMSE |
|---|---:|---:|---:|---:|---:|---:|
| baseline_lightgbm | 44.363324 | 38.197537 | 28.706143 | 37.089001 | 24.196932 | 277.051432 |
| B_retrained | 44.351923 | 38.282503 | 28.705249 | 37.113225 | 24.181184 | 277.090008 |
| time_shift_off_same_head | 45.439161 | 38.882164 | 28.461855 | 37.594393 | 24.413199 | 276.009705 |
| Full_joint | 45.527395 | 38.991026 | 28.449478 | 37.655966 | 24.435378 | 276.220102 |
| event_off_same_head | 45.568393 | 38.985182 | 28.449953 | 37.667843 | 24.444942 | 276.722540 |
| Anchor_joint | 45.686916 | 39.059766 | 28.472317 | 37.739666 | 24.466696 | 276.458418 |
| baseline_xgboost | 45.145121 | 39.445534 | 29.089035 | 37.893230 | 25.616703 | 277.592321 |
| baseline_patchtst | 53.501437 | 46.911823 | 32.445498 | 44.286252 | 27.149797 | 272.618378 |
| baseline_gatets_adapter | 53.019460 | 49.251394 | 34.930911 | 45.733922 | 28.023761 | 273.226007 |
| baseline_dlinear | 56.465809 | 49.041675 | 33.944802 | 46.484096 | 28.163137 | 272.688602 |
| baseline_linear | 56.656372 | 49.122996 | 33.917877 | 46.565748 | 28.249373 | 272.965903 |
| baseline_timexer | 57.137619 | 50.510306 | 33.846947 | 47.164958 | 30.492645 | 279.177599 |
| baseline_mlp | 55.812092 | 52.185477 | 33.993851 | 47.330473 | 28.454989 | 276.583470 |
| baseline_no_fields | 59.243585 | 48.438593 | 35.206416 | 47.629532 | 29.083388 | 269.245212 |
| baseline_full | 58.539772 | 51.062159 | 34.539951 | 48.047294 | 28.745636 | 270.159152 |
| baseline_pricefm | 59.073542 | 58.600027 | 35.631514 | 51.101694 | 30.392849 | 277.450533 |
| baseline_lasso_arx72_cp | 70.847356 | 71.729639 | 34.576354 | 59.051117 | 42.702281 | 288.006437 |

`B_retrained` 是本轮基础树；`baseline_full` 是原版 Full；`Full_joint` 是最新完整联合模型。`Anchor_joint`、`event_off_same_head`、`time_shift_off_same_head` 是同一检查点的关闭干预，不是重新训练消融。PriceFM 仍为原单地区、从零训练的适配版本。

本轮 LightGBM 的宏 MAE 37.089001，Full_joint 为 37.655966（高 1.53%，NSW1/QLD1 变差，TAS1 改善）。历史分支相对 Anchor_joint 降低宏 MAE 0.083700，但尚未抵消联合校正相对基础树的损失，不能宣称完整方法优于 LightGBM。

- [最新 seed 2026 主表](results/main_seed2026.csv) · [51 行分地区表](results/regional_seed2026.csv)
- [完整最终测试报告](../v5_joint_history_20261020_final_seed2026_r1/report.md) · [原始汇总指标](../v5_joint_history_20261020_final_seed2026_r1/summary.json)
- [冻结检查点及源码哈希](../v5_joint_history_20261020_final_seed2026_r1/freeze.json) · [归档来源与哈希](../v5_joint_history_20261020_final_seed2026_r1/archive_provenance.json)

## 原三 seed 基线归档

### 完成状态与归档结果

新增 18/18 格已完成，失败 0 格，最后一组于 2026-10-13 10:33 UTC（北京时间 18:33）结束。联合 108/108 格通过产物校验、父子源码快照校验与跨运行输入/样本/标签对齐校验。

按三个地区、三个 seed 的等权宏平均 MAE 排名，LightGBM 仍第一（37.101，QS 24.191）；TimeXer 第五（46.624，QS 30.014），旧 Full 第十（48.464，QS 28.506），PriceFM 当前单地区适配版第十一（51.162，QS 30.939）。QS 为共同五分位数评分代理，不能把这些固定配置结果解释成各原论文模型的最优能力。

- [12 模型总表](results/main.csv)
- [36 行分地区表](results/regional.csv)
- [108 行逐 seed 表](results/per_seed.csv)
- [完整联合报告](results/combined_report.json.gz) 与 [新增模型报告](results/extension_report.json.gz)
- [归档来源、SHA256 与完整性记录](results/provenance.json)

归档保留原报告数值，不重新训练或改动排名。完整 checkpoint 和逐样本预测仍保存在忽略跟踪的 `outputs/forecasting/aligned_baselines_20261013_extension/`。新 [Full v2 方案](../../../docs/full_v2_architecture_20261013.md) 尚未实现或训练，不计入这 108 格。

## 共同实验边界

- 直接复用父实验经过 SHA256 校验的七块输入缓存、真实标签、预测起点和目标时间戳，不重新生成特征或拟合 scaler。
- 每小时一个 origin，72 小时历史，未来 24 小时；每个地区独立训练，不读取其他地区数据。
- 训练 2015–2021；选择 2022 上半年；校准 2022 下半年；回顾性测试 2023–2024。新增模型不使用测试标签做优化或选模。
- 保持公共 identity + train-zscore 目标空间；五分位数 `[0.05, 0.1, 0.5, 0.9, 0.95]`，共同 pinball 损失。
- 与原神经基线相同：Adam、weight decay 1e-4、batch 256、最多 100 epoch、patience 15；按选择期五分位平均 pinball 选择 checkpoint。
- 固定学习率 TimeXer 1e-4、PriceFM 1e-3；分别取官方 TimeXer parser 和 PriceFM Keras Adam 默认值。没有额外调参预算或测试集选择。
- 预测还原原始 AUD/MWh，统一分位排序后 q50 的 MAE/RMSE、五分位 QS proxy 和 80%/90% 区间指标；QS 不是精确 CRPS。

## 模型适配

**TimeXer**：官方固定版本的目标 patch/global-token、self-attention、exogenous cross-attention 保留。历史价格/负荷和历史日历通过原编码进入模型；未来已知外生变量、日历和上下文通过 horizon-length 投影成为额外的外生变量 token。输出头扩展为 24×5，保留内部归一化和目标反归一化。配置采用官方 NP EPF preset 的 width 512、8 heads、3 layers、FFN 512、patch 24、dropout 0.1；历史长度统一为 72。具体来源和差异见 `forecasting/vendor/timexer/PROVENANCE.md`。

**PriceFM**：锁定官方 2026 MoE 版本，将 TensorFlow 的双分支投影、softmax gate、swish experts、图加权池化及中位数锚点/绝对增量分位头移植至 PyTorch。emb_dim=32、4 experts，所有七块公共输入都有计算路径。按本赛道使用单节点 graph gate=1，从零训练；没有跨地区预训练、邻区输入或外部 checkpoint。这里评价的是单地区适配，不能据此声称评价了 PriceFM 完整跨市场基础模型能力。见 `forecasting/vendor/pricefm/PROVENANCE.md`。

## 运行和产物

配置：`configs/aemo_forecast_aligned_20261013_extension.yaml`。

```bash
.venv/bin/python -m forecasting.aligned_experiment prepare \
  --config configs/aemo_forecast_aligned_20261013_extension.yaml \
  --run-root outputs/forecasting/aligned_baselines_20261013_extension \
  --extend-from outputs/forecasting/aligned_baselines_20261012

.venv/bin/python -m forecasting.aligned_experiment run \
  --run-root outputs/forecasting/aligned_baselines_20261013_extension
```

新目录保存独立 source/config snapshot、jobs.json、controller.log、progress.json。各格 `runs/{model}/seed{seed}/{region}/` 保存 best_model.pt、progress.jsonl、test.npz、metrics.json、status.json。父实验保持不变；data_cache 是通向父缓存的只读用途软链接，每个 worker 重新校验内容。

运行结束自动生成 `combined_report.json`，以及 `combined_report_tables/{main,regional,per_seed}.csv`，完整时分别含 12、36、108 行。每行保留真实来源 source/config hash；不会把父、子实验伪装成同一个代码版本。任意模型缺格或样本、预处理、输入、标签不一致时，不生成联合排名。

可单独更新审核报告：

```bash
.venv/bin/python -m forecasting.aligned_extension \
  --run-root outputs/forecasting/aligned_baselines_20261013_extension \
  --output outputs/forecasting/aligned_baselines_20261013_extension/combined_report.json
```

开发期 GPU smoke 记录在本目录 `gpu_smoke.json`：只使用训练/选择/校准段，检验真实输入形状、finite training 和公共输出接口，不作为质量排名。
