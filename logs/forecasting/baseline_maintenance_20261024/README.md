# 核心基线维护（2026-10-24）

本轮协议：[限定范围与预定开发门槛](../../../docs/baseline_maintenance_20261024.md)。最终主表仍由 [统一入口](../main_comparison_20261023/README.md)读取；诊断样本不是正式测试结果。

## 数据可用性结论

对 NSW1/QLD1/TAS1 的九个外生 NPZ 完整检查所有已保存 source_*_unix，不存在来源时间晚于 origin 的记录。三个原始小时 CSV 的 available_at 均被定义为 delivery_start+1h。气价为各 gas day 的实际计划价格算术均值，使用前七个已结束 gas day；不是结算加权日均价，不能套用 INT041/INT042 的发布规则。

这只能证明现有归档时间代理内部一致，**未完成真实公开发布时间与修订版本审计**。官方 MMS 数据模型将 LASTCHANGED 定义为记录修改时间，不等于公众可获取时刻；目前没有覆盖全部2015–2024记录的首次公开发布时间台账。不能据此声称无泄露，也不能凭任意1h/1day延迟伪造修复。维持统一历史缓存和回顾性比较的范围；严格实时/无污染结论暂不可验收。当前没有发现足以证明必须更换所有输入数值的证据，因此不盲目重跑19个基线。

证据：[完整时间戳核查](data_availability.json)。官方资料：
- [AEMO MMS 数据与模型入口](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/market-management-system-mms-data)
- [MMS 数据模型字段定义](https://di-help.docs.public.aemo.com.au/Content/Data_Model/MMS_Data_Model_Report_53.pdf)
- [DWGM 各计划时点价格资料](https://www.aemo.com.au/energy-systems/gas/declared-wholesale-gas-market-dwgm/data-dwgm/vic-wholesale-price-withdrawals)

## 核心对照冻结

LightGBM、XGBoost、PatchTST 三模型×三地区×三seed，共27个原始产物 SHA 已重新核验，记录于 [core_freeze.json](core_freeze.json)。原2022H1选择规则/固定配置保留，不追加测试驱动调参，不把神经轮数和树迭代数强行视为相同成本。新增方法以这些冻结预测作核心参照；尚不能宣称已执行公平HPO。

## 适配诊断

TimesFM 的固定128起点抽样中，ridge1 在 NSW1/QLD1/TAS1 均改善 MAE；完整2022H1确认进行中。只有开发门槛通过、配置冻结后才运行新测试并替换来源。裸模型仅用于诊断，不因抽样更好直接入榜。

TabPFN 已修复任意子集调用改变生产查询形状的问题：按同一视图的4096行规范块推理，再取回请求的行，保持顺序、重复索引和最后不足4096行的块；重新 fit 清空缓存。完整有序生产调用与旧路径一致，因此保留原九格结果，不伪称重跑。

真实权重开发验证使用 NSW1、seed2026、第一预测步、完整61273行训练上下文：4096行独立重放最大差0.0，规范块取子集差0.0，直接8行查询差0.03486684。证据见 [tabpfn_development.json](tabpfn_development.json)。这只是一个地区/一个预测步的开发证据，不能称24步×9格全量重放；原测试批次抽样不通过的记录继续保留。5项调用契约测试验证顺序、重复索引、尾块与非法索引。

旧基础模型原始预测、原审计和 SHA 不改写；新适配器身份包含 query_batch_policy，不能冒充旧实验生成源码。


TimesFM 设计诊断：[timesfm_design.json](timesfm_design.json)。逐origin回归使静态气价中心化为0；128个固定开发起点的12列矩阵秩为5–6。ridge1将最大外推算子行范数从2.60降至1.56。该诊断没有使用标签数值，使用float64数学算子，不等价于官方JAX float32的逐位复现，也不足以单独解释全部误差；正式修复是否成立仍由预定完整开发门槛决定。

正式 TimesFM 适配器与开发诊断在 NSW1 固定前8个开发起点、两种预定配置上逐位一致，见 [timesfm_adapter_parity.json](timesfm_adapter_parity.json)。它验证代码路径一致，不代替完整开发/测试评估。
