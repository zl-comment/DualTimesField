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

TimesFM 与 TabPFN 开发诊断进行中。旧正式结果保留，只有完整开发确认并冻结修复配置后才运行新测试、替换来源。旧基础模型生产结果和预测哈希不改写。
