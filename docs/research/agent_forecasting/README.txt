Agent 电价预测研究资料库
建立日期：2026-10-28 UTC

用途
持续查阅论文、已核实的开源框架和本项目的设计边界。本文档不是实验成功声明。

最新结果（2026-10-31，Asia/Shanghai）
r12_totaldemand_information_protocol_20261031.txt：用户确认代理特征后的2023三地区S1 A/B冻结协议。
分支research/agent-totaldemand-information-20261031-r12，基点9b4bc64；执行提交f5a9541138bcc6f69269c552635ae9f87158b1fe。
2023三区各365日完整运行，A/B各1095单元、seed2026，无缺失/失败/回退；34项关键测试及独立实际结果复算通过。
宏QS改善4.6008%（相对变化95%CI −6.5517%..−3.2941%），宏MAE改善4.3208%。
宏/QLD1/TAS1的C80偏差恶化超过1个百分点；26/29条件通过，组合判定NO_USEFUL_GAIN，按冻结规则停止S2/S3。
这表示损失改善但覆盖保护未通过，不表示完全没有预测收益；没有执行2024或三seed Agent实验，也没有推送。
完整结果、实际资源与独立审查：../../../logs/forecasting/agent_totaldemand_information_r12_20261031/results_cn.txt。

historical_replay_protocol_20261031_r11.txt：用户确认历史回溯后提出的R11方案，2023信息筛选、2024 Agent历史评价。
分支research/agent-historical-replay-20261031-r11以a568f7a为基点；原R10结论不改。
已实现72/25/24小时历史输入、原生分位评分和S1配对评价；24项关键合成测试通过，记录在logs/forecasting/r11_*tests*。
2023三地区各365/365、2024各366/366，结构覆盖均100%；来源审计执行提交b3680d0。
详细结果：../../../logs/forecasting/agent_historical_source_r11_20261031/results_cn.txt。
R11当时记录：新小时需求代理编码已列明近似与局限，尚待用户确认；当时未运行模型、真实评分或三seed实验，未推送。
后续确认与实际模型结果见上方R12，旧协议和审计身份不回写。

R10历史记录
information_first_plan_20261031_r10.txt：r9 后的修改计划，先核实未来 TOTALDEMAND 的信息增益，
再检验 Agent 的独立贡献；四个设计差距、A/B/C/D、时点、停止规则、预算与实现顺序均在其中。
information_first_review_20261031_r10.txt：三路独立审查意见、官方来源和未解决的证据边界。
计划分支 research/agent-information-first-plan-20261031-r10，基点 9a1cd340c1c02fda5dba6b11bc270e69355390d2。
实施进度：遵从用户最新指令，已在此分支完成S0最小读取/时间覆盖审计实现，先提交协议/测试后执行。
r10_s0_source_audit_protocol_20261031.txt为实施协议，执行提交d91d60c02533d93bc120e12ae7cc9c807f087f30。
结果见../../../logs/forecasting/agent_totaldemand_s0_20261031/results_cn.txt及review.txt。
7项相关测试通过；三地区各334/365=91.51%结构覆盖，需求定义仍未证实一致，S0未放行。
S1/S2/S3和三地区三seed预测实验尚未执行；没有推送。本段之后是原资料库历史记录。

入口
1. papers/RE-Price/RE-Price.pdf：用户提供的13页原论文，原样保存。
2. papers/RE-Price/RE-Price.fulltext.txt：pdftotext -layout 提取全文，可全文搜索；图表以PDF为准。
3. papers/RE-Price/review_20261028.txt：全文方法审查，按PDF页码定位。
4. library.json：文献来源、大小、SHA-256和阅读状态。
5. frameworks_20261028.json：已审查GitHub项目、固定版本、许可、适用性与证据链接。
6. proposal_20261028.txt：严格免任务训练方案、已有工作、最小比较和实施前置条件。
7. next_step_plan_20261028.txt：多agent研究后的分阶段计划；下一项仅G0证据通路与CPU工具验收。
8. g0_protocol_20261028.txt：已提交冻结的G0协议；执行源码提交9740509d8a28b7b05ce3ce59286afa49d9180e85。
9. ../../../logs/forecasting/agent_evidence_contract_g0_20261028/README.txt：G0结果、实际资源、协议差异和停止判定。

资料/计划分支：research/agent-zeroshot-feasibility-20261028，已推送至origin，提交a090efe05766fcfb7ebad8d7e04305ffcbe74d3c。
独立G0分支：research/agent-evidence-contract-g0-20261028，代码与结果本地提交。
资料阶段只有文件归档、论文阅读、公开资料与代码审查。后续G0执行了1次CPU预测工具验收及有上限的官方公告采集；没有进行准确性实验。
39项合成测试通过；CPU工具通过；6条选中公告链的合法数值输入通路为0，整体G0未通过，G1/G2未运行。
本次没有模型训练、权重下载、新包安装、实验外部LLM调用或部署。源文审阅由研究agent完成，不能替代协议文字中的人工审核或真人金标准。
GitHub上的源代码经连接器只读核查，未安装或运行。PDF及提取全文保存在本地并由所在目录的.gitignore排除，防止后续无意公开；审查摘要与索引可单独版本化。

用户要求
新批量实验必须先新建独立研究分支；实验协议、配置与必要测试先冻结，再运行。
最新研究目标为无需电价任务训练、在线利用最新信息。此前讨论的输出头微调不属于这个严格方案。
阶段关系见next_step_plan_20261028.txt；本次G0以g0_protocol_20261028.txt为冻结依据，实际结果见上述归档入口，不自动放行后续阶段。
“首次”和“准确”均需要独立文献与实证支持，不能作为已实现属性。
