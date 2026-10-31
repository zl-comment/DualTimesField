Agent 电价预测研究资料库
建立日期：2026-10-28 UTC

用途
持续查阅论文、已核实的开源框架和本项目的设计边界。本文档不是实验成功声明。

最新计划（2026-10-31，Asia/Shanghai）
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
