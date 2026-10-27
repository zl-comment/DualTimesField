from pathlib import Path
import csv,json,hashlib
R=Path('/root/workspace/Code/DualTimesField');O=Path('/root/workspace/Code/research_progress_report_20261026');C=R/'logs/forecasting/main_comparison_20261023'
def rows(p):
 d=list(csv.DictReader(open(p)))
 for x in d:
  for k,v in x.items():
   if v=='': x[k]=None
   else:
    try:x[k]=float(v)
    except:pass
 return d
main=rows(C/'main.csv');regional=rows(C/'regional.csv');m={x['model']:x for x in main};r={(x['model'],x['region']):x for x in regional}
compare=[]
for name in m:
 d={'reference':name,'method':'Full_joint','sign_convention':'负数=Full_joint误差下降，正数=退化'}
 for k in ['mae_mean','quantile_score_proxy_mean','rmse_pooled_mean']:
  if m[name][k] is not None:d[k+'_relative_change_pct']=100*(m['Full_joint'][k]-m[name][k])/m[name][k]
 compare.append(d)
rc=[]
for region in ['NSW1','QLD1','TAS1']:
 d={'region':region}
 for k in ['mae_mean','quantile_score_proxy_mean','rmse_pooled_mean']:
  d[k+'_relative_change_pct']=100*(r['Full_joint',region][k]-r['baseline_lightgbm',region][k])/r['baseline_lightgbm',region][k]
 rc.append(d)
result={
 'report_date':'2026-10-26','research_status':'实现和回顾性同协议比较已完成，较强基线的稳健总体提升尚未达成，下一步优先解释并约束条件校正退化。',
 'primary_source':str(C),
 'main_table':main,'regional_table':regional,'historical_controls':rows(C/'historical_controls.csv'),
 'comparison_vs_all':compare,'regional_change_vs_lightgbm':rc,
 'protocol':{'test_period':'2023—2024','regions':['NSW1','QLD1','TAS1'],'neural_seeds':[2026,2027,2028],'main_outputs':25,'baseline_outputs':19,'joint_outputs':6,'region_seed_cells':225,'history_outputs':2,'history_cells':18,'origins_per_region':17521,'horizon':24,'historical_input_hours':72,'cells_per_region_per_output':420504,'timezone':'AEST','price':'原始 RRP，AUD/MWh','tree_training':'2015—2021','neural_training':'2018-01-01 至2022-01-01，chronological OOF，严格成熟度/purging','selection':'2022H1','2022H2':'联合模型仅作为成熟历史记忆；不参与checkpoint/alpha选择','reporting':'先对每地区每seed的origin×horizon逐单元评分；每seed三地区等权平均，再报告三个seed均值±样本SD。RMSE是各地区pooled RMSE算术均值，非跨地区合并平方再开根。','point':'Q表使用排序后q50；P表使用native_point','probability':'QS=2×五分位平均pinball，分位数[.05,.10,.50,.90,.95]；P表无此QS','independence':'重叠窗口不是独立事件。联合模型树固定2026，仅神经初始化和训练顺序变化；B_retrained为同一冻结树重复；Moirai/TimesFM/Chronos为确定性seed重复。','scope':'相同目标与评分，并非完全相同信息范围：联合方法有成熟历史/误差；除TabPFN外时序基础模型使用受限输入。'},
 'interpretation':[
  'LightGBM在19个外部基线中宏MAE和QS最佳。B_retrained QS略低，是同一方法管线的冻结树参考，不能作为另一个独立随机基线。',
  'Full_joint MAE=37.7828±0.4389，QS=24.3922±0.1875；对LightGBM MAE退化1.8366%、QS退化0.8302%，地区平均RMSE改善0.6017%。不能宣称整体超过最强树基线。',
  '对XGBoost MAE改善0.3886%、QS改善4.8003%；对PatchTST MAE改善15.5950%、QS改善11.2476%。这说明相对部分神经基线有优势，但不证明事件模块贡献。',
  '区域差异突出：NSW MAE改善0.6706%、QLD退化6.6281%、TAS改善0.6652%；QLD QS同时退化4.1618%。NSW与TAS的小幅收益无法抵消QLD退化。',
  'Full_joint相对Anchor_joint MAE改善0.3297%、QS改善0.3061%；关闭event后MAE反而略低、QS略高，关闭time_shift后三项误差均略低。当前同检查点干预不能支持事件/时间位移带来一致增益；它们不是重训消融。',
  '覆盖率：Full_joint中央80%/90%区间实际覆盖68.56%/82.70%，低于名义值11.44/7.30个百分点，且较LightGBM的72.50%/85.10%进一步下降。区间平均宽度从84.08/141.35缩为73.23/118.08，不能仅把区间变窄当改进。',
  'LightGBM-CP保持中位数MAE，但覆盖96.44%/98.67%，宽度188.63/278.67，QS恶化至30.9729，表现为过度保守而非全面校准改善。LightGBM-QRA RMSE269.4659为当前25行最低，但MAE和QS比原树差，呈现损失权衡。',
  'DDNN-JSU宏QS140.86±90.76、90%区间宽度4034.65±3598.32，显示当前实现/配置存在明显不稳定；不能据此否定该论文在原协议的结论。',
  '基础模型预训练重叠未审计、输入范围不齐且未执行统一HPO；这些数值是当前适配和预算下的表现，不是架构普遍排名。'],
 'development_evidence':{
  'calibration_robustness':{'source':str(R/'logs/forecasting/v5_calibration_robustness_20261023/report.md'),'physical_fits':36,'logical_arms':54,'epochs':20,'years':[2021,2022],'finding':'reference、multiwindow、bounded_calibration均未通过两年×三个seed全部稳健门槛。仅bounded guarded的seed均值全过，但逐seed失败，故不得称通过。','bounded_guarded_2022':{'mae':68.8435,'mae_sd':1.3340,'qs':42.0012,'qs_sd':.3934},'caution':'与2023—2024主表分开；multiwindow与reference共用训练轨迹，raw不是独立重复训练。'},
  'mechanism_state':{'source':str(R/'logs/forecasting/v5_mechanism_state_20261023/README.md'),'finding':'原bounded输出中弱位置证据组伤害更高：raw仅4/6地区年份，guarded仅1/6，假设未获一致支持。状态分组/原树误差混杂，非因果证据。','status':'67项实现测试通过；3seed×3臂×2epoch共9次短流程仅验证工程；完整54次20轮开发确认尚未启动，无科学验收资格。'},
  'event_diagnosis_2022':{'source':str(R/'logs/forecasting/lightgbm_event_diagnosis_20261013/report.md'),'finding':'2022开发期更支持预测时可识别的高价条件下水平与幅度低估，而非整体时间错位/碎片化。高价状态和极端短尖峰需区分。','H2_oracle_time_shift_gain_pct':2.73,'H2_oracle_level_gain_pct':38.0,'oracle_caution':'均使用未来标签，只是诊断上限，不能写为可部署提升或可学习性证明。','H2_high_predispatch_actual_gt_q50_pct':[95.05,95.25,90.39],'H2_high_predispatch_actual_gt_q95_pct':[6.34,2.18,32.38],'region_order':['NSW1','QLD1','TAS1']},
  'old_spike_ranking':'20261020档案为旧Full_joint单seed，不能用来声称最新版本尖峰结果；本报告另复算最新主表预测。'},
 'audit_cautions':['publication.json仍保留21输出189格的旧计数，与当前CSV和verification.json不同；以当前25行225格与文件hash为准。','真实公开发布时间及后续修订审计未完成；已有来源时间代理无晚于起点，不等于严格实时无泄露。','本轮训练20epochs，raw/guarded各2/9选择位于末两轮，不能宣称充分收敛。','回顾性测试，无显著性认证；均值±SD不能当作时间泛化置信区间。'],
 'next_steps':[
  {'priority':1,'action':'将研究目标分为常态误差保护、高价条件响应、极端事件风险三项；采用预测时可观测条件做分组，区分实际值筛选的事后诊断与可部署触发条件。','success':'总体MAE/QS不劣于强树，同时条件偏差/覆盖有稳定改善。'},
  {'priority':2,'action':'先加入同等成熟历史输入的简单状态校准、线性/MLP残差头以及固定收缩控制，必要时分地区检验QLD失效。','success':'复杂双场应超过这些信息范围匹配的对照，且事件/位移需要重训消融支持。'},
  {'priority':3,'action':'用训练期固定状态检验残差信号跨窗口稳定性、样本支持与校正伤害；机理约束先验证假设再启动完整预注册确认。','success':'保留反向结果；与相同平均门约束比较，不把总体缩小校正当状态机制。'},
  {'priority':4,'action':'同时报告实际>300/>1000切片MAE、bias与事件发生/峰值/持续时间；按唯一交付小时和固定提前量统计事件，报告删失/稀少样本数。','success':'事件概率需要单独概率头及Brier/PR-AUC/校准，不能把q95>阈值当已估计事件概率。'},
  {'priority':5,'action':'检验条件分位覆盖、WIS或公共五分位QS和区间宽度；在滚动开发窗口内冻结配置，新增未查看时段作为未来外部确认。','success':'用按时间块或事件块的配对bootstrap处理重叠窗口，报告差值区间；不把三个seed当时间独立重复。'},
  {'priority':6,'action':'补齐真实发布时间/修订审计、基础模型预训练重叠边界和等预算调参；扩展整套树+神经seed重训。','success':'明确公平性适用范围及计算预算，避免局部适配结果扩展为普遍SOTA结论。'}],
}
sp=O/'latest_spike_diagnostics_macro.csv'
if sp.exists():
 result['new_spike_diagnosis']={'status':'本报告对当前全部25输出×9格冻结预测轻量复算，非新训练；同版本/同测试协议。','protocol':'以实际值>300、>1000、负价、0到300分组的逐单元诊断；极端组为高价组子集，不互斥。另保留按24h窗口max/mean划分的旧式切片，不能与实际高价单元MAE混淆。','caution':'按真实高价筛选的负bias不自动证明条件中位数失准；只能描述已发生高价时的误差，需结合预测时可识别状态的覆盖和事件概率检验。','inventory':rows(O/'latest_spike_inventory.csv'),'macro':rows(sp),'regional':rows(O/'latest_spike_diagnostics_regional.csv'),'provenance':str(O/'latest_spike_diagnostics_provenance.json'),'script':str(O/'audit_results.py')}
json.dump(result,open(O/'results_audit.json','w'),ensure_ascii=False,indent=2)
print('written',O/'results_audit.json')
