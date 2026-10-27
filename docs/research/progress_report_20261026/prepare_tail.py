import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).parent
d=pd.read_csv(O/'latest_spike_diagnostics_macro.csv').set_index(['model','band'])
models=['baseline_lightgbm','baseline_xgboost','baseline_lightgbm_qra','baseline_patchtst','baseline_dlinear','B_retrained','Anchor_joint','Full_joint','event_off_same_head','time_shift_off_same_head']
labels=['LightGBM','XGBoost','LightGBM+QRA','PatchTST','DLinear','B_retrained','Anchor_joint','Full_joint','关闭事件项','关闭时间移动']
bands=['negative','normal_0_to_300','high_gt300','extreme_gt1000']
rows=[]
for m,l in zip(models,labels):rows.append([l]+[f'{d.loc[(m,b),"mae_mean"]:.3f}' for b in bands])
qrows=[]
for m,l in zip(['baseline_lightgbm','baseline_lightgbm_qra','baseline_patchtst','Full_joint'],['LightGBM','LightGBM+QRA','PatchTST','Full_joint']):qrows.append([l]+[f'{d.loc[(m,b),"quantile_score_mean"]:.3f}' for b in bands])
payload={'paragraphs':[
'本报告在2026年10月26日对当前25个输出×9格保存预测进行了额外复算。225份预测的SHA256与登记一致，真实标签、预测起点和交付时间完全对齐，逐格全样本MAE重现误差小于10⁻⁸。本节不新增训练，也不修改主表选择规则。按各地区分别评分、三地区等权、再对三个seed求均值；完整25行分层结果见附录C。',
'事后分组采用负价y<0、常态0≤y≤300、高价y>300，并在高价内另列极端y>1000；最后两组有包含关系，不应相加。这里筛选的是起点—步长单元中的实际交付价格，不是“24小时窗口里出现过一次尖峰”的整窗指标。阈值仅用于报告诊断，不能视为预测时已知状态。',
'Full_joint相对LightGBM：高价MAE由988.399降至967.603（改善2.10%），极端MAE由3472.494降至3444.268（改善0.81%）；负价由30.618降至28.540（改善6.79%），常态由22.822升至23.948（退化4.93%）。这说明最新方法确实存在已发生尖峰时的局部点误差改善，但高频常态样本的代价足以抵消它。负价在跨地区平均改善，不代表QLD1负价也改善。',
'Full_joint并非尖峰切片的最佳方法：LightGBM+QRA在y>300的MAE更低，DLinear在y>1000的MAE更低；PatchTST两个尖峰切片均优于Full_joint，但总体MAE更差。关闭时间移动后的尖峰MAE也略低于完整方法，因此不能将当前尖峰收益直接归功于时间移动模块。',
'三地区中，实际>1000的唯一交付小时分别只有85、81、18个；它们不等于独立事件数，连续尖峰需进一步合并。重叠滚动窗口将同一小时多次计入不同提前量，不能把相应2040、1944、432个评分单元当成独立尖峰。按真实高价筛选后的偏差、覆盖率也不自动证明条件中位数或预测时可识别状态失准，后续应增加独立事件定义及时间块置信区间。'],
 'tables':[{'title':'表3　当前冻结预测的分层MAE，均值，AUD/MWh（完整25行见附录C）','headers':['模型/输出','负价 <0','常态 0–300','高价 >300','极端 >1000'],'rows':rows},{'title':'表4　同一切片的五分位QS，均值，AUD/MWh；分组均为事后分析','headers':['模型/输出','负价 <0','常态 0–300','高价 >300','极端 >1000'],'rows':qrows}]}
(O/'tail_report_payload.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
vals=[100*(d.loc[('Full_joint',b),'mae_mean']/d.loc[('baseline_lightgbm',b),'mae_mean']-1) for b in bands]
fig,ax=plt.subplots(figsize=(9,3.8));ax.barh(range(4),vals,color=['#1e687e' if v<0 else '#d8773b' for v in vals]);ax.set_yticks(range(4),['Negative: y < 0','Normal: 0 <= y <= 300','High: y > 300','Extreme: y > 1000']);ax.axvline(0,color='#333333',lw=.8);ax.set_xlim(-8.6,6.5);ax.invert_yaxis();ax.set_xlabel('Full_joint vs LightGBM: MAE change (%) | negative = improvement');ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
for i,v in enumerate(vals):ax.text(v+(.15 if v>0 else -.15),i,f'{v:+.2f}%',va='center',ha='left' if v>0 else 'right')
fig.tight_layout();fig.savefig(O/'tail_tradeoff.png',dpi=220);plt.close(fig)
