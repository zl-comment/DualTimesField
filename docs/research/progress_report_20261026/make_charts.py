from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
O=Path(__file__).parent; R=Path('/root/workspace/Code/DualTimesField/logs/forecasting/main_comparison_20261023')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
d=pd.read_csv(R/'main.csv'); reg=pd.read_csv(R/'regional.csv')
def label(x):
 return x.replace('baseline_','').replace('lasso_arx72_cp','Lasso-ARX72+CP').replace('gatets_adapter','GateTS').replace('chronos2_zeroshot','Chronos-2').replace('moirai2_zeroshot','Moirai-2').replace('timesfm25_xreg','TimesFM-2.5+XReg').replace('tabpfn_v3','TabPFN-3')
sel=d[(d.role.str.contains('baseline')) | (d.model=='Full_joint')].sort_values('mae_mean')
fig,ax=plt.subplots(figsize=(9,7.2))
colors=['#d8773b' if x=='Full_joint' else '#1e687e' if x=='baseline_lightgbm' else '#91a6b4' for x in sel.model]
y=np.arange(len(sel));ax.barh(y,sel.mae_mean,color=colors,xerr=sel.mae_std,capsize=2)
ax.set_yticks(y, [label(x) for x in sel.model]);ax.invert_yaxis();ax.set_xlabel('Macro MAE (AUD/MWh), mean of 3 seeds');ax.set_xlim(0,89);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
for i,v in enumerate(sel.mae_mean):ax.text(v+1,i,f'{v:.2f}',va='center',fontsize=9)
fig.tight_layout();fig.savefig(O/'comparison_mae.png');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,3.7))
for ax,col,title in zip(axs,['mae_mean','quantile_score_proxy_mean'],['MAE','Five-quantile QS']):
 for j,(m,c) in enumerate([('baseline_lightgbm','#1e687e'),('baseline_xgboost','#91a6b4'),('Full_joint','#d8773b')]):
  a=reg[reg.model==m].set_index('region').loc[['NSW1','QLD1','TAS1']]
  ax.bar(np.arange(3)+(j-1)*.24,a[col],width=.23,color=c,label=label(m))
 ax.set_xticks(range(3),['NSW1','QLD1','TAS1']);ax.set_title(title);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axs[0].set_ylabel('AUD/MWh');axs[1].legend(fontsize=8);fig.tight_layout();fig.savefig(O/'regional_comparison.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,3.8))
q=d.set_index('model').loc[['baseline_lightgbm','baseline_lightgbm_cp','baseline_lightgbm_qra','Full_joint','Anchor_joint']]
x=np.arange(len(q));ax.bar(x-.16,q.coverage_80_mean*100,.31,label='80% interval',color='#1e687e');ax.bar(x+.16,q.coverage_90_mean*100,.31,label='90% interval',color='#d8773b');ax.axhline(80,color='#1e687e',ls='--',lw=1);ax.axhline(90,color='#d8773b',ls='--',lw=1);ax.set_xticks(x,[label(s) for s in q.index]);ax.tick_params(axis='x',labelrotation=12);ax.set_ylim(55,102);ax.set_ylabel('Empirical coverage (%)');ax.legend(ncol=2,loc='upper right');fig.tight_layout();fig.savefig(O/'coverage.png');plt.close(fig)
