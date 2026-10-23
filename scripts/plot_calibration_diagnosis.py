"""Static research figure from the frozen three-seed diagnostic CSV."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'logs/forecasting/v5_calibration_diagnosis_20261023'


def main():
    with (DATA/'groups.csv').open() as stream:
        rows=[r for r in csv.DictReader(stream) if r['scope']=='aligned_2023_2024']
    def get(region,component,group,key):
        r=next(r for r in rows if (r['region'],r['component'],r['group'])==(region,component,group))
        return float(r[key])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
    colors={'Anchor':'#f39c42','Full':'#2667a4','B':'#6a727b'}
    regions=['NSW1','QLD1','TAS1'];x=np.arange(3)
    for i,name in enumerate(('Anchor','Full')):
        values=[get(r,name,'all','mae_minus_B_mean') for r in regions]
        bars=axes[0,0].bar(x+(i-.5)*.35,values,.35,label=name,color=colors[name])
        axes[0,0].bar_label(bars,fmt='%.2f',padding=3)
    axes[0,0].axhline(0,color='black',lw=.7);axes[0,0].set_xticks(x,regions)
    axes[0,0].set_ylabel('MAE minus the same frozen tree')
    axes[0,0].set_title('(a) Region-level correction cost');axes[0,0].legend(frameon=False)
    months=sorted({r['group'] for r in rows if len(r['group'])==7 and r['group'][4]=='-'})
    for name in ('Anchor','Full'):
        axes[0,1].plot(range(24),[get('QLD1',name,m,'mae_minus_B_mean') for m in months],
                       label=name,color=colors[name],marker='o',markersize=3)
    axes[0,1].axhline(0,color='black',lw=.7);axes[0,1].set_xticks(range(0,24,3),months[::3],rotation=30)
    axes[0,1].set_ylabel('Monthly MAE difference');axes[0,1].set_title('(b) QLD1: persistent, seasonal regressions')
    names=['B','Anchor','Full','Full_tree_width'];x=np.arange(4)
    coverage=[100*get('QLD1',n,'all','coverage80_mean') for n in names]
    bars=axes[1,0].bar(x,coverage,color=['#6a727b','#f39c42','#2667a4','#63a5c8'])
    axes[1,0].bar_label(bars,fmt='%.1f%%',padding=3)
    axes[1,0].axhline(80,color='black',ls='--',lw=.8,label='Nominal 80%')
    axes[1,0].set_xticks(x,['Tree','Anchor','Full','Full + tree width'])
    axes[1,0].set_ylim(0,90);axes[1,0].set_ylabel('80% interval coverage (%)')
    axes[1,0].set_title('(c) Restoring width keeps Full MAE unchanged');axes[1,0].legend(frameon=False)
    bins=['-inf:0','0:100','100:300','300:1000','1000:inf'];x=np.arange(5)
    total=get('QLD1','Full','all','n_cells_per_seed')
    for i,metric in enumerate(('mae','qs')):
        values=[get('QLD1','Full','actual_diagnostic_only/'+b,metric+'_minus_B_mean')*
                get('QLD1','Full','actual_diagnostic_only/'+b,'n_cells_per_seed')/total for b in bins]
        axes[1,1].bar(x+(i-.5)*.35,values,.35,label=metric.upper(),color=['#2667a4','#bf6549'][i])
    axes[1,1].axhline(0,color='black',lw=.7);axes[1,1].set_xticks(x,['<0','0–100','100–300','300–1000','>=1000'])
    axes[1,1].set_xlabel('Observed price (diagnostic only; AUD/MWh)')
    axes[1,1].set_ylabel('Weighted contribution to Full minus tree')
    axes[1,1].set_title('(d) Common prices hurt MAE; extremes hurt QS');axes[1,1].legend(frameon=False)
    fig.suptitle('Frozen Full_joint diagnosis | 2023–2024 | mean over 3 neural seeds',fontsize=14)
    fig.savefig(DATA/'diagnosis.png',dpi=180)
    fig.savefig(DATA/'diagnosis.svg')
    svg=DATA/'diagnosis.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':main()
