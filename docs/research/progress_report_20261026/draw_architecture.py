from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties
O=Path(__file__).parent
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fig,ax=plt.subplots(figsize=(10,5.8));ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis('off')
def box(x,y,w,h,text,color='#EEF4F7'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',fc=color,ec='#648293',lw=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontproperties=font,fontsize=10)
def arrow(x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':'#486A7D','lw':1.3})
box(.15,4.7,2.45,1,'起点已知公共输入\n历史72h＋外生预测')
box(3.2,4.7,2.6,1,'冻结 LightGBM\n24h × 5分位 B')
box(6.4,4.7,3.35,1,'非对称 Anchor 校正 A\n位置 δ；下/上尺度 s−、s+')
arrow(2.65,5.2,3.14,5.2);arrow(5.86,5.2,6.34,5.2)
box(.15,2.7,2.45,1.2,'成熟历史与误差统计\n24h标签完整可见\n候选回看≤730天')
box(3.2,2.7,2.6,1.2,'历史趋势—事件分解\nFourier趋势 T\n6个局部脉冲 E')
box(6.4,2.7,3.35,1.2,'当前状态检索与时间对齐\n最多16候选＋空候选\n趋势/事件采用门')
arrow(2.66,3.3,3.14,3.3);arrow(5.86,3.3,6.34,3.3)
arrow(8.05,4.65,8.05,3.96)
box(3.2,.65,2.6,1.2,'有界历史残差 R\n共同平移五个分位\n原始输出 Q = A + R')
box(6.4,.65,3.35,1.2,'选择期冻结 checkpoint / α\n正式输出 B + α(Q−B)\n测试期不按结果重新选优','#F7EADF')
arrow(8.05,2.65,4.5,1.91);arrow(5.86,1.25,6.34,1.25)
ax.text(.15,.12,'当前版本：asymmetric_balance_20261023；基础树冻结，神经模块联合训练；当前未来真实电价不可见。',fontproperties=font,fontsize=8.5,color='#486A7D')
fig.tight_layout();fig.savefig(O/'architecture.png',dpi=220,bbox_inches='tight');plt.close(fig)
