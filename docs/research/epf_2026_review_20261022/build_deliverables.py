import collections
import csv
import hashlib
import html
import json
import re
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from docx import Document
from docx.shared import Inches, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from synthesis import OUT, REPO, DATE, IDENTITIES, BASELINES, ARCHITECTURES, PROTOCOL, EXTRA_SOURCES


def readj(name):
    return json.loads((OUT / name).read_text())


def txt(v):
    if v is None:
        return ''
    if isinstance(v, list):
        return '\n'.join(txt(x) for x in v)
    if isinstance(v, dict):
        return '\n'.join(f'{k}: {txt(val)}' for k, val in v.items())
    return str(v)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


input_metadata = readj('input_metadata.json')
assert digest(OUT/'input_rows.json') == input_metadata['input_rows_sha256'], 'Extracted input hash mismatch'
original = readj('input_rows.json')
orig = {r['ID']: r for r in original}
reviews = sum((readj(n) for n in ['papers_p_review.json', 'papers_j_review.json', 'papers_ec_review.json']), [])
assert len(reviews) == 109, len(reviews)
assert collections.Counter(r['id'] for r in reviews) == collections.Counter(list(orig)), 'ID coverage mismatch'
# Separate fragment access and old-version full text from current full-text access.
for r in reviews:
    r['reviewer_evidence_status'] = r['evidence_status']
    if r['id'] in ('J15','J26','J35'):
        r['evidence_status'] = '正文片段'
    if r['id'] in ('J31','E05'):
        r['evidence_status'] = '前版全文/终版摘要'
review = {r['id']: r for r in reviews}
reviews = sorted(reviews, key=lambda r: r['id'])
required = ['id','evidence_status','market_verified','task_verified','reported_results','comparability','sources']
for r in reviews:
    assert all(k in r for k in required), r['id']
    assert r['sources'], r['id']
    assert r['comparability'] != '可直接排名', 'Requires independent protocol proof'
rankings = readj('local_rankings.json')
regional = readj('local_regional.json')
audit = readj('local_audit.json')
assert audit['passed'] and audit['rescored_artifacts'] == 51
counts = dict(collections.Counter(r['evidence_status'] for r in reviews))
comp_counts = dict(collections.Counter(r['comparability'] for r in reviews))
full = next(r for r in rankings if r['model'] == 'Full_joint')
tree = next(r for r in rankings if r['model'] == 'baseline_lightgbm')
assert full['mae_rank_13'] == 2 and full['mae_rank_all17'] == 4
delta_mae = (full['mae'] / tree['mae'] - 1) * 100
delta_qs = (full['quantile_score_proxy'] / tree['quantile_score_proxy'] - 1) * 100


def links_for(ids):
    return '\n'.join(dict.fromkeys(s['url'] for i in ids for s in review[i]['sources']))


def source_summary(r):
    return '\n'.join(f"{s.get('url','')} | {s.get('location','')}" for s in r.get('sources', []))


baseline_rows = []
for b in BASELINES:
    baseline_rows.append({
        '综合优先级': b['priority'], '论文ID': ', '.join(b['ids']),
        '论文题目': '\n'.join(f"{i} {review[i]['title_verified']}" for i in b['ids']),
        '建议补充的模型': b['models'], '实施动作': b['action'], '为什么需要': b['why'],
        '公平复现要求': b['protocol'], '预计成本（判断）': b['cost'], '验收指标': b['success'],
        '限制与风险': b['risk'], '本轮核验深度': '\n'.join(f"{i}: {review[i]['evidence_status']}" for i in b['ids']),
        '代码/项目入口（可用性见总表）': '\n'.join(f"{i}: {review[i].get('code_url') or '本轮未确认'}" for i in b['ids']),
        '来源': links_for(b['ids']),
    })

paper_rows = []
for r in reviews:
    o = orig[r['id']]
    paper_rows.append({
        'ID': r['id'], '核验标题': r['title_verified'], '本轮证据深度': r['evidence_status'],
        '市场': r['market_verified'], '任务': r['task_verified'], '测试期': r.get('test_period'),
        '分辨率': r.get('resolution'), '预测跨度': r.get('horizon'), '核实的结果（原文协议）': r['reported_results'],
        '与本地可比性': r['comparability'], '为什么不能直接排名': r.get('comparability_reason'),
        '审阅员基线建议': r.get('baseline_priority'), '推荐原因': r.get('baseline_reason'),
        '指标定义': r.get('metrics_definition'), '价格尺度/变换': r.get('target_transform'),
        '预测起点/更新/切分': r.get('origin_protocol'), '总体数据期': r.get('data_period'),
        '文中实际基线': r.get('paper_baselines'), '代码/项目入口': r.get('code_url') or '本轮未确认',
        '复现可行性': r.get('reproduction_feasibility'), '可借鉴结构': r.get('architecture_idea'),
        '结构/协议风险': r.get('architecture_risk'), '来源与位置': source_summary(r),
        '备注/版本': r.get('notes'), '附件年份口径（非全部重核）': o.get('年份口径'),
        '附件日期（非全部重核）': o.get('发表日期与版本说明'), '附件DOI': o.get('DOI'),
        '附件原URL': o.get('论文 URL'), '审阅日期': DATE,
    })

local_rows = []
for r in rankings:
    name, kind, note = IDENTITIES[r['model']]
    local_rows.append({
        'MAE名次／13输出': r.get('mae_rank_13'), 'QS名次／13输出': r.get('quantile_score_proxy_rank_13'),
        'MAE名次／17输出': r['mae_rank_all17'], 'QS名次／17输出': r['quantile_score_proxy_rank_all17'],
        '模型': name, '输出标识': r['model'], '身份': kind,
        'MAE AUD/MWh': r['mae'], 'QS5': r['quantile_score_proxy'],
        '地区平均RMSE': r['rmse_pooled'], 'RMSE名次／17': r['rmse_pooled_rank_all17'],
        '80%覆盖率': r['coverage_80'], '90%覆盖率': r['coverage_90'],
        'NSW1 MAE': r['NSW1_mae'], 'QLD1 MAE': r['QLD1_mae'], 'TAS1 MAE': r['TAS1_mae'],
        '种子': 2026, '测试期': '2023–2024', '复现身份/信息差异': note,
    })

with (REPO/'logs/forecasting/aligned_baselines_20261013_extension/results/main.csv').open() as f:
    seeds = list(csv.DictReader(f))
seeds.sort(key=lambda r: float(r['mae']))
seed_rows = [{
    'MAE排序（仅旧12输出）': i+1, '模型': IDENTITIES.get('baseline_'+r['model'], (r['model'],))[0],
    'MAE': float(r['mae']), 'QS5': float(r['quantile_score_proxy']), '地区平均RMSE': float(r['rmse_pooled']),
    '80%覆盖率': float(r['coverage_80']), '90%覆盖率': float(r['coverage_90']),
    '80%宽度': float(r['interval_width_80']), '90%宽度': float(r['interval_width_90']),
    '口径': 'seed 2026/2027/2028 的分数平均；不是预测集成；不含 Full_joint',
} for i,r in enumerate(seeds)]
assert len(seed_rows) == 12

with (REPO/'logs/forecasting/v5_anchor_architecture_20261022/macro.csv').open() as f:
    dev = list(csv.DictReader(f))
dev_rows = [{
    '开发年度': int(r['fold']), '方案': r['variant'], '选择方式': r['mode'],
    'Full MAE': float(r['mae']), 'Full QS5': float(r['qs']),
    'Anchor MAE': float(r['anchor_mae']), 'Anchor QS5': float(r['anchor_qs']),
    '回退数': int(r['fallback_count']),
    '评价限制': '三地区宏平均；单种子6轮screen；不是2023–2024测试；没有候选通过全部预注册标准',
} for r in dev]

evidence_rows = []
for r in reviews:
    for s in r['sources']:
        evidence_rows.append({'ID': r['id'], '证据深度': r['evidence_status'], 'URL': s.get('url'),
                              '位置': s.get('location'), '支持的判断': s.get('support'), '日期': DATE})
for i,title,url,loc in EXTRA_SOURCES:
    evidence_rows.append({'ID': i, '证据深度':'官方仓库页面', 'URL': url, '位置':loc, '支持的判断':title, '日期':DATE})

au_ids = [r['id'] for r in reviews if re.search(r'NEM|AEMO|澳大利亚|澳洲|NSW|QLD|Victoria|维多利亚', txt(r.get('market_verified')), re.I)]
au_rows = [r for r in paper_rows if r['ID'] in au_ids]
arch_rows = [{
    '结构候选': a['name'], '优先级': a['priority'], '文献ID': ', '.join(a['ids']),
    '借鉴与原创边界': a['idea'], '改动位置': a['location'], '参数化/规则': a['formula'],
    '针对的问题': a['why'], '必要对照实验': a['experiment'], '现有证据': a['evidence'],
    '主要风险': a['risk'], '采纳条件': a['decision'], '来源':links_for(a['ids']),
} for a in ARCHITECTURES]

findings = [
 ('范围', '逐条审查附件109条记录；这是用户提供的检索集，不是经穷尽检索证明的2026全部论文。包含跨年修订、综述、分类/日内/远期等不同任务。'),
 ('核验深度', '；'.join(f'{k} {v}条' for k,v in counts.items())+'。全文表示取得正文并核过相关方法/实验章节，并非每篇全表复算；旧版与片段另外列明。未复跑外部论文，缺失值未补造。'),
 ('能够给出的本地名次', f'旧 Full_joint：在“原12输出 + Full_joint”的13输出集合中 MAE/QS均第2；若加重训B和3个同检查点干预，17输出中均第4，RMSE第8。集合包含模型/消融/适配，并非13或17篇独立论文。'),
 ('本地数值', f'Full_joint MAE {full["mae"]:.6f}、QS5 {full["quantile_score_proxy"]:.6f}；原LightGBM MAE {tree["mae"]:.6f}、QS5 {tree["quantile_score_proxy"]:.6f}。Full的MAE高{delta_mae:.2f}%，QS高{delta_qs:.2f}%，没有超过树的宏指标。'),
 ('地区差异', 'Full_joint 在TAS1的MAE优于原树（28.449478 vs 28.706143），在NSW1与QLD1更差。不能把宏平均掩盖的地区差异写成一致改善。'),
 ('公平性边界', '51份文件已重新评分、校验哈希及origin/target/actual逐项对齐；Full额外使用成熟历史/误差，HPO和在线信息权限尚未完全公平。当前是同样本同评分的回顾性描述排序。'),
 ('不能给出的名次', '不能给出“在109篇论文里第几”或世界前百分之几；没有外部论文已证明与本地市场、测试时间、粒度、信息集、训练与评分完全一致。'),
 ('当前新方案', '2021/2022开发screen已经完成，但仅位置等候选尚无本地2023–2024正式成绩；没有候选通过全部预注册标准。不能拿旧Full_joint名次代表新方案。'),
 ('计算方法', '本地QS5=2×五分位平均pinball；E04/J34所称CRPS为99分位平均pinball且省略2倍，PriceFM为另7分位AQL。系数、网格、市场、任务都不同，不能直接换算排序。'),
 ('复现命名', 'PriceFM=单区从零训练适配；TimeXer/GateTS新增外生/分位接口；PatchTST本地重写；Lasso-ARX72+CP不是官方LEAR。应按实际身份报告。'),
 ('补基线顺序', '第一批：LEAR+CP / DDNN-JSU、树+CP/QRA/简单残差、NBEATSx或NHITS、TabPFN；XMatch并行核实现。第二批：Chronos-2+CING-LEAR/简单集成、QNBM/NBMLSS、在线结构。NEM文献复现另设轨道。'),
 ('新结构方向', '先验证非对称单调分位头、支持度检索门、等信息在线校正；分布参数头与点轨迹协调次之；联合概率场景层在需要跨时依赖时升级。全部为研究提案，不保证改善。'),
 ('日期与版本', '按具体版本记录结果，保留2025首发/2026修订区分；C12同题两个SSRN版本不重复当作两项成果；原表核实等级不继承为本轮核验。'),
]

WB_PATH = OUT/'2026电价论文_市场结果与基线结构总表.xlsx'
BASELINE_PATH = OUT/'建议补充的基线论文_实施清单.xlsx'
DOCX_PATH = OUT/'2026电价论文_对比结论与结构建议.docx'
HTML_PATH = OUT/'2026电价论文_对比结论与结构建议.html'

wb = Workbook()
wb.remove(wb.active)
table_counter = 0
navy = '17324D'
teal = '087F8C'
gray = '5D6B78'


def safe_cell(cell, value):
    if value is None:
        return
    if isinstance(value, (list,dict)):
        value = txt(value)
    if isinstance(value, str):
        value = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', value)
        assert len(value) < 32767, 'Excel text too long'
        cell.value = value
        cell.data_type = 's'  # external text must never become spreadsheet formulas
    else:
        cell.value = value


def add_sheet(name, title, subtitle, rows, widths=None, freeze='C5'):
    global table_counter
    ws = wb.create_sheet(name)
    headers = list(rows[0]) if rows else ['无记录']
    n = len(headers)
    ws.merge_cells(start_row=1,start_column=1,end_row=1,end_column=max(n,2))
    ws.cell(1,1,title).font = Font(name='微软雅黑',size=16,bold=True,color='FFFFFF')
    ws.cell(1,1).fill = PatternFill('solid',fgColor=navy)
    ws.row_dimensions[1].height = 32
    ws.merge_cells(start_row=2,start_column=1,end_row=2,end_column=max(n,2))
    ws.cell(2,1,subtitle).font = Font(name='微软雅黑',size=10,color=gray)
    ws.cell(2,1).alignment = Alignment(wrap_text=True,vertical='center')
    ws.row_dimensions[2].height = 40
    ws.cell(3,1,'← 阅读与结论').hyperlink = "#'阅读与结论'!A1"
    ws.cell(3,1).font = Font(color=teal,underline='single')
    for j,h in enumerate(headers,1):
        c=ws.cell(4,j,h);c.fill=PatternFill('solid',fgColor=navy)
        c.font=Font(name='微软雅黑',bold=True,color='FFFFFF',size=10)
        c.alignment=Alignment(wrap_text=True,vertical='center')
    ws.row_dimensions[4].height=32
    for i,r in enumerate(rows,5):
        for j,h in enumerate(headers,1):
            c=ws.cell(i,j);safe_cell(c,r.get(h))
            c.font=Font(name='微软雅黑',size=10,color='243746')
            c.alignment=Alignment(wrap_text=True,vertical='top')
            if isinstance(c.value,float):
                c.number_format='0.0000'
            if '覆盖率' in h and isinstance(c.value,(int,float)):
                c.number_format='0.00%'
            if isinstance(c.value,str) and c.value.startswith('https://') and '\n' not in c.value and ' | ' not in c.value:
                c.hyperlink=c.value;c.font=Font(name='微软雅黑',size=10,color=teal,underline='single')
            if c.value in ('P0','P1','P2'):
                c.fill=PatternFill('solid',fgColor={'P0':'D8EEE9','P1':'FFF0CC','P2':'E7EAF0'}[c.value])
            if c.value in ('全文','摘要','元数据','未能获取','身份冲突'):
                c.fill=PatternFill('solid',fgColor={'全文':'D8EEE9','摘要':'FFF0CC','元数据':'E7EAF0','未能获取':'FADBD8','身份冲突':'FADBD8'}[c.value])
        ws.row_dimensions[i].height=75 if name not in ('本地同样本排名','分地区结果','三种子基线','新方案开发对照') else 42
    for j,h in enumerate(headers,1):
        numeric = any(isinstance(r.get(h),(int,float)) for r in rows)
        default_width = 15 if numeric else 28
        width=(widths or {}).get(h, default_width)
        ws.column_dimensions[get_column_letter(j)].width=width
    if rows:
        table_counter+=1
        tab=Table(displayName=f'EPFTable{table_counter}',ref=f'A4:{get_column_letter(n)}{4+len(rows)}')
        tab.tableStyleInfo=TableStyleInfo(name='TableStyleMedium2',showRowStripes=True,showColumnStripes=False)
        ws.add_table(tab)
    ws.freeze_panes=freeze
    ws.sheet_view.showGridLines=False
    ws.sheet_properties.pageSetUpPr.fitToPage=True
    ws.page_setup.orientation='landscape';ws.page_setup.paperSize=ws.PAPERSIZE_A3
    ws.page_setup.fitToWidth=1;ws.page_setup.fitToHeight=0
    ws.print_title_rows='1:4'
    ws.print_options.horizontalCentered=True
    return ws


add_sheet('阅读与结论','2026 电力价格预测文献审查与本地定位',
          f'审查日期 {DATE} | 109条输入记录 | 3个文献审阅智能体 + 主代理本地复算与综合 | 推荐优先级为项目判断',
          [{'主题':a,'结论与边界':b} for a,b in findings],{'主题':24,'结论与边界':130},freeze='B5')
add_sheet('本地同样本排名','本地同样本排序：旧 Full_joint 与既有输出',
          '13输出=原12输出+Full_joint；17输出另含重训B和3个同检查点干预。非独立论文榜、非完全等信息榜。越低越好。',
          local_rows,{'模型':40,'输出标识':32,'复现身份/信息差异':70},freeze='F5')
add_sheet('分地区结果','51份预测的地区评分',
          '同region的origin、目标时刻和actual逐项一致；每region 17,521个origin × 24h；不可把重叠预测当独立样本做显著性。',
          [{'地区':r['region'],'模型标识':r['model'],'MAE':r['mae'],'QS5':r['quantile_score_proxy'],'RMSE':r['rmse_pooled'],
            '80%覆盖率':r['coverage_80'],'90%覆盖率':r['coverage_90'],'origin数':r['n_origins'],'评价单元数':r['n_cells']} for r in regional],
          {'模型标识':35},freeze='C5')
add_sheet('三种子基线','旧12输出：三种子分数平均，仅作稳健性背景',
          'seed 2026/2027/2028；未包含新Full_joint；不能将这里的平均分数与单种子主榜混排。',
          seed_rows,{'模型':42,'口径':80},freeze='C5')
add_sheet('新方案开发对照','新 Anchor 架构候选：已完成开发筛选，尚未正式晋升',
          '36组拟合/216轮；6方案×3地区×2年，单种子、6轮screen；2021/2022均开发期。raw与guarded不得混成纯结构效应。',
          dev_rows,{'评价限制':90,'方案':24},freeze='D5')
add_sheet('109篇论文结果总表','109条文献：市场、任务、结果、比较条件与证据',
          '原文数值只属于该行所述版本和协议，不参与跨论文总排名。“未核”不等于论文未提供；全文不等于复跑实验。',
          paper_rows,{'ID':9,'核验标题':58,'市场':36,'任务':38,'核实的结果（原文协议）':86,'为什么不能直接排名':66,
                      '指标定义':62,'价格尺度/变换':50,'预测起点/更新/切分':66,'来源与位置':85,'备注/版本':70},freeze='D5')
add_sheet('澳洲市场重点','澳洲论文重点：同市场也不能直接比较公布误差',
          '按本轮核验市场字段筛选；不同年份、半小时/小时、一步/24h/一周、归一化/原价均已分别记录。',
          au_rows,{'ID':9,'核验标题':58,'市场':35,'核实的结果（原文协议）':86,'为什么不能直接排名':65,'来源与位置':80},freeze='D5')
add_sheet('需补基线论文','建议补充的基线论文与实施顺序',
          '综合P0=优先补齐主张所需对照；P1=第二批；P2=目标扩展/高成本。与逐篇审阅建议可能不同，综合考虑重叠和资源。',
          baseline_rows,{'论文题目':75,'建议补充的模型':43,'为什么需要':65,'公平复现要求':80,'限制与风险':70,'来源':80},freeze='E5')
add_sheet('新结构讨论','可移植结构与必要实验：文献证据和我们的提案分开',
          '所有候选尚未在本地主测试上验证；不要把论文在其数据上的收益解释成我们模型的预期收益。',
          arch_rows,{'结构候选':45,'参数化/规则':90,'必要对照实验':75,'现有证据':72,'主要风险':65,'来源':80},freeze='D5')
add_sheet('指标协议对齐','计算方法是否一致：16项核对表',
          '可计算名次要求预测对象、数据、可用信息与评分均相容；换系数只能解决很小一部分差异。',
          [{'核对项':a,'本地协议':b,'论文中常见差异':c,'统一方法':d} for a,b,c,d in PROTOCOL],
          {'核对项':25,'本地协议':70,'论文中常见差异':90,'统一方法':90},freeze='B5')
add_sheet('逐项证据','论文证据索引',
          '优先原文、出版方、作者稿和官方仓库；逐条定位。摘要中的结果不升级为全文核验。',
          evidence_rows,{'ID':10,'URL':85,'位置':65,'支持的判断':105},freeze='C5')
add_sheet('原始109条','附件原始109条记录（保留输入，不继承其结论）',
          '本表为附件转录，原“A/B/C”核实等级是输入声明；请以“109篇论文结果总表”的本轮核验为准。',
          original,{'ID':10,'论文题目':65,'实际比较的基线':80,'市场与预测任务':65,'备注与代码链接':80},freeze='C5')

audit_rows=[{'类别':'输入附件','对象':input_metadata['original_name'],'值/状态':input_metadata['original_sha256'],'备注':'审查时记录的原附件SHA256；重建使用归档转录'},
            {'类别':'归档转录','对象':'input_rows.json','值/状态':digest(OUT/'input_rows.json'),'备注':'重建时校验SHA256，共109条'},
            {'类别':'范围','对象':'109条记录','值/状态':json.dumps(counts,ensure_ascii=False),'备注':'全文=相关方法/实验段；未复跑外部实验'},
            {'类别':'本地','对象':'local_audit.json','值/状态':'通过：51份重评分；13项归档hash核对','备注':'所有51份预测另记录sha256；origin/targets/actuals一致'},
            {'类别':'比较','对象':'可比性分类计数','值/状态':json.dumps(comp_counts,ensure_ascii=False),'备注':'没有外部论文被准许直接插入本地榜'}]
for n in ['papers_p_review.json','papers_j_review.json','papers_ec_review.json','local_audit.json','local_rankings.json']:
    audit_rows.append({'类别':'本轮审查产物','对象':str(OUT/n),'值/状态':digest(OUT/n),'备注':'SHA256'})
for e in audit['evidence']:
    audit_rows.append({'类别':'预测文件','对象':e['path'],'值/状态':e['sha256'],
                       '备注':f"{e['region']} / {e['model']}；aligned={e['aligned_origins_targets_labels']}；rescored={e['scores_recomputed']}"})
add_sheet('审计信息','本轮文件与预测审计',
          '审阅未修改原输入、训练代码或已归档预测；独立输出可追溯到原始文件与SHA256。',
          audit_rows,{'类别':22,'对象':120,'值/状态':95,'备注':85},freeze='C5')
wb.save(WB_PATH)

# A small standalone workbook lets the user hand off baseline implementation
# without carrying the whole literature audit.
complete_wb=wb
wb=Workbook();wb.remove(wb.active)
add_sheet('阅读与结论','建议补充的基线：实施清单',
          '从完整研究总表抽出；没有启动新增训练。优先级是本项目实施判断，并非论文质量榜。',
          [{'主题':'优先级','结论与边界':'P0优先补齐当前研究主张的必要对照；P1第二批；P2高成本或联合情景方向。'},
           {'主题':'公平性','结论与边界':'相同市场、切分、origin、可用外生与历史、调参/更新预算和评分；72h与扩展长期信息分轨。'},
           {'主题':'复现身份','结论与边界':'严格区分官方实现、移植、结构适配与论文完整实验复现；未核代码不可当作已可运行。'},
           {'主题':'完整材料','结论与边界':str(WB_PATH)}],{'主题':24,'结论与边界':130},freeze='B5')
add_sheet('需补基线论文','基线论文与实施要求',
          '列出论文ID/题目、模型、为什么补、协议、成本、代码核验状态与来源。',
          baseline_rows,{'论文题目':75,'建议补充的模型':44,'公平复现要求':80,'为什么需要':65,'限制与风险':70,'来源':80},freeze='E5')
recommended_ids=set(i for b in BASELINES for i in b['ids'])
add_sheet('涉及论文核验','实施清单涉及论文的逐条证据',
          '这些原文数字均不能直接插入本地榜；先按统一协议复现。',
          [r for r in paper_rows if r['ID'] in recommended_ids],
          {'ID':9,'核验标题':58,'核实的结果（原文协议）':86,'为什么不能直接排名':66,'来源与位置':85},freeze='D5')
wb.save(BASELINE_PATH)
wb=complete_wb

# The same report content feeds HTML and Word, avoiding inconsistent narratives.
blocks=[]
def p(s): blocks.append(('p',s))
def h(s): blocks.append(('h',s))
def t(headers,rows): blocks.append(('t',(headers,rows)))

p(f'审查日期：{DATE}。依据用户附件的109条记录，由3个文献审阅智能体分组核验，主代理核查关键原文并复算本地51份冻结预测。核验深度：'+ '、'.join(f'{k}{v}条' for k,v in counts.items())+'。全文指相关方法与实验章节；外部论文未在本轮重新训练。')
h('我们现在能排到哪里')
p(findings[2][1])
p(findings[3][1])
p('这不是“在2026所有论文中第二名”。这109条记录涵盖不同市场和任务，也未证明穷尽全部2026研究。外部论文没有满足与本地同数据、同时间、同信息和同评分的直接排名条件。')
t(['模型/输出','MAE','QS5','MAE名次 13/17','QS名次 13/17'],[
    [IDENTITIES[r['model']][0],f"{r['mae']:.4f}",f"{r['quantile_score_proxy']:.4f}",
     f"{r.get('mae_rank_13') or '—'} / {r['mae_rank_all17']}",
     f"{r.get('quantile_score_proxy_rank_13') or '—'} / {r['quantile_score_proxy_rank_all17']}"] for r in rankings])
p('13输出集合仍含早期模型及消融，不是13篇独立论文；17输出多了重训B、Anchor_joint、event_off、time_shift_off。后三个是同检查点关闭干预。完整分地区数据和旧12基线三种子均值见Excel，后者未与单种子Full_joint混排。')
p('Full_joint 在 TAS1 的 MAE 为28.4495，优于原树28.7061；NSW1和QLD1分别45.5274/38.9910，差于原树44.3633/38.1975。RMSE在17输出中第8；它使用中位数预测评分，不能据此推断均值预测的最优性。')
p('本地评分：澳洲NEM三个地区，原始RRP（AUD/MWh），小时级，72小时历史预测未来24小时，逐小时滚动，2023–2024，每区17,521个起点。Full使用额外成熟历史/误差，其他基线未全部获得等价的长期信息和在线校准机会；系统等预算调参也未完成。因此这是一份同样本、同评分的描述性排序。')

h('新方案的结果与旧模型必须分开')
p('当前平衡损失与架构候选只完成2021/2022开发筛选：6方案×3地区×2年度，36次拟合、216轮，单种子、每模型6轮。没有候选通过全部预注册标准，尚无新方案在2023–2024上的正式排名。')
gd={(r['fold'],r['variant']):r for r in dev if r['mode']=='guarded'}
t(['开发候选（guarded）','2021 MAE / QS','2022 MAE / QS'],[
    [v, ' / '.join(f"{float(gd[('2021',v)][k]):.4f}" for k in ['mae','qs']),
        ' / '.join(f"{float(gd[('2022',v)][k]):.4f}" for k in ['mae','qs'])]
    for v in ['balanced','identity_anchor','location_only','separate_heads','staged','wide_head']])
p('仅位置校正结合既定选择/收缩流程是值得继续验证的候选，但TAS1/2021相对B的MAE增加约0.03887，未通过全部条件；raw结果也不是每年MAE与QS同时改善。选择的检查点和收缩系数可能变化，不能单独归因为“关闭尺度解决了问题”。独立头、加宽头、3+3分阶段也未给出跨年一致胜出证据。')

h('为什么外部论文数字不能直接插入我们的榜')
spot_ids=['P20','P11','E07','J09','J30','P04','E04','J34','P23','P30']
t(['记录','市场/任务与测试','原文核验结果','比较限制'],[
    [i, f"{review[i]['market_verified']}；{review[i].get('test_period','')}；{review[i].get('horizon','')}",
     review[i]['reported_results'],review[i]['comparability_reason']] for i in spot_ids])
p('例如P20的NSW 24h结果82.409不能因为大于我们的45.527就判定我们胜出：其2025H1、半小时48步、历史长度和输入条件均不同。E07的NSW MAE3.784也不能证明其强于我们：它是半小时一步并有不同测试期和选择流程。完整109条的数值、缺失项、版本与出处在Excel。')

h('计算公式：有的同名指标并不相同')
p('令 u=y−ŷq，pinball损失 ρq(u)=u·(q−1[u<0])。我们的 QS5=2×mean(ρq)，平均覆盖起点、24个horizon与Q={0.05,0.10,0.50,0.90,0.95}，最后三地区等权。这个五点分数不能写成精确CRPS。')
p('连续CRPS(F,y)=2∫₀¹ρq(y−F⁻¹(q))dq。E04和J34用99分位平均pinball并省略2倍；PriceFM v4用另7个分位的AQL。即使补上系数，分位网格、权重和任务仍不同；五点输出也不足以无假设还原密网格积分。')
p('覆盖率需要与区间宽度一起比较；90%区间覆盖98%未必更好。负价/零价市场的MAPE、归一化误差、分类Accuracy，以及策略利润也不能与原价MAE混排。')
t(['核对项','本地与差异','需要怎么统一'],[[a,b+'；'+c,d] for a,b,c,d in PROTOCOL])

h('已有“复现”的准确身份')
t(['实现','当前身份与限制'],[[IDENTITIES[k][0],IDENTITIES[k][2]] for k in ['baseline_pricefm','baseline_timexer','baseline_gatets_adapter','baseline_patchtst','baseline_lasso_arx72_cp','baseline_mlp']])
p('建议论文中使用“在统一本地协议下的适配实现”这类准确命名。官方骨干对齐、实现移植通过局部测试、原论文完整实验复现是不同层次。尤其不能用当前PriceFM成绩宣称优于完整跨区预训练基础模型。')

h('需要补哪些基线论文')
p('P0表示当前研究主张所需的优先对照，P1为第二批，P2为目标扩展或高成本项目；这是结合本项目资源的综合判断，不等于论文质量排名。完整论文题名、链接、代码可用性和实施要求另见Excel的“需补基线论文”。')
for b in BASELINES:
    h(f"{b['priority']} · {b['models']}（{', '.join(b['ids'])}）")
    p(b['why']+' '+b['action']+'。')
    p('对齐要求：'+b['protocol']+' 成本：'+b['cost']+'。风险：'+b['risk'])
p('另保留日/周季节性Naive和B+简单残差校正作为便宜机制对照。周季节项也需要168h信息，在扩展信息轨道统一提供；不要只给某个基线额外数据。无需把所有2026年分解+群智能调参混合网络都加入：优先覆盖强统计、树后校准、外生神经、预训练和检索机制。')

h('新结构：从哪个问题出发、怎样证伪')
for a in ARCHITECTURES:
    h(a['name']+f"（{', '.join(a['ids'])}）")
    p(a['idea']+' '+a['why'])
    p('改动：'+a['location']+'。'+a['formula'])
    p('对照：'+a['experiment'])
    p('证据与风险：'+a['evidence']+' '+a['risk'])
    p('采纳条件：'+a['decision'])
p('关于创新性：历史检索与外生条件的组合已有相关研究，P30和J21（目前仅摘要）尤其接近。我们的贡献需要落在明确的因果信息协议、校正或检索机制，以及超过强对照的可验证增益，不能仅凭组合了这些模块就宣称新颖。')
p('不建议直接把当前损失替换成Chung校准损失。P16中NHITS+QRA pinball的CRPS为16.87，Chung为21.74，且两组同时更换线性/神经头；这既没有证明替换有效，也不是隔离损失效应的纯对照。')

h('下一轮实验应该如何组织')
p('第一步，冻结两条协议：A为当前NEM滚动24h轨道，按72h公共输入与长期成熟信息分别报告；B为文献对齐轨道，例如P20的2025H1半小时24h，或epftoolbox的日前任务。各轨道单独排名。')
p('第二步，先加入等信息的树后校准、正式LEAR/DDNN-JSU和一个NBEATSx/NHITS强神经基线；TabPFN、Chronos-2/CING-LEAR与XMatch按代码和预训练审计结果推进。所有方法记录调参、重训、上下文和推断成本。')
p('第三步，只改变一个机制：以balanced、仅位置为对照测试非对称单调头，再单独测试支持度门控。相同B、历史池、总训练预算和选择规则；采用多种子并保留全部地区/季节门槛，不因本次观察临时放宽。')
p('第四步，2023–2024已经看过，后续可继续作回顾性诊断但不能叫盲测。正式新结论需要预先冻结的未来留出或尚未参与选择的数据；统计检验应处理重叠预测的时间相关，报告差异的不确定性而非只报名次。')

h('证据、版本与交付文件')
p('109条全部有逐项记录。摘要或元数据不可获得的切分、误差、代码被留作“未核”，没有估算名次。部分论文有日期/版本和实验描述冲突，已逐行记录；风险标记不等于认定泄漏或不当行为。原附件保留为独立工作表。')
p('审阅覆盖：'+json.dumps(counts,ensure_ascii=False)+'；可比性分类：'+json.dumps(comp_counts,ensure_ascii=False)+'。')
selected=sorted(set(spot_ids+[i for b in BASELINES for i in b['ids']]+[i for a in ARCHITECTURES for i in a['ids']]))
for i in selected:
    r=review[i]; p(f"[{i}] {r['title_verified']}。"+source_summary(r))
for i,title,url,loc in EXTRA_SOURCES:
    p(f'[{i}] {title}。{url}；{loc}')
p('本地可复核：'+str(OUT/'local_audit.json')+'；'+str(REPO/'logs/forecasting/v5_joint_history_20261020_final_seed2026_r1/report.md')+'；'+str(REPO/'logs/forecasting/v5_anchor_architecture_20261022/interpretation.md'))

title='2026电价论文：我们的定位、补充基线与结构选择'
html_parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
'''<style>body{font-family:system-ui,"Noto Sans CJK SC",sans-serif;background:#f3f6f8;color:#243746;margin:0;line-height:1.75}main{max-width:1120px;margin:auto;padding:40px;background:white}h1{color:#17324d;font-size:30px}h2{color:#087f8c;font-size:21px;margin-top:36px;border-bottom:1px solid #dbe5ea;padding-bottom:8px}p{margin:12px 0}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.6}th{background:#17324d;color:white;text-align:left}td,th{padding:9px;border:1px solid #dbe5ea;vertical-align:top}tr:nth-child(even){background:#f3f7fa}.table-wrap{overflow:auto}a{color:#087f8c;overflow-wrap:anywhere}@media print{body{background:white}main{padding:0}h2{break-after:avoid}tr{break-inside:avoid}table{font-size:10px}}</style><main>''',
f'<h1>{html.escape(title)}</h1>']


def linkify(s):
    pattern=r'https?://[^\s；，。|]+'
    parts=[];last=0
    for m in re.finditer(pattern,s):
        parts.append(html.escape(s[last:m.start()]))
        u=m.group();parts.append(f'<a href="{html.escape(u,quote=True)}">{html.escape(u)}</a>');last=m.end()
    parts.append(html.escape(s[last:]));return ''.join(parts).replace('\n','<br>')


doc=Document()
doc.core_properties.title=title
doc.core_properties.subject='109条文献逐项核验；本地预测重评分；基线与架构建议'
sec=doc.sections[0]
sec.top_margin=Inches(.65);sec.bottom_margin=Inches(.65)
sec.left_margin=Inches(.65);sec.right_margin=Inches(.65)
for style_name in ['Normal','Title','Heading 1','Heading 2']:
    st=doc.styles[style_name];st.font.name='Arial';st._element.rPr.rFonts.set(qn('w:eastAsia'),'微软雅黑')
doc.styles['Normal'].font.size=Pt(10)
doc.styles['Normal'].paragraph_format.space_after=Pt(7)
doc.add_heading(title,0)
for typ,val in blocks:
    if typ=='h':
        html_parts.append(f'<h2>{html.escape(val)}</h2>');doc.add_heading(val,1)
    elif typ=='p':
        html_parts.append(f'<p>{linkify(val)}</p>');doc.add_paragraph(val)
    else:
        headers,rows=val
        html_parts.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>')
        table=doc.add_table(rows=1,cols=len(headers));table.style='Light Shading Accent 1'
        for i,x in enumerate(headers):table.rows[0].cells[i].text=str(x)
        repeat=OxmlElement('w:tblHeader');table.rows[0]._tr.get_or_add_trPr().append(repeat)
        for row in rows:
            html_parts.append('<tr>'+''.join('<td>'+linkify(str(x))+'</td>' for x in row)+'</tr>')
            cells=table.add_row().cells
            for i,x in enumerate(row):
                cells[i].text=str(x)
                for para in cells[i].paragraphs:
                    for run in para.runs:run.font.size=Pt(8)
        html_parts.append('</tbody></table></div>')
        doc.add_paragraph('')
html_parts.append('</main></html>')
HTML_PATH.write_text('\n'.join(html_parts),encoding='utf-8')
doc.save(DOCX_PATH)

# Re-open independently to check row coverage, numerical cells, formula injection,
# hyperlinks, archive structure, and all expected outputs before declaring completion.
check=load_workbook(WB_PATH,data_only=False)
assert len(check['109篇论文结果总表'].tables)==1
ids=[check['109篇论文结果总表'].cell(i,1).value for i in range(5,114)]
assert len(ids)==109 and set(ids)==set(orig)
assert check['本地同样本排名'].max_row==21
assert check['原始109条'].max_row==113
assert all(c.data_type!='f' for ws in check for row in ws for c in row), 'Unexpected spreadsheet formula'
assert len(Document(DOCX_PATH).paragraphs)>80
baseline_check=load_workbook(BASELINE_PATH,data_only=False)
assert baseline_check['需补基线论文'].max_row==4+len(BASELINES)
assert baseline_check['涉及论文核验'].max_row==4+len(recommended_ids)
assert '<html lang="zh-CN">' in HTML_PATH.read_text()
validation={
    'passed':True,'input_records':109,'review_records':len(reviews),'evidence_counts':counts,
    'comparability_counts':comp_counts,'australian_records':len(au_rows),'evidence_links':len(evidence_rows),
    'local_prediction_artifacts_rescored':51,'local_output_count':17,'baseline_recommendation_groups':len(BASELINES),
    'architecture_proposals':len(ARCHITECTURES),'sheet_names':check.sheetnames,
    'outputs':[{ 'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in [WB_PATH,BASELINE_PATH,DOCX_PATH,HTML_PATH]],
    'assertions':['all109IDsExactlyOnce','noExternalDirectRanking','localAuditPassed','excelReopened','docxReopened','noFormulaInjection'],
}
(OUT/'review_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
(OUT/'all_reviews.json').write_text(json.dumps(reviews,ensure_ascii=False,indent=2))
(OUT/'baseline_recommendations.json').write_text(json.dumps(baseline_rows,ensure_ascii=False,indent=2))
print(json.dumps(validation,ensure_ascii=False,indent=2))
