"""Presentation only: derive R9 tables from completed, independently checked results."""
import csv
import json
from pathlib import Path

import numpy as np


def main():
    cfg = json.loads(Path('configs/agent_forecasting/r9_development_20261030.json').read_text())
    root = Path(cfg['output_root'])
    report_dir = Path(__file__).resolve().parent
    summary = json.loads((root / 'results/summary.json').read_text())
    audit = json.loads((root / 'independent_audit.json').read_text())
    matrix = json.loads((root / 'commands/matrix.json').read_text())
    if not audit['passed']:
        raise ValueError('independent result audit must pass before reporting')
    metrics = ('mae', 'quantile_score', 'coverage_80', 'coverage_90', 'width_80', 'width_90')
    records = []
    for result in summary['results']:
        for arm, values in result['arms'].items():
            records.append({'region': result['region'], 'seed': result['seed'], 'arm': arm,
                            **{key: values[key] for key in metrics},
                            'n_origins': values['n_origins'], 'n_cells': values['n_cells'],
                            'decision_failures': values['decision_failures']})
    with (report_dir / 'all_region_seed_metrics.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)

    lines = [
        'R9 已确认缺陷修复与三地区、三个 seed 实验结果',
        f"分支：{cfg['branch']}", f"基点：{cfg['base_commit']}",
        f"预测与评分冻结提交：{summary['execution_commit']}",
        '实现提交：83d5a4c9102a7390f876718e64ad1ea9990b509a。归档提交为添加本报告的 Git commit。',
        '', '主要结论',
        '本轮工程修复验收通过；当前 Agent 修正策略的预测增益不成立。',
        'D 在全部9个地区/seed组合中，MAE和QS均差于基础模型A，也均差于固定规则B。',
        '三地区宏平均再跨seed汇总：D相对A的MAE增加12.47%、QS增加12.79%。',
        'D宏平均优于单次LLM C，但这一点不足以证明比现有数值预测更有效。',
        '全部C/D决策均成功，因此本轮精度退化不能归因于调用失败或失败样本被剔除。',
        '结论限定于当前历史输入、五种有限修正动作的r9策略；本轮未验证实时新公告输入。',
        '三个seed的A/B预测完全一致；D有20/159个地区起报点动作变化，详见seed_stability.txt。',
        '', '修复验收',
        '完整回归：263 passed、89 subtests passed，exit 0；独立核验器另有 1 项合成测试通过。',
        '修复范围：三镜像反例/有限证明范围，完整请求 schema，真实调用期限，原子发布，',
        'seed 传递、D 三次总调用上限、失败不漏样本。本轮没有扩展新预测机制或追加调参。',
        'M1 工程门禁：ENGINEERING_PASS_REDUCIBILITY_REFUTED。194 个有限场景中，',
        '1852 条 D 轨迹、1746 次失败探针，5 条有效 typed witness 来自同一个三镜像场景。',
        '此反例不证明真实 LLM 选择优势；原 M1 metrics=NOT_EVALUATED、advance_m1=false。',
        '外层完整七项终审及 3/3 控制器日志未在旧库找到；本轮保存自己的真实证据，没有重写历史。',
        '', '实验口径',
        '2022 开发集；NSW1 / QLD1 / TAS1；seed 2026 / 2027 / 2028；每格 53 个起报点、24 小时。',
        '合计159个不同地区×起报单元、3816个地区×时点标签；三个seed重复后每臂477次决策，不增加独立样本量。',
        '每地区每 seed 重新运行 A，不复用旧预测。B/C/D 共享对应 A，所有计划起报纳入主指标。',
        'A=冻结 Chronos-2；B=固定规则；C=单次 LLM；D=工具 Agent。无训练或权重下载。',
        'C/D 决策失败保持失败状态，预测保留逐字节 A。表中失败次数是决策失败，不删预测样本。',
        '模型输入与评分标签分开；工具只读当前起报已可见历史。',
        '2023–2024 未评分、未传入预测器；源 CSV 包含该时段，因此不声称其文件字节从未被读取。',
        '本轮沿用 r8 历史数据有限修正机制，不是原 M1 的实时 TOTALDEMAND 公告输入机制。',
        '归档 available_at 是历史可用时间代理，不能据此声称已验证真实在线公告接收时点。',
        'temperature=0，seed 重复不是独立数据集或三个训练模型；标准差只描述本轮重复间变化。',
        'MAE/QS 单位 AUD/MWh，越小越好；QS 为五个分位数 pinball loss 均值的两倍。',
        '', '九组完整结果：每项为 MAE / QS',
        '地区 | seed | A | B | C | D | C失败 | D失败',
    ]
    for result in summary['results']:
        arms = result['arms']
        cells = [f"{arms[a]['mae']:.4f} / {arms[a]['quantile_score']:.4f}" for a in 'ABCD']
        lines.append(' | '.join([result['region'], str(result['seed']), *cells,
                                str(arms['C']['decision_failures']), str(arms['D']['decision_failures'])]))
    lines += ['', '地区三个 seed 均值 ± 样本标准差', '地区 | arm | MAE | QS | 80%覆盖 | 90%覆盖']
    for region in cfg['regions']:
        for arm in 'ABCD':
            samples = [row['arms'][arm] for row in summary['results'] if row['region'] == region]
            cells = []
            for key in metrics[:4]:
                values = [s[key] for s in samples]
                cells.append(f'{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}')
            lines.append(' | '.join([region, arm, *cells]))
    lines += ['', '三地区宏平均，再对三个 seed 汇总', 'arm | MAE | QS | 80%覆盖 | 90%覆盖']
    for arm in 'ABCD':
        cells = [f"{summary['across_seeds'][arm][k]['mean']:.4f} ± {summary['across_seeds'][arm][k]['sample_std']:.4f}"
                 for k in metrics[:4]]
        lines.append(' | '.join([arm, *cells]))
    lines += ['', '配对差值 D - 对照（负值代表 D 更好）', '地区 | seed | D-A MAE/QS | D-B MAE/QS | D-C MAE/QS']
    for result in summary['results']:
        cells = [f"{result['increments']['D-'+a]['mae']:+.4f} / {result['increments']['D-'+a]['quantile_score']:+.4f}"
                 for a in 'ABC']
        lines.append(' | '.join([result['region'], str(result['seed']), *cells]))

    res = audit['resources']
    lines += ['', '实际资源与一致性核验',
        f"完整矩阵墙钟：{matrix['wall_seconds']:.3f} 秒（{matrix['wall_seconds']/60:.2f} 分钟）；36 个阶段均正常退出。",
        f"阶段 CPU 合计：user={res['summed_user_cpu_seconds']:.3f}s，system={res['summed_system_cpu_seconds']:.3f}s。",
        f"单个阶段进程峰值 RSS 最大值：{res['maximum_stage_peak_rss_bytes']/1024**3:.3f} GiB；不是所有并发进程的总峰值。",
        'CPU/RSS 为各预测/决策进程在发布前的采样，含各自导入开销；不含协调器、评分、归档及远端服务。',
        '矩阵墙钟含启动/发布；阶段墙钟重叠，不把其合计当总耗时。完整修复测试另用时721.51秒。',
        f"A 批次：{audit['base_batches']}；LLM 实际 HTTP 尝试：{audit['llm_calls']}，分臂={audit['llm_calls_by_arm']}。",
        f"服务端返回 token：{audit['wire_reported_tokens']}；缺 usage 调用={audit['usage_missing_calls']}。",
        f"服务端原始 model ID 计数：{audit['response_models']}；未返回 model 调用={audit['response_model_missing_calls']}。",
        '本地数值推理 CPU、两核 affinity、Torch 两线程、CUDA 不可见；远端 LLM GPU/功耗/费用未观测，不声称整体零 GPU/零成本。',
        f"独立审计通过：{audit['verified_stage_count']} 个阶段，{audit['metric_comparisons']} 项指标比较，最大绝对误差={audit['maximum_absolute_metric_error']:.3g}。",
        f"决策失败：{audit['decision_failures_by_arm']}；原因：{audit['failure_reasons']}。",
        f"成功动作分布：{audit['successful_actions']}。",
        '', '产物', f'完整原始产物：{root.resolve()}',
        f'全部地区/seed/arm 指标 CSV：{report_dir / "all_region_seed_metrics.csv"}',
        '原始输入、独立标签、36 份预测、逐起报决策、原始 wire、attempt journal、命令和实际退出码均保留。',
        '本报告与结果归档提交到独立分支；未推送，未覆盖旧输出或更改正式主表。',
    ]
    with (report_dir / 'results_cn.txt').open('x') as stream:
        stream.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
