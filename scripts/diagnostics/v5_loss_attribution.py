"""Read frozen forecasts only. Label-aware oracles are diagnostic, never models.

Run from the repository root with .venv/bin/python scripts/diagnostics/v5_loss_attribution.py.
All groupings and oracle controls are retrospective descriptive analyses.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/forecasting/v5_loss_attribution_20261019"
RUN = ROOT / "outputs/forecasting/v5_conditional_event_transfer_20261017_r1"
PARENT = ROOT / "outputs/forecasting/v5_calibrated_history_20261015_r3/development/predictions"
REGIONS = ("NSW1", "QLD1", "TAS1")
LEVELS = np.array([.05, .10, .50, .90, .95])
AEST = timezone(timedelta(hours=10))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def verify_freeze():
    frozen = json.loads((OUT / "freeze.json").read_text())
    for relative, expected in frozen["files"].items():
        assert sha(ROOT / relative) == expected["sha256"], relative
    return frozen


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def aligned(reference, other):
    for key in ("origin_unix", "target_time_unix", "actual", "quantile_levels"):
        assert np.array_equal(reference[key], other[key]), key
    assert np.array_equal(reference["quantile_levels"], LEVELS)
    assert np.isfinite(other["quantiles"]).all()
    assert (np.diff(other["quantiles"], axis=-1) >= 0).all()


def qs(y, q):
    error = y[..., None] - q
    return 2 * np.maximum(LEVELS * error, (LEVELS - 1) * error).mean(axis=-1)


def score(y, q):
    return {"mae": float(np.abs(y - q[..., 2]).mean()), "qs": float(qs(y, q).mean())}


def bands(y):
    return {"negative": y < 0, "ordinary_0_300": (y >= 0) & (y <= 300), "spike_above_300": y > 300}


def effect(y, before, after, mask):
    n = int(mask.sum())
    if not n:
        return {"cells": 0, "mae_contribution": 0., "qs_contribution": 0.}
    d = np.abs(y - after[..., 2]) - np.abs(y - before[..., 2])
    dq = qs(y, after) - qs(y, before)
    return {"cells": n, "mae_before": float(np.abs(y - before[..., 2])[mask].mean()),
            "mae_after": float(np.abs(y - after[..., 2])[mask].mean()),
            "conditional_mae_delta": float(d[mask].mean()),
            "mae_contribution": float(d[mask].sum() / y.size),
            "qs_contribution": float(dq[mask].sum() / y.size),
            "improved_fraction": float((d[mask] < -1e-9).mean()),
            "worsened_fraction": float((d[mask] > 1e-9).mean())}


def compare(y, before, after, groups):
    return {name: effect(y, before, after, mask) for name, mask in groups.items()}


def interval_oracle(y, anchor, lower, upper):
    """Exact per-cell q50 MAE optimum in an allowed correction interval."""
    assert np.all(lower <= upper + 1e-12)
    return anchor + np.clip(y - anchor, lower, upper)


def pulse_payload(memory, indices):
    widths = memory["event_widths"][indices]
    valid = memory["event_record_indices"][indices] >= 0
    assert np.all(widths[valid] > 0)
    safe_widths = np.where(valid, widths, 1.)
    centers = memory["event_centers"][indices]
    pulses = memory["event_amplitudes"][indices, :, None] * np.exp(
        -.5 * ((np.arange(24)[None, None] - centers[..., None]) / safe_widths[..., None]) ** 2)
    return np.where(valid[..., None], pulses, 0.) * memory["event_weights"][indices, :, None]


def subset_oracle(y, anchor, trend_term, gate_event, payload):
    """Choose one of 16 top-four event subsets per origin using all 24 labels.

    Retain original weights (no renormalization), geometry, learned gate and trend.
    Empty/full subsets are included. This is hindsight, not a deployable selector.
    """
    choices = []
    for subset in range(16):
        selected = np.array([(subset >> k) & 1 for k in range(4)])
        event = np.clip((payload * selected[None, :, None]).sum(axis=1), 0, 150)
        choices.append(anchor + np.clip(trend_term + gate_event * event, -150, 150))
    choices = np.stack(choices, axis=1)
    losses = np.abs(y[:, None, :] - choices).mean(axis=2)
    best = losses.argmin(axis=1)
    return choices[np.arange(len(y)), best], best


def monthly_anchor(region):
    b = load(PARENT / "B" / f"{region}_v1+v2.npz")
    c = load(PARENT / "C_roll" / f"{region}_v1+v2.npz")
    # Legacy raw artifacts omit metadata; explicitly reconstruct the frozen
    # horizon grid and levels, then verify against current canonical artifacts.
    for name, value in (("B", b), ("C_roll", c)):
        value['target_time_unix'] = value['origin_unix'][:, None] + np.arange(24)[None] * 3600
        value['quantile_levels'] = LEVELS
        canonical = load(RUN / 'predictions' / f'{region}_{name}.npz')
        ix = np.searchsorted(value['origin_unix'], canonical['origin_unix'])
        for key in ('origin_unix', 'target_time_unix', 'actual', 'quantiles'):
            assert np.array_equal(value[key][ix], canonical[key]), (name, key)
    aligned(b, c)
    result = {}
    for month in (10, 11, 12):
        start = int(datetime(2022, month, 1, tzinfo=AEST).timestamp())
        end = int(datetime(2023, 1, 1, tzinfo=AEST).timestamp()) if month == 12 else int(datetime(2022, month + 1, 1, tzinfo=AEST).timestamp())
        # Match latest audit's conservative label-availability boundary.
        rows = (b["origin_unix"] >= start) & (b["origin_unix"] + 24 * 3600 < end)
        y, qb, qc = b["actual"][rows], b["quantiles"][rows], c["quantiles"][rows]
        result[str(month)] = {"origins": int(rows.sum()), "B": score(y, qb), "C_roll": score(y, qc),
                              "bands": compare(y, qb, qc, bands(y))}
    return result


def training_replay(region, memory, november_gate, november_indices):
    """Apply already-trained saved gates to October; no optimizer or selection.

    October is an in-sample fit diagnostic, NEVER a validation score.
    """
    head = json.loads((RUN / 'models' / f'{region}_TE_full.json').read_text())
    gates = np.clip(memory['features'] @ np.asarray(head['weight']).T + np.asarray(head['bias']), 0, 1)
    np.testing.assert_allclose(gates[november_indices, :, 0], november_gate['gate_trend'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(gates[november_indices, :, 1], november_gate['gate_event'], rtol=0, atol=1e-12)
    anchor = load(PARENT / 'C_roll' / f'{region}_v1+v2.npz')
    start = int(datetime(2022, 10, 1, tzinfo=AEST).timestamp())
    cutoff = int(datetime(2022, 11, 1, tzinfo=AEST).timestamp())
    rows = (anchor['origin_unix'] >= start) & (anchor['origin_unix'] + 86400 < cutoff)
    orig, y, q0 = anchor['origin_unix'][rows], anchor['actual'][rows], anchor['quantiles'][rows]
    ix = np.searchsorted(memory['origin_unix'], orig)
    assert np.array_equal(memory['origin_unix'][ix], orig)
    assert len(ix) == 720
    t = gates[ix, :, 0] * memory['trend'][ix]
    e = gates[ix, :, 1] * memory['event'][ix]
    qt = q0 + np.clip(t, -150, 150)[..., None]
    qf = q0 + np.clip(t+e, -150, 150)[..., None]
    return {'scope': 'October in-sample diagnostic of saved fitted head; NOT validation',
            'origins': len(ix), 'anchor': score(y, q0), 'trend': score(y, qt), 'full': score(y, qf),
            'trend_mae_delta': float(np.abs(y-qt[..., 2]).mean()-np.abs(y-q0[..., 2]).mean()),
            'event_mae_delta': float(np.abs(y-qf[..., 2]).mean()-np.abs(y-qt[..., 2]).mean()),
            'mean_trend_gate': float(gates[ix, :, 0].mean()), 'mean_event_gate': float(gates[ix, :, 1].mean())}


def analyze_region(region):
    names = ("B", "C_roll", "TE_full", "TE_event_off", "TE_time_shift_off")
    artifacts = {name: load(RUN / "predictions" / f"{region}_{name}.npz") for name in names}
    ref = artifacts["B"]
    for value in artifacts.values():
        aligned(ref, value)
    q = {name: value["quantiles"] for name, value in artifacts.items()}
    y, origins = ref["actual"], ref["origin_unix"]
    memory = load(RUN / "features" / f"{region}_full.npz")
    gate = load(RUN / "gates" / f"{region}_TE_full.npz")
    assert np.array_equal(origins, gate["origin_unix"])
    indices = np.searchsorted(memory["origin_unix"], origins)
    assert np.array_equal(memory["origin_unix"][indices], origins)
    trend, event = memory["trend"][indices], memory["event"][indices]
    gt, ge = gate["gate_trend"], gate["gate_event"]
    assert ((gt >= 0) & (gt <= 1) & (ge >= 0) & (ge <= 1)).all()
    assert (event >= 0).all()
    trend_term = gt * trend
    q["event_only_fixed_gate"] = q["C_roll"] + np.clip(ge * event, -150, 150)[..., None]
    replay = q["C_roll"] + np.clip(trend_term + ge * event, -150, 150)[..., None]
    np.testing.assert_allclose(replay, q["TE_full"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(q["C_roll"] + np.clip(trend_term, -150, 150)[..., None], q["TE_event_off"], rtol=0, atol=1e-9)
    for intervention in ("TE_event_off", "TE_time_shift_off"):
        g = load(RUN / "gates" / f"{region}_{intervention}.npz")
        assert np.array_equal(g["origin_unix"], origins)
        assert np.array_equal(g["gate_trend"], gt) and np.array_equal(g["gate_event"], ge)
    payload = pulse_payload(memory, indices)
    np.testing.assert_allclose(np.clip(payload.sum(axis=1), 0, 150), event, rtol=0, atol=1e-9)
    # Independently check every selected candidate precedes the query by >=24h.
    maturity_counts = {}
    for key in ("event_record_indices", "trend_ids"):
        ids = memory[key][indices]
        valid = ids >= 0
        now = origins.reshape((len(origins),) + (1,) * (ids.ndim - 1))
        assert np.all(np.broadcast_to(now, ids.shape)[valid] >= memory["origin_unix"][ids[valid]] + 86400)
        maturity_counts[key] = int(valid.sum())
    groups = {"all": np.ones(y.shape, bool), **bands(y)}
    day = np.array([datetime.fromtimestamp(int(t), AEST).day for t in origins])
    for lo, hi in ((1, 7), (8, 14), (15, 21), (22, 30)):
        groups[f"origin_Nov_{lo:02}_{hi:02}"] = np.broadcast_to(((day >= lo) & (day <= hi))[:, None], y.shape)
    for lo in (0, 6, 12, 18):
        groups[f"lead_{lo}_{lo+5}"] = np.broadcast_to((np.arange(24) >= lo) & (np.arange(24) < lo + 6), y.shape)
    b50 = q["B"][..., 2]
    for name, mask in (("B_negative", b50 < 0), ("B_0_100", (b50 >= 0) & (b50 <= 100)),
                       ("B_100_300", (b50 > 100) & (b50 <= 300)), ("B_above_300", b50 > 300)):
        groups[name] = mask
    # z column6 = spare capacity, already standardized/clipped then divided by3.
    # These thresholds mean +/-1 training standard deviation, not physical MW.
    spare = memory["features"][indices, :, 6]
    groups.update(spare_low=spare < -1/3, spare_mid=(spare >= -1/3) & (spare <= 1/3), spare_high=spare > 1/3)
    comparisons = {"anchor": ("B", "C_roll"), "trend": ("C_roll", "TE_event_off"),
                   "event": ("TE_event_off", "TE_full"), "total": ("B", "TE_full"),
                   "time_movement": ("TE_time_shift_off", "TE_full")}
    effects = {key: compare(y, q[a], q[b], groups) for key, (a, b) in comparisons.items()}
    for metric in ("mae_contribution", "qs_contribution"):
        np.testing.assert_allclose(sum(effects[s]["all"][metric] for s in ("anchor", "trend", "event")), effects["total"]["all"][metric], atol=1e-10)
        for s in comparisons:
            np.testing.assert_allclose(sum(effects[s][k][metric] for k in bands(y)), effects[s]["all"][metric], atol=1e-10)
    losses = {name: np.abs(y - value[..., 2]) for name, value in q.items()}
    l0, lt, le, lf = (losses[name] for name in ("C_roll", "TE_event_off", "event_only_fixed_gate", "TE_full"))
    shapley_t, shapley_e = .5 * (lt - l0 + lf - le), .5 * (le - l0 + lf - lt)
    np.testing.assert_allclose(shapley_t + shapley_e, lf - l0, atol=1e-10)
    shapley = {"trend": float(shapley_t.mean()), "event": float(shapley_e.mean()),
               "interaction": float((lf - lt - le + l0).mean())}
    a50 = q["C_roll"][..., 2]
    # Exact hindsight interval optima; each cell gets its own label-aware gate.
    oracle_preds = {
        "event_gate_given_fixed_trend": interval_oracle(y, a50, np.clip(trend_term, -150, 150), np.clip(trend_term + event, -150, 150)),
        "trend_gate_only": interval_oracle(y, a50, np.minimum(trend, 0), np.maximum(trend, 0)),
        "joint_trend_event_gates": interval_oracle(y, a50, np.clip(np.minimum(trend, 0), -150, 150), np.clip(np.maximum(trend, 0) + event, -150, 150)),
    }
    subset_prediction, subset_ids = subset_oracle(y, a50, trend_term, ge, payload)
    oracle_preds["top4_subset_per_origin_fixed_gates"] = subset_prediction
    np.testing.assert_array_less(np.abs(y-oracle_preds["event_gate_given_fixed_trend"]), lf + 1e-8)
    np.testing.assert_array_less(np.abs(y-oracle_preds["joint_trend_event_gates"]), np.minimum(l0, lf) + 1e-8)
    assert np.abs(y-subset_prediction).mean() <= lf.mean() + 1e-9
    oracles = {name: {"mae": float(np.abs(y-pred).mean()),
                     "bands": {band: {"cells": int(mask.sum()),
                                        "mae_contribution": float(np.abs(y-pred)[mask].sum()/y.size),
                                        "conditional_mae": float(np.abs(y-pred)[mask].mean()) if mask.any() else None}
                               for band, mask in bands(y).items()}}
               for name, pred in oracle_preds.items()}
    oracles["top4_subset_per_origin_fixed_gates"]["selected_subset_counts"] = np.bincount(subset_ids, minlength=16).tolist()
    proposal_diagnostics = {}
    dt = np.clip(trend_term, -150, 150)
    de = q["TE_full"][..., 2] - q["TE_event_off"][..., 2]
    for name, mask in groups.items():
        if not mask.any():
            continue
        need = y - q["TE_event_off"][..., 2]
        wrong_t = ((trend_term > 1e-9) & (y-a50 < 0)) | ((trend_term < -1e-9) & (y-a50 > 0))
        active_t = np.abs(trend_term) > 1e-9
        active_e = de > 1e-9
        proposal_diagnostics[name] = {
            "mean_signed_trend_correction": float(dt[mask].mean()),
            "mean_event_correction": float(de[mask].mean()),
            "mean_anchor_underprediction": float((y-a50)[mask].mean()),
            "active_trend_wrong_direction_fraction": float(wrong_t[mask & active_t].mean()) if (mask & active_t).any() else None,
            "active_event_on_already_overpredicted_fraction": float((need[mask & active_e] < 0).mean()) if (mask & active_e).any() else None,
            "event_proposal_above_1_fraction": float((event[mask] > 1).mean()),
            "selected_unweighted_atom_above_1_fraction": float((np.max(np.divide(payload, memory['event_weights'][indices, :, None], out=np.zeros_like(payload), where=memory['event_weights'][indices, :, None]>0), axis=1)[mask] > 1).mean()),
        }
    return {"origins": len(origins), "scores": {name: score(y, value) for name, value in q.items()},
            "effects": effects, "shapley_mae": shapley, "oracles_NOT_DEPLOYABLE": oracles,
            "proposal_diagnostics": proposal_diagnostics,
            "unique_spike_hours": int(len(np.unique(ref['target_time_unix'][y > 300]))),
            "monthly_anchor": monthly_anchor(region),
            "October_IN_SAMPLE_replay": training_replay(region, memory, gate, indices),
            "checks": {"grid_equal": True, "full_replay": True, "event_payload_replay": True,
                       "same_head_gates": True, "additivity": True, "oracle_bounds": True,
                       "candidate_maturity_violations": 0, "candidate_references": maturity_counts}}


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"]*len(headers)) + "|"] + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]


def report(result):
    regions = result["regions"]
    lines = ["# V5 冻结版本：损失来源诊断", "", f"冻结提交：`{result['frozen_commit']}`；当前诊断未重训、未选择模型、未读取最终测试。",
             "", "最新实验为 2022 年 11 月回顾性开发评价，三个地区 seed 2026、每地区696个起报窗口。全部误差在原始价格空间计算；正的误差变化表示变差。", "",
             "诊断结论：11月的主要损失来自历史趋势修正，其次是事件修正；在本轮宏平均下C_roll本身略有收益。历史提案存在事后可用空间，但现有采用规则没有实现该空间；不能由oracle成绩推导出状态特征足以学习理想决策。", "",
             "## 1. 同参数逐层归因", "", "趋势使用完整模型原有门控，事件通过同权重关闭事件的干预计算；不是另拟合的 T_only。三项变化精确加总为 Full−LightGBM。", ""]
    rows = []
    for region, r in regions.items():
        rows.append([region] + [f"{r['scores'][m]['mae']:.6f}" for m in ('B','C_roll','TE_event_off','TE_full')] +
                    [f"{r['effects'][s]['all']['mae_contribution']:+.6f}" for s in ('anchor','trend','event','total')])
    rows.append(["宏平均"] + [f"{np.mean([r['scores'][m]['mae'] for r in regions.values()]):.6f}" for m in ('B','C_roll','TE_event_off','TE_full')] +
                [f"{np.mean([r['effects'][s]['all']['mae_contribution'] for r in regions.values()]):+.6f}" for s in ('anchor','trend','event','total')])
    lines += table(["地区","LightGBM","C_roll","加趋势","加事件 Full","基础校正Δ","趋势Δ","事件Δ","总Δ"], rows)
    lines += ["", "## 2. 价格区间贡献", "", "以下为对全样本 MAE 的贡献，已乘区间占比；不能与尖峰条件 MAE 混用。", ""]
    rows = []
    for region, r in regions.items():
        for band in ('negative','ordinary_0_300','spike_above_300'):
            rows.append([region,band,r['effects']['total'][band]['cells']] + [f"{r['effects'][s][band]['mae_contribution']:+.6f}" for s in ('anchor','trend','event','total')])
    lines += table(["地区","区间","预测单元数","基础校正Δ","趋势Δ","事件Δ","总Δ"],rows)
    lines += ["", "## 3. 基础校正是否随月份反转", "", "每月只保留 origin+24h 严格早于次月的窗口，10/11/12月分别为720/696/720个。与旧报告12月721个窗口的边界不同。未在12月运行最新Full。", ""]
    rows = []
    for region,r in regions.items():
        for month,m in r['monthly_anchor'].items():
            rows.append([region,month,m['origins'],f"{m['B']['mae']:.6f}",f"{m['C_roll']['mae']:.6f}",f"{m['C_roll']['mae']-m['B']['mae']:+.6f}"])
    lines += table(["地区","2022月份","窗口数","LightGBM MAE","C_roll MAE","变化"],rows)
    lines += ["", "### 10月拟合段与11月评价段的差异", "", "使用已保存的门控参数重放10月，不重新训练。10月是训练内成绩，不能与11月一起宣称样本外收益。", ""]
    lines += table(["地区","10月趋势ΔMAE（训练内）","11月趋势ΔMAE","10月事件ΔMAE（训练内）","11月事件ΔMAE"],
                   [[region,f"{r['October_IN_SAMPLE_replay']['trend_mae_delta']:+.6f}",f"{r['effects']['trend']['all']['mae_contribution']:+.6f}",f"{r['October_IN_SAMPLE_replay']['event_mae_delta']:+.6f}",f"{r['effects']['event']['all']['mae_contribution']:+.6f}"] for region,r in regions.items()])
    lines += ["", "## 4. 改变归因顺序的检查", "", "Shapley归因平均‘先趋势后事件’和‘先事件后趋势’两种顺序，保持全部模型参数不变；它是预测干预归因，不是市场因果效应。", ""]
    lines += table(["地区","趋势Shapley ΔMAE","事件Shapley ΔMAE","交互项"],[[region]+[f"{r['shapley_mae'][k]:+.6f}" for k in ('trend','event','interaction')] for region,r in regions.items()])
    lines += ["", "## 5. 理想选择上限——使用真实标签，不能部署或列入模型榜单", "",
              "逐单元oracle利用真实未来价格选择[0,1]门控，给出乐观的可改善空间上限（MAE下界），只回答已有提案有没有潜在纠错方向和幅度，不证明可预测。随机方向的提案也可能在事后选择中获益，因此该结果不能证明历史信息本身的增量价值。没有用oracle训练或选择任何模型。", "",
              "top4子集oracle对每个origin的整条24小时曲线，从16个事件子集中选择最优；保持原候选、幅度、位置、权重（不重新归一化）、门控和趋势。它只检查已检索top4，不能代表整个历史候选库。", ""]
    oracle_names = ('event_gate_given_fixed_trend','trend_gate_only','joint_trend_event_gates','top4_subset_per_origin_fixed_gates')
    lines += table(["地区","C_roll","当前Full","理想事件门控/固定趋势","理想趋势门控/无事件","理想双门控","理想top4子集/固定门控"],
                   [[region,f"{r['scores']['C_roll']['mae']:.6f}",f"{r['scores']['TE_full']['mae']:.6f}"] + [f"{r['oracles_NOT_DEPLOYABLE'][k]['mae']:.6f}" for k in oracle_names] for region,r in regions.items()])
    lines += ["", "## 6. 提案方向和事件覆盖", "", "方向错误表示已采用趋势与基础预测所需修正方向相反；事件误用表示采用正事件时，无事件模型已经高估。覆盖阈值固定为修正提案>1 AUD/MWh，仅供诊断。", ""]
    rows=[]
    for region,r in regions.items():
        d=r['proposal_diagnostics']['all']; spike=r['proposal_diagnostics']['spike_above_300']
        rows.append([region,f"{d['mean_signed_trend_correction']:.4f}",f"{100*d['active_trend_wrong_direction_fraction']:.2f}%",f"{100*d['active_event_on_already_overpredicted_fraction']:.2f}%",f"{100*spike['event_proposal_above_1_fraction']:.2f}%",r['unique_spike_hours']])
    lines += table(["地区","平均趋势修正","采用趋势方向错误率","采用事件时已高估比例","真实尖峰处事件提案覆盖","独立尖峰小时"],rows)
    lines += ["", "## 7. 时间移动与状态分组", ""]
    lines += table(["地区","移动导致总MAE变化","移动导致尖峰条件MAE变化"],[[region,f"{r['effects']['time_movement']['all']['conditional_mae_delta']:+.6f}",f"{r['effects']['time_movement']['spike_above_300']['conditional_mae_delta']:+.6f}"] for region,r in regions.items()])
    lines += ["", "完整逐周、预测步长、基础价格水平、备用容量状态分组见 attribution.json。备用容量阈值为标准化后的±1个训练期标准差，不是物理MW阈值；全部分组为事后描述，不作显著性结论。", "",
              "## 8. 审计范围和限制", "", "- 冻结文件在分析前后逐一校验；预测网格、标签和分位数顺序一致。",
              "- 完整预测和事件脉冲从保存的提案、门控、候选参数重放；所有选中历史满足24小时标签成熟边界。",
              "- 分层MAE/QS及价格区间贡献严格可加；oracle包含原预测或相应零修正边界。",
              "- 历史提案库从2022年10月开始；不是跨年度同日期历史验证。双场只在7–8月训练，采用模块只在10月训练。",
              "- 实际外生数据首次发布时间仍仅有原归档代理审计，不能扩大为已证明完全无泄漏。",
              "- 11月独立尖峰小时仅4/18/6，滚动窗口重复单元不能当独立事件；没有计算或宣称统计显著性。",
              "- 当前oracle只诊断已生成趋势提案、已选择事件及其聚合；没有搜索整个未检索的历史库，没有证明现有状态特征可学习到这些理想决策。", "",
              "## 9. 本轮判断与下一步边界", "",
              "1. 11月宏MAE退化主要来自趋势修正；改变趋势/事件的归因顺序后结论不变。不能只处理尖峰误报而忽略趋势分支。",
              "2. 已保存门控在10月训练段有收益，在11月三个地区均退化。证据支持训练段到评价段的泛化失败，但当前实验无法单独区分过拟合、误差状态漂移、特征不足或训练预算影响。",
              "3. 正事件确实在真实尖峰区间降低了一些误差，但普通时段的代价更大。事件已参与预测，不等于事件产生净收益。",
              "4. 固定部署趋势、仅理想调整已有聚合事件门控时，NSW1和QLD1仍弱于LightGBM。这一有限可行域内，单改事件采用无法修复全部退化。更换事件库、候选或幅度后需重新检验，不能扩大结论。",
              "5. 优先验证趋势误差的符号与幅度能否跨时期预测，再验证事件触发；使用跨季节严格样本外误差库和时间前推验证，而不是直接增加神经网络层数或用11月oracle生成训练目标。",
              "6. 本步骤只完成冻结和诊断。未更换默认模型、未重训、未运行跨年度历史库、未启动最终评价。", ""]
    (OUT / "report.md").write_text("\n".join(lines))


def main():
    frozen = verify_freeze()
    result = {"frozen_commit": frozen['frozen_commit'], "diagnostic_code_sha256": sha(Path(__file__)),
              "oracle_warning": "Hindsight labels used. Not deployable predictions or model selection.",
              "regions": {region: analyze_region(region) for region in REGIONS}}
    verify_freeze()
    result['freeze_verified_before_and_after'] = True
    (OUT / "attribution.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report(result)
    print(OUT / "report.md")
    for region,r in result['regions'].items():
        print(region, 'stage_deltas', {k:r['effects'][k]['all']['mae_contribution'] for k in ('anchor','trend','event','total')})
        print('ORACLES', {k:v['mae'] for k,v in r['oracles_NOT_DEPLOYABLE'].items()})


if __name__ == "__main__":
    main()
