"""Score frozen main-table forecasts on R12 targets; no model execution."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ('NSW1', 'QLD1', 'TAS1')
TAUS = np.array([.05, .1, .5, .9, .95])
METRICS = ('mae', 'qs', 'rmse', 'c80', 'c90', 'width80', 'width90')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def align(origins, targets, actual, requested, truth):
    if origins.ndim != 1 or np.any(np.diff(origins) <= 0):
        raise ValueError('source origins must be unique and increasing')
    idx = np.searchsorted(origins, requested)
    if np.any(idx >= len(origins)) or not np.array_equal(origins[idx], requested):
        raise ValueError('missing requested origins')
    if not np.array_equal(targets[idx], requested[:, None] + np.arange(24) * 3600):
        raise ValueError('target times differ')
    selected = actual[idx]
    if not (np.array_equal(selected, truth) or
            np.array_equal(selected, truth.astype(np.float32).astype(np.float64))):
        raise ValueError('labels differ beyond exact float32 rounding')
    return idx, float(np.max(np.abs(selected - truth)))


def score(y, quantiles=None, point=None):
    result = dict.fromkeys(METRICS)
    if quantiles is not None:
        if quantiles.shape != (*y.shape, 5) or not np.isfinite(quantiles).all():
            raise ValueError('invalid quantile predictions')
        q = np.sort(quantiles, axis=-1)
        point = q[..., 2]
        error = y[..., None] - q
        result['qs'] = float(2 * np.maximum(TAUS * error, (TAUS - 1) * error).mean())
        for level, lo, hi in ((80, 1, 3), (90, 0, 4)):
            result[f'c{level}'] = float(((y >= q[..., lo]) & (y <= q[..., hi])).mean())
            result[f'width{level}'] = float((q[..., hi] - q[..., lo]).mean())
    if point is None or point.shape != y.shape or not np.isfinite(point).all():
        raise ValueError('invalid point predictions')
    result.update(mae=float(np.abs(y - point).mean()), rmse=float(np.sqrt(np.mean((y - point) ** 2))))
    return result


def mean_metrics(rows):
    return {key: float(np.mean([r[key] for r in rows])) if all(r[key] is not None for r in rows)
            else None for key in METRICS}


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.monotonic()
    table = ROOT / 'logs/forecasting/main_comparison_20261023'
    latest = ROOT / 'logs/forecasting/agent_totaldemand_information_r12_20261031'
    out = ROOT / 'logs/forecasting/r13_main_table_comparison_20261102'
    out.mkdir(exist_ok=False)
    with (table / 'main.csv').open() as f:
        identities = {r['model']: r for r in csv.DictReader(f)}
    verification = json.loads((table / 'verification.json').read_text())
    sources = [r for r in verification['artifacts'] if r['model'] in identities]
    assert len(sources) == 225
    assert {(r['model'], r['region'], r['seed']) for r in sources} == {
        (m, r, s) for m in identities for r in REGIONS for s in (2026, 2027, 2028)}
    with np.load(latest / 'paired_predictions.npz', allow_pickle=False) as p:
        origins, truth, a, b = (p[k].copy() for k in ('origin_unix', 'actual', 'A', 'B'))
    expected = int(np.datetime64('2022-12-31T14:00:00', 's').astype('int64')) + np.arange(365) * 86400
    assert np.array_equal(origins, expected) and truth.shape == (365, 3, 24)
    assert np.isfinite(truth).all() and (np.diff(a, axis=-1) >= 0).all() and (np.diff(b, axis=-1) >= 0).all()
    cells, audit = [], []
    for n, record in enumerate(sources, 1):
        path = Path(record['path'])
        digest = sha(path)
        if digest != record['sha256']:
            raise ValueError(f'changed frozen source: {path}')
        r = REGIONS.index(record['region'])
        with np.load(path, allow_pickle=False) as p:
            meta = json.loads(p['metadata_json'].item())
            assert meta['region'] == record['region'] and meta['seed'] == record['seed']
            idx, difference = align(p['origin_unix'], p['target_time_unix'], p['actual'], origins, truth[:, r])
            if identities[record['model']]['table'] == 'Q':
                assert np.array_equal(p['quantile_levels'], TAUS)
                scores = score(truth[:, r], quantiles=p['quantiles'][idx])
            else:
                scores = score(truth[:, r], point=p['native_point'][idx])
        cells.append(dict(model=record['model'], region=record['region'], seed=record['seed'], **scores))
        audit.append(dict(model=record['model'], region=record['region'], seed=record['seed'],
                          path=str(path), sha256=digest, matched_origins=len(idx), label_rounding_max=difference))
        if n % 15 == 0:
            print(f'verified and scored {n}/225 frozen artifacts', flush=True)
    macro, summary = [], []
    for model, ident in identities.items():
        per_seed = []
        for seed in (2026, 2027, 2028):
            selected = [r for r in cells if r['model'] == model and r['seed'] == seed]
            assert len(selected) == 3
            row = dict(model=model, seed=seed, **mean_metrics(selected))
            macro.append(row)
            per_seed.append(row)
        row = dict(model=model, role=ident['role'], seeds=3, **mean_metrics(per_seed))
        row.update({key + '_std': float(np.std([r[key] for r in per_seed], ddof=1))
                    if row[key] is not None else None for key in METRICS})
        summary.append(row)
    controls = []
    for arm, predictions in (('A', a), ('B', b)):
        rows = [dict(model=f'R12_{arm}', region=region, seed=2026,
                     **score(truth[:, r], quantiles=predictions[:, r])) for r, region in enumerate(REGIONS)]
        cells.extend(rows)
        row = dict(model=f'R12_{arm}', role='demand_proxy_candidate' if arm == 'B' else 'control_only',
                   seeds=1, **mean_metrics(rows), **{key + '_std': None for key in METRICS})
        (summary if arm == 'B' else controls).append(row)
    method_models = {m for m in identities if m.startswith('baseline_') or m in
                     ('Full_joint', 'chronos2_zeroshot', 'moirai2_zeroshot', 'timesfm25_xreg', 'tabpfn_v3')}
    method_models.add('R12_B')
    assert len(method_models) == 21
    for row in summary:
        for key in ('mae', 'qs', 'rmse'):
            valid = [r for r in summary if r[key] is not None]
            methods = [r for r in valid if r['model'] in method_models]
            row[key + '_rank_all'] = 1 + sum(r[key] < row[key] for r in valid) if row[key] is not None else None
            row[key + '_rank_methods'] = (1 + sum(r[key] < row[key] for r in methods)
                if row[key] is not None and row['model'] in method_models else None)
    summary.sort(key=lambda r: (r['mae'], r['model']))
    write_csv(out / 'ranking_all_outputs.csv', summary)
    write_csv(out / 'ranking_methods.csv', [r for r in summary if r['model'] in method_models])
    write_csv(out / 'regional_per_seed.csv', cells)
    write_csv(out / 'macro_per_seed.csv', macro)
    write_csv(out / 'control_A.csv', controls)
    report = dict(base_commit='e9c788b',
        execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        script_sha256=sha(__file__), protocol_sha256=sha(ROOT / 'docs/research/agent_forecasting/r13_main_table_comparison_20261102.txt'),
        comparison='same 2023 daily targets; different information and issue-time protocols; exploratory only',
        sources=audit, r12_predictions_sha256=sha(latest / 'paired_predictions.npz'),
        main_table_sha256=sha(table / 'main.csv'), verification_sha256=sha(table / 'verification.json'),
        n_origins_per_region=365, n_regions=3, n_cells_per_system=26280,
        candidate=next(r for r in summary if r['model'] == 'R12_B'), control_A=controls[0],
        rank_counts=dict(all_mae=26, all_qs=24, methods_mae=21, methods_qs=19),
        max_label_rounding_difference=max(r['label_rounding_max'] for r in audit),
        prediction_calls=0, training_calls=0, llm_calls=0, seconds=time.monotonic()-started)
    report['output_sha256'] = {p.name: sha(p) for p in sorted(out.glob('*.csv'))}
    with (out / 'audit.json').open('x') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({k:report[k] for k in ('candidate','rank_counts','max_label_rounding_difference','seconds')}, indent=2))


if __name__ == '__main__':
    main()
