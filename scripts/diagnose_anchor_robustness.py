"""Read-only decomposition of frozen predictions; never fit or select a model."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from forecasting.full_v5_anchor_balance import experiment as e
from forecasting.full_v5_anchor_balance.confirmation_experiment import shape_diagnostics

ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / 'outputs/forecasting/v5_latest_aligned_20261023_r2'
DEVELOPMENT = ROOT / 'outputs/forecasting/v5_anchor_confirmation_20261022_r1'
OUTPUT = ROOT / 'logs/forecasting/v5_calibration_diagnosis_20261023'
LEVELS = np.array(e.LEVELS)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def groups(origins, baseline, actual):
    dates = e.local_dates(origins)
    years = dates.astype('datetime64[Y]').astype(str)
    months = dates.astype('datetime64[M]').astype(str)
    quarters = np.array([m[:4] + 'Q' + str((int(m[-2:])-1)//3+1) for m in months])
    result = {'all': np.ones(actual.shape, dtype=bool)}
    for values in (years, quarters, months):
        for value in np.unique(values):
            result[str(value)] = np.broadcast_to((values == value)[:, None], actual.shape)
    for start in range(0, 24, 6):
        result[f'horizon/{start+1:02d}-{start+6:02d}'] = np.broadcast_to(
            (np.arange(24) >= start) & (np.arange(24) < start+6), actual.shape)
    for name, values in (('tree_q50', baseline[..., 2]), ('actual_diagnostic_only', actual)):
        for low, high in zip((-np.inf, 0, 100, 300, 1000), (0, 100, 300, 1000, np.inf)):
            result[f'{name}/{low}:{high}'] = (values >= low) & (values < high)
    return result


def scores(actual, quantiles):
    y, q = np.asarray(actual, dtype=np.float64), np.asarray(quantiles, dtype=np.float64)
    error = y[..., None] - q
    pb = np.maximum(LEVELS * error, (LEVELS - 1) * error).mean(axis=0)
    return dict(mae=float(np.abs(error[:, 2]).mean()), qs=float(2 * pb.mean()),
        bias=float(-error[:, 2].mean()), rmse=float(np.sqrt(np.square(error[:, 2]).mean())),
        coverage80=float(((y >= q[:, 1]) & (y <= q[:, 3])).mean()),
        coverage90=float(((y >= q[:, 0]) & (y <= q[:, 4])).mean()),
        width80=float((q[:, 3]-q[:, 1]).mean()), width90=float((q[:, 4]-q[:, 0]).mean()),
        lower90=float((q[:, 2]-q[:, 0]).mean()), upper90=float((q[:, 4]-q[:, 2]).mean()),
        below_q05=float((y < q[:, 0]).mean()), above_q95=float((y > q[:, 4]).mean()),
        **{f'pinball_{level:g}': float(value) for level, value in zip(LEVELS, pb)})


def decompose(origins, y, b, a, f):
    """Algebraic width/position interventions are diagnostics, not trained arms."""
    y, b, a, f = (np.asarray(value, dtype=np.float64) for value in (y, b, a, f))
    location = a[..., 2] - b[..., 2]
    correction = f[..., 2] - a[..., 2]
    predictions = dict(B=b, Anchor=a, Full=f,
        Anchor_location_only=b+location[..., None],
        Anchor_scale_only=a-location[..., None],
        Full_tree_width=b+(f[..., 2]-b[..., 2])[..., None])
    for q in predictions.values():
        if q.shape != (*y.shape, 5) or not np.isfinite(q).all() or (np.diff(q, axis=-1) < -1e-8).any():
            raise ValueError('Invalid diagnostic quantiles')
    result = []
    for group, mask in groups(origins, b, y).items():
        if not mask.any():
            continue
        delta, history = location[mask], correction[mask]
        shape = dict(location_mean=float(delta.mean()), location_abs=float(np.abs(delta).mean()),
            history_mean=float(history.mean()), history_abs=float(np.abs(history).mean()),
            opposing_fraction=float((delta*history < 0).mean()))
        base_score = scores(y[mask], b[mask])
        for component, q in predictions.items():
            values = scores(y[mask], q[mask])
            result.append(dict(group=group, component=component, n_cells=int(mask.sum()),
                n_origins=int(mask.any(axis=1).sum()), **values, **shape,
                mae_minus_B=values['mae']-base_score['mae'], qs_minus_B=values['qs']-base_score['qs']))
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    sources, rows, selections, shape = {}, [], [], {}
    def load(path, expected=None):
        sha = e.digest(path)
        if expected is not None and sha != expected:
            raise ValueError(f'Frozen prediction changed: {path}')
        sources[str(path)] = sha
        with np.load(path, allow_pickle=False) as arrays:
            return {k: arrays[k].copy() for k in arrays.files}
    report_path = LATEST / 'report.json'
    report = json.loads(report_path.read_text())
    sources[str(report_path)] = e.digest(report_path)
    for region in ('NSW1', 'QLD1', 'TAS1'):
        for seed in (2026, 2027, 2028):
            arrays = {}
            for label, model in (('B', 'B_retrained'), ('Anchor', 'Anchor_joint'), ('Full', 'Full_joint')):
                record = report[f'{model}/{seed}/{region}']
                arrays[label] = load(Path(record['path']), record['sha256'])
            b = arrays['B']
            for value in arrays.values():
                for key in ('actual', 'origin_unix', 'target_time_unix'):
                    np.testing.assert_array_equal(b[key], value[key])
            y, base, anchor, full = b['actual'], b['quantiles'], arrays['Anchor']['quantiles'], arrays['Full']['quantiles']
            identity = dict(scope='aligned_2023_2024', region=region, seed=seed, mode='guarded')
            rows.extend(dict(identity, **row) for row in decompose(b['origin_unix'], y, base, anchor, full))
            shape[f'aligned/{region}/{seed}'] = shape_diagnostics(y, base, anchor, full)
            path = LATEST / 'fits' / region / f'seed{seed}' / 'selection.json'
            chosen = json.loads(path.read_text())['selection']['guarded']
            sources[str(path)] = e.digest(path)
            for component, key, q in (('B','baseline',base),('Anchor','anchor',anchor),('Full','full',full)):
                actual_score = scores(y.reshape(-1), q.reshape(-1, 5))
                selections.append(dict(identity, component=component, epoch=chosen['epoch'], alpha=chosen['alpha'],
                    selection_mae=chosen[key]['mae'], selection_qs=chosen[key]['qs'],
                    selection_bias=chosen[key]['bias'], selection_coverage80=chosen[key]['coverage_80'],
                    evaluation_mae=actual_score['mae'], evaluation_qs=actual_score['qs'],
                    evaluation_bias=actual_score['bias'], evaluation_coverage80=actual_score['coverage80']))
            for fold in ('2021', '2022'):
                directory = DEVELOPMENT / region / fold / f'seed_{seed}' / 'asymmetric_scale'
                for mode in ('raw', 'guarded'):
                    value = load(directory / f'asymmetric_scale_{mode}.npz')
                    identity = dict(scope='development_'+fold, region=region, seed=seed, mode=mode)
                    rows.extend(dict(identity, **row) for row in decompose(value['origin_unix'], value['actual'],
                        value['baseline'], value['anchor'], value['quantiles']))
    write_csv(OUTPUT / 'per_seed_groups.csv', rows)
    write_csv(OUTPUT / 'selection_vs_evaluation.csv', selections)
    keys = ('scope', 'region', 'mode', 'group', 'component')
    aggregate = {}
    for row in rows:
        aggregate.setdefault(tuple(row[k] for k in keys), []).append(row)
    summary = []
    for identity, values in sorted(aggregate.items()):
        if {r['seed'] for r in values} != {2026, 2027, 2028} or len(values) != 3:
            raise ValueError('Incomplete diagnostic seed grid')
        row = dict(zip(keys, identity))
        row.update(seeds=3, n_cells_per_seed=values[0]['n_cells'])
        for metric in values[0]:
            if metric in (*keys, 'seed', 'n_cells', 'n_origins'):
                continue
            row[metric+'_mean'] = float(np.mean([r[metric] for r in values]))
            row[metric+'_std'] = float(np.std([r[metric] for r in values], ddof=1))
        summary.append(row)
    write_csv(OUTPUT / 'groups.csv', summary)
    e.write_json(OUTPUT / 'shape.json', shape)
    if any(e.digest(path) != expected for path, expected in sources.items()):
        raise ValueError('A frozen input changed during diagnosis')
    e.write_json(OUTPUT / 'verification.json', dict(
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        script_sha256=e.digest(Path(__file__)), sources=sources,
        no_training=True, no_selection=True, inputs_unchanged=True,
        retrospective_diagnostics_only=True, rows=len(rows), summary_rows=len(summary),
        output_sha256={p.name: e.digest(p) for p in OUTPUT.iterdir() if p.is_file()}))
    print(json.dumps(dict(status='complete', rows=len(rows), summary_rows=len(summary), output=str(OUTPUT))))


if __name__ == '__main__':
    main()
