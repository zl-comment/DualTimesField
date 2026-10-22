"""Read-only integrity audit of a completed, fixed 36-fit architecture screen."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

VARIANTS = ('balanced', 'identity_anchor', 'location_only', 'separate_heads', 'staged', 'wide_head')
REGIONS, FOLDS = ('NSW1', 'QLD1', 'TAS1'), ('2021', '2022')


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ordered(q):
    return q.shape[-1] == 5 and np.isfinite(q).all() and (np.diff(q, axis=-1) >= 0).all()


def mixed(baseline, raw, alpha):
    if alpha == 0:
        return baseline.copy()
    if alpha == 1:
        return raw.copy()
    return (1 - alpha) * baseline.astype(np.float64) + alpha * raw.astype(np.float64)


def audit_prediction(saved, selection, variant, reference):
    """Check independent archives and exact saved blend; no metric selection."""
    for key in ('origin_unix', 'actual', 'baseline', 'target_time_unix'):
        require(np.array_equal(saved[key], reference[key]), f'{key} differs across variants/modes')
    origin, b, y = saved['origin_unix'], saved['baseline'], saved['actual']
    require(b.shape == (len(origin), 24, 5) and y.shape == (len(origin), 24), 'Prediction shapes')
    require(np.isfinite(y).all() and np.all(np.diff(origin) > 0), 'Invalid targets/origins')
    require(np.array_equal(saved['target_time_unix'], origin[:, None] + np.arange(24)[None] * 3600), 'Horizon times')
    require(np.array_equal(saved['quantile_levels'], np.array([.05, .1, .5, .9, .95])), 'Quantile levels')
    require(float(saved['alpha']) == selection['alpha'] and int(saved['epoch']) == selection['epoch'], 'NPZ selection differs')
    for key in ('baseline', 'raw_anchor', 'raw_quantiles', 'anchor', 'quantiles'):
        require(saved[key].shape == b.shape and ordered(saved[key]), f'Invalid quantiles: {key}')
    for key, raw in (('anchor', 'raw_anchor'), ('quantiles', 'raw_quantiles')):
        require(np.array_equal(saved[key], mixed(b, saved[raw], selection['alpha'])), f'Alpha mixing differs: {key}')
    details = dict(n_origins=len(origin), n_cells=int(y.size), alpha=selection['alpha'], epoch=selection['epoch'])
    if variant == 'identity_anchor':
        require(np.array_equal(saved['raw_anchor'], b), 'Identity Anchor is not exactly B')
        details['raw_anchor_exactly_B'] = True
    if variant == 'location_only':
        a64, b64 = saved['raw_anchor'].astype(np.float64), b.astype(np.float64)
        width_error = np.abs((a64 - a64[..., 2:3]) - (b64 - b64[..., 2:3]))
        scale = np.maximum(1, np.maximum(np.abs(a64).max(axis=-1, keepdims=True),
                                        np.abs(b64).max(axis=-1, keepdims=True)))
        tolerance = 4 * np.finfo(saved['raw_anchor'].dtype).eps * scale
        require(np.all(width_error <= tolerance), 'Location-only Anchor changed quantile widths beyond rounding')
        details.update(max_width_difference=float(width_error.max()),
                       max_width_tolerance=float(tolerance.max()), width_rounding_bound_passed=True)
    return details


def audit_training(log, variant, expected_orders):
    require(len(log) == 6 and [r['epoch'] for r in log] == list(range(1, 7)), 'Expected six consecutive epochs')
    orders = [row['order_sha256'] for row in log]
    require(orders == expected_orders, 'Training permutation hashes differ across variants')
    if variant == 'staged':
        require([r['stage'] for r in log] == ['anchor'] * 3 + ['history'] * 3, 'Invalid staged schedule')
        frozen = log[2]['fixed_anchor_sha256']
        for row in log[:3]:
            require(not row['selection_eligible'] and 'raw' not in row and 'guarded' not in row, 'Pretraining selected a checkpoint')
            require(all(v['trainable'] == 0 for k, v in row['parameters'].items()
                        if k not in ('state', 'calibration')), 'History trainable during Anchor pretraining')
        for row in log[3:]:
            state = row['frozen_anchor']
            require(state['before_sha256'] == state['after_sha256'] == frozen, 'Frozen Anchor hash changed')
            require(state['identical'] and state['max_parameter_difference'] == 0 and
                    state['max_probe_prediction_difference'] == 0, 'Frozen Anchor differs')
            require(all(row['parameters'][k]['trainable'] == 0 for k in ('state', 'calibration')), 'Anchor parameters remain trainable')
    else:
        require(all(r['stage'] == 'joint' and r['selection_eligible'] for r in log), 'Expected six joint selection epochs')
        require(all('full_pinball' in r['terms'] for r in log), 'Balanced objective terms missing')
    return dict(epochs=6, matched_training_orders=True, staged_freeze_verified=variant == 'staged')


def audit(run, repo):
    require(read_json(run / 'progress.json').get('stage') == 'complete', 'Run is not complete; audit refused')
    manifest, barrier, report = (read_json(run / name) for name in ('manifest.json', 'selection_complete.json', 'report.json'))
    expected = {f'{r}/{f}/{v}' for r in REGIONS for f in FOLDS for v in VARIANTS}
    require(set(barrier) == expected, 'Selection barrier must contain exactly 36 fits')
    require(set(report) == {f'{r}/{f}' for r in REGIONS for f in FOLDS}, 'Report cells differ')
    require(manifest['config']['epochs'] == 6 and manifest['config']['profile'] == 'screen', 'Wrong fixed budget')
    require(tuple(manifest['config']['variants']) == VARIANTS, 'Manifest variants differ')
    checks, details = [], {}
    def check(name, action):
        try:
            value = action()
            checks.append(dict(check=name, passed=True))
            if value is not None:
                details[name] = value
        except Exception as exc:
            checks.append(dict(check=name, passed=False, error=f'{type(exc).__name__}: {exc}'))
    def hash_check(path, sha):
        require(digest(path) == sha, f'Hash mismatch: {path}')
    for name, sha in manifest['source_sha256'].items():
        check(f'current_source/{name}', lambda p=repo / name, h=sha: hash_check(p, h))
        check(f'source_snapshot/{name}', lambda p=run / 'source' / name, h=sha: hash_check(p, h))
    for name, sha in manifest['bundles'].items():
        check(f'bundle/{name}', lambda p=repo / name, h=sha: hash_check(p, h))
    check('protocol_snapshot', lambda: hash_check(run / 'protocol.md', manifest['protocol_sha256']))
    check('protocol_current', lambda: hash_check(repo / 'docs/v5_anchor_architecture_20261022.md', manifest['protocol_sha256']))
    check('source_diff', lambda: hash_check(run / 'source.diff', manifest['diff_sha256']))
    for region in REGIONS:
        for fold in FOLDS:
            cell = f'{region}/{fold}'
            require(set(report[cell]) == set(VARIANTS), f'Incomplete report variants: {cell}')
            expected_orders = [r['order_sha256'] for r in read_json(run / cell / 'balanced/training.json')]
            with np.load(run / cell / 'balanced/balanced_raw.npz', allow_pickle=False) as reference_file:
                reference = {k: reference_file[k] for k in ('origin_unix', 'actual', 'baseline', 'target_time_unix')}
            sample = manifest['samples'][cell]['evaluation']
            check(f'{cell}/evaluation_sample', lambda s=sample, r=reference:
                  require(len(r['origin_unix']) == s['count'] and
                          hashlib.sha256(r['origin_unix'].astype('<i8').tobytes()).hexdigest() == s['origin_sha256'],
                          'Evaluation origins differ from preregistered manifest'))
            for variant in VARIANTS:
                key, directory = f'{cell}/{variant}', run / cell / variant
                log = read_json(directory / 'training.json')
                check(f'{key}/training', lambda l=log, v=variant, o=expected_orders: audit_training(l, v, o))
                local_metrics = read_json(directory / 'metrics.json')
                check(f'{key}/local_report', lambda m=local_metrics, r=report[cell][variant]:
                      require(m == r, 'Local metrics differ from global report'))
                for mode in ('raw', 'guarded'):
                    selection = barrier[key]['selection'][mode]
                    checkpoint = directory / f'{mode}.pt'
                    saved_hashes = barrier[key]['checkpoints']
                    matches = [sha for name, sha in saved_hashes.items() if (repo / name).resolve() == checkpoint.resolve()]
                    check(f'{key}/{mode}/checkpoint_hash', lambda hs=matches, p=checkpoint:
                          require(len(hs) == 1 and digest(p) == hs[0], 'Checkpoint missing or changed since barrier'))
                    def selection_check(k=key, c=cell, v=variant, m=mode, s=selection, p=checkpoint,
                                        frozen=log[2].get('fixed_anchor_sha256')):
                        saved = torch.load(p, map_location='cpu', weights_only=True)
                        require(saved['variant'] == v and saved['seed'] == manifest['config']['seed'], 'Checkpoint identity')
                        require(saved['selection'] == s == report[c][v][m]['selection'], 'Checkpoint/barrier/report selection mismatch')
                        require(s['epoch'] in (range(4, 7) if v == 'staged' else range(1, 7)), 'Ineligible selected epoch')
                        require(s['stage'] == ('history' if v == 'staged' else 'joint'), 'Wrong selected stage')
                        require(s['alpha'] in manifest['config']['alphas'], 'Undeclared alpha')
                        if v == 'staged':
                            anchor_hash = hashlib.sha256()
                            for name, value in sorted(saved['state_dict'].items()):
                                if name.startswith(('state.', 'calibration.')):
                                    anchor_hash.update(name.encode())
                                    anchor_hash.update(str((value.dtype, tuple(value.shape))).encode())
                                    anchor_hash.update(value.contiguous().numpy().tobytes())
                            require(anchor_hash.hexdigest() == frozen, 'Checkpoint Anchor differs from frozen epoch3')
                        if m == 'guarded':
                            require(s['fallback'] == (s['alpha'] == 0), 'Fallback/alpha disagree')
                        if m == 'raw':
                            require(s['alpha'] == 1 and not s['fallback'], 'Raw checkpoint must be unblended')
                    check(f'{key}/{mode}/selection', selection_check)
                    def prediction_check(p=directory / f'{variant}_{mode}.npz', s=selection, v=variant, r=reference):
                        with np.load(p, allow_pickle=False) as saved:
                            return audit_prediction(saved, s, v, r)
                    check(f'{key}/{mode}/prediction', prediction_check)
    return dict(passed=all(row['passed'] for row in checks), run=str(run.resolve()), expected_fits=36,
                completed_fits=len(barrier), checks_passed=sum(row['passed'] for row in checks),
                checks_failed=sum(not row['passed'] for row in checks), checks=checks, details=details,
                audited_artifact_sha256={name: digest(run / name) for name in
                                        ('manifest.json', 'selection_complete.json', 'report.json', 'protocol.md')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().with_name('audit.json'))
    args = parser.parse_args()
    try:
        result = audit(args.run.resolve(), args.repo.resolve())
    except Exception as exc:
        result = dict(passed=False, run=str(args.run.resolve()), fatal_error=f'{type(exc).__name__}: {exc}')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'details')}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
