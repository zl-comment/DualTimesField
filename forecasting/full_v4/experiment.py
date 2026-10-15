"""Frozen V4: three GPU region pipelines, OOF features, five contrasts and scoring."""
import argparse
import concurrent.futures
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import yaml

from forecasting.aligned_experiment import atomic_json, read_json, sha, freeze_sources, verify_snapshot, timestamp
from forecasting.aligned_inputs import _array_hash
from forecasting.aligned_scoring import make_prediction_artifact, save_prediction_artifact, score_predictions, QUANTILE_LEVELS
from forecasting.common.cache import load_split
from forecasting.datasets import load_forecast_config
from .schema import VARIANTS, OUTPUT_KEYS, subset, tree_features
from .memory import HistoricalMemory, load_availability
from .crossfit import generate_frontends, load_features
from .model import fit_residual_model, predict_residual_model, residual_input_provenance, _load_manifest, crossing_rate

SPLITS = ('train', 'selection', 'calibration', 'test')


def validate_config(config):
    if list(config['variants']) != list(VARIANTS) or config['seeds'] != [2026]:
        raise ValueError('V4 requires the declared five contrasts and seed 2026')
    if len(config['gpu_ids']) != len(config['regions']) or len(set(config['gpu_ids'])) != len(config['gpu_ids']):
        raise ValueError('Assign one distinct GPU per region')
    if not config.get('smoke'):
        if config['regions'] != ['NSW1', 'QLD1', 'TAS1'] or config.get('max_train_origins') or config.get('max_eval_origins'):
            raise ValueError('Formal run requires all regions and untruncated origins')
        if config['lightgbm']['n_estimators'] != 2000 or config['lightgbm']['early_stopping_rounds'] != 50:
            raise ValueError('Formal tree budget changed')


def prepare(config_path, root):
    project = Path.cwd().resolve(); root = Path(root).resolve()
    cfg = yaml.safe_load(Path(config_path).read_text()); validate_config(cfg)
    if root.exists():
        raise FileExistsError('Use a fresh run root')
    baseline = (project / cfg['baseline_root']).resolve()
    original = read_json(baseline / 'jobs.json'); verify_snapshot(baseline, original)
    base = load_forecast_config(project / cfg['baseline_config'])
    if base['data'] != yaml.safe_load((baseline / 'config.yaml').read_text())['data']:
        raise ValueError('Canonical data configuration differs from baseline')
    for path, digest in original['data_hashes'].items():
        if sha(Path(path)) != digest:
            raise ValueError(f'Baseline data changed: {path}')
    root.mkdir(parents=True)
    shutil.copyfile(config_path, root / 'config.yaml')
    shutil.copyfile(project / 'docs/full_v4_design_20261014.md', root / 'design.md')
    atomic_json(root / 'base_config.json', base)
    files, source_hash = freeze_sources(project, root / 'source')
    manifest = dict(experiment_id=cfg['experiment_id'], created_at=timestamp(),
        project_root=str(project), baseline_root=str(baseline), baseline_manifest_hash=sha(baseline / 'jobs.json'),
        base_config_hash=sha(root / 'base_config.json'), config_hash=sha(root / 'config.yaml'),
        design_hash=sha(root / 'design.md'), source_files=files, source_hash=source_hash,
        cache_hashes=original['cache_hashes'], data_hashes=original['data_hashes'],
        smoke=cfg.get('smoke', False), jobs=[dict(variant=v, region=r, seed=2026)
            for v in VARIANTS for r in cfg['regions']])
    atomic_json(root / 'jobs.json', manifest)
    atomic_json(root / 'launch_state.json', dict(status='prepared', jobs=len(manifest['jobs'])))
    return manifest


def verify_run(root):
    manifest = read_json(root / 'jobs.json'); verify_snapshot(root, manifest)
    if sha(root / 'base_config.json') != manifest['base_config_hash'] or sha(root / 'design.md') != manifest['design_hash']:
        raise ValueError('Frozen plan or data configuration changed')
    baseline = Path(manifest['baseline_root'])
    if sha(baseline / 'jobs.json') != manifest['baseline_manifest_hash']:
        raise ValueError('Baseline manifest changed')
    return manifest


def completed(folder, job, manifest):
    if not (folder / 'state.json').exists():
        return False
    state = read_json(folder / 'state.json')
    if state.get('status') != 'completed':
        return False
    for key, value in {**job, 'source_hash': manifest['source_hash'], 'config_hash': manifest['config_hash']}.items():
        if state.get(key) != value:
            raise ValueError('Completed tree identity differs')
    hashes = read_json(folder / 'output_hashes.json')
    required = {'metrics.json', *(f'{s}_predictions.npz' for s in SPLITS[1:])}
    if not required <= set(hashes):
        raise ValueError('Completed outputs incomplete')
    for name, digest in hashes.items():
        if sha(folder / name) != digest:
            raise ValueError('Completed tree output changed')
    _load_manifest(folder / 'models')
    if sha(folder / 'models' / 'manifest.json') != state['model_manifest_sha256']:
        raise ValueError('Completed tree model changed')
    return True


def fit_tree(root, region, variant, memory, splits, cfg, manifest):
    job = dict(variant=variant, region=region, seed=2026)
    folder = root / 'runs' / variant / 'seed2026' / region
    folder.mkdir(parents=True, exist_ok=True)
    if completed(folder, job, manifest):
        return
    state = dict(**job, status='running', pid=os.getpid(), started_at=timestamp(),
                 source_hash=manifest['source_hash'], config_hash=manifest['config_hash'])
    atomic_json(folder / 'state.json', state)
    try:
        mode = {'v4_current_residual': 'current', 'v4_forced_history': 'forced'}.get(variant, 'full')
        feature_root = root / 'frontends' / region / mode
        data = {s: load_features(feature_root / s) for s in ('oof', *SPLITS[1:])}
        train_indices = np.searchsorted(splits['train'].origin_unix, memory.origins[data['oof']['rows']])
        if not np.array_equal(splits['train'].origin_unix[train_indices], memory.origins[data['oof']['rows']]):
            raise ValueError('OOF samples not in canonical training set')
        # The tree's target scale uses only its actual OOF training rows.
        scale = max(1., float(np.std(splits['train'].actual[train_indices].astype(np.float64))))
        tr = subset(splits['train'], train_indices, scale=scale)
        def inputs(name):
            item = data[name]
            canonical = splits['train' if name == 'oof' else name]
            positions = np.searchsorted(canonical.origin_unix, memory.origins[item['rows']])
            if not np.array_equal(canonical.origin_unix[positions], memory.origins[item['rows']]):
                raise ValueError('Feature timestamps do not align')
            split = subset(canonical, positions, scale=scale)
            extra = np.empty((len(split), 0), dtype=np.float32) if variant == 'v4_lgb_oof_rows' else tree_features(item, memory.levels[item['rows']])
            prior = np.zeros((len(split), 24), dtype=np.float64) if variant in ('v4_lgb_oof_rows', 'v4_direct_tree') else item['P']
            return split, extra, prior
        tr, xt, pt = inputs('oof'); sel, xs, ps = inputs('selection')
        models = folder / 'models'
        if models.exists():
            if (models / 'manifest.sha256').exists():
                _, checkpoint = _load_manifest(models)
                for name, split, extra, prior in (('train', tr, xt, pt), ('selection', sel, xs, ps)):
                    if checkpoint['model_metadata'][f'{name}_input_provenance'] != residual_input_provenance(split, extra, prior):
                        raise ValueError('Tree resume feature identity mismatch')
            else:
                models.rename(folder / f'interrupted_models_{time.time_ns()}')
        if not models.exists():
            fit_residual_model(tr, sel, xt, xs, pt, ps, model_directory=models,
                               progress_path=folder / 'progress.json', seed=2026, parameters=cfg['lightgbm'])
        metrics = {}; diagnostics = {}
        for name in SPLITS[1:]:
            split, extra, prior = inputs(name)
            q = predict_residual_model(models, split, extra, prior, threads=cfg['lightgbm']['n_jobs'])
            provenance = residual_input_provenance(split, extra, prior)
            canonical = splits[name]
            if cfg.get('smoke'):
                pos = np.searchsorted(canonical.origin_unix, split.origin_unix)
                canonical = subset(canonical, pos)
            meta = dict(canonical.metadata, seed=2026, source=variant, model=variant,
                output_type='quantile', prediction_space='raw_rrp', base_input_hash=canonical.metadata['input_hash'],
                input_hash=provenance['effective_input_hash'], input_provenance=provenance,
                source_hash=manifest['source_hash'], config_hash=manifest['config_hash'],
                tree_target_scale=scale, historical_memory='growing_asof_static_frontend_and_tree',
                tree_training_origins=len(tr), tree_engine='full_v3_verified_native_lightgbm')
            artifact = make_prediction_artifact(actual=canonical.actual, origin_unix=canonical.origin_unix,
                target_time_unix=canonical.target_time_unix, metadata=meta, quantiles=q, quantile_levels=QUANTILE_LEVELS)
            save_prediction_artifact(folder / f'{name}_predictions.npz', artifact)
            metrics[name] = score_predictions(artifact)
            item = data[name]
            if variant == 'v4_lgb_oof_rows':
                diagnostics[name] = dict(frontend_used=False, unsorted_crossing_rate=crossing_rate(
                    models, split, extra, threads=cfg['lightgbm']['n_jobs']))
                continue
            diagnostics[name] = {k: float(np.abs(item[k] - canonical.actual).mean()) for k in ('P', 'P0')}
            diagnostics[name]['frontend_used'] = True
            diagnostics[name].update(gC_mean=float(item['gC'].mean()), gD_mean=float(item['gD'].mean()),
                null_C_rate=float((item['gC'] == 0).mean()), null_D_rate=float((item['gD'] == 0).mean()),
                correction_mae=float(np.abs(q[..., 2] - prior).mean()))
            diagnostics[name]['unsorted_crossing_rate'] = crossing_rate(models, split, extra, threads=cfg['lightgbm']['n_jobs'])
            diagnostics[name]['slices'] = {}
            y = canonical.actual
            for label, mask in dict(ordinary_0_300=(y >= 0) & (y <= 300), high_gt300=y > 300, negative=y < 0).items():
                diagnostics[name]['slices'][label] = dict(n_cells=int(mask.sum()), **{
                    k: float(np.abs(pred[mask] - y[mask]).mean()) if mask.any() else None
                    for k, pred in dict(P0=item['P0'], P=item['P'], final_q50=q[..., 2]).items()})
            if variant != 'v4_lgb_oof_rows':
                train_p = data['oof']['P']
                diagnostics[name]['prior_outside_oof_horizon_range'] = float(((item['P'] < train_p.min(0)) | (item['P'] > train_p.max(0))).mean())
            weights = item['weights']
            diagnostics[name]['adjacent_weight_total_variation'] = float(.5 * np.abs(np.diff(weights, axis=2)).sum(axis=1).mean())
        atomic_json(folder / 'metrics.json', metrics)
        atomic_json(folder / 'frontend_diagnostics.json', diagnostics)
        hashes = {p.name: sha(p) for p in folder.glob('*_predictions.npz')}
        hashes.update({n: sha(folder / n) for n in ('metrics.json', 'frontend_diagnostics.json')})
        atomic_json(folder / 'output_hashes.json', hashes)
        state.update(status='completed', completed_at=timestamp(), model_manifest_sha256=sha(models / 'manifest.json'))
        atomic_json(folder / 'state.json', state)
    except Exception:
        state.update(status='failed', error=traceback.format_exc())
        atomic_json(folder / 'state.json', state)
        raise


def worker(root, region):
    root = Path(root).resolve()
    if Path(__file__).resolve() != root / 'source/forecasting/full_v4/experiment.py':
        raise ValueError('Worker must use frozen source')
    manifest = verify_run(root)
    cfg = yaml.safe_load((root / 'config.yaml').read_text()); validate_config(cfg)
    if region not in cfg['regions']:
        raise ValueError('Region not in frozen matrix')
    for path, digest in manifest['data_hashes'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Raw archive changed since preparation')
    baseline = Path(manifest['baseline_root']); original = read_json(baseline / 'jobs.json')
    splits = {s: load_split(baseline, original, region, s) for s in SPLITS}
    base = read_json(root / 'base_config.json')
    availability = load_availability(Path(base['project_root']) / base['data']['region_files'][region], region)
    memory = HistoricalMemory(splits, availability, config=cfg['memory'])
    candidates = memory.candidate_indices(memory.origins)
    audit_dir = root / 'memory' / region; audit_dir.mkdir(parents=True, exist_ok=True)
    np.save(audit_dir / 'origins.npy', memory.origins)
    np.save(audit_dir / 'indices.npy', candidates)
    np.save(audit_dir / 'max_availability.npy', memory.candidate_max_availability(candidates))
    valid = candidates >= 0
    if np.any(memory.candidate_max_availability(candidates)[valid] > np.broadcast_to(memory.origins[:, None], candidates.shape)[valid]):
        raise AssertionError('Unavailable historical outcome')
    atomic_json(audit_dir / 'manifest.json', dict(future_publication_violations=0,
        files={p.name: sha(p) for p in audit_dir.glob('*.npy')}))
    try:
        generate_frontends(root, region, memory, candidates, splits, cfg, base, manifest, 'cuda:0')
    except Exception:
        atomic_json(root / 'frontends' / region / 'state.json', dict(status='failed', error=traceback.format_exc()))
        raise
    # Neural fits are done; independent native tree jobs can share CPU resources.
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['tree_workers_per_region']) as pool:
        futures = [pool.submit(fit_tree, root, region, v, memory, splits, cfg, manifest) for v in VARIANTS]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def launch(root):
    root = Path(root).resolve()
    with (root / 'launch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify_run(root)
        cfg = yaml.safe_load((root / 'config.yaml').read_text()); validate_config(cfg)
        atomic_json(root / 'launch_state.json', dict(status='running', started_at=timestamp(), pid=os.getpid()))
        def run(region, gpu):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            with (root / f'{region}.log').open('a') as stream:
                return region, subprocess.call([sys.executable, '-u', '-m', 'forecasting.full_v4.experiment',
                    'worker', '--run-root', str(root), '--region', region], cwd=root / 'source', env=env,
                    stdout=stream, stderr=subprocess.STDOUT)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cfg['regions'])) as pool:
            futures = [pool.submit(run, r, gpu) for r, gpu in zip(cfg['regions'], cfg['gpu_ids'])]
            results = dict(f.result() for f in concurrent.futures.as_completed(futures))
        if any(results.values()):
            atomic_json(root / 'launch_state.json', dict(status='failed', exits=results, completed_at=timestamp()))
            raise RuntimeError(f'Region worker failed: {results}')
        if not cfg.get('smoke'):
            from .report import write_report
            try:
                summary = write_report(root)
                if not summary['comparison_available'] or summary['errors']:
                    raise RuntimeError('Completed predictions did not pass report audit')
                archive = Path(manifest['project_root']) / 'logs/forecasting' / root.name
                archive.mkdir(parents=True, exist_ok=True)
                for name in ('report.md', 'results.csv', 'summary.json'):
                    shutil.copyfile(root / name, archive / name)
            except Exception:
                atomic_json(root / 'launch_state.json', dict(status='failed', stage='report',
                    error=traceback.format_exc(), completed_at=timestamp()))
                raise
        atomic_json(root / 'launch_state.json', dict(status='completed', exits=results, completed_at=timestamp()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'launch', 'worker', 'report'))
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--config', default='configs/full_v4/experiment.yaml')
    parser.add_argument('--region')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.config, args.run_root)
    elif args.action == 'launch':
        launch(args.run_root)
    elif args.action == 'worker':
        worker(args.run_root, args.region)
    else:
        from .report import write_report
        print(write_report(args.run_root)['progress'])


if __name__ == '__main__':
    main()
