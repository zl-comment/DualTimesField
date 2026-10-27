"""Verified, stage-separated frozen sources for mature-residual controls.

Selection opens only 2020/2021/2022-H1 OOF arrays. L0 adds the archived 2022-H2
deployment fold. L1 arrays are inaccessible without a hashed, passing L0/R28
decision from the same run. No model is instantiated, fitted, or inferred here.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

HOUR, DAY = 3600, 86400
TZ = timezone(timedelta(hours=10))
LEVELS = np.asarray([.05, .1, .5, .9, .95], dtype=np.float64)
FOLDS = ('2020', '2021', '2022', '2022H2')
FOLD_IDS = {name: i for i, name in enumerate((*FOLDS, '2023', '2024'))}
SCORE_ATOL = 1e-9


def timestamp_aest(value):
    if isinstance(value, (int, np.integer)):
        return int(value)
    time = datetime.fromisoformat(str(value))
    if time.tzinfo is None:
        time = time.replace(tzinfo=TZ)
    return int(time.timestamp())


def purged_period_rows(origins, start, end):
    """Project protocol: complete labels must mature STRICTLY before end."""
    values = np.asarray(origins)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError('Expected an integer origin vector')
    start, end = timestamp_aest(start), timestamp_aest(end)
    if start >= end:
        raise ValueError('Invalid period')
    return np.flatnonzero((values >= start) & (values+DAY < end))


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class SourceRegistry:
    """Snapshot verified source hashes; subsequent checks always rehash bytes."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.records = {}

    def register(self, path, expected=None, kind='source'):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f'Source escapes project root: {path}')
        key = str(path)
        if key not in self.records:
            self.records[key] = dict(path=key, sha256=_digest(path), bytes=path.stat().st_size,
                                     kind=kind, expected_sha256=expected)
        record = self.records[key]
        if expected is not None and record['sha256'] != expected:
            raise ValueError(f'Frozen source hash mismatch: {path}')
        if expected is not None:
            record['expected_sha256'] = expected
        return record['sha256']

    def read_json(self, path, expected=None):
        self.register(path, expected, 'metadata')
        return json.loads(Path(path).read_text())

    def verify_unchanged(self):
        for path, record in self.records.items():
            if _digest(path) != record['sha256']:
                raise ValueError(f'Source changed after verification: {path}')
        return len(self.records)


def require_l1_permission(l0_decision):
    """Verify a persisted decision BEFORE opening any formal array or label."""
    if not isinstance(l0_decision, dict) or set(l0_decision) != {'path', 'sha256', 'run_id'}:
        raise PermissionError('L1 needs a persisted passing L0 decision token')
    path = Path(l0_decision['path'])
    if not path.is_file() or _digest(path) != l0_decision['sha256']:
        raise PermissionError('L0 decision file is absent or its hash changed')
    decision = json.loads(path.read_text())
    if (decision.get('passed') is not True or decision.get('stage') != 'L0'
            or decision.get('primary_arm') != 'R28' or not decision.get('run_id')
            or decision['run_id'] != l0_decision['run_id']):
        raise PermissionError('L1 requires a passing L0/R28 decision from the same frozen run')
    return decision


def _score(actual, quantiles):
    y, q = np.asarray(actual, dtype=np.float64), np.asarray(quantiles, dtype=np.float64)
    if q.shape != (*y.shape, 5) or not y.size or not np.isfinite(y).all() or not np.isfinite(q).all():
        raise ValueError('Invalid archived score grid')
    if np.any(np.diff(q, axis=-1) < 0):
        raise ValueError('Archived quantiles crossed')
    error = y[..., None]-q
    return dict(mae=float(np.abs(error[..., 2]).mean()),
        qs=float(2*np.maximum(LEVELS*error, (LEVELS-1)*error).mean()),
        coverage_80=float(((y >= q[..., 1]) & (y <= q[..., 3])).mean()),
        coverage_90=float(((y >= q[..., 0]) & (y <= q[..., 4])).mean()),
        bias=float((q[..., 2]-y).mean()))


def _score_check(actual, quantiles, expected, description):
    computed = _score(actual, quantiles)
    archived = {key: expected[key] for key in ('mae', 'coverage_80', 'coverage_90')}
    archived['qs'] = expected['qs'] if 'qs' in expected else expected['quantile_score_proxy']
    if 'bias' in expected:
        archived['bias'] = expected['bias']
    errors = {key: abs(computed[key]-float(value)) for key, value in archived.items()}
    if any(error > SCORE_ATOL for error in errors.values()):
        raise ValueError(f'Archived score reproduction failed: {description}: {errors}')
    return dict(description=description, passed=True, maximum_absolute_error=max(errors.values()),
                tolerance=SCORE_ATOL, computed=computed, archived=archived,
                compared_metrics=list(archived))


def _fold_edges(name):
    if name == '2022H2':
        return '2022-07-01', '2023-01-01', '2022-01-01'
    year = int(name)
    return f'{year}-01-01', '2022-07-01' if year == 2022 else f'{year+1}-01-01', f'{year-1}-07-01'


def _load_oof(region, fold, bundle_root, registry, trusted):
    directory = bundle_root/'oof'/region/fold
    path = directory/'bundle.npz'
    meta = registry.read_json(path.with_suffix('.json'), trusted[str(path.with_suffix('.json').resolve())])
    config = registry.read_json(directory/'config.json', meta['configuration_sha256'])
    registry.register(path, meta['array_sha256'], 'oof_prediction')
    if trusted[str(path.resolve())] != meta['array_sha256']:
        raise ValueError('OOF bundle disagrees with the archived development input manifest')
    start, end, train_end = _fold_edges(fold)
    if (meta['scope'] != 'chronological_oof' or meta['region'] != region or meta['seed'] != 2026
            or meta['year'] != int(fold[:4]) or meta['final_half'] != (fold == '2022H2')):
        raise ValueError('OOF fold identity changed')
    wanted = {'train': {'start': '2015-01-01', 'end_exclusive': train_end},
        'selection': {'start': train_end, 'end_exclusive': start},
        'calibration': {'start': start, 'end_exclusive': end}}
    if config['data']['split_boundaries'] != wanted or config['data']['load_end_exclusive'] != end:
        raise ValueError('OOF temporal configuration changed')
    expected_models = {f'models/h{h:02d}_q{q:02d}.txt' for h in range(24) for q in range(5)}
    if set(meta['model_files']) != expected_models | {'models/manifest.json'}:
        raise ValueError('Incomplete frozen 24-by-5 tree grid')
    for relative, expected in meta['model_files'].items():
        registry.register(directory/relative, expected, 'frozen_tree')
    tree = registry.read_json(directory/'models/manifest.json', meta['model_files']['models/manifest.json'])
    model = tree['model_metadata']
    if (model['model'] != 'lightgbm' or model['seed'] != 2026 or model['quantile_levels'] != LEVELS.tolist()
            or tree['train_metadata']['region'] != region
            or tree['train_metadata']['preprocessing_hash'] != meta['prediction']['preprocessing_hash']):
        raise ValueError('Frozen tree identity or preprocessing differs from the OOF bundle')
    if len(tree['files']) != 120:
        raise ValueError('Incomplete tree manifest')
    for entry in tree['files']:
        if meta['model_files']['models/'+entry['path']] != entry['sha256']:
            raise ValueError('Tree file hashes disagree between manifests')
    # Read origins FIRST. No future-label array is accessed before its period
    # and metadata guards pass. Only these six fields are decompressed.
    with np.load(path, allow_pickle=False) as archive:
        origin = archive['origin']
        expected = np.arange(timestamp_aest(start), timestamp_aest(end)-DAY, HOUR, dtype=np.int64)
        if not np.array_equal(origin, expected):
            raise ValueError(f'OOF hourly/purged grid differs: {region}/{fold}')
        block = {key: archive[key] for key in ('actual', 'baseline', 'z', 'trained_before', 'selected_before')}
    n = len(origin)
    for key, shape in (('actual', (n, 24)), ('baseline', (n, 24, 5)), ('z', (n, 24, 16))):
        if block[key].shape != shape or block[key].dtype != np.float32 or not np.isfinite(block[key]).all():
            raise ValueError(f'Invalid original FP32 {key} array')
    if np.any(np.diff(block['baseline'], axis=-1) < 0):
        raise ValueError('Unordered OOF baseline')
    for key, field in (('trained_before', 'trained_through'), ('selected_before', 'selected_through')):
        if block[key].shape != (n,) or not np.all(block[key] == meta[field]) or not np.all(block[key] < origin):
            raise ValueError('OOF training/selection permission violation')
    if meta['trained_through'] != timestamp_aest(train_end) or meta['selected_through'] != timestamp_aest(start)-HOUR:
        raise ValueError('OOF frozen cutoff differs from its known schedule')
    block.update(origin=origin, fold_id=np.full(n, FOLD_IDS[fold], dtype=np.int16))
    catalog = dict(region=region, fold=fold, fold_id=FOLD_IDS[fold], origins=n,
        start=start, end_exclusive=end, first_origin=int(origin[0]), last_origin=int(origin[-1]),
        trained_before=meta['trained_through'], selected_before=meta['selected_through'],
        bundle_path=str(path.resolve()), bundle_sha256=meta['array_sha256'],
        tree_manifest_sha256=meta['model_files']['models/manifest.json'],
        preprocessing_hash=meta['prediction']['preprocessing_hash'],
        source_dtype='float32', query_purge='origin+24h < period_end',
        availability_audit=meta['availability_audit'])
    return block, catalog


def _load_development_reference(region, data, robustness, registry, report, selections, seeds):
    full, checks = {}, []
    rows = np.concatenate((data['selection_rows'], data['evaluation_rows']))
    eval_origins = data['origin'][data['evaluation_rows']]
    for seed in seeds:
        directory = robustness/region/'2022'/f'seed_{seed}'/'reference'
        path = directory/'reference_guarded.npz'
        registry.register(path, kind='historical_neural_reference')
        record = selections[f'{region}/2022/{seed}/reference']
        selected = record['selection']['guarded']
        checkpoint = directory/'guarded.pt'
        checkpoint_hash = record['checkpoints'][str(checkpoint.resolve())]
        registry.register(checkpoint, checkpoint_hash, 'checkpoint_hash_only')
        metrics = registry.read_json(directory/'metrics.json')
        if metrics != report[f'{region}/2022/{seed}/reference'] or metrics['guarded']['selection'] != selected:
            raise ValueError('Historical reference metrics/selection manifest changed')
        with np.load(path, allow_pickle=False) as archive:
            origins = archive['origin_unix']
            if not np.array_equal(origins, data['origin'][rows]) or np.any(origins >= timestamp_aest('2023-01-01')):
                raise ValueError('Historical reference must be the exact purged 2022 grid')
            for key in ('actual', 'baseline'):
                if not np.array_equal(archive[key], data[key][rows]):
                    raise ValueError('Historical Full differs from the current B or targets')
            if not np.array_equal(archive['target_time_unix'], origins[:, None]+np.arange(24)*HOUR):
                raise ValueError('Historical Full target grid differs')
            if float(archive['alpha']) != selected['alpha'] or int(archive['epoch']) != selected['epoch']:
                raise ValueError('Historical Full alpha/epoch changed')
            if not np.array_equal(archive['quantile_levels'], LEVELS):
                raise ValueError('Historical Full quantile levels differ')
            q = archive['quantiles']
        h2 = np.searchsorted(origins, eval_origins)
        if not np.array_equal(origins[h2], eval_origins):
            raise ValueError('Historical Full H2 alignment failed')
        checks.append(_score_check(data['actual'][rows], q, metrics['guarded']['metrics']['all']['full'], f'{region}/{seed}/Full_2022'))
        checks.append(_score_check(data['actual'][data['evaluation_rows']], q[h2], metrics['guarded']['metrics']['2022H2']['full'], f'{region}/{seed}/Full_H2'))
        full[int(seed)] = dict(quantiles=q[h2], origin=eval_origins.copy(), epoch=int(selected['epoch']),
            alpha=float(selected['alpha']), checkpoint_path=str(checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
            prediction_path=str(path.resolve()), prediction_sha256=registry.records[str(path.resolve())]['sha256'],
            fit_scope='historical_2022_development_asymmetric_reference; neural train<2021, selection2021; not final formal fit')
    return full, checks


def load_and_verify_sources(cfg, stage='selection', l0_decision=None):
    """Return minimal arrays, query indices, descriptive Fulls and source hashes.

    cfg requires root/bundle_root/robustness_root/regions; seeds defaults to the
    three original neural seeds. stage='selection' never opens H2 labels or any
    neural prediction array. L0 Full references are H2-aligned development
    outputs. stage='L1' additionally requires formal_root/cache_root and a valid
    persisted permission token; only then can formal arrays be opened.
    """
    if stage not in ('selection', 'L0', 'L1'):
        raise ValueError('Unknown source-access stage')
    permission = require_l1_permission(l0_decision) if stage == 'L1' else None
    root = Path(cfg['root']).resolve()
    resolve = lambda key: (root/Path(cfg[key])).resolve()
    registry = SourceRegistry(root)
    if permission is not None:
        registry.register(l0_decision['path'], l0_decision['sha256'], 'frozen_L0_permission')
    bundles, robustness = resolve('bundle_root'), resolve('robustness_root')
    manifest = registry.read_json(robustness/'manifest.json')
    if manifest['config']['tree_seed'] != 2026 or manifest['config']['profile'] != 'full':
        raise ValueError('Expected the complete seed-2026 archived OOF source')
    report = registry.read_json(robustness/'report.json')
    selections = registry.read_json(robustness/'selection_complete.json') if stage == 'L0' else None
    seeds = tuple(cfg.get('seeds', (2026, 2027, 2028)))
    if seeds != (2026, 2027, 2028) or tuple(cfg['regions']) != ('NSW1', 'QLD1', 'TAS1'):
        raise ValueError('The frozen three regions and reference seeds are required')
    regions, catalog, checks = {}, [], []
    folds = FOLDS[:3] if stage == 'selection' else FOLDS
    for region in cfg['regions']:
        blocks = []
        for fold in folds:
            block, item = _load_oof(region, fold, bundles, registry, manifest['inputs'])
            blocks.append(block)
            catalog.append(item)
        data = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
        if np.any(np.diff(data['origin']) <= 0):
            raise ValueError('OOF folds overlap or are out of order')
        data['selection_rows'] = purged_period_rows(data['origin'], '2022-01-01', '2022-07-01')
        data['evaluation_rows'] = purged_period_rows(data['origin'], '2022-07-01', '2023-01-01')
        if len(data['selection_rows']) != 4320 or (stage != 'selection' and len(data['evaluation_rows']) != 4392):
            raise ValueError('Required 2022 selection/evaluation rows are missing')
        data['query_rows'] = data['selection_rows'] if stage == 'selection' else data['evaluation_rows']
        if data['origin'][0] > data['origin'][data['query_rows'][0]]-730*DAY:
            raise ValueError('Library does not cover the first query 730-day lower boundary')
        data['full_reference'] = {}
        periods = [('2022H1', data['selection_rows']), ('2022H2', data['evaluation_rows'])]
        if stage != 'selection':
            periods.append(('all', np.concatenate((data['selection_rows'], data['evaluation_rows']))))
        for period, rows in periods:
            if len(rows):
                archived = report[f'{region}/2022/2026/reference']['B'][period]
                checks.append(_score_check(data['actual'][rows], data['baseline'][rows], archived, f'{region}/B/{period}'))
        if stage == 'L0':
            data['full_reference'], full_checks = _load_development_reference(region, data, robustness, registry, report, selections, seeds)
            checks.extend(full_checks)
        regions[region] = data
    if stage == 'L1':
        formal_checks = _load_formal_after_permission(cfg, regions, registry, permission)
        checks.extend(formal_checks)
    # Query arrays never gain a neural-seed axis. Only historical Full has one.
    metadata = dict(stage=stage, tree_seed=2026, folds=catalog, fold_id_map=FOLD_IDS,
        score_reproduction=checks, L1_permission=permission,
        simple_model_replicated_across_neural_seeds=False, new_inference=0, training=0,
        source_policy='archived OOF model changes at recorded deployment boundaries; no retrospective reforecasting',
        history_maturity='j+24h<=t', query_purge='origin+24h<period_end',
        coordinate_policy='unchanged archived fold-standardized FP32 z; no common-coordinate refit',
        actual_dtype_policy=('archived FP32 OOF targets and B retained, FP64 only for arithmetic/scoring'
            if stage != 'L1' else 'original FP64 formal targets/B retained; historical FP32 values are exactly cast to FP64 when concatenated'),
        limitations=['Archive timestamp proxies, not reconstructed realtime releases.',
                     'H1 and H2 use the pre-existing tree update schedule, not one tree for the entire year.',
                     'OOF end-of-fold purging gaps are preserved; missing windows are not reconstructed.'])
    return dict(regions=regions, sources=list(registry.records.values()), metadata=metadata)


def _load_formal_after_permission(cfg, regions, registry, permission):
    """Conditional L1 only: archived arrays and cache; no booster inference."""
    if permission is None:
        raise PermissionError('Missing verified L0 permission')
    root = Path(cfg['root']).resolve()
    formal = (root/Path(cfg['formal_root'])).resolve()
    cache = (root/Path(cfg.get('cache_root', 'outputs/forecasting/aligned_baselines_20261012/data_cache'))).resolve()
    manifest = registry.read_json(formal/'manifest.json')
    report = registry.read_json(formal/'report.json')
    selections = registry.read_json(formal/'selection_complete.json')
    if (manifest['config']['tree_seed'] != 2026 or manifest['config']['train_end'] != '2022-01-01'
            or manifest['config']['selection_end'] != '2022-07-01'):
        raise ValueError('Formal frozen tree update schedule changed')
    checks = []
    # Import the deterministic feature construction only after permission.
    from .full_v5.data import build_z
    for region, data in regions.items():
        sources = {}
        for seed in cfg.get('seeds', (2026, 2027, 2028)):
            for model in ('B_retrained', 'Full_joint'):
                entry = report[f'{model}/{seed}/{region}']
                path = formal/'runs'/model/f'seed{seed}'/region/'test.npz'
                registry.register(path, entry['sha256'], 'permitted_L1_prediction')
                with np.load(path, allow_pickle=False) as archive:
                    values = {key: archive[key] for key in ('origin_unix', 'target_time_unix', 'actual', 'quantiles', 'quantile_levels', 'metadata_json')}
                meta = json.loads(str(values['metadata_json']))
                selected = selections[f'{region}/{seed}']['selection']['guarded']
                checkpoint = formal/'fits'/region/f'seed{seed}'/'guarded.pt'
                ckhash = selections[f'{region}/{seed}']['checkpoints'][str(checkpoint.resolve())]
                registry.register(checkpoint, ckhash, 'checkpoint_hash_only')
                alpha = 0. if model == 'B_retrained' else selected['alpha']
                if (entry['epoch'] != selected['epoch'] or entry['alpha'] != alpha or meta['alpha'] != alpha
                        or entry['checkpoint_sha256'] != ckhash or meta['checkpoint_sha256'] != ckhash):
                    raise ValueError('Formal source checkpoint/epoch/alpha changed')
                if not np.array_equal(values['quantile_levels'], LEVELS):
                    raise ValueError('Formal quantile levels changed')
                checks.append(_score_check(values['actual'], values['quantiles'], entry['scores'], f'{region}/{seed}/{model}/formal_unpurged'))
                values.update(metadata=meta, selected=selected, checkpoint_sha256=ckhash,
                              source_path=str(path), source_sha256=entry['sha256'])
                sources[(seed, model)] = values
        base = sources[(2026, 'B_retrained')]
        origins = base['origin_unix']
        if (origins.ndim != 1 or np.any(np.diff(origins) <= 0) or origins[0] < timestamp_aest('2023-01-01')
                or origins[-1] >= timestamp_aest('2025-01-01')):
            raise ValueError('Formal source must contain only chronological 2023/2024 origins')
        if not np.array_equal(base['target_time_unix'], origins[:, None]+np.arange(24)*HOUR):
            raise ValueError('Formal target grid changed')
        for (seed, model), values in sources.items():
            for key in ('origin_unix', 'target_time_unix', 'actual'):
                if not np.array_equal(values[key], base[key]):
                    raise ValueError('Formal neural seeds/models have different sample grids')
            if model == 'B_retrained' and not np.array_equal(values['quantiles'], base['quantiles']):
                raise ValueError('The deterministic tree differs across neural reference seeds')
        cache_root = cache/region
        hashes_path = cache_root/'hashes.json'
        hashes = registry.read_json(hashes_path, manifest['inputs_sha256'][str(hashes_path.resolve())])
        cache_meta_path = cache_root/'test/metadata.json'
        cache_meta = registry.read_json(cache_meta_path, hashes['test/metadata.json'])
        for key in ('region', 'input_hash', 'sample_hash', 'preprocessing_hash'):
            if cache_meta[key] != base['metadata'][key]:
                raise ValueError('Formal B and the frozen feature cache differ')
        cached = {}
        for name in ('origin_unix', 'target_time_unix', 'actual', 'input__future_exogenous',
                     'input__ctf_exogenous', 'input__quantile_exogenous'):
            relative = f'test/{name}.npy'
            registry.register(cache_root/relative, hashes[relative], 'permitted_L1_cache')
            cached[name.removeprefix('input__')] = np.load(cache_root/relative, allow_pickle=False)
        for key in ('origin_unix', 'target_time_unix', 'actual'):
            if not np.array_equal(cached[key], base[key]):
                raise ValueError('Formal B and cache arrays differ')
        tree_path = Path(cfg['bundle_root'])/'oof'/region/'2022H2/models/manifest.json'
        tree_path = (root/tree_path).resolve()
        if registry.records[str(tree_path)]['sha256'] != manifest['inputs_sha256'][str(tree_path)]:
            raise ValueError('Formal baseline does not link to the same 2022H2 frozen tree')
        tree = registry.read_json(tree_path)
        if tree['train_metadata']['preprocessing_hash'] != cache_meta['preprocessing_hash']:
            raise ValueError('Formal z preprocessing differs from the deployed tree')
        z = build_z(base['quantiles'], cached).astype(np.float32)
        n_old = len(data['origin'])
        # Preserve all released historical windows, including year-boundary
        # windows, in memory. Only query/scoring indices undergo yearly purging.
        years = (origins+10*HOUR).astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
        incoming = dict(origin=origins, actual=base['actual'], baseline=base['quantiles'], z=z,
            fold_id=np.asarray([FOLD_IDS[str(y)] for y in years], dtype=np.int16),
            trained_before=np.full(len(origins), timestamp_aest('2022-01-01'), dtype=np.int64),
            selected_before=np.full(len(origins), timestamp_aest('2022-07-01')-HOUR, dtype=np.int64))
        for key, value in incoming.items():
            data[key] = np.concatenate((data[key], value))
        qrows = np.concatenate([purged_period_rows(origins, f'{year}-01-01', f'{year+1}-01-01') for year in (2023, 2024)])
        data['evaluation_rows'] = data['query_rows'] = n_old+qrows
        data['full_reference'] = {int(seed): dict(quantiles=sources[(seed, 'Full_joint')]['quantiles'][qrows],
            origin=origins[qrows], epoch=int(sources[(seed, 'Full_joint')]['selected']['epoch']),
            alpha=float(sources[(seed, 'Full_joint')]['selected']['alpha']),
            checkpoint_sha256=sources[(seed, 'Full_joint')]['checkpoint_sha256'],
            prediction_path=sources[(seed, 'Full_joint')]['source_path'],
            prediction_sha256=sources[(seed, 'Full_joint')]['source_sha256'],
            fit_scope='archived final formal guarded fit; no reselection') for seed in cfg.get('seeds', (2026, 2027, 2028))}
    return checks
