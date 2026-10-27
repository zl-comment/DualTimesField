"""Frozen upper-scale removal across the complete saved prediction archive.

Only formal Full_joint guarded is primary. Outcomes are used after the fixed
transform for scoring, never for routing, selecting checkpoints or changing alpha.
"""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..full_v5_correction_audit.saved import Sources, validate_same_checkpoint
from ..full_v5_correction_audit.metrics import score
from .core import remove_upper_scale, common_history_audit, compute_comparison


def _json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _signature(*arrays):
    return [(tuple(a.shape), str(a.dtype), hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
            for a in (np.asarray(value) for value in arrays)]


def _masks(actual, origins):
    """Disjoint bins plus explicitly overlapping ordinary-price protection."""
    y = np.asarray(actual)
    groups = {'actual/negative': y < 0,
        'actual/0_to_100': (y >= 0) & (y < 100),
        'actual/100_to_300': (y >= 100) & (y < 300),
        'actual/300_to_1000': (y >= 300) & (y < 1000),
        'actual/1000_plus': y >= 1000,
        'actual/0_to_300': (y >= 0) & (y < 300)}
    local = (np.asarray(origins)+10*3600).astype('datetime64[s]')
    years = local.astype('datetime64[Y]').astype(int)+1970
    for year in np.unique(years):
        groups[f'year/{year}'] = np.broadcast_to((years == year)[:, None], y.shape)
    return groups


def evaluate_decision(rows, cfg):
    """Apply the prespecified gate to primary rows only; never tune a policy.

    Each row needs role/scope/variant/mode/representation/region/seed/group,
    cells/empty, qs_delta, point_equal, lower_quantiles_equal,
    quantiles_ordered and common_history_passed. Missing/empty strata fail.
    """
    tolerance = float(cfg.get('numeric_tolerance', 1e-10))
    if tolerance != 1e-10:
        raise ValueError('The frozen numerical tolerance is 1e-10, not a noninferiority margin')
    regions, seeds = list(cfg['regions']), list(cfg['seeds'])
    required_groups = ['all', *cfg['decision_price_groups']]
    primary = [row for row in rows if row.get('role') == 'primary' and row.get('scope') == 'formal'
               and row.get('variant') == 'Full_joint' and row.get('mode') == 'guarded'
               and row.get('representation') == 'selected']
    expected = {(region, int(seed)) for region in regions for seed in seeds}
    all_rows = [row for row in primary if row['group'] == 'all']
    observed = [(row['region'], int(row['seed'])) for row in all_rows]
    complete = len(observed) == len(expected) and set(observed) == expected
    expected_count = int(cfg.get('expected_primary_cases', len(expected)))
    complete = complete and len(observed) == expected_count
    criteria = [dict(name='complete_primary_matrix', passed=complete,
                     expected=expected_count, observed=len(observed))]
    for name in ('point_equal', 'lower_quantiles_equal', 'quantiles_ordered', 'common_history_passed'):
        criteria.append(dict(name=name, passed=complete and all(row.get(name) is True for row in all_rows)))
    values = {}
    for region in regions:
        for group in required_groups:
            selected = [row for row in primary if row['region'] == region and row['group'] == group]
            represented = [int(row['seed']) for row in selected]
            valid = (len(selected) == len(seeds) and set(represented) == set(seeds)
                     and all(not row.get('empty', True) and int(row.get('cells', 0)) > 0
                             and np.isfinite(row.get('qs_delta', np.nan)) for row in selected))
            value = float(np.mean([row['qs_delta'] for row in selected])) if valid else None
            values[(region, group)] = value
            criteria.append(dict(name='regional_qs_nonincrease', region=region, group=group,
                passed=bool(valid and value <= tolerance), qs_delta=value, nonempty=valid,
                tolerance=tolerance, statistical_noninferiority_claim=False))
    region_all = [values[(region, 'all')] for region in regions]
    macro = float(np.mean(region_all)) if region_all and all(value is not None for value in region_all) else None
    criteria.append(dict(name='macro_qs_strict_improvement', qs_delta=macro,
                         passed=bool(complete and macro is not None and macro < -tolerance), tolerance=tolerance))
    passed = all(item['passed'] for item in criteria)
    return dict(passed=passed,
        decision='eligible_for_prospective_L1_design' if passed else 'stop_global_upper_removal',
        primary='formal/Full_joint/guarded/selected', primary_cases=len(observed),
        primary_macro_qs_delta=macro, criteria=criteria,
        failed_criteria=[item for item in criteria if not item['passed']],
        supporting_results_used_for_selection=False, scientific_confirmation=False,
        instruction='A passing screen permits a separate prospective L1 design only; no automatic training or replacement.')


def _load_formal(root, report, selections, sources, region, seed, model):
    entry = report[f'{model}/{seed}/{region}']
    path = root/'runs'/model/f'seed{seed}'/region/'test.npz'
    data = sources.load(path, entry['sha256'])
    mode = 'raw' if model == 'Full_joint_raw' else 'guarded'
    record = selections[f'{region}/{seed}']
    selected = record['selection'][mode]
    checkpoint = root/'fits'/region/f'seed{seed}'/f'{mode}.pt'
    ckhash = sources.register(checkpoint, record['checkpoints'][str(checkpoint.resolve())], 'checkpoint_hash_only')
    metadata = json.loads(str(data['metadata_json']))
    alpha = 0. if model == 'B_retrained' else selected['alpha']
    if (entry['epoch'] != selected['epoch'] or entry['alpha'] != alpha
            or entry['checkpoint_sha256'] != ckhash or metadata['checkpoint_sha256'] != ckhash
            or metadata['alpha'] != alpha):
        raise ValueError('Formal archive differs from frozen checkpoint/alpha selection')
    return data, entry, dict(source_path=str(path.resolve()), source_sha256=entry['sha256'],
                            checkpoint_path=str(checkpoint.resolve()), checkpoint_sha256=ckhash,
                            epoch=int(entry['epoch']), alpha=float(entry['alpha']),
                            applied_alpha=float(entry['alpha']))


def _save_case(destination, identity, actual, baseline, original, origins, target_times,
               source_metadata, anchor, tolerance):
    """The intervention is computed before any outcome-dependent masks exist."""
    b, f, y = (np.asarray(value, dtype=np.float64) for value in (baseline, original, actual))
    upper_off = remove_upper_scale(b, f)
    if upper_off.dtype != np.float64:
        raise ValueError('Upper intervention must use FP64')
    point_equal = bool(np.array_equal(upper_off[..., 2], f[..., 2]))
    lower_equal = bool(np.array_equal(upper_off[..., :3], f[..., :3]))
    ordered = bool(np.all(np.diff(upper_off, axis=-1) >= 0))
    if not point_equal or not lower_equal or not ordered:
        raise ValueError('Fixed upper removal changed q05/q10/q50 or crossed quantiles')
    if anchor is None:
        common = dict(available=False, passed=None,
            reason='Formal raw has no same-checkpoint raw Anchor artifact; guarded Anchor was not borrowed.')
    else:
        common = dict(available=True, **common_history_audit(b, np.asarray(anchor, dtype=np.float64), f,
                                                         source_metadata.get('applied_alpha', source_metadata['alpha'])))
    metadata = dict(identity, **source_metadata,
        transformation='q05/q10/q50 unchanged; q90/q95 = Bq + (Full50 - B50)',
        point_equal=point_equal, lower_quantiles_equal=lower_equal, quantiles_ordered=ordered,
        common_history=common, outcome_routing=False, checkpoint_reselected=False, alpha_reselected=False,
        numeric_tolerance=tolerance)
    path = destination/'predictions'/Path(identity['case_id']+'.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, origin_unix=origins, target_time_unix=target_times,
        actual=y, baseline=b, original=f, upper_off=upper_off,
        quantile_levels=np.asarray([.05,.10,.50,.90,.95], dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
    temporary.replace(path)
    comparison = compute_comparison(y, f, upper_off, _masks(y, origins))
    baseline_scores = score(y, b)
    comparison['overall'].update({f'baseline_{key}': value for key, value in baseline_scores.items()})
    for loss in ('mae', 'qs'):
        comparison['overall'][f'original_vs_baseline_{loss}_delta'] = comparison['overall'][f'before_{loss}']-baseline_scores[loss]
        comparison['overall'][f'upper_off_vs_baseline_{loss}_delta'] = comparison['overall'][f'after_{loss}']-baseline_scores[loss]
    rows = []
    checks = dict(point_equal=point_equal, lower_quantiles_equal=lower_equal, quantiles_ordered=ordered,
                  common_history_passed=common.get('passed'))
    for group, values in [('all', comparison['overall']), *comparison['groups'].items()]:
        rows.append(dict(identity, epoch=source_metadata['epoch'], alpha=source_metadata['alpha'],
                         group=group, **checks, **values))
    with path.open('rb') as stream:
        output_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest = dict(identity, path=str(path.resolve()), sha256=output_hash, bytes=path.stat().st_size,
        origins=int(len(y)), cells=int(y.size), **source_metadata, **checks, common_history=common)
    return rows, manifest


def _aggregate(rows):
    """Equal neural-seed means within regions, then equal-region macro means."""
    identity_keys = ('role','scope','variant','mode','representation','fold','group')
    identities = sorted({tuple(str(row.get(key, '')) for key in identity_keys) for row in rows})
    region_rows, macro_rows = [], []
    metric_names = ('mae','qs','rmse','bias','coverage80','coverage90','width80','width90','under_q05','over_q95',
                    'pinball_q05','pinball_q10','pinball_q50','pinball_q90','pinball_q95')
    fields = [f'{arm}_{metric}' for arm in ('before','after','baseline') for metric in metric_names]
    fields += ['mae_delta','qs_delta','mae_benefit','mae_harm','qs_benefit','qs_harm']
    fields += [f'pinball_{name}_{suffix}' for name in ('q05','q10','q50','q90','q95')
               for suffix in ('delta','benefit','harm')]
    fields += [f'{arm}_vs_baseline_{loss}_delta' for arm in ('original','upper_off') for loss in ('mae','qs')]
    for values in identities:
        identity = dict(zip(identity_keys, values))
        selected = [row for row in rows if tuple(str(row.get(key, '')) for key in identity_keys) == values]
        regions = sorted({row['region'] for row in selected})
        local = []
        for region in regions:
            cases = [row for row in selected if row['region'] == region]
            nonempty = [row for row in cases if not row['empty']]
            row = dict(identity, region=region, seeds=len(cases), nonempty_seeds=len(nonempty),
                       cells=int(cases[0]['cells']), empty=not bool(nonempty))
            if nonempty:
                for field in fields:
                    if field in nonempty[0]:
                        numbers = [float(case[field]) for case in nonempty]
                        row[f'{field}_mean'] = float(np.mean(numbers))
                        row[f'{field}_seed_sd'] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else 0.
            region_rows.append(row)
            local.append(row)
        available = [row for row in local if not row['empty']]
        macro = dict(identity, regions=len(regions), nonempty_regions=len(available),
                     empty=not bool(available), partial_group=bool(len(available) != len(regions)))
        if available:
            for field in fields:
                key = f'{field}_mean'
                if key in available[0]:
                    macro[key] = float(np.mean([row[key] for row in available]))
            # Neural seed spread, with region macro computed within each seed.
            for field in fields:
                if f'{field}_mean' not in macro:
                    continue
                per_seed = []
                for seed in sorted({int(row['seed']) for row in selected}):
                    cases = [row for row in selected if int(row['seed']) == seed and not row['empty']]
                    if cases:
                        per_seed.append(float(np.mean([row[field] for row in cases])))
                macro[f'{field}_seed_sd'] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.
        macro_rows.append(macro)
    return region_rows, macro_rows


def audit(output: Path, cfg: dict) -> dict:
    destination = Path(output)/'audit'
    destination.mkdir(parents=True, exist_ok=False)
    tolerance = float(cfg.get('numeric_tolerance', 1e-10))
    if tolerance != 1e-10 or cfg.get('primary', 'formal/Full_joint/guarded/selected') != 'formal/Full_joint/guarded/selected':
        raise ValueError('Primary comparison and numerical tolerance are frozen')
    if cfg.get('training_allowed', False) or cfg.get('outcome_routing_allowed', False):
        raise ValueError('No fitting or outcome routing is permitted')
    sources, rows, artifacts = Sources(), [], []
    started = time.monotonic()

    def record(identity, data, baseline, metadata, anchor=None):
        new_rows, artifact = _save_case(destination, identity, data['actual'], baseline, data['quantiles'],
            data['origin_unix'], data['target_time_unix'], metadata, anchor, tolerance)
        rows.extend(new_rows)
        artifacts.append(artifact)
        progress = dict(stage='frozen_upper_removal', completed_cases=len(artifacts),
                        expected_cases=cfg.get('expected_cases'), last_case=identity['case_id'],
                        elapsed_seconds=time.monotonic()-started)
        _json(destination/'progress.json', progress)
        print(f"[upper-scale] {len(artifacts)}/{cfg.get('expected_cases', '?')} {identity['case_id']}", flush=True)

    formal = Path(cfg['formal_root'])
    report = sources.read_json(formal/'report.json')
    selections = sources.read_json(formal/'selection_complete.json')
    formal_samples = {}
    for region in cfg['regions']:
        for seed in cfg['seeds']:
            base, base_entry, base_meta = _load_formal(formal, report, selections, sources, region, seed, 'B_retrained')
            baseline = base['quantiles']
            signature = _signature(base['origin_unix'], base['target_time_unix'], base['actual'], baseline)
            if region in formal_samples and formal_samples[region] != signature:
                raise ValueError('Formal sample or fixed tree differs across neural seeds')
            formal_samples[region] = signature
            anchor, anchor_entry, anchor_meta = _load_formal(formal, report, selections, sources, region, seed, 'Anchor_joint')
            if _signature(anchor['origin_unix'], anchor['target_time_unix'], anchor['actual']) != signature[:3]:
                raise ValueError('Formal Anchor sample mismatch')
            for model in cfg.get('formal_models', ('Full_joint','Full_joint_raw')):
                data, entry, meta = _load_formal(formal, report, selections, sources, region, seed, model)
                if _signature(data['origin_unix'], data['target_time_unix'], data['actual']) != signature[:3]:
                    raise ValueError('Formal candidate sample mismatch')
                primary = model == 'Full_joint'
                mode = 'guarded' if primary else 'raw'
                if primary:
                    validate_same_checkpoint({'full':entry,'anchor':anchor_entry}, ('full','anchor'))
                identity = dict(case_id=f'formal/{model}/{mode}/seed{seed}/{region}',
                    role='primary' if primary else 'sensitivity', scope='formal', variant=model,
                    mode=mode, representation='selected', region=region, seed=int(seed), fold='2023-2024')
                meta.update(baseline_source_path=base_meta['source_path'], baseline_source_sha256=base_meta['source_sha256'])
                if primary:
                    meta.update(anchor_source_path=anchor_meta['source_path'], anchor_source_sha256=anchor_meta['source_sha256'])
                record(identity, data, baseline, meta, anchor['quantiles'] if primary else None)

    development = Path(cfg['robustness_root'])
    verification = sources.read_json(development/'independent_verification.json')
    sources.register(development/'report.json', verification['report_sha256'])
    sources.register(development/'selection_complete.json', verification['selection_sha256'])
    report = sources.read_json(development/'report.json')
    selections = sources.read_json(development/'selection_complete.json')
    development_samples = {}
    for region in cfg['regions']:
        for fold in cfg['folds']:
            for seed in cfg['seeds']:
                for variant in cfg.get('development_variants', ('reference','bounded_calibration')):
                    key = f'{region}/{fold}/{seed}/{variant}'
                    frozen, archived = selections[key], report[key]
                    for mode in cfg.get('modes', ('raw','guarded')):
                        path = development/region/str(fold)/f'seed_{seed}'/variant/f'{variant}_{mode}.npz'
                        expected = verification['source_predictions'][str(path.resolve())]
                        data = sources.load(path, expected)
                        signature = _signature(data['origin_unix'], data['target_time_unix'], data['actual'], data['baseline'])
                        sample_key = f'{region}/{fold}'
                        if sample_key in development_samples and development_samples[sample_key] != signature:
                            raise ValueError('Development samples or baseline differ across arms/modes/seeds')
                        development_samples[sample_key] = signature
                        selected = frozen['selection'][mode]
                        if (selected != archived[mode]['selection'] or int(data['epoch']) != selected['epoch']
                                or float(data['alpha']) != selected['alpha']):
                            raise ValueError('Development selected epoch/alpha differs from frozen archive')
                        alpha = float(data['alpha'])
                        baseline = data['baseline'].astype(np.float64)
                        for output_key, raw_key in (('anchor','raw_anchor'),('quantiles','raw_quantiles')):
                            q = data[raw_key].astype(np.float64)
                            mixed = baseline if alpha == 0 else q if alpha == 1 else (1-alpha)*baseline+alpha*q
                            if not np.allclose(data[output_key], mixed, rtol=0, atol=tolerance):
                                raise ValueError('Archived alpha blend does not reproduce selected output')
                        checkpoint = path.parent/f'{mode}.pt'
                        ckhash = sources.register(checkpoint, frozen['checkpoints'][str(checkpoint.resolve())], 'checkpoint_hash_only')
                        identity = dict(case_id=f'development/{region}/{fold}/seed{seed}/{variant}/{mode}',
                            role='support', scope='development', variant=variant, mode=mode,
                            representation='selected', region=region, seed=int(seed), fold=str(fold))
                        metadata = dict(source_path=str(path.resolve()), source_sha256=expected,
                            checkpoint_path=str(checkpoint.resolve()), checkpoint_sha256=ckhash,
                            epoch=int(selected['epoch']), alpha=alpha, applied_alpha=alpha)
                        record(identity, data, baseline, metadata, data['anchor'])

    if len(artifacts) != int(cfg.get('expected_cases', len(artifacts))):
        raise ValueError('Incomplete frozen intervention matrix')
    regional, macro = _aggregate(rows)
    decision = evaluate_decision(rows, cfg)
    _csv(destination/'scores.csv', rows)
    _csv(destination/'regional.csv', regional)
    _csv(destination/'macro.csv', macro)
    _json(destination/'prediction_manifest.json', artifacts)
    _json(destination/'sources.json', list(sources.records.values()))
    _json(destination/'decision.json', decision)
    _json(destination/'definitions.json', dict(
        delta='after upper_off loss minus original loss; negative is improvement',
        primary='formal/Full_joint/guarded/selected; raw and development never select the intervention',
        formula='first three quantiles copied; upper two = B upper + (original q50 - B q50)',
        comparison='original and upper_off share each source checkpoint and alpha; no fitting or reselection',
        ordinary='[0,300), with separately protected [0,100); overlapping groups must not be added',
        extreme='actual >=1000; retrospective outcome grouping only',
        disjoint_prices='negative, [0,100), [100,300), [300,1000), [1000,infinity)',
        year='AEST forecast-origin year',
        common_history='compared to same-checkpoint Anchor where available, with FP32 rounding bound; raw formal gap retained',
        qs='2 * mean of five pinball losses; q-specific pinball columns use no factor 2',
        benefit='mean(max(original loss - upper_off loss,0)); harm reverses the difference',
        neural_seeds='optimization replicates, not independent markets',
        forecast_cells='overlapping origin-horizon cells, not independent events',
        statistical_claim='numerical prespecified screening only; no significance or statistical noninferiority claim',
        artifacts='FP64 compact paired predictions, full origins; no duplicated neural checkpoint files'))
    summary = dict(cases=len(artifacts), primary_cases=sum(a['role']=='primary' for a in artifacts),
        sensitivity_cases=sum(a['role']=='sensitivity' for a in artifacts),
        support_cases=sum(a['role']=='support' for a in artifacts), score_rows=len(rows),
        regional_rows=len(regional), macro_rows=len(macro), output=str(destination.resolve()),
        source_files=len(sources.records), prediction_files=len(artifacts),
        all_point_predictions_unchanged=all(a['point_equal'] for a in artifacts),
        all_lower_quantiles_unchanged=all(a['lower_quantiles_equal'] for a in artifacts),
        all_quantiles_ordered=all(a['quantiles_ordered'] for a in artifacts),
        missing_same_checkpoint_anchor_cases=sum(not a['common_history']['available'] for a in artifacts),
        decision=decision, elapsed_seconds=time.monotonic()-started,
        trained_models=0, checkpoints_reselected=0, alphas_reselected=0, official_table_modified=False)
    _json(destination/'summary.json', summary)
    _json(destination/'progress.json', dict(stage='complete', cases=len(artifacts), elapsed_seconds=summary['elapsed_seconds']))
    return summary
