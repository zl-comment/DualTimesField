"""Frozen, label-free event allocation using matched archived predictions.

Only the nine guarded formal Full/Trend pairs are available. The intervention
copies Trend q05/q10/q50 and Full q90/q95 without sorting, refitting or changing
the saved checkpoint, alpha, or the nonlinear event/trend interaction.
"""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .full_v5_correction_audit.saved import Sources, validate_same_checkpoint
from .full_v5_upper_scale_audit.audit import _load_formal, _signature
from .full_v5_upper_scale_audit.core import compute_comparison
from .full_v5_event_allocation import event_upper_only, audit_event_pair


COMPARISONS = {
    'upper_only_vs_original': ('original', 'upper_only'),
    'event_off_vs_original': ('original', 'event_off'),
    'upper_only_vs_event_off': ('event_off', 'upper_only'),
}
CHECKS = ('same_checkpoint', 'event_pair_passed', 'lower_equals_trend',
          'upper_equals_full', 'quantiles_ordered')


def _json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _masks(actual, origins):
    """Outcome groups are retrospective only; time groups use AEST origins."""
    y = np.asarray(actual)
    groups = {
        'actual/negative': y < 0,
        'actual/0_to_100': (y >= 0) & (y < 100),
        'actual/100_to_300': (y >= 100) & (y < 300),
        'actual/300_to_1000': (y >= 300) & (y < 1000),
        'actual/1000_plus': y >= 1000,
        'actual/0_to_300': (y >= 0) & (y < 300),
    }
    local = (np.asarray(origins) + 10*3600).astype('datetime64[s]')
    years = local.astype('datetime64[Y]').astype(int) + 1970
    months = local.astype('datetime64[M]').astype(int) % 12
    quarters = months // 3 + 1
    for year in np.unique(years):
        groups[f'year/{year}'] = np.broadcast_to((years == year)[:, None], y.shape)
        for quarter in np.unique(quarters[years == year]):
            mask = (years == year) & (quarters == quarter)
            groups[f'quarter/{year}Q{quarter}'] = np.broadcast_to(mask[:, None], y.shape)
    for lead in range(y.shape[1]):
        groups[f'lead/{lead:02d}'] = np.broadcast_to(np.arange(y.shape[1]) == lead, y.shape)
    return groups


def validate_saved_pair(full_entry, trend_entry, full_metadata, trend_metadata):
    """Reject cross-checkpoint, cross-alpha and cross-input pairing."""
    validate_same_checkpoint({'full': full_entry, 'trend': trend_entry}, ('full', 'trend'))
    for key in ('checkpoint_sha256', 'alpha', 'input_hash', 'sample_hash',
                'preprocessing_hash', 'output_mode', 'region', 'seed', 'split',
                'model_version', 'protocol_id', 'prediction_space', 'price_transform',
                'tree_seed', 'source'):
        if key not in full_metadata or key not in trend_metadata:
            raise ValueError(f'Missing saved-pair identity field: {key}')
        if full_metadata[key] != trend_metadata[key]:
            raise ValueError(f'Saved Full and Trend identity differs: {key}')
    for entry, metadata in ((full_entry, full_metadata), (trend_entry, trend_metadata)):
        if any(entry[key] != metadata[key] for key in ('checkpoint_sha256', 'alpha', 'output_mode')):
            raise ValueError('Prediction metadata differs from its archived selection entry')
    if full_metadata['output_mode'] != 'guarded':
        raise ValueError('Only matched guarded Full/Trend outputs are available')
    return True


def evaluate_decision(rows, cfg):
    """Apply the frozen screen, without selecting a region or intervention.

    Inputs are numeric row dictionaries, not CSV strings. Every expected seed,
    comparison and protected outcome group must be present and nonempty.
    """
    tolerance = float(cfg.get('numeric_tolerance', 1e-10))
    coverage_tolerance = float(cfg.get('coverage_tolerance', .01))
    if tolerance != 1e-10 or coverage_tolerance != .01:
        raise ValueError('The numerical and coverage tolerances are frozen')
    regions, seeds = list(cfg['regions']), list(cfg['seeds'])
    expected = {(region, int(seed)) for region in regions for seed in seeds}
    primary = [row for row in rows if row.get('role') == 'primary'
        and row.get('scope') == 'formal' and row.get('variant') == 'Full_joint'
        and row.get('mode') == 'guarded' and row.get('representation') == 'selected']
    criteria = []
    complete = True
    for comparison in COMPARISONS:
        selected = [r for r in primary if r.get('comparison') == comparison and r.get('group') == 'all']
        observed = [(r['region'], int(r['seed'])) for r in selected]
        valid = len(observed) == len(expected) and set(observed) == expected
        valid = valid and len(observed) == int(cfg.get('expected_cases', len(expected)))
        complete = complete and valid
        criteria.append(dict(name='complete_primary_matrix', comparison=comparison, passed=valid,
                             expected=len(expected), observed=len(observed)))
    overall = [r for r in primary if r.get('group') == 'all']
    for check in CHECKS:
        criteria.append(dict(name=check, passed=complete and all(r.get(check) is True for r in overall)))

    def mean(region, comparison, group, metric):
        selected = [r for r in primary if r.get('comparison') == comparison
                    and r.get('region') == region and r.get('group') == group]
        represented = [int(r['seed']) for r in selected]
        valid = (len(selected) == len(seeds) and set(represented) == set(seeds)
                 and all(not r.get('empty', True) and int(r.get('cells', 0)) > 0
                         and np.isfinite(r.get(metric, np.nan)) for r in selected))
        return float(np.mean([r[metric] for r in selected])) if valid else None

    for region in regions:
        for group in ['all', *cfg['decision_price_groups']]:
            for loss in ('mae', 'qs'):
                value = mean(region, 'upper_only_vs_original', group, f'{loss}_delta')
                criteria.append(dict(name=f'regional_{loss}_nonincrease', region=region,
                    comparison='upper_only_vs_original', group=group, delta=value,
                    nonempty=value is not None, passed=value is not None and value <= tolerance,
                    tolerance=tolerance, statistical_noninferiority_claim=False))
        value = mean(region, 'upper_only_vs_event_off', 'all', 'qs_delta')
        criteria.append(dict(name='regional_qs_nonincrease', region=region,
            comparison='upper_only_vs_event_off', group='all', delta=value,
            nonempty=value is not None, passed=value is not None and value <= tolerance,
            tolerance=tolerance, statistical_noninferiority_claim=False))
        for level in (80, 90):
            before = mean(region, 'upper_only_vs_original', 'all', f'before_coverage{level}')
            after = mean(region, 'upper_only_vs_original', 'all', f'after_coverage{level}')
            worsening = None if before is None or after is None else abs(after-level/100)-abs(before-level/100)
            criteria.append(dict(name='regional_coverage_deviation', region=region, level=level,
                before=before, after=after, absolute_deviation_worsening=worsening,
                tolerance=coverage_tolerance,
                passed=worsening is not None and worsening <= coverage_tolerance+tolerance))
    macro_deltas = {}
    for comparison in ('upper_only_vs_original', 'upper_only_vs_event_off'):
        values = [mean(region, comparison, 'all', 'qs_delta') for region in regions]
        value = float(np.mean(values)) if values and all(v is not None for v in values) else None
        macro_deltas[comparison] = value
        criteria.append(dict(name='macro_qs_strict_improvement', comparison=comparison,
            delta=value, passed=complete and value is not None and value < -tolerance,
            tolerance=tolerance))
    passed = all(item['passed'] for item in criteria)
    return dict(passed=passed,
        decision='eligible_for_prospective_L1_design' if passed else 'stop_upper_only_event_allocation',
        primary='formal/Full_joint/guarded/selected', primary_cases=len({(r['region'], int(r['seed'])) for r in overall}),
        macro_qs_deltas=macro_deltas, criteria=criteria,
        failed_criteria=[item for item in criteria if not item['passed']],
        supporting_results_used_for_selection=False, scientific_confirmation=False,
        instruction='A passing screen permits a separate prospective L1 design only; no automatic training or replacement.')


def _save_case(destination, identity, actual, baseline, original, trend, origins,
               target_times, source_metadata, tolerance):
    """Build all outputs before creating any outcome-dependent groups."""
    b, f, t = (np.asarray(value, dtype=np.float64) for value in (baseline, original, trend))
    event_check = audit_event_pair(b, f, t, source_metadata['alpha'])
    upper_only = event_upper_only(f, t)
    checks = dict(same_checkpoint=source_metadata['same_checkpoint'],
        event_pair_passed=bool(event_check['passed']),
        lower_equals_trend=bool(np.array_equal(upper_only[..., :3], t[..., :3])),
        upper_equals_full=bool(np.array_equal(upper_only[..., 3:], f[..., 3:])),
        quantiles_ordered=bool(np.all(np.diff(upper_only, axis=-1) >= 0)))
    if not all(checks.values()):
        raise ValueError('A structural invariant of the frozen event allocation failed')
    metadata = dict({**identity, **source_metadata, **checks}, event_pair=event_check,
        transformation='upper_only q05/q10/q50 = event_off; q90/q95 = original Full; no sorting',
        outcome_routing=False, checkpoint_reselected=False, alpha_reselected=False,
        trend_event_nonlinear_coupling='preserved through the matched saved Full and event_off outputs',
        numeric_tolerance=tolerance)
    path = Path(destination)/'predictions'/Path(identity['case_id']+'.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(actual, dtype=np.float64)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, origin_unix=origins, target_time_unix=target_times,
        actual=y, baseline=b, original=f, event_off=t, upper_only=upper_only,
        quantile_levels=np.asarray([.05, .10, .50, .90, .95], dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
    temporary.replace(path)
    groups = _masks(y, origins)
    base_scores = compute_comparison(y, b, b, groups)
    base_by_group = {'all': base_scores['overall'], **base_scores['groups']}
    arms, rows = dict(original=f, event_off=t, upper_only=upper_only), []
    for comparison, (before, after) in COMPARISONS.items():
        result = compute_comparison(y, arms[before], arms[after], groups)
        for group, values in [('all', result['overall']), *result['groups'].items()]:
            baseline_values = {key.removeprefix('before_'): value
                               for key, value in base_by_group[group].items() if key.startswith('before_')}
            values.update({f'baseline_{key}': value for key, value in baseline_values.items()})
            if not values['empty']:
                for loss in ('mae', 'qs'):
                    values[f'{before}_vs_baseline_{loss}_delta'] = values[f'before_{loss}']-baseline_values[loss]
                    values[f'{after}_vs_baseline_{loss}_delta'] = values[f'after_{loss}']-baseline_values[loss]
            rows.append(dict(identity, comparison=comparison, before_arm=before, after_arm=after,
                epoch=source_metadata['epoch'], alpha=source_metadata['alpha'], group=group,
                **checks, **values))
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest = dict({**identity, **source_metadata, **checks}, path=str(path.resolve()),
        sha256=digest, bytes=path.stat().st_size, origins=len(y), cells=int(y.size), event_pair=event_check)
    return rows, manifest


def _aggregate(rows):
    identities = ('role', 'scope', 'variant', 'mode', 'representation', 'fold', 'comparison', 'before_arm', 'after_arm', 'group')
    groups = sorted({tuple(str(r.get(key, '')) for key in identities) for r in rows})
    excluded = set(identities) | set(CHECKS) | {'case_id', 'region', 'seed', 'epoch', 'alpha', 'cells', 'empty'}
    fields = sorted({key for row in rows for key, value in row.items()
                     if key not in excluded and isinstance(value, (int, float)) and not isinstance(value, bool)})
    regional, macro = [], []
    for group in groups:
        identity = dict(zip(identities, group))
        selected = [r for r in rows if tuple(str(r.get(key, '')) for key in identities) == group]
        local = []
        for region in sorted({r['region'] for r in selected}):
            cases = [r for r in selected if r['region'] == region]
            nonempty = [r for r in cases if not r['empty']]
            result = dict(identity, region=region, seeds=len(cases), nonempty_seeds=len(nonempty),
                          cells=int(cases[0]['cells']), empty=not bool(nonempty))
            for field in fields:
                values = [float(r[field]) for r in nonempty if field in r]
                if values:
                    result[f'{field}_mean'] = float(np.mean(values))
                    result[f'{field}_seed_sd'] = float(np.std(values, ddof=1)) if len(values)>1 else 0.
            regional.append(result)
            local.append(result)
        available = [r for r in local if not r['empty']]
        result = dict(identity, regions=len(local), nonempty_regions=len(available), empty=not bool(available),
                      partial_group=len(available) != len(local))
        for field in fields:
            values = [r[f'{field}_mean'] for r in available if f'{field}_mean' in r]
            if values:
                result[f'{field}_mean'] = float(np.mean(values))
                seed_macros = [np.mean([r[field] for r in selected if int(r['seed']) == seed
                               and not r['empty'] and field in r]) for seed in sorted({int(r['seed']) for r in selected})]
                seed_macros = [v for v in seed_macros if np.isfinite(v)]
                result[f'{field}_seed_sd'] = float(np.std(seed_macros, ddof=1)) if len(seed_macros)>1 else 0.
        macro.append(result)
    return regional, macro


def audit(output: Path, cfg: dict) -> dict:
    destination = Path(output)/'audit'
    destination.mkdir(parents=True, exist_ok=False)
    tolerance = float(cfg.get('numeric_tolerance', 1e-10))
    if tolerance != 1e-10 or cfg.get('primary', 'formal/Full_joint/guarded/selected') != 'formal/Full_joint/guarded/selected':
        raise ValueError('The primary scope and numerical tolerance are frozen')
    for key in ('training_allowed', 'update_main_table', 'checkpoint_reselection_allowed',
                'alpha_reselection_allowed', 'outcome_routing_allowed'):
        if cfg.get(key, False):
            raise ValueError(f'This audit forbids {key}')
    started, sources, rows, artifacts = time.monotonic(), Sources(), [], []
    formal = Path(cfg['formal_root'])
    report = sources.read_json(formal/'report.json')
    selections = sources.read_json(formal/'selection_complete.json')
    signatures = {}
    for region in cfg['regions']:
        for seed in cfg['seeds']:
            base, base_entry, base_meta = _load_formal(formal, report, selections, sources, region, seed, 'B_retrained')
            full, full_entry, metadata = _load_formal(formal, report, selections, sources, region, seed, 'Full_joint')
            trend, trend_entry, trend_meta = _load_formal(formal, report, selections, sources, region, seed, 'event_off_same_head')
            full_metadata, trend_metadata = (json.loads(str(value['metadata_json'])) for value in (full, trend))
            same_checkpoint = validate_saved_pair(full_entry, trend_entry, full_metadata, trend_metadata)
            # B is stored under the guarded checkpoint identity but has alpha=0.
            if any(base_entry[key] != full_entry[key] for key in ('checkpoint_sha256', 'epoch')):
                raise ValueError('Baseline reference has a different checkpoint identity')
            base_metadata = json.loads(str(base['metadata_json']))
            for key in ('input_hash', 'sample_hash', 'preprocessing_hash'):
                if base_metadata[key] != full_metadata[key]:
                    raise ValueError(f'Baseline and Full input/sample identity differs: {key}')
            signature = _signature(base['origin_unix'], base['target_time_unix'], base['actual'], base['quantiles'])
            if region in signatures and signature != signatures[region]:
                raise ValueError('Fixed tree or sample differs across neural seeds')
            signatures[region] = signature
            for values in (full, trend):
                if _signature(values['origin_unix'], values['target_time_unix'], values['actual']) != signature[:3]:
                    raise ValueError('Baseline, Full and Trend sample grids differ')
            identity = dict(case_id=f'formal/Full_joint/guarded/seed{seed}/{region}', role='primary',
                scope='formal', variant='Full_joint', mode='guarded', representation='selected',
                region=region, seed=int(seed), fold='2023-2024')
            metadata.update(same_checkpoint=same_checkpoint,
                baseline_source_path=base_meta['source_path'], baseline_source_sha256=base_meta['source_sha256'],
                trend_source_path=trend_meta['source_path'], trend_source_sha256=trend_meta['source_sha256'],
                sample_hash=full_metadata['sample_hash'], input_hash=full_metadata['input_hash'],
                preprocessing_hash=full_metadata['preprocessing_hash'])
            new_rows, artifact = _save_case(destination, identity, full['actual'], base['quantiles'],
                full['quantiles'], trend['quantiles'], full['origin_unix'], full['target_time_unix'], metadata, tolerance)
            rows.extend(new_rows)
            artifacts.append(artifact)
            _json(destination/'progress.json', dict(stage='frozen_event_allocation', completed_cases=len(artifacts),
                expected_cases=cfg.get('expected_cases', 9), last_case=identity['case_id'], elapsed_seconds=time.monotonic()-started))
            print(f"[event-allocation] {len(artifacts)}/{cfg.get('expected_cases', 9)} {identity['case_id']}", flush=True)
    if len(artifacts) != int(cfg.get('expected_cases', 9)):
        raise ValueError('The complete nine-case primary matrix is required')
    excluded = [dict(scope='formal', mode='raw', cases=len(cfg['regions'])*len(cfg['seeds']),
                    reason='No saved same-checkpoint raw event_off/Trend; guarded outputs cannot substitute.'),
                dict(scope='development', modes=['raw', 'guarded'], variants=['reference', 'bounded_calibration'],
                    cases=len(cfg['regions'])*len(cfg.get('folds', ['2021', '2022']))*len(cfg['seeds'])*4,
                    reason='Development saved NPZ contain Full and Anchor, but no same-checkpoint Trend output.')]
    regional, macro = _aggregate(rows)
    decision = evaluate_decision(rows, cfg)
    _csv(destination/'scores.csv', rows)
    _csv(destination/'regional.csv', regional)
    _csv(destination/'macro.csv', macro)
    _json(destination/'prediction_manifest.json', artifacts)
    _json(destination/'sources.json', list(sources.records.values()))
    _json(destination/'excluded.json', excluded)
    _json(destination/'decision.json', decision)
    _json(destination/'definitions.json', dict(
        transformation='upper_only[..., :3] = event_off[..., :3]; upper_only[..., 3:] = Full[..., 3:]; no sorting',
        comparisons=COMPARISONS, delta='after loss minus before loss; negative is improvement',
        contrast_questions={
            'upper_only_vs_original': 'Does removing direct event loading from q05/q10/q50 help while retaining exactly the original q90/q95? It cannot add upper-quantile or q90-threshold gains over Full.',
            'upper_only_vs_event_off': 'Does retaining the saved upper-quantile event contribution help beyond closing the direct event term? The first three quantiles are identical.',
            'event_off_vs_original': 'Descriptive complete direct-event-off comparator with the same checkpoint and learned gates.'},
        primary='formal/Full_joint/guarded/selected, all archived origins and 24 horizons',
        event='Full minus event_off preserves saved trend/event tanh coupling; it is not an independently added raw event proposal',
        selection='source checkpoint, epoch and alpha frozen; no training, routing, reselection or new inference',
        qs='2 * mean of five pinball losses; per-quantile pinball has no factor 2',
        benefit='mean(max(before loss - after loss, 0)); harm reverses the difference',
        price_groups='actual values used only after intervention for retrospective scoring; [0,300) overlaps [0,100) and [100,300)',
        time_groups='AEST forecast-origin year and quarter; lead/00 is the origin hour, through lead/23',
        coverage='absolute deviation of regional seed-mean coverage from nominal 0.8/0.9; maximum allowed worsening .01',
        aggregation='equal neural-seed means within region, then equal-region macro; seed SD is descriptive',
        statistical_claim='No significance or statistical noninferiority claim; overlapping forecast cells are not independent events',
        baseline='B is a fixed reference, not an extra intervention; all three arms retain B-relative MAE/QS'))
    summary = dict(cases=len(artifacts), primary_cases=len(artifacts), prediction_files=len(artifacts),
        comparisons=list(COMPARISONS), score_rows=len(rows), regional_rows=len(regional), macro_rows=len(macro),
        all_identity_and_structural_checks_passed=all(all(a[key] for key in CHECKS) for a in artifacts),
        scientific_questions={
            'upper_only_vs_original': 'Remove direct event loading only from the lower three quantiles; upper-tail predictions are exactly unchanged.',
            'upper_only_vs_event_off': 'Measure the value of retaining the saved upper-tail event contribution.'},
        failure_scope='A failed screen stops this fixed allocation candidate; it does not establish that all upper-tail event models are ineffective.',
        excluded=excluded, source_files=len(sources.records), output=str(destination.resolve()), decision=decision,
        elapsed_seconds=time.monotonic()-started, trained_models=0, checkpoints_reselected=0, alphas_reselected=0,
        new_inference=0, official_table_modified=False)
    _json(destination/'summary.json', summary)
    _json(destination/'progress.json', dict(stage='complete', cases=len(artifacts), elapsed_seconds=summary['elapsed_seconds']))
    return summary
