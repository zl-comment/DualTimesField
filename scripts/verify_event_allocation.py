"""Independently verify frozen event-allocation artifacts using only NumPy.

No forecasting/core/scorer/audit modules are imported. Run after the frozen
audit completes: python scripts/verify_event_allocation.py OUTPUT_DIRECTORY.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


LEVELS = (.05, .10, .50, .90, .95)
NAMES = ('q05', 'q10', 'q50', 'q90', 'q95')
PAIRS = {'upper_only_vs_original': ('original', 'upper_only'),
         'event_off_vs_original': ('original', 'event_off'),
         'upper_only_vs_event_off': ('event_off', 'upper_only')}
CHECKS = ('same_checkpoint', 'event_pair_passed', 'lower_equals_trend',
          'upper_equals_full', 'quantiles_ordered')
IDENTITY = ('role', 'scope', 'variant', 'mode', 'representation', 'fold',
            'comparison', 'before_arm', 'after_arm', 'group')
NUMERIC_ATOL = 1e-9  # Independent reduction-order check, not a decision margin.


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def primitives(y, q):
    residual = y-q[..., 2]
    result = dict(mae=np.abs(residual), rmse=residual*residual, bias=-residual,
        coverage80=(y >= q[..., 1]) & (y <= q[..., 3]),
        coverage90=(y >= q[..., 0]) & (y <= q[..., 4]),
        width80=q[..., 3]-q[..., 1], width90=q[..., 4]-q[..., 0],
        under_q05=y < q[..., 0], over_q95=y > q[..., 4])
    losses = []
    for i, (tau, name) in enumerate(zip(LEVELS, NAMES)):
        error = y-q[..., i]
        loss = np.where(error >= 0, tau*error, (tau-1)*error)
        result[f'pinball_{name}'] = loss
        losses.append(loss)
    result['qs'] = (2/5)*sum(losses)
    return result


def masks(y, origins):
    result = {'all': np.ones(y.shape, dtype=bool),
        'actual/negative': y < 0, 'actual/0_to_100': (0 <= y) & (y < 100),
        'actual/100_to_300': (100 <= y) & (y < 300),
        'actual/300_to_1000': (300 <= y) & (y < 1000),
        'actual/1000_plus': y >= 1000, 'actual/0_to_300': (0 <= y) & (y < 300)}
    time = (origins+36000).astype('datetime64[s]')
    year = time.astype('datetime64[Y]').astype(int)+1970
    quarter = (time.astype('datetime64[M]').astype(int) % 12)//3+1
    for yr in np.unique(year):
        result[f'year/{yr}'] = np.broadcast_to((year == yr)[:, None], y.shape)
        for qt in np.unique(quarter[year == yr]):
            result[f'quarter/{yr}Q{qt}'] = np.broadcast_to(((year == yr) & (quarter == qt))[:, None], y.shape)
    for lead in range(y.shape[1]):
        value = np.zeros(y.shape, dtype=bool)
        value[:, lead] = True
        result[f'lead/{lead:02d}'] = value
    assert np.all(sum(result[k].astype(int) for k in ('actual/negative', 'actual/0_to_100',
        'actual/100_to_300', 'actual/300_to_1000', 'actual/1000_plus')) == 1)
    return result


def comparison_cells(y, before_q, after_q, before, after, baseline):
    cells = {f'before_{k}': v for k, v in before.items()}
    cells.update({f'after_{k}': v for k, v in after.items()})
    cells.update({f'baseline_{k}': v for k, v in baseline.items()})
    for name in ('mae', 'qs', *(f'pinball_{q}' for q in NAMES)):
        d = after[name]-before[name]
        cells[f'{name}_delta'] = d
        cells[f'{name}_benefit'] = np.where(d < 0, -d, 0.)
        cells[f'{name}_harm'] = np.where(d > 0, d, 0.)
        if name in ('mae', 'qs'):
            cells[f'{name}_harm_rate'] = d > 1e-7
    for label, names in (('lower', NAMES[:2]), ('upper', NAMES[3:])):
        d = sum(after[f'pinball_{q}']-before[f'pinball_{q}'] for q in names)
        cells[f'{label}_tail_qs_delta'] = d
        cells[f'{label}_tail_qs_benefit'] = np.where(d < 0, -d, 0.)
        cells[f'{label}_tail_qs_harm'] = np.where(d > 0, d, 0.)
    residual = y-before_q[..., 2]
    move = after_q[..., 2]-before_q[..., 2]
    same_direction = move*residual > 0
    cells.update(wrong_direction_rate=move*residual < 0,
        target_crossing_rate=same_direction & (np.abs(move) > np.abs(residual)),
        harmful_overshoot_rate=same_direction & (np.abs(move) > 2*np.abs(residual)),
        correction_from_exact_baseline_rate=(residual == 0) & (move != 0),
        zero_correction_rate=move == 0, mean_abs_median_delta=np.abs(move),
        mean_abs_quantile_delta=np.abs(after_q-before_q).mean(axis=-1),
        mean_signed_median_delta=move)
    return cells


def common_event(b, f, t, alpha):
    """Independent two-rounded-additions error analysis, never empirical tol."""
    assert 0 <= alpha <= 1 and np.all(f >= t)
    if alpha == 0:
        np.testing.assert_array_equal(f, b)
        np.testing.assert_array_equal(t, b)
        return dict(common_shift_max_abs_error=0., rounding_bound_max=0., max_error_to_bound_ratio=0.,
            event_increment_min=0., event_increment_max=0., median_event_increment_mean=0.,
            positive_event_fraction=0., theoretical_event_upper_bound=0., event_range_rounding_bound_max=0.)
    u32, u64 = np.finfo(np.float32).eps/2, np.finfo(np.float64).eps/2
    g32, g64 = u32/(1-u32), 3*u64/(1-3*u64)
    sub32 = float(np.nextafter(np.float32(0), np.float32(1)))
    pairs = []
    for q in (f, t):
        if alpha == 1:
            pairs.append((np.abs(q), np.zeros_like(q)))
        else:
            magnitude = (1-alpha)*np.abs(b)
            p = (np.abs(q)+(1+g64)*magnitude)/(1-g64)
            pairs.append((p, g64*(magnitude+p)))
    (pf, rf), (pt, rt) = pairs
    bound = g32*(pf+pt+pf[..., 2:3]+pt[..., 2:3])+rf+rt+rf[..., 2:3]+rt[..., 2:3]
    bound += g64*(np.abs(f)+np.abs(t)+np.abs(f[..., 2:3])+np.abs(t[..., 2:3]))+2*alpha*sub32
    increment = f-t
    err = np.abs(increment-increment[..., 2:3])
    assert np.all(err <= bound), 'Event common shift violates its frozen rounding bound'
    range_bound = g32*(pf+pt)+rf+rt+g64*(np.abs(f)+np.abs(t)+300*alpha)+alpha*sub32
    assert np.all(increment <= 300*alpha+range_bound), 'Event exceeds its saturated 300*alpha range'
    ratio = np.divide(err, bound, out=np.zeros_like(err), where=bound > 0)
    return dict(common_shift_max_abs_error=float(err.max()), rounding_bound_max=float(bound.max()),
        max_error_to_bound_ratio=float(ratio.max()), event_increment_min=float(increment.min()),
        event_increment_max=float(increment.max()), median_event_increment_mean=float(increment[..., 2].mean()),
        positive_event_fraction=float(np.mean(increment[..., 2] > 0)), theoretical_event_upper_bound=300*alpha,
        event_range_rounding_bound_max=float(range_bound.max()))


def verify(root):
    root = Path(root).resolve()
    hashes, max_error, checked_metrics = {}, 0., 0

    def assert_hash(path, expected):
        key = str(Path(path).resolve())
        if key not in hashes:
            hashes[key] = sha(key)
        assert hashes[key] == expected, key

    def close(actual, expected, label):
        nonlocal max_error, checked_metrics
        error = abs(float(actual)-float(expected))
        assert np.isfinite(error) and error <= NUMERIC_ATOL, (label, actual, expected, error)
        max_error = max(max_error, error)
        checked_metrics += 1

    assert read(root/'progress.json')['stage'] == 'complete'
    manifest, recorded = read(root/'manifest.json'), read(root/'verification.json')
    cfg = manifest['configuration']
    assert recorded['run_id'] == manifest['run_id']
    assert recorded['trained_models'] == recorded['checkpoints_reselected'] == recorded['alphas_reselected'] == 0
    assert not recorded['formal_tables_changed'] and not recorded['scientific_confirmation']
    for path, expected in manifest['sources'].items():
        assert_hash(path, expected)
        relative = Path(path).relative_to(Path(cfg['root']))
        assert_hash(root/'source'/relative, expected)
    for relative, expected in recorded['artifact_sha256'].items():
        assert_hash(root/relative, expected)
    for stage in ('audit', 'episodes'):
        marker = read(root/f'{stage}_complete.json')
        payload = {k: v for k, v in marker.items() if k != 'completion_id'}
        assert hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() == marker['completion_id']
        observed = {str(p.relative_to(root)): sha(p) for p in (root/stage).rglob('*') if p.is_file()}
        assert observed == marker['artifact_sha256']
    for source in read(root/'audit/sources.json'):
        assert_hash(source['path'], source['sha256'])
    index = read(root/'audit/prediction_manifest.json')
    assert len(index) == cfg['expected_cases'] == cfg['expected_primary_cases'] == 9
    with (root/'audit/scores.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(r['case_id'], r['comparison'], r['group']): r for r in rows}
    assert len(lookup) == len(rows)
    frozen = read(Path(cfg['formal_root'])/'selection_complete.json')
    source_report = read(Path(cfg['formal_root'])/'report.json')
    independent, seen, empty_groups = [], set(), 0
    max_attribution, max_common, alpha_zero = 0., 0., 0
    cross_seed = {}
    identity_keys = ('checkpoint_sha256', 'alpha', 'input_hash', 'sample_hash', 'preprocessing_hash',
        'output_mode', 'region', 'seed', 'split', 'model_version', 'protocol_id', 'prediction_space',
        'price_transform', 'tree_seed', 'source')
    for number, item in enumerate(index, 1):
        assert (item['role'], item['scope'], item['variant'], item['mode'], item['representation']) == (
            'primary', 'formal', 'Full_joint', 'guarded', 'selected')
        assert_hash(item['path'], item['sha256'])
        assert_hash(item['checkpoint_path'], item['checkpoint_sha256'])
        data, source_data, source_meta = load(item['path']), {}, {}
        meta = json.loads(str(data['metadata_json']))
        for key in ('case_id', 'region', 'seed', 'epoch', 'alpha', 'applied_alpha', 'checkpoint_sha256',
                    'source_path', 'source_sha256', 'trend_source_path', 'trend_source_sha256',
                    'baseline_source_path', 'baseline_source_sha256'):
            assert meta[key] == item[key], (item['case_id'], key)
        assert item['applied_alpha'] == item['alpha']
        assert meta['outcome_routing'] is meta['checkpoint_reselected'] is meta['alpha_reselected'] is False
        selection = frozen[f"{item['region']}/{item['seed']}"]
        selected = selection['selection']['guarded']
        assert selected['epoch'] == item['epoch'] and selected['alpha'] == item['alpha']
        assert selection['checkpoints'][str(Path(item['checkpoint_path']).resolve())] == item['checkpoint_sha256']
        for arm, model, prefix in (('original', 'Full_joint', ''),
                                   ('event_off', 'event_off_same_head', 'trend_'),
                                   ('baseline', 'B_retrained', 'baseline_')):
            path, expected = item[prefix+'source_path'], item[prefix+'source_sha256']
            assert_hash(path, expected)
            source_data[arm] = load(path)
            source_meta[arm] = json.loads(str(source_data[arm]['metadata_json']))
            entry = source_report[f"{model}/{item['seed']}/{item['region']}"]
            assert entry['sha256'] == expected and entry['epoch'] == item['epoch']
            assert entry['checkpoint_sha256'] == item['checkpoint_sha256']
            wanted_alpha = 0. if arm == 'baseline' else item['alpha']
            assert entry['alpha'] == source_meta[arm]['alpha'] == wanted_alpha
            assert source_meta[arm]['checkpoint_sha256'] == item['checkpoint_sha256']
            np.testing.assert_array_equal(data[arm], source_data[arm]['quantiles'])
            for field in ('actual', 'origin_unix', 'target_time_unix'):
                np.testing.assert_array_equal(data[field], source_data[arm][field])
        for key in identity_keys:
            assert source_meta['original'][key] == source_meta['event_off'][key], (item['case_id'], key)
        for key in ('input_hash', 'sample_hash', 'preprocessing_hash'):
            assert source_meta['baseline'][key] == source_meta['original'][key]
        signature = tuple(hashlib.sha256(np.ascontiguousarray(data[k]).tobytes()).hexdigest()
                          for k in ('actual', 'origin_unix', 'target_time_unix', 'baseline'))
        if item['region'] in cross_seed:
            assert signature == cross_seed[item['region']]
        cross_seed[item['region']] = signature
        y, b, f, t, u = (data[k].astype(np.float64, copy=False) for k in
                        ('actual', 'baseline', 'original', 'event_off', 'upper_only'))
        assert u.dtype == np.float64 and f.shape == t.shape == u.shape == (*y.shape, 5)
        for q in (b, f, t, u):
            assert np.isfinite(q).all() and np.all(np.diff(q, axis=-1) >= 0)
        np.testing.assert_array_equal(u[..., :3], t[..., :3])
        np.testing.assert_array_equal(u[..., 3:], f[..., 3:])
        np.testing.assert_array_equal(data['quantile_levels'], np.asarray(LEVELS))
        np.testing.assert_array_equal(data['target_time_unix'], data['origin_unix'][:, None]+np.arange(24)*3600)
        assert np.all(np.diff(data['origin_unix']) > 0)
        event = common_event(b, f, t, item['alpha'])
        for key, value in event.items():
            close(value, item['event_pair'][key], (item['case_id'], 'event_bound', key))
        assert all(item[key] is meta[key] is True for key in CHECKS)
        max_common = max(max_common, event['common_shift_max_abs_error'])
        if item['alpha'] == 0:
            np.testing.assert_array_equal(u, b)
            alpha_zero += 1
        quantiles = dict(original=f, event_off=t, upper_only=u, baseline=b)
        primitives_by_arm = {key: primitives(y, q) for key, q in quantiles.items()}
        for name in NAMES[3:]:
            np.testing.assert_array_equal(primitives_by_arm['upper_only'][f'pinball_{name}'],
                                          primitives_by_arm['original'][f'pinball_{name}'])
        for name in ('mae', 'rmse', 'bias', *(f'pinball_{q}' for q in NAMES[:3])):
            np.testing.assert_array_equal(primitives_by_arm['upper_only'][name], primitives_by_arm['event_off'][name])
        group_masks = masks(y, data['origin_unix'])
        for pair, (before, after) in PAIRS.items():
            cells = comparison_cells(y, quantiles[before], quantiles[after], primitives_by_arm[before],
                                     primitives_by_arm[after], primitives_by_arm['baseline'])
            for group, mask in group_masks.items():
                key = (item['case_id'], pair, group)
                saved = lookup[key]
                seen.add(key)
                count = int(mask.sum())
                assert int(saved['cells']) == count and (saved['empty'] == 'True') == (count == 0)
                row = {key: item[key] for key in ('case_id', 'role', 'scope', 'variant', 'mode',
                       'representation', 'region', 'seed', 'fold')}
                row.update(comparison=pair, before_arm=before, after_arm=after, group=group,
                           cells=count, empty=count == 0, **{key: True for key in CHECKS})
                for flag in CHECKS:
                    assert saved[flag] == 'True'
                if count == 0:
                    assert saved['qs_delta'] == ''
                    empty_groups += 1
                    independent.append(row)
                    continue
                computed = {name: float(value[mask].mean()) for name, value in cells.items()}
                for label in ('before', 'after', 'baseline'):
                    computed[f'{label}_rmse'] = float(np.sqrt(computed[f'{label}_rmse']))
                for arm, label in ((before, 'before'), (after, 'after')):
                    for loss in ('mae', 'qs'):
                        computed[f'{arm}_vs_baseline_{loss}_delta'] = computed[f'{label}_{loss}']-computed[f'baseline_{loss}']
                for field, value in computed.items():
                    close(value, saved[field], (key, field))
                if pair in ('upper_only_vs_original', 'upper_only_vs_event_off'):
                    changed = NAMES[:3] if pair == 'upper_only_vs_original' else NAMES[3:]
                    contribution = (2/5)*sum(computed[f'pinball_{name}_delta'] for name in changed)
                    error = abs(computed['qs_delta']-contribution)
                    bound = 32*np.finfo(np.float64).eps*(abs(computed['before_qs'])+abs(computed['after_qs'])+abs(contribution)+1)
                    assert error <= bound, (key, 'quantile QS attribution', error, bound)
                    max_attribution = max(max_attribution, error)
                row.update(computed)
                independent.append(row)
            del cells
        print(f'[independent-event] verified {number}/{len(index)} cases', flush=True)
    assert seen == set(lookup)
    aggregate_count = 0
    for filename, macro in (('regional.csv', False), ('macro.csv', True)):
        with (root/'audit'/filename).open() as stream:
            summaries = list(csv.DictReader(stream))
        unique = set()
        for saved in summaries:
            identity = tuple(saved[k] for k in IDENTITY)+(('macro',) if macro else (saved['region'],))
            assert identity not in unique
            unique.add(identity)
            cases = [r for r in independent if all(str(r[k]) == saved[k] for k in IDENTITY)
                     and (macro or r['region'] == saved['region'])]
            valid = [r for r in cases if not r['empty']]
            assert cases and (saved['empty'] == 'True') == (not valid)
            for label, number in saved.items():
                suffix = '_seed_sd' if label.endswith('_seed_sd') else '_mean' if label.endswith('_mean') else None
                if suffix is None or number == '':
                    continue
                field = label[:-len(suffix)]
                assert any(field in r for r in valid), (filename, field)
                if macro:
                    by = 'region' if suffix == '_mean' else 'seed'
                    values = [float(np.mean([r[field] for r in valid if r[by] == value and field in r]))
                              for value in sorted({r[by] for r in valid})]
                else:
                    values = [r[field] for r in valid if field in r]
                expected = float(np.mean(values)) if suffix == '_mean' else float(np.std(values, ddof=1)) if len(values)>1 else 0.
                close(expected, number, (filename, identity, label))
                aggregate_count += 1
        expected_identities = {tuple(str(r[k]) for k in IDENTITY)+(('macro',) if macro else (r['region'],)) for r in independent}
        assert unique == expected_identities
    decision = check_decision(independent, cfg, read(root/'audit/decision.json'), close)
    result = dict(status='passed', run_id=manifest['run_id'], cases=len(index), comparisons=list(PAIRS),
        score_groups_checked=len(seen), empty_score_groups_retained=empty_groups,
        scalar_metrics_checked=checked_metrics, aggregate_scalar_metrics_checked=aggregate_count,
        maximum_metric_abs_error=max_error, maximum_quantile_qs_attribution_abs_error=max_attribution,
        maximum_common_event_shift_error_within_frozen_bound=max_common,
        upper_quantiles_exactly_equal_full=True, lower_quantiles_exactly_equal_same_head_event_off=True,
        qs_delta_quantile_attribution_verified=True, all_quantiles_ordered=True,
        alpha_zero_exact_cases=alpha_zero, source_checkpoint_epoch_alpha_all_verified=True,
        files_hash_verified=len(hashes), independent_decision=decision,
        independent_implementation='Standalone NumPy piecewise pinball, interval calculations and event rounding analysis; no production imports.',
        trained_models=0, original_artifacts_written=False,
        limitations=['Retrospective fixed-output allocation, not a retrained model or new blind test.',
                     'Seed variation and overlapping cells are not independent market-event replications.',
                     'No same-checkpoint raw/development event-off artifacts were substituted.'])
    (root/'independent_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return result


def check_decision(rows, cfg, saved, close):
    """Reconstruct every prespecified decision condition from independent rows."""
    assert cfg['numeric_tolerance'] == 1e-10 and cfg['coverage_tolerance'] == .01
    expected = {(r, int(s)) for r in cfg['regions'] for s in cfg['seeds']}
    flags, predicted, macro_values = {}, {}, {}

    def add(name, passed, *, region='', comparison='', group='', level='', **values):
        key = (name, region, comparison, group, level)
        assert key not in flags
        flags[key] = bool(passed)
        predicted[key] = values

    def mean(region, comparison, group, metric):
        selected = [r for r in rows if (r['region'], r['comparison'], r['group']) == (region, comparison, group)]
        valid = len(selected) == len(cfg['seeds']) and {r['seed'] for r in selected} == set(cfg['seeds'])
        valid = valid and all(not r['empty'] and r['cells'] > 0 and metric in r for r in selected)
        return float(np.mean([r[metric] for r in selected])) if valid else None

    for comparison in PAIRS:
        selected = [r for r in rows if r['comparison'] == comparison and r['group'] == 'all']
        complete = len(selected) == len(expected) and {(r['region'], r['seed']) for r in selected} == expected
        add('complete_primary_matrix', complete, comparison=comparison, expected=len(expected), observed=len(selected))
    for check in CHECKS:
        add(check, all(r[check] for r in rows))
    for region in cfg['regions']:
        for group in ('all', *cfg['decision_price_groups']):
            for loss in ('mae', 'qs'):
                value = mean(region, 'upper_only_vs_original', group, f'{loss}_delta')
                add(f'regional_{loss}_nonincrease', value is not None and value <= 1e-10,
                    region=region, comparison='upper_only_vs_original', group=group,
                    delta=value, nonempty=value is not None)
        value = mean(region, 'upper_only_vs_event_off', 'all', 'qs_delta')
        add('regional_qs_nonincrease', value is not None and value <= 1e-10, region=region,
            comparison='upper_only_vs_event_off', group='all', delta=value, nonempty=value is not None)
        for level in (80, 90):
            before = mean(region, 'upper_only_vs_original', 'all', f'before_coverage{level}')
            after = mean(region, 'upper_only_vs_original', 'all', f'after_coverage{level}')
            difference = None if before is None or after is None else abs(after-level/100)-abs(before-level/100)
            add('regional_coverage_deviation', difference is not None and difference <= .01+1e-10,
                region=region, level=level, before=before, after=after, absolute_deviation_worsening=difference)
    for comparison in ('upper_only_vs_original', 'upper_only_vs_event_off'):
        values = [mean(region, comparison, 'all', 'qs_delta') for region in cfg['regions']]
        value = float(np.mean(values)) if all(v is not None for v in values) else None
        macro_values[comparison] = value
        add('macro_qs_strict_improvement', value is not None and value < -1e-10, comparison=comparison, delta=value)
    seen = set()
    for criterion in saved['criteria']:
        key = tuple(criterion.get(k, '') for k in ('name', 'region', 'comparison', 'group', 'level'))
        assert key not in seen and criterion['passed'] == flags[key], key
        seen.add(key)
        for field, value in predicted[key].items():
            if value is None or isinstance(value, bool):
                assert criterion[field] is value
            else:
                close(value, criterion[field], ('decision', key, field))
    assert seen == set(flags)
    passed = all(flags.values())
    assert saved['passed'] == passed
    assert saved['decision'] == ('eligible_for_prospective_L1_design' if passed else 'stop_upper_only_event_allocation')
    assert saved['primary_cases'] == len(expected)
    assert saved['failed_criteria'] == [c for c in saved['criteria'] if not c['passed']]
    for key, value in macro_values.items():
        close(value, saved['macro_qs_deltas'][key], ('decision', key, 'macro'))
    return dict(passed=passed, decision=saved['decision'], macro_qs_deltas=macro_values,
                criteria_checked=len(flags), failed_criteria=sum(not value for value in flags.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = verify(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
