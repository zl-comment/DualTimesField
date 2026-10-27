"""Independent NumPy verification of completed upper-scale audit artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1]).resolve()

def read(path):
    return json.loads(Path(path).read_text())

def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

hashes = {}
def assert_hash(path, expected):
    key = str(Path(path).resolve())
    if key not in hashes:
        hashes[key] = sha(key)
    assert hashes[key] == expected, key

def primitives(y, q):
    e50 = y-q[..., 2]
    result = dict(mae=np.abs(e50), rmse=e50**2, bias=-e50,
                  coverage80=(y >= q[..., 1]) & (y <= q[..., 3]),
                  coverage90=(y >= q[..., 0]) & (y <= q[..., 4]),
                  width80=q[..., 3]-q[..., 1], width90=q[..., 4]-q[..., 0],
                  under_q05=y < q[..., 0], over_q95=y > q[..., 4])
    pbs = []
    for k, (tau, name) in enumerate(zip((.05, .1, .5, .9, .95), ('q05','q10','q50','q90','q95'))):
        err = y-q[..., k]
        # Direct piecewise definition, independent of the production scorer.
        pb = np.where(err >= 0, tau*err, (tau-1)*err)
        result['pinball_'+name] = pb
        pbs.append(pb)
    result['qs'] = 2*np.stack(pbs, axis=-1).mean(axis=-1)
    return result

def reductions(values, mask):
    out = {name: float(value[mask].mean()) for name, value in values.items()}
    out['rmse'] = float(np.sqrt(out['rmse']))
    return out

def masks(y, origins):
    result = {'all': np.ones(y.shape, dtype=bool),
        'actual/negative': y < 0,
        'actual/0_to_100': (0 <= y) & (y < 100),
        'actual/100_to_300': (100 <= y) & (y < 300),
        'actual/300_to_1000': (300 <= y) & (y < 1000),
        'actual/1000_plus': y >= 1000,
        'actual/0_to_300': (0 <= y) & (y < 300)}
    local_year = (origins+36000).astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
    result.update({f'year/{year}': np.broadcast_to((local_year == year)[:, None], y.shape)
                   for year in np.unique(local_year)})
    disjoint = [result[name] for name in ('actual/negative', 'actual/0_to_100', 'actual/100_to_300',
                                         'actual/300_to_1000', 'actual/1000_plus')]
    np.testing.assert_array_equal(np.sum(disjoint, axis=0), np.ones(y.shape))
    return result

def common_shift(b, a, f, alpha):
    if alpha == 0:
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(f, b)
        return 0.
    unit32 = np.finfo(np.float32).eps/2
    unit64 = np.finfo(np.float64).eps/2
    g32, g64 = unit32/(1-unit32), 3*unit64/(1-3*unit64)
    if alpha == 1:
        pf = np.abs(f)
        rf, ra = np.zeros_like(f), np.zeros_like(a)
    else:
        z = (1-alpha)*np.abs(b)
        pf = (np.abs(f)+(1+g64)*z)/(1-g64)
        pa = (np.abs(a)+(1+g64)*z)/(1-g64)
        rf, ra = g64*(z+pf), g64*(z+pa)
    bound = g32*(pf+pf[..., 2:3])+rf+ra+rf[..., 2:3]+ra[..., 2:3]
    bound += g64*(np.abs(f)+np.abs(a)+np.abs(f[..., 2:3])+np.abs(a[..., 2:3]))
    bound += alpha*float(np.nextafter(np.float32(0), np.float32(1)))
    err = np.abs((f-a)-(f[..., 2:3]-a[..., 2:3]))
    assert np.all(err <= bound), 'common-history premise violates its frozen FP32/FP64 bound'
    return float(np.max(err))

assert read(root/'progress.json')['stage'] == 'complete'
manifest, verification = read(root/'manifest.json'), read(root/'verification.json')
cfg = manifest['configuration']
assert verification['run_id'] == manifest['run_id']
assert verification['trained_models'] == verification['checkpoints_reselected'] == verification['alphas_reselected'] == 0
assert not verification['formal_tables_changed'] and not verification['scientific_confirmation']
for relative, expected in verification['artifact_sha256'].items():
    assert_hash(root/relative, expected)
marker = read(root/'audit_complete.json')
identity = {key: value for key, value in marker.items() if key != 'completion_id'}
assert hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == marker['completion_id']
observed_files = {str(path.relative_to(root)): sha(path) for path in (root/'audit').rglob('*') if path.is_file()}
assert observed_files == marker['artifact_sha256']
index = read(root/'audit/prediction_manifest.json')
assert len(index) == cfg['expected_cases'] == 90
with (root/'audit/scores.csv').open() as stream:
    saved_rows = list(csv.DictReader(stream))
row_lookup = {(row['case_id'], row['group']): row for row in saved_rows}
assert len(row_lookup) == len(saved_rows), 'Duplicate case/group score rows'
formal_selections = read(Path(cfg['formal_root'])/'selection_complete.json')
dev_selections = read(Path(cfg['robustness_root'])/'selection_complete.json')
max_metric_error, max_common_shift_error, checked_groups, common_cases, empty_groups = 0., 0., 0, 0, 0
max_upper_tail_attribution_error = 0.
independent_rows = []
alpha_zero_cases = 0
checked_metrics = 0
for number, item in enumerate(index, 1):
    assert_hash(item['path'], item['sha256'])
    assert_hash(item['source_path'], item['source_sha256'])
    assert_hash(item['checkpoint_path'], item['checkpoint_sha256'])
    data = load(item['path'])
    original_source = load(item['source_path'])
    meta = json.loads(str(data['metadata_json']))
    for field in ('case_id','role','scope','variant','mode','representation','region','seed','fold',
                  'source_path','source_sha256','checkpoint_path','checkpoint_sha256','epoch','alpha',
                  'applied_alpha'):
        assert meta[field] == item[field], (item['case_id'], field)
    assert meta['applied_alpha'] == meta['alpha'], 'Selected alpha must remain the applied alpha'
    assert meta['outcome_routing'] is False and meta['checkpoint_reselected'] is False and meta['alpha_reselected'] is False
    if item['scope'] == 'formal':
        frozen = formal_selections[f"{item['region']}/{item['seed']}"]
        source_meta = json.loads(str(original_source['metadata_json']))
        assert source_meta['checkpoint_sha256'] == item['checkpoint_sha256']
        assert source_meta['alpha'] == item['alpha']
        baseline_source = load(meta['baseline_source_path'])
        assert_hash(meta['baseline_source_path'], meta['baseline_source_sha256'])
        np.testing.assert_array_equal(data['baseline'], baseline_source['quantiles'])
        for field in ('origin_unix','target_time_unix','actual'):
            np.testing.assert_array_equal(data[field], baseline_source[field])
        anchor = None
        if item['variant'] == 'Full_joint':
            assert item['role'] == 'primary' and item['mode'] == 'guarded'
            assert_hash(meta['anchor_source_path'], meta['anchor_source_sha256'])
            anchor_source = load(meta['anchor_source_path'])
            anchor = anchor_source['quantiles'].astype(np.float64)
            anchor_metadata = json.loads(str(anchor_source['metadata_json']))
            assert anchor_metadata['checkpoint_sha256'] == item['checkpoint_sha256']
            assert anchor_metadata['alpha'] == item['alpha']
            for field in ('origin_unix','target_time_unix','actual'):
                np.testing.assert_array_equal(data[field], anchor_source[field])
        else:
            assert item['role'] == 'sensitivity' and item['mode'] == 'raw'
            assert item['common_history']['available'] is False
    else:
        assert item['role'] == 'support'
        frozen = dev_selections[f"{item['region']}/{item['fold']}/{item['seed']}/{item['variant']}"]
        np.testing.assert_array_equal(data['baseline'], original_source['baseline'])
        anchor = original_source['anchor'].astype(np.float64)
        assert int(original_source['epoch']) == item['epoch'] and float(original_source['alpha']) == item['alpha']
    selected = frozen['selection'][item['mode']]
    assert selected['epoch'] == item['epoch'] and selected['alpha'] == item['alpha']
    assert frozen['checkpoints'][str(Path(item['checkpoint_path']).resolve())] == item['checkpoint_sha256']
    for field in ('origin_unix','target_time_unix','actual'):
        np.testing.assert_array_equal(data[field], original_source[field])
    np.testing.assert_array_equal(data['original'], original_source['quantiles'])
    np.testing.assert_array_equal(data['target_time_unix'], data['origin_unix'][:, None]+np.arange(24)*3600)
    assert np.all(np.diff(data['origin_unix']) > 0)
    np.testing.assert_array_equal(data['quantile_levels'], np.array([.05,.1,.5,.9,.95]))
    y,b,f,q = (np.asarray(data[k], dtype=np.float64) for k in ('actual','baseline','original','upper_off'))
    assert q.dtype == np.float64 and np.all(np.isfinite(q)) and np.all(np.diff(q, axis=-1) >= 0)
    np.testing.assert_array_equal(q[..., :3], f[..., :3])
    # Independent derivation via baseline upper gaps around the preserved median.
    expected = f[..., 2:3]+(b[..., 3:]-b[..., 2:3])
    transform_bound = 8*np.finfo(np.float64).eps*(np.abs(f[..., 2:3])+np.abs(b[..., 3:])+np.abs(b[..., 2:3])+1)
    assert np.all(np.abs(q[..., 3:]-expected) <= transform_bound)
    if item['alpha'] == 0:
        np.testing.assert_array_equal(q, b)
        alpha_zero_cases += 1
    if anchor is not None:
        error = common_shift(b, anchor, f, item['alpha'])
        max_common_shift_error = max(max_common_shift_error, error)
        common_cases += 1
        assert item['common_history']['passed'] is True
    before, after, bs = primitives(y,f), primitives(y,q), primitives(y,b)
    for metric in ('mae','rmse','bias','under_q05','pinball_q05','pinball_q10','pinball_q50'):
        np.testing.assert_array_equal(before[metric], after[metric])
    group_masks = masks(y,data['origin_unix'])
    for group, mask in group_masks.items():
        saved = row_lookup[(item['case_id'], group)]
        count = int(mask.sum())
        assert int(saved['cells']) == count
        checked_groups += 1
        row = {key: item[key] for key in ('case_id','role','scope','variant','mode','representation','fold','region','seed')}
        row.update(group=group, cells=count, empty=count == 0)
        if count == 0:
            assert saved['empty'] == 'True' and saved['qs_delta'] == ''
            empty_groups += 1
            independent_rows.append(row)
            continue
        assert saved['empty'] == 'False'
        computed = {}
        for label, cells in (('before',before),('after',after)):
            computed.update({f'{label}_{key}': value for key,value in reductions(cells,mask).items()})
        for metric in ('mae','qs','pinball_q05','pinball_q10','pinball_q50','pinball_q90','pinball_q95'):
            change = (after[metric]-before[metric])[mask]
            computed.update({f'{metric}_delta': float(change.mean()),
                             f'{metric}_benefit': float((-change[change < 0]).sum()/count),
                             f'{metric}_harm': float(change[change > 0].sum()/count)})
        if group == 'all':
            computed.update({f'baseline_{key}': value for key,value in reductions(bs,mask).items()})
            for metric in ('mae','qs'):
                computed[f'original_vs_baseline_{metric}_delta'] = computed[f'before_{metric}']-computed[f'baseline_{metric}']
                computed[f'upper_off_vs_baseline_{metric}_delta'] = computed[f'after_{metric}']-computed[f'baseline_{metric}']
        for field,value in computed.items():
            error = abs(value-float(saved[field]))
            max_metric_error = max(max_metric_error,error)
            checked_metrics += 1
            assert error <= 1e-9, (item['case_id'],group,field,error)
        assert computed['mae_delta'] == computed['pinball_q05_delta'] == computed['pinball_q10_delta'] == computed['pinball_q50_delta'] == 0.
        for metric in ('mae','rmse','bias','under_q05','pinball_q05','pinball_q10','pinball_q50'):
            assert computed[f'before_{metric}'] == computed[f'after_{metric}']
        attributed_delta = (2/5)*(computed['pinball_q90_delta']+computed['pinball_q95_delta'])
        attribution_error = abs(computed['qs_delta']-attributed_delta)
        attribution_bound = 32*np.finfo(np.float64).eps*(
            abs(computed['before_qs'])+abs(computed['after_qs'])+abs(attributed_delta)+1)
        assert attribution_error <= attribution_bound, (item['case_id'],group,'upper-tail QS attribution')
        max_upper_tail_attribution_error = max(max_upper_tail_attribution_error, attribution_error)
        row.update(computed)
        row.update(point_equal=True,lower_quantiles_equal=True,
                   quantiles_ordered=True,common_history_passed=anchor is not None)
        independent_rows.append(row)
    if number % 15 == 0:
        print(f'independently verified {number}/{len(index)} cases', flush=True)
# Verify reported equal-seed regional and equal-region macro statistics from
# independently calculated case scores, including the descriptive seed spread.
aggregate_metrics_checked = 0
identity_fields = ('role','scope','variant','mode','representation','fold','group')
def same_identity(case, saved):
    return all(str(case.get(k, '')) == saved[k] for k in identity_fields)

for filename, is_macro in (('regional.csv',False),('macro.csv',True)):
    with (root/'audit'/filename).open() as stream:
        summaries = list(csv.DictReader(stream))
    for saved in summaries:
        cases = [r for r in independent_rows if same_identity(r,saved)
                 and (is_macro or r['region'] == saved['region'])]
        valid = [r for r in cases if not r['empty']]
        assert cases and (saved['empty'] == 'True') == (not valid)
        for label, number in saved.items():
            if not (label.endswith('_mean') or label.endswith('_seed_sd')) or number == '':
                continue
            suffix = '_mean' if label.endswith('_mean') else '_seed_sd'
            field = label[:-len(suffix)]
            if not valid or field not in valid[0]:
                continue
            if is_macro and suffix == '_mean':
                regions = sorted({r['region'] for r in valid})
                values = [np.mean([r[field] for r in valid if r['region']==region]) for region in regions]
                expected = float(np.mean(values))
            elif is_macro:
                seeds = sorted({r['seed'] for r in valid})
                values = [np.mean([r[field] for r in valid if r['seed']==seed]) for seed in seeds]
                expected = float(np.std(values,ddof=1)) if len(values)>1 else 0.
            else:
                values = [r[field] for r in valid]
                expected = float(np.mean(values)) if suffix == '_mean' else (
                    float(np.std(values,ddof=1)) if len(values)>1 else 0.)
            error = abs(expected-float(number))
            max_metric_error = max(max_metric_error,error)
            aggregate_metrics_checked += 1
            assert error <= 1e-9, (filename,saved['group'],label,error)
# Reconstruct the decision from independently recomputed PRIMARY rows only.
primary = [r for r in independent_rows if r['role']=='primary' and r['scope']=='formal' and
           r['variant']=='Full_joint' and r['mode']=='guarded' and r['representation']=='selected']
primary_all = [r for r in primary if r['group']=='all']
expected_pairs = {(region,seed) for region in cfg['regions'] for seed in cfg['seeds']}
assert len(primary_all) == cfg['expected_primary_cases'] == 9
assert {(r['region'],r['seed']) for r in primary_all} == expected_pairs
assert all(r['point_equal'] and r['lower_quantiles_equal'] and r['quantiles_ordered'] and r['common_history_passed'] for r in primary_all)
regional = []
for region in cfg['regions']:
    for group in ('all',*cfg['decision_price_groups']):
        selected = [r for r in primary if r['region']==region and r['group']==group]
        present = len(selected)==len(cfg['seeds']) and {r['seed'] for r in selected}==set(cfg['seeds'])
        valid = present and all(r['cells']>0 and not r['empty'] for r in selected)
        delta = float(np.mean([r['qs_delta'] for r in selected])) if valid else None
        regional.append(dict(region=region,group=group,qs_delta=delta,
                             passed=bool(valid and delta <= 1e-10),nonempty=valid))
all_deltas = [r['qs_delta'] for r in regional if r['group']=='all']
macro = float(np.mean(all_deltas)) if all(v is not None for v in all_deltas) else None
passed = all(r['passed'] for r in regional) and macro is not None and macro < -1e-10
decision = read(root/'audit/decision.json')
assert decision['passed'] == passed
assert decision['decision'] == ('eligible_for_prospective_L1_design' if passed else 'stop_global_upper_removal')
assert abs(decision['primary_macro_qs_delta']-macro) <= 1e-9
for row in regional:
    recorded = next(c for c in decision['criteria'] if c['name']=='regional_qs_nonincrease' and
                    c['region']==row['region'] and c['group']==row['group'])
    assert recorded['passed']==row['passed'] and recorded['nonempty']==row['nonempty']
    assert (recorded['qs_delta'] is None and row['qs_delta'] is None) or abs(recorded['qs_delta']-row['qs_delta']) <= 1e-9
result = dict(status='passed', run_id=manifest['run_id'],
    independent_implementation='Standalone NumPy piecewise pinball and interval arithmetic; no imports from audit/core/scorers',
    cases=len(index), primary_cases=len(primary_all), score_groups_checked=checked_groups,
    scalar_metrics_checked=checked_metrics, maximum_metric_abs_error=max_metric_error,
    aggregate_scalar_metrics_checked=aggregate_metrics_checked,
    all_q05_q10_q50_and_mae_unchanged=True, all_output_quantiles_ordered=True,
    all_point_rmse_bias_under_q05_and_lower_pinballs_unchanged=True,
    qs_delta_entirely_upper_pinball_verified=True,
    maximum_upper_tail_attribution_abs_error=max_upper_tail_attribution_error,
    alpha_zero_exact_cases=alpha_zero_cases, source_checkpoint_epoch_alpha_all_verified=True,
    same_checkpoint_common_history_cases=common_cases,
    missing_raw_formal_anchor_cases=len(index)-common_cases,
    max_common_shift_error_within_frozen_bound=max_common_shift_error,
    empty_score_groups_retained=empty_groups, artifact_hashes_verified=len(verification['artifact_sha256']),
    independent_decision=dict(passed=passed,decision=decision['decision'],primary_macro_qs_delta=macro,
                              regional_criteria=regional, supporting_modes_used=False),
    trained_models=0, original_artifacts_written=False,
    limitations=['This verifies a retrospective frozen intervention, not a retrained model or causal effect.',
                 'The numerical screen does not establish statistical noninferiority or new blind-test performance.',
                 'The nine formal raw cases retain the missing same-checkpoint Anchor evidence gap.'])
(root/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
