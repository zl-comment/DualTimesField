"""Synthetic and artifact-integrity checks; no training or evaluation-file access."""
from copy import deepcopy
import csv
import hashlib
import json

import numpy as np
import pytest

from forecasting.full_v5_mature_controls import (
    DAY, HOUR, SELECTION_START, SELECTION_END, apply_shared_beta,
    build_residual_controls, matched_raw_residual, recent_same_hour_residual,
    rolling_residual_28d, select_shared_beta,
)
from forecasting.full_v5_mature_evaluation import (
    apply_decision_rules, daily_paired_losses, paired_cell_losses,
    paired_day_block_inference, paired_scores, score_quantiles,
    REQUIRED_INTEGRITY_CHECKS,
)


def library(days=50):
    rng = np.random.default_rng(41)
    n = days*24
    origin = SELECTION_START+np.arange(n, dtype=np.int64)*HOUR
    median = rng.uniform(-50, 150, (n, 24)).astype(np.float32)
    baseline = median[..., None]+np.array([-30, -20, 0, 20, 30], dtype=np.float32)
    actual = median+rng.normal(size=(n, 24)).astype(np.float32)*25
    return dict(origin=origin, actual=actual, baseline=baseline,
        z=rng.normal(size=(n, 24, 16)).astype(np.float32),
        fold_id=(np.arange(n)//(24*20)).astype(np.int16))


def independent_candidates(data, row, method):
    t = int(data['origin'][row])
    mature = [j for j, origin in enumerate(data['origin']) if int(origin)+DAY <= t]
    if method == 'R28':
        return [j for j in mature if t-28*DAY < int(data['origin'][j])]
    same = [j for j in mature if (int(data['origin'][j])+10*HOUR)//HOUR % 24 == (t+10*HOUR)//HOUR % 24]
    if method == 'Recent16':
        return same[-16:]
    pool = [j for j in same if int(data['origin'][j]) >= t-730*DAY]
    selected = pool[-8:]
    distances = []
    for j in pool:
        # Source FP32 distances are essential to reproducing stable ties in
        # the existing prefilter. Candidate eligibility is independently looped.
        difference = data['z'][j][:, [1, 2, 3, 6, 7, 8]]-data['z'][row][:, [1, 2, 3, 6, 7, 8]]
        distances.append(float(np.mean(difference*difference)))
    for _, j in sorted(zip(distances, pool), key=lambda pair:(pair[0], pair[1])):
        if j not in selected:
            selected.append(j)
        if len(selected) == 16:
            break
    return selected


def test_candidate_ids_maturity_strict_window_and_residuals_match_independent_loops():
    data = library()
    rng = np.random.default_rng(9)
    rows = np.r_[0, 23, 24, 28*24, rng.choice(np.arange(30*24, 49*24), size=6, replace=False)]
    outputs = build_residual_controls(data, rows)
    for method, result in outputs.items():
        for position, row in enumerate(rows):
            selected = independent_candidates(data, row, method)
            count = int(result['counts'][position])
            assert count == len(selected)
            np.testing.assert_array_equal(result['candidate_ids'][position, :count], selected)
            assert np.all(result['candidate_ids'][position, count:] == -1)
            if selected:
                expected = np.median(data['actual'][selected].astype(np.float64)-data['baseline'][selected, :, 2].astype(np.float64), axis=0)
                np.testing.assert_array_equal(result['median'][position], expected)
                np.testing.assert_array_equal(result['candidate_fold_id'][position, :count], data['fold_id'][selected])
                assert np.all(result['candidate_age_hours'][position, :count] >= 24)
            else:
                assert result['fallback'][position] and np.all(result['median'][position] == 0)
    # At exactly 28 days, the lower boundary j=t-28d is excluded, while the
    # right boundary j=t-24h is included. Hourly R28 uses all origin hours.
    selected = outputs['R28']['candidate_ids'][3]
    assert 0 not in selected and 28*24-24 in selected


def test_matched_prefilter_preserves_730_day_left_boundary_and_recent_first_stable_ties():
    data = library(days=1)
    n = 750
    data = dict(origin=SELECTION_START+np.arange(n, dtype=np.int64)*DAY,
        actual=np.zeros((n, 24)), baseline=np.broadcast_to([-2., -1., 0., 1., 2.], (n, 24, 5)).copy(),
        z=np.zeros((n, 24, 16), dtype=np.float32), fold_id=np.zeros(n, dtype=np.int16))
    result = matched_raw_residual(data['origin'], data['actual'], data['baseline'], data['z'], np.array([749]))
    selected = result['candidate_ids'][0]
    np.testing.assert_array_equal(selected, [*range(741, 749), *range(19, 27)])
    assert data['origin'][selected[8]] == data['origin'][749]-730*DAY


def test_current_unmature_future_and_unavailable_labels_cannot_change_any_candidate_prediction():
    data = library()
    row = 35*24+7
    before = build_residual_controls(data, np.array([row]))
    changed = deepcopy(data)
    unavailable = changed['origin']+DAY > changed['origin'][row]
    changed['actual'][unavailable] = np.nan
    after = build_residual_controls(changed, np.array([row]))
    for method in before:
        for field in ('median', 'candidate_ids', 'candidate_fold_id', 'counts', 'fallback'):
            np.testing.assert_array_equal(before[method][field], after[method][field])
        np.testing.assert_array_equal(apply_shared_beta(data['baseline'][[row]], before[method]['median'], .75),
                                      apply_shared_beta(data['baseline'][[row]], after[method]['median'], .75))


def test_beta_zero_is_bitwise_original_and_common_shift_never_crosses_quantiles():
    baseline = np.array([[[-0., -0., 0., 0., 0.], [-100., -60., -20., 30., 90.]]], dtype=np.float32)
    median = np.array([[1e6, -1e6]])
    fallback = apply_shared_beta(baseline, median, 0)
    assert fallback.dtype == baseline.dtype and fallback.tobytes() == baseline.tobytes()
    corrected = apply_shared_beta(baseline, median, .75)
    np.testing.assert_array_equal(corrected-baseline.astype(np.float64), np.broadcast_to([[[22.5], [-22.5]]], baseline.shape))
    assert np.all(np.diff(corrected, axis=-1) >= 0)
    with pytest.raises(ValueError):
        apply_shared_beta(baseline, median, .75, clip=31)
    with pytest.raises(ValueError):
        apply_shared_beta(baseline, median, np.array([.25, .5]))


def test_simple_controls_are_one_deterministic_model_unaffected_by_neural_seed_metadata():
    data = library()
    rows = np.array([32*24, 34*24+3])
    first = build_residual_controls(dict(data, neural_seed=2026), rows)
    second = build_residual_controls(dict(data, neural_seed=2028), rows)
    assert set(first) == {'R28', 'Recent16', 'Matched16'}
    for arm in first:
        for field in first[arm]:
            np.testing.assert_array_equal(first[arm][field], second[arm][field])
        assert apply_shared_beta(data['baseline'][rows], first[arm]['median'], .5).shape == (len(rows), 24, 5)


def test_beta_is_one_purged_h1_global_choice_and_ties_use_zero():
    origin = SELECTION_START+np.arange(3, dtype=np.int64)*HOUR
    baseline = np.broadcast_to([-2., -1., 0., 1., 2.], (3, 24, 5)).copy()
    regions = {region:dict(origin=origin, actual=np.full((3, 24), actual), baseline=baseline,
                          median=np.full((3, 24), 4.)) for region,actual in [('NSW1', 1.), ('QLD1', 3.), ('TAS1', 5.)]}
    result = select_shared_beta(regions)
    # Independently compute the cross-region normalized two-loss objective.
    expected = []
    for beta in (0., .25, .5, .75, 1.):
        losses = []
        for data in regions.values():
            q = data['baseline']+beta*4
            error = data['actual'][..., None]-q
            pb = np.where(error >= 0, error*np.array([.05, .1, .5, .9, .95]), -error*(1-np.array([.05, .1, .5, .9, .95])))
            losses.append([np.abs(error[..., 2]).mean(), 2*pb.mean()])
        expected.append(np.mean(losses, axis=0))
    feasible = [i for i,x in enumerate(expected) if np.all(x <= expected[0]+1e-10)]
    objectives = [sum(expected[i]/expected[0]) for i in feasible]
    chosen = min((i for i,o in zip(feasible, objectives) if o <= min(objectives)+1e-12))
    assert result['beta'] == (0., .25, .5, .75, 1.)[chosen]
    assert len(result['curve']) == 5 and not isinstance(result['beta'], dict)
    for data in regions.values():
        data['median'][:] = 0
    assert select_shared_beta(regions)['beta'] == 0.
    invalid = deepcopy(regions)
    invalid['NSW1']['origin'][-1] = SELECTION_END-DAY
    with pytest.raises(ValueError):
        select_shared_beta(invalid)


def test_scoring_matches_archived_raw_price_convention_and_gain_harm_close():
    actual = np.array([[-80., 0., 2., 1500.]])
    baseline = np.broadcast_to([-20., -10., 0., 10., 20.], (1, 4, 5)).copy()
    candidate = baseline+np.array([[[-25.], [0.], [5.], [30.]]])
    scores = score_quantiles(actual, baseline)
    errors = actual[..., None]-baseline
    losses = np.where(errors>=0, errors*np.array([.05,.1,.5,.9,.95]), -errors*(1-np.array([.05,.1,.5,.9,.95])))
    assert scores['mae'] == np.abs(actual).mean()
    assert scores['qs'] == 2*losses.mean()
    paired = paired_scores(actual, baseline, candidate)
    for loss in ('mae', 'qs'):
        assert paired[f'{loss}_delta'] == pytest.approx(paired[f'{loss}_harm']-paired[f'{loss}_benefit'])
    assert paired['harmful_overshoot_rate'] == .25


def test_daily_sums_preserve_purged_boundary_cell_weighting():
    start = SELECTION_START
    origins = start+np.r_[np.arange(24), 24]*HOUR
    y = np.zeros((len(origins), 24))
    baseline = np.broadcast_to([-2., -1., 0., 1., 2.], (*y.shape, 5)).copy()
    prediction = baseline.copy()
    prediction[:24] += 1.
    prediction[24:] += 25.
    daily = daily_paired_losses(origins, y, baseline, prediction)
    np.testing.assert_array_equal(daily['cells'], [24*24, 24])
    assert daily['after_mae_sum'].sum()/daily['cells'].sum() == pytest.approx(49/25)
    assert daily['after_mae_sum'].sum()/daily['cells'].sum() == pytest.approx(paired_scores(y, baseline, prediction)['after_mae'])
    assert np.mean(daily['after_mae_sum']/daily['cells']) != pytest.approx(49/25)


def test_day_bootstrap_synchronizes_regions_and_does_not_resample_neural_seeds():
    days = 35
    dates = SELECTION_START+np.arange(days)*DAY
    effect = np.arange(days, dtype=float)-17
    counts = np.full(days, 576)
    daily = {}
    for region, sign in [('NSW1', 1), ('QLD1', -1)]:
        daily[region] = dict(date_unix=dates, cells=counts,
            before_mae_sum=100*counts, after_mae_sum=(100+sign*effect)*counts,
            before_qs_sum=50*counts, after_qs_sum=(50+sign*effect)*counts)
    result = paired_day_block_inference(daily)
    for block in result['blocks']:
        assert block['available'] and block['replicates']==2000
        for metric in ('mae', 'qs'):
            assert block['macro'][metric] == dict(estimate=0., lower=0., upper=0.)
            assert block['regional']['NSW1'][metric]['upper'] > 0
            assert block['regional']['QLD1'][metric]['lower'] < 0
    assert not result['decision_uses_confidence_interval']
    with pytest.raises(ValueError):
        paired_day_block_inference(daily, dict(bootstrap_replicates=10))
    misaligned = deepcopy(daily)
    misaligned['QLD1']['date_unix'] = dates+DAY
    with pytest.raises(ValueError):
        paired_day_block_inference(misaligned)


def decision_fixture():
    cfg = dict(regions=['NSW1', 'QLD1', 'TAS1'], required_integrity_checks=list(REQUIRED_INTEGRITY_CHECKS))
    rows = [dict(region=region, arm='R28', cells=24, empty=False,
        before_mae=10., after_mae=9.9, before_qs=8., after_qs=8.,
        before_coverage80=.7, after_coverage80=.7, before_coverage90=.85, after_coverage90=.85)
        for region in cfg['regions']]
    integrity = {key: True for key in cfg['required_integrity_checks']}
    return cfg, rows, integrity


def test_l0_requires_positive_beta_integrity_region_losses_and_coverage_without_diagnostic_rescue():
    cfg, rows, integrity = decision_fixture()
    assert apply_decision_rules(rows, .5, integrity, cfg)['passed']
    assert not apply_decision_rules(rows, 0., integrity, cfg)['passed']
    assert not apply_decision_rules(rows, .5, {}, cfg)['passed']
    altered = deepcopy(rows)
    altered[0]['after_mae'] = 10.101
    altered[1]['after_mae'] = 1.
    assert not apply_decision_rules(altered, .5, integrity, cfg)['passed']
    altered = deepcopy(rows)
    altered[0]['after_coverage90'] = .839
    assert not apply_decision_rules(altered, .5, integrity, cfg)['passed']
    # Diagnostic or Full wins, price slices and copied seed rows cannot replace
    # exactly one deterministic R28 prediction per expected region.
    diagnostic = [dict(row, arm='Matched16', after_mae=1., after_qs=1.) for row in rows]
    assert not apply_decision_rules(rows[1:]+diagnostic, .5, integrity, cfg)['passed']
    assert not apply_decision_rules(rows+[dict(rows[0], seed=2027)], .5, integrity, cfg)['passed']
    equal = [dict(row, after_mae=10., after_qs=8.) for row in rows]
    assert not apply_decision_rules(equal, .5, integrity, cfg)['passed']


@pytest.mark.parametrize('start,end', [('2022-01-01', '2022-07-01'),
    ('2022-07-01', '2023-01-01'), ('2023-01-01', '2024-01-01'), ('2024-01-01', '2025-01-01')])
def test_purge_excludes_whole_24_hour_label_windows_at_halfyear_and_year_boundaries(start, end):
    from forecasting.full_v5_mature_sources import purged_period_rows, timestamp_aest
    left, right = timestamp_aest(start), timestamp_aest(end)
    # Synthetic integer timestamps only: this test never opens annual arrays.
    origin = np.array([left-HOUR, left, right-DAY-HOUR, right-DAY, right-HOUR, right], dtype=np.int64)
    np.testing.assert_array_equal(purged_period_rows(origin, start, end), [1, 2])
    assert np.all(origin[purged_period_rows(origin, start, end)]+DAY < right)


def test_baseline_archive_reproduction_and_old_source_hashes_fail_on_changes(tmp_path):
    from forecasting.full_v5_mature_sources import _score_check, SourceRegistry
    y = np.array([[-10., 0., 10., 1000.]])
    q = np.broadcast_to([-20., -5., 0., 5., 20.], (*y.shape, 5)).copy()
    expected = dict(mae=255., qs=136.6, coverage_80=.25, coverage_90=.75, bias=-250.)
    # Pinball is independently accumulated at each supplied quantile here.
    expected['qs'] = 2*sum(sum((tau*(float(yy)-float(qq)) if yy>=qq else (1-tau)*(float(qq)-float(yy)))
        for tau,qq in zip((.05,.1,.5,.9,.95),quantiles)) for yy,quantiles in zip(y.ravel(),q.reshape(-1,5)))/(y.size*5)
    assert _score_check(y, q, expected, 'synthetic/B/H2')['passed']
    with pytest.raises(ValueError):
        _score_check(y, q, dict(expected, mae=expected['mae']+.01), 'synthetic/B/H2')
    old = tmp_path/'old_predictions.npz'
    old.write_bytes(b'original frozen output')
    digest = hashlib.sha256(old.read_bytes()).hexdigest()
    registry = SourceRegistry(tmp_path)
    assert registry.register(old, digest) == digest
    assert registry.verify_unchanged()==1
    with pytest.raises(ValueError):
        registry.register(old, '0'*64)
    old.write_bytes(b'changed output')
    with pytest.raises(ValueError):
        registry.verify_unchanged()


def test_l0_failure_or_wrong_identity_blocks_l1_before_any_array_open(tmp_path, monkeypatch):
    from forecasting.full_v5_mature_sources import load_and_verify_sources, require_l1_permission
    def forbidden_load(*args, **kwargs):
        raise AssertionError('An array was opened before passing L0 permission')
    monkeypatch.setattr(np, 'load', forbidden_load)
    with pytest.raises(PermissionError):
        load_and_verify_sources({}, stage='L1')
    path = tmp_path/'L0_decision.json'
    base = dict(stage='L0', primary_arm='R28', passed=True, run_id='same-frozen-run')
    for change in (dict(passed=False), dict(stage='selection'), dict(primary_arm='Matched16'), dict(run_id='different-run')):
        path.write_text(json.dumps(dict(base, **change)))
        token = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), run_id='same-frozen-run')
        with pytest.raises(PermissionError):
            load_and_verify_sources({}, stage='L1', l0_decision=token)
    path.write_text(json.dumps(base))
    token = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), run_id='same-frozen-run')
    assert require_l1_permission(token)==base
    path.write_text(json.dumps(dict(base, passed=False)))
    with pytest.raises(PermissionError):
        require_l1_permission(token)


def test_synthetic_l0_forward_saves_single_simple_arms_empty_slices_and_day_evidence(tmp_path):
    from forecasting.full_v5_mature_forward import evaluate_forward_stage
    from forecasting.full_v5_mature_sources import _score_check, timestamp_aest, purged_period_rows, require_l1_permission
    start = timestamp_aest('2022-07-01')
    origins = start+np.arange(-40, 35, dtype=np.int64)*DAY
    query = purged_period_rows(origins, '2022-07-01', '2023-01-01')
    assert len(query) == 35
    baseline = np.broadcast_to(np.array([-30., -20., 0., 20., 30.], dtype=np.float32),
                               (len(origins), 24, 5)).copy()
    actual = np.full((len(origins), 24), 5., dtype=np.float32)
    cfg = dict(regions=['NSW1', 'QLD1', 'TAS1'], seeds=[2026, 2027, 2028],
               required_integrity_checks=list(REQUIRED_INTEGRITY_CHECKS))
    loaded = dict(regions={}, sources=[], metadata=dict(score_reproduction=[]))
    for region in cfg['regions']:
        loaded['regions'][region] = dict(origin=origins.copy(), actual=actual.copy(),
            baseline=baseline.copy(), z=np.zeros((len(origins), 24, 16), dtype=np.float32),
            fold_id=(origins>=start).astype(np.int16), query_rows=query.copy(),
            full_reference={seed:dict(origin=origins[query].copy(), quantiles=baseline[query].copy(),
                                     epoch=0, alpha=0., source='synthetic') for seed in cfg['seeds']})
        loaded['metadata']['score_reproduction'].append(_score_check(actual[query], baseline[query],
            dict(mae=5., qs=3.8, coverage_80=1., coverage_90=1., bias=-5.), f'{region}/B/2022H2'))
    old = tmp_path/'existing_result.txt'
    old.write_bytes(b'unchanged prior evidence')
    expected_hash = hashlib.sha256(old.read_bytes()).hexdigest()
    barrier_calls = []
    def barrier(sources):
        assert sources == loaded['sources']
        assert hashlib.sha256(old.read_bytes()).hexdigest() == expected_hash
        barrier_calls.append(True)
        return dict(protected_files_unchanged=1, source_files_unchanged=2, source_files_verified=0)
    decision = evaluate_forward_stage(tmp_path, cfg, loaded, 1., 'L0', barrier, 'synthetic-frozen-run')
    assert barrier_calls == [True]
    assert decision['passed'] and decision['primary_arm']=='R28' and decision['stage']=='L0'
    assert decision['run_id']=='synthetic-frozen-run' and decision['decision']=='proceed_to_frozen_L1'
    decision_path = tmp_path/'L0'/'decision.json'
    permission = require_l1_permission(dict(path=str(decision_path),
        sha256=hashlib.sha256(decision_path.read_bytes()).hexdigest(), run_id=decision['run_id']))
    assert permission == decision == json.loads(decision_path.read_text())
    integrity = json.loads((tmp_path/'L0'/'integrity.json').read_text())
    assert set(integrity['checks']) == set(REQUIRED_INTEGRITY_CHECKS)
    assert all(integrity['checks'].values())
    for region in cfg['regions']:
        with np.load(tmp_path/'L0'/region/'predictions.npz', allow_pickle=False) as saved:
            for arm in ('B', 'R28', 'Recent16', 'Matched16'):
                assert saved[arm].shape == (35, 24, 5)
            assert not any(name.startswith(('R28_seed', 'Recent16_seed', 'Matched16_seed')) for name in saved.files)
            assert len([name for name in saved.files if name.startswith('Full_joint_seed')]) == 3
            np.testing.assert_array_equal(saved['R28'][..., 2], np.full((35, 24), 5.))
        with np.load(tmp_path/'L0'/region/'R28_daily_losses.npz', allow_pickle=False) as daily:
            assert len(daily['date_unix']) == 35
            np.testing.assert_array_equal(daily['cells'], np.full(35, 24))
            assert daily['before_mae_sum'].sum()==35*24*5 and daily['after_mae_sum'].sum()==0
    with (tmp_path/'L0'/'paired_groups.csv').open() as stream:
        groups = list(csv.DictReader(stream))
    main = [row for row in groups if row['arm']=='R28' and row['group']=='all']
    assert len(main)==3 and {row['region'] for row in main}==set(cfg['regions'])
    empty = [row for row in groups if row['group'] in ('quarter/2022Q1', 'quarter/2022Q2', 'quarter/2022Q4', 'actual/1000_plus')]
    assert len(empty)==3*7*4
    assert all(row['empty']=='True' and row['cells']=='0' and row['mae_delta']=='' for row in empty)
    inference = json.loads((tmp_path/'L0'/'paired_inference.json').read_text())
    assert set(inference['2022H2']) == {'R28_minus_B', 'Recent16_minus_B', 'Matched16_minus_B',
                                      'Recent16_minus_R28', 'Matched16_minus_R28'}
    assert all(block['available'] for block in inference['2022H2']['R28_minus_B']['blocks'])


def test_existing_output_run_preserves_prior_artifacts(tmp_path, monkeypatch):
    from scripts import compare_mature_residual_controls as runner
    # The failing invocation must not install process-wide timers or resource
    # limits in pytest, or reach any real source lookup or array loading.
    monkeypatch.setattr(runner.resource, 'setrlimit', lambda *args: None)
    monkeypatch.setattr(runner.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(runner.signal, 'alarm', lambda *args: None)
    def forbidden_freeze(*args, **kwargs):
        raise AssertionError('Existing output must be rejected before freezing inputs')
    monkeypatch.setattr(runner, 'freeze', forbidden_freeze)
    marker = tmp_path/'marker.txt'
    original = b'prior run evidence must remain unchanged\n'
    marker.write_bytes(original)
    with pytest.raises(FileExistsError):
        runner.run(tmp_path, dict(max_cpu_core_hours=8, max_wall_hours=3))
    assert marker.read_bytes() == original
    assert not (tmp_path/'failure.json').exists()
    assert not (tmp_path/'resources.json').exists()
    assert set(tmp_path.iterdir()) == {marker}
