"""No-training checks of saturated event allocation and diagnostic boundaries."""
import inspect
import json

import numpy as np
import pytest
import torch

from forecasting.full_v5_event_allocation import audit_event_pair, event_upper_only
from scripts.summarize_event_allocation_episodes import diagnose_series, episode_ranges


def test_event_allocation_copies_exact_inputs_including_negative_prices_without_labels():
    trend = np.array([[-90., -70., -40., -20., -10.], [1., 1., 1., 1., 1.]])
    full = trend+np.array([[25.], [0.]])
    before_full, before_trend = full.copy(), trend.copy()
    result = event_upper_only(full, trend)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result[..., :3], trend[..., :3])
    np.testing.assert_array_equal(result[..., 3:], full[..., 3:])
    np.testing.assert_array_equal(full, before_full)
    np.testing.assert_array_equal(trend, before_trend)
    assert np.all(np.diff(result, axis=-1) >= 0)
    assert list(inspect.signature(event_upper_only).parameters) == ['full', 'trend']
    with pytest.raises(TypeError):
        event_upper_only(full, trend, actual=np.zeros(2))
    with pytest.raises((ValueError, AssertionError)):
        event_upper_only(full, None)
    with pytest.raises((ValueError, AssertionError)):
        event_upper_only(trend, full)


@pytest.mark.parametrize('bounded', [False, True])
def test_real_forward_retains_retrieval_and_saturation_when_only_event_allocation_changes(bounded):
    from forecasting.full_v5_anchor_balance.confirmation_model import AsymmetricScaleModel
    from forecasting.full_v5_calibration_robustness.model import BoundedCalibrationModel
    from test_full_v5_anchor_architecture_experiment import synthetic_store
    torch.set_num_threads(2)
    torch.manual_seed(103)
    model = (BoundedCalibrationModel if bounded else AsymmetricScaleModel)().eval()
    batch = synthetic_store().batch([7, 8])
    batch['memory_actual'][..., 8] += 3000  # A mature historical event, not the query outcome.
    with torch.no_grad():
        model.calibration[-1].bias.copy_(torch.tensor([.12, -.2, .3]))
        model.gates[-1].bias.copy_(torch.tensor([1., 1.]))
        full = model(batch)
        trend = model(batch, event_off=True)
    for key in ('anchor', 'trend', 'gates', 'weights'):
        torch.testing.assert_close(full[key], trend[key], rtol=0, atol=0)
    assert torch.all(full['event'] >= 0) and torch.any(full['event'] > 0)
    assert torch.count_nonzero(trend['event']) == 0
    saturated_event = full['correction']-trend['correction']
    assert torch.all(saturated_event >= 0)
    # The actual saturated event is not its pre-tanh proposal. Adding that
    # proposal directly is a materially different intervention.
    assert torch.max(torch.abs(saturated_event-full['event'])).item() > 1.
    baseline, f, t = (x.numpy() for x in (batch['baseline'], full['prediction'], trend['prediction']))
    assert audit_event_pair(baseline, f, t, 1.)['passed']
    allocated = event_upper_only(f, t)
    np.testing.assert_array_equal(allocated[..., :3], t[..., :3])
    np.testing.assert_array_equal(allocated[..., 3:], f[..., 3:])
    # Independently express the new forward from the actual model's returned
    # corrections; final FP32 addition can only agree within its fixed bound.
    expected = trend['anchor'].double()+trend['correction'].double()[..., None]
    expected[..., 3:] += saturated_event.double()[..., None]
    bound = 8*np.finfo(np.float32).eps*(np.abs(allocated)+np.abs(trend['anchor'].numpy())+1)
    assert np.all(np.abs(allocated-expected.numpy()) <= bound)


def test_saturation_allows_event_difference_above_150_and_suppresses_already_saturated_trend():
    anchor = np.broadcast_to([-60., -30., -10., 30., 80.], (3, 5)).copy()
    t = np.array([-1500., 0., 1500.])
    e = np.array([3000., 10., 1500.])
    trend = anchor+150*np.tanh(t[:, None]/150)
    full = anchor+150*np.tanh((t+e)[:, None]/150)
    result = audit_event_pair(anchor, full, trend, 1.)
    assert result['passed'] and result['event_increment_max'] > 299.
    assert 0 <= (full-trend)[2, 2] < 1e-5
    assert abs((full-trend)[1, 2]-e[1]) > .01
    upper_only = event_upper_only(full, trend)
    np.testing.assert_array_equal(upper_only[..., 3:], full[..., 3:])
    with pytest.raises((ValueError, AssertionError)):
        audit_event_pair(anchor, trend+301., trend, 1.)


@pytest.mark.parametrize('alpha', [0., .25, .5, 1.])
def test_original_alpha_is_retained_and_zero_alpha_requires_exact_baseline(alpha):
    baseline = np.array([[-100., -50., -10., 50., 200.]], dtype=np.float32).astype(np.float64)
    trend = baseline+4.25
    full = trend+23.5
    blend = lambda q: baseline if alpha == 0 else q if alpha == 1 else (1-alpha)*baseline+alpha*q
    selected_full, selected_trend = blend(full), blend(trend)
    assert audit_event_pair(baseline, selected_full, selected_trend, alpha)['passed']
    np.testing.assert_array_equal(event_upper_only(selected_full, selected_trend), blend(event_upper_only(full, trend)))
    if alpha == 0:
        np.testing.assert_array_equal(event_upper_only(selected_full, selected_trend), baseline)
        with pytest.raises((ValueError, AssertionError)):
            audit_event_pair(baseline, full, trend, 0.)


def test_pair_bound_accepts_fp32_addition_and_rejects_noncommon_or_negative_event():
    baseline = np.array([[-7000., -30., 1., 15000., 30000.]], dtype=np.float32)
    anchor = baseline+np.float32(.3137)
    trend, full = anchor+np.float32(.1173), anchor+np.float32(.3175)
    assert np.ptp((full-trend).astype(np.float64)) > 0
    assert audit_event_pair(baseline, full, trend, 1.)['passed']
    bad = full.astype(np.float64)
    bad[..., 4] += 1.
    with pytest.raises((ValueError, AssertionError)):
        audit_event_pair(baseline, bad, trend, 1.)
    bad = full.copy()
    bad[..., 0] = np.nextafter(trend[..., 0], np.float32(-np.inf))
    with pytest.raises((ValueError, AssertionError)):
        audit_event_pair(baseline, bad, trend, 1.)


def test_saved_pair_requires_same_checkpoint_epoch_alpha_and_information_set():
    from forecasting.full_v5_event_allocation_audit import validate_saved_pair
    entry = dict(checkpoint_sha256='same-checkpoint', epoch=7, alpha=.5, output_mode='guarded')
    metadata = dict(checkpoint_sha256='same-checkpoint', alpha=.5, input_hash='inputs',
        sample_hash='samples', preprocessing_hash='preprocess', output_mode='guarded', region='NSW1',
        seed=2026, split='test', model_version='frozen', protocol_id='frozen-protocol',
        prediction_space='price', price_transform='identity', tree_seed=2026, source='same')
    assert validate_saved_pair(entry, entry.copy(), metadata, metadata.copy())
    for field, wrong in (('checkpoint_sha256', 'another'), ('epoch', 8), ('alpha', 1.)):
        with pytest.raises(ValueError):
            validate_saved_pair(entry, dict(entry, **{field: wrong}), metadata, metadata.copy())
    for field in ('input_hash', 'sample_hash', 'preprocessing_hash'):
        with pytest.raises(ValueError):
            validate_saved_pair(entry, entry.copy(), metadata, dict(metadata, **{field: 'another'}))


def test_saved_case_never_routes_on_outcomes_and_preserves_comparison_invariants(tmp_path):
    from forecasting.full_v5_event_allocation_audit import _save_case
    from scripts.summarize_event_allocation_episodes import summarize
    baseline = np.broadcast_to([-30., -20., 0., 20., 30.], (1, 24, 5)).copy()
    trend, full = baseline+3., baseline+8.
    origins = np.array([1672531200], dtype=np.int64)
    targets = origins[:, None]+np.arange(24)*3600
    actual = np.full((1, 24), 3.)
    identity = dict(case_id='formal/Full_joint/guarded/seed2026/NSW1',
        role='primary', scope='formal', variant='Full_joint', mode='guarded', representation='selected',
        region='NSW1', seed=2026, fold='2023-2024')
    selection = dict(epoch=7, alpha=1., checkpoint_sha256='unchanged-checkpoint', same_checkpoint=True)
    rows, first = _save_case(tmp_path/'original_labels'/'audit', identity, actual, baseline, full,
                            trend, origins, targets, selection, 1e-10)
    _, changed = _save_case(tmp_path/'changed_labels'/'audit', identity, actual+100000., baseline,
                           full, trend, origins, targets, selection, 1e-10)
    with np.load(first['path'], allow_pickle=False) as a, np.load(changed['path'], allow_pickle=False) as b:
        np.testing.assert_array_equal(a['upper_only'], b['upper_only'])
        assert str(a['metadata_json']) == str(b['metadata_json'])
        meta = json.loads(str(a['metadata_json']))
        assert meta['epoch']==7 and meta['alpha']==1. and not meta['outcome_routing']
    for row in rows:
        if row['empty']:
            assert row['cells']==0 and 'qs_delta' not in row
        elif row['comparison']=='upper_only_vs_original':
            assert row['pinball_q90_delta']==row['pinball_q95_delta']==0.
            assert row['before_over_q95']==row['after_over_q95']
        elif row['comparison']=='upper_only_vs_event_off':
            assert row['mae_delta']==row['pinball_q05_delta']==row['pinball_q10_delta']==row['pinball_q50_delta']==0.
    # The standalone diagnostic keeps all four arms and both quantiles for
    # each lead even when this small fixture contains no actual episodes.
    manifest = tmp_path/'original_labels'/'audit'/'prediction_manifest.json'
    manifest.write_text(json.dumps([first]))
    diagnostic = summarize(tmp_path/'original_labels', {})
    assert diagnostic['summary_rows']==24*4*2 and diagnostic['actual_episode_peak_rows']==0
    assert diagnostic['empty_episode_groups']==24*4*2
    with pytest.raises(FileExistsError):
        summarize(tmp_path/'original_labels', {})


def decision_fixture():
    cfg = dict(regions=['NSW1', 'QLD1', 'TAS1'], seeds=[2026, 2027, 2028],
               decision_price_groups=['actual/negative', 'actual/0_to_300', 'actual/0_to_100', 'actual/1000_plus'])
    rows = []
    for region in cfg['regions']:
        for seed in cfg['seeds']:
            for comparison in ('upper_only_vs_original', 'event_off_vs_original', 'upper_only_vs_event_off'):
                for group in ('all', *cfg['decision_price_groups']):
                    rows.append(dict(role='primary', scope='formal', variant='Full_joint',
                        mode='guarded', representation='selected', region=region, seed=seed,
                        group=group, comparison=comparison, qs_delta=-.1, mae_delta=-.01,
                        cells=100, empty=False, before_coverage80=.75, after_coverage80=.76,
                        before_coverage90=.85, after_coverage90=.86, same_checkpoint=True,
                        event_pair_passed=True, lower_equals_trend=True, upper_equals_full=True,
                        quantiles_ordered=True))
    return cfg, rows


def test_decision_requires_region_negative_price_and_both_comparators():
    from forecasting.full_v5_event_allocation_audit import evaluate_decision
    cfg, rows = decision_fixture()
    assert evaluate_decision(rows, cfg)['passed']
    for r in rows:
        if r['region']=='NSW1' and r['comparison']=='upper_only_vs_original' and r['group']=='actual/negative':
            r['mae_delta'] = .01
    assert not evaluate_decision(rows, cfg)['passed']
    cfg, rows = decision_fixture()
    for r in rows:
        if r['comparison']=='upper_only_vs_event_off':
            r['qs_delta'] = 0.
    assert not evaluate_decision(rows, cfg)['passed']


def test_decision_rejects_coverage_damage_empty_tail_or_missing_primary():
    from forecasting.full_v5_event_allocation_audit import evaluate_decision
    cfg, rows = decision_fixture()
    for r in rows:
        if r['region']=='QLD1' and r['comparison']=='upper_only_vs_original' and r['group']=='all':
            r['after_coverage90'] = .83
    assert not evaluate_decision(rows, cfg)['passed']
    cfg, rows = decision_fixture()
    for r in rows:
        if r['region']=='TAS1' and r['group']=='actual/1000_plus':
            r.update(cells=0, empty=True)
            r.pop('qs_delta')
    assert not evaluate_decision(rows, cfg)['passed']
    cfg, rows = decision_fixture()
    raw = [dict(r, role='sensitivity', mode='raw', variant='Full_joint_raw', qs_delta=-10.) for r in rows]
    assert not evaluate_decision([r for r in rows if r['region']!='NSW1']+raw, cfg)['passed']


def test_episode_counting_splits_gaps_and_aest_years_and_rejects_duplicate_targets():
    start = int(np.datetime64('2023-12-31T13:00:00').astype('datetime64[s]').astype(np.int64))
    times = start+np.array([0, 1, 2, 4])*3600  # 23:00 AEST, midnight, 01:00, 03:00.
    assert episode_ranges(times, np.ones(4, dtype=bool)) == [(0, 1), (1, 3), (3, 4)]
    with pytest.raises(ValueError):
        episode_ranges(np.array([0, 0]), np.ones(2, dtype=bool))


def test_episode_diagnostics_use_hourly_confusion_and_preserve_empty_peak_errors():
    times = np.arange(7)*3600
    y = np.array([0., 1200., 2000., 0., 0., 1500., 0.])
    q = np.array([1100., 1300., 1250., 0., 0., 900., 0.])
    result, peaks = diagnose_series(times, y, q)
    assert (result['hourly_tp'], result['hourly_fp'], result['hourly_fn']) == (2, 1, 1)
    assert (result['actual_episodes'], result['detected_actual_episodes'], result['missed_actual_episodes']) == (2, 1, 1)
    # A forecast run beginning one hour early still overlaps the first actual
    # episode; it contributes an hourly FP, not a wholly false-alarm episode.
    assert result['false_alarm_predicted_episodes'] == 0
    assert peaks[0]['prediction_peak'] == 1300. and peaks[0]['peak_time_error_hours'] == -1.
    empty, detail = diagnose_series(times, np.zeros(7), np.zeros(7))
    assert empty['actual_episodes'] == 0 and not detail
    assert empty['mean_peak_amplitude_error'] is None and empty['mean_abs_peak_time_error_hours'] is None
