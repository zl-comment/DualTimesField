import json

import pytest

from scripts import update_latest_main_comparison as report


def fixture_rows():
    return [dict(model='Full_joint', role='latest_full_method', model_version='test',
                 input_scope='public_inputs_plus_mature_history_and_errors',
                 seed_scope='neural only', table='Q', pretraining_overlap='not_applicable', region=region, seed=seed, fallback=(seed == 2028),
                 **{metric: float(10 * si + ri) for metric in report.METRICS})
            for si, seed in enumerate(report.SEEDS) for ri, region in enumerate(report.REGIONS)]


def test_sd_is_across_region_macro_seeds_not_across_nine_cells():
    main, regional, per_seed = report.summarize(fixture_rows())
    assert len(main) == 1 and len(regional) == len(per_seed) == 3
    assert [row['mae'] for row in per_seed] == [1., 11., 21.]
    assert main[0]['mae_mean'] == 11.
    assert main[0]['mae_std'] == 10.  # sample SD, ddof=1
    assert main[0]['fallback_cells'] == 3
    assert main[0]['seed_scope'] == 'neural only'


def test_partial_or_duplicated_seed_grid_cannot_publish():
    rows = fixture_rows()
    with pytest.raises(ValueError, match='Incomplete three-seed'):
        report.summarize(rows[:-1])
    with pytest.raises(ValueError, match='Duplicate'):
        report.summarize(rows + rows[:1])


def test_incomplete_run_does_not_replace_existing_main_table(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    output = tmp_path / 'main'
    run.mkdir()
    output.mkdir()
    (run / 'progress.json').write_text(json.dumps(dict(stage='training_selection')))
    (output / 'main.csv').write_text('previous verified table\n')
    monkeypatch.setattr(report, 'RUN', run)
    monkeypatch.setattr(report, 'OUTPUT', output)
    with pytest.raises(ValueError, match='incomplete'):
        report.main()
    assert (output / 'main.csv').read_text() == 'previous verified table\n'


def test_point_table_does_not_fabricate_quantile_metrics():
    rows = fixture_rows()
    point = [dict(r, model='point_model', table='P', **{m: None for m in report.METRICS if m not in ('mae', 'rmse_pooled')}) for r in rows]
    main, regional, macro = report.summarize(rows + point)
    p = next(r for r in main if r['model'] == 'point_model')
    assert p['mae_mean'] == 11.
    assert p['quantile_score_proxy_mean'] is None
    assert p['coverage_80_std'] is None
    assert all(r['quantile_score_proxy'] is None for r in macro if r['model'] == 'point_model')


def test_snapshot_validation_ignores_worktree_but_rejects_snapshot_tampering(tmp_path):
    source = tmp_path / 'source/forecasting/lightgbm/model.py'
    source.parent.mkdir(parents=True)
    source.write_text('frozen code')
    protocol = tmp_path / 'protocol.md'
    protocol.write_text('frozen protocol')
    digest = report.experiment.c.base.digest
    manifest = {'source_sha256': {str(report.ROOT / 'forecasting/lightgbm/model.py'): digest(source)},
                'inputs_sha256': {}, 'protocol_sha256': digest(protocol)}
    report.verify_frozen_joint_manifest(manifest, tmp_path)
    source.write_text('tampered code')
    with pytest.raises(ValueError, match='Frozen source snapshot'):
        report.verify_frozen_joint_manifest(manifest, tmp_path)


def test_replacement_cannot_silently_fall_back_on_missing_evidence(tmp_path, monkeypatch):
    from scripts import update_latest_main_comparison as report
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    with pytest.raises(FileNotFoundError):
        report.validate_maintenance_source('timesfm25_xreg', {
            'run_root':'missing_run','manifest_sha256':'not-a-real-sha'})


def test_maintenance_source_binds_selected_config_and_complete_grid(tmp_path, monkeypatch):
    import yaml
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    run=tmp_path/'repair';run.mkdir()
    selection=tmp_path/'selection.json'
    selection.write_text(json.dumps({'accepted':True,'test_used_for_selection':False,'selected_ridge':1.}))
    digest=report.experiment.c.base.digest
    pin={'path':'selection.json','sha256':digest(selection)}
    (run/'maintenance_selection.json').write_text(selection.read_text())
    manifest={'models':['timesfm25_xreg'],'maintenance_selection_sha256':pin['sha256'],'jobs':[
        {'model':'timesfm25_xreg','region':r,'seed':s} for r in report.REGIONS for s in report.SEEDS]}
    (run/'jobs.json').write_text(json.dumps(manifest))
    config={'maintenance_selection':pin,'foundation_models':{'timesfm25_xreg':{'ridge':1.}}}
    (run/'config.yaml').write_text(yaml.safe_dump(config))
    record={'run_root':'repair','manifest_sha256':digest(run/'jobs.json'),'selection':pin}
    assert report.validate_maintenance_source('timesfm25_xreg',record)[0]==run
    config['foundation_models']['timesfm25_xreg']['ridge']=0.
    (run/'config.yaml').write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='candidate'):
        report.validate_maintenance_source('timesfm25_xreg',record)
    (run/'config.yaml').write_text(yaml.safe_dump({**config,'foundation_models':{'timesfm25_xreg':{'ridge':1.}}}))
    selection.write_text(json.dumps({'accepted':False,'test_used_for_selection':True,'selected_ridge':1.}))
    with pytest.raises(ValueError,match='selection changed'):
        report.validate_maintenance_source('timesfm25_xreg',record)
