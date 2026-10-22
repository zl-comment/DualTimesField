import json

import pytest

from scripts import update_latest_main_comparison as report


def fixture_rows():
    return [dict(model='Full_joint', role='latest_full_method', model_version='test',
                 input_scope='public_inputs_plus_mature_history_and_errors',
                 seed_scope='neural only', region=region, seed=seed, fallback=(seed == 2028),
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
