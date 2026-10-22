"""Explicit correction provenance, complete-grid replacement, and no double count."""
import csv
import json

import pytest

from forecasting.aligned_expansion_report import build_multi_run_report, write_multi_run_report
from test_aligned_report import make_run, write_cell


def fixture_run(root, models, *, error=1., completed=True):
    manifest = make_run(root, models=models, completed=False)
    manifest.update(run_id=root.name, cache_hashes={}, source_hash=f'source-{root.name}',
                    config_hash=f'config-{root.name}')
    (root / 'jobs.json').write_text(json.dumps(manifest))
    if completed:
        for job in manifest['jobs']:
            write_cell(root, manifest, job, error=error)
    return manifest


def test_replacement_keeps_older_sources_audited_without_double_counting(tmp_path):
    old, new = tmp_path / 'original', tmp_path / 'repair'
    before = fixture_run(old, ('tree', 'ddnn_jsu'))
    fixture_run(new, ('ddnn_jsu',), error=100.)  # worse score must still be selected explicitly
    hashes = {p: p.read_bytes() for p in old.rglob('*') if p.is_file()}
    report = build_multi_run_report([old, new], replacements={'ddnn_jsu': new})
    assert report['comparison_available']
    assert report['progress']['source_completed_cells'] == 27
    assert report['progress']['completed_cells'] == report['progress']['expected_cells'] == 18
    assert report['progress']['superseded_cells'] == 9
    assert report['cross_run_alignment_validation'] == 'passed_all_27_artifacts'
    assert report['tables']['Q']['ddnn_jsu']['macro_regions']['mae'] == 100.
    assert report['tables']['Q']['tree']['macro_regions']['mae'] == 1.
    assert report['tables']['Q']['ddnn_jsu']['provenance']['run_id'] == 'repair'
    assert report['tables']['Q']['tree']['provenance']['run_id'] == 'original'
    assert report['tables']['Q']['ddnn_jsu']['supersedes'][0]['previous_source']['run_id'] == 'original'
    assert all(p.read_bytes() == content for p, content in hashes.items())
    path = tmp_path / 'report.json'
    write_multi_run_report(path, report)
    with (tmp_path / 'report_tables/per_seed.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 18
    # A superseded artifact is still audited; corruption must not disappear.
    (old / before['jobs'][9]['output'] / 'test.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash mismatch'):
        build_multi_run_report([old, new], replacements={'ddnn_jsu': new})


def test_reject_duplicate_without_explicit_source_and_invalid_replacements(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    fixture_run(old, ('ddnn_jsu', 'tree')); fixture_run(new, ('ddnn_jsu',))
    with pytest.raises(ValueError, match='explicit replacement'):
        build_multi_run_report([old, new])
    with pytest.raises(ValueError, match='duplicate model'):
        build_multi_run_report([old, new], replacements={'tree': old})
    with pytest.raises(ValueError, match='does not provide'):
        build_multi_run_report([old, new], replacements={'ddnn_jsu': tmp_path / 'absent'})
    with pytest.raises(ValueError, match='Duplicate run paths'):
        build_multi_run_report([old, old])


def test_incomplete_replacement_never_falls_back_to_old_score_or_keeps_stale_csv(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    fixture_run(old, ('ddnn_jsu', 'tree')); fixture_run(new, ('ddnn_jsu',), completed=False)
    report = build_multi_run_report([old, new], replacements={'ddnn_jsu': new})
    assert not report['comparison_available'] and 'tables' not in report
    assert report['progress']['expected_cells'] == 18
    assert report['progress']['completed_cells'] == 9
    directory = tmp_path / 'report_tables'; directory.mkdir()
    (directory / 'main.csv').write_text('stale')
    write_multi_run_report(tmp_path / 'report.json', report)
    assert not (directory / 'main.csv').exists()


def test_replacement_requires_same_grid_and_same_samples(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    fixture_run(old, ('ddnn_jsu',)); manifest = fixture_run(new, ('ddnn_jsu',))
    fewer = dict(manifest, jobs=manifest['jobs'][:-1])
    (new / 'jobs.json').write_text(json.dumps(fewer))
    with pytest.raises(ValueError, match='region/seed coverage'):
        build_multi_run_report([old, new], replacements={'ddnn_jsu': new})
    (new / 'jobs.json').write_text(json.dumps(manifest))
    for job in manifest['jobs']:
        write_cell(new, manifest, job, actual_offset=1.)
    with pytest.raises(ValueError, match='actual'):
        build_multi_run_report([old, new], replacements={'ddnn_jsu': new})
