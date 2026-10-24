import copy
import numpy as np
import pytest
from forecasting.foundation_audit import validate_audits
from forecasting.foundation_candidate import _append_model


def fixture_audit():
    manifest = {'jobs': [{'model': 'm', 'seed': 2026, 'region': 'NSW1'}]}
    audits = {'m': {'model': 'm', 'cells': {'2026/NSW1': {
        'state': 'completed', 'rescore_ok': True, 'chunk_reassembly_ok': True}},
        'identity_within_seed_consistent': True, 'replay': {'passed': True}}}
    return manifest, audits


@pytest.mark.parametrize('failure', ['missing_model', 'missing_cell', 'failed_replay', 'no_replay', 'pending', 'bad_chunks'])
def test_collection_rejects_incomplete_or_failed_audits(failure):
    manifest, audits = fixture_audit()
    if failure == 'missing_model': audits.clear()
    elif failure == 'missing_cell': audits['m']['cells'].clear()
    elif failure == 'failed_replay': audits['m']['replay']['passed'] = False
    elif failure == 'no_replay': audits['m']['replay'] = None
    elif failure == 'pending': audits['m']['cells']['2026/NSW1']['state'] = 'pending'
    else: audits['m']['cells']['2026/NSW1']['chunk_reassembly_ok'] = False
    with pytest.raises(ValueError): validate_audits(manifest, audits)


def test_complete_audit_is_accepted():
    validate_audits(*fixture_audit())


def test_candidate_std_is_across_seed_macros_not_regions():
    row = {'regions': {}}
    for region, offset in zip(('NSW1', 'QLD1', 'TAS1'), (0., 100., 200.)):
        row['regions'][region] = {'scores': {
            'seed_summary': {'mae': {'mean': offset + 2., 'std': 1.}},
            'per_seed': {str(seed): {'mae': offset + i} for seed, i in zip((2026,2027,2028),(1.,2.,3.))}},
            'seed_evidence': {'interpretation': 'independent'}}
    main, regional, macro, per_seed = [], [], [], []
    _append_model(main, regional, macro, per_seed, 'tabpfn_v3', row, 'Q', None,
                  {'run_id':'test','source_hash':'s','config_hash':'c'})
    assert main[0]['mae_mean'] == 102.
    assert main[0]['mae_std'] == 1.
    assert regional[0]['mae_mean'] == 2.
    assert 'mae' not in regional[0]


def test_candidate_refresh_replaces_foundation_rows_already_in_main(tmp_path):
    import csv
    from forecasting.foundation_candidate import _existing_rows
    for name in ('main.csv', 'regional.csv', 'macro_per_seed.csv', 'per_seed.csv'):
        with (tmp_path / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['model', 'table'])
            writer.writeheader()
            writer.writerows([{'model': 'baseline_lightgbm', 'table': 'Q'},
                              {'model': 'tabpfn_v3', 'table': 'Q'}])
    for rows in _existing_rows(tmp_path):
        assert [r['model'] for r in rows] == ['baseline_lightgbm']
