"""Independent, no-training verification of the already completed L0 archive."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1]).resolve()

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}

def independent(y, b, q):
    y, b, q = (np.asarray(v, dtype=np.float64) for v in (y, b, q))
    losses = []
    for x in (b, q):
        residual = y[..., None] - x
        # Scalar formula in a loop deliberately avoids importing the audit scorer.
        pb = np.stack([np.where(residual[..., k] >= 0, tau * residual[..., k],
                               (tau-1) * residual[..., k])
                       for k, tau in enumerate((.05, .1, .5, .9, .95))], axis=-1)
        losses.append((np.abs(y-x[..., 2]), 2 * pb.mean(axis=-1)))
    result = {}
    for k, label in enumerate(('mae', 'qs')):
        delta = losses[1][k] - losses[0][k]
        result.update({f'after_{label}': float(losses[1][k].mean()),
                       f'before_{label}': float(losses[0][k].mean()),
                       f'{label}_delta': float(delta.mean()),
                       f'{label}_benefit': float((-delta[delta < 0]).sum()/delta.size),
                       f'{label}_harm': float(delta[delta > 0].sum()/delta.size)})
    return result

progress = read(root/'progress.json')
assert progress['stage'] == 'complete'
manifest = read(root/'manifest.json')
verification = read(root/'verification.json')
assert verification['trained_models'] == 0 and verification['formal_tables_changed'] is False
assert verification['scientific_confirmation'] is False
assert verification['run_id'] == manifest['run_id']
for relative, expected in verification['artifact_sha256'].items():
    assert sha(root/relative) == expected, relative
for stage in ('saved', 'replay'):
    marker = read(root/f'{stage}_complete.json')
    identity = {k: v for k, v in marker.items() if k != 'completion_id'}
    assert hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == marker['completion_id']
    assert marker['run_id'] == manifest['run_id']
    observed = {str(p.relative_to(root)): sha(p) for p in (root/stage).rglob('*') if p.is_file()}
    assert observed == marker['artifact_sha256']
summary = read(root/'replay/summary.json')
max_score_error, max_budget_error = 0., 0.
checked_files, pair_results, zero_anchor_survival = [], [], []
for case in summary['cases']:
    checkpoint = Path(case['checkpoint'])
    assert sha(checkpoint) == case['checkpoint_sha256']
    selected_record = read(checkpoint.parent/'selection.json')['selection'][case['mode']]
    assert selected_record['epoch'] == case['epoch'] and selected_record['alpha'] == case['alpha']
    assert sha(case['saved_predictions']) == case['saved_predictions_sha256']
    old = load(case['saved_predictions'])
    arms = {name: load(record['path']) for name, record in case['arms'].items()}
    orig = arms['original']
    positions = np.searchsorted(old['origin_unix'], orig['origin_unix'])
    for name in ('origin_unix', 'target_time_unix', 'actual', 'baseline'):
        np.testing.assert_array_equal(orig[name], old[name][positions])
    for name in ('raw_quantiles', 'raw_anchor', 'quantiles', 'anchor'):
        np.testing.assert_allclose(orig[name], old[name][positions], rtol=0, atol=manifest['configuration']['replay_atol'])
    for arm, data in arms.items():
        record = case['arms'][arm]
        assert sha(record['path']) == record['sha256']
        assert float(data['alpha']) == case['alpha'] and int(data['epoch']) == case['epoch']
        assert str(data['checkpoint_sha256']) == case['checkpoint_sha256']
        np.testing.assert_array_equal(data['target_time_unix'], data['origin_unix'][:, None]+np.arange(24)*3600)
        mask = data['candidate_mask']
        assert np.all((data['candidate_origins']+86400 <= data['origin_unix'][:, None])[mask])
        for mode, qkey in (('unshrunk', 'raw_quantiles'), ('chosen', 'quantiles')):
            q = data[qkey]
            assert np.isfinite(q).all() and np.all(np.diff(q, axis=-1) >= 0)
            calculated = independent(data['actual'], data['baseline'], q)
            for metric, value in calculated.items():
                error = abs(value-record[mode][metric])
                max_score_error = max(max_score_error, error)
                assert error < 1e-9, (arm, mode, metric, error)
        b, q, alpha = data['baseline'], data['raw_quantiles'], case['alpha']
        selected = b if alpha == 0 else q if alpha == 1 else (1-alpha)*b.astype(np.float64)+alpha*q.astype(np.float64)
        np.testing.assert_array_equal(data['quantiles'], selected)
        checked_files.append(record['path'])
    b = orig['baseline'].astype(np.float64)
    delta = orig['raw_quantiles'].astype(np.float64)-b
    gate = orig['evidence_gates'][..., 0].astype(np.float64)
    state = arms['external_final_state']
    uniform = arms['external_matched_uniform']
    np.testing.assert_allclose(state['raw_quantiles'], b+gate[..., None]*delta, rtol=0, atol=1e-10)
    state_size = np.abs(state['raw_quantiles']-b).sum(axis=(1, 2))
    uniform_size = np.abs(uniform['raw_quantiles']-b).sum(axis=(1, 2))
    error = np.max(np.abs(state_size-uniform_size)/np.maximum(1., state_size))
    assert error < 1e-10
    max_budget_error = max(max_budget_error, float(error))
    paired = independent(orig['actual'], uniform['quantiles'], state['quantiles'])
    pair_results.append(dict(region=case['region'], fold=case['fold'], mode=case['mode'],
                             mae_delta=paired['mae_delta'], qs_delta=paired['qs_delta']))
    zero_anchor_survival.append(case['assertions']['zero_gate_surviving_history_max'])
# Reconcile published summaries with every per-seed/per-region saved score row.
with (root/'saved/formal_scores.csv').open() as stream:
    formal_rows = list(csv.DictReader(stream))
for entry in read(root/'saved/summary.json')['formal_macro']:
    selected = [r for r in formal_rows if r['model'] == entry['model'] and r['group'] == 'all']
    for metric in ('mae', 'qs'):
        assert abs(np.mean([float(r[f'after_{metric}']) for r in selected])-entry[f'{metric}_mean']) < 1e-9
with (root/'saved/development_scores.csv').open() as stream:
    development_rows = list(csv.DictReader(stream))
empty_covered_weak = [r for r in development_rows if r['group'] == 'covered_weak_location' and r['empty'] == 'True']
assert all(r['mae_delta'] == '' for r in empty_covered_weak)
result = dict(status='passed', verification_scope='independent archived-forecast arithmetic, metadata and artifact integrity; no rerun of full training',
    audit_run_id=manifest['run_id'], checked_stage_artifacts=len(verification['artifact_sha256']),
    replay_cases=len(summary['cases']), replay_prediction_files=len(checked_files),
    maximum_independent_metric_error=max_score_error, maximum_budget_relative_error=max_budget_error,
    all_case_checkpoint_hash_epoch_alpha_matches=True, all_archived_original_replays_match=True,
    all_selected_blends_exact=True, all_candidate_labels_mature=True,
    empty_covered_weak_rows_retained=len(empty_covered_weak),
    zero_anchor_gate_with_surviving_history_cases=sum(v > 0 for v in zero_anchor_survival),
    external_state_minus_same_origin_budget_uniform=pair_results,
    limitations=['Monthly replay samples are retrospective diagnostics, not independent market replications.',
                'No significance, state-conditional calibration, causal mechanism or new blind-test claim follows.',
                'The independent checker verifies forecast artifacts and checkpoint bytes; complete source publication times remain outside this check.'],
    trained_models=0, old_artifacts_written=False)
(root/'independent_verification.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
print(json.dumps(result, indent=2, ensure_ascii=False))
