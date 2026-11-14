"""R23 opportunity-loss diagnosis over one saved development replay.

Per-origin post-hoc oracle selection over the materialized candidates is a
diagnostic upper bound with selection bias. It is not a deployable score, and
these values must never enter any decision input for the same origins.

Decomposition (per region, QS primary): with mat = oracle over all
materialized candidates and elig = oracle over identity plus evidence-eligible
candidates,

    G = QS(identity) - QS(mat)                  (generation headroom)
    A = QS(elig) - QS(mat)                      (admission loss)
    S = QS(committed) - QS(elig)                (selection and risk loss)

so that QS(identity) - QS(committed) = G - A - S.
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_r22_blocked_cv import REGIONS, _load_parent, _load_aligned_data
from forecasting.adaptive_agent.replay import aggregate_scores, bundle_name, digest, sha_file

IDENTITY_ID = 'lightgbm-input:identity'
METRICS = ('qs', 'mae')
NUMERIC_FIELDS = (
    'identity', 'oracle_materialized', 'oracle_eligible', 'committed',
    'generation_headroom_identity_minus_oracle_materialized',
    'admission_loss_oracle_eligible_minus_oracle_materialized',
    'selection_risk_loss_committed_minus_oracle_eligible',
)
# Protocol stages that end an origin before the structured evidence report
# admits any operation. Their headroom is attributed to admission.
EARLY_STOP_REASONS = frozenset({
    'identity_no_mature_cohort', 'identity_insufficient_paired_support',
    'identity_no_operation_passed_structured_evidence',
})


def origin_diagnosis(actual, candidates, eligible, committed, identity_id=IDENTITY_ID):
    """Decompose one origin's QS/MAE opportunity loss.

    candidates maps candidate id to (24, 5) quantiles and must contain the
    identity candidate. eligible lists non-identity evidence-admitted ids;
    identity is always a legal choice.
    """
    actual = np.asarray(actual, dtype=float)
    if actual.shape != (24,) or not np.isfinite(actual).all():
        raise ValueError('Diagnosis requires 24 finite actuals')
    if identity_id not in candidates:
        raise ValueError('Identity candidate missing from materialized set')
    unknown = [cid for cid in eligible if cid not in candidates]
    if unknown:
        raise ValueError('Eligible ids absent from materialized set: ' + ','.join(unknown))
    committed = np.asarray(committed, dtype=float)
    if committed.shape != (24, 5) or not np.isfinite(committed).all():
        raise ValueError('Committed forecast must be finite 24x5 quantiles')

    per_candidate = {cid: aggregate_scores(actual, np.asarray(q)) for cid, q in candidates.items()}
    legal = [identity_id] + sorted(eligible)
    result = {}
    for metric in METRICS:
        scores = {cid: row[metric] for cid, row in per_candidate.items()}
        best_materialized = min(scores, key=scores.get)
        best_legal = min(legal, key=scores.get)
        identity_score = scores[identity_id]
        materialized_score = scores[best_materialized]
        eligible_score = scores[best_legal]
        committed_score = aggregate_scores(actual, committed)[metric]
        block = {
            'identity': identity_score,
            'oracle_materialized': materialized_score,
            'oracle_eligible': eligible_score,
            'committed': committed_score,
            'generation_headroom_identity_minus_oracle_materialized':
                identity_score - materialized_score,
            'admission_loss_oracle_eligible_minus_oracle_materialized':
                eligible_score - materialized_score,
            'selection_risk_loss_committed_minus_oracle_eligible':
                committed_score - eligible_score,
        }
        assert tuple(block) == NUMERIC_FIELDS
        result[metric] = block
        result[f'oracle_materialized_id_{metric}'] = best_materialized
        result[f'oracle_eligible_id_{metric}'] = best_legal
    return result


def region_summary(origin_rows):
    """Aggregate per-origin diagnoses and trace facts into one region block."""
    summary = {
        'origins': len(origin_rows),
        'stop_reasons': dict(sorted(Counter(
            row['stop_reason'] for row in origin_rows).items())),
        'early_stop_origins': sum(row['stop_reason'] in EARLY_STOP_REASONS
                                  for row in origin_rows),
        'eligible_nonempty_origins': sum(bool(row['eligible']) for row in origin_rows),
        'committed_nonidentity_origins': sum(row['committed_id'] != IDENTITY_ID
                                             for row in origin_rows),
        'oracle_materialized_pick_counts': dict(sorted(Counter(
            row['diagnosis']['oracle_materialized_id_qs'] for row in origin_rows).items())),
        'oracle_materialized_nonidentity_origins': sum(
            row['diagnosis']['oracle_materialized_id_qs'] != IDENTITY_ID
            for row in origin_rows),
        'committed_matches_oracle_eligible': sum(
            row['committed_id'] == row['diagnosis']['oracle_eligible_id_qs']
            for row in origin_rows),
    }
    for metric in METRICS:
        summary[metric] = {field: float(np.mean(
            [row['diagnosis'][metric][field] for row in origin_rows]))
            for field in NUMERIC_FIELDS}
    return summary


def losses(block):
    """QS decomposition of one region block, per the frozen R24 plan notation."""
    qs = block['qs']
    return {
        'G_generation_headroom': qs['generation_headroom_identity_minus_oracle_materialized'],
        'A_admission_loss': qs['admission_loss_oracle_eligible_minus_oracle_materialized'],
        'S_selection_risk_loss': qs['selection_risk_loss_committed_minus_oracle_eligible'],
    }


def branch_decision(region_equal_losses):
    """Apply the pre-registered branch rule from the frozen R24 execution plan."""
    G = region_equal_losses['G_generation_headroom']
    A = region_equal_losses['A_admission_loss']
    S = region_equal_losses['S_selection_risk_loss']
    if G < 0.05:
        return ('parameterize_operations_first',
                'Generation headroom below 0.05 QS: the current operation space has no meat.')
    if S > A:
        return ('proceed_with_r24a_loop',
                'Decision quality is the bottleneck; the loop targets the largest loss.')
    return ('admission_contract_first',
            'Admission filtering dominates the reachable loss; review the contract first.')


def load_origin_row(region_dir, region, origin, aligned_actual, row):
    directory = region_dir / bundle_name(region, origin)
    with np.load(directory / 'materialized_candidates.npz', allow_pickle=False) as bundle:
        if set(bundle.files) != {'candidate_ids', 'forecast_ids', 'quantiles', 'target_time_unix'}:
            raise ValueError('Unexpected materialized candidate schema')
        candidate_ids = [str(value) for value in bundle['candidate_ids']]
        if len(set(candidate_ids)) != len(candidate_ids) or IDENTITY_ID not in candidate_ids:
            raise ValueError('Materialized candidates must be unique and contain identity')
        if not np.array_equal(bundle['target_time_unix'], row['arrays']['target_time_unix']):
            raise ValueError('Materialized target axis differs from committed predictions')
        candidates = {cid: bundle['quantiles'][index].copy()
                      for index, cid in enumerate(candidate_ids)}
    trace = row['trace']
    report = trace.get('evidence_report') or {}
    eligible = [str(value) for value in (report.get('eligible_candidate_ids') or [])]
    stop_reason = trace.get('stop_reason')
    if not isinstance(stop_reason, str) or not stop_reason:
        raise ValueError('Trace lacks a stop reason')
    if eligible and stop_reason in EARLY_STOP_REASONS:
        raise ValueError('Early-stopped origin cannot have eligible operations')
    diagnosis = origin_diagnosis(aligned_actual, candidates, eligible,
                                 row['arrays']['agent'])
    return {'eligible': eligible, 'stop_reason': stop_reason,
            'committed_id': str(row['record']['agent']['candidate_id']),
            'diagnosis': diagnosis}


def analyze(r23_root, aligned_baseline_root):
    split = 'calibration'
    manifests, _, rows = _load_parent(r23_root)
    origins = sorted({origin for region, origin in rows if region == REGIONS[0]})
    if any(sorted({origin for region, origin in rows if region == name}) != origins
           for name in REGIONS):
        raise ValueError('Regions do not share one origin schedule')
    aligned = _load_aligned_data(aligned_baseline_root, split, manifests, origins)
    regions = {}
    for region in REGIONS:
        region_dir = Path(r23_root) / region / 'origins'
        origin_rows = [load_origin_row(region_dir, region, origin,
                                       aligned[region]['actual'][index], rows[(region, origin)])
                       for index, origin in enumerate(origins)]
        regions[region] = region_summary(origin_rows)
    region_equal = {
        metric: {field: float(np.mean([regions[region][metric][field] for region in REGIONS]))
                 for field in NUMERIC_FIELDS}
        for metric in METRICS}
    region_equal['losses'] = losses(region_equal)
    decision, rationale = branch_decision(region_equal['losses'])
    return {
        'status': 'COMPLETE_OPPORTUNITY_LOSS_DIAGNOSIS',
        'design': {
            'split': split, 'period': '2022H2', 'diagnostic_only': True,
            'oracle_definition': 'per-origin post-hoc minimum over candidate sets',
            'oracle_caveat': ('post-hoc selection is an upward-biased upper bound; '
                              'not a deployable score; never a decision input'),
            'G': 'QS(identity) - QS(oracle over all materialized candidates)',
            'A': 'QS(oracle over identity plus eligible) - QS(oracle materialized)',
            'S': 'QS(committed) - QS(oracle over identity plus eligible)',
            'identity_check': 'QS(identity) - QS(committed) = G - A - S',
            'branch_rule': 'frozen in r24_execution_plan_20261114.md before this run',
        },
        'inputs': {
            'r23_root': str(Path(r23_root).resolve()),
            'aligned_baseline_root': str(Path(aligned_baseline_root).resolve()),
            'manifest_ids': {region: digest(manifests[region]) for region in REGIONS},
            'analysis_sources': {path.name: sha_file(path) for path in (
                Path(__file__), Path(__file__).with_name('analyze_r22_blocked_cv.py'))},
        },
        'regions': {region: {**regions[region], 'losses': losses(regions[region])}
                    for region in REGIONS},
        'region_equal': region_equal,
        'branch_decision': {'action': decision, 'rationale': rationale,
                            'region_equal_losses': region_equal['losses']},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('r23-root', 'aligned-baseline-root', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.r23_root, args.aligned_baseline_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output),
                      'branch_decision': result['branch_decision']['action']}))


if __name__ == '__main__':
    main()
