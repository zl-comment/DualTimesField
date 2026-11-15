"""Pre-registered offline ablation of the R23 admission contract.

Re-admits the saved structured evidence under frozen variants and scores a
deterministic commit policy on identical origins. No model inference, no API
calls. Variant list, commit rule and adoption rule are frozen in
r24_step3_admission_ablation_plan_20261115.md before any result.
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
METRICS = ('qs', 'mae', 'rmse', 'coverage90')
MIN_WIN_FRACTION = .625
VETO_STOP_REASON = 'risk_hard_veto_revert_identity'
ADOPTION_TIE_ORDER = ('V3_drop_current_change', 'V2_support_3',
                      'V1_drop_half_day', 'V4_state_gate_only')

VARIANTS = {
    'V0_baseline': dict(min_support=5, require_median=True,
                        min_win_fraction=MIN_WIN_FRACTION, require_half_day=True,
                        require_current_change=True),
    'V1_drop_half_day': dict(min_support=5, require_median=True,
                             min_win_fraction=MIN_WIN_FRACTION, require_half_day=False,
                             require_current_change=True),
    'V2_support_3': dict(min_support=3, require_median=True,
                         min_win_fraction=MIN_WIN_FRACTION, require_half_day=True,
                         require_current_change=True),
    'V3_drop_current_change': dict(min_support=5, require_median=True,
                                   min_win_fraction=MIN_WIN_FRACTION, require_half_day=True,
                                   require_current_change=False),
    'V4_state_gate_only': dict(min_support=5, require_median=False,
                               min_win_fraction=0.0, require_half_day=False,
                               require_current_change=True),
}


def rejection_reasons(effect, coverage_grade, config):
    """Evaluate one saved operation effect under one variant configuration."""
    paired = effect['paired_qs_gain']
    horizon = effect['horizon_qs_gain']
    similarity = effect['similarity_weighted']
    first12 = (horizon['hours_1_6']['mean'] + horizon['hours_7_12']['mean']) / 2
    last12 = (horizon['hours_13_18']['mean'] + horizon['hours_19_24']['mean']) / 2
    reasons = []
    if effect['common_support'] < config['min_support']:
        reasons.append('insufficient_common_support')
    if paired['mean_qs_gain'] <= 0:
        reasons.append('nonpositive_mean_qs_gain')
    if config['require_median'] and paired['median_qs_gain'] <= 0:
        reasons.append('nonpositive_median_qs_gain')
    if paired['win_fraction'] < config['min_win_fraction']:
        reasons.append('weak_win_fraction')
    if config['require_half_day']:
        if first12 <= 0:
            reasons.append('nonpositive_first12_qs_gain')
        if last12 <= 0:
            reasons.append('nonpositive_last12_qs_gain')
    if not similarity.get('available'):
        reasons.append('similarity_evidence_unavailable')
    else:
        if similarity['effective_support'] < config['min_support']:
            reasons.append('insufficient_similarity_effective_support')
        if similarity['mean_qs_gain'] <= 0:
            reasons.append('nonpositive_similarity_weighted_qs_gain')
    if (config['require_current_change']
            and effect['current_forecast']['current_disagreement']
            ['median_absolute_difference'] <= 0):
        reasons.append('zero_current_forecast_change')
    if coverage_grade != 'sufficient':
        reasons.append('cohort_quality_not_sufficient')
    return reasons


def eligible_under(report, config):
    """Return {candidate id: rejection reasons} for one saved evidence report."""
    coverage_grade = (report.get('cohort_quality') or {}).get('coverage_grade')
    return {candidate_id: rejection_reasons(effect, coverage_grade, config)
            for candidate_id, effect in report['operation_effects'].items()}


def select_commit_for(report, rejections):
    """Pre-registered deterministic choice: no eligible -> None (identity)."""
    eligible = sorted(cid for cid, reasons in rejections.items() if not reasons)
    if not eligible:
        return None
    if len(eligible) == 1:
        return eligible[0]

    def rank(cid):
        effect = report['operation_effects'][cid]
        similarity = effect['similarity_weighted']
        return (-similarity.get('mean_qs_gain', float('-inf')),
                -effect['paired_qs_gain']['mean_qs_gain'], cid)

    return min(eligible, key=rank)


def load_materialized(region_dir, region, origin, target_axis):
    directory = region_dir / bundle_name(region, origin)
    with np.load(directory / 'materialized_candidates.npz', allow_pickle=False) as bundle:
        if set(bundle.files) != {'candidate_ids', 'forecast_ids', 'quantiles', 'target_time_unix'}:
            raise ValueError('Unexpected materialized candidate schema')
        candidate_ids = [str(value) for value in bundle['candidate_ids']]
        if len(set(candidate_ids)) != len(candidate_ids) or IDENTITY_ID not in candidate_ids:
            raise ValueError('Materialized candidates must be unique and contain identity')
        if not np.array_equal(bundle['target_time_unix'], target_axis):
            raise ValueError('Materialized target axis differs from committed predictions')
        return {cid: bundle['quantiles'][index].copy()
                for index, cid in enumerate(candidate_ids)}


def variant_block(region_rows, candidates_by_origin, actual, identity, variant):
    """Score one variant's deterministic policy for one region."""
    config = VARIANTS[variant]
    committed, eligible_nonempty, committed_ids = [], 0, Counter()
    for origin, row in region_rows.items():
        evidence = row['trace'].get('evidence_report') or {}
        if evidence.get('operation_effects'):
            committed_id = select_commit_for(evidence, eligible_under(evidence, config))
        else:
            committed_id = None
        if committed_id is None:
            committed_id = IDENTITY_ID
        else:
            eligible_nonempty += 1
        committed_ids[committed_id] += 1
        committed.append(candidates_by_origin[origin][committed_id])
    scores = aggregate_scores(actual, np.stack(committed))
    identity_scores = aggregate_scores(actual, identity)
    return {
        'eligible_nonempty_origins': eligible_nonempty,
        'committed_nonidentity_origins': sum(
            count for cid, count in committed_ids.items() if cid != IDENTITY_ID),
        'committed_candidate_counts': dict(sorted(committed_ids.items())),
        'metrics': scores,
        'deltas_vs_identity': {metric: scores[metric] - identity_scores[metric]
                               for metric in METRICS},
    }


def veto_audit_block(region_rows, candidates_by_origin, actual, identity):
    """What-if outcome of candidates reverted by hard vetoes, grouped by code."""
    selected = [(origin, row['trace'].get('stage_b_selected_candidate_id'))
                for origin, row in region_rows.items()
                if row['trace'].get('stop_reason') == VETO_STOP_REASON]
    for origin, candidate_id in selected:
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f'Hard-veto origin lacks a stage B selection: {origin}')
        if candidate_id not in candidates_by_origin[origin]:
            raise ValueError(f'Vetoed selection not materialized: {candidate_id}')
    reason_codes = Counter()
    for origin, _ in selected:
        codes = list(region_rows[origin]['trace'].get('risk_vetoes') or [])
        reason = region_rows[origin]['trace'].get('final_commit_reason', '')
        if reason.startswith('hard_veto:') and reason[len('hard_veto:'):] not in codes:
            codes.append(reason[len('hard_veto:'):])
        reason_codes.update(sorted(set(codes)))
    if not selected:
        return {'origins': 0, 'reason_codes': {}}
    position = {origin: index for index, origin in enumerate(region_rows)}
    rows_selected = [position[origin] for origin, _ in selected]
    quantiles = np.stack([candidates_by_origin[origin][candidate_id]
                          for origin, candidate_id in selected])
    whatif = aggregate_scores(actual[rows_selected], quantiles)
    base = aggregate_scores(actual[rows_selected], identity[rows_selected])
    return {
        'origins': len(selected),
        'reason_codes': dict(sorted(reason_codes.items())),
        'metrics': whatif,
        'deltas_vs_identity': {metric: whatif[metric] - base[metric] for metric in METRICS},
    }


def adoption_decision(variants):
    """Apply the pre-registered adoption rule; V0 is the baseline, never adopted."""
    qualifying = []
    for name in ADOPTION_TIE_ORDER:
        ok = True
        for region, block in variants[name].items():
            baseline = variants['V0_baseline'][region]
            if block['eligible_nonempty_origins'] <= baseline['eligible_nonempty_origins']:
                ok = False
            if (block['deltas_vs_identity']['qs'] >= 0
                    or block['deltas_vs_identity']['mae'] >= 0):
                ok = False
            if block['deltas_vs_identity']['coverage90'] < -0.005:
                ok = False
        if ok:
            qualifying.append(name)
    if not qualifying:
        return {'action': 'keep_r23_admission_proceed_r24a', 'variant': None,
                'qualifying_variants': [],
                'rationale': 'No variant improves QS and MAE vs identity in every region '
                             'while raising eligibility; the contract stands.'}
    newly = {name: float(np.mean([
        variants[name][region]['eligible_nonempty_origins']
        - variants['V0_baseline'][region]['eligible_nonempty_origins']
        for region in REGIONS])) for name in qualifying}
    best = min(qualifying, key=lambda name: (newly[name], ADOPTION_TIE_ORDER.index(name)))
    return {'action': 'adopt_variant_into_r24a_contract', 'variant': best,
            'qualifying_variants': qualifying,
            'region_equal_newly_eligible_origins': newly,
            'rationale': 'Most conservative qualifying variant per the frozen rule.'}


def analyze(r23_root, aligned_baseline_root):
    split = 'calibration'
    manifests, _, rows = _load_parent(r23_root)
    origins = sorted({origin for region, origin in rows if region == REGIONS[0]})
    if any(sorted({origin for region, origin in rows if region == name}) != origins
           for name in REGIONS):
        raise ValueError('Regions do not share one origin schedule')
    aligned = _load_aligned_data(aligned_baseline_root, split, manifests, origins)

    variants = {name: {} for name in VARIANTS}
    binding_reasons, vetoes = {}, {}
    for region in REGIONS:
        region_dir = Path(r23_root) / region / 'origins'
        region_rows = {origin: rows[(region, origin)] for origin in origins}
        candidates_by_origin = {
            origin: load_materialized(region_dir, region, origin,
                                      row['arrays']['target_time_unix'])
            for origin, row in region_rows.items()}
        identity = np.stack([row['arrays']['base'] for row in region_rows.values()])
        actual = aligned[region]['actual']

        v0_binding = Counter()
        for origin, row in region_rows.items():
            evidence = row['trace'].get('evidence_report') or {}
            if not evidence.get('operation_effects'):
                continue
            rejections = eligible_under(evidence, VARIANTS['V0_baseline'])
            stored = sorted(evidence.get('eligible_candidate_ids') or [])
            recomputed = sorted(cid for cid, reasons in rejections.items() if not reasons)
            if recomputed != stored:
                raise ValueError(f'V0 reproduction mismatch at {region} {origin}')
            v0_binding.update(reason for reasons in rejections.values() for reason in reasons)
        binding_reasons[region] = dict(sorted(v0_binding.items()))

        for name in VARIANTS:
            variants[name][region] = variant_block(
                region_rows, candidates_by_origin, actual, identity, name)
        vetoes[region] = veto_audit_block(region_rows, candidates_by_origin, actual, identity)

    region_equal = {name: {
        'deltas_vs_identity': {metric: float(np.mean(
            [variants[name][region]['deltas_vs_identity'][metric] for region in REGIONS]))
            for metric in METRICS},
        'eligible_nonempty_origins': float(np.mean(
            [variants[name][region]['eligible_nonempty_origins'] for region in REGIONS])),
        'committed_nonidentity_origins': float(np.mean(
            [variants[name][region]['committed_nonidentity_origins'] for region in REGIONS])),
    } for name in VARIANTS}
    return {
        'status': 'COMPLETE_ADMISSION_ABLATION',
        'design': {
            'split': split, 'period': '2022H2', 'diagnostic_only': True,
            'variants': {name: dict(config) for name, config in VARIANTS.items()},
            'commit_rule': ('none->identity; one->it; two->higher similarity weighted '
                            'mean gain, then paired mean gain, then smallest id'),
            'adoption_rule': 'frozen in r24_step3_admission_ablation_plan_20261115.md',
            'v0_reproduction_guard': 'recomputed V0 equals stored eligibility everywhere',
        },
        'inputs': {
            'r23_root': str(Path(r23_root).resolve()),
            'aligned_baseline_root': str(Path(aligned_baseline_root).resolve()),
            'manifest_ids': {region: digest(manifests[region]) for region in REGIONS},
            'analysis_sources': {path.name: sha_file(path) for path in (
                Path(__file__), Path(__file__).with_name('analyze_r22_blocked_cv.py'))},
        },
        'variants': variants,
        'region_equal': region_equal,
        'v0_binding_reasons': binding_reasons,
        'veto_audit': vetoes,
        'adoption_decision': adoption_decision(variants),
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
                      'adoption_decision': result['adoption_decision']['action'],
                      'variant': result['adoption_decision'].get('variant')}))


if __name__ == '__main__':
    main()
