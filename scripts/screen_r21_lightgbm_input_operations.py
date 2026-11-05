"""Development-only screen for audited frozen-LightGBM input interventions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.adaptive_agent.context import ContextRepository
from forecasting.adaptive_agent.input_operations import OPERATIONS, apply_operation
from forecasting.adaptive_agent.predictors.lightgbm import FrozenLightGBMAdapter
from forecasting.adaptive_agent.replay import aggregate_scores, select_indices, sha_file, write_json
from forecasting.aligned_inputs import INPUT_KEYS


BASELINE = Path('/root/workspace/Code/DualTimesField/outputs/forecasting/aligned_baselines_20261012')
REGIONS = ('NSW1', 'QLD1', 'TAS1')
DEVELOPMENT_SPLIT = 'calibration'


def flatten(blocks):
    return np.concatenate([np.asarray(blocks[key]).reshape(-1) for key in INPUT_KEYS])


def affected_training_support(train_directory, operation):
    if operation.operation_id == 'identity':
        return {'affected_features': 0, 'zero_within_minmax_fraction': 1.0,
                'zero_within_q005_q995_fraction': 1.0}
    values = np.load(train_directory / ('input__' + operation.block + '.npy'), mmap_mode='r')
    if values.ndim == 2:
        selected = np.asarray(values[:, operation.column_start:operation.column_stop])
    else:
        selected = np.asarray(values[:, operation.row_start:operation.row_stop,
                                     operation.column_start:operation.column_stop]).reshape(len(values), -1)
    lower, upper = np.min(selected, axis=0), np.max(selected, axis=0)
    qlower, qupper = np.quantile(selected, [.005, .995], axis=0)
    return {
        'affected_features': int(selected.shape[1]),
        'zero_within_minmax_fraction': float(np.mean((lower <= 0) & (upper >= 0))),
        'zero_within_q005_q995_fraction': float(np.mean((qlower <= 0) & (qupper >= 0))),
        'observed_min': float(selected.min()), 'observed_max': float(selected.max()),
    }


def qs_per_origin(actual, quantiles):
    levels = np.array([.05, .1, .5, .9, .95])
    error = actual[..., None] - quantiles
    return 2 * np.maximum(levels * error, (levels - 1) * error).mean(axis=(1, 2))


def metrics(actual, prediction, identity):
    score = aggregate_scores(actual, prediction)
    q = qs_per_origin(actual, prediction)
    base = qs_per_origin(actual, identity)
    return {**score, 'origin_qs_improvement_fraction': float(np.mean(q < base)),
            'origin_qs_tie_fraction': float(np.mean(q == base)),
            'conditional_oracle_qs_gain': float(np.maximum(base - q, 0).mean()),
            'mean_qs_delta_vs_identity': float(np.mean(q - base))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--origins', type=int, default=181)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.origins < 121:
        raise ValueError('--origins must leave at least one post-design validation origin')

    branch = subprocess.check_output(
        ['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()

    result = {
        'status': 'DEVELOPMENT_OPERATOR_SCREEN',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': {
            'branch': branch,
            'commit': commit,
            'base_commit': '4f3aa6c',
            'script_sha256': sha_file(Path(__file__).resolve()),
            'input_operations_sha256': sha_file(
                ROOT / 'forecasting/adaptive_agent/input_operations.py'),
            'lightgbm_adapter_sha256': sha_file(
                ROOT / 'forecasting/adaptive_agent/predictors/lightgbm.py'),
        },
        'split': DEVELOPMENT_SPLIT, 'seed': 2026, 'origins_per_region': args.origins,
        'design_origins_per_region': 120, 'validation_origins_per_region': args.origins - 120,
        'selection_rule_fixed_before_scores': {
            'structural': 'all replacement zeros within training per-feature min/max and >=95% within q0.5%-q99.5%',
            'design': 'improves >=15% design origins in every region; worst regional aggregate QS degradation <=2%',
            'ranking': 'top four by equal-region conditional oracle QS gain on design only',
            'validation': 'reported after selection and never changes admission',
        },
        'semantics': 'zero means training-normalization mean counterfactual, not missingness or column deletion',
        'regions': {}, 'training_support': {}, 'operations': {},
    }
    predictions, actuals = {}, {}
    for region in REGIONS:
        repository = ContextRepository(BASELINE / 'data_cache' / region / DEVELOPMENT_SPLIT)
        indices = select_indices(len(repository.origins), str(args.origins))
        actual = np.load(repository.directory / 'actual.npy', mmap_mode='r')[indices].copy()
        adapter = FrozenLightGBMAdapter(
            BASELINE / 'runs/lightgbm/seed2026' / region,
            region=region, seed=2026, threads=4)
        region_predictions = {}
        for operation_id in OPERATIONS:
            matrix = []
            for index in indices:
                context = repository.context(index)
                blocks, _ = apply_operation(context.model_inputs, operation_id)
                matrix.append(flatten(blocks))
            region_predictions[operation_id] = adapter.predict_matrix(np.stack(matrix))
        predictions[region], actuals[region] = region_predictions, actual
        result['regions'][region] = {
            'indices': indices,
            'first_origin': int(repository.origins[indices[0]]),
            'last_origin': int(repository.origins[indices[-1]]),
            'model_manifest_sha256': sha_file(adapter.anchor.directory / 'manifest.json'),
        }
        result['training_support'][region] = {
            operation_id: affected_training_support(
                BASELINE / 'data_cache' / region / 'train', operation)
            for operation_id, operation in OPERATIONS.items()}

    design = slice(0, 120)
    validation = slice(120, args.origins)
    eligible = []
    for operation_id, operation in OPERATIONS.items():
        row = {'spec': operation.__dict__, 'design': {}, 'validation': {}}
        for region in REGIONS:
            identity = predictions[region]['identity']
            row['design'][region] = metrics(actuals[region][design],
                                            predictions[region][operation_id][design], identity[design])
            row['validation'][region] = metrics(actuals[region][validation],
                                                predictions[region][operation_id][validation], identity[validation])
        structural = all(
            result['training_support'][region][operation_id]['zero_within_minmax_fraction'] == 1
            and result['training_support'][region][operation_id]['zero_within_q005_q995_fraction'] >= .95
            for region in REGIONS)
        base = np.mean([row['design'][r]['qs'] for r in REGIONS])
        identity_qs = np.mean([metrics(actuals[r][design], predictions[r]['identity'][design],
                                       predictions[r]['identity'][design])['qs'] for r in REGIONS])
        worst_degradation = max(100 * (row['design'][r]['qs'] - metrics(
            actuals[r][design], predictions[r]['identity'][design],
            predictions[r]['identity'][design])['qs']) / metrics(
                actuals[r][design], predictions[r]['identity'][design],
                predictions[r]['identity'][design])['qs'] for r in REGIONS)
        improvement_floor = min(row['design'][r]['origin_qs_improvement_fraction'] for r in REGIONS)
        oracle_gain = float(np.mean([row['design'][r]['conditional_oracle_qs_gain'] for r in REGIONS]))
        admitted_by_rule = (operation_id != 'identity' and structural
                            and improvement_floor >= .15 and worst_degradation <= 2.0)
        row['admission'] = {
            'structural_support': structural,
            'design_improvement_fraction_floor': improvement_floor,
            'design_worst_region_qs_degradation_percent': worst_degradation,
            'design_region_equal_qs': float(base),
            'design_identity_region_equal_qs': float(identity_qs),
            'design_conditional_oracle_qs_gain': oracle_gain,
            'passes_rule_before_top_four_cap': admitted_by_rule,
        }
        if admitted_by_rule:
            eligible.append((oracle_gain, operation_id))
        result['operations'][operation_id] = row
    selected = [key for _, key in sorted(eligible, reverse=True)[:4]]
    result['admitted_operation_ids'] = ['identity', *selected]
    result['rejected_operation_ids'] = [key for key in OPERATIONS if key not in result['admitted_operation_ids']]
    result['status'] = ('ADMITTED' if selected else 'NO_NONIDENTITY_OPERATION_ADMITTED')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({'status': result['status'],
                      'admitted_operation_ids': result['admitted_operation_ids'],
                      'output': str(args.output)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
