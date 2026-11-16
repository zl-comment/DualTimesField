"""Pre-registered offline screen of the parameterized demand-window family.

Selection split (2022H1), frozen seed-2026 LightGBM, offline inference only.
Decision rule frozen in r24_parameterization_prescreen_plan_20261116.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.adaptive_agent.context import ContextRepository
from forecasting.adaptive_agent.input_operations import apply_operation
from forecasting.adaptive_agent.predictors.lightgbm import FrozenLightGBMAdapter
from forecasting.adaptive_agent.replay import aggregate_scores, sha_file, write_json
from forecasting.aligned_inputs import INPUT_KEYS

BASELINE = Path('/root/workspace/Code/DualTimesField/outputs/forecasting/aligned_baselines_20261012')
REGIONS = ('NSW1', 'QLD1', 'TAS1')
SPLIT = 'selection'
GRID = ('identity', 'history_demand_all_mean', 'history_demand_last12_mean',
        'history_demand_last24_mean', 'history_demand_last36_mean',
        'history_demand_last48_mean', 'history_demand_last60_mean')
BASELINE_GRID = ('history_demand_last24_mean', 'history_demand_last48_mean')


def flatten(blocks):
    return np.concatenate([np.asarray(blocks[key]).reshape(-1) for key in INPUT_KEYS])


def qs_per_origin(actual, quantiles):
    levels = np.array([.05, .1, .5, .9, .95])
    error = actual[..., None] - quantiles
    return 2 * np.maximum(levels * error, (levels - 1) * error).mean(axis=(1, 2))


def oracle_qs_gain(actual, quantiles_by_op, ops):
    """Post-hoc per-origin best over ops minus identity; upward-biased bound."""
    best = np.min(np.stack([qs_per_origin(actual, quantiles_by_op[op]) for op in ops]), axis=0)
    identity = qs_per_origin(actual, quantiles_by_op['identity'])
    return float(np.mean(identity - best))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    predictions, actuals = {}, {}
    for region in REGIONS:
        repository = ContextRepository(BASELINE / 'data_cache' / region / SPLIT)
        adapter = FrozenLightGBMAdapter(BASELINE / 'runs/lightgbm/seed2026' / region,
                                        region=region, seed=2026, threads=4)
        actual = np.load(repository.directory / 'actual.npy', mmap_mode='r').copy()
        matrices = {op: [] for op in GRID}
        for index in range(len(repository.origins)):
            context = repository.context(index)
            for op in GRID:
                blocks, _ = apply_operation(context.model_inputs, op)
                matrices[op].append(flatten(blocks))
        predictions[region] = {op: adapter.predict_matrix(np.stack(matrices[op]))
                               for op in GRID}
        actuals[region] = actual
        print(region, 'predicted', len(repository.origins), 'origins x', len(GRID), 'ops',
              flush=True)

    result = {
        'status': 'PARAMETERIZED_DEMAND_SCREEN',
        'split': SPLIT, 'seed': 2026, 'grid': list(GRID),
        'decision_rule': 'frozen in r24_parameterization_prescreen_plan_20261116.md',
        'analysis_sources': {
            Path(__file__).name: sha_file(Path(__file__).resolve()),
            'input_operations.py': sha_file(ROOT / 'forecasting/adaptive_agent/input_operations.py')},
        'regions': {}, 'oracle_qs_gain': {}, 'decision': {},
    }
    for region in REGIONS:
        identity = predictions[region]['identity']
        rows = {}
        for op in GRID:
            score = aggregate_scores(actuals[region], predictions[region][op])
            q, base = (qs_per_origin(actuals[region], predictions[region][op]),
                       qs_per_origin(actuals[region], identity))
            rows[op] = {
                'qs': score['qs'], 'mae': score['mae'], 'rmse': score['rmse'],
                'coverage90': score['coverage90'],
                'qs_delta_vs_identity': score['qs'] - aggregate_scores(
                    actuals[region], identity)['qs'],
                'mae_delta_vs_identity': score['mae'] - aggregate_scores(
                    actuals[region], identity)['mae'],
                'origin_qs_improvement_fraction': float(np.mean(q < base)),
            }
        result['regions'][region] = {
            'origins': int(len(actuals[region])),
            'operations': rows,
            'oracle_qs_gain_expanded': oracle_qs_gain(
                actuals[region], predictions[region], GRID[1:]),
            'oracle_qs_gain_baseline_grid': oracle_qs_gain(
                actuals[region], predictions[region], BASELINE_GRID),
        }

    expanded = {r: result['regions'][r]['oracle_qs_gain_expanded'] for r in REGIONS}
    baseline = {r: result['regions'][r]['oracle_qs_gain_baseline_grid'] for r in REGIONS}
    result['oracle_qs_gain'] = {
        'region_equal_expanded': float(np.mean(list(expanded.values()))),
        'region_equal_baseline_grid': float(np.mean(list(baseline.values()))),
        'region_equal_growth': float(np.mean(list(expanded.values()))
                                    - np.mean(list(baseline.values())))}
    co_improved = {op: sum(
        result['regions'][r]['operations'][op]['qs_delta_vs_identity'] < 0
        and result['regions'][r]['operations'][op]['mae_delta_vs_identity'] < 0
        for r in REGIONS) for op in GRID[1:]}
    result['decision'] = {
        'co_improving_region_counts': co_improved,
        'condition_1_exists_window_improving_two_regions': any(v >= 2 for v in co_improved.values()),
        'condition_2_oracle_growth_at_least_0_05':
            result['oracle_qs_gain']['region_equal_growth'] >= 0.05,
        'parameterization_proceeds': bool(
            any(v >= 2 for v in co_improved.values())
            and result['oracle_qs_gain']['region_equal_growth'] >= 0.05),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({'oracle_growth': result['oracle_qs_gain']['region_equal_growth'],
                      'decision': result['decision']['parameterization_proceeds'],
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
