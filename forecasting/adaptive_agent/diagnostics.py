"""Numerical evidence over visible matured episodes, with explicit support."""
from __future__ import annotations

import numpy as np
from ..api_agent.workflow import quantile_loss


def within_history_diagnostic(context, registry, profiles=('base', 'calendar', 'price_only')):
    """Compatibility entrypoint for the R17 48→24 profile diagnostic."""
    if len(set(profiles)) != len(profiles) or set(profiles) - {'base', 'calendar', 'price_only'}:
        raise ValueError('Unregistered within-history profile')
    if 'chronos' not in registry.adapters:
        return {'diagnostic_kind': 'within_visible_history_hindcast', 'available': False,
                'reason': 'chronos_not_available', 'logical_inference_units': 0,
                'numeric_inferences': 0, 'sample_count': 0, 'profiles': {}}
    result = evaluate_recent_history(context, [f'chronos:{p}:72' for p in profiles], registry)
    result['profiles'] = {p: result['candidates'][f'chronos:{p}:72']['metrics'] for p in profiles}
    result['history_context_hours'] = 48
    result['available_history_hours'] = 72
    return result


def evaluate_recent_history(context, candidate_ids, registry, *, max_units=40):
    """Candidate-ID variant of the visible-history diagnostic, budget bounded."""
    from ..api_agent.workflow import predictor_input
    from ..chronos2_zeroshot.model import predict_common_quantiles
    from .contracts import digest
    ids = list(candidate_ids)
    if len(set(ids)) != len(ids) or len(ids) > min(max_units, 40):
        raise ValueError('Recent-history diagnostic budget exceeded or duplicate candidates')
    for candidate in ids:
        if candidate not in registry.specs or registry.specs[candidate].model_id != 'chronos':
            raise ValueError('Reduced-context hindcast only supports registered Chronos candidates')
    adapter = registry.adapters.get('chronos')
    if adapter is None:
        return {'diagnostic_kind': 'within_visible_history_hindcast', 'available': False,
                'sample_count': 0, 'logical_inference_units': 0, 'numeric_inferences': 0, 'candidates': {}}
    if digest(adapter.identity) != digest(registry.model_identities['chronos']):
        raise ValueError('Frozen Chronos identity changed')
    outputs = {}
    for candidate in ids:
        spec = registry.specs[candidate]
        row = predictor_input(context.snapshot, spec.profile_id, hindcast=True)
        window = min(spec.window_hours, 48)
        row['target'] = row['target'][-window:]
        if 'past_covariates' in row:
            row['past_covariates'] = {key: value[-window:] for key, value in row['past_covariates'].items()}
        quantiles = predict_common_quantiles(adapter.pipeline, [row], batch_size=1, context_length=window)[0]
        outputs[candidate] = {'metrics': scores(context.snapshot.price[-24:], quantiles),
                              'sample_count': 1, 'hindcast_context_hours': window,
                              'forecast_context_hours': spec.window_hours,
                              'context_length_differs': window != spec.window_hours}
    return {'diagnostic_kind': 'within_visible_history_hindcast', 'available': True,
            'logical_inference_units': len(ids), 'numeric_inferences': len(ids),
            'sample_count': 1 if ids else 0, 'horizon_hours': 24,
            'target_end_exclusive': context.origin_unix, 'candidates': outputs,
            'lightgbm_supported': False,
            'limitation': 'Uses at most 48 observed context hours to predict the final24 observed hours. This can differ from the main forecast window. Foundation pretraining overlap unknown.'}


def scores(actual, quantiles):
    actual, q = np.asarray(actual), np.sort(np.asarray(quantiles), axis=-1)
    if actual.shape != (24,) or q.shape != (24, 5) or not np.isfinite(actual).all() or not np.isfinite(q).all():
        raise ValueError('Diagnostic must use complete finite 24-hour targets')
    return {'qs': quantile_loss(actual, q), 'mae': float(np.abs(actual - q[:, 2]).mean()),
            'coverage90': float(((actual >= q[:, 0]) & (actual <= q[:, -1])).mean()),
            'width90': float((q[:, -1] - q[:, 0]).mean()),
            'qs_first12': quantile_loss(actual[:12], q[:12]),
            'qs_last12': quantile_loss(actual[12:], q[12:])}


def compare_candidates(artifacts):
    """Forecast disagreement only; never treats candidate values as targets."""
    values = list(artifacts.values()) if isinstance(artifacts, dict) else list(artifacts)
    if not values:
        return {'diagnostic_kind': 'prediction_disagreement', 'sample_count': 0, 'candidates': {}, 'pairs': []}
    for item in values:
        if item.context_id != values[0].context_id or not np.array_equal(item.target_time_unix, values[0].target_time_unix):
            raise ValueError('Disagreement requires one context and target axis')
    summaries = {item.candidate_id: {'median_mean': float(item.quantiles[:, 2].mean()),
                  'width90_mean': float((item.quantiles[:, -1] - item.quantiles[:, 0]).mean())}
                 for item in values}
    pairs = [{'a': a.candidate_id, 'b': b.candidate_id,
              'median_absolute_difference': float(np.abs(a.quantiles[:, 2] - b.quantiles[:, 2]).mean())}
             for i, a in enumerate(values) for b in values[i + 1:]]
    return {'diagnostic_kind': 'prediction_disagreement', 'sample_count': 0,
            'future_actuals_used': False, 'candidates': summaries, 'pairs': pairs}


def _role(episode, model_identity):
    origin = episode['context'].origin_unix
    role = episode.get('data_role', episode['context'].metadata.get('split', 'unknown'))
    train_end = model_identity.get('train_end_unix')
    selection_end = model_identity.get('selection_end_unix')
    if role == 'train' or (train_end is not None and origin < train_end):
        return 'training_in_sample'
    if role == 'selection' or (selection_end is not None and origin < selection_end):
        return 'model_selection_exposed'
    # An episode's split name alone cannot prove model pretraining exclusion.
    if train_end is None or selection_end is None:
        return 'training_or_selection_overlap_unknown'
    return 'post_selection_holdout'


def evaluate_episodes(context, candidate_ids, episodes, registry, *, max_units=40):
    """Recompute frozen forecasts on eligible full historical contexts.

    Recomputed results are never described as predictions published historically.
    Logical candidate×episode units are capped even when registry caches hit.
    """
    candidates = list(candidate_ids)
    episodes = list(episodes)
    if len(set(candidates)) != len(candidates) or len(candidates) > 5 or len(episodes) > 8:
        raise ValueError('Invalid diagnostic candidate/episode bounds')
    if max_units < 0 or len(candidates) * len(episodes) > min(max_units, 40):
        raise ValueError('Historical inference budget exceeded')
    if len({item['episode_id'] for item in episodes}) != len(episodes):
        raise ValueError('Duplicate historical diagnostic episode')
    visible = min(int(context.cutoff), int(context.memory_watermark))
    for episode in episodes:
        historical = episode['context']
        target_end = int(historical.snapshot.target_time_unix[-1]) + 3600
        if (historical.region != context.region or historical.origin_unix >= context.origin_unix
                or int(episode['outcome_available_at']) > visible
                or int(episode['outcome_available_at']) < target_end
                or target_end > visible or historical.cutoff > visible):
            raise ValueError('Historical diagnostic episode is not mature and visible')
        y = np.asarray(episode['actual'])
        if y.shape != (24,) or not np.isfinite(y).all():
            raise ValueError('Historical outcomes incomplete')
    diagnostic_registry = registry.fork(max_current_candidates=40)
    rows, failures = [], []
    for candidate_id in candidates:
        if candidate_id not in registry.specs:
            raise ValueError('Unknown historical candidate')
        spec = registry.specs[candidate_id]
        identity = registry.model_identities[spec.model_id]
        for episode in episodes:
            try:
                artifact = diagnostic_registry.forecast(episode['context'], candidate_id)
                rows.append({'candidate_id': candidate_id, 'episode_id': episode['episode_id'],
                             'forecast_id': artifact.identity(),
                             'data_role': _role(episode, identity),
                             'diagnostic_kind': 'retrospective_frozen_recompute',
                             'metrics': scores(episode['actual'], artifact.quantiles)})
            except (ValueError, RuntimeError, OSError) as exc:
                failures.append({'candidate_id': candidate_id, 'episode_id': episode['episode_id'],
                                 'reason': type(exc).__name__, 'detail': str(exc)[:200]})
    aggregates = {}
    for candidate_id in candidates:
        selected = [row for row in rows if row['candidate_id'] == candidate_id]
        holdout = [row for row in selected if row['data_role'] == 'post_selection_holdout']
        def mean(items):
            return ({key: float(np.mean([item['metrics'][key] for item in items]))
                     for key in items[0]['metrics']} if items else None)
        aggregates[candidate_id] = {
            'sample_count': len(selected), 'holdout_sample_count': len(holdout),
            'metrics': mean(selected), 'post_selection_holdout_metrics': mean(holdout),
            'data_roles': sorted({row['data_role'] for row in selected}),
            'missing_count': len(episodes) - len(selected)}
    return {'diagnostic_kind': 'retrospective_frozen_recompute', 'horizon_hours': 24,
            'sample_count': len(episodes), 'logical_inference_units': len(candidates) * len(episodes),
            'numeric_inferences': diagnostic_registry.stats['numeric_inferences'],
            'episodes': [item['episode_id'] for item in episodes],
            'candidates': aggregates, 'rows': rows, 'failures': failures,
            'limitation': 'Retrospective recomputation, not historical published forecasts. Training and selection overlap are flagged; unknown overlap is not unbiased evidence.'}
