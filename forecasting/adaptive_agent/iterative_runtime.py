"""R20 incremental, paired diagnostics on one immutable mature episode cohort.

Only this subclass changes diagnostic caching. R18/R19 ToolHost semantics stay
unchanged. A failed candidate/episode attempt is a cached charged observation,
never an invitation to silently retry until the model produces a favorable row.
"""
from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np

from .contracts import digest, plain
from .diagnostics import _role, compare_candidates, scores
from .review import validate_artifact
from .runtime import ToolHost


class IterativeToolHost(ToolHost):
    def __init__(self, registry, evidence, analogs, feedback, *, information_scope='full_evidence'):
        super().__init__(registry, evidence, analogs, feedback, information_scope=information_scope)
        self._registry_binding = self._registry_identity()
        self._cohort_locked = False
        self._locked_episode_ids = ()
        self._cohort = ()
        self._cohort_binding = None
        self._cohort_id = None
        self._diagnostic_registry = registry.fork(max_current_candidates=5)
        self._diagnostic_cache = {}
        self._candidate_bindings = {}
        self._charged_diagnostic_units = 0

    def _registry_identity(self):
        return digest({'release': self.registry.release_id,
                       'specs': {k: asdict(v) for k, v in self.registry.specs.items()},
                       'models': self.registry.model_identities,
                       'live_models': {k: v.identity for k, v in self.registry.adapters.items()}})

    def _check(self, context):
        super()._check(context)
        if self._registry_identity() != self._registry_binding:
            raise ValueError('Iterative diagnostic registry/model/release changed')
        if self._cohort_locked:
            if tuple(self.episode_ids) != self._locked_episode_ids:
                raise ValueError('Iterative diagnostic episode cohort is immutable')
            if self._episode_binding(context, self._cohort) != self._cohort_binding:
                raise ValueError('Iterative diagnostic historical inputs/outcomes changed')

    @staticmethod
    def _episode_binding(context, episodes):
        return digest({'context': context.identity(), 'episodes': [
            {'episode_id': row['episode_id'], 'context_id': row['context'].identity(),
             'actual': row['actual'], 'available_at': row['outcome_available_at'],
             'data_role': row.get('data_role', 'unknown')}
            for row in episodes]})

    def _lock_cohort(self, context):
        if self._cohort_locked:
            raise ValueError('Cannot replace the first iterative diagnostic cohort')
        if len(self.episode_ids) > 8 or len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError('Invalid iterative diagnostic cohort size or duplicate ID')
        rows = tuple(self.analogs.get_episode(context, key) for key in self.episode_ids)
        visible = min(context.cutoff, context.memory_watermark)
        for row in rows:
            historical = row['context']
            available = row['outcome_available_at']
            if (historical.region != context.region or historical.origin_unix >= context.origin_unix
                    or available > visible or available < int(historical.snapshot.target_time_unix[-1]) + 3600
                    or historical.cutoff > visible):
                raise ValueError('Iterative diagnostic cohort contains an immature/foreign episode')
            actual = np.asarray(row['actual'], dtype=float).copy()
            if actual.shape != (24,) or not np.isfinite(actual).all():
                raise ValueError('Iterative diagnostic requires 24 finite mature targets')
            actual.setflags(write=False)
            row['actual'] = actual
        self._cohort = rows
        self._locked_episode_ids = tuple(self.episode_ids)
        self._cohort_binding = self._episode_binding(context, rows)
        self._cohort_id = digest({'registry': self._registry_binding, 'cohort': self._cohort_binding})
        self._cohort_locked = True

    def _models(self):
        models = [key for key in self.artifacts if key in self.registry.specs]
        if len(models) > 5:
            raise ValueError('Iterative diagnosis supports at most five current model candidates')
        return models

    def _candidate_binding(self, context, candidate_id):
        current = validate_artifact(context, self.artifacts[candidate_id])
        spec = self.registry.specs[candidate_id]
        expected_input = digest({'release': self.registry.release_id, 'context': context.identity(),
                                 'spec': asdict(spec), 'model': self.registry.model_identities[spec.model_id]})
        if current.input_hash != expected_input:
            raise ValueError('Current candidate input identity does not match frozen registry')
        binding = digest({'registry': self._registry_binding, 'current_input': current.input_hash,
                          'current_forecast': current.identity(), 'cohort': self._cohort_id,
                          'spec': asdict(spec)})
        if candidate_id in self._candidate_bindings and self._candidate_bindings[candidate_id] != binding:
            raise ValueError('A previously diagnosed candidate/input changed')
        return binding

    def _cache_key(self, context, candidate_id, episode):
        return digest({'candidate': self._candidate_binding(context, candidate_id),
                       'episode_id': episode['episode_id'],
                       'historical_context': episode['context'].identity()})

    def _score_episode(self, actual, quantiles):
        return scores(actual, quantiles)

    def actions(self, context, state):
        offered = super().actions(context, state)
        offered = {key: value for key, value in offered.items() if not key.startswith('diagnose:')
                   and not (self._cohort_locked and key.startswith('retrieve_analogs:'))}
        models = self._models()
        if self._cohort_locked and self._cohort and models:
            pending = sum(self._cache_key(context, candidate, episode) not in self._diagnostic_cache
                          for candidate in models for episode in self._cohort)
            offered['diagnose:' + '|'.join(models)] = {
                'description': 'Compare every generated model on the same locked mature cohort; '
                               'charge only previously unattempted model/episode pairs.',
                'kind': 'compare_candidates', 'costs': {'historical_evaluations': pending}}
        return offered

    def execute(self, context, action_id, state):
        self._check(context)
        if not action_id.startswith('diagnose:'):
            result = super().execute(context, action_id, state)
            if action_id.startswith('retrieve_analogs:'):
                self._lock_cohort(context)
                result['observation'].update({'cohort_id': self._cohort_id,
                                              'cohort_locked': True,
                                              'cohort_size': len(self._cohort)})
                self.audit.append({'action': action_id, 'locked_cohort': {
                    'cohort_id': self._cohort_id, 'episodes': list(self._locked_episode_ids)}})
            return result
        offered = self.actions(context, state)
        if action_id not in offered:
            raise ValueError('Diagnosis must include all generated models on the first locked cohort')
        models = self._models()
        jobs = [(candidate, episode, self._cache_key(context, candidate, episode))
                for candidate in models for episode in self._cohort]
        pending = [(candidate, episode, key) for candidate, episode, key in jobs if key not in self._diagnostic_cache]
        external_used = state.get('costs', {}).get('historical_evaluations', 0)
        if len(pending) > min(40 - self._charged_diagnostic_units, 40 - external_used):
            raise ValueError('Iterative historical inference budget exhausted before execution')
        numeric_before = self._diagnostic_registry.stats['numeric_inferences']
        attempted = 0
        for candidate, episode, key in pending:
            binding = self._candidate_binding(context, candidate)
            self._candidate_bindings[candidate] = binding
            # Record/charge before touching the numerical model. The default
            # unresolved entry also prevents silent retry after interruption.
            self._diagnostic_cache[key] = {'candidate_id': candidate, 'episode_id': episode['episode_id'],
                                           'status': 'failure', 'reason': 'InterruptedAttempt'}
            self._charged_diagnostic_units += 1
            attempted += 1
            try:
                artifact = self._diagnostic_registry.forecast(episode['context'], candidate)
                model_identity = self.registry.model_identities[self.registry.specs[candidate].model_id]
                row = {'candidate_id': candidate, 'episode_id': episode['episode_id'],
                       'forecast_id': artifact.identity(), 'input_hash': artifact.input_hash,
                       'data_role': _role(episode, model_identity), 'status': 'success',
                       'diagnostic_kind': 'retrospective_frozen_recompute',
                       'metrics': self._score_episode(episode['actual'], artifact.quantiles)}
                self._diagnostic_cache[key] = row
            except Exception as exc:
                self._diagnostic_cache[key] = {'candidate_id': candidate, 'episode_id': episode['episode_id'],
                                               'status': 'failure', 'reason': type(exc).__name__,
                                               'detail': str(exc)[:200]}
                if isinstance(exc, TimeoutError):
                    # An outer deadline may interrupt one call; do not start
                    # another numeric call after that signal was consumed.
                    break
        rows = [self._diagnostic_cache[key] for _, _, key in jobs if key in self._diagnostic_cache]
        success = {(row['candidate_id'], row['episode_id']): row for row in rows if row['status'] == 'success'}
        paired = [episode_id for episode_id in self._locked_episode_ids
                  if all((candidate, episode_id) in success for candidate in models)]
        holdout = [episode_id for episode_id in paired if all(
            success[candidate, episode_id]['data_role'] == 'post_selection_holdout' for candidate in models)]
        def mean(selected):
            return ({metric: float(np.mean([row['metrics'][metric] for row in selected]))
                     for metric in selected[0]['metrics']} if selected else None)
        aggregates = {}
        default_id = self.registry.default_id
        for candidate in models:
            available = [row for row in rows if row['candidate_id'] == candidate and row['status'] == 'success']
            matched = [success[candidate, episode_id] for episode_id in paired]
            aggregates[candidate] = {
                'metrics': mean(matched), 'sample_count': len(paired),
                'paired_episode_ids': paired, 'successful_count': len(available),
                'missing_count': len(self._cohort) - len(available),
                'original_cohort_size': len(self._cohort),
                'paired_support_fraction': len(paired) / len(self._cohort),
                'post_selection_holdout_metrics': mean([success[candidate, key] for key in holdout]),
                'holdout_sample_count': len(holdout),
                'data_roles': sorted({row['data_role'] for row in available}),
                'failed_episode_ids': [row['episode_id'] for row in rows
                                       if row['candidate_id'] == candidate and row['status'] == 'failure'],
            }
            if candidate != default_id and default_id in models:
                gains = [success[default_id, episode_id]['metrics']['qs']
                         - success[candidate, episode_id]['metrics']['qs']
                         for episode_id in paired]
                aggregates[candidate]['paired_vs_default'] = ({
                    'default_candidate_id': default_id,
                    'gain_definition': 'default_qs_minus_candidate_qs',
                    'mean_qs_gain': float(np.mean(gains)),
                    'median_qs_gain': float(np.median(gains)),
                    'minimum_qs_gain': float(np.min(gains)),
                    'maximum_qs_gain': float(np.max(gains)),
                    'qs_gain_std': float(np.std(gains)),
                    'win_count': int(np.sum(np.asarray(gains) > 0)),
                    'tie_count': int(np.sum(np.asarray(gains) == 0)),
                    'loss_count': int(np.sum(np.asarray(gains) < 0)),
                    'win_fraction': float(np.mean(np.asarray(gains) > 0)),
                } if gains else None)
        failures = [row for row in rows if row['status'] == 'failure']
        unattempted = len(jobs) - len(rows)
        result = {
            'status': 'partial_failure' if failures or unattempted else 'available',
            'diagnostic_kind': 'paired_retrospective_frozen_recompute', 'horizon_hours': 24,
            'cohort_id': self._cohort_id, 'cohort_locked': True, 'cohort_size': len(self._cohort),
            'episodes': list(self._locked_episode_ids), 'sample_count': len(paired),
            'paired_episode_ids': paired, 'paired_support_fraction': len(paired) / len(self._cohort),
            'paired_comparison': True, 'comparison_available': len(models) > 1 and bool(paired),
            'low_support': len(paired) < 3, 'candidates': aggregates,
            'failures': failures, 'unattempted_count': unattempted,
            'logical_inference_units': attempted,
            'cumulative_logical_inference_units': self._charged_diagnostic_units,
            'numeric_inferences': self._diagnostic_registry.stats['numeric_inferences'] - numeric_before,
            'cumulative_numeric_inferences': self._diagnostic_registry.stats['numeric_inferences'],
            'cache_hit_count': len(jobs) - len(pending), 'cache_miss_count': len(pending),
            'current_forecast_comparison': compare_candidates(self.artifacts),
            'limitation': 'Metrics use the common successful episode intersection for every candidate. '
                          'Failures remain charged and cached; loss of cohort support is explicit. '
                          'Retrospective recomputation and training/selection overlap do not imply unbiased accuracy.',
        }
        full = plain({**result, 'rows': rows})
        self.audit.append({'action': action_id, 'complete_diagnostic': full})
        observation = plain(result)
        self.audit.append({'action': action_id, 'observation': observation})
        return {'status': result['status'], 'observation': observation, 'candidate_ids': [],
                'costs': {'historical_evaluations': attempted}}


class R22IterativeToolHost(IterativeToolHost):
    """R22-only bounded diagnostic aggregates; R20/R21 output stays unchanged."""

    def __init__(self, *args, bootstrap_seed=20261111, bootstrap_replicates=2000,
                 material_qs_loss_threshold=.01, **kwargs):
        super().__init__(*args, **kwargs)
        if (isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int)
                or isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int)
                or bootstrap_replicates < 100
                or isinstance(material_qs_loss_threshold, bool)
                or not isinstance(material_qs_loss_threshold, (int, float))
                or not math.isfinite(material_qs_loss_threshold)
                or material_qs_loss_threshold < 0):
            raise ValueError('Invalid R22 diagnostic uncertainty configuration')
        self.bootstrap_seed = bootstrap_seed
        self.bootstrap_replicates = bootstrap_replicates
        self.material_qs_loss_threshold = float(material_qs_loss_threshold)
        self._r22_retrieval = None

    def _score_episode(self, actual, quantiles):
        from ..api_agent.workflow import quantile_loss

        actual = np.asarray(actual, dtype=float)
        quantiles = np.sort(np.asarray(quantiles, dtype=float), axis=-1)
        result = scores(actual, quantiles)
        for index, (start, stop) in enumerate(((0, 6), (6, 12), (12, 18), (18, 24)), 1):
            result[f'qs_bucket_{index}'] = quantile_loss(actual[start:stop], quantiles[start:stop])
        return result

    @staticmethod
    def _summary(values):
        values = np.asarray(values, dtype=float)
        return {'mean': float(values.mean()), 'median': float(np.median(values)),
                'minimum': float(values.min()), 'maximum': float(values.max()),
                'std': float(values.std())}

    def _paired_summary(self, default_rows, candidate_rows):
        episode_ids = sorted(set(default_rows) & set(candidate_rows))
        gains = np.asarray([default_rows[key]['metrics']['qs']
                            - candidate_rows[key]['metrics']['qs'] for key in episode_ids])
        if not len(gains):
            return None
        coverage_deltas = np.asarray([
            candidate_rows[key]['metrics']['coverage90'] - default_rows[key]['metrics']['coverage90']
            for key in episode_ids])
        width_deltas = np.asarray([
            candidate_rows[key]['metrics']['width90'] - default_rows[key]['metrics']['width90']
            for key in episode_ids])
        mae_gains = np.asarray([default_rows[key]['metrics']['mae']
                                - candidate_rows[key]['metrics']['mae'] for key in episode_ids])
        calibration_damage = np.asarray([
            abs(candidate_rows[key]['metrics']['coverage90'] - .9)
            - abs(default_rows[key]['metrics']['coverage90'] - .9) for key in episode_ids])
        bucket_gains = {
            f'hours_{start}_{stop}': self._summary([
                default_rows[key]['metrics'][f'qs_bucket_{index}']
                - candidate_rows[key]['metrics'][f'qs_bucket_{index}'] for key in episode_ids])
            for index, (start, stop) in enumerate(((1, 6), (7, 12), (13, 18), (19, 24)), 1)
        }
        lower_count = max(1, int(math.ceil(len(gains) * .25)))
        sorted_gains = np.sort(gains)
        material_losses = gains < -self.material_qs_loss_threshold
        rng = np.random.default_rng(self.bootstrap_seed)
        bootstrap_means = gains[rng.integers(0, len(gains),
                                              size=(self.bootstrap_replicates, len(gains)))].mean(axis=1)
        if len(gains) == 1:
            leave_one_out = gains.copy()
        else:
            leave_one_out = np.asarray([np.delete(gains, index).mean() for index in range(len(gains))])
        distances = {item['episode_id']: float(item['distance'])
                     for item in (self._r22_retrieval or {}).get('items', [])}
        if set(episode_ids) <= set(distances):
            weights = np.asarray([1 / (1 + distances[key]) for key in episode_ids])
            weighted_mean = float(np.average(gains, weights=weights))
            effective_support = float(weights.sum() ** 2 / np.square(weights).sum())
            weighted_available = True
        else:
            weighted_mean = effective_support = None
            weighted_available = False
        return {
            'common_support': len(gains),
            'horizon_qs_gain': bucket_gains,
            'mae_gain': self._summary(mae_gains),
            'coverage90_delta': self._summary(coverage_deltas),
            'width90_delta': self._summary(width_deltas),
            'calibration_damage': self._summary(calibration_damage),
            'downside': {
                'worst_paired_qs_loss': float(max(0., -gains.min())),
                'lower_tail_mean_qs_gain': float(sorted_gains[:lower_count].mean()),
                'loss_fraction': float(np.mean(gains < 0)),
                'material_loss_fraction': float(material_losses.mean()),
                'material_qs_loss_threshold': self.material_qs_loss_threshold,
            },
            'similarity_weighted': {
                'available': weighted_available, 'mean_qs_gain': weighted_mean,
                'effective_support': effective_support,
                'sign_agreement_with_unweighted': (
                    bool(np.sign(weighted_mean) == np.sign(gains.mean()))
                    if weighted_available else None),
            },
            'bootstrap_mean_qs_gain': {
                'role': 'small_sample_supporting_evidence_not_significance_test',
                'seed': self.bootstrap_seed, 'replicates': self.bootstrap_replicates,
                'percentiles': [2.5, 97.5],
                'interval': [float(value) for value in np.percentile(bootstrap_means, [2.5, 97.5])],
            },
            'leave_one_episode_out': {
                'minimum_mean_qs_gain': float(leave_one_out.min()),
                'maximum_mean_qs_gain': float(leave_one_out.max()),
                'sign_stable': bool(np.all(np.sign(leave_one_out) == np.sign(gains.mean()))),
            },
        }

    def _current_comparison(self, observation):
        comparison = observation['current_forecast_comparison']
        artifacts = list(self.artifacts.values())
        comparison['candidates'] = {
            artifact.candidate_id: {
                'q05_mean': float(artifact.quantiles[:, 0].mean()),
                'median_mean': float(artifact.quantiles[:, 2].mean()),
                'q95_mean': float(artifact.quantiles[:, -1].mean()),
                'width90_mean': float((artifact.quantiles[:, -1] - artifact.quantiles[:, 0]).mean()),
            } for artifact in artifacts
        }
        comparison['pairs'] = [{
            'a': first.candidate_id, 'b': second.candidate_id,
            'median_absolute_difference': float(np.abs(
                first.quantiles[:, 2] - second.quantiles[:, 2]).mean()),
            'q05_absolute_difference': float(np.abs(
                first.quantiles[:, 0] - second.quantiles[:, 0]).mean()),
            'q95_absolute_difference': float(np.abs(
                first.quantiles[:, -1] - second.quantiles[:, -1]).mean()),
            'width90_absolute_difference': float(np.abs(
                (first.quantiles[:, -1] - first.quantiles[:, 0])
                - (second.quantiles[:, -1] - second.quantiles[:, 0])).mean()),
        } for index, first in enumerate(artifacts) for second in artifacts[index + 1:]]

    def execute(self, context, action_id, state):
        result = super().execute(context, action_id, state)
        if action_id.startswith('retrieve_analogs:'):
            self._r22_retrieval = plain(result['observation'])
            return result
        if not action_id.startswith('diagnose:'):
            return result
        complete = next(item['complete_diagnostic'] for item in reversed(self.audit)
                        if item.get('action') == action_id and 'complete_diagnostic' in item)
        rows = [row for row in complete['rows'] if row['status'] == 'success']
        by_candidate = {}
        for row in rows:
            by_candidate.setdefault(row['candidate_id'], {})[row['episode_id']] = row
        default_id = self.registry.default_id
        for candidate_id, candidate in result['observation']['candidates'].items():
            if candidate_id == default_id or default_id not in by_candidate:
                continue
            summary = self._paired_summary(by_candidate[default_id], by_candidate.get(candidate_id, {}))
            if summary is not None:
                candidate['paired_vs_default'].update(summary)
        self._current_comparison(result['observation'])
        result['observation']['r22_aggregate_schema'] = 'r22-paired-diagnostic-v1'
        observation = plain(result['observation'])
        result['observation'] = observation
        self.audit.append({'action': action_id, 'r22_observation': observation})
        return result
