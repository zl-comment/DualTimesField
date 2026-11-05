"""R20 incremental, paired diagnostics on one immutable mature episode cohort.

Only this subclass changes diagnostic caching. R18/R19 ToolHost semantics stay
unchanged. A failed candidate/episode attempt is a cached charged observation,
never an invitation to silently retry until the model produces a favorable row.
"""
from __future__ import annotations

from dataclasses import asdict

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
                       'metrics': scores(episode['actual'], artifact.quantiles)}
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
