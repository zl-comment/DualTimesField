"""Concrete tool host connecting the complete R18 modules to typed decisions."""
from __future__ import annotations

from itertools import combinations

from .contracts import digest, plain
from .diagnostics import compare_candidates, evaluate_episodes
from .events import EventResolver
from .pooling import pool_quantiles
from .state import summarize_state


class ToolHost:
    def __init__(self, registry, evidence, analogs, feedback, *, information_scope='full_evidence'):
        if information_scope not in ('full_evidence', 'common_input'):
            raise ValueError('Unknown information scope')
        self.registry, self.evidence = registry, evidence
        self.analogs, self.feedback = analogs, feedback
        self.information_scope = information_scope
        self.context_id = None
        self.artifacts, self.documents, self.event_facts, self.episode_ids = {}, {}, [], []
        self.read_document_ids, self.interpreted_document_ids = set(), set()
        self.audit = []

    def _check(self, context):
        if self.context_id is not None and self.context_id != context.identity():
            raise ValueError('ToolHost belongs to another origin; create a new host')
        self.context_id = context.identity()

    def default(self, context):
        self._check(context)
        artifact = self.registry.forecast(context, self.registry.default_id)
        self.artifacts[artifact.candidate_id] = artifact
        return artifact

    def artifact(self, context, candidate_id):
        self._check(context)
        if candidate_id not in self.artifacts:
            raise ValueError('Candidate has not been generated in this context')
        return self.artifacts[candidate_id]

    def initial_observation(self, context):
        self._check(context)
        summary = summarize_state(context)
        return {'numerical_state': summary, 'information_scope': self.information_scope,
                'evidence_status': self.evidence.search(context, limit=1)['status']
                    if self.information_scope == 'full_evidence' else 'disabled_common_input',
                'registered_models': list(self.registry.adapters),
                'forecast_semantics': 'all values are frozen predictions, not future observations',
                'default_summary': compare_candidates(self.artifacts)}

    def actions(self, context, state):
        self._check(context)
        actions = {}

        def add(key, description, kind, **costs):
            actions[key] = {'description': description, 'kind': kind, 'costs': costs}

        add('inspect_state', 'Inspect visible numerical state and currently mature feedback.', 'assess_state')
        if 'chronos' in self.registry.adapters:
            add('inspect_recent_fit', 'Compare three Chronos profiles on the last already observed day (48-to-24 hindcast).',
                'compare_candidates', historical_evaluations=3)
        if self.information_scope == 'full_evidence':
            add('search_evidence', 'Search verified, cutoff-visible notices and event evidence.', 'interpret_evidence')
            add('inspect_feedback', 'Read only fully mature errors from this experiment arm.', 'assess_state')
            if not self.episode_ids:
                for strategy in ('recent', 'seasonal', 'state', 'event', 'diverse'):
                    if strategy == 'event' and not self.event_facts:
                        continue
                    add('retrieve_analogs:' + strategy, 'Retrieve up to eight mature historical episodes using ' + strategy,
                        'retrieve_analogs', episodes=8)
            for evidence_id, record in self.documents.items():
                if evidence_id not in self.read_document_ids:
                    add('read_evidence:' + evidence_id,
                        'Read exact quoted event candidates and publication/version metadata from ' + evidence_id,
                        'interpret_evidence')
                    continue
                if evidence_id in self.interpreted_document_ids:
                    continue
                for choice, description in EventResolver().resolve(record)['choices'].items():
                    add(f'interpret_event:{evidence_id}:{choice}', description + ' from ' + evidence_id,
                        'interpret_evidence')
        for candidate_id in self.registry.available():
            if candidate_id not in self.artifacts:
                add('forecast:' + candidate_id, 'Generate frozen candidate ' + candidate_id,
                    'compose_forecast_request', candidates=1)
        generated = list(self.artifacts)
        if len(generated) > 1:
            add('compare:' + '|'.join(generated), 'Compare current candidate medians, interval widths and disagreement.',
                'compare_candidates')
            # Pool only actual model forecasts, not recursively pooled forecasts.
            models = [k for k in generated if k in self.registry.specs]
            for a, b in combinations(models, 2):
                for weight in (.25, .5, .75):
                    add(f'pool:{a}|{b}|{weight:g}', f'Generate registered quantile pool: {weight:g} {a} + {1-weight:g} {b}.',
                        'compare_candidates', candidates=1)
        if self.episode_ids:
            models = [k for k in generated if k in self.registry.specs]
            if models:
                add('diagnose:' + '|'.join(models), 'Evaluate generated models on the selected mature analog episodes.',
                    'compare_candidates', historical_evaluations=len(models) * len(self.episode_ids))
        return actions

    def execute(self, context, action_id, state):
        self._check(context)
        offered = self.actions(context, state)
        if action_id not in offered:
            raise ValueError('Action is unavailable for this snapshot')
        costs, candidates = {}, []
        if action_id == 'inspect_state':
            feedback = self.feedback.snapshot(context) if self.information_scope == 'full_evidence' else []
            result = summarize_state(context, feedback)
        elif action_id == 'inspect_feedback':
            rows = self.feedback.snapshot(context)
            result = {'rows': rows[-8:], 'total_mature_count': len(rows), 'arm_id': self.feedback.arm_id,
                      'ledger_fingerprint': digest({'rows': rows, 'arm_id': self.feedback.arm_id,
                                                    'watermark': context.memory_watermark})}
        elif action_id == 'search_evidence':
            result = self.evidence.search(context, limit=3)
            self.audit.append({'action': action_id, 'source_warehouse_fingerprint': result['warehouse_fingerprint']})
            for record in result['items']:
                self.documents[record['evidence_id']] = record
            # Full original bodies stay in the immutable warehouse. LLM gets
            # bounded excerpts and exact IDs for quoted typed interpretations.
            result = {**result, 'items': [{**record, 'payload': {'text': record['payload']['text'][:1200]}}
                                         for record in result['items']]}
            # API-visible identity binds only the returned as-of evidence. The
            # frozen full warehouse hash is audit-only: future records must not
            # perturb a past decision's state even through opaque hash changes.
            result['warehouse_fingerprint'] = digest({key: value for key, value in result.items()
                                                       if key != 'warehouse_fingerprint'})
        elif action_id.startswith('read_evidence:'):
            evidence_id = action_id[len('read_evidence:'):]
            record = self.evidence.read(context, evidence_id)
            self.documents[evidence_id] = record
            quoted = EventResolver().resolve(record)
            result = {'status': 'read',
                      'evidence_metadata': {key: value for key, value in record.items() if key != 'payload'},
                      'quoted_candidates': quoted['facts'], 'choices': quoted['choices'],
                      'text_scope': 'Exact candidate quote spans, not the entire source document.'}
            self.read_document_ids.add(evidence_id)
        elif action_id.startswith('interpret_event:'):
            suffix = action_id[len('interpret_event:'):]
            evidence_id, choice = suffix.rsplit(':', 1)
            result = EventResolver().resolve(self.documents[evidence_id], choice)
            self.event_facts.extend(result['facts'])
            self.interpreted_document_ids.add(evidence_id)
        elif action_id.startswith('retrieve_analogs:'):
            strategy = action_id.split(':', 1)[1]
            result = self.analogs.retrieve(context, strategy=strategy, limit=8,
                                          event_tags=[r['event_type'] for r in self.event_facts])
            self.audit.append({'action': action_id, 'source_warehouse_fingerprint': result['warehouse_fingerprint']})
            result['warehouse_fingerprint'] = digest({key: value for key, value in result.items()
                                                       if key != 'warehouse_fingerprint'})
            self.episode_ids = [r['episode_id'] for r in result['items']]
            costs = {'episodes': len(self.episode_ids)}
        elif action_id.startswith('forecast:'):
            candidate_id = action_id[len('forecast:'):]
            artifact = self.registry.forecast(context, candidate_id)
            self.artifacts[candidate_id] = artifact
            candidates, costs = [candidate_id], {'candidates': 1}
            result = compare_candidates([artifact])
        elif action_id.startswith('compare:'):
            result = compare_candidates(self.artifacts)
        elif action_id.startswith('pool:'):
            a, b, weight = action_id[len('pool:'):].split('|')
            artifact = pool_quantiles(self.artifacts[a], self.artifacts[b], float(weight))
            self.artifacts[artifact.candidate_id] = artifact
            candidates, costs = [artifact.candidate_id], {'candidates': 1}
            result = compare_candidates([artifact])
            result['pool_semantics'] = 'same-quantile weighted average; not exact mixture quantiles'
        elif action_id.startswith('diagnose:'):
            ids = action_id[len('diagnose:'):].split('|')
            episodes = [self.analogs.get_episode(context, key) for key in self.episode_ids]
            full = evaluate_episodes(context, ids, episodes, self.registry,
                                     max_units=40 - state['costs']['historical_evaluations'])
            self.audit.append({'action': action_id, 'complete_diagnostic': full})
            result = {key: value for key, value in full.items() if key != 'rows'}
            costs = {'historical_evaluations': full['logical_inference_units']}
        elif action_id == 'inspect_recent_fit':
            from .diagnostics import evaluate_recent_history
            ids = ['chronos:' + p + ':72' for p in ('base', 'calendar', 'price_only')]
            result = evaluate_recent_history(context, ids, self.registry,
                                            max_units=40 - state['costs']['historical_evaluations'])
            costs = {'historical_evaluations': result['logical_inference_units']}
        else:
            raise ValueError('Unregistered execution route')
        observation = plain(result)
        self.audit.append({'action': action_id, 'observation': observation})
        return {'status': result.get('status', 'ok'), 'observation': observation,
                'candidate_ids': candidates, 'costs': costs}
