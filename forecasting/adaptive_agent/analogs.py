"""Mature historical episode retrieval; contexts/targets never enter tool summaries."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math

import numpy as np

from .state import summarize_state

STRATEGIES = ('recent', 'seasonal', 'state', 'event', 'diverse')
_AEST = timezone(timedelta(hours=10))


class AnalogStore:
    def __init__(self, episodes=()):
        self._episodes = {}
        for value in episodes:
            row = {key: item if key == 'context' else deepcopy(item) for key, item in value.items()}
            context = row.get('context')
            if not row.get('episode_id') or context is None:
                raise ValueError('Episode requires immutable ID and full ForecastContext')
            row.setdefault('region', context.region)
            row.setdefault('origin_unix', context.origin_unix)
            if row['region'] != context.region or row['origin_unix'] != context.origin_unix:
                raise ValueError('Episode context identity mismatch')
            actual = np.asarray(row.get('actual'), dtype=float).copy()
            if actual.shape != (24,) or not np.isfinite(actual).all():
                raise ValueError('Episode requires 24 finite mature observations')
            availability = row.get('outcome_available_at')
            if isinstance(availability, bool) or not isinstance(availability, int) or availability < context.origin_unix + 24 * 3600:
                raise ValueError('Episode outcomes cannot mature before the final interval ends')
            actual.setflags(write=False)
            row['actual'] = actual
            state = summarize_state(context)
            row.setdefault('similarity_fields', state['similarity_fields'])
            row.setdefault('state_tags', state['state_tags'])
            row.setdefault('original_forecast_ids', [])
            row.setdefault('data_role', 'unknown')
            row.setdefault('metrics', {})
            for feature in row['similarity_fields'].values():
                if not math.isfinite(float(feature)):
                    raise ValueError('Nonfinite episode similarity feature')
            if row['episode_id'] in self._episodes:
                raise ValueError('Duplicate immutable episode ID')
            self._episodes[row['episode_id']] = row
        payload = [{'id': r['episode_id'], 'context_id': r['context'].identity(),
                    'available_at': r['outcome_available_at'], 'actual': r['actual'].tolist(),
                    'state_tags': r['state_tags'], 'similarity_fields': r['similarity_fields'],
                    'metrics': r['metrics'], 'data_role': r['data_role']}
                   for r in sorted(self._episodes.values(), key=lambda x: x['episode_id'])]
        self.fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def _visible(self, context):
        cutoff = min(context.cutoff, context.memory_watermark)
        return [r for r in self._episodes.values() if r['region'] == context.region
                and r['origin_unix'] < context.origin_unix and r['outcome_available_at'] <= cutoff]

    def eligible_count(self, context):
        """Count mature same-region records without retrieving targets or ranking."""
        return len(self._visible(context))

    def get_episode(self, context, episode_id):
        row = next((r for r in self._visible(context) if r['episode_id'] == episode_id), None)
        if row is None:
            raise ValueError('Episode unavailable at this cutoff/watermark/region')
        # Keep immutable context object, clone mutable dictionaries and target array.
        return {key: value if key == 'context' else deepcopy(value) for key, value in row.items()}

    def retrieve(self, context, strategy='diverse', limit=8, event_tags=()):
        if strategy not in STRATEGIES or not 1 <= limit <= 8:
            raise ValueError('Unregistered analog strategy or excess sample budget')
        visible = self._visible(context)
        fields = summarize_state(context)['similarity_fields']
        names = sorted(fields)
        scales = {k: max(float(np.std([r['similarity_fields'].get(k, fields[k]) for r in visible])) if visible else 0.,
                         abs(fields[k]) * .01, 1e-6) for k in names}
        now = datetime.fromtimestamp(context.origin_unix, _AEST)
        tags = set(event_tags)
        ranked = []
        for row in visible:
            distance = sum(((row['similarity_fields'].get(k, fields[k]) - fields[k]) / scales[k]) ** 2
                           for k in names) ** .5
            date = datetime.fromtimestamp(row['origin_unix'], _AEST)
            month_gap = min((now.month - date.month) % 12, (date.month - now.month) % 12)
            age_days = (context.origin_unix - row['origin_unix']) / 86400
            if strategy == 'event' and not tags.intersection(row['state_tags']):
                continue
            score = {'recent': age_days / 30 + distance, 'seasonal': month_gap + distance,
                     'state': distance, 'event': distance, 'diverse': distance}[strategy]
            ranked.append((score, row['episode_id'], row, distance))
        ranked.sort(key=lambda x: (x[0], x[1]))
        pool = ranked[:32]
        selected = []
        while pool and len(selected) < limit:
            if strategy == 'diverse' and selected:
                # Penalize repeated state labels while retaining standardized similarity.
                seen_tags = {tag for item in selected for tag in item[2]['state_tags']}
                pool.sort(key=lambda item: (item[0] + len(seen_tags.intersection(item[2]['state_tags'])), item[1]))
            item = pool.pop(0)
            # At most one target window per day; no overlapping 24-hour outcomes.
            if any(abs(item[2]['origin_unix'] - old[2]['origin_unix']) < 86400 for old in selected):
                continue
            selected.append(item)
        items = [{'episode_id': row['episode_id'], 'origin_unix': row['origin_unix'],
                  'outcome_available_at': row['outcome_available_at'],
                  'state_tags': list(row['state_tags']), 'distance': float(distance),
                  'metrics': deepcopy(row['metrics']), 'data_role': row['data_role']}
                 for _, _, row, distance in selected]
        return {'status': 'available' if items else 'insufficient_mature_history',
                'strategy': strategy, 'items': items, 'support_count': len(items),
                'eligible_count': len(visible), 'warehouse_fingerprint': self.fingerprint,
                'low_support': len(items) < 3}
