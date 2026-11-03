"""Bounded, on-demand frozen predictions with release/input/weight identities."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import copy
import hashlib
import numpy as np
from .contracts import digest


@dataclass(frozen=True)
class ForecastSpec:
    model_id: str
    profile_id: str
    window_hours: int


class ForecastRegistry:
    def __init__(self, adapters, release_id, *, max_current_candidates=5, default_id=None,
                 cache_size=256):
        if not release_id or max_current_candidates < 1 or cache_size < 1:
            raise ValueError('Invalid registry configuration')
        if set(adapters) - {'lightgbm', 'chronos'} or not adapters:
            raise ValueError('Only registered frozen model adapters are allowed')
        self.adapters = dict(adapters)
        self.release_id = str(release_id)
        self.max_current_candidates = int(max_current_candidates)
        self.cache_size = int(cache_size)
        self.specs = {}
        if 'lightgbm' in adapters:
            self.specs['lightgbm:full:72'] = ForecastSpec('lightgbm', 'full', 72)
        if 'chronos' in adapters:
            for profile in ('base', 'calendar', 'price_only'):
                for window in (24, 48, 72):
                    self.specs[f'chronos:{profile}:{window}'] = ForecastSpec('chronos', profile, window)
        self.default_id = default_id or ('lightgbm:full:72' if 'lightgbm' in adapters else 'chronos:base:72')
        if self.default_id not in self.specs:
            raise ValueError('Default prediction is not registered')
        self.model_identities = {key: copy.deepcopy(value.identity) for key, value in adapters.items()}
        self._cache = OrderedDict()
        self._requested = {}
        self.stats = {'logical_requests': 0, 'numeric_inferences': 0, 'cache_hits': 0}

    def available(self):
        return {key: asdict(value) for key, value in self.specs.items()}

    def fork(self, *, max_current_candidates=40):
        return type(self)(self.adapters, self.release_id,
                          max_current_candidates=max_current_candidates,
                          default_id=self.default_id, cache_size=self.cache_size)

    def forecast(self, context, candidate_id):
        from .contracts import ForecastArtifact
        if candidate_id not in self.specs:
            raise ValueError('Unregistered forecast action')
        spec = self.specs[candidate_id]
        if digest(self.adapters[spec.model_id].identity) != digest(self.model_identities[spec.model_id]):
            raise ValueError('Frozen adapter identity changed')
        context_id = context.identity()
        requested = self._requested.setdefault(context_id, set())
        if candidate_id not in requested and len(requested) >= self.max_current_candidates:
            raise ValueError('Current candidate budget exhausted')
        # Consume the unique candidate budget before execution: repeated failures
        # cannot bypass the cap by asking for more expensive alternatives.
        requested.add(candidate_id)
        self.stats['logical_requests'] += 1
        input_hash = digest({'release': self.release_id, 'context': context_id,
                             'spec': asdict(spec), 'model': self.model_identities[spec.model_id]})
        if input_hash in self._cache:
            self.stats['cache_hits'] += 1
            self._cache.move_to_end(input_hash)
            return self._cache[input_hash]
        self.stats['numeric_inferences'] += 1
        quantiles = np.asarray(self.adapters[spec.model_id].predict(
            context, spec.profile_id, spec.window_hours), dtype=np.float64)
        if quantiles.shape != (24, 5) or not np.isfinite(quantiles).all():
            raise ValueError('Predictor violated 24-hour five-quantile output contract')
        artifact = ForecastArtifact(
            candidate_id=candidate_id, context_id=context_id,
            quantiles=np.sort(quantiles, axis=-1),
            target_time_unix=context.snapshot.target_time_unix,
            model_id=spec.model_id, input_hash=input_hash,
            provenance={'release_id': self.release_id, 'spec': asdict(spec),
                        'model_identity': copy.deepcopy(self.model_identities[spec.model_id]),
                        'context_id': context_id, 'cutoff': int(context.cutoff),
                        'output_sha256': hashlib.sha256(np.sort(quantiles, axis=-1).tobytes()).hexdigest(),
                        'training_performed': False})
        self._cache[input_hash] = artifact
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return artifact
