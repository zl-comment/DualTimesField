"""Frozen LightGBM candidates that differ only by an admitted input operation."""
from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .contracts import ForecastArtifact, digest, plain
from .input_operations import OPERATIONS


@dataclass(frozen=True)
class InputForecastSpec:
    model_id: str
    operation_id: str
    operation_sha256: str


def load_operator_admission(path, *, seed):
    """Validate the immutable development-only operation admission artifact."""
    path = Path(path)
    raw = path.read_bytes()
    record = json.loads(raw)
    admitted = record.get('admitted_operation_ids')
    if (record.get('status') != 'ADMITTED' or record.get('split') != 'calibration'
            or record.get('seed') != seed or not isinstance(admitted, list)
            or not admitted or admitted[0] != 'identity'
            or len(admitted) != len(set(admitted))):
        raise ValueError('Invalid R21 development operation admission artifact')
    if record.get('selection_rule_fixed_before_scores', {}).get('validation') != (
            'reported after selection and never changes admission'):
        raise ValueError('R21 admission artifact does not isolate validation reporting')
    for operation_id in admitted:
        if operation_id not in OPERATIONS:
            raise ValueError('R21 admission names an unknown input operation')
        row = record.get('operations', {}).get(operation_id, {})
        spec = row.get('spec')
        if spec != asdict(OPERATIONS[operation_id]):
            raise ValueError('R21 admitted operation spec changed')
        if operation_id != 'identity' and not row.get('admission', {}).get('passes_rule_before_top_four_cap'):
            raise ValueError('R21 admitted operation did not pass the fixed design rule')
    eligible = [(row.get('admission', {}).get('design_conditional_oracle_qs_gain'), key)
                for key, row in record.get('operations', {}).items()
                if row.get('admission', {}).get('passes_rule_before_top_four_cap')]
    expected = [key for _, key in sorted(eligible, reverse=True)[:4]]
    if admitted != ['identity', *expected]:
        raise ValueError('R21 admitted operation ranking does not reproduce from design scores')
    return {
        'path': str(path.resolve()),
        'sha256': hashlib.sha256(raw).hexdigest(),
        'admitted_operation_ids': tuple(admitted),
        'screen_source': record['source'],
        'semantics': record['semantics'],
        'selection_rule_fixed_before_scores': record['selection_rule_fixed_before_scores'],
    }


class InputForecastRegistry:
    """One frozen predictor family with a finite audited operation registry."""
    def __init__(self, adapter, release_id, *, max_current_candidates=5, cache_size=256):
        if not release_id or max_current_candidates < 1 or cache_size < 1:
            raise ValueError('Invalid R21 registry configuration')
        admitted = tuple(adapter.admitted_operation_ids)
        if not admitted or admitted[0] != 'identity':
            raise ValueError('R21 registry requires identity as its default')
        self.adapters = {'lightgbm': adapter}
        self.release_id = str(release_id)
        self.max_current_candidates = int(max_current_candidates)
        self.cache_size = int(cache_size)
        self.admitted_operation_ids = admitted
        self.specs = {
            'lightgbm-input:' + key: InputForecastSpec(
                'lightgbm', key, OPERATIONS[key].identity())
            for key in admitted
        }
        self.default_id = 'lightgbm-input:identity'
        self.model_identities = {'lightgbm': copy.deepcopy(adapter.identity)}
        self._cache = OrderedDict()
        self._requested = {}
        self.stats = {'logical_requests': 0, 'numeric_inferences': 0, 'cache_hits': 0}

    def available(self):
        return {key: asdict(value) for key, value in self.specs.items()}

    def fork(self, *, max_current_candidates=40):
        return type(self)(self.adapters['lightgbm'], self.release_id,
                          max_current_candidates=max_current_candidates,
                          cache_size=self.cache_size)

    def forecast(self, context, candidate_id):
        if candidate_id not in self.specs:
            raise ValueError('Unregistered R21 input-operation candidate')
        adapter = self.adapters['lightgbm']
        if digest(adapter.identity) != digest(self.model_identities['lightgbm']):
            raise ValueError('Frozen R21 LightGBM adapter identity changed')
        context_id = context.identity()
        requested = self._requested.setdefault(context_id, set())
        if candidate_id not in requested and len(requested) >= self.max_current_candidates:
            raise ValueError('Current candidate budget exhausted')
        requested.add(candidate_id)
        self.stats['logical_requests'] += 1
        spec = self.specs[candidate_id]
        input_hash = digest({'release': self.release_id, 'context': context_id,
                             'spec': asdict(spec), 'model': self.model_identities['lightgbm']})
        if input_hash in self._cache:
            self.stats['cache_hits'] += 1
            self._cache.move_to_end(input_hash)
            return self._cache[input_hash]
        self.stats['numeric_inferences'] += 1
        raw = adapter.predict(context, spec.operation_id)
        quantiles = np.asarray(raw.quantiles, dtype=np.float64)
        if (raw.candidate_id != candidate_id or raw.context_id != context_id
                or raw.model_id != 'lightgbm' or quantiles.shape != (24, 5)
                or not np.isfinite(quantiles).all()):
            raise ValueError('R21 adapter violated its frozen forecast contract')
        quantiles = np.sort(quantiles, axis=-1)
        artifact = ForecastArtifact(
            candidate_id=candidate_id, context_id=context_id, quantiles=quantiles,
            target_time_unix=context.snapshot.target_time_unix,
            model_id='lightgbm', input_hash=input_hash,
            provenance={
                'release_id': self.release_id,
                'spec': asdict(spec),
                'model_identity': copy.deepcopy(self.model_identities['lightgbm']),
                'context_id': context_id,
                'cutoff': int(context.cutoff),
                'adapter_forecast_id': raw.identity(),
                'adapter_input_hash': raw.input_hash,
                'input_operation': plain(raw.provenance['input_operation']),
                'output_sha256': hashlib.sha256(quantiles.tobytes()).hexdigest(),
                'training_performed': False,
            })
        self._cache[input_hash] = artifact
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return artifact
