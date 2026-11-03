"""Immutable host-side objects. No evaluation labels belong in ForecastContext."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import hashlib
import json
from numbers import Integral, Real
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np

from forecasting.api_agent.workflow import Snapshot


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def frozen(value):
    if isinstance(value, np.ndarray):
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result
    if isinstance(value, Mapping):
        return MappingProxyType({k: frozen(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(frozen(v) for v in value)
    return value


@dataclass(frozen=True)
class ForecastContext:
    snapshot: Snapshot
    model_inputs: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    cutoff: int | None = None
    memory_watermark: int | None = None

    def __post_init__(self):
        for value in (self.snapshot.origin_unix, self.cutoff, self.memory_watermark):
            if value is not None and (isinstance(value, bool) or not isinstance(value, Integral)):
                raise ValueError('Context timestamps must be integer Unix seconds')
        cutoff = self.snapshot.origin_unix if self.cutoff is None else int(self.cutoff)
        watermark = cutoff if self.memory_watermark is None else int(self.memory_watermark)
        if cutoff != self.snapshot.origin_unix or watermark > cutoff:
            raise ValueError('Context must use its origin cutoff and a nonfuture memory watermark')
        if {'actual', 'target', 'labels'} & set(self.model_inputs):
            raise ValueError('Evaluation labels cannot enter model inputs')
        for name, value in self.model_inputs.items():
            if not np.isfinite(np.asarray(value)).all():
                raise ValueError('Nonfinite canonical input: ' + name)
        if {'actual', 'target', 'labels'} & set(self.metadata):
            raise ValueError('Evaluation labels cannot enter metadata')
        digest(self.metadata)
        object.__setattr__(self, 'model_inputs', frozen(self.model_inputs))
        object.__setattr__(self, 'metadata', frozen(self.metadata))
        object.__setattr__(self, 'cutoff', cutoff)
        object.__setattr__(self, 'memory_watermark', watermark)

    @property
    def region(self):
        return self.snapshot.region

    @property
    def origin_unix(self):
        return self.snapshot.origin_unix

    def identity(self):
        return digest({'snapshot': self.snapshot.identity(), 'model_inputs': self.model_inputs,
                       'metadata': self.metadata, 'cutoff': self.cutoff,
                       'memory_watermark': self.memory_watermark})


@dataclass(frozen=True)
class ForecastArtifact:
    candidate_id: str
    context_id: str
    quantiles: np.ndarray
    target_time_unix: np.ndarray
    model_id: str
    input_hash: str
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if not all(isinstance(v, str) and v for v in (self.candidate_id, self.context_id,
                                                     self.model_id, self.input_hash)):
            raise ValueError('Forecast identifiers must be nonempty strings')
        q, axis = np.asarray(self.quantiles), np.asarray(self.target_time_unix)
        if q.shape != (24, 5) or not np.isfinite(q).all():
            raise ValueError('Forecast must contain finite 24x5 native quantiles')
        if axis.shape != (24,) or axis.dtype.kind not in 'iu' or not np.all(np.diff(axis) == 3600):
            raise ValueError('Forecast needs an hourly int64 target axis')
        digest(self.provenance)
        object.__setattr__(self, 'quantiles', frozen(q.astype(np.float64)))
        object.__setattr__(self, 'target_time_unix', frozen(axis.astype(np.int64)))
        object.__setattr__(self, 'provenance', frozen(self.provenance))

    def identity(self):
        return digest({f.name: getattr(self, f.name) for f in fields(self)})


@dataclass(frozen=True)
class BudgetLimits:
    api_calls: int = 8
    tool_calls: int = 12
    candidates: int = 5
    episodes: int = 8
    historical_evaluations: int = 40
    api_timeout_seconds: float = 30
    origin_deadline_seconds: float = 180

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError('Budgets must be finite positive numbers')
            if not f.name.endswith('_seconds') and not isinstance(value, Integral):
                raise ValueError('Count budgets must be integers')
