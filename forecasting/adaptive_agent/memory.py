"""Append-only, experiment-arm isolated forecasts and fully mature outcome feedback."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import threading

import numpy as np

from ..api_agent.workflow import quantile_loss
from .contracts import plain, ForecastArtifact


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class FeedbackLedger:
    def __init__(self, arm_id, path=None):
        if not isinstance(arm_id, str) or not arm_id:
            raise ValueError('Feedback ledger needs an experiment arm ID')
        self.arm_id = arm_id
        self.path = Path(path) if path is not None else None
        self._events, self._forecasts, self._outcomes = [], {}, {}
        self._lock = threading.RLock()
        self._disk_size = 0
        self._disk_mtime_ns = None
        if self.path is not None and self.path.exists():
            with self.path.open() as stream:
                for line in stream:
                    self._accept(json.loads(line))
            stat = self.path.stat()
            self._disk_size, self._disk_mtime_ns = stat.st_size, stat.st_mtime_ns

    def _validate_event(self, event):
        if not isinstance(event, dict) or set(event) != {
                'arm_id', 'kind', 'payload', 'sequence', 'previous_hash', 'event_hash'}:
            raise ValueError('Malformed feedback event schema')
        if event.get('arm_id') != self.arm_id:
            raise ValueError('Cross-arm feedback access')
        expected_previous = self._events[-1]['event_hash'] if self._events else None
        if (isinstance(event.get('sequence'), bool) or event.get('sequence') != len(self._events)
                or event.get('previous_hash') != expected_previous):
            raise ValueError('Broken append-only feedback chain')
        body = {k: v for k, v in event.items() if k != 'event_hash'}
        if _digest(body) != event.get('event_hash'):
            raise ValueError('Feedback hash mismatch')
        payload = event['payload']
        if not isinstance(payload, dict) or not isinstance(payload.get('forecast_id'), str):
            raise ValueError('Malformed feedback payload')
        key = payload['forecast_id']
        if event['kind'] == 'forecast':
            expected = {'forecast_id', 'candidate_id', 'context_id', 'model_id', 'region',
                        'origin_unix', 'issued_at', 'target_time_unix', 'quantiles', 'input_hash', 'provenance'}
            if set(payload) != expected:
                raise ValueError('Malformed forecast payload schema')
            if key in self._forecasts:
                raise ValueError('Repeated immutable forecast record')
            for field in ('origin_unix', 'issued_at'):
                if isinstance(payload[field], bool) or not isinstance(payload[field], int):
                    raise ValueError('Forecast timestamps must be integer Unix seconds')
            if payload['issued_at'] != payload['origin_unix'] or payload['region'] not in ('NSW1', 'QLD1', 'TAS1'):
                raise ValueError('Forecast issue time or region is invalid')
            q, axis = np.asarray(payload['quantiles']), np.asarray(payload['target_time_unix'])
            if axis.dtype.kind not in 'iu' or not np.array_equal(axis, payload['origin_unix'] + np.arange(24) * 3600):
                raise ValueError('Recorded forecast target axis is invalid')
            rebuilt = ForecastArtifact(payload['candidate_id'], payload['context_id'], q, axis,
                                       payload['model_id'], payload['input_hash'], payload['provenance'])
            if (np.diff(q, axis=1) < 0).any() or rebuilt.identity() != key:
                raise ValueError('Recorded forecast content or immutable identity is invalid')
        elif event['kind'] == 'outcome':
            if set(payload) != {'forecast_id', 'outcome_available_at', 'actual', 'metrics'}:
                raise ValueError('Malformed outcome payload schema')
            if key not in self._forecasts or key in self._outcomes:
                raise ValueError('Outcome requires one existing forecast')
            availability = payload['outcome_available_at']
            forecast = self._forecasts[key]
            if (isinstance(availability, bool) or not isinstance(availability, int)
                    or availability < forecast['target_time_unix'][-1] + 3600):
                raise ValueError('Recorded outcome is not fully mature')
            y = np.asarray(payload['actual'], dtype=float)
            if y.shape != (24,) or not np.isfinite(y).all():
                raise ValueError('Recorded outcome must contain all 24 finite targets')
            if payload['metrics'] != self._metrics(y, np.asarray(forecast['quantiles'])):
                raise ValueError('Recorded feedback metrics do not match forecast and outcome')
        else:
            raise ValueError('Unknown feedback event kind')

    def _accept(self, event):
        self._validate_event(event)
        destination = self._forecasts if event['kind'] == 'forecast' else self._outcomes
        destination[event['payload']['forecast_id']] = deepcopy(event['payload'])
        self._events.append(deepcopy(event))

    @staticmethod
    def _metrics(y, q):
        return {'qs': quantile_loss(y, q),
                'mae': float(np.abs(y - q[:, 2]).mean()),
                'coverage_90': float(((y >= q[:, 0]) & (y <= q[:, -1])).mean()),
                'width_90': float((q[:, -1] - q[:, 0]).mean())}

    def _append(self, kind, payload):
        event = {'arm_id': self.arm_id, 'kind': kind, 'payload': payload,
                 'sequence': len(self._events),
                 'previous_hash': self._events[-1]['event_hash'] if self._events else None}
        event['event_hash'] = _digest(event)
        self._validate_event(event)
        if self.path is not None:
            import fcntl
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a+') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                # Constant-cost append guard: do not reread the full growing
                # ledger on every forecast/outcome. Full hash/schema validation
                # is performed on reload; size/mtime detects another writer.
                stat = os.fstat(stream.fileno())
                if (stat.st_size != self._disk_size or self._disk_mtime_ns is not None
                        and stat.st_mtime_ns != self._disk_mtime_ns):
                    raise RuntimeError('Stale ledger writer: reopen ledger before appending')
                stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                stat = os.fstat(stream.fileno())
                self._disk_size, self._disk_mtime_ns = stat.st_size, stat.st_mtime_ns
        self._accept(event)

    def append_forecast(self, context, artifact):
        if artifact.context_id != context.identity():
            raise ValueError('Forecast does not belong to this context')
        quantiles = np.asarray(artifact.quantiles, dtype=float)
        target = np.asarray(artifact.target_time_unix, dtype=np.int64)
        if quantiles.shape != (24, 5) or not np.isfinite(quantiles).all() or (np.diff(quantiles, axis=1) < 0).any():
            raise ValueError('Invalid recorded forecast quantiles')
        if not np.array_equal(target, context.snapshot.target_time_unix):
            raise ValueError('Forecast target mismatch')
        key = artifact.identity()
        payload = {'forecast_id': key, 'candidate_id': artifact.candidate_id,
                   'context_id': context.identity(), 'model_id': artifact.model_id,
                   'region': context.region, 'origin_unix': context.origin_unix,
                   'issued_at': context.cutoff, 'target_time_unix': target.tolist(),
                   'quantiles': quantiles.tolist(), 'input_hash': artifact.input_hash,
                   'provenance': plain(artifact.provenance)}
        with self._lock:
            if key in self._forecasts:
                if self._forecasts[key] != payload:
                    raise ValueError('Cannot overwrite immutable forecast')
                return key
            self._append('forecast', payload)
        return key

    def append_outcome(self, forecast_id, actual, available_at):
        with self._lock:
            if forecast_id not in self._forecasts:
                raise ValueError('Unknown forecast in this arm')
            forecast = self._forecasts[forecast_id]
            if isinstance(available_at, bool) or not isinstance(available_at, int) or available_at < forecast['target_time_unix'][-1] + 3600:
                raise ValueError('Outcome not fully mature at declared availability')
            y = np.asarray(actual, dtype=float)
            if y.shape != (24,) or not np.isfinite(y).all():
                raise ValueError('Only complete 24-hour outcomes are accepted')
            q = np.asarray(forecast['quantiles'])
            payload = {'forecast_id': forecast_id, 'outcome_available_at': available_at,
                       'actual': y.tolist(), 'metrics': self._metrics(y, q)}
            if forecast_id in self._outcomes:
                if self._outcomes[forecast_id] != payload:
                    raise ValueError('Cannot revise a matured outcome')
                return
            self._append('outcome', payload)

    def snapshot(self, context):
        """Read-only JSON metrics; future completion order cannot bypass watermark."""
        cutoff = min(context.cutoff, context.memory_watermark)
        rows = []
        with self._lock:
            for key, outcome in self._outcomes.items():
                forecast = self._forecasts[key]
                if forecast['region'] != context.region or forecast['issued_at'] > cutoff:
                    continue
                if outcome['outcome_available_at'] > cutoff or forecast['origin_unix'] >= context.origin_unix:
                    continue
                rows.append({k: deepcopy(forecast[k]) for k in
                             ('forecast_id', 'candidate_id', 'context_id', 'model_id', 'region', 'origin_unix')})
                rows[-1].update({'outcome_available_at': outcome['outcome_available_at'],
                                 'metrics': deepcopy(outcome['metrics']), 'arm_id': self.arm_id})
        return sorted(rows, key=lambda r: (r['origin_unix'], r['forecast_id']))

    @property
    def fingerprint(self):
        return self._events[-1]['event_hash'] if self._events else _digest({'arm_id': self.arm_id})
