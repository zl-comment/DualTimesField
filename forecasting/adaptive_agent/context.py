"""Canonical cache broker: historical outcomes released only after maturity."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from ..aligned_inputs import INPUT_KEYS
from ..api_agent.workflow import Snapshot
from .contracts import ForecastContext


class ContextRepository:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / 'metadata.json').read_text())
        self.origins = np.load(self.directory / 'origin_unix.npy', mmap_mode='r', allow_pickle=False)
        self.targets = np.load(self.directory / 'target_time_unix.npy', mmap_mode='r', allow_pickle=False)
        self.inputs = {k: np.load(self.directory / f'input__{k}.npy', mmap_mode='r', allow_pickle=False)
                       for k in INPUT_KEYS}
        if (np.any(np.diff(self.origins) != 3600) or
                not np.array_equal(self.targets, self.origins[:, None] + 3600 * np.arange(24))):
            raise ValueError('Expected canonical hourly origin/target grid')
        if any(len(x) != len(self.origins) for x in self.inputs.values()):
            raise ValueError('Canonical inputs are not aligned')
        self.outcome_reads = []

    def context(self, index):
        if not 0 <= index < len(self.origins):
            raise ValueError('Origin index outside cache')
        inputs = {key: np.asarray(value[index]) for key, value in self.inputs.items()}
        history = inputs['history_values'].astype(np.float64)
        snapshot = Snapshot(self.metadata['region'], int(self.origins[index]), self.targets[index],
                            history[:, 0] * self.metadata['target_std'] + self.metadata['target_mean'],
                            history[:, 1], inputs['history_calendar'], inputs['future_calendar'],
                            self.metadata['preprocessing_hash'])
        return ForecastContext(snapshot, inputs, self.metadata)

    def mature_actual(self, index, cutoff):
        if not 0 <= index < len(self.origins):
            raise ValueError('Outcome index outside cache')
        available = int(self.targets[index, -1]) + 3600
        if available > cutoff:
            raise ValueError('Requested outcomes have not fully matured')
        # No actual file is opened until the availability check passes.
        y = np.load(self.directory / 'actual.npy', mmap_mode='r', allow_pickle=False)[index].copy()
        self.outcome_reads.append({'index': int(index), 'available_at': available, 'cutoff': int(cutoff)})
        return y

    def episodes(self, context, limit=32):
        if (context.region != self.metadata['region'] or isinstance(limit, bool)
                or not isinstance(limit, int) or not 1 <= limit <= 32):
            raise ValueError('Invalid historical repository request')
        visible = min(context.cutoff, context.memory_watermark)
        latest = int(np.searchsorted(self.origins + 86400, visible, side='right')) - 1
        if latest < 0:
            return []
        # Mature-only daily grid, selected solely by timestamp (never outcomes).
        # Keep the recent half and spread the other half across older history,
        # giving seasonal/diverse retrieval support beyond the last 32 days.
        # This reservoir uses this cache's split only; it does not claim access
        # to a full training-year archive when replaying another split.
        daily = list(range(latest, -1, -24))
        recent_count = min((limit + 1) // 2, len(daily))
        indices = daily[:recent_count]
        older = daily[recent_count:]
        older_count = min(limit - len(indices), len(older))
        if older_count:
            positions = np.linspace(0, len(older) - 1, older_count, dtype=int)
            indices += [older[int(position)] for position in positions]
        return [{'episode_id': f"{context.region}:{int(self.origins[i])}",
                 'context': self.context(i), 'actual': self.mature_actual(i, visible),
                 'outcome_available_at': int(self.origins[i]) + 86400,
                 'data_role': self.metadata['split'],
                 'retrieval_scope': 'current_split_only',
                 'reservoir_policy': 'recent_half_plus_uniform_older_half_daily_v1'} for i in indices]

    def evaluation_actual(self, indices):
        """Evaluation-only entry point, never exposed to ToolHost or Kev."""
        return np.load(self.directory / 'actual.npy', mmap_mode='r', allow_pickle=False)[indices].copy()
