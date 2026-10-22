"""Device-resident gathers of the existing causal HistoryStore records.

This adapter changes storage only. Candidate selection and statistics are
computed by HistoryStore.record, then checked against HistoryStore.batch.
Current targets remain a separate training-only interface.
"""
import numpy as np
import torch

from ..full_v5_joint.protocol import DAY, assert_mature


class DeviceHistoryStore:
    def __init__(self, store, rows, device='cpu', progress=None):
        self.a, self.origins = store.a, store.origins
        self.device = torch.device(device)
        n = len(store.origins)
        ids = np.full((n, store.k), -1, dtype=np.int64)
        stats = np.zeros((n, 24, 8), dtype=np.float32)
        self.prepared = np.zeros(n, dtype=bool)
        for count, row in enumerate(np.unique(rows), 1):
            ids[row], stats[row] = store.record(int(row))
            self.prepared[row] = True
            if progress and count % 2000 == 0:
                progress(count)
        mask, safe = ids >= 0, np.maximum(ids, 0)
        assert_mature(store.origins, store.origins[safe], mask)
        age = np.where(mask, (store.origins[:, None] - store.origins[safe]) / DAY, 0).astype('f')
        self.data = {k: torch.as_tensor(store.a[k], device=self.device)
                     for k in ('history', 'future', 'z', 'baseline', 'lags', 'actual')}
        self.ids = torch.as_tensor(safe, device=self.device)
        self.mask = torch.as_tensor(mask, device=self.device)
        self.stats = torch.as_tensor(stats, device=self.device)
        self.age = torch.as_tensor(age, device=self.device)

    def _rows(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        if not self.prepared[rows].all():
            raise ValueError('Unprepared causal record requested')
        return torch.as_tensor(rows, device=self.device)

    def batch(self, rows):
        rows = self._rows(rows)
        ids, mask = self.ids[rows], self.mask[rows]
        result = {k: self.data[k][rows] for k in ('history', 'future', 'z', 'baseline')}
        result.update(stats=self.stats[rows], mask=mask, age=self.age[rows])
        for key, value in self.data.items():
            gathered = value[ids]
            shaped_mask = mask.reshape(*mask.shape, *([1] * (gathered.ndim - 2)))
            result['memory_' + key] = torch.where(shaped_mask, gathered, 0.)
        return result

    def targets(self, rows):
        return self.data['actual'][self._rows(rows)]

    def verify(self, reference, rows):
        actual = self.batch(rows)
        expected = reference.batch(rows)
        if actual.keys() != expected.keys() or 'actual' in actual:
            raise ValueError('Predictor interface changed')
        for key in actual:
            torch.testing.assert_close(actual[key].cpu(), expected[key], rtol=0, atol=0)
        return dict(rows=len(rows), exact=True, current_targets_in_predictor=False)
