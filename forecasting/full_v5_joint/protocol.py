"""Single temporal contract. No anniversary-based retrieval or final-test stage."""
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import numpy as np

TZ = timezone(timedelta(hours=10))
HOUR = 3600
DAY = 24 * HOUR


def timestamp(value):
    return int(datetime.fromisoformat(value).replace(tzinfo=TZ).timestamp())


@dataclass(frozen=True)
class Fold:
    year: int
    final_half: bool = False

    @property
    def train_end(self):
        return '2022-01-01' if self.final_half else f'{self.year-1}-07-01'

    @property
    def prediction_start(self):
        return '2022-07-01' if self.final_half else f'{self.year}-01-01'

    @property
    def prediction_end(self):
        return '2022-07-01' if self.year == 2022 and not self.final_half else f'{self.year+1}-01-01'

    def configure(self, config):
        import copy
        if self.year not in range(2017, 2023) or (self.final_half and self.year != 2022):
            raise ValueError('Development folds restricted to2017–2022')
        c=copy.deepcopy(config)
        c['data'].update(split_names=['train','selection','calibration'],relabel_splits=True,
                         load_end_exclusive=self.prediction_end,
                         split_boundaries={
                             'train':{'start':'2015-01-01','end_exclusive':self.train_end},
                             'selection':{'start':self.train_end,'end_exclusive':self.prediction_start},
                             'calibration':{'start':self.prediction_start,'end_exclusive':self.prediction_end}})
        return c


def segment_rows(origins, start, end):
    origins=np.asarray(origins)
    return np.flatnonzero((origins>=timestamp(start)) & (origins+DAY<timestamp(end)))


def assert_forecast_provenance(origins, trained_before, selected_before):
    origins=np.asarray(origins)
    if not np.all((np.asarray(trained_before)<origins) & (np.asarray(selected_before)<origins)):
        raise ValueError('Historical forecasts must precede each query in both fitting and selection')


def assert_mature(query_origins, history_origins, mask):
    q,h,m=np.asarray(query_origins),np.asarray(history_origins),np.asarray(mask,dtype=bool)
    if h.shape!=m.shape or h.shape[0]!=len(q):
        raise ValueError('Invalid candidate grid')
    if np.any((h+DAY>q[:,None]) & m):
        raise ValueError('History labels not mature at query')
