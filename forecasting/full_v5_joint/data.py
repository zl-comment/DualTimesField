"""OOF bundles, causal retrieval and label-separated prediction batches."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from .protocol import DAY, HOUR, assert_forecast_provenance, assert_mature
from ..full_v5.data import build_z


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def bundle_from_aligned(split, quantiles, trained_before, selected_before):
    q=np.sort(np.asarray(quantiles,dtype=np.float64),axis=-1)
    if q.shape!=(len(split),24,5):
        raise ValueError('Expected raw-price five quantiles')
    assert_forecast_provenance(split.origin_unix,trained_before,selected_before)
    inp=split.inputs
    z=build_z(q,{**inp,'target_time_unix':split.target_time_unix})
    history=np.concatenate((inp['history_values'],inp['history_calendar']),axis=-1)
    future=np.concatenate((inp['future_calendar'],inp['future_exogenous'],inp['ctf_exogenous'],
                           inp['quantile_exogenous'],np.broadcast_to(inp['origin_context'][:,None,:],(len(split),24,1)),q/300),axis=-1)
    raw_price=inp['history_values'][:,:,0]*split.metadata['target_std']+split.metadata['target_mean']
    lags=np.stack((raw_price[:,:24],raw_price[:,24:48],raw_price[:,48:]),axis=-1)
    return dict(origin=split.origin_unix.astype(np.int64),actual=split.actual.astype(np.float32),
                baseline=q.astype(np.float32),z=z.astype(np.float32),history=history.astype(np.float32),
                future=future.astype(np.float32),lags=lags.astype(np.float32),
                trained_before=np.full(len(split),trained_before,dtype=np.int64),
                selected_before=np.full(len(split),selected_before,dtype=np.int64))


class HistoryStore:
    """No current targets in batch(); training labels returned separately."""
    def __init__(self, arrays, *, candidates=16, history_days=730):
        self.a={k:np.asarray(v) for k,v in arrays.items()}
        self.k=int(candidates); self.days=int(history_days)
        if self.k<1 or self.days<28:
            raise ValueError('Positive candidates and at least28days')
        a=self.a; self.origins=a['origin']
        if self.origins.ndim!=1 or not len(self.origins) or np.any(np.diff(self.origins)<=0):
            raise ValueError('Unique chronological origins required')
        n=len(self.origins)
        expected={'actual':(n,24),'baseline':(n,24,5),'z':(n,24,16),
                  'history':(n,72,12),'future':(n,24,31),'lags':(n,24,3),
                  'trained_before':(n,),'selected_before':(n,)}
        for key,shape in expected.items():
            if key not in a or a[key].shape!=shape or not np.isfinite(a[key]).all():
                raise ValueError(f'Invalid {key}: expected{shape}')
        if np.any(np.diff(a['baseline'],axis=-1)<0):
            raise ValueError('Crossed baseline quantiles')
        assert_forecast_provenance(self.origins,a['trained_before'],a['selected_before'])
        self._cache={}

    @classmethod
    def from_paths(cls, paths, **kwargs):
        blocks=[]
        identity=None
        if not paths:
            raise ValueError("No OOF bundles")
        for path in paths:
            path=Path(path)
            metadata=json.loads(path.with_suffix('.json').read_text())
            if metadata.get('array_sha256')!=digest(path):
                raise ValueError(f'Bundle hash mismatch: {path}')
            if metadata.get('scope')!='chronological_oof':
                raise ValueError('Formal joint training only accepts chronological OOF bundles')
            current=(metadata.get('region'), metadata.get('seed'))
            if None in current or (identity is not None and current!=identity):
                raise ValueError('Missing or mixed region/seed provenance')
            identity=current
            if metadata.get('configuration_sha256')!=digest(path.parent/'config.json'):
                raise ValueError('Fold configuration hash mismatch')
            model_files=metadata.get('model_files', {})
            if not model_files or any(digest(path.parent/f)!=sha for f,sha in model_files.items()):
                raise ValueError('Missing or changed source tree models')
            with np.load(path,allow_pickle=False) as archive:
                block={k:archive[k] for k in archive.files}
            if not np.all(block['trained_before']==metadata['trained_through']) or not np.all(block['selected_before']==metadata['selected_through']):
                raise ValueError('Forecast cutoff manifest mismatch')
            blocks.append(block)
        arrays={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
        return cls(arrays,**kwargs)

    def record(self, i):
        if i in self._cache:return self._cache[i]
        a=self.a; t=self.origins[i]
        hi=np.searchsorted(self.origins,t-DAY,side='right')
        lo=np.searchsorted(self.origins,t-self.days*DAY)
        pool=np.arange(lo,hi)
        pool=pool[self.origins[pool]%DAY==t%DAY]
        # Deterministic prefilter includes recent cases plus state-near cases;
        # no query labels, anniversaries, or 28day exponential suppression.
        ids=np.full(self.k,-1,dtype=np.int64)
        if len(pool):
            dims=[1,2,3,6,7,8]
            dist=((a['z'][pool][:,:,dims]-a['z'][i][:,dims])**2).mean(axis=(1,2))
            chosen=list(pool[-min(len(pool),(self.k+1)//2):])
            for p in pool[np.argsort(dist,kind='stable')]:
                if int(p) not in chosen:chosen.append(int(p))
                if len(chosen)>=self.k:break
            ids[:len(chosen)]=chosen
        stats=np.zeros((24,8),np.float32)
        for wi,days in enumerate((7,28)):
            left=np.searchsorted(self.origins,t-days*DAY)
            rows=np.arange(left,hi)
            stats[:,6+wi]=min(len(rows)/(days*24),1.)
            if not len(rows):continue
            residual=a['actual'][rows]-a['baseline'][rows,:,2]
            stats[:,wi]=np.median(residual,axis=0)/300
            stats[:,2+wi]=np.median(np.abs(residual),axis=0)/300
            if days==28:
                for col,l,u,nominal in ((4,1,3,.8),(5,0,4,.9)):
                    stats[:,col]=((a['actual'][rows]>=a['baseline'][rows,:,l]) & (a['actual'][rows]<=a['baseline'][rows,:,u])).mean(axis=0)-nominal
        self._cache[i]=(ids,np.clip(stats,-1,1))
        return self._cache[i]

    def batch(self, rows, *, device='cpu'):
        rows=np.asarray(rows,dtype=np.int64)
        records=[self.record(int(i)) for i in rows]
        ids=np.stack([r[0] for r in records]);mask=ids>=0;safe=np.maximum(ids,0)
        origins=self.origins[safe]
        assert_mature(self.origins[rows],origins,mask)
        result={k:self.a[k][rows] for k in ('history','future','z','baseline')}
        result['stats']=np.stack([r[1] for r in records])
        for key in ('history','future','z','baseline','lags','actual'):
            value=self.a[key][safe].copy()
            value[~mask]=0
            result['memory_'+key]=value
        result['mask']=mask
        result['age']=np.where(mask,(self.origins[rows,None]-origins)/DAY,0).astype(np.float32)
        # Query actual is deliberately absent from the predictor interface.
        return {k:torch.as_tensor(v,device=device) for k,v in result.items()}

    def targets(self, rows, *, device='cpu'):
        return torch.as_tensor(self.a['actual'][rows],device=device)
