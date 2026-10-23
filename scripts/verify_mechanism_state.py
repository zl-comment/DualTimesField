"""Recompute all saved scores and replay the short integration matrix."""
import argparse
import json
from pathlib import Path
import numpy as np
from forecasting.full_v5_mechanism_state import runner as r
from forecasting.full_v5_mechanism_state.policy import fit_policy, classify


def verify(output, replay=False):
    output=Path(output).resolve()
    manifest=json.loads((output/'manifest.json').read_text());cfg=manifest['config']
    r.verify_manifest(manifest)
    records=json.loads((output/'selection_complete.json').read_text())
    r.verify_barrier(records,cfg,output)
    reports=json.loads((output/'report.json').read_text())
    count=0;max_score=0.;max_replay=0.;replayed=0
    for region,device in zip(cfg['regions'],cfg['devices']):
        r.c.configure_device(device)
        store,paths=r.base.load_store(manifest['bundle_root'],region,cfg['tree_seed'])
        inputs,_=r.load_inputs(store,paths)
        splits={f:r.split_rows(store,f,cfg) for f in cfg['folds']}
        adapted=None
        if replay:
            adapted,_=r.adapted_store(store,inputs,splits,('evaluation',),device)
        for fold,rows in splits.items():
            policy=json.loads((output/region/fold/'policy.json').read_text())
            train=rows['train']
            expected=fit_policy(inputs[train],store.a['actual'][train],store.a['baseline'][train],store.origins[train])
            if expected!=policy:raise ValueError('Training-only policy does not reproduce')
            ids=rows['evaluation'];states=classify(inputs[ids],policy['thresholds'])
            for seed in cfg['seeds']:
                for arm in cfg['variants']:
                    path=r.arm_path(output,region,fold,seed,arm)
                    key=f'{region}/{fold}/{seed}/{arm}'
                    for mode in ('raw','guarded'):
                        with np.load(path/f'{mode}.npz',allow_pickle=False) as z:
                            if not np.array_equal(z['origin_unix'],store.origins[ids]) or not np.array_equal(z['actual'],store.a['actual'][ids]) or not np.array_equal(z['baseline'],store.a['baseline'][ids]):
                                raise ValueError('Prediction grid/label/baseline mismatch')
                            if not np.array_equal(z['target_time_unix'],store.origins[ids,None]+np.arange(24)[None]*3600):
                                raise ValueError('Target time grid mismatch')
                            if not np.array_equal(z['mechanism_state'],states):raise ValueError('State identity changed')
                            selected=records[key]['selection'][mode]
                            if float(z['alpha'])!=selected['alpha'] or int(z['epoch'])!=selected['epoch']:
                                raise ValueError('Checkpoint/alpha selection changed')
                            for raw,name in [('raw_anchor','anchor'),('raw_quantiles','quantiles')]:
                                np.testing.assert_array_equal(r.blend(z['baseline'],z[raw],selected['alpha']),z[name])
                            if arm=='bounded_reference':gate=np.ones((*states.shape,3))
                            elif arm=='state_evidence':gate=np.asarray(policy['gates'])[states]
                            else:gate=np.where((states==5)[...,None],0,np.asarray(policy['constant_gates']))
                            b,a=z['baseline'].astype(float),z['raw_anchor'].astype(float)
                            delta=a[...,2]-b[...,2]
                            tol=2e-5+np.max(np.abs(b),axis=-1)*5e-7
                            if np.any(np.abs(delta)>30*gate[...,0]+tol):raise ValueError('Location bound violated')
                            for cols,g in [([0,1],1),([3,4],2)]:
                                old=b[...,cols]-b[...,2,None]
                                new=a[...,cols]-a[...,2,None]
                                if np.any(np.abs(new-old)>.1*gate[...,g,None]*np.abs(old)+tol[...,None]):
                                    raise ValueError('Tail bound violated')
                            if np.any(np.diff(a,axis=-1)<-1e-5):raise ValueError('Crossed quantiles')
                            groups=r.base.evaluation_groups(z['origin_unix'],z['baseline'])
                            for group,mask in groups.items():
                                if not mask.any():continue
                                for component,field in [('baseline','baseline'),('anchor','anchor'),('full','quantiles')]:
                                    actual=r.c.score(z['actual'][mask],z[field][mask])
                                    saved=reports[key][mode]['metrics'][group][component]
                                    for metric,value in actual.items():max_score=max(max_score,abs(value-saved[metric]))
                            if replay:
                                model,_=r.base.restore_checkpoint(path/f'{mode}.pt',model_builder=r.build_model)
                                model.to(device)
                                # Full integration smoke; full matrix uses the first original batch.
                                n=len(ids) if cfg['scope']=='integration_smoke_not_evidence' else min(len(ids),cfg['prediction_batch_size'])
                                prediction=r.base.predict(model,adapted,ids[:n],cfg['prediction_batch_size'])
                                for name,field in [('anchor','raw_anchor'),('full','raw_quantiles')]:
                                    diff=float(np.max(np.abs(prediction[name]-z[field][:n])))
                                    max_replay=max(max_replay,diff)
                                replayed+=1
                            count+=1
    if max_score>1e-10 or max_replay>1e-6:raise ValueError('Score/checkpoint replay mismatch')
    r.verify_manifest(manifest);r.verify_barrier(records,cfg,output)
    result=dict(prediction_files=count,max_score_difference=max_score,replayed_checkpoints=replayed,
        max_replay_difference=max_replay,all_policies_refit_from_training_only=True,
        scope=cfg['scope'],scientific_effectiveness_verified=False)
    r.base.write_json(output/'verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output');parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify(args.output,args.replay),indent=2))
