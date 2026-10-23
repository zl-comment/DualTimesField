"""Independent frozen-selection, full-prediction, and CPU-history replay audit."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from forecasting.full_v5_anchor_balance.selection import RTOL
from forecasting.full_v5_calibration_robustness import experiment as e
from forecasting.full_v5_calibration_robustness.model import build_model


def independent_score(y,q):
    y,q=np.asarray(y,dtype=np.float64),np.asarray(q,dtype=np.float64)
    error=y[...,None]-q
    levels=np.array([.05,.1,.5,.9,.95])
    return dict(mae=float(np.abs(error[...,2]).mean()),
        qs=float(2*np.maximum(levels*error,(levels-1)*error).mean()),
        bias=float((-error[...,2]).mean()),
        coverage_80=float(((y>=q[...,1])&(y<=q[...,3])).mean()),
        coverage_90=float(((y>=q[...,0])&(y<=q[...,4])).mean()))


def independent_multiwindow(log):
    """Reconstruct the winner from all saved alpha/window scores, not flags."""
    eligible=[]
    for row in log:
        record=row['multiwindow']
        for value in record['alpha_audit']:
            groups=[dict(baseline=record['baseline'],**value['aggregate']),*value['quarters'].values()]
            safe=all(g[component][metric]<=g['baseline'][metric]+RTOL*abs(g['baseline'][metric])
                for g in groups for component in ('anchor','full') for metric in ('mae','qs'))
            improves=any(value['aggregate']['full'][metric]<record['baseline'][metric]-RTOL*abs(record['baseline'][metric])
                         for metric in ('mae','qs'))
            assert safe==value['safe'] and improves==value['improves']
            if safe and improves:
                relative=[(g['full']['qs']-g['baseline']['qs'])/max(abs(g['baseline']['qs']),1e-12)
                          for g in value['quarters'].values()]
                key=(max(relative),sum(relative)/len(relative),value['aggregate']['full']['mae'],
                     value['alpha'],row['epoch'])
                eligible.append((key,row['epoch'],value['alpha']))
    if eligible:
        _,epoch,alpha=min(eligible,key=lambda row:row[0])
        return epoch,alpha,False
    return min((r['multiwindow']['full']['mae'],r['epoch']) for r in log)[1],0.,True


def score_audit(root):
    manifest=json.loads((root/'manifest.json').read_text());cfg=manifest['config']
    progress=json.loads((root/'progress.json').read_text())
    assert progress['stage']=='complete'
    e.verify_manifest(manifest)
    assert progress['selection_sha256']==e.base.digest(root/'selection_complete.json')
    assert progress['report_sha256']==e.base.digest(root/'report.json')
    records=json.loads((root/'selection_complete.json').read_text())
    e.verify_barrier(records,cfg)
    report=json.loads((root/'report.json').read_text())
    assert set(report)==e.expected_keys(cfg)
    maxdiff=0.;sources={};count=0;old_prediction_replays=0;selection_replays=0
    for region in cfg['regions']:
        for fold in cfg['folds']:
            for seed in cfg['seeds']:
                reference=None
                for arm in e.ARMS:
                    key=f'{region}/{fold}/{seed}/{arm}'
                    record=records[key]
                    for mode in ('raw','guarded'):
                        path=e.arm_path(root,region,fold,seed,arm)/f'{arm}_{mode}.npz'
                        with np.load(path,allow_pickle=False) as arrays:
                            z={k:arrays[k] for k in arrays.files}
                        sources[str(path)]=e.base.digest(path);count+=1
                        if reference is None:reference=z
                        for name in ('origin_unix','target_time_unix','actual','baseline','quantile_levels'):
                            np.testing.assert_array_equal(z[name],reference[name])
                        np.testing.assert_array_equal(z['target_time_unix'],z['origin_unix'][:,None]+np.arange(24)[None]*3600)
                        selected=record['selection'][mode]
                        assert z['alpha']==selected['alpha'] and z['epoch']==selected['epoch']
                        for raw_name,name in (('raw_anchor','anchor'),('raw_quantiles','quantiles')):
                            alpha=selected['alpha'];b=z['baseline'];raw=z[raw_name]
                            expected=b if alpha==0 else raw if alpha==1 else (1-alpha)*b.astype(np.float64)+alpha*raw.astype(np.float64)
                            np.testing.assert_array_equal(z[name],expected)
                            assert np.isfinite(z[name]).all() and (np.diff(z[name],axis=-1)>=0).all()
                        groups=e.base.evaluation_groups(z['origin_unix'],z['baseline'])
                        for group,mask in groups.items():
                            if not mask.any():continue
                            recorded=report[key][mode]['metrics'][group]
                            assert recorded['n_origins']==int(mask.sum()) and recorded['n_cells']==int(mask.sum())*24
                            for component,name in (('baseline','baseline'),('anchor','anchor'),('full','quantiles')):
                                calculated=independent_score(z['actual'][mask],z[name][mask])
                                for metric,value in calculated.items():
                                    difference=abs(value-recorded[component][metric]);maxdiff=max(maxdiff,difference)
                                    assert difference<1e-10,(key,mode,group,metric,difference)
                        if arm=='reference':
                            old=e.arm_path(cfg['old_reference'],region,fold,seed,'asymmetric_scale')/f'asymmetric_scale_{mode}.npz'
                            with np.load(old,allow_pickle=False) as arrays:
                                for name in z:
                                    np.testing.assert_array_equal(z[name],arrays[name])
                            old_prediction_replays+=1
                        if arm=='bounded_calibration':
                            q,b=z['raw_anchor'].astype(np.float64),z['baseline'].astype(np.float64)
                            assert np.max(np.abs(q[...,2]-b[...,2]))<=30.01
                            for side in (0,1,3,4):
                                old_width=np.abs(b[...,side]-b[...,2]);new_width=np.abs(q[...,side]-q[...,2])
                                assert (new_width>=.9*old_width-.01).all() and (new_width<=1.1*old_width+.01).all()
                    if arm=='multiwindow':
                        log=json.loads((e.arm_path(root,region,fold,seed,'reference')/'training.json').read_text())
                        epoch,alpha,fallback=independent_multiwindow(log)
                        selected=record['selection']['guarded']
                        assert (epoch,alpha,fallback)==(selected['epoch'],selected['alpha'],selected['fallback'])
                        selection_replays+=1
                e.verify_reference(e.arm_path(root,region,fold,seed,'reference'),
                    e.arm_path(cfg['old_reference'],region,fold,seed,'asymmetric_scale'))
    return dict(prediction_files=count,all_group_scores_verified=True,max_abs_metric_difference=maxdiff,
        historical_predictions_bitwise_exact=old_prediction_replays,multiwindow_selections_reconstructed=selection_replays,
        bound_checks_all_origins=True,original_inputs_and_72_fit_archive_unchanged=True,
        selection_sha256=e.base.digest(root/'selection_complete.json'),
        report_sha256=e.base.digest(root/'report.json'),source_predictions=sources)


def replay_region(spec):
    root,region,device=spec;root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text());cfg=manifest['config']
    records=json.loads((root/'selection_complete.json').read_text())
    e.verify_barrier(records,cfg);e.c.configure_device(device)
    store,_=e.base.load_store(manifest['bundle_root'],region,cfg['tree_seed'])
    results={}
    for fold in cfg['folds']:
        ids=e.base.split_rows(store.origins,fold,'full')['evaluation']
        size=cfg['prediction_batch_size']
        starts=sorted({0,(len(ids)//2//size)*size,((len(ids)-1)//size)*size})
        for seed in cfg['seeds']:
            for arm,modes in (('multiwindow',('guarded',)),('bounded_calibration',('raw','guarded'))):
                key=f'{region}/{fold}/{seed}/{arm}'
                for mode in modes:
                    record=records[key]
                    model,selected=e.base.restore_checkpoint(Path(record['checkpoint_paths'][mode]),model_builder=build_model)
                    model.to(device).eval()
                    path=e.arm_path(root,region,fold,seed,arm)/f'{arm}_{mode}.npz'
                    with np.load(path,allow_pickle=False) as arrays:
                        raw_anchor,raw_full=arrays['raw_anchor'],arrays['raw_quantiles']
                    maxdiff=0.;samples=0
                    with torch.no_grad():
                        for start in starts:
                            chosen=ids[start:start+size]
                            # The independent original CPU interface replaces the device adapter.
                            batch={k:v.to(device) for k,v in store.batch(chosen).items()}
                            prediction=model(batch)
                            for name,expected in (('anchor',raw_anchor),('prediction',raw_full)):
                                actual=prediction[name].cpu().numpy()
                                reference=expected[start:start+len(chosen)]
                                maxdiff=max(maxdiff,float(np.max(np.abs(actual-reference))))
                                np.testing.assert_array_equal(actual,reference)
                            samples+=len(chosen)
                    results[f'{key}/{mode}']=dict(bitwise_exact=True,max_abs_difference=maxdiff,
                        sampled_origins=samples,batch_starts=starts,checkpoint_sha256=e.base.digest(record['checkpoint_paths'][mode]))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();root=args.run.resolve()
    audit=score_audit(root)
    if args.replay:
        cfg=json.loads((root/'manifest.json').read_text())['config']
        audit['checkpoint_replay']=e.c.run_pool(replay_region,[(str(root),r,d) for r,d in zip(cfg['regions'],cfg['devices'])])
    audit['verified_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    audit['audit_script_sha256']=e.base.digest(Path(__file__))
    e.base.write_json(root/'independent_verification.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('source_predictions','checkpoint_replay')},indent=2))


if __name__=='__main__':main()
