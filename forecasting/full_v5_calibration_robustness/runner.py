"""Correctly scope physical fits by region/fold/seed checkpoint paths.

The original training module is retained byte-for-byte for the frozen r2 run.
This orchestration layer changes neither fitting nor candidate selection.
"""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo

from . import experiment as e

PROTOCOL = Path('docs/v5_calibration_orchestration_fix_20261023.md')


def verify_barrier(records,cfg):
    if set(records)!=e.expected_keys(cfg):
        raise ValueError('All robustness fits must finish before evaluation')
    physical={r['checkpoint_paths']['raw'] for r in records.values()}
    if len(physical)!=2*len(records)//3:
        raise ValueError('Expected two physical checkpoint paths per three logical arms')
    root=Path(next(iter(physical))).parents[4]
    for key,record in records.items():
        region,fold,seed,arm=key.split('/')
        trajectory='reference' if arm in ('reference','multiwindow') else arm
        directory=e.arm_path(root,region,fold,int(seed),trajectory)
        expected=dict(raw=str(directory/'raw.pt'),
                      guarded=str(directory/('multiwindow.pt' if arm=='multiwindow' else 'guarded.pt')))
        if record['checkpoint_paths']!=expected:
            raise ValueError('Checkpoint paths do not match the logical region/fold/seed/arm')
        if record['epochs_completed']!=cfg['epochs']:
            raise ValueError('Incomplete epoch budget')
        e.c.verify_checkpoints(record)
        if not set(expected.values())<=record['checkpoints'].keys():
            raise ValueError('Selected checkpoint missing from frozen hashes')
    for region in cfg['regions']:
        for fold in cfg['folds']:
            for seed in cfg['seeds']:
                a,b=(records[f'{region}/{fold}/{seed}/{arm}'] for arm in ('reference','multiwindow'))
                if a['fit_id']!=b['fit_id'] or a['checkpoint_paths']['raw']!=b['checkpoint_paths']['raw']:
                    raise ValueError('Selection-only arm must share its reference trajectory')


def completed_records(output,cfg):
    paths=[e.arm_path(output,r,f,s,v)/'selection.json' for r in cfg['regions'] for f in cfg['folds']
           for s in cfg['seeds'] for v in e.TRAJECTORIES]
    if not all(p.exists() for p in paths):return None
    result={}
    for region in cfg['regions']:
        for fold in cfg['folds']:
            for seed in cfg['seeds']:
                for trajectory in e.TRAJECTORIES:
                    directory=e.arm_path(output,region,fold,seed,trajectory)
                    record=json.loads((directory/'selection.json').read_text())
                    expected_modes=('raw','guarded','multiwindow') if trajectory=='reference' else ('raw','guarded')
                    expected_paths={str(directory/f'{mode}.pt') for mode in expected_modes}
                    if set(record['checkpoints'])!=expected_paths or set(record['selection'])!=set(expected_modes):
                        raise ValueError('Completed physical fit has mismatched checkpoint sources')
                    e.c.verify_checkpoints(record)
                    if trajectory=='reference':
                        e.verify_reference(directory,e.arm_path(cfg['old_reference'],region,fold,seed,'asymmetric_scale'))
                    for arm in (('reference','multiwindow') if trajectory=='reference' else (trajectory,)):
                        mode='multiwindow' if arm=='multiwindow' else 'guarded'
                        result[f'{region}/{fold}/{seed}/{arm}']=dict(
                            fit_id=record['fit_id'],trajectory=trajectory,epochs_completed=record['epochs_completed'],
                            selection=dict(raw=record['selection']['raw'],guarded=record['selection'][mode]),
                            checkpoint_paths=dict(raw=str(directory/'raw.pt'),guarded=str(directory/f'{mode}.pt')),
                            checkpoints=record['checkpoints'])
    verify_barrier(result,cfg)
    return result


def freeze_orchestration(output,manifest):
    path=output/'orchestration.json'
    if path.exists():
        verify_orchestration(output)
        return
    # Freeze the new orchestration before any evaluation, preserving the original
    # training manifest and all its source hashes instead of rewriting history.
    snapshots={str(Path(__file__).resolve()):e.base.digest(Path(__file__)),
               str(PROTOCOL.resolve()):e.base.digest(PROTOCOL)}
    checkpoint_sources={str(p):e.base.digest(p) for p in output.glob('*/*/seed_*/*/*.pt')
                        if (p.parent/'selection.json').exists()}
    selection_sources={str(p):e.base.digest(p) for p in output.glob('*/*/seed_*/*/selection.json')}
    record=dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        training_manifest_sha256=e.base.digest(output/'manifest.json'),training_run_id=manifest['run_id'],
        sources=snapshots,completed_checkpoint_sources=checkpoint_sources,
        completed_selection_sources=selection_sources,
        correction='physical fits identified by region-scoped raw checkpoint paths, not unscoped fit_id',
        no_training_or_selection_rule_changes=True)
    failure=output/'failure.json'
    if failure.exists():
        record['prior_failure_sha256']=e.base.digest(failure)
        failure.rename(output/'failure_before_orchestration_fix.json')
    shutil.copyfile(Path(__file__),output/'orchestration_source.py')
    shutil.copyfile(PROTOCOL,output/'orchestration_protocol.md')
    e.base.write_json(path,record)


def verify_orchestration(output):
    record=json.loads((output/'orchestration.json').read_text())
    if record['training_manifest_sha256']!=e.base.digest(output/'manifest.json'):
        raise ValueError('Original training manifest changed')
    for kind in ('sources','completed_checkpoint_sources','completed_selection_sources'):
        if any(e.base.digest(p)!=sha for p,sha in record[kind].items()):
            raise ValueError(f'Frozen orchestration evidence changed: {kind}')


def evaluate_region(spec):
    # Spawned processes import modules afresh; install the corrected barrier in
    # each worker. The actual frozen evaluation/model/data functions are reused.
    e.verify_barrier=verify_barrier
    verify_orchestration(Path(spec[3]))
    return e.evaluate_region(spec)


def run(cfg,bundles,output,resume=False):
    manifest=e.prepare(output,cfg,bundles,resume)
    e.verify_manifest(manifest)
    freeze_orchestration(output,manifest)
    specs=[(region,cfg,str(bundles),str(output),manifest['run_id'],device)
           for region,device in zip(cfg['regions'],cfg['devices'])]
    completed=completed_records(output,cfg)
    reused=completed is not None
    if completed is None:
        e.base.write_json(output/'progress.json',dict(stage='training_selection',
            physical_fits=2*len(e.expected_keys(cfg))//3,logical_fits=len(e.expected_keys(cfg))))
        completed=e.c.run_pool(e.train_region,specs)
    e.verify_manifest(manifest);verify_barrier(completed,cfg);verify_orchestration(output)
    barrier=output/'selection_complete.json'
    if barrier.exists() and json.loads(barrier.read_text())!=completed:
        raise ValueError('Frozen selection barrier changed')
    e.base.write_json(barrier,completed)
    count=len({r['checkpoint_paths']['raw'] for r in completed.values()})
    e.base.write_json(output/'progress.json',dict(stage='evaluation',physical_fits=count,
        logical_fits=len(completed),reused_completed_training=reused,selection_sha256=e.base.digest(barrier)))
    results=e.c.run_pool(evaluate_region,specs)
    e.verify_manifest(manifest);verify_barrier(completed,cfg);verify_orchestration(output)
    if set(results)!=e.expected_keys(cfg):raise ValueError('Incomplete evaluation matrix')
    e.base.write_json(output/'report.json',results)
    from .report import write_report
    write_report(output,results,cfg)
    e.base.write_json(output/'progress.json',dict(stage='complete',physical_fits=count,logical_fits=len(completed),
        reused_completed_training=reused,selection_sha256=e.base.digest(barrier),
        report_sha256=e.base.digest(output/'report.json'),orchestration_sha256=e.base.digest(output/'orchestration.json')))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--bundle-root',default=e.base.BUNDLE_ROOT)
    parser.add_argument('--devices',nargs=3,default=['cuda:3','cuda:6','cuda:7'])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--development',action='store_true')
    args=parser.parse_args(argv)
    if not args.development:raise ValueError('--development required')
    output,bundles=Path(args.output).resolve(),Path(args.bundle_root).resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with (output.parent/(output.name+'.lock')).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This robustness output already has an active runner') from None
        if output.exists() and not args.resume:raise FileExistsError('Output exists; use --resume')
        try:run(e.configuration(args.devices),bundles,output,args.resume)
        except Exception as error:
            if output.exists():e.base.write_json(output/'failure.json',dict(type=type(error).__name__,message=str(error)))
            raise


if __name__=='__main__':main()
