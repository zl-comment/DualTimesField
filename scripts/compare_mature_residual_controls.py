"""Committed, CPU-only mature residual experiment with a hard L0/L1 barrier.

No runtime overrides, model fitting, GPU use, publication, or network writes.
An existing output directory is an error, so old or partial results cannot be
silently overwritten. The sole parameter is selected before H2 labels open.
"""
from __future__ import annotations

import os
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT/'configs/mature_residual_controls.json'
BASE = 'e00604107b83ce61f862901cef5c513b27ea7f42'
BRANCH = 'research/v5-mature-residual-controls-20261027'

from scripts.audit_correction_value import digest, git, load_completed, verify_frozen, write_json
from scripts.audit_event_allocation import protected_files as preceding_protected_files
from scripts.audit_upper_scale import verify_snapshots


def configuration():
    cfg = json.loads(CONFIG.read_text())
    if cfg['base_commit'] != BASE or cfg['branch'] != BRANCH or git('branch','--show-current') != BRANCH:
        raise ValueError('The new experiment must run on its dedicated branch from the prescribed base')
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=ROOT,check=True)
    if git('status','--porcelain'):
        raise ValueError('Protocol, configuration, implementation, and tests must first be committed; worktree must be clean')
    if subprocess.check_output(['git','show','HEAD:configs/mature_residual_controls.json'],cwd=ROOT) != CONFIG.read_bytes():
        raise ValueError('Runtime configuration differs from the committed version')
    if cfg['max_cpu_core_hours'] != 8 or cfg['max_wall_hours'] != 3:
        raise ValueError('The resource budget is frozen')
    cfg['root'] = str(ROOT)
    return cfg


def protected_files(cfg):
    resolved = {key:str((ROOT/value).resolve()) if key.endswith(('_root','_archive')) else value
                for key,value in cfg.items()}
    files = set(preceding_protected_files(resolved))
    archive = ROOT/cfg['event_audit_archive']
    manifest = json.loads((archive/'archive_manifest.json').read_text())
    for name, record in manifest['files'].items():
        if digest(archive/name) != record['sha256']:
            raise ValueError(f'Previous event archive changed: {name}')
    files.update(path for path in archive.rglob('*') if path.is_file())
    old = ROOT/cfg['event_audit_root']
    manifest = json.loads((old/'manifest.json').read_text())
    verify_frozen(manifest)
    verify_snapshots(old,manifest)
    for stage in ('audit','episodes'):
        load_completed(old,stage,manifest['run_id'])
    files.update(path for path in old.rglob('*') if path.is_file())
    # The formal arrays are protected by byte hashes only. Hashing does not
    # decompress labels, score them, or construct any new formal prediction.
    return sorted(files)


def source_files():
    files = {ROOT/'AGENTS.md', CONFIG,
        ROOT/'docs/mature_residual_controls_protocol_20261027.md',
        ROOT/'docs/mature_residual_controls_requested_plan_20261027.md',
        ROOT/'docs/mature_residual_controls_test_evidence_20261027.json'}
    for folder, pattern in [('forecasting','*.py'),('src/dualfield','*.py'),('scripts','*.py'),('tests','*.py')]:
        files.update((ROOT/folder).rglob(pattern))
    return sorted(files)


def freeze(output,cfg):
    evidence = json.loads((ROOT/'docs/mature_residual_controls_test_evidence_20261027.json').read_text())
    if evidence['exit_code'] != 0 or not evidence['passed']:
        raise ValueError('Committed preflight tests did not pass')
    for relative, expected in evidence['tested_sources'].items():
        if digest(ROOT/relative) != expected:
            raise ValueError(f'Tested implementation changed: {relative}')
    sources = {}
    for source in source_files():
        copy = output/'source'/source.relative_to(ROOT)
        copy.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,copy)
        sources[str(source)] = digest(source)
    inputs = {str(path):digest(path) for path in protected_files(cfg)}
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),base_commit=BASE,head=git('rev-parse','HEAD'),
        branch=BRANCH,working_tree_status=git('status','--short'),configuration=cfg,sources=sources,
        protected_inputs=inputs,experiment_type='mature residual controls; H1 beta, H2 L0, conditional L1',
        environment=dict(python=sys.version,platform=platform.platform(),hostname=platform.node(),
            app_connection='remote-ssh-discovered:ssh89',packages={name:importlib.metadata.version(name)
            for name in ('numpy','scipy','pandas','pytest')},cpu_threads=1,cuda_visible_devices=''),
        test_evidence_sha256=digest(ROOT/'docs/mature_residual_controls_test_evidence_20261027.json'))
    manifest['run_id'] = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    write_json(output/'manifest.json',manifest)
    verify_frozen(manifest)
    return manifest


def resource_snapshot(start, cpu_start):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    seconds = usage.ru_utime+usage.ru_stime+child.ru_utime+child.ru_stime-cpu_start
    return dict(wall_seconds=time.monotonic()-start,cpu_seconds=seconds,cpu_core_hours=seconds/3600,
        peak_rss_kib=usage.ru_maxrss,threads=1,gpu_seconds=0,gpu_hours=0,neural_training=0,
        lightgbm_retraining=0,new_model_inference=0,max_cpu_core_hours=8,max_wall_hours=3)


def _budget_exceeded(signum, frame):
    raise TimeoutError('Frozen CPU/wall resource budget reached; no rescue run is allowed')


def run(output,cfg):
    import numpy as np
    from forecasting.full_v5_mature_controls import rolling_residual_28d, select_shared_beta
    from forecasting.full_v5_mature_sources import load_and_verify_sources
    from forecasting.full_v5_mature_forward import evaluate_forward_stage, write_csv, independent_ids
    start = time.monotonic()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_start = usage.ru_utime+usage.ru_stime+child.ru_utime+child.ru_stime
    signal.signal(signal.SIGALRM,_budget_exceeded)
    signal.signal(signal.SIGXCPU,_budget_exceeded)
    signal.alarm(int(cfg['max_wall_hours']*3600))
    limit = int(usage.ru_utime+usage.ru_stime+cfg['max_cpu_core_hours']*3600)
    resource.setrlimit(resource.RLIMIT_CPU,(limit,limit+1))
    manifest = None
    stages = []
    owned_output = False
    try:
        output.mkdir(parents=True,exist_ok=False)
        owned_output = True
        print('[freeze] verify source, old results and formal-table byte hashes',flush=True)
        manifest = freeze(output,cfg)
        def barrier(records):
            for record in records:
                if digest(record['path']) != record['sha256']:
                    raise ValueError(f"Source changed during the run: {record['path']}")
            verify_frozen(manifest)
            verify_snapshots(output,manifest)
            usage = resource_snapshot(start,cpu_start)
            if usage['cpu_core_hours'] > 8 or usage['wall_seconds'] > 3*3600:
                raise TimeoutError('Frozen resource budget exceeded')
            return dict(source_records_verified=len(records),protected_old_files_unchanged=len(manifest['protected_inputs']),
                        source_snapshots_unchanged=len(manifest['sources']),resource=usage)
        print('[selection] load only archived <=2022H1 arrays',flush=True)
        loaded = load_and_verify_sources(cfg,stage='selection')
        selection = output/'selection'
        selection.mkdir()
        inputs = {}
        for region,data in loaded['regions'].items():
            rows = data['selection_rows']
            controls = rolling_residual_28d(data['origin'],data['actual'],data['baseline'],rows,fold_id=data['fold_id'])
            probes = np.unique(np.r_[0,len(rows)-1,np.random.default_rng(20261027).choice(len(rows),64,replace=False)])
            for position in probes:
                ids = independent_ids(data,int(rows[position]),'R28')
                np.testing.assert_array_equal(ids,controls['candidate_ids'][position,:controls['counts'][position]])
                expected = np.median(data['actual'][ids].astype(np.float64)-data['baseline'][ids,:,2].astype(np.float64),axis=0)
                np.testing.assert_array_equal(expected,controls['median'][position])
            inputs[region] = {key:data[key][rows] for key in ('origin','actual','baseline')}
            inputs[region]['median'] = controls['median']
            np.savez_compressed(selection/f'{region}_R28.npz',**inputs[region],
                **{key:value for key,value in controls.items() if key not in ('method','median')},
                library_origin=data['origin'],library_fold_id=data['fold_id'])
        choice = select_shared_beta(inputs,cfg)
        choice.update(run_id=manifest['run_id'],selection_only=True,H2_labels_opened=False,
            source_barrier=barrier(loaded['sources']))
        write_json(selection/'source_manifest.json',dict(sources=loaded['sources'],metadata=loaded['metadata']))
        write_json(selection/'beta_selection.json',choice)
        write_csv(selection/'beta_curve.csv',[dict(beta=row['beta'],eligible=row['eligible'],
            macro_mae=row['macro']['mae'],macro_qs=row['macro']['qs'],objective=row['standardized_relative_loss_sum']) for row in choice['curve']])
        choice_path = selection/'beta_selection.json'
        choice_hash = digest(choice_path)
        write_json(selection/'selection_frozen.json',dict(run_id=manifest['run_id'],beta=choice['beta'],
            sha256=choice_hash,cutoff='2022-07-01 00:00 AEST',query_purge='origin+24h<cutoff',
            frozen_before_H2_array_access=True))
        stages.append('selection')
        del loaded, inputs
        print(f"[selection] frozen shared beta={choice['beta']}; now open H2",flush=True)
        loaded = load_and_verify_sources(cfg,stage='L0')
        decision = evaluate_forward_stage(output,cfg,loaded,choice['beta'],'L0',barrier,manifest['run_id'])
        stages.append('L0')
        if digest(choice_path) != choice_hash:
            raise ValueError('Beta selection changed after H2 access')
        if not decision['passed']:
            if (output/'L1').exists():
                raise AssertionError('L1 output exists despite L0 failure')
            write_json(output/'L1_not_run.json',dict(run_id=manifest['run_id'],reason='L0_failed',
                L0_decision_sha256=digest(output/'L0/decision.json'),formal_arrays_opened=False,
                new_formal_predictions=0,no_rescue=True))
            print('[L0] FAIL: stop. L1 arrays remain unopened.',flush=True)
        else:
            del loaded
            token = dict(path=str(output/'L0/decision.json'),sha256=digest(output/'L0/decision.json'),run_id=manifest['run_id'])
            print('[L0] PASS: activate committed frozen L1',flush=True)
            loaded = load_and_verify_sources(cfg,stage='L1',l0_decision=token)
            decision = evaluate_forward_stage(output,cfg,loaded,choice['beta'],'L1',barrier,manifest['run_id'])
            stages.append('L1')
        final_barrier = barrier(loaded['sources'])
        if digest(choice_path) != choice_hash:
            raise ValueError('Frozen beta choice changed')
        write_json(output/'completion.json',dict(run_id=manifest['run_id'],completed=stages,
            disposition=decision['decision'],final_stage_passed=decision['passed'],
            protected_files_unchanged=final_barrier['protected_old_files_unchanged'],
            source_snapshots_unchanged=final_barrier['source_snapshots_unchanged'],
            main_table_modified=False,pushed=False,formal_scope='previously inspected; any L1 is retrospective'))
        artifacts = {str(path.relative_to(output)):digest(path) for path in sorted(output.rglob('*')) if path.is_file()}
    except BaseException as error:
        if owned_output:
            write_json(output/'failure.json',dict(error_type=type(error).__name__,message=str(error),
                completed=stages,stopped=True,no_rescue=True))
        raise
    finally:
        signal.alarm(0)
        if owned_output:
            write_json(output/'resources.json',resource_snapshot(start,cpu_start))
    artifacts['resources.json'] = digest(output/'resources.json')
    write_json(output/'artifact_manifest.json',dict(run_id=manifest['run_id'],files=artifacts))
    return json.loads((output/'completion.json').read_text())


def main():
    if len(sys.argv) != 1:
        raise ValueError('Runtime overrides are forbidden; run the committed configuration without arguments')
    cfg = configuration()
    output = ROOT/cfg['output_root']
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        result = run(output,cfg)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
