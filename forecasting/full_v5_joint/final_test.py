"""Frozen seed2026 evaluation on the exact original2023–2024 test origins."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from ..aligned_inputs import AlignedSplit, INPUT_KEYS
from ..aligned_scoring import (load_prediction_artifact, make_prediction_artifact,
                               save_prediction_artifact, assert_comparable, score_predictions)
from ..full_v5.data import set_stage, load_cache_split, atomic_json
from ..lightgbm.predictor import FrozenLightGBMPredictor
from .data import HistoryStore, bundle_from_aligned, digest
from .model import JointHistoryModel, LEVELS
from .protocol import timestamp

REGIONS = ('NSW1', 'QLD1', 'TAS1')
VARIANTS = {'Anchor_joint': {'history_off': True}, 'Full_joint': {},
            'event_off_same_head': {'event_off': True},
            'time_shift_off_same_head': {'time_shift_off': True}}


def combine_memory(history, test):
    if history['origin'].max() >= test['origin'].min():
        raise ValueError('Historical and test origins overlap')
    if np.any(test['origin'] < timestamp('2023-01-01')) or np.any(test['origin'] >= timestamp('2025-01-01')):
        raise ValueError('Unexpected test origin interval')
    return {k: np.concatenate((history[k], test[k])) for k in history}


def region_run(args):
    torch.set_num_threads(2)
    set_stage('final-test')
    root, development, repo = Path(args.output), Path(args.development), Path(args.repo)
    region = args.region
    output = root/region
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def progress(stage, **kw):
        atomic_json(output/'progress.json',dict(stage=stage,elapsed_seconds=time.monotonic()-started,**kw))
    progress('verify_frozen_checkpoint')
    frozen = json.loads((root/'freeze.json').read_text())
    checkpoint_path = development/'joint'/region/'model.pt'
    if digest(checkpoint_path) != frozen['checkpoints'][region]:
        raise ValueError('Checkpoint changed after test freeze')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = checkpoint['configuration']
    if cfg['seed'] != 2026:
        raise ValueError('Expected seed2026')
    log = json.loads((development/'joint'/region/'training.json').read_text())
    if checkpoint['best_epoch'] != min(log,key=lambda x:x['selection']['qs'])['epoch']:
        raise ValueError('Checkpoint does not match validation-only selection')
    model = JointHistoryModel().eval()
    model.load_state_dict(checkpoint['state_dict'])
    for p in model.parameters():
        p.requires_grad_(False)
    progress('load_canonical_test')
    arrays, meta = load_cache_split(repo/'outputs/forecasting/aligned_baselines_20261012/data_cache',region,'test')
    split = AlignedSplit({k:arrays[k] for k in INPUT_KEYS},arrays['actual'],arrays['origin_unix'],
                         arrays['target_time_unix'],meta,arrays['target'])
    reference = load_prediction_artifact(repo/f'outputs/forecasting/aligned_baselines_20261012/runs/lightgbm/seed2026/{region}/test.npz')
    for key in ('actual','origin_unix','target_time_unix'):
        if not np.array_equal(getattr(split,key),getattr(reference,key)):
            raise ValueError('Canonical test/reference mismatch: '+key)
    progress('predict_frozen_tree',origins=len(split))
    tree = FrozenLightGBMPredictor(development/'oof'/region/'2022H2/models',seed=2026,region=region,threads=4)
    q = tree.predict(split)
    test = bundle_from_aligned(split,q,timestamp('2022-01-01'),timestamp('2022-07-01'))
    progress('load_mature_historical_memory')
    paths = [development/'oof'/region/fold/'bundle.npz' for fold in ('2017','2018','2019','2020','2021','2022','2022H2')]
    history = HistoryStore.from_paths(paths,candidates=cfg['candidates'],history_days=cfg['history_days'])
    offset = len(history.origins)
    store = HistoryStore(combine_memory(history.a,test),candidates=cfg['candidates'],history_days=cfg['history_days'])
    del history, test
    provenance = dict(region=region,seed=2026,best_epoch=checkpoint['best_epoch'],checkpoint_sha256=digest(checkpoint_path),
                      tree=tree.provenance,canonical_test_metadata=meta,origins=len(split),
                      protocol='exact_original_test_samples_frozen_parameters_mature_online_memory',
                      historical_bundles={str(p):digest(p) for p in paths},
                      input_scope='canonical_seven_blocks_plus_causal_mature_history_and_residuals',
                      interpretation='Retrospective test; same-head interventions are not retrained ablations')
    atomic_json(output/'manifest.json',provenance)
    metrics={}
    def save(name,prediction):
        artifact = make_prediction_artifact(actual=split.actual,origin_unix=split.origin_unix,target_time_unix=split.target_time_unix,
                    metadata=dict(meta,seed=2026,source='full_v5_joint_final_test:'+name,output_type='quantile',prediction_space='raw_rrp',
                                  checkpoint_sha256=provenance['checkpoint_sha256'],input_scope=provenance['input_scope']),
                    quantiles=prediction,quantile_levels=LEVELS)
        assert_comparable(reference,artifact,allow_different_seed=False)
        save_prediction_artifact(output/f'{name}.npz',artifact)
        metrics[name]=score_predictions(artifact)
        atomic_json(output/'metrics.json',metrics)
    save('B_retrained',q)
    for name, controls in VARIANTS.items():
        predictions=[]
        with torch.inference_mode():
            for start in range(0,len(split),cfg['batch_size']):
                rows=np.arange(offset+start,offset+min(start+cfg['batch_size'],len(split)))
                result=model(store.batch(rows),**controls)['prediction'].cpu().numpy()
                predictions.append(result)
                if start==0 or (start//cfg['batch_size']+1)%100==0:
                    progress('predict_joint',variant=name,processed=min(start+len(rows),len(split)),origins=len(split))
        save(name,np.concatenate(predictions))
    # Re-score every completed original baseline seed on exactly these same samples.
    for run in ('aligned_baselines_20261012','aligned_baselines_20261013_extension'):
        for path in sorted((repo/'outputs/forecasting'/run/'runs').glob(f'*/seed2026/{region}/test.npz')):
            artifact=load_prediction_artifact(path)
            assert_comparable(reference,artifact,allow_different_seed=False)
            name='baseline_'+path.parents[2].name
            metrics[name]=score_predictions(artifact)
            provenance.setdefault('baseline_artifacts',{})[name]=dict(path=str(path),sha256=digest(path))
    if len([k for k in metrics if k.startswith('baseline_')]) != 12:
        raise ValueError('Missing original baseline cells')
    atomic_json(output/'metrics.json',metrics)
    atomic_json(output/'manifest.json',provenance)
    if digest(checkpoint_path)!=frozen['checkpoints'][region]:
        raise ValueError('Checkpoint modified during evaluation')
    progress('complete',origins=len(split),models=len(metrics))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True)
    p.add_argument('--development',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--region',choices=REGIONS)
    args=p.parse_args()
    if args.region:
        region_run(args)
        return
    root=Path(args.output)
    def execute(region):
        command=[sys.executable,'-u','-m','forecasting.full_v5_joint.final_test','--repo',args.repo,
                 '--development',args.development,'--output',args.output,'--region',region]
        with (root/f'{region}.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(root/f'{region}_process.json',dict(pid=child.pid,state='running'))
            code=child.wait()
            atomic_json(root/f'{region}_process.json',dict(pid=child.pid,state='complete' if code==0 else 'failed',returncode=code))
        return region,code
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=dict(pool.map(execute,REGIONS))
    if any(results.values()):
        atomic_json(root/'status.json',dict(state='failed',returncodes=results))
        raise SystemExit(1)
    scores={r:json.loads((root/r/'metrics.json').read_text()) for r in REGIONS}
    macro={m:{key:float(np.mean([scores[r][m][key] for r in REGIONS])) for key in
              ('mae','rmse_pooled','quantile_score_proxy','coverage_80','coverage_90','interval_width_80','interval_width_90')}
           for m in scores[REGIONS[0]]}
    atomic_json(root/'summary.json',dict(regions=scores,macro=macro,seed=2026,test='2023–2024'))
    lines=['# 原协议2023—2024最终评估：seed2026','',
           '固定检查点；精确复用原基线测试样本。历史模型允许更新已成熟记忆，参数不更新；回顾性评估，非盲测。', '',
           '| 模型 | NSW1 MAE | QLD1 MAE | TAS1 MAE | 宏MAE | 宏QS | 地区平均RMSE |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for m in sorted(macro,key=lambda m:macro[m]['mae']):
        vals=[scores[r][m]['mae'] for r in REGIONS]+[macro[m][k] for k in ('mae','quantile_score_proxy','rmse_pooled')]
        lines.append('| '+m+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines+=['','B_retrained为本轮重新训练的基础树；baseline_lightgbm为原总榜树，二者分别保留。',
            'Anchor_joint、event_off_same_head、time_shift_off_same_head为同一检查点的关闭干预，不是重新训练消融。',
            '基线输入为公共七块；联合模型额外使用成熟历史和误差，作为方法扩展明确标记。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    atomic_json(root/'status.json',dict(state='complete',returncodes=results,models=len(macro)))


if __name__=='__main__':main()
