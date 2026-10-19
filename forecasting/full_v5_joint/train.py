"""One optimizer and one temporal split for every differentiable component."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import yaml

from .data import HistoryStore, digest
from .model import JointHistoryModel, training_loss, LEVELS
from .protocol import segment_rows


def score(actual, prediction):
    error = actual[..., None] - prediction
    q = np.asarray(LEVELS)
    return dict(mae=float(np.abs(error[..., 2]).mean()),
                qs=float(2 * np.maximum(q * error, (q - 1) * error).mean()))


@torch.no_grad()
def predict(model, store, rows, batch_size=16, device='cpu', **controls):
    model.eval()
    outputs = []
    for offset in range(0, len(rows), batch_size):
        batch = store.batch(rows[offset:offset+batch_size], device=device)
        outputs.append(model(batch, **controls)['prediction'].cpu().numpy())
    return np.concatenate(outputs)


def fit(store, train_rows, selection_rows, cfg, output):
    if not len(train_rows) or not len(selection_rows):
        raise ValueError('Training and selection must both be nonempty')
    if store.origins[train_rows].max() + 86400 >= store.origins[selection_rows].min():
        raise ValueError('Training labels overlap selection origins')
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    torch.set_num_threads(2)
    device = cfg['device']
    model = JointHistoryModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-4)
    best, state, log = float('inf'), None, []
    started = time.monotonic()
    for epoch in range(cfg['epochs']):
        model.train()
        order = rng.permutation(train_rows)
        total = 0.
        for offset in range(0, len(order), cfg['batch_size']):
            rows = order[offset:offset + cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            result = model(store.batch(rows, device=device))
            loss = training_loss(result, store.targets(rows, device=device),
                                 reconstruction_weight=cfg['reconstruction_weight'], gate_penalty=cfg['gate_penalty'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite joint training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
            if offset == 0 or (offset // cfg['batch_size'] + 1) % 100 == 0:
                progress = dict(stage='joint_training', epoch=epoch+1, epochs=cfg['epochs'],
                                processed_origins=min(offset+len(rows),len(order)), epoch_origins=len(order),
                                elapsed_seconds=time.monotonic()-started)
                temporary = output/'progress.json.tmp'
                temporary.write_text(json.dumps(progress, indent=2))
                temporary.replace(output/'progress.json')
        pred = predict(model, store, selection_rows, cfg['batch_size'], device)
        metrics = score(store.a['actual'][selection_rows], pred)
        log.append(dict(epoch=epoch+1, training_loss=total/len(order), selection=metrics))
        if metrics['qs'] < best:
            best = metrics['qs']
            state = copy.deepcopy(model.state_dict())
            best_epoch = epoch+1
            torch.save(dict(state_dict=state, best_epoch=best_epoch, configuration=cfg), output/'best.pt.tmp')
            (output/'best.pt.tmp').replace(output/'best.pt')
        (output/'training.json').write_text(json.dumps(log, indent=2))
        print(json.dumps(log[-1]), flush=True)
    if state is None:
        raise ValueError('At least one epoch required')
    model.load_state_dict(state)
    torch.save(dict(state_dict=state, best_epoch=best_epoch, configuration=cfg), output/'model.pt')
    return model, best_epoch


def evaluate(model, store, rows, cfg, output):
    if not len(rows):
        raise ValueError('Empty evaluation')
    result = {}
    target = store.a['actual'][rows]
    variants = {'B': None, 'Anchor_joint': {'history_off': True}, 'Full_joint': {},
                'event_off_same_head': {'event_off': True}, 'time_shift_off_same_head': {'time_shift_off': True}}
    for name, controls in variants.items():
        prediction = (store.a['baseline'][rows] if controls is None else
                      predict(model, store, rows, cfg['batch_size'], cfg['device'], **controls))
        if not np.isfinite(prediction).all() or (np.diff(prediction, axis=-1) < 0).any():
            raise ValueError('Invalid prediction')
        np.savez_compressed(output/f'{name}.npz', origin_unix=store.origins[rows],
                            target_time_unix=store.origins[rows,None]+np.arange(24)[None]*3600,
                            actual=target, quantiles=prediction, quantile_levels=LEVELS)
        result[name] = score(target, prediction)
    (output/'metrics.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/full_v5_joint/protocol.yaml')
    p.add_argument('--bundles', nargs='+', required=True, help='2017..2022H1 then2022H2 bundles in chronological order')
    p.add_argument('--region', choices=['NSW1','QLD1','TAS1'], required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    metadata = [json.loads(Path(f).with_suffix('.json').read_text()) for f in args.bundles]
    expected = [(y, False) for y in cfg['oof_years']] + [(2022, True)]
    if [(m['year'], m['final_half']) for m in metadata] != expected:
        raise ValueError('Complete ordered OOF folds required, including final deployment fold')
    if any(m['region'] != args.region or m['seed'] != cfg['seed'] for m in metadata):
        raise ValueError('Mixed region or seed')
    store = HistoryStore.from_paths(args.bundles, candidates=cfg['candidates'], history_days=cfg['history_days'])
    train_rows = segment_rows(store.origins, cfg['train_start'], cfg['train_end'])
    select_rows = segment_rows(store.origins, cfg['train_end'], cfg['selection_end'])
    eval_rows = segment_rows(store.origins, cfg['selection_end'], cfg['evaluation_end'])
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(scope='retrospective_development_not_final_test', region=args.region, config=cfg,
                    parent_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                    source_sha256={str(f): digest(f) for f in sorted(Path('forecasting/full_v5_joint').glob('*.py'))},
                    bundles={str(f): digest(f) for f in args.bundles},
                    rows=dict(train=len(train_rows), selection=len(select_rows), evaluation=len(eval_rows)),
                    availability_audit='archive_timestamp_proxies_only',
                    online_updates='mature memory and residual statistics only; all parameters frozen after selection')
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
    model, epoch = fit(store, train_rows, select_rows, cfg, root)
    metrics = evaluate(model, store, eval_rows, cfg, root)
    (root/'progress.json').write_text(json.dumps(dict(stage='complete', best_epoch=epoch, evaluated_origins=len(eval_rows)), indent=2))
    print(json.dumps(dict(best_epoch=epoch, metrics=metrics), indent=2))


if __name__ == '__main__':
    main()
