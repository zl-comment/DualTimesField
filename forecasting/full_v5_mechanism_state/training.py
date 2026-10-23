"""Original fixed-budget optimizer/selection, with a frozen mechanism policy.

Copied from the archived confirmation fit to avoid mutating frozen provenance.
"""
import hashlib
import json
import time
import numpy as np
import torch
from ..full_v5_anchor_balance import experiment as base
from ..full_v5_anchor_balance.confirmation_experiment import (fingerprint, cpu_state, save_torch, verify_checkpoints, parameter_counts)
from ..full_v5_anchor_balance.objective import balance_loss
from ..full_v5_anchor_balance.selection import select_candidate, selection_key
from .model import build_model


def fit(store, rows, variant, seed, cfg, output, run_id, policy):
    """A last.pt is the epoch commit. Partial subsequent epochs are replayed.

    Selected model states live in that same atomic commit, avoiding a mismatch
    between raw/guarded checkpoints and optimizer state after interruption.
    Evaluation labels are never read here.
    """
    output.mkdir(parents=True, exist_ok=True)
    identity = fingerprint(dict(run=run_id, path=str(output.resolve()), policy=fingerprint(policy), variant=variant, seed=seed,
                                train=rows['train'].tolist(), selection=rows['selection'].tolist()))
    complete = output / 'selection.json'
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved['fit_id'] != identity:
            raise ValueError('Completed fit identity changed')
        verify_checkpoints(saved)
        return saved
    model = build_model(variant, seed).to(store.device)
    if variant != 'bounded_reference':
        model.set_policy(policy)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    rng = np.random.default_rng(seed)
    best_raw = best_guard = None
    best_states, log, start = {}, [], 1
    resume_path = output / 'last.pt'
    if resume_path.exists():
        saved = torch.load(resume_path, map_location='cpu', weights_only=True)
        if saved['fit_id'] != identity:
            raise ValueError('Resume fit identity changed')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        if store.device.type == 'cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'], store.device)
        best_raw, best_guard = saved['raw'], saved['guarded']
        best_states, log, start = saved['best_states'], saved['log'], saved['epoch'] + 1
    ids = rows['selection']
    actual, baseline = (store.a[k][ids] for k in ('actual', 'baseline'))
    started = time.monotonic()
    for epoch in range(start, cfg['epochs'] + 1):
        model.train()
        order, total, terms_total = rng.permutation(rows['train']), 0., {}
        epoch_started = time.monotonic()
        for offset in range(0, len(order), cfg['batch_size']):
            batch_ids = order[offset:offset + cfg['batch_size']]
            batch = store.batch(batch_ids)
            optimizer.zero_grad(set_to_none=True)
            result = model(batch)
            loss, terms = balance_loss(result, store.targets(batch_ids), batch['baseline'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite confirmation objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(batch_ids)
            for name, value in terms.items():
                terms_total[name] = terms_total.get(name, 0.) + float(value.detach()) * len(batch_ids)
            if offset == 0 or (offset // cfg['batch_size'] + 1) % 50 == 0:
                base.write_json(output / 'progress.json', dict(stage='training_selection', epoch=epoch,
                    epochs=cfg['epochs'], processed=offset + len(batch_ids), count=len(order),
                    elapsed_seconds=time.monotonic() - started))
        prediction = base.predict(model, store, ids, cfg['prediction_batch_size'])
        raw = dict(epoch=epoch, alpha=1., fallback=False, **base.diagnostics(actual, baseline, prediction))
        guarded = select_candidate(actual, baseline, prediction['anchor'], prediction['full'],
                                   alphas=cfg['alphas'], epoch=epoch)
        if best_raw is None or raw['full']['qs'] < best_raw['full']['qs']:
            best_raw, best_states['raw'] = raw, cpu_state(model)
        if best_guard is None or selection_key(guarded) < selection_key(best_guard):
            best_guard, best_states['guarded'] = guarded, cpu_state(model)
        log.append(dict(epoch=epoch, loss=total / len(order),
            terms={k: v / len(order) for k, v in terms_total.items()}, raw=raw, guarded=guarded,
            order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
            elapsed_seconds=time.monotonic() - epoch_started))
        save_torch(resume_path, dict(fit_id=identity, model=cpu_state(model), optimizer=optimizer.state_dict(),
            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(store.device) if store.device.type == 'cuda' else None,
            epoch=epoch, raw=best_raw, guarded=best_guard, best_states=best_states, log=log))
        base.write_json(output / 'training.json', log)
        print(f'{output}: epoch={epoch}/{cfg["epochs"]} loss={log[-1]["loss"]:.6f} '
              f'selection_QS={raw["full"]["qs"]:.6f} alpha={guarded["alpha"]}', flush=True)
    # Recover training.json too if interruption followed last.pt's atomic rename.
    base.write_json(output / 'training.json', log)
    for mode, selection in (('raw', best_raw), ('guarded', best_guard)):
        save_torch(output / f'{mode}.pt', dict(state_dict=best_states[mode], variant=variant,
                                              seed=seed, selection=selection, policy_sha256=fingerprint(policy)))
    record = dict(fit_id=identity, selection=dict(raw=best_raw, guarded=best_guard),
                  parameters=parameter_counts(model), epochs_completed=len(log),
                  checkpoints={str(output / f'{mode}.pt'): base.digest(output / f'{mode}.pt')
                               for mode in ('raw', 'guarded')})
    base.write_json(complete, record)
    base.write_json(output / 'progress.json', dict(stage='selection_complete', epochs_completed=len(log)))
    return record
