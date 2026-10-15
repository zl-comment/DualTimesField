"""Deterministic, resumable prefix fits and label-free front-end predictions."""
import copy
import random
import time
from pathlib import Path
import numpy as np
import torch
from forecasting.aligned_experiment import atomic_json, sha
from .frontend import Frontend, forecast_loss
from .schema import OUTPUT_KEYS


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model(path, config, mode, device):
    model = Frontend(mode, config['width'], config['dropout'], config['memory_dropout']).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval()


@torch.no_grad()
def infer(model, bank, rows, batch_size=128, *, force_null=False, audit_weights=False):
    model.eval()
    collected = {k: [] for k in OUTPUT_KEYS}
    weights = []
    for start in range(0, len(rows), batch_size):
        ix = rows[start:start + batch_size]
        result = model(bank.batch(ix, with_memory=model.mode != 'current' and not force_null), force_null=force_null)
        for key in OUTPUT_KEYS:
            value = result[key].cpu().numpy().astype(np.float64)
            if key not in ('gC', 'gD'):
                value *= bank.sigma
                if key in ('P', 'P0', 'C0'):
                    value += bank.levels[ix, None]
            collected[key].append(value)
        if audit_weights:
            weights.append(result['weights'].cpu().numpy())
    output = {k: np.concatenate(v) for k, v in collected.items()}
    if audit_weights:
        output['weights'] = np.concatenate(weights)
    if not all(np.isfinite(v).all() for v in output.values()):
        raise FloatingPointError('Nonfinite front-end prediction')
    return output


def train_frontend(bank, train_rows, selection_rows, config, mode, directory, *, seed=2026, fixed_epochs=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    torch.set_num_threads(config['torch_threads'])
    model = Frontend(mode, config['width'], config['dropout'], config['memory_dropout']).to(bank.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    rng = np.random.default_rng(seed)
    batch_size = config['batch_size']
    started = time.monotonic()
    # Prefix-local pretraining; not a pretraining pass over the full data archive.
    for epoch in range(config['pretrain_epochs']):
        model.train()
        losses = []
        for rows in np.array_split(rng.permutation(train_rows), max(1, int(np.ceil(len(train_rows) / batch_size)))):
            ix = torch.as_tensor(rows, device=bank.device)
            c, d, amp = model.reconstruct(bank.flat[ix], bank.curves[ix], masked=True)
            loss = (c + d - bank.curves[ix]).abs().mean() + .01 * c.diff(n=2, dim=-1).square().mean() + .001 * amp.abs().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); losses.append(float(loss.detach()))
        atomic_json(directory / 'progress.json', dict(stage='pretrain', epoch=epoch + 1,
            epochs=config['pretrain_epochs'], loss=float(np.mean(losses)), seconds=time.monotonic() - started))
    # Reset optimizer so joint fitting has the same schedule after every pretraining stage.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    best, best_epoch, stale, trace = float('inf'), 0, 0, []
    for epoch in range(fixed_epochs if fixed_epochs is not None else config['epochs']):
        model.train()
        losses = []
        for rows in np.array_split(rng.permutation(train_rows), max(1, int(np.ceil(len(train_rows) / batch_size)))):
            result = model(bank.batch(rows, with_memory=mode != 'current'))
            loss = forecast_loss(result, bank.curves[torch.as_tensor(rows, device=bank.device)])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training objective')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); losses.append(float(loss.detach()))
        if fixed_epochs is None:
            prediction = infer(model, bank, selection_rows, batch_size)['P']
            truth = bank.curves[torch.as_tensor(selection_rows, device=bank.device)].cpu().numpy().astype(np.float64) * bank.sigma + bank.levels[selection_rows, None]
            score = float(np.abs(prediction - truth).mean())
        else:
            score = None
        trace.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), selection_mae=score))
        if fixed_epochs is not None or score < best:
            best, best_epoch, stale = score if score is not None else best, epoch + 1, 0
            torch.save(model.state_dict(), directory / 'model.pt.tmp')
            (directory / 'model.pt.tmp').replace(directory / 'model.pt')
        else:
            stale += 1
        atomic_json(directory / 'progress.json', dict(stage='joint', epoch=epoch + 1, selected_epoch=best_epoch,
            selection_mae=score, seconds=time.monotonic() - started, trace=trace))
        if fixed_epochs is None and stale >= config['patience']:
            break
    result = dict(mode=mode, seed=seed, selected_epoch=best_epoch, selected_mae=None if fixed_epochs is not None else best,
                  fixed_epochs=fixed_epochs, train_rows=len(train_rows), selection_rows=len(selection_rows),
                  seconds=time.monotonic() - started, trace=trace, model_sha256=sha(directory / 'model.pt'))
    atomic_json(directory / 'fit.json', result)
    return load_model(directory / 'model.pt', config, mode, bank.device), result
