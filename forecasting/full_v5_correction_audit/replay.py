"""Frozen-checkpoint interventions; all new artifacts live under output/replay.

This is an inference audit, not a training or selection entry point. Internal
Anchor interventions always execute the archived JointHistoryModel.forward,
including its subtraction of the *intervened* location from trend proposals.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from ..full_v5_anchor_balance import experiment as base
from ..full_v5_anchor_balance.confirmation_experiment import configure_device
from ..full_v5_anchor_balance.selection import blend
from ..full_v5_calibration_robustness.model import BoundedCalibrationModel
from ..full_v5_joint.data import HistoryStore, digest
from ..full_v5_joint.protocol import DAY, HOUR, timestamp
from ..full_v5_mechanism_state.data import load_inputs
from ..full_v5_mechanism_state.policy import classify
from .metrics import score_and_gain, movement

ARMS = ('original', 'coverage_only', 'constant', 'state')
EXTERNAL_ARMS = ('external_final_state', 'external_matched_uniform')
OUTPUT_KEYS = ('prediction', 'anchor', 'correction', 'trend', 'event', 'gates', 'weights')


class InterventionModel(BoundedCalibrationModel):
    """Change only the calibration gate; retain the original complete forward."""

    def __init__(self, width=24):
        super().__init__(width)
        self._audit_gate = None

    def set_gate(self, gate):
        if gate is None:
            self._audit_gate = None
            return self
        value = torch.as_tensor(gate)
        if value.shape != (3,) and (value.ndim != 3 or value.shape[1:] != (24, 3)):
            raise ValueError('Gate must have shape [3] or [origin,24,3]')
        if not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
            raise ValueError('Finite gates in [0,1] required')
        self._audit_gate = value
        return self

    def calibrate_anchor(self, baseline, features):
        raw = self.calibration(features)
        # The no-intervention arithmetic has the same order as the old class.
        location = 30.0 * torch.tanh(150 * raw[..., 0].tanh() / 30.0)
        lower_delta = .1 * torch.tanh(.5 * raw[..., 1].tanh() / .1)
        upper_delta = .1 * torch.tanh(.5 * raw[..., 2].tanh() / .1)
        gate = self._audit_gate
        if gate is not None:
            gate = gate.to(device=raw.device, dtype=raw.dtype)
            if gate.ndim == 3 and gate.shape[:2] != raw.shape[:2]:
                raise ValueError('Gate origin grid differs from the current batch')
            location = gate[..., 0] * location
            lower_delta = gate[..., 1] * lower_delta
            upper_delta = gate[..., 2] * upper_delta
        lower, upper = 1 + lower_delta, 1 + upper_delta
        scale = torch.stack((lower, lower, torch.ones_like(lower), upper, upper), dim=-1)
        anchor = baseline[..., 2:3] + location[..., None] + scale * (baseline - baseline[..., 2:3])
        self.last_location = location
        self.last_scale = scale
        return anchor, location


def monthly_positions(origins, year):
    """All AEST hours on each month's first day, intersected with scored rows."""
    origins = np.asarray(origins, dtype=np.int64)
    if origins.ndim != 1 or np.any(np.diff(origins) <= 0):
        raise ValueError('Sorted unique origins required')
    wanted = np.array([timestamp(f'{int(year)}-{m:02d}-01') + h * HOUR
                       for m in range(1, 13) for h in range(24)], dtype=np.int64)
    return np.flatnonzero(np.isin(origins, wanted))


def budget_controls(baseline, original, location_gate):
    """Label-free external final-g and exact per-origin L1-budget control.

    These controls gate the *same original final delta*. They do not claim to
    match the internally recomputed state intervention, which may change sign.
    The budget sums absolute movement over 24 horizons and all five quantiles.
    """
    b, q, gate = (np.asarray(v, dtype=np.float64) for v in (baseline, original, location_gate))
    if b.ndim != 3 or b.shape[1:] != (24, 5) or q.shape != b.shape or gate.shape != b.shape[:2]:
        raise ValueError('Expected B/Q[origin,24,5] and gate[origin,24]')
    if not all(np.isfinite(v).all() for v in (b, q, gate)) or (gate < 0).any() or (gate > 1).any():
        raise ValueError('Finite predictions and gates in [0,1] required')
    if (np.diff(b, axis=-1) < 0).any() or (np.diff(q, axis=-1) < 0).any():
        raise ValueError('Ordered input quantiles required')
    delta = q - b
    original_budget = np.abs(delta).sum(axis=(1, 2))
    state_delta = gate[..., None] * delta
    target_budget = np.abs(state_delta).sum(axis=(1, 2))
    constant = np.divide(target_budget, original_budget, out=np.zeros_like(target_budget),
                         where=original_budget > 0)
    uniform_delta = constant[:, None, None] * delta
    state_q, uniform_q = b + state_delta, b + uniform_delta
    state_budget = np.abs(state_q - b).sum(axis=(1, 2))
    uniform_budget = np.abs(uniform_q - b).sum(axis=(1, 2))
    error = np.abs(uniform_budget - state_budget) / np.maximum(1., target_budget)
    if error.max(initial=0) > 1e-10:
        raise AssertionError('External final-g budget matching failed')
    if (np.diff(state_q, axis=-1) < 0).any() or (np.diff(uniform_q, axis=-1) < 0).any():
        raise AssertionError('Shared-quantile convex shrinkage crossed quantiles')
    return dict(quantiles_final_state=state_q, quantiles_matched_uniform=uniform_q,
                applied_constant=constant, original_budget=original_budget,
                state_budget=state_budget, uniform_budget=uniform_budget,
                relative_budget_error=error, feasible=np.ones(len(b), dtype=bool),
                zero_budget=original_budget == 0)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _save_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def _path(cfg, key):
    path = Path(cfg[key])
    return path if path.is_absolute() else Path(cfg['root']) / path


def _compare(actual, expected, cfg, what):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape:
        raise AssertionError(f'{what}: shape differs: {a.shape} != {b.shape}')
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    result = dict(max_abs=float(difference.max(initial=0)),
                  mean_abs=float(difference.mean()) if difference.size else 0.,
                  bit_identical=bool(np.array_equal(a, b)),
                  atol=float(cfg.get('replay_atol', 1e-6)), rtol=float(cfg.get('replay_rtol', 0)))
    if not np.allclose(a, b, atol=result['atol'], rtol=result['rtol']):
        raise AssertionError(f'{what}: archived replay failed; {result}. Tolerance was not relaxed.')
    return result


@torch.no_grad()
def _forward(model, store, rows, gate=None):
    model.set_gate(gate)
    batch = store.batch(rows, device=next(model.parameters()).device)
    result = model(batch)
    values = {key: result[key].cpu().numpy() for key in OUTPUT_KEYS}
    values.update(location=model.last_location.cpu().numpy(), scale=model.last_scale.cpu().numpy())
    return values


@torch.no_grad()
def _sampled_replay(model, store, evaluation_rows, sample_positions, original_batch_size):
    """Recreate old full-evaluation batches, then retain only requested samples.

    FP32 GEMM and solver choices can depend on batch shape. Rebatching monthly
    samples into a new 288-row sequence is not a strict archived replay.
    """
    chunks = {key: [] for key in (*OUTPUT_KEYS, 'location', 'scale')}
    replayed = 0
    for start in np.unique((sample_positions // original_batch_size) * original_batch_size):
        end = min(int(start) + original_batch_size, len(evaluation_rows))
        positions = sample_positions[(sample_positions >= start) & (sample_positions < end)]
        value = _forward(model, store, evaluation_rows[int(start):end])
        replayed += end - int(start)
        for key in chunks:
            chunks[key].append(value[key][positions - start])
    return {key: np.concatenate(value) for key, value in chunks.items()}, replayed


@torch.no_grad()
def _interventions(model, store, evaluation_rows, positions, gate_arrays, old_batch_size):
    """Use archived batch shapes for every arm as well as for original replay.

    Only the selected rows are intervened on. Other rows pad the original batch
    shape with unit gates; rows have no cross-row operations in this network.
    """
    chunks = {arm: {key: [] for key in (*OUTPUT_KEYS, 'location', 'scale')} for arm in gate_arrays}
    for start in np.unique((positions // old_batch_size) * old_batch_size):
        end = min(int(start) + old_batch_size, len(evaluation_rows))
        mask = (positions >= start) & (positions < end)
        picked = positions[mask] - start
        for arm, gates in gate_arrays.items():
            if arm == 'original':
                gate = None
            else:
                gate = np.ones((end-int(start), 24, 3), dtype=np.float32)
                gate[picked] = gates[mask]
            value = _forward(model, store, evaluation_rows[int(start):end], gate)
            for key in chunks[arm]:
                chunks[arm][key].append(value[key][picked])
    return {arm: {key: np.concatenate(v) for key, v in values.items()} for arm, values in chunks.items()}


@torch.no_grad()
def _assertions(model, store, row):
    """Actual forwards, not assertions on stale copied prediction batches."""
    device = next(model.parameters()).device
    batch = store.batch([row], device=device)
    if 'actual' in batch:
        raise AssertionError('Query label entered the predictor batch')
    model.set_gate(None)
    plain = model(batch)
    original_model = BoundedCalibrationModel().to(device).eval()
    original_model.load_state_dict(model.state_dict())
    reference = original_model(batch)
    for key in OUTPUT_KEYS:
        torch.testing.assert_close(plain[key], reference[key], rtol=0, atol=0)
    model.set_gate(torch.ones(3))
    unit = model(batch)
    for key in OUTPUT_KEYS:
        torch.testing.assert_close(unit[key], reference[key], rtol=0, atol=0)
    # An artificial zero gate also tests covered rows, not only whatever rare
    # coverage pattern happened to be included in the monthly sample.
    model.set_gate(torch.zeros(3))
    zero = model(batch)
    expected = batch['baseline']
    error = (zero['anchor'] - expected).abs()
    rounding = 4 * torch.finfo(expected.dtype).eps * (expected.abs() + expected[..., 2:3].abs() + 1)
    if (error > rounding).any():
        raise AssertionError('Zero Anchor gates did not return B within FP32 cancellation error')
    b_np, q_np = expected.cpu().numpy(), zero['prediction'].cpu().numpy()
    if not np.array_equal(blend(b_np, q_np, 0.), b_np):
        raise AssertionError('Final lambda=0 did not return the exact baseline')
    # A zero Anchor gate is explicitly NOT asserted to zero Full. Record its
    # surviving history. A deterministic nonzero-location probe additionally
    # establishes that the override is not merely passing identity outputs.
    saved_bias = model.calibration[-1].bias.detach().clone()
    model.calibration[-1].bias[0].add_(.2)
    model.set_gate(None)
    nonzero = model(batch)
    location_max = float(model.last_location.abs().max())
    if location_max <= 0:
        raise AssertionError('The nonzero-location calibration probe was ineffective')
    original_model.load_state_dict(model.state_dict())
    nonzero_reference = original_model(batch)
    for key in OUTPUT_KEYS:
        torch.testing.assert_close(nonzero[key], nonzero_reference[key], rtol=0, atol=0)
    model.calibration[-1].bias.copy_(saved_bias)
    # Perturb current and all not-yet-mature labels for ONE query. Older labels
    # that are legitimate memory for this query remain untouched.
    arrays = dict(store.a)
    arrays['actual'] = arrays['actual'].copy()
    immature = store.origins + DAY > store.origins[row]
    arrays['actual'][immature] += np.float32(12345.)
    changed = HistoryStore(arrays, candidates=store.k, history_days=store.days)
    changed_batch = changed.batch([row], device=device)
    for key in batch:
        torch.testing.assert_close(batch[key], changed_batch[key], rtol=0, atol=0)
    model.set_gate(None)
    changed_output = model(changed_batch)
    restored_output = model(batch)
    for key in OUTPUT_KEYS:
        torch.testing.assert_close(changed_output[key], restored_output[key], rtol=0, atol=0)
    if (torch.diff(zero['prediction'], dim=-1) < 0).any():
        raise AssertionError('Zero-gate output quantiles crossed')
    return dict(original_class_exact=True, unit_gate_exact=True, nonzero_location_exact=True,
                nonzero_location_probe_max=location_max, zero_anchor_max_abs=float(error.max()),
                zero_gate_full_max_abs=float((zero['prediction']-expected).abs().max()),
                zero_gate_full_is_baseline=bool(torch.equal(zero['prediction'], expected)),
                zero_gate_surviving_history_max=float(zero['correction'].abs().max()),
                final_lambda_zero_exact=True, perturbed_immature_rows=int(immature.sum()),
                current_and_immature_label_invariance_exact=True, quantiles_monotone=True)


def audit_replay(output: Path, cfg: dict) -> dict:
    """Run predeclared monthly frozen-checkpoint diagnostics; never fit anything."""
    if tuple(cfg.get('internal_interventions', ())) != ARMS:
        raise ValueError('Configured internal interventions differ from the implemented arms')
    if tuple(cfg.get('external_interventions', ())) != EXTERNAL_ARMS:
        raise ValueError('Configured external interventions differ from the implemented arms')
    destination = Path(output).resolve() / 'replay'
    destination.mkdir(parents=True, exist_ok=False)
    device = torch.device(cfg.get('device', 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'Requested {device}, but CUDA is unavailable; no silent device fallback')
    configure_device(device)
    torch.set_num_threads(int(cfg.get('cpu_threads', 2)))
    robust = _path(cfg, 'robustness_root')
    bundle_root = _path(cfg, 'bundle_root')
    diagnosis = _path(cfg, 'mechanism_diagnosis_root')
    manifest_path = robust / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    archived_batch_size = int(manifest['config']['prediction_batch_size'])
    if int(cfg.get('batch_size', archived_batch_size)) != archived_batch_size:
        raise ValueError('Strict replay must use the archived prediction batch size')
    seed = int(cfg.get('replay_seed', 2026))
    summary = dict(scope='frozen_checkpoint_interventions_not_retraining', seed=seed,
        device=str(device), torch_version=torch.__version__, archived_batch_size=archived_batch_size,
        sampling='each month first AEST day, all 24 origin hours, intersected with original evaluation rows',
        internal_arms=list(ARMS), external_controls=list(EXTERNAL_ARMS),
        budget='per-origin sum absolute final movement over 24 hours x 5 quantiles; no labels',
        internal_state_budget_matched=False, cases=[], manifest_sha256=digest(manifest_path))
    started = time.monotonic()
    for region in cfg['regions']:
        store, paths = base.load_store(bundle_root, region, 2026)
        inputs, input_audit = load_inputs(store, paths)
        fold_pieces = []
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                fold_pieces.append(np.full(len(archive['origin']), path.parent.name))
        fold_names = np.concatenate(fold_pieces)
        for fold in cfg['folds']:
            year = int(fold)
            policy_path = diagnosis / region / str(year) / 'policy.json'
            policy = json.loads(policy_path.read_text())
            split = base.split_rows(store.origins, str(year), 'full')
            if int(policy['training_last_origin']) != int(store.origins[split['train'][-1]]):
                raise ValueError('Policy training cutoff does not match the development fold')
            fit = robust / region / str(year) / f'seed_{seed}' / 'bounded_calibration'
            selection = json.loads((fit / 'selection.json').read_text())
            for mode in ('raw', 'guarded'):
                checkpoint = fit / f'{mode}.pt'
                prediction_path = fit / f'bounded_calibration_{mode}.npz'
                expected_hashes = selection['checkpoints']
                matches = [sha for p, sha in expected_hashes.items() if Path(p).name == checkpoint.name]
                if len(matches) != 1 or digest(checkpoint) != matches[0]:
                    raise ValueError('Frozen checkpoint identity mismatch')
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if saved.get('variant') != 'bounded_calibration' or int(saved.get('seed', -1)) != seed:
                    raise ValueError('Unexpected checkpoint model variant or neural seed')
                if saved['selection'] != selection['selection'][mode]:
                    raise ValueError('Frozen checkpoint selection differs from its manifest')
                model = InterventionModel().to(device).eval()
                model.load_state_dict(saved['state_dict'])
                with np.load(prediction_path, allow_pickle=False) as archive:
                    origins = archive['origin_unix']
                    if not np.array_equal(origins, store.origins[split['evaluation']]):
                        raise ValueError('Archived evaluation origins differ from the causal store')
                    positions = monthly_positions(origins, year)
                    rows = split['evaluation'][positions]
                    if not len(rows):
                        raise ValueError('No predeclared monthly replay samples')
                    actual, baseline = archive['actual'][positions], archive['baseline'][positions]
                    old_raw = archive['raw_quantiles'][positions]
                    old_anchor = archive['raw_anchor'][positions]
                    old_chosen, old_chosen_anchor = archive['quantiles'][positions], archive['anchor'][positions]
                    alpha = float(archive['alpha'])
                    if alpha != float(saved['selection']['alpha']):
                        raise ValueError('Archived alpha differs from the frozen checkpoint')
                    if int(archive['epoch']) != int(saved['selection']['epoch']):
                        raise ValueError('Archived epoch differs from the frozen checkpoint')
                if not np.array_equal(actual, store.a['actual'][rows]) or not np.array_equal(baseline, store.a['baseline'][rows]):
                    raise ValueError('Archived labels or baseline differ from the OOF store')
                states = classify(inputs[rows], policy['thresholds'])
                state_gates = np.asarray(policy['gates'], dtype=np.float32)[states]
                coverage = (states != 5)[..., None]
                gate_arrays = dict(original=np.ones((*states.shape, 3), dtype=np.float32),
                    coverage_only=np.broadcast_to(coverage, (*states.shape, 3)).astype(np.float32),
                    constant=(coverage * np.asarray(policy['constant_gates'], dtype=np.float32)).astype(np.float32),
                    state=state_gates)
                case_dir = destination / region / str(year) / mode
                case_dir.mkdir(parents=True, exist_ok=True)
                # Fail closed before writing any intervention predictions if the
                # original checkpoint cannot reproduce the saved old forecast.
                original, replayed = _sampled_replay(model, store, split['evaluation'], positions, archived_batch_size)
                audit = dict(raw_quantiles=_compare(original['prediction'], old_raw, cfg, 'raw quantiles'),
                    raw_anchor=_compare(original['anchor'], old_anchor, cfg, 'raw Anchor'),
                    chosen_quantiles=_compare(blend(baseline, original['prediction'], alpha), old_chosen, cfg, 'chosen quantiles'),
                    chosen_anchor=_compare(blend(baseline, original['anchor'], alpha), old_chosen_anchor, cfg, 'chosen Anchor'))
                assertions = _assertions(model, store, int(rows[0]))
                interventions = _interventions(model, store, split['evaluation'], positions,
                    {k: v for k, v in gate_arrays.items() if k != 'original'}, archived_batch_size)
                interventions['original'] = original
                candidates = np.stack([store.record(int(row))[0] for row in rows])
                mask = candidates >= 0
                safe = np.maximum(candidates, 0)
                candidate_origins = np.where(mask, store.origins[safe], -1)
                if np.any((candidate_origins + DAY > store.origins[rows, None]) & mask):
                    raise AssertionError('Unmatured candidate entered replay')
                common = dict(origin_unix=store.origins[rows], target_time_unix=store.origins[rows, None]+np.arange(24)*HOUR,
                    actual=actual, baseline=baseline, mechanism_inputs=inputs[rows], states=states,
                    store_row_ids=rows, evaluation_positions=positions, query_fold=fold_names[rows],
                    rolling_error_stats=np.stack([store.record(int(row))[1] for row in rows]),
                    evidence_gates=state_gates, policy_support=np.asarray(policy['support'])[states],
                    policy_coherence=np.asarray(policy['coherence'])[states],
                    policy_block_signals=np.asarray(policy['block_signals'])[states],
                    policy_day_counts=np.asarray(policy['day_counts'])[states],
                    candidate_ids=candidates, candidate_mask=mask, candidate_origins=candidate_origins,
                    candidate_age_days=np.where(mask, (store.origins[rows, None]-candidate_origins)/DAY, 0),
                    candidate_fold=np.where(mask, fold_names[safe], ''),
                    selected_before=np.where(mask, store.a['selected_before'][safe], -1),
                    trained_before=np.where(mask, store.a['trained_before'][safe], -1),
                    alpha=np.array(alpha), epoch=np.array(saved['selection']['epoch']),
                    checkpoint_sha256=np.array(matches[0]), policy_sha256=np.array(digest(policy_path)),
                    quantile_levels=np.array([.05, .1, .5, .9, .95]))
                case = dict(region=region, fold=year, mode=mode, seed=seed, n_origins=len(rows),
                    alpha=alpha, epoch=int(saved['selection']['epoch']), checkpoint=str(checkpoint),
                    checkpoint_sha256=matches[0], saved_predictions=str(prediction_path),
                    saved_predictions_sha256=digest(prediction_path), policy=str(policy_path),
                    policy_sha256=digest(policy_path), replayed_padding_origins=int(replayed),
                    replay=audit, assertions=assertions, input_audit=input_audit, arms={})
                for arm in ARMS:
                    value = interventions[arm]
                    raw_q, raw_a = value['prediction'], value['anchor']
                    q, a = blend(baseline, raw_q, alpha), blend(baseline, raw_a, alpha)
                    if (np.diff(raw_q, axis=-1) < 0).any() or (np.diff(q, axis=-1) < 0).any():
                        raise AssertionError(f'Crossed quantiles in {arm}')
                    uncovered = states == 5
                    uncovered_anchor_max = 0.
                    if arm != 'original' and uncovered.any():
                        difference = np.abs(raw_a[uncovered]-baseline[uncovered])
                        rounding = 4*np.finfo(raw_a.dtype).eps*(np.abs(baseline[uncovered])
                            + np.abs(baseline[..., 2:3][uncovered])+1)
                        if np.any(difference > rounding):
                            raise AssertionError(f'{arm}: uncovered zero gates did not return Anchor to B')
                        uncovered_anchor_max = float(difference.max())
                    path = case_dir / f'{arm}.npz'
                    _save_npz(path, **common, raw_quantiles=raw_q, quantiles=q, raw_anchor=raw_a, anchor=a,
                        raw_correction=value['correction'], correction=alpha*value['correction'],
                        pre_tanh_trend=value['trend'], pre_tanh_event=value['event'],
                        adoption_gates=value['gates'], match_weights=value['weights'],
                        location=value['location'], scale=value['scale'], constraint_gates=gate_arrays[arm])
                    case['arms'][arm] = dict(path=str(path), sha256=digest(path),
                        uncovered_cells=int(uncovered.sum()),
                        uncovered_anchor_max_abs=uncovered_anchor_max if arm != 'original' else None,
                        unshrunk=score_and_gain(actual, baseline, raw_q), chosen=score_and_gain(actual, baseline, q),
                        vs_original_unshrunk=score_and_gain(actual, original['prediction'], raw_q),
                        movement_unshrunk=movement(baseline, raw_a, raw_q), movement_chosen=movement(baseline, a, q))
                controls = budget_controls(baseline, original['prediction'], state_gates[..., 0])
                for arm, key in (('external_final_state', 'quantiles_final_state'),
                                 ('external_matched_uniform', 'quantiles_matched_uniform')):
                    raw_q = controls[key]
                    q = blend(baseline, raw_q, alpha)
                    path = case_dir / f'{arm}.npz'
                    _save_npz(path, **common, raw_quantiles=raw_q, quantiles=q,
                        applied_constant=controls['applied_constant'], original_budget=controls['original_budget'],
                        state_budget=controls['state_budget'], uniform_budget=controls['uniform_budget'],
                        relative_budget_error=controls['relative_budget_error'], zero_budget=controls['zero_budget'])
                    case['arms'][arm] = dict(path=str(path), sha256=digest(path),
                        unshrunk=score_and_gain(actual, baseline, raw_q), chosen=score_and_gain(actual, baseline, q),
                        vs_original_unshrunk=score_and_gain(actual, original['prediction'], raw_q))
                case['external_budget_max_relative_error'] = float(controls['relative_budget_error'].max())
                case['internal_state_budget_matching_claimed'] = False
                _write_json(case_dir / 'audit.json', case)
                summary['cases'].append(case)
                _write_json(destination / 'progress.json', dict(completed_cases=len(summary['cases']),
                    region=region, fold=year, mode=mode, elapsed_seconds=time.monotonic()-started))
    summary['elapsed_seconds'] = time.monotonic()-started
    summary['n_cases'] = len(summary['cases'])
    _write_json(destination / 'summary.json', summary)
    return summary
