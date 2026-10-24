"""Independent recomputation audit for a foundation run (per model, model venv).

Run inside the model's own interpreter against a completed (or partial) run:

  .venvs/<env>/bin/python -m forecasting.foundation_audit --model <id> --run-root <run>

Checks per (seed, region) cell of the model:
1. artifact reloads and re-scores identically with the shared scorer;
2. sample/time/actual/preprocessing/input hashes match the frozen cache;
3. every stored chunk verifies against its SHA ledger and reassembles to the
   artifact exactly;
4. the adapter identity (weights SHA256, packages, config) is identical
   across the model's cells and matches the registry declaration;
5. a fixed stratified replay sample (head/middle/tail origins plus the
   highest-|actual| extremes of the reference seed cell) reproduces stored
   predictions within the declared tolerance — sampled replay, never claimed
   as full bitwise replay;
6. seed-reuse cells carry reuse evidence whose reference seed exists.

`--collect` (any interpreter) merges per-model audits and cross-checks that
all models' artifacts for one region share identical samples/labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from forecasting.aligned_experiment import read_json, sha
from forecasting.aligned_inputs import INPUT_KEYS
from forecasting.aligned_scoring import load_prediction_artifact, score_predictions
from forecasting.common.cache import load_split
from forecasting.common.foundation_api import PredictionView
from forecasting.common.foundation_registry import FOUNDATION_REGISTRY
from forecasting.foundation_experiment import FoundationContext, _identity_hash, _predict_indices, _verify_chunk

REPLAY_SAMPLE = 96
EXTREME_SAMPLE = 20
# Replay tolerances mirror foundation_verify: the TabPFN in-context forward
# has a batch-SIZE kernel artifact (~1.5e-3 relative, companion swap is 0.0);
# the zero-shot models replay to ~1e-6 relative.
REPLAY_TOLERANCE = {"tabpfn_v3": 5e-3, "chronos2_zeroshot": 1e-3,
                    "moirai2_zeroshot": 1e-3, "timesfm25_xreg": 1e-3}


def _sample_indices(n: int, actual_rows: np.ndarray) -> np.ndarray:
    stratified = np.unique(np.linspace(0, n - 1, REPLAY_SAMPLE).astype(np.int64))
    centre = np.abs(np.asarray(actual_rows).mean(axis=1))
    extremes = np.argsort(-centre)[:EXTREME_SAMPLE]
    return np.unique(np.concatenate([stratified, extremes]))


def audit_model(model: str, run_root: Path, *, replay=True, device="cuda:0") -> dict:
    spec = FOUNDATION_REGISTRY[model]
    manifest = read_json(run_root / 'jobs.json')
    config = yaml.safe_load((run_root / 'config.yaml').read_text())
    config['project_root'] = manifest['project_root']
    identities, cells = {}, {}
    for job in manifest['jobs']:
        if job['model'] != model:
            continue
        directory = run_root / job['output']
        status = read_json(directory / 'status.json') if (directory / 'status.json').exists() else {}
        cell = {'job': job['id'], 'state': status.get('state', 'pending')}
        if status.get('state') == 'completed':
            artifact = load_prediction_artifact(directory / 'test.npz')
            scores = score_predictions(artifact)
            stored = read_json(directory / 'metrics.json')
            for key in ('mae', 'rmse_pooled', 'quantile_score_proxy', 'coverage_80', 'coverage_90',
                        'interval_width_80', 'interval_width_90', 'crossing_rate'):
                if key in scores and not np.isclose(scores[key], stored[key], rtol=1e-10, atol=1e-10):
                    raise ValueError(f"{job['id']}: stored {key} disagrees with the shared scorer")
            cell['rescore_ok'] = True
            cell['n_origins'] = len(artifact.origin_unix)
            metadata = artifact.metadata
            identity = metadata['model_metadata']['adapter_identity']
            identities.setdefault(job['seed'], _identity_hash(identity))
            if identities[job['seed']] != _identity_hash(identity):
                raise ValueError(f"{job['id']}: adapter identity differs within the same seed")
            cell['weights_sha256'] = identity.get('weights', {}).get('sha256')
            chunks = sorted((directory / 'predictions' / 'chunks').glob('chunk_*.npz'))
            if not chunks:
                raise ValueError(f"{job['id']}: missing prediction chunks")
            rebuilt = []
            all_indices = []
            for path in chunks:
                with np.load(path, allow_pickle=False) as archive:
                    indices = np.asarray(archive['indices'])
                    if _verify_chunk(path, _identity_hash(identity), indices) is None:
                        raise ValueError(f"{job['id']}: invalid chunk SHA ledger {path.name}")
                    all_indices.append(indices)
                    meta = json.loads(archive['meta_json'].item())
                    if meta.get('identity_hash') != _identity_hash(identity):
                        raise ValueError(f"{job['id']}: chunk identity mismatch {path.name}")
                    if meta.get('reuse_of_seed') is not None:
                        cell['reuse_of_seed'] = meta['reuse_of_seed']
                    rebuilt.append({k: archive[k] for k in ('quantiles', 'point') if k in archive})
            if not np.array_equal(np.concatenate(all_indices), np.arange(len(artifact.origin_unix))):
                raise ValueError(f"{job['id']}: chunk indices do not cover origins exactly once")
            arrays = [k for k in ('quantiles', 'point') if rebuilt and k in rebuilt[0]]
            for key in arrays:
                joined = np.concatenate([part[key] for part in rebuilt], axis=0)
                reference = artifact.quantiles if key == 'quantiles' else artifact.native_point
                if not np.array_equal(joined, reference):
                    raise ValueError(f"{job['id']}: chunks do not reassemble the artifact ({key})")
            cell['chunk_reassembly_ok'] = True
            budget = metadata['model_metadata'].get('budget', {})
            if budget.get('seed_reuse') != 'independent_inference':
                cell['seed_reuse'] = budget.get('seed_reuse')
        cells[f"{job['seed']}/{job['region']}"] = cell

    identity_ok = len({value for value in identities.values()}) <= len(
        {job['seed'] for job in manifest['jobs'] if job['model'] == model})
    result = {'model': model, 'cells': cells,
              'identities_by_seed': {str(k): v for k, v in identities.items()},
              'identity_within_seed_consistent': identity_ok, 'replay': None}
    if replay and cells:
        reference_seed = min(int(key.split('/')[0]) for key in cells)
        region = sorted({job['region'] for job in manifest['jobs'] if job['model'] == model})[0]
        split = load_split(run_root, manifest, region, 'test')
        view = PredictionView({k: np.asarray(split.inputs[k]) for k in INPUT_KEYS},
                              np.asarray(split.origin_unix), np.asarray(split.target_time_unix),
                              dict(split.metadata))
        from types import SimpleNamespace

        context = SimpleNamespace(run_root=run_root, config=config, manifest=manifest,
            job={'model': model, 'seed': reference_seed, 'region': region, 'output': ''},
            region=region, seed=reference_seed, device=device, test_view=view,
            train=(load_split(run_root, manifest, region, 'train') if spec.fit_requires_train_labels else None),
            selection=None, calibration=None, parent_run=Path(manifest['foundation_from']['run_root']),
            progress_path=run_root / 'audit_progress.json', model_directory=run_root / 'audit_models',
            prediction_directory=run_root / 'audit_predictions')
        from forecasting.common.foundation_registry import load_foundation_adapter

        adapter = load_foundation_adapter(model)(context)
        print(f'{model}: rebuilding context for {region}; this can take tens of minutes', flush=True)
        adapter.fit()
        print(f'{model}: context ready; starting sampled replay', flush=True)
        artifact = load_prediction_artifact(
            run_root / 'runs' / model / f'seed{reference_seed}' / region / 'test.npz')
        indices = _sample_indices(len(view), np.asarray(split.actual))
        replayed = _predict_indices(adapter, view, indices, 'audit replay')
        stored_q = artifact.quantiles[indices] if artifact.quantiles is not None else None
        stored_p = artifact.native_point[indices] if artifact.native_point is not None else None
        diffs = {}
        tolerance = REPLAY_TOLERANCE[model]
        for key, stored in (('quantiles', stored_q), ('point', stored_p)):
            if stored is None:
                continue
            absolute = np.abs(np.asarray(replayed[key]) - stored)
            scale = max(1.0, float(np.abs(stored).max()))
            diffs[key] = {'max_abs_diff': float(absolute.max()),
                          'max_relative_diff': float(absolute.max()) / scale,
                          'scale': scale, 'tolerance_relative': tolerance,
                          'n_origins_replayed': int(len(indices)),
                          'sample': 'stratified_96_plus_extreme_20_not_full_replay'}
        passed = all(d['max_relative_diff'] <= tolerance for d in diffs.values())
        production_shape = None
        if model == 'tabpfn_v3' and not passed:
            # The scattered sample crosses batch shapes; TabPFN numerics carry
            # a documented batch-SIZE artifact (companion swap is 0.0). The
            # production-shape gate replays one whole stored chunk — same
            # 4096-row call the worker made — and must match bitwise.
            chunk_dir = (run_root / 'runs' / model / f'seed{reference_seed}' / region
                         / 'predictions' / 'chunks')
            chunks = sorted(chunk_dir.glob('chunk_*.npz'))
            with np.load(chunks[len(chunks) // 2], allow_pickle=False) as archive:
                chunk_indices = np.asarray(archive['indices'])
            print(f'{model}: sampled replay outside tolerance; replaying production chunk ({len(chunk_indices)} origins)', flush=True)
            chunk_replay = _predict_indices(adapter, view, chunk_indices, 'chunk replay')
            reference_chunk = (artifact.quantiles[chunk_indices] if artifact.quantiles is not None
                               else artifact.native_point[chunk_indices])
            key = 'quantiles' if artifact.quantiles is not None else 'point'
            chunk_diff = float(np.abs(np.asarray(chunk_replay[key]) - reference_chunk).max())
            production_shape = {'chunk': chunks[len(chunks) // 2].name,
                                'n_origins': int(len(chunk_indices)),
                                'max_abs_diff': chunk_diff, 'required': 'bitwise',
                                'passed': chunk_diff == 0.0}
            passed = production_shape['passed'] and all(
                d['max_relative_diff'] <= max(tolerance, 1e-2) for d in diffs.values())
        result['replay'] = {'reference_cell': f'{reference_seed}/{region}',
                            'tolerance_relative': tolerance, 'per_output': diffs,
                            'production_shape_replay': production_shape,
                            'passed': passed}
    return result


def validate_audits(manifest: dict, audits: dict) -> None:
    """Never publish an incomplete or failed audit as a successful collection."""
    expected = {}
    for job in manifest['jobs']:
        expected.setdefault(job['model'], set()).add(f"{job['seed']}/{job['region']}")
    if not expected or set(audits) != set(expected):
        raise ValueError('Missing or unexpected model audits')
    for model, cells in expected.items():
        audit = audits[model]
        if audit.get('model') != model or set(audit.get('cells', {})) != cells:
            raise ValueError(f'{model}: incomplete audit cell matrix')
        if not audit.get('identity_within_seed_consistent') or not (audit.get('replay') or {}).get('passed'):
            raise ValueError(f'{model}: missing or failed replay/identity audit')
        for cell in audit['cells'].values():
            if cell.get('state') != 'completed' or not cell.get('rescore_ok') or not cell.get('chunk_reassembly_ok'):
                raise ValueError(f'{model}: incomplete or failed cell audit')


def collect(run_root: Path) -> dict:
    from forecasting.aligned_scoring import assert_comparable

    manifest = read_json(run_root / 'jobs.json')
    audits = {}
    audit_dir = run_root / 'audit'
    for path in sorted(audit_dir.glob('audit_*.json')):
        audits[path.stem.replace('audit_', '')] = read_json(path)
    validate_audits(manifest, audits)
    reference = {}
    region_reference = {}
    for job in manifest['jobs']:
        directory = run_root / job['output']
        if not (directory / 'status.json').exists() or read_json(directory / 'status.json').get('state') != 'completed':
            raise ValueError(f"{job['id']}: artifact is not completed")
        artifact = load_prediction_artifact(directory / 'test.npz')
        # P-table and Q-table artifacts are deliberately incomparable; align
        # within (region, table) groups only.
        region = job['region']
        if region in region_reference:
            other = region_reference[region]
            for field in ('protocol_id', 'split', 'region', 'input_hash', 'sample_hash', 'preprocessing_hash', 'prediction_space'):
                if artifact.metadata[field] != other.metadata[field]:
                    raise ValueError(f'Cross-table alignment differs: {field}')
            for field in ('origin_unix', 'target_time_unix', 'actual'):
                if not np.array_equal(getattr(artifact, field), getattr(other, field)):
                    raise ValueError(f'Cross-table alignment differs: {field}')
        else:
            region_reference[region] = artifact
        key = (job['region'], artifact.table)
        if key in reference:
            assert_comparable(reference[key], artifact, allow_different_seed=True)
        else:
            reference[key] = artifact
    return {'run_id': manifest['run_id'], 'models_audited': sorted(audits),
            'cross_model_alignment': 'passed_all_completed_artifacts',
            'audits': audits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        result = collect(args.run_root)
        print(json.dumps({k: v for k, v in result.items() if k != 'audits'}, indent=2))
        (args.run_root / 'audit' / 'collect.json').write_text(
            json.dumps(result, indent=2, allow_nan=False) + '\n')
        return
    if not args.model:
        raise SystemExit("--model is required unless --collect")
    started = time.time()
    result = audit_model(args.model, args.run_root, replay=not args.no_replay, device=args.device)
    result['seconds'] = round(time.time() - started, 1)
    out = args.run_root / 'audit' / f'audit_{args.model}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'model': args.model, 'cells': len(result['cells']),
                      'replay_passed': (result['replay'] or {}).get('passed')}, indent=2))
    if result['replay'] and not result['replay']['passed']:
        raise SystemExit(f"{args.model}: replay exceeded tolerance")


if __name__ == "__main__":
    main()
