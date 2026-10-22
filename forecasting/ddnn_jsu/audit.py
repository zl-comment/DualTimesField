"""Independently replay all repaired JSU checkpoints and verify their true tails."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import johnsonsu
import torch
import yaml

from ..aligned_experiment import read_json, sha, verify_snapshot
from ..aligned_report import build_aligned_report
from ..aligned_scoring import QUANTILE_LEVELS, assert_comparable, load_prediction_artifact
from ..common.cache import load_split
from ..common.native_trainer import block_tensors
from .model import DDNNJSU, _flatten, jsu_quantiles


def audit(original, repaired, *, device='cpu'):
    original, repaired = Path(original).resolve(), Path(repaired).resolve()
    old_manifest, manifest = (read_json(root / 'jobs.json') for root in (original, repaired))
    verify_snapshot(original, old_manifest)
    verify_snapshot(repaired, manifest)
    if sha(Path(__file__).with_name('model.py')) != manifest['source_files']['forecasting/ddnn_jsu/model.py']:
        raise ValueError('Auditor model code differs from the repaired frozen source')
    report = build_aligned_report(repaired)
    if not report['comparison_available'] or report['progress']['completed_cells'] != 9:
        raise ValueError('Nine complete repaired cells required before auditing results')
    old_config, config = (yaml.safe_load((root / 'config.yaml').read_text()) for root in (original, repaired))
    if old_config['training'] != config['training']:
        raise ValueError('Training configuration changed beyond the numerical repair')
    if old_config['model_baselines']['ddnn_jsu'] != config['model_baselines']['ddnn_jsu']:
        raise ValueError('Network hyperparameters changed')
    torch.set_num_threads(config['experiment']['cpu_threads_per_job'])
    device = torch.device(device)
    levels = torch.tensor(QUANTILE_LEVELS, dtype=torch.float64, device=device)
    rows = []
    for region in manifest['regions']:
        splits = {name: load_split(repaired, manifest, region, name) for name in ('selection', 'test')}
        for seed in manifest['seeds']:
            relative = Path('runs/ddnn_jsu') / f'seed{seed}' / region
            directory, old_directory = repaired / relative, original / relative
            saved = torch.load(directory / 'best_model.pt', map_location='cpu', weights_only=True)
            result = load_prediction_artifact(directory / 'test.npz')
            old_result = load_prediction_artifact(old_directory / 'test.npz')
            assert_comparable(old_result, result)
            if result.metadata['model_metadata']['inverse_cdf'] != dict(
                    precision='float64', argument_clipping=False, prediction_clipping=False, nonfinite_policy='raise'):
                raise ValueError('Missing unbounded inverse-CDF declaration')
            width = sum(int(np.prod(a.shape[1:])) for a in splits['test'].inputs.values())
            model = DDNNJSU(width, result.actual.shape[1], saved['metadata']['params']).to(device)
            model.load_state_dict(saved['model_state'])
            model.eval()
            history = result.metadata['model_metadata']['training_history']
            old_history = old_result.metadata['model_metadata']['training_history']
            if history['learning_rate'] != old_history['learning_rate']:
                raise ValueError('Effective learning rate changed')
            if history['best_epoch'] != int(np.argmin(history['selection_curve'])):
                raise ValueError('Checkpoint does not minimize its selection curve')
            stage_audits = {}
            with torch.no_grad():
                for stage, split in splits.items():
                    tensors = block_tensors(split, device, include_target=False)
                    mu, sigma, xi, lam = model(_flatten(tensors))
                    standardized = jsu_quantiles(mu, sigma, xi, lam, levels).cpu().numpy()
                    mean, std = split.metadata['target_mean'], split.metadata['target_std']
                    raw = standardized * std + mean
                    mu, sigma, xi, lam = (x.double().cpu().numpy() for x in (mu, sigma, xi, lam))
                    scipy_standardized = johnsonsu.ppf(QUANTILE_LEVELS,
                        a=(-mu / sigma)[..., None], b=(1 / sigma)[..., None],
                        loc=xi[..., None], scale=lam[..., None])
                    np.testing.assert_allclose(standardized, scipy_standardized, rtol=1e-11, atol=1e-8)
                    # Independent normal quantiles: do not reuse Torch's erfinv.
                    arguments = mu[..., None] + sigma[..., None] * ndtri(QUANTILE_LEVELS)
                    stage_audits[stage] = dict(finite=bool(np.isfinite(raw).all()),
                        former_clamp_boundary_exceedances=int((np.abs(arguments) > 12).sum()),
                        max_absolute_prediction=float(np.abs(raw).max()),
                        scipy_max_absolute_standardized_difference=float(np.abs(standardized - scipy_standardized).max()))
                    if stage == 'test':
                        # Some CUDA/CPU backends may round network matmuls
                        # differently; this audit requires exact saved predictions.
                        if not np.array_equal(raw, result.quantiles):
                            raise ValueError('Checkpoint replay differs from the stored test prediction')
                        stage_audits[stage]['checkpoint_prediction_max_abs_difference'] = 0.
                    else:
                        error = split.actual[..., None] - raw
                        pinball = float(np.maximum(QUANTILE_LEVELS * error,
                                                    (QUANTILE_LEVELS - 1) * error).mean())
                        if not np.isclose(pinball, history['best_selection_value'], rtol=1e-12, atol=1e-12):
                            raise ValueError('Selection checkpoint metric cannot be reproduced')
                        stage_audits[stage]['selected_mean_pinball'] = pinball
            def train_losses(path):
                return [json.loads(line)['train_loss'] for line in path.read_text().splitlines()
                        if json.loads(line).get('event') == 'epoch']
            old_losses, new_losses = (train_losses(d / 'progress.json') for d in (old_directory, directory))
            n = min(len(old_losses), len(new_losses))
            loss_diff = max(abs(a - b) for a, b in zip(old_losses[:n], new_losses[:n]))
            rows.append(dict(region=region, seed=seed, old_best_epoch=old_history['best_epoch'],
                new_best_epoch=history['best_epoch'], epochs_executed=history['epochs_executed'],
                effective_learning_rate=history['learning_rate'], common_training_epochs=n,
                training_loss_max_abs_difference=loss_diff, stages=stage_audits,
                checkpoint_sha256=sha(directory / 'best_model.pt'), prediction_sha256=sha(directory / 'test.npz')))
            print(f'Verified {region}/{seed}: epoch={history["best_epoch"]}, loss_delta={loss_diff}', flush=True)
    return dict(original_run=str(original), repaired_run=str(repaired), cells=rows,
                all_nine_replayed_exactly=True, scipy_inverse_cdf_checks_passed=True,
                same_samples_and_training_budget=True, corrected_scored_macro=report['tables']['Q']['ddnn_jsu']['macro_regions'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--repaired', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.original, args.repaired, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
