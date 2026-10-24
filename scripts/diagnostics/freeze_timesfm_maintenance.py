"""Freeze the preregistered TimesFM repair decision using selection data only."""
import json
from pathlib import Path
import numpy as np
import yaml
from forecasting.common.cache import load_split
from scripts.diagnostics.baseline_maintenance import ROOT, RUN, REGIONS, digest, save

WORK = ROOT / 'outputs/forecasting/baseline_maintenance_20261024'
ARCHIVE = ROOT / 'logs/forecasting/baseline_maintenance_20261024'


def gate(scores):
    if set(scores) != set(REGIONS):
        raise ValueError('All three development regions are required')
    for pair in scores.values():
        if set(pair) != {'original', 'ridge1'} or not all(np.isfinite(v) and v > 0 for v in pair.values()):
            raise ValueError('Finite positive original/candidate development MAEs required')
    original = float(np.mean([scores[r]['original'] for r in REGIONS]))
    candidate = float(np.mean([scores[r]['ridge1'] for r in REGIONS]))
    changes = {r: scores[r]['ridge1'] / scores[r]['original'] - 1 for r in REGIONS}
    return dict(accepted=candidate < original and max(changes.values()) <= 0.05,
                original_macro_mae=original, ridge1_macro_mae=candidate,
                regional_relative_changes=changes)


def freeze():
    manifest = json.loads((RUN / 'jobs.json').read_text())
    evidence = {}
    for stage, folder in [('sample', 'timesfm_sample'), ('complete_selection', 'timesfm_full_selection')]:
        scores, files = {}, {}
        for region in REGIONS:
            split = load_split(RUN, manifest, region, 'selection')
            path = WORK / folder / f'{region}.npz'
            record = json.loads(path.with_suffix('.json').read_text())
            if record['split'] != 'selection':
                raise ValueError('Selection data only')
            expected = np.arange(len(split)) if stage == 'complete_selection' else np.unique(np.linspace(0, len(split)-1, 128).astype(int))
            with np.load(path, allow_pickle=False) as a:
                for actual, wanted in [(a['indices'], expected), (a['origin_unix'], split.origin_unix[expected]), (a['actual'], split.actual[expected])]:
                    np.testing.assert_array_equal(actual, wanted)
                target = a['actual'].astype(np.float64)
                scores[region] = {}
                for name in ('original', 'ridge1'):
                    prediction = a[name].astype(np.float64)
                    if prediction.shape != target.shape or not np.isfinite(prediction).all():
                        raise ValueError('Invalid development predictions')
                    scores[region][name] = float(np.abs(prediction - target).mean())
            files[region] = dict(path=str(path), sha256=digest(path),
                metadata_sha256=digest(path.with_suffix('.json')), count=len(expected),
                weights=record['weights'], script_sha256=record['script_sha256'])
        evidence[stage] = dict(scores=scores, gate=gate(scores), files=files)
    accepted = all(v['gate']['accepted'] for v in evidence.values())
    record = dict(accepted=accepted, selected_ridge=1.0 if accepted else 0.0,
        rule='macro MAE strictly improves; no region worsens >5%; both fixed sample and complete 2022H1',
        split='selection', seed=2026, test_used_for_selection=False,
        protocol_sha256=digest(ROOT/'docs/baseline_maintenance_20261024.md'),
        original_jobs_sha256=digest(RUN/'jobs.json'), evidence=evidence)
    dest = ARCHIVE / 'timesfm_selection.json'
    if dest.exists():
        if json.loads(dest.read_text()) != record:
            raise ValueError('Refusing to overwrite a different frozen selection')
    else:
        save(dest, record)
    if accepted:
        config = yaml.safe_load((ROOT/'configs/timesfm25_xreg/aligned.yaml').read_text())
        config['foundation_models']['timesfm25_xreg']['ridge'] = 1.0
        config['experiment']['interpreters']['timesfm25_xreg'] = str(ROOT.parent/'DualTimesField-foundation-baselines/.venvs/timesfm25/bin/python')
        config['experiment']['cpu_threads_per_job'] = 4
        config['maintenance_selection'] = dict(path=str(dest.relative_to(ROOT)), sha256=digest(dest),
            split='2022H1', candidates=[0.0, 1.0], selected_ridge=1.0,
            bare_model='diagnostic_only_not_a_selection_candidate',
            budget='two predefined XReg settings; not equal-budget HPO across all models')
        output = ROOT/'configs/timesfm25_xreg/maintenance_20261024.yaml'
        serialized = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        if output.exists() and output.read_text() != serialized:
            raise ValueError('Refusing to overwrite frozen repair config')
        output.write_text(serialized)
    print(json.dumps({k: v['gate'] for k,v in evidence.items()}, indent=2))
    print('Accepted:', accepted)


if __name__ == '__main__':
    freeze()
