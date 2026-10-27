"""Read-only rescoring of archived predictions; write only output/saved/.

No checkpoints are loaded. Frozen output decompositions are descriptive and
order dependent; unavailable trend/event components are explicitly recorded.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ..full_v5_mechanism_state.policy import STATES, classify
from .metrics import LEVELS, anchor_stages, movement, score, score_and_gain

FORMAL_MODELS = ('B_retrained', 'Full_joint', 'Anchor_joint',
                 'event_off_same_head', 'time_shift_off_same_head', 'Full_joint_raw')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def state_groups(states, gates):
    """Keep uncovered separate, including empty covered weak strata."""
    s, g = np.asarray(states), np.asarray(gates)
    if g.shape != (*s.shape, 3) or np.any((s < 0) | (s >= len(STATES))):
        raise ValueError('State/gate grid mismatch')
    covered = s != 5
    result = dict(all=np.ones(s.shape, bool), covered=covered, uncovered=~covered,
        weak_location=g[..., 0] <= .25, stronger_location=g[..., 0] > .25,
        covered_weak_location=covered & (g[..., 0] <= .25),
        covered_stronger_location=covered & (g[..., 0] > .25))
    result.update({f'state/{name}': s == i for i, name in enumerate(STATES)})
    for i, name in ((1, 'lower'), (2, 'upper')):
        result[f'covered_weak_{name}'] = covered & (g[..., i] <= .25)
        result[f'covered_stronger_{name}'] = covered & (g[..., i] > .25)
    return result


def price_groups(actual):
    """Outcome strata are retrospective diagnostics, never routing inputs."""
    y = np.asarray(actual)
    return {'actual/negative': y < 0, 'actual/0_to_100': (y >= 0) & (y < 100),
            'actual/100_to_300': (y >= 100) & (y < 300),
            'actual/300_to_1000': (y >= 300) & (y < 1000), 'actual/1000_plus': y >= 1000}


def validate_same_checkpoint(entries, names):
    """A frozen-output intervention must retain checkpoint, epoch AND alpha."""
    if len(names) < 2:
        raise ValueError('At least two outputs are needed for an intervention check')
    reference = entries[names[0]]
    for name in names[1:]:
        if any(entries[name][key] != reference[key] for key in ('checkpoint_sha256', 'epoch', 'alpha')):
            raise ValueError('Interventions must share checkpoint, epoch and alpha')


def _signature(*arrays):
    result = []
    for array in arrays:
        value = np.ascontiguousarray(array)
        result.append((value.shape, str(value.dtype), hashlib.sha256(value.tobytes()).hexdigest()))
    return result


class Sources:
    def __init__(self):
        self.records = {}

    def register(self, path, expected=None, kind='source'):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.records:
            self.records[key] = dict(path=key, sha256=digest(path), bytes=path.stat().st_size,
                                     kind=kind, expected_sha256=expected)
        record = self.records[key]
        if expected is not None:
            if record['sha256'] != expected:
                raise ValueError(f'Archived source hash mismatch: {path}')
            record['expected_sha256'] = expected
        return record['sha256']

    def read_json(self, path):
        self.register(path)
        return json.loads(Path(path).read_text())

    def load(self, path, expected):
        self.register(path, expected, 'prediction')
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        y, q = values['actual'], values['quantiles']
        if y.ndim != 2 or y.shape[1] != 24 or q.shape != (*y.shape, 5):
            raise ValueError(f'Unexpected forecast grid: {path}')
        if not np.array_equal(values['quantile_levels'], LEVELS):
            raise ValueError(f'Wrong quantile levels: {path}')
        origins = values['origin_unix']
        if origins.shape != (len(y),) or np.any(np.diff(origins) <= 0):
            raise ValueError(f'Nonchronological origins: {path}')
        if not np.array_equal(values['target_time_unix'], origins[:, None]+3600*np.arange(24)[None]):
            raise ValueError(f'Target times do not match 24 hourly steps: {path}')
        score(y, q)  # Validate finite outcomes and ordered quantiles too.
        return values


def _compare(computed, archived, mapping):
    errors = [abs(computed[left]-float(archived[right])) for left, right in mapping.items()]
    if any(not np.isclose(computed[left], float(archived[right]), rtol=1e-10, atol=1e-8)
           for left, right in mapping.items()):
        raise ValueError('Saved metrics disagree with independent FP64 rescoring')
    return max(errors, default=0.)


def _rows(actual, baseline, prediction, groups, identity, gates=None):
    result = []
    for name, mask in groups.items():
        row = dict(identity, group=name, cells=int(mask.sum()), origins=int(mask.any(axis=1).sum()),
                   empty=not bool(mask.any()))
        if mask.any():
            row.update(score_and_gain(actual[mask], baseline[mask], prediction[mask]))
            if gates is not None:
                row.update({f'mean_{head}_evidence_gate': float(gates[..., i][mask].mean())
                            for i, head in enumerate(('location', 'lower', 'upper'))})
        result.append(row)
    return result


def _chain(actual, stages, identity, groups):
    rows, closure = [], []
    for group, mask in groups.items():
        if not mask.any():
            continue
        totals = dict(mae_delta=0., qs_delta=0.)
        for (before_name, before), (after_name, after) in zip(stages, stages[1:]):
            values = score_and_gain(actual[mask], before[mask], after[mask])
            rows.append(dict(identity, group=group, before_stage=before_name, stage=after_name,
                             cells=int(mask.sum()), **values))
            for metric in totals:
                totals[metric] += values[metric]
        net = score_and_gain(actual[mask], stages[0][1][mask], stages[-1][1][mask])
        errors = {f'{metric}_closure_error': abs(totals[metric]-net[metric]) for metric in totals}
        if max(errors.values()) > 1e-7:
            raise ValueError('Sequential loss decomposition did not close')
        closure.append(dict(identity, group=group, **errors))
    return rows, closure


def _formal(cfg, sources):
    root = Path(cfg['formal_root'])
    report = sources.read_json(root/'report.json')
    selections = sources.read_json(root/'selection_complete.json')
    rows, stages, closures, moves, comparisons = [], [], [], [], []
    samples, max_difference = {}, 0.
    for region in cfg['regions']:
        for seed in cfg['seeds']:
            predictions, meta = {}, {}
            for model in FORMAL_MODELS:
                entry = report[f'{model}/{seed}/{region}']
                path = root/'runs'/model/f'seed{seed}'/region/'test.npz'
                data = sources.load(path, entry['sha256'])
                metadata = json.loads(str(data['metadata_json']))
                if metadata['checkpoint_sha256'] != entry['checkpoint_sha256'] or metadata['alpha'] != entry['alpha']:
                    raise ValueError('Formal checkpoint/alpha metadata mismatch')
                mode = 'raw' if model == 'Full_joint_raw' else 'guarded'
                record = selections[f'{region}/{seed}']
                checkpoint = root/'fits'/region/f'seed{seed}'/f'{mode}.pt'
                checkpoint_hash = sources.register(checkpoint, record['checkpoints'][str(checkpoint.resolve())], 'checkpoint_hash_only')
                selected = record['selection'][mode]
                expected_alpha = 0. if model == 'B_retrained' else selected['alpha']
                if checkpoint_hash != entry['checkpoint_sha256'] or selected['epoch'] != entry['epoch'] or entry['alpha'] != expected_alpha:
                    raise ValueError('Formal output differs from frozen selection record')
                current = _signature(data['origin_unix'], data['target_time_unix'], data['actual'])
                if region in samples and current != samples[region]:
                    raise ValueError('Formal model/seed sample mismatch')
                samples[region] = current
                y = data['actual'].astype(np.float64)
                predictions[model] = data['quantiles'].astype(np.float64)
                meta[model] = entry
                result = score(y, predictions[model])
                max_difference = max(max_difference, _compare(result, entry['scores'],
                    dict(mae='mae', qs='quantile_score_proxy', coverage80='coverage_80', coverage90='coverage_90')))
            b, a, f = (predictions[key] for key in ('B_retrained', 'Anchor_joint', 'Full_joint'))
            validate_same_checkpoint(meta, ('Full_joint', 'Anchor_joint', 'event_off_same_head', 'time_shift_off_same_head'))
            baseline_key = f'{region}/baseline'
            signature = _signature(b)
            if baseline_key in samples and samples[baseline_key] != signature:
                raise ValueError('Formal tree differs across neural seeds')
            samples[baseline_key] = signature
            groups = {'all': np.ones(y.shape, bool), **price_groups(y)}
            local_year = ((data['origin_unix']+10*3600).astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970)
            for year in np.unique(local_year):
                groups[f'year/{year}'] = np.broadcast_to((local_year == year)[:, None], y.shape)
            for model, q in predictions.items():
                entry = meta[model]
                identity = dict(scope='formal', region=region, seed=seed, model=model,
                    mode=entry['output_mode'], epoch=entry['epoch'], alpha=entry['alpha'])
                rows += _rows(y, b, q, groups, identity)
            identity = dict(scope='formal', region=region, seed=seed, mode='guarded',
                            epoch=meta['Full_joint']['epoch'], alpha=meta['Full_joint']['alpha'])
            sequence = anchor_stages(b, a)+[('trend_history', predictions['event_off_same_head']),
                                            ('event_given_trend', f)]
            rr, cc = _chain(y, sequence, identity, groups)
            stages += rr
            closures += cc
            moves.append(dict(identity, **movement(b, a, f)))
            comparisons.append(dict(identity, contrast='Full_to_time_shift_off_same_checkpoint',
                **score_and_gain(y, f, predictions['time_shift_off_same_head'])))
    macro = []
    for model in FORMAL_MODELS:
        selected = [r for r in rows if r['model'] == model and r['group'] == 'all']
        entry = dict(model=model, regions=len(cfg['regions']), seeds=len(cfg['seeds']))
        for metric in ('mae', 'qs', 'coverage80', 'coverage90', 'width80', 'width90'):
            per_seed = [float(np.mean([r[f'after_{metric}'] for r in selected if r['seed'] == seed])) for seed in cfg['seeds']]
            entry[f'{metric}_mean'] = float(np.mean(per_seed))
            entry[f'{metric}_seed_sd'] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.
        macro.append(entry)
    table_check = dict(status='not_comparable_subset', rows=0, max_abs_difference=None)
    if set(cfg['regions']) == {'NSW1','QLD1','TAS1'} and set(cfg['seeds']) == {2026,2027,2028}:
        table_path = Path(cfg['root'])/'logs/forecasting/main_comparison_20261023/main.csv'
        sources.register(table_path)
        with table_path.open(newline='') as stream:
            published = {row['model']: row for row in csv.DictReader(stream)}
        difference = 0.
        for row in macro:
            previous = published[row['model']]
            difference = max(difference, _compare(row, previous,
                dict(mae_mean='mae_mean', mae_seed_sd='mae_std', qs_mean='quantile_score_proxy_mean',
                     qs_seed_sd='quantile_score_proxy_std', coverage80_mean='coverage_80_mean',
                     coverage90_mean='coverage_90_mean', width80_mean='interval_width_80_mean',
                     width90_mean='interval_width_90_mean')))
        table_check = dict(status='reproduced', rows=len(macro), max_abs_difference=difference)
    return dict(scores=rows, attribution=stages, closure=closures, movement=moves,
                interventions=comparisons, macro=macro, metric_max_difference=max_difference,
                main_table_reproduction=table_check,
                prediction_files=len(FORMAL_MODELS)*len(cfg['regions'])*len(cfg['seeds']))


def _mechanism_inputs(root, region, origins, sources):
    audit = sources.read_json(root/region/'input_audit.json')
    fields = {}
    wanted = ('max_spare_capacity_mw', 'demand50_mw', 'uigf_mw', 'predispatch_covered',
              'ic_import_room_mw', 'ic_export_room_mw', 'ic_covered')
    for path, expected in audit['sources'].items():
        if Path(path).suffix != '.npz':
            continue
        sources.register(path, expected, 'mechanism_input_archive')
        with np.load(path, allow_pickle=False) as archive:
            archive_origins = archive['forecast_origin_unix']
            ids = np.searchsorted(archive_origins, origins)
            if np.any(ids >= len(archive_origins)) or not np.array_equal(archive_origins[ids], origins):
                raise ValueError('Mechanism source missing forecast origins')
            for key in ('source_run_unix', 'source_last_changed_unix', 'source_stpasa_run_unix'):
                if key in archive and np.any(archive[key][ids] > origins):
                    raise ValueError('Future source vintage in audit inputs')
            fields.update({key: archive[key][ids] for key in wanted if key in archive})
    # Match the archived float32 subtraction before conversion to float64.
    return np.stack((fields['max_spare_capacity_mw'], fields['demand50_mw']-fields['uigf_mw'],
        fields['ic_import_room_mw'], fields['ic_export_room_mw'], fields['predispatch_covered'],
        fields['ic_covered']), axis=-1).astype(np.float64)


def _development(cfg, sources):
    root, mechanism = Path(cfg['robustness_root']), Path(cfg['mechanism_diagnosis_root'])
    verification = sources.read_json(root/'independent_verification.json')
    sources.register(root/'selection_complete.json', verification['selection_sha256'])
    selections = sources.read_json(root/'selection_complete.json')
    sources.register(root/'report.json', verification['report_sha256'])
    report = sources.read_json(root/'report.json')
    mechanism_manifest = sources.read_json(mechanism/'manifest.json')
    sources.register(mechanism/'diagnosis.csv')
    with (mechanism/'diagnosis.csv').open(newline='') as stream:
        old_rows = {(r['region'], r['fold'], int(r['seed']), r['mode'], r['group']): r for r in csv.DictReader(stream)}
    rows, attribution, closure, moves, checkpoints, policies = [], [], [], [], [], []
    archive_difference, score_difference = 0., 0.
    samples, nfiles = {}, 0
    for region in cfg['regions']:
        for fold in cfg['folds']:
            policy_path = mechanism/region/str(fold)/'policy.json'
            sources.register(policy_path, mechanism_manifest['policies'][str(policy_path.resolve())])
            policy = sources.read_json(policy_path)
            inputs, states, gates, groups = None, None, None, None
            for seed in cfg['seeds']:
                for variant in ('reference', 'bounded_calibration'):
                    archived = report[f'{region}/{fold}/{seed}/{variant}']
                    frozen = selections[f'{region}/{fold}/{seed}/{variant}']
                    for mode in ('raw', 'guarded'):
                        path = root/region/str(fold)/f'seed_{seed}'/variant/f'{variant}_{mode}.npz'
                        expected = verification['source_predictions'].get(str(path.resolve()))
                        if expected is None:
                            raise ValueError(f'No archived prediction hash: {path}')
                        data = sources.load(path, expected)
                        nfiles += 1
                        y, b, a, f = (data[key].astype(np.float64) for key in ('actual', 'baseline', 'anchor', 'quantiles'))
                        sample_key = f'{region}/{fold}'
                        signature = _signature(data['origin_unix'], data['target_time_unix'], data['actual'], data['baseline'])
                        if sample_key in samples and signature != samples[sample_key]:
                            raise ValueError('Development arm/mode/seed samples or baseline differ')
                        samples[sample_key] = signature
                        if inputs is None:
                            inputs = _mechanism_inputs(mechanism, region, data['origin_unix'], sources)
                            states = classify(inputs, policy['thresholds'])
                            gates = np.asarray(policy['gates'], dtype=np.float64)[states]
                            groups = {**state_groups(states, gates), **price_groups(y)}
                            for i, name in enumerate(STATES):
                                policies.append(dict(region=region, fold=fold, state=name,
                                    evaluation_cells=int((states == i).sum()),
                                    training_cell_count=policy['cell_counts'][i],
                                    minimum_training_support_days=min(policy['day_counts'][i]),
                                    support=policy['support'][i],
                                    **{f'{head}_gate': policy['gates'][i][j] for j, head in enumerate(('location','lower','upper'))}))
                        selected = archived[mode]['selection']
                        if selected != frozen['selection'][mode]:
                            raise ValueError('Development report differs from frozen selection record')
                        if int(data['epoch']) != selected['epoch'] or float(data['alpha']) != selected['alpha']:
                            raise ValueError('Development selection metadata changed')
                        alpha = float(data['alpha'])
                        # Check exact archived blend convention; no checkpoint reselection.
                        for key, rawkey in (('anchor', 'raw_anchor'), ('quantiles', 'raw_quantiles')):
                            expected_q = b if alpha == 0 else (data[rawkey].astype(np.float64) if alpha == 1 else
                                         (1-alpha)*b+alpha*data[rawkey].astype(np.float64))
                            if not np.allclose(data[key], expected_q, rtol=0, atol=1e-9):
                                raise ValueError('Saved alpha blend does not reproduce prediction')
                        identity = dict(scope='development', region=region, fold=fold, seed=seed,
                                        variant=variant, mode=mode, epoch=selected['epoch'], alpha=alpha)
                        checkpoint = path.parent/f'{mode}.pt'
                        checkpoint_hash = sources.register(checkpoint, frozen['checkpoints'][str(checkpoint.resolve())], 'checkpoint_hash_only')
                        checkpoints.append(dict(identity, checkpoint_sha256=checkpoint_hash))
                        for output_name, q in (('anchor', a), ('full', f)):
                            values = score(y, q)
                            score_difference = max(score_difference, _compare(values, archived[mode]['metrics']['all'][output_name],
                                dict(mae='mae', qs='qs', coverage80='coverage_80', coverage90='coverage_90')))
                            scored = _rows(y, b, q, groups, dict(identity, output=output_name, representation='selected'), gates)
                            rows += scored
                            if variant == 'bounded_calibration' and output_name == 'anchor':
                                for row in scored:
                                    key = (region, str(fold), seed, mode, row['group'])
                                    if key in old_rows and not row['empty']:
                                        old = old_rows[key]
                                        if row['cells'] != int(old['cells']):
                                            raise ValueError('Mechanism group membership changed')
                                        for new, previous in (('mae_delta','anchor_mae_delta'), ('mae_harm','anchor_harm'),
                                                               ('mae_benefit','anchor_benefit'), ('wrong_direction_rate','wrong_direction_rate')):
                                            error = abs(row[new]-float(old[previous]))
                                            archive_difference = max(archive_difference, error)
                                            if error > 1e-8:
                                                raise ValueError('Saved state diagnosis failed independent replay')
                        # Both selected output and its UNBLENDED same checkpoint are
                        # available: separate alpha from the distinct raw checkpoint.
                        for representation, aa, ff in (('selected', a, f),
                                ('unblended_same_checkpoint', data['raw_anchor'].astype(np.float64), data['raw_quantiles'].astype(np.float64))):
                            ident = dict(identity, representation=representation)
                            chain = anchor_stages(b, aa)+[('combined_history', ff)]
                            rr, cc = _chain(y, chain, ident, {'all': groups['all'], **price_groups(y)})
                            attribution += rr
                            closure += cc
                            moves.append(dict(ident, **movement(b, aa, ff)))
    return dict(scores=rows, attribution=attribution, closure=closure, movement=moves,
                checkpoints=checkpoints, policies=policies, prediction_files=nfiles,
                metric_max_difference=score_difference, old_diagnosis_max_difference=archive_difference)


def _weak_strong_summary(rows):
    result = []
    keys = sorted({(r['region'], str(r['fold']), r['variant'], r['mode'], r['output']) for r in rows})
    metrics = ('mae_delta','qs_delta','mae_harm','mae_benefit','qs_harm','qs_benefit',
               'wrong_direction_rate','target_crossing_rate','harmful_overshoot_rate','before_mae',
               'lower_tail_qs_delta','upper_tail_qs_delta')
    for region, fold, variant, mode, output in keys:
        for group in ('weak_location','stronger_location','covered_weak_location','covered_stronger_location',
                      'uncovered','covered_weak_lower','covered_stronger_lower','covered_weak_upper','covered_stronger_upper'):
            selected = [r for r in rows if (r['region'],str(r['fold']),r['variant'],r['mode'],r['output'],r['group']) ==
                        (region,fold,variant,mode,output,group)]
            row = dict(region=region, fold=fold, variant=variant, mode=mode, output=output, group=group,
                       cells=selected[0]['cells'], seeds=len(selected), empty=selected[0]['empty'])
            if not row['empty']:
                row.update({metric: float(np.mean([r[metric] for r in selected])) for metric in metrics})
            result.append(row)
    return result


def audit_saved(output: Path, cfg: dict) -> dict:
    """Audit archived predictions without training, modifying or selecting models."""
    if float(cfg.get('weak_location_threshold', .25)) != .25:
        raise ValueError('This replay requires the original frozen weak-location threshold .25')
    destination = Path(output)/'saved'
    destination.mkdir(parents=True, exist_ok=False)
    sources = Sources()
    formal = _formal(cfg, sources)
    development = _development(cfg, sources)
    for prefix, result in (('formal', formal), ('development', development)):
        for key, value in result.items():
            if isinstance(value, list):
                write_csv(destination/f'{prefix}_{key}.csv', value)
    weak = _weak_strong_summary(development['scores'])
    write_csv(destination/'weak_strong_seed_means.csv', weak)
    records = list(sources.records.values())
    write_json(destination/'sources.json', records)
    definitions = dict(loss_delta='after minus before; negative is improvement',
        gross_benefit='mean(max(before loss - after loss, 0))',
        gross_harm='mean(max(after loss - before loss, 0))',
        qs='twice the mean of five pinball losses; per-quantile columns are unmultiplied pinball',
        weak_location='g_location <= .25, the original diagnostic threshold',
        weak_tail='g_lower/upper <= .25 is exploratory; original protocol only prespecified location threshold',
        sample_unit='origin-horizon cell; overlapping windows are NOT independent market events',
        seeds='neural optimization replicates, NOT independent market realizations',
        direction='q50 correction times pre-correction residual < 0; all group cells are denominator',
        target_crossing='same direction and |correction| > |residual|; not necessarily harmful',
        harmful_overshoot='same direction and |correction| > 2|residual|',
        attribution='ordered output decomposition B -> location -> lower -> upper -> history; order dependent, not retraining',
        formal_history='Anchor -> same-checkpoint event_off -> Full: trend-only and event incremental effect including saturation',
        unavailable='development NPZ lacks trend/event separately; formal raw NPZ lacks raw Anchor; these components are NOT inferred',
        maturity='this rescoring does not certify source first-publication times or rerun temporal retrieval audits',
        actual_price_groups='post-hoc outcome diagnostics, forbidden as prediction routing inputs',
        actual_price_boundaries='negative; [0,100); [100,300); [300,1000); [1000,infinity), AUD/MWh',
        formal_raw='independently selected raw checkpoint, not generally the guarded checkpoint before alpha',
        modifications='none to predictions, checkpoints, selection, or existing main tables')
    write_json(destination/'definitions.json', definitions)
    summary = dict(scope='saved_prediction_L0_only', formal_prediction_files=formal['prediction_files'],
        development_prediction_files=development['prediction_files'],
        formal_metric_max_difference=formal['metric_max_difference'],
        development_metric_max_difference=development['metric_max_difference'],
        old_mechanism_diagnosis_max_difference=development['old_diagnosis_max_difference'],
        formal_macro=formal['macro'], main_table_reproduction=formal['main_table_reproduction'], source_files=len(records),
        rows={f'{prefix}_{key}': len(value) for prefix, result in (('formal',formal),('development',development))
              for key,value in result.items() if isinstance(value,list)},
        output=str(destination.resolve()), no_model_training=True,
        no_checkpoint_loading=True, no_checkpoint_or_alpha_reselection=True)
    write_json(destination/'summary.json', summary)
    return summary
