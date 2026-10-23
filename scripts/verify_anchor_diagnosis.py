"""Independently reconcile every diagnostic source and weighted decomposition."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'logs/forecasting/v5_calibration_diagnosis_20261023'


def main():
    evidence=json.loads((OUTPUT/'verification.json').read_text())
    for name,expected in evidence['sources'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
    for name,expected in evidence['output_sha256'].items():
        assert hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest()==expected,name
    with (OUTPUT/'per_seed_groups.csv').open() as stream:rows=list(csv.DictReader(stream))
    with (OUTPUT/'groups.csv').open() as stream:summary=list(csv.DictReader(stream))
    report=json.loads((ROOT/'outputs/forecasting/v5_latest_aligned_20261023_r2/report.json').read_text())
    maxdiff=0.;score_checks=0;old_reports={}
    for row in rows:
        if row['group']!='all' or row['component'] not in ('B','Anchor','Full'):continue
        if row['scope']=='aligned_2023_2024':
            model={'B':'B_retrained','Anchor':'Anchor_joint','Full':'Full_joint'}[row['component']]
            saved=report[f"{model}/{row['seed']}/{row['region']}"]['scores']
            keys=(('mae','mae'),('qs','quantile_score_proxy'),('coverage80','coverage_80'),('width80','interval_width_80'))
        else:
            key=(row['region'],row['scope'][-4:],row['seed'])
            if key not in old_reports:
                path=ROOT/'outputs/forecasting/v5_anchor_confirmation_20261022_r1'/key[0]/key[1]/f'seed_{key[2]}'/'asymmetric_scale/metrics.json'
                old_reports[key]=json.loads(path.read_text())
            component={'B':'baseline','Anchor':'anchor','Full':'full'}[row['component']]
            saved=old_reports[key][row['mode']]['metrics']['all'][component]
            keys=(('mae','mae'),('qs','qs'),('coverage80','coverage_80'),('bias','bias'))
        for a,b in keys:
            diff=abs(float(row[a])-saved[b]);maxdiff=max(maxdiff,diff)
            assert np.isclose(float(row[a]),saved[b],atol=1e-10,rtol=0),(row['scope'],row['region'],a,diff)
        score_checks+=1
    groups={}
    for row in rows:
        identity=tuple(row[k] for k in ('scope','region','seed','mode'))
        groups.setdefault(identity,[]).append(row)
    for identity,selected in groups.items():
        allrows={r['component']:r for r in selected if r['group']=='all'}
        for a,b in (('Anchor','Anchor_location_only'),('Full','Full_tree_width'),('B','Anchor_scale_only')):
            assert np.isclose(float(allrows[a]['mae']),float(allrows[b]['mae']),atol=1e-10,rtol=0),(identity,a,b)
        for prefix in ('actual_diagnostic_only/','tree_q50/','horizon/'):
            buckets=[r for r in selected if r['component']=='Full' and r['group'].startswith(prefix)]
            count=sum(int(r['n_cells']) for r in buckets)
            assert count==int(allrows['Full']['n_cells'])
            for metric in ('mae_minus_B','qs_minus_B'):
                value=sum(int(r['n_cells'])*float(r[metric]) for r in buckets)/count
                assert np.isclose(value,float(allrows['Full'][metric]),atol=1e-10,rtol=0)
    keys=('scope','region','mode','group','component')
    aggregation={}
    for row in rows:aggregation.setdefault(tuple(row[k] for k in keys),[]).append(row)
    for row in summary:
        values=aggregation[tuple(row[k] for k in keys)]
        assert {v['seed'] for v in values}=={'2026','2027','2028'} and len(values)==3
        for metric in ('mae','qs','mae_minus_B','qs_minus_B','location_mean','history_mean'):
            assert np.mean([float(v[metric]) for v in values])==float(row[metric+'_mean'])
            assert np.std([float(v[metric]) for v in values],ddof=1)==float(row[metric+'_std'])
    audit=dict(source_scores_matched=True,source_score_sets_checked=score_checks,
        max_abs_difference=maxdiff,output_sets_checked=len(groups),
        diagnostic_intervention_mae_invariance=True,weighted_group_decomposition_reconciled=True,
        three_seed_aggregates_recomputed=True,sources_unchanged=True,
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUTPUT/'independent_verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
