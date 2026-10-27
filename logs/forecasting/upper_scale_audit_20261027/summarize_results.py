"""Produce readable tables from the verified CSVs; no new predictions or selection."""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

root = Path(sys.argv[1]).resolve()
scores = pd.read_csv(root / 'scores.csv')
regional = pd.read_csv(root / 'regional.csv')
macro = pd.read_csv(root / 'macro.csv')
primary = scores[(scores.role == 'primary') & (scores.scope == 'formal')
                 & (scores.variant == 'Full_joint') & (scores['mode'] == 'guarded')
                 & (scores.representation == 'selected')]
all_rows = primary[primary.group == 'all']
assert len(all_rows) == 9
groups = ['all', 'actual/0_to_300', 'actual/0_to_100', 'actual/1000_plus']
columns = ['region', 'group', 'cells', 'before_qs_mean', 'after_qs_mean', 'qs_delta_mean',
           'before_pinball_q90_mean', 'after_pinball_q90_mean',
           'before_pinball_q95_mean', 'after_pinball_q95_mean',
           'before_coverage80_mean', 'after_coverage80_mean',
           'before_coverage90_mean', 'after_coverage90_mean',
           'before_width80_mean', 'after_width80_mean',
           'before_width90_mean', 'after_width90_mean']
regional[(regional.role == 'primary') & regional.group.isin(groups)][columns].to_csv(
    root / 'primary_summary.csv', index=False)
prices = primary[primary.group.str.startswith('actual/') & (primary.group != 'actual/0_to_300')].copy()
counts = all_rows.set_index(['region', 'seed']).cells
prices['cell_share'] = [row.cells / counts.loc[(row.region, row.seed)] for row in prices.itertuples()]
prices['qs_contribution'] = prices.qs_delta * prices.cell_share
price_macro = prices.groupby('group')[['cell_share', 'qs_contribution']].mean().reset_index()
price_macro.to_csv(root / 'primary_price_contribution.csv', index=False)
prices.groupby(['region', 'group'])[['cell_share', 'qs_contribution']].mean().reset_index().to_csv(
    root / 'primary_price_contribution_by_region.csv', index=False)
primary_macro = macro[(macro.role == 'primary') & (macro.group == 'all')].iloc[0]
closure = abs(price_macro.qs_contribution.sum() - primary_macro.qs_delta_mean)
assert closure < 1e-10
report = {key: float(primary_macro[key]) for key in (
    'baseline_mae_mean', 'before_mae_mean', 'after_mae_mean', 'baseline_qs_mean',
    'before_qs_mean', 'after_qs_mean', 'qs_delta_mean', 'qs_delta_seed_sd',
    'pinball_q90_delta_mean', 'pinball_q95_delta_mean',
    'qs_benefit_mean', 'qs_harm_mean',
    'before_coverage80_mean', 'after_coverage80_mean', 'before_coverage90_mean', 'after_coverage90_mean',
    'before_width80_mean', 'after_width80_mean', 'before_width90_mean', 'after_width90_mean')}
report['weighted_price_contribution_closure_error'] = float(closure)
report['decision'] = json.loads((root / 'decision.json').read_text())['decision']
report['interpretation'] = 'Descriptive complete-archive intervention; seed SD is not a confidence interval.'
(root / 'readable_summary.json').write_text(json.dumps(report, indent=2) + '\n')
inputs = ['scores.csv', 'regional.csv', 'macro.csv', 'decision.json']
outputs = ['primary_summary.csv', 'primary_price_contribution.csv',
           'primary_price_contribution_by_region.csv', 'readable_summary.json']
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
provenance = dict(script_sha256=sha(Path(__file__)),
    input_sha256={name: sha(root / name) for name in inputs},
    output_sha256={name: sha(root / name) for name in outputs},
    weighted_price_contribution_closure_error=float(closure))
(root / 'summary_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
print(json.dumps(report, indent=2))
