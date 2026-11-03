"""Deterministic summaries of currently visible numerical observations."""
from __future__ import annotations

import numpy as np


def summarize_state(context, feedback=()):
    snapshot = context.snapshot
    price = np.asarray(snapshot.price, dtype=float)
    demand = np.asarray(snapshot.demand_std, dtype=float)
    days = []
    for start in (0, 24, 48):
        p, d = price[start:start + 24], demand[start:start + 24]
        days.append({'price_mean': float(p.mean()), 'price_median': float(np.median(p)),
                     'price_std': float(p.std()), 'price_range': float(np.ptp(p)),
                     'demand_std_mean': float(d.mean()),
                     'max_absolute_hourly_price_change': float(np.abs(np.diff(p)).max())})
    pstd = max(float(price[:48].std()), 1e-8)
    tags = []
    if price[-24:].std() > 1.5 * pstd:
        tags.append('higher_recent_volatility')
    if abs(price[-24:].mean() - price[:48].mean()) > pstd:
        tags.append('price_level_change')
    if demand[-24:].mean() - demand[:48].mean() > .5:
        tags.append('higher_standardized_demand')
    metrics = {}
    for row in feedback:
        # Feedback must be independently timestamped, not caller-trusted future summaries.
        if row.get('outcome_available_at', float('inf')) > min(context.cutoff, context.memory_watermark):
            continue
        if row.get('region') != context.region:
            continue
        key = row.get('candidate_id', row.get('model_id', 'unknown'))
        metrics.setdefault(key, []).append(row['metrics'])
    recent = {key: {'sample_count': len(rows), **{
        metric: float(np.mean([r[metric] for r in rows]))
        for metric in ('qs', 'mae', 'coverage_90') if all(metric in r for r in rows)}}
        for key, rows in metrics.items()}
    return {'context_id': context.identity(), 'region': context.region,
            'origin_unix': context.origin_unix, 'cutoff': context.cutoff,
            'history_hours': len(price), 'days_oldest_first': days,
            'units': {'price': 'raw RRP', 'demand': 'frozen standardized demand'},
            'similarity_fields': {'price_mean': float(price[-24:].mean()),
                                  'price_std': float(price[-24:].std()),
                                  'demand_std_mean': float(demand[-24:].mean())},
            'state_tags': tags or ['no_registered_state_flag'],
            'nonfinite_count': int((~np.isfinite(price)).sum() + (~np.isfinite(demand)).sum()),
            'mature_feedback': recent,
            'interpretation': 'Observed numerical flags; not causal explanations.'}
