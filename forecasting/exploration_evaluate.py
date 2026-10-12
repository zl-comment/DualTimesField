"""Strict, seed-wise evaluation of aligned forecasting experiments.

Example::

    python -m forecasting.exploration_evaluate summary \
      --source 'trunk=outputs/trunk/seed{seed}/{region}.npz' --output summary.json
    python -m forecasting.exploration_evaluate compare \
      --source 'trunk=outputs/trunk/seed{seed}/{region}.npz' \
      --source 'full=outputs/full/seed{seed}/{region}.npz' \
      --reference trunk --output comparison.json

Prices must already be in AUD/MWh. NPZs contain one seed, not ensembles.
Quantile columns are q05/q10/q50/q90/q95. Optional ``levels`` or
``quantile_levels`` metadata are verified; legacy files without either use
this explicitly declared fixed column convention. Quantiles are rearranged
once for all scores; the original crossing rate is also reported.

CRPS~ is twice the equal-weight mean of five pinball losses; the levels are
nonuniform, so this is a score proxy, not an exact CRPS integral. Comparisons
average LOSSES across seeds. Seven-day moving blocks sample the complete
hourly origin grid jointly across regions. Empty-tail origins remain on that
grid with zero numerator/count; no event-conditioned timeline compression is
performed. Tail means are ratios of resampled loss sums and event counts.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

REGIONS = ("NSW1", "QLD1", "TAS1")
SEEDS = (2026, 2027, 2028)
LEVELS = np.array([0.05, 0.10, 0.50, 0.90, 0.95])
COMPARISON_METRICS = (
    "mae", "crps_quantile_approx", "crps_negative", "crps_spike",
    "pinball_q05_negative_hours", "pinball_q95_spike_hours",
)


@dataclass
class Forecast:
    point: np.ndarray
    quantile: np.ndarray
    actual: np.ndarray
    origin_unix: np.ndarray
    crossing_rate: float
    path: str = ""


def load_forecast(path: str | Path) -> Forecast:
    """Validate one raw, unensembled forecast before allowing any scoring."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"point", "quantile", "actual", "origin_unix"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        point, quantile, actual, origins = (
            np.asarray(archive[k], dtype=np.float64)
            for k in ("point", "quantile", "actual", "origin_unix")
        )
        for key in ("levels", "quantile_levels"):
            if key in archive:
                levels = np.asarray(archive[key])
                if levels.shape != LEVELS.shape or not np.allclose(levels, LEVELS, rtol=0, atol=1e-7):
                    raise ValueError(f"{path}: {key} must be {LEVELS.tolist()}")
    if actual.ndim != 2 or not all(actual.shape):
        raise ValueError(f"{path}: actual must have nonempty shape [N,H]")
    if point.shape != actual.shape or quantile.shape != (*actual.shape, len(LEVELS)):
        raise ValueError(f"{path}: point/quantile shapes must be [N,H]/[N,H,5]")
    if origins.shape != (actual.shape[0],):
        raise ValueError(f"{path}: origin_unix must have shape [N]")
    for name, values in (("point", point), ("quantile", quantile), ("actual", actual), ("origin_unix", origins)):
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: {name} contains non-finite values")
    if not np.equal(origins, np.floor(origins)).all():
        raise ValueError(f"{path}: origin_unix must be integer Unix seconds")
    if len(origins) > 1 and not np.all(np.diff(origins) == 3600):
        raise ValueError(f"{path}: origins must be a complete increasing hourly grid")
    crossing = float(np.any(np.diff(quantile, axis=-1) < 0, axis=-1).mean())
    return Forecast(point, np.sort(quantile, axis=-1), actual, origins.astype(np.int64), crossing, str(path))


def check_alignment(left: Forecast, right: Forecast, *, same_actual: bool = True) -> None:
    if left.actual.shape != right.actual.shape or not np.array_equal(left.origin_unix, right.origin_unix):
        raise ValueError(f"Forecast grids are not aligned: {left.path} vs {right.path}")
    if same_actual and not np.array_equal(left.actual, right.actual):
        raise ValueError(f"Actual prices differ: {left.path} vs {right.path}")


def load_source(pattern: str, regions: Sequence[str] = REGIONS, seeds: Sequence[int] = SEEDS) -> dict:
    if "{region}" not in pattern or "{seed}" not in pattern:
        raise ValueError("Source patterns must contain both {region} and {seed}; ensembles are not seed runs")
    result = {}
    first_region = None
    paths = set()
    for region in regions:
        result[region] = {}
        first_seed = None
        for seed in seeds:
            path = Path(pattern.format(region=region, seed=seed)).resolve()
            if path in paths:
                raise ValueError(f"Source reuses a file across seeds or regions: {path}")
            paths.add(path)
            forecast = load_forecast(path)
            if first_seed is not None:
                check_alignment(first_seed, forecast)
            else:
                first_seed = forecast
            if first_region is not None:
                check_alignment(first_region, forecast, same_actual=False)
            else:
                first_region = forecast
            result[region][seed] = forecast
    return result


def check_sources(reference: Mapping, candidate: Mapping) -> None:
    if set(reference) != set(candidate):
        raise ValueError("Sources have different regions")
    for region in reference:
        if set(reference[region]) != set(candidate[region]):
            raise ValueError(f"Sources have different seeds for {region}")
        for seed in reference[region]:
            check_alignment(reference[region][seed], candidate[region][seed])


def pinball(forecast: Forecast) -> np.ndarray:
    error = forecast.actual[..., None] - forecast.quantile
    return np.maximum(LEVELS * error, (LEVELS - 1) * error)


def _mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    return float(values[mask].mean()) if mask.any() else None


def _score_arrays(forecast: Forecast) -> dict:
    error = forecast.point - forecast.actual
    losses = pinball(forecast)
    q05, q95 = forecast.quantile[..., 0], forecast.quantile[..., -1]
    width = q95 - q05
    return {"absolute_error": np.abs(error), "squared_error": error ** 2,
            "losses": losses, "crps": 2 * losses.mean(axis=-1), "width": width,
            "interval_score": width + 20 * np.maximum(q05 - forecast.actual, 0)
                              + 20 * np.maximum(forecast.actual - q95, 0)}


def _delivery_support(mask: np.ndarray, negative: np.ndarray, spike: np.ndarray) -> dict:
    """Count distinct delivery timestamps on the validated complete hourly grid.

    Horizon h of origin i delivers at origin[0] + (i+h)*3600. Merge each
    horizon into three short boolean timelines instead of constructing and
    sorting a fresh [N,H] timestamp array for every metric/bin.
    """
    n, horizon = mask.shape
    seen = np.zeros((3, n + horizon - 1), dtype=bool)
    for h in range(horizon):
        selected = mask[:, h]
        seen[0, h:h + n] |= selected
        seen[1, h:h + n] |= selected & negative[:, h]
        seen[2, h:h + n] |= selected & spike[:, h]
    counts = seen.sum(axis=1)
    return {"n_unique_delivery_hours": int(counts[0]),
            "n_unique_negative_delivery_hours": int(counts[1]),
            "n_unique_spike_delivery_hours": int(counts[2])}


def score(forecast: Forecast, mask: np.ndarray | None = None, *, _arrays: dict | None = None) -> dict:
    """Score selected origin/horizon cells, retaining explicit support counts."""
    actual, point, quantile = forecast.actual, forecast.point, forecast.quantile
    if mask is None:
        mask = np.ones(actual.shape, dtype=bool)
    if mask.shape != actual.shape or mask.dtype != bool:
        raise ValueError("Score mask must be boolean [N,H]")
    arrays = _score_arrays(forecast) if _arrays is None else _arrays
    loss, crps = arrays["losses"], arrays["crps"]
    q05, q95 = quantile[..., 0], quantile[..., -1]
    spike, negative = actual > 300.0, actual < 0.0
    n_per_origin = mask.sum(axis=1)
    valid = n_per_origin > 0
    window_rmse = np.sqrt(np.divide((arrays["squared_error"] * mask).sum(axis=1), n_per_origin,
                                   out=np.zeros(len(actual)), where=valid))
    width, interval_score = arrays["width"], arrays["interval_score"]
    mse = _mean(arrays["squared_error"], mask)
    return {
        "n_observations": int(mask.sum()), "n_origins": int(valid.sum()),
        "n_negative": int((mask & negative).sum()), "n_spike": int((mask & spike).sum()),
        **_delivery_support(mask, negative, spike),
        "mae": _mean(arrays["absolute_error"], mask),
        "rmse_pooled": float(np.sqrt(mse)) if mse is not None else None,
        "rmse_window_mean": float(window_rmse[valid].mean()) if valid.any() else None,
        "crps_quantile_approx": _mean(crps, mask),
        "crps_negative": _mean(crps, mask & negative), "crps_spike": _mean(crps, mask & spike),
        "pinball_q05_negative_hours": _mean(loss[..., 0], mask & negative),
        "pinball_q95_spike_hours": _mean(loss[..., -1], mask & spike),
        "spike_hit_rate_q95": _mean(q95 > 300, mask & spike),
        "spike_false_alarm_rate_q95": _mean(q95 > 300, mask & ~spike),
        "negative_hit_rate_q05": _mean(q05 < 0, mask & negative),
        "negative_false_alarm_rate_q05": _mean(q05 < 0, mask & ~negative),
        "coverage_90": _mean((actual >= q05) & (actual <= q95), mask),
        "interval_width_90": _mean(width, mask), "ais_90": _mean(interval_score, mask),
    }


def _aggregate(values: Sequence[dict]) -> dict:
    result = {}
    for key in values[0]:
        items = [v[key] for v in values]
        if isinstance(items[0], dict):
            result[key] = _aggregate(items)
        elif isinstance(items[0], (str, list)):
            continue
        else:
            valid = [x for x in items if x is not None]
            result[key] = {
                "mean": float(np.mean(valid)) if valid else None,
                "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else None,
                "n": len(valid),
            }
    return result


def summarize_forecast(forecast: Forecast) -> dict:
    # origin_unix is the delivery timestamp of horizon 1; AEMO uses fixed AEST.
    deliveries = forecast.origin_unix[:, None] + np.arange(forecast.actual.shape[1])[None, :] * 3600
    months = (deliveries + 10 * 3600).astype("datetime64[s]").astype("datetime64[M]").astype(str)
    actual = forecast.actual
    arrays = _score_arrays(forecast)
    masks = {"negative": actual < 0, "0_to_100": (actual >= 0) & (actual <= 100),
             "100_to_300": (actual > 100) & (actual <= 300), "spike": actual > 300}
    return {
        "overall": score(forecast, _arrays=arrays), "quantile_crossing_rate_before_rearrangement": forecast.crossing_rate,
        "price_bands": {name: score(forecast, mask, _arrays=arrays) for name, mask in masks.items()},
        "months_aest": {month: score(forecast, months == month, _arrays=arrays) for month in np.unique(months)},
        "horizons": {str(h + 1): score(forecast, np.broadcast_to(np.arange(actual.shape[1]) == h, actual.shape), _arrays=arrays)
                     for h in range(actual.shape[1])},
    }


def summarize_source(source: Mapping) -> dict:
    per_seed, region_summaries = {}, {}
    for region, forecasts in source.items():
        per_seed[region] = {str(seed): summarize_forecast(forecast) for seed, forecast in forecasts.items()}
        region_summaries[region] = _aggregate(list(per_seed[region].values()))
    seeds = list(next(iter(source.values())))
    # Average regions within each seed, then report seed variability; regions are
    # not treated as extra independent training replicates.
    macro_seed = {}
    for seed in seeds:
        rows = [per_seed[region][str(seed)]["overall"] for region in source]
        macro_seed[str(seed)] = {
            key: float(np.mean([row[key] for row in rows])) if all(row[key] is not None for row in rows) else None
            for key in rows[0]
        }
    return {"per_seed": per_seed, "region_seed_summary": region_summaries,
            "macro_region_per_seed": macro_seed, "macro_region_seed_summary": _aggregate(list(macro_seed.values()))}


def origin_loss_parts(forecast: Forecast, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Loss SUM and observation count per original origin, including empty tails."""
    actual = forecast.actual
    mask = np.ones(actual.shape, dtype=bool)
    if metric == "mae":
        hourly = np.abs(forecast.point - actual)
    else:
        losses = pinball(forecast)
        hourly = 2 * losses.mean(axis=-1)
        if metric in ("crps_negative", "pinball_q05_negative_hours"):
            mask = actual < 0
        elif metric in ("crps_spike", "pinball_q95_spike_hours"):
            mask = actual > 300
        elif metric != "crps_quantile_approx":
            raise ValueError(f"Unsupported comparison metric: {metric}")
        if metric == "pinball_q05_negative_hours":
            hourly = losses[..., 0]
        elif metric == "pinball_q95_spike_hours":
            hourly = losses[..., -1]
    return (hourly * mask).sum(axis=1), mask.sum(axis=1)


def _block_sums(values: np.ndarray, starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    prefix = np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)], axis=0)
    return (prefix[starts + lengths[None, :]] - prefix[starts]).sum(axis=1)


def _inference(estimate: float | None, replicates: np.ndarray, alpha: float) -> dict:
    valid = replicates[np.isfinite(replicates)]
    if estimate is None or not len(valid):
        return {"difference": estimate, "ci_low": None, "ci_high": None, "p_value": None,
                "bootstrap_valid_replicates": int(len(valid))}
    # Bootstrap null distribution centered at the observed difference.
    p = (1 + np.count_nonzero(np.abs(valid - estimate) >= abs(estimate))) / (len(valid) + 1)
    low, high = np.quantile(valid, [alpha / 2, 1 - alpha / 2])
    return {"difference": estimate, "ci_low": float(low), "ci_high": float(high),
            "p_value": float(p), "bootstrap_valid_replicates": int(len(valid))}


def compare_sources(reference: Mapping, candidate: Mapping, *, replicates: int = 19999,
                    block_hours: int = 168, random_seed: int = 2026, alpha: float = 0.05,
                    metrics: Sequence[str] = COMPARISON_METRICS) -> dict:
    check_sources(reference, candidate)
    if replicates < 2 or block_hours < 1 or not 0 < alpha < 1:
        raise ValueError("Require replicates >= 2, block_hours >= 1 and 0 < alpha < 1")
    regions = list(reference)
    first = next(iter(reference[regions[0]].values()))
    n = len(first.origin_unix)
    # Ordinary noncircular moving blocks, with a truncated final block.
    length = min(block_hours, n)
    lengths = np.full((n + length - 1) // length, length, dtype=int)
    lengths[-1] = n - int(lengths[:-1].sum())
    rng = np.random.default_rng(random_seed)
    starts = rng.integers(0, n - length + 1, size=(replicates, len(lengths)))
    results = {}
    for metric in metrics:
        differences, counts = [], []
        for region in regions:
            per_seed = []
            denominator = None
            for seed in reference[region]:
                ref_sum, count = origin_loss_parts(reference[region][seed], metric)
                new_sum, new_count = origin_loss_parts(candidate[region][seed], metric)
                if not np.array_equal(count, new_count):
                    raise ValueError("Paired event supports differ")
                per_seed.append(new_sum - ref_sum)
                denominator = count
            differences.append(np.mean(per_seed, axis=0))
            counts.append(denominator)
        diff = np.stack(differences, axis=1)
        count = np.stack(counts, axis=1)
        total_counts = count.sum(axis=0)
        estimates = np.divide(diff.sum(axis=0), total_counts,
                              out=np.full(len(regions), np.nan), where=total_counts > 0)
        boot_count = _block_sums(count, starts, lengths)
        boot_diff = np.divide(_block_sums(diff, starts, lengths), boot_count,
                              out=np.full((replicates, len(regions)), np.nan), where=boot_count > 0)
        rows = {region: _inference(float(estimates[i]) if total_counts[i] else None, boot_diff[:, i], alpha)
                | {"n_observations": int(total_counts[i]), "n_origins": n,
                   "n_event_origins": int(np.count_nonzero(count[:, i]))}
                for i, region in enumerate(regions)}
        macro = float(estimates.mean()) if np.isfinite(estimates).all() else None
        # Same starts for all regions preserve cross-market dependence.
        rows["macro_regions"] = _inference(macro, boot_diff.mean(axis=1), alpha)
        results[metric] = rows
    return {"direction": "candidate minus reference; negative favors candidate",
            "bootstrap": {"method": "paired noncircular moving blocks on complete hourly origin grid",
                          "block_hours_requested": block_hours, "block_hours_used": length,
                          "replicates": replicates, "random_seed": random_seed, "confidence": 1 - alpha,
                          "n_origins": n, "possible_full_block_starts": n - length + 1,
                          "seed_aggregation": "mean losses, not mean forecasts",
                          "region_aggregation": "equal-weight mean of region scores with joint block starts",
                          "tail_aggregation": "resampled loss sum / resampled event count"},
            "metrics": results}


def holm_adjust(comparisons: Mapping, alpha: float = 0.05) -> None:
    """One declared family over all candidate/metric/region tests in this run."""
    rows = [row for comparison in comparisons.values() for regions in comparison["metrics"].values()
            for row in regions.values()]
    finite = sorted((row for row in rows if row["p_value"] is not None), key=lambda row: row["p_value"])
    largest = 0.0
    for i, row in enumerate(finite):
        largest = max(largest, (len(finite) - i) * row["p_value"])
        row["p_value_holm"] = min(1.0, largest)
        row["reject_holm"] = row["p_value_holm"] <= alpha
    for row in rows:
        row.setdefault("p_value_holm", None)
        row.setdefault("reject_holm", None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("summary", "compare"))
    parser.add_argument("--source", action="append", required=True, help="name=path with {region} and {seed}")
    parser.add_argument("--reference", help="Source name used as comparator (compare only)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=19999)
    parser.add_argument("--block-hours", type=int, default=168)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()
    sources = {}
    for item in args.source:
        name, separator, pattern = item.partition("=")
        if not separator or not name or not pattern or name in sources:
            parser.error("Sources must have unique nonempty names in name=pattern form")
        sources[name] = pattern
    if args.command == "compare" and (args.reference not in sources or len(sources) < 2):
        parser.error("compare requires --reference naming one of at least two sources")
    result = {"schema_version": 1, "regions": list(REGIONS), "seeds": list(SEEDS),
              "quantile_levels": LEVELS.tolist(), "quantile_policy": "ascending rearrangement; crossing rate recorded",
              "legacy_level_metadata": "absent metadata uses declared fixed q05/q10/q50/q90/q95 column convention",
              "price_units": "AUD/MWh", "month_timezone": "AEST UTC+10; delivery time",
              "support_definition": "n_observations/n_negative/n_spike count forecast-origin x horizon cells; n_unique_*_delivery_hours deduplicate actual delivery timestamps, without assuming temporal independence",
              "crps_quantile_approx_definition": "2 * equal-weight mean pinball at five nonuniform quantile levels; not exact CRPS",
              "source_patterns": sources, "summaries": {}, "comparisons": {}}
    reference_name = args.reference if args.command == "compare" else next(iter(sources))
    reference = load_source(sources[reference_name])
    result["summaries"][reference_name] = summarize_source(reference)
    for name, pattern in sources.items():
        if name == reference_name:
            continue
        candidate = load_source(pattern)
        check_sources(reference, candidate)
        result["summaries"][name] = summarize_source(candidate)
        if args.command == "compare":
            result["comparisons"][name] = compare_sources(reference, candidate, replicates=args.replicates,
                                                          block_hours=args.block_hours, random_seed=args.bootstrap_seed)
        del candidate
    if args.command == "compare":
        result["reference"] = reference_name
        result["holm_family"] = "all finite tests across candidates, metrics, individual regions and macro regions"
        holm_adjust(result["comparisons"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}: {len(sources)} sources, {len(REGIONS)} regions, {len(SEEDS)} seeds")


if __name__ == "__main__":
    main()
