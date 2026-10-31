"""Paired S1 evaluation on a complete daily axis; no data loading or inference."""
from __future__ import annotations

import numpy as np

from .historical_replay import QUANTILES, score_native

REGIONS = ("NSW1", "QLD1", "TAS1")
BLOCK_DAYS = 7
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261031
METRICS = ("mae", "quantile_score", "coverage_80", "coverage_90", "width_80", "width_90")


def _relative(before: float, after: float) -> float | None:
    return float((after - before) / before) if before > 0 else None


def _macro(scores: list[dict | None]) -> dict | None:
    if any(score is None for score in scores):
        return None  # An empty region cannot silently disappear from the macro.
    result = {name: float(np.mean([score[name] for score in scores])) for name in METRICS}
    result["pinball_by_quantile"] = {
        str(float(q)): float(np.mean([score["pinball_by_quantile"][str(float(q))] for score in scores]))
        for q in QUANTILES
    }
    for level in (80, 90):
        result[f"coverage_distance_{level}"] = float(np.mean([
            abs(score[f"coverage_{level}"] - level / 100) for score in scores
        ]))
    return result


def _bootstrap(a_daily: np.ndarray, b_daily: np.ndarray, common: np.ndarray) -> dict:
    """Noncircular moving blocks, shared by both arms and all three regions."""
    days = len(common)
    result = {
        "method": "synchronous_non_circular_moving_day_blocks",
        "block_days": BLOCK_DAYS, "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED, "date_axis_length": days,
        "valid_replicates": 0, "relative_qs_ci95": None,
    }
    if days < BLOCK_DAYS:
        return {**result, "status": "fewer_than_seven_planned_days"}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    starts = rng.integers(0, days - BLOCK_DAYS + 1,
                          size=(BOOTSTRAP_REPLICATES, (days + BLOCK_DAYS - 1) // BLOCK_DAYS))
    indices = (starts[..., None] + np.arange(BLOCK_DAYS)).reshape(BOOTSTRAP_REPLICATES, -1)[:, :days]
    # Missing days stay in their original positions and travel with each block.
    count = common[indices].sum(axis=1)
    a_sum = np.where(common, a_daily, 0.0)[indices].sum(axis=1)
    b_sum = np.where(common, b_daily, 0.0)[indices].sum(axis=1)
    a_region = np.divide(a_sum, count, out=np.full_like(a_sum, np.nan), where=count > 0)
    b_region = np.divide(b_sum, count, out=np.full_like(b_sum, np.nan), where=count > 0)
    a_macro, b_macro = a_region.mean(axis=1), b_region.mean(axis=1)
    valid = (count > 0).all(axis=1) & (a_macro > 0) & np.isfinite(a_macro) & np.isfinite(b_macro)
    result["valid_replicates"] = int(valid.sum())
    if not valid.all():
        # Do not condition the CI on a selected subset of successful draws.
        return {**result, "status": "undefined_resampled_region_or_reference_loss"}
    relative = (b_macro - a_macro) / a_macro
    return {**result, "status": "complete",
            "relative_qs_ci95": np.quantile(relative, [0.025, 0.975]).tolist()}


def compare_s1(a: np.ndarray, b: np.ndarray, y: np.ndarray,
               source_complete: np.ndarray, dates, regions=REGIONS) -> dict:
    """Evaluate native A/B forecasts under the fixed R10 section-8 rules.

    Shapes: A/B [days,3,24,5], labels [days,3,24], source coverage bool [days,3].
    ``dates`` is the consecutive planned date axis; absent data remains NaN.
    A/y define the primary region-day set. B must cover every such unit, with
    failures already recorded and replaced by A in the prediction ledger.
    Any unresolved B absence blocks acceptance; finite-subset scores then are
    diagnostics only, even if B availability remains above 95%.
    The source mask is reported separately and never drops a prediction: a
    caller-supplied B=A fallback contributes its actual zero paired change.

    Macro losses are equal-weight means of regional scores, including inside
    every bootstrap draw. Coverage protection averages regional absolute
    distances from nominal coverage, so opposite errors cannot cancel.
    """
    a, b, y = (np.asarray(value, dtype=np.float64) for value in (a, b, y))
    source = np.asarray(source_complete)
    dates = np.asarray(dates, dtype="datetime64[D]")
    regions = tuple(regions)
    days = len(dates)
    if (dates.ndim != 1 or not days or np.isnat(dates).any()
            or np.any(np.diff(dates) != np.timedelta64(1, "D"))):
        raise ValueError("dates must retain the nonempty consecutive planned daily axis")
    if len(regions) != 3 or len(set(regions)) != 3:
        raise ValueError("exactly three distinct regions are required")
    if (a.shape != (days, 3, 24, 5) or b.shape != a.shape
            or y.shape != (days, 3, 24) or source.shape != (days, 3)
            or source.dtype != np.dtype(bool)):
        raise ValueError("expected paired [days,3,24,5] forecasts, [days,3,24] labels, and bool source mask")

    a_available = np.isfinite(a).all(axis=(2, 3))
    b_available = np.isfinite(b).all(axis=(2, 3))
    label_available = np.isfinite(y).all(axis=2)
    primary = a_available & label_available
    common = primary & b_available
    regional = {}
    a_scores, b_scores = [], []
    for r, region in enumerate(regions):
        mask = common[:, r]
        sa = score_native(a[mask, r], y[mask, r]) if mask.any() else None
        sb = score_native(b[mask, r], y[mask, r]) if mask.any() else None
        a_scores.append(sa)
        b_scores.append(sb)
        regional[region] = {
            "planned_days": days, "common_days": int(mask.sum()),
            "primary_days": int(primary[:, r].sum()),
            "b_unresolved_on_primary_days": int((primary[:, r] & ~b_available[:, r]).sum()),
            "source_complete_days": int(source[:, r].sum()),
            "source_coverage": float(source[:, r].mean()),
            "a_prediction_availability": float(a_available[:, r].mean()),
            "b_prediction_availability": float(b_available[:, r].mean()),
            "label_availability": float(label_available[:, r].mean()),
            "common_scoring_rate": float(mask.mean()),
            "source_missing_scored_days": int((mask & ~source[:, r]).sum()),
            "A": sa, "B": sb,
            "relative_qs_change": _relative(sa["quantile_score"], sb["quantile_score"]) if sa else None,
            "relative_mae_change": _relative(sa["mae"], sb["mae"]) if sa else None,
            "coverage_distance_increase": {
                str(level): float(abs(sb[f"coverage_{level}"] - level / 100)
                                  - abs(sa[f"coverage_{level}"] - level / 100)) if sa else None
                for level in (80, 90)
            },
        }

    ma, mb = _macro(a_scores), _macro(b_scores)
    macro = {
        "A": ma, "B": mb,
        "relative_qs_change": _relative(ma["quantile_score"], mb["quantile_score"]) if ma else None,
        "relative_mae_change": _relative(ma["mae"], mb["mae"]) if ma else None,
        "coverage_distance_increase": {
            str(level): float(mb[f"coverage_distance_{level}"] - ma[f"coverage_distance_{level}"]) if ma else None
            for level in (80, 90)
        },
    }
    daily = []
    for q in (a, b):
        error = y[..., None] - q
        pb = np.maximum(QUANTILES * error, (QUANTILES - 1) * error)
        daily.append(2 * pb.mean(axis=(2, 3)))
    bootstrap = _bootstrap(*daily, common)

    checks = []

    def check(name, passed, **details):
        checks.append({"name": name, "passed": bool(passed), **details})

    for region, row in regional.items():
        check(f"{region}/b_covers_primary", row["b_unresolved_on_primary_days"] == 0,
              unresolved_days=row["b_unresolved_on_primary_days"], maximum=0, category="data")
        for field in ("source_coverage", "a_prediction_availability", "b_prediction_availability", "common_scoring_rate"):
            check(f"{region}/{field}", row[field] >= 0.95, value=row[field], minimum=0.95, category="data")
    check("macro/qs_improvement", ma is not None and ma["quantile_score"] > 0
          and mb["quantile_score"] <= 0.99 * ma["quantile_score"],
          relative_change=macro["relative_qs_change"], maximum=-0.01, category="effect")
    check("macro/mae_protection", ma is not None and mb["mae"] <= 1.01 * ma["mae"],
          relative_change=macro["relative_mae_change"], maximum=0.01, category="effect")
    for region, row in regional.items():
        sa, sb = row["A"], row["B"]
        check(f"{region}/qs_protection", sa is not None and sb["quantile_score"] <= 1.02 * sa["quantile_score"],
              relative_change=row["relative_qs_change"], maximum=0.02, category="effect")
    for name, row in [("macro", macro), *regional.items()]:
        for level in (80, 90):
            increase = row["coverage_distance_increase"][str(level)]
            check(f"{name}/coverage_{level}_protection", increase is not None and increase <= 0.01,
                  distance_increase=increase, maximum=0.01, category="effect")
    ci = bootstrap["relative_qs_ci95"]
    check("bootstrap/upper_bound_below_zero", ci is not None and ci[1] < 0,
          upper_bound=ci[1] if ci is not None else None, category="inference")

    failed = [item["name"] for item in checks if not item["passed"]]
    if any(not item["passed"] for item in checks if item["category"] == "data"):
        decision = "DATA_OR_EXECUTION_BLOCKED"
    elif days < BLOCK_DAYS:
        decision = "INCONCLUSIVE"
    elif any(not item["passed"] for item in checks if item["category"] == "effect"):
        decision = "NO_USEFUL_GAIN"
    elif not checks[-1]["passed"]:
        decision = "INCONCLUSIVE"
    else:
        decision = "PASS_INFORMATION"
    return {
        "decision": decision, "passed": decision == "PASS_INFORMATION",
        "planned_dates": dates.astype(str).tolist(), "regions": list(regions),
        "primary_mask": primary.tolist(), "common_mask": common.tolist(),
        "scores_are_partial_diagnostics": bool(np.any(primary & ~b_available)),
        "regional": regional, "macro": macro,
        "bootstrap": bootstrap, "criteria": checks, "failed_criteria": failed,
        "coverage_macro_definition": "mean_of_regional_absolute_distances_from_nominal",
        "scope": "paired S1 information screen; historical results are exploratory, not prospective confirmation",
    }
