"""Frozen r6 protocol constants, action space, fixed rule B and strict decision parser.

Every constant here is predeclared before any 2022 result was produced; nothing in
this module may be changed after results exist (score-driven tuning is forbidden).
"""
from __future__ import annotations

import json
from datetime import timedelta, timezone
from typing import Any, Mapping

import numpy as np

REGIONS = ("NSW1", "QLD1", "TAS1")
# The holdout CSV stores every timestamp in fixed AEST (UTC+10) year-round, and
# ACTIVE_ROUND declares the origin grid in AEST; no DST handling is needed.
AEST = timezone(timedelta(hours=10))

DEV_YEAR = 2022
DEV_SPLIT = "validation"          # holdout CSV split label for 2022 rows; 'test' rows are forbidden
GRID_FIRST_DATE = "2022-01-01"    # local-midnight origin grid, every 7 days
GRID_STEP_DAYS = 7
HISTORY_HOURS = 72
HORIZON_HOURS = 24
LABEL_WINDOW_MAX_HOURS = 26       # DST fall-back makes one 24-row window span 25 wall hours

QUANTILE_LEVELS = (0.05, 0.10, 0.50, 0.90, 0.95)
SEED = 2026
SEEDS = (2026, 2027, 2028)        # request seeds; temperature=0 need not yield distinct samples

ACTIONS = ("KEEP", "SHIFT_UP", "SHIFT_DOWN", "WIDEN", "NARROW")
SHIFT_FRACTION = 0.25             # of std of the last 24 visible prices
SPREAD_FACTOR = 1.5               # WIDEN multiplies the spread around q50 by 1.5; NARROW divides by 1.5

# Fixed rule B (predeclared, deterministic, no fitting).
RULE_MOMENTUM_THRESHOLD = 0.5     # |mean(last24) - mean(prev48)| / std(72h), floored denominator
RULE_VOL_HIGH = 0.15              # mean |hourly diff| / max(mean(last24), 1.0) above -> WIDEN
RULE_VOL_LOW = 0.05               # below -> NARROW

LLM_BASE_URL = "http://127.0.0.1:18011/v1"
LLM_MODEL = "qwen3-30b3a"
LLM_TEMPERATURE = 0.0
LLM_TIMEOUT_SECONDS = 180
C_MAX_TOKENS = 600
D_MAX_TOKENS = 700
D_MAX_TURNS = 3
D_MAX_TOOL_CALLS = 4

ACTION_DEFINITIONS = {
    "KEEP": "leave the base quantile forecast unchanged",
    "SHIFT_UP": f"move every quantile up by {SHIFT_FRACTION} of the standard deviation of the last 24 visible hourly prices",
    "SHIFT_DOWN": f"move every quantile down by {SHIFT_FRACTION} of the standard deviation of the last 24 visible hourly prices",
    "WIDEN": f"multiply the spread of quantiles around q50 by {SPREAD_FACTOR}",
    "NARROW": f"divide the spread of quantiles around q50 by {SPREAD_FACTOR}",
}


class DecisionParseError(Exception):
    def __init__(self, reason_code: str, detail: str = ""):
        super().__init__(f"decision rejected: {reason_code} {detail}")
        self.reason_code = reason_code


class DuplicateJSONKeyError(ValueError):
    pass


def json_loads_unique(raw_text: str):
    """Reject duplicate JSON fields at the decision/tool-argument boundary."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateJSONKeyError(f"duplicate field {key!r}")
            result[key] = value
        return result

    return json.loads(raw_text, object_pairs_hook=unique_object)


def apply_action(base_quantiles: np.ndarray, history_prices: np.ndarray, action: str) -> np.ndarray:
    """Apply one frozen allowlisted action to sorted base quantiles [24, 5]."""
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    base = np.sort(np.asarray(base_quantiles, dtype=np.float64), axis=-1)
    scale = float(np.std(np.asarray(history_prices[-24:], dtype=np.float64)))
    q50 = base[:, 2]
    if action == "KEEP":
        out = base.copy()
    elif action == "SHIFT_UP":
        out = base + SHIFT_FRACTION * scale
    elif action == "SHIFT_DOWN":
        out = base - SHIFT_FRACTION * scale
    elif action == "WIDEN":
        out = q50[:, None] + (base - q50[:, None]) * SPREAD_FACTOR
    else:  # NARROW
        out = q50[:, None] + (base - q50[:, None]) / SPREAD_FACTOR
    return np.sort(out, axis=-1)


def rule_b(history_prices: np.ndarray) -> str:
    """Predeclared fixed-rule decision from the 72h as-of-visible price history."""
    h = np.asarray(history_prices, dtype=np.float64)
    if h.shape != (HISTORY_HOURS,):
        raise ValueError(f"expected {HISTORY_HOURS} visible prices, got {h.shape}")
    mean_last24 = float(h[-24:].mean())
    mean_prev48 = float(h[-72:-24].mean())
    std72 = float(h.std())
    vol = float(np.abs(np.diff(h[-25:])).mean())
    momentum = (mean_last24 - mean_prev48) / max(std72, 1.0)
    vol_ratio = vol / max(mean_last24, 1.0)
    if momentum > RULE_MOMENTUM_THRESHOLD:
        return "SHIFT_UP"
    if momentum < -RULE_MOMENTUM_THRESHOLD:
        return "SHIFT_DOWN"
    if vol_ratio > RULE_VOL_HIGH:
        return "WIDEN"
    if vol_ratio < RULE_VOL_LOW:
        return "NARROW"
    return "KEEP"


def evidence_digest(record: Mapping[str, Any], base_quantiles: np.ndarray) -> dict:
    """Identical evidence payload for arms B/C/D; strictly as-of-visible inputs only."""
    h = np.asarray(record["history_prices"], dtype=np.float64)
    d = np.asarray(record["history_demand"], dtype=np.float64)
    q = np.sort(np.asarray(base_quantiles, dtype=np.float64), axis=-1)
    mean_last24 = float(h[-24:].mean())
    mean_prev48 = float(h[-72:-24].mean())
    return {
        "region": record["region"],
        "origin_local": record["origin_local"],
        "day_of_week": record["origin_dow"],
        "month": record["origin_month"],
        "history_72h": {
            "mean": round(float(h.mean()), 2),
            "mean_last24h": round(mean_last24, 2),
            "mean_prev48h": round(mean_prev48, 2),
            "std_last24h": round(float(h[-24:].std()), 2),
            "min": round(float(h.min()), 2),
            "max": round(float(h.max()), 2),
            "last_price": round(float(h[-1]), 2),
        },
        "momentum_norm": round((mean_last24 - mean_prev48) / max(float(h.std()), 1.0), 3),
        "vol_ratio": round(float(np.abs(np.diff(h[-25:])).mean()) / max(mean_last24, 1.0), 3),
        "demand_72h": {
            "mean_last24h": round(float(d[-24:].mean()), 1),
            "mean_prev48h": round(float(d[-72:-24].mean()), 1),
        },
        "base_forecast": {
            "q50_mean": round(float(q[:, 2].mean()), 2),
            "q50_min": round(float(q[:, 2].min()), 2),
            "q50_max": round(float(q[:, 2].max()), 2),
            "q10_mean": round(float(q[:, 1].mean()), 2),
            "q90_mean": round(float(q[:, 3].mean()), 2),
            "width80_mean": round(float((q[:, 3] - q[:, 1]).mean()), 2),
        },
    }


def parse_decision(raw_text: str) -> str:
    """Strict parser: the whole reply must be one JSON object with a valid action."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise DecisionParseError("EMPTY_OUTPUT")
    try:
        obj = json_loads_unique(raw_text)
    except DuplicateJSONKeyError as exc:
        raise DecisionParseError("DUPLICATE_JSON_KEY", str(exc)) from exc
    except ValueError as exc:
        raise DecisionParseError("INVALID_JSON", str(exc)[:120]) from exc
    if not isinstance(obj, dict):
        raise DecisionParseError("NOT_AN_OBJECT")
    if set(obj) != {"action", "reason"}:
        raise DecisionParseError("INVALID_FIELDS")
    action = obj.get("action")
    if action not in ACTIONS:
        raise DecisionParseError("INVALID_ACTION", repr(action)[:80])
    if not isinstance(obj["reason"], str):
        raise DecisionParseError("INVALID_REASON")
    return action
