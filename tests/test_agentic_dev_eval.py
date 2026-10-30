"""r6 minimal dev-eval unit tests (no network, no chronos; real-CSV grid test is
skipped automatically when the data root is absent)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting.agentic_dev_eval import arms, data, metrics, protocol

DATA_ROOT = Path(data.DEFAULT_DATA_ROOT)


# ---------- protocol: actions ----------

def _base_q():
    return np.tile(np.array([[50.0, 70.0, 100.0, 130.0, 150.0]]), (24, 1))


def _flat_history(value=100.0, n=72):
    return np.full(n, value)


def test_apply_action_keep_is_identity():
    q = _base_q()
    out = protocol.apply_action(q, _flat_history(), "KEEP")
    assert np.array_equal(out, np.sort(q, axis=-1))


def test_apply_action_shift_uses_std_of_last_24h():
    history = np.concatenate([np.full(48, 100.0), np.full(24, 200.0)])  # std of last 24 = 0
    out = protocol.apply_action(_base_q(), history, "SHIFT_UP")
    assert np.allclose(out, np.sort(_base_q(), axis=-1))  # zero scale -> no shift
    wobbling = np.concatenate([np.full(48, 100.0), 100.0 + 4.0 * np.sin(np.arange(24))])
    scale = wobbling[-24:].std()
    out2 = protocol.apply_action(_base_q(), wobbling, "SHIFT_UP")
    assert np.allclose(out2, np.sort(_base_q(), axis=-1) + protocol.SHIFT_FRACTION * scale)


def test_apply_action_widen_narrow_around_q50_and_monotone():
    base = _base_q()
    wide = protocol.apply_action(base, _flat_history(), "WIDEN")
    narrow = protocol.apply_action(base, _flat_history(), "NARROW")
    q50 = np.sort(base, axis=-1)[:, 2]
    assert np.allclose(wide[:, 4], q50 + (150.0 - q50) * protocol.SPREAD_FACTOR)
    assert np.allclose(narrow[:, 0], q50 + (50.0 - q50) / protocol.SPREAD_FACTOR)
    for out in (wide, narrow):
        assert np.all(np.diff(out, axis=-1) >= 0)
        assert np.allclose(out[:, 2], q50)  # median never moves


# ---------- protocol: fixed rule B ----------

def test_rule_b_branches():
    rising = np.concatenate([np.full(48, 100.0), np.full(24, 140.0)])
    assert protocol.rule_b(rising) == "SHIFT_UP"
    falling = np.concatenate([np.full(48, 140.0), np.full(24, 100.0)])
    assert protocol.rule_b(falling) == "SHIFT_DOWN"
    calm = np.full(72, 100.0)
    assert protocol.rule_b(calm) == "NARROW"  # vol_ratio 0 < 0.05
    wobbly = np.concatenate([np.full(48, 100.0), 100.0 + 30.0 * np.sin(np.arange(24))])
    assert protocol.rule_b(wobbly) == "WIDEN"


def test_rule_b_deterministic_and_shape_guarded():
    h = np.full(72, 55.0) + np.arange(72) * 0.01
    assert protocol.rule_b(h) == protocol.rule_b(h.copy())
    with pytest.raises(ValueError):
        protocol.rule_b(np.full(48, 1.0))


# ---------- protocol: strict decision parser ----------

def test_parse_decision_accepts_valid_object():
    assert protocol.parse_decision('{"action": "WIDEN", "reason": "wide tail"}') == "WIDEN"


@pytest.mark.parametrize("raw", ["", "not json", '["KEEP"]', '{"action": "SHRINK"}',
                                 '{"action": 7}', 'prefix {"action": "KEEP"}'])
def test_parse_decision_rejects(raw):
    with pytest.raises(protocol.DecisionParseError):
        protocol.parse_decision(raw)


# ---------- metrics ----------

def test_score_arrays_hand_computed():
    q = np.array([[[10.0, 20.0, 30.0, 40.0, 50.0], [10.0, 20.0, 30.0, 40.0, 50.0]]])
    y = np.array([[29.0, 51.0]])
    result = metrics.score_arrays(q, y)
    assert result["mae"] == pytest.approx(11.0)  # |30-29|=1, |30-51|=21
    # pinball cell y=29: l.05=0.95 l.10=0.9 l.50=0.5 l.90=1.1 l.95=1.05
    # pinball cell y=51: l.05=2.05 l.10=3.1 l.50=10.5 l.90=9.9 l.95=0.95
    pin = {0.05: (0.95 + 2.05) / 2, 0.10: (0.9 + 3.1) / 2, 0.50: (0.5 + 10.5) / 2,
           0.90: (1.1 + 9.9) / 2, 0.95: (1.05 + 0.95) / 2}
    expected_qs = 2 * np.mean(list(pin.values()))
    assert result["quantile_score"] == pytest.approx(expected_qs)
    assert result["coverage_80"] == pytest.approx(0.5)  # 29 inside [20,40], 51 outside
    assert result["coverage_90"] == pytest.approx(0.5)  # 51 also above q95=50
    assert result["width_80"] == pytest.approx(20.0)


def test_arm_metrics_failures_never_score():
    q = np.tile(np.array([[[10.0, 20.0, 30.0, 40.0, 50.0]]]), (2, 1, 1))
    y = np.array([[30.0], [30.0]])
    codes = np.array([0, 1])
    result = metrics.arm_metrics_with_failures(q, y, codes, np.array([True, True]))
    assert result["n_success"] == 1 and result["n_fail"] == 1
    assert result["mae"] == pytest.approx(0.0)


# ---------- arms C/D with fake transports ----------

def _record_dict():
    return {
        "region": "NSW1",
        "origin_local": "2022-05-07 00:00:00+10:00",
        "origin_dow": "Saturday",
        "origin_month": 5,
        "history_prices": np.full(72, 100.0) + np.linspace(0, 10, 72),
        "history_demand": np.full(72, 8000.0),
    }


def _fake_transport_ok(messages, *, tools=None, max_tokens=0, tag=""):
    return {"content": '{"action": "WIDEN", "reason": "volatile"}', "tool_calls": [],
            "finish_reason": "stop", "usage": {"prompt_tokens": 11, "completion_tokens": 7}}


def test_decide_c_success_and_failure_paths():
    record = _record_dict()
    base = _base_q()
    decision = arms.decide_c(record, base, transport=_fake_transport_ok)
    assert decision["action"] == "WIDEN" and decision["exit_code"] == 0
    assert decision["llm_calls"] == 1 and decision["tokens"] == {"prompt": 11, "completion": 7}

    def bad(messages, *, tools=None, max_tokens=0, tag=""):
        return {"content": "I think widening", "tool_calls": [], "finish_reason": "stop",
                "usage": {"prompt_tokens": 5, "completion_tokens": 5}}

    decision = arms.decide_c(record, base, transport=bad)
    assert decision["exit_code"] == 1 and decision["failure_reason"] == "INVALID_JSON"


def _tiny_frame():
    index = pd.date_range("2022-05-04 00:00", periods=400, freq="h", tz="Australia/Sydney")
    frame = pd.DataFrame({
        "delivery_start_aest": index.tz_convert("UTC"),
        "available_at_aest": index.tz_convert("UTC") + pd.Timedelta(hours=1),
        "rrp_aud_per_mwh": np.full(400, 100.0) + np.sin(np.arange(400) * np.pi / 12) * 20,
        "total_demand_mw": np.full(400, 8000.0),
        "split": ["validation"] * 400,
    })
    frame["_local"] = frame["delivery_start_aest"].dt.tz_convert("Australia/Sydney")
    frame["_year"] = frame["_local"].dt.year
    return frame


def _agent_transport(script):
    calls = {"i": 0}
    def transport(messages, *, tools=None, max_tokens=0, tag=""):
        response = script[min(calls["i"], len(script) - 1)]
        calls["i"] += 1
        return response
    return transport


def test_decide_d_tool_then_decision():
    record = _record_dict()
    frame = _tiny_frame()
    script = [
        {"content": "", "tool_calls": [{"id": "t1", "name": "history_window", "arguments": {"hours": 24}}],
         "finish_reason": "tool_calls", "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
        {"content": '{"action": "SHIFT_UP", "reason": "uptrend"}', "tool_calls": [],
         "finish_reason": "stop", "usage": {"prompt_tokens": 200, "completion_tokens": 9}},
    ]
    decision = arms.decide_d(record, _base_q(), frame, transport=_agent_transport(script))
    assert decision["action"] == "SHIFT_UP" and decision["exit_code"] == 0
    assert decision["tool_calls"] == 1 and decision["llm_calls"] == 2
    assert decision["tokens"] == {"prompt": 300, "completion": 19}


def test_decide_d_budget_and_allowlist_guards():
    record = _record_dict()
    frame = _tiny_frame()
    always_tools = [{"content": "", "tool_calls": [{"id": "t", "name": "history_window",
                                                    "arguments": {"hours": 48}}],
                     "finish_reason": "tool_calls", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}]
    decision = arms.decide_d(record, _base_q(), frame, transport=_agent_transport(always_tools))
    assert decision["exit_code"] == 1
    assert decision["failure_reason"] in ("BUDGET_EXHAUSTED_NO_DECISION",
                                          "BUDGET_EXHAUSTED_TOO_MANY_TOOL_CALLS")

    forbidden = [{"content": "", "tool_calls": [{"id": "t", "name": "fetch_labels",
                                                 "arguments": {}}],
                  "finish_reason": "tool_calls", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}]
    decision = arms.decide_d(record, _base_q(), frame, transport=_agent_transport(forbidden))
    assert decision["failure_reason"] == "TOOL_NOT_ALLOWLISTED"


def test_tool_executor_only_returns_visible_rows():
    record = _record_dict()
    frame = _tiny_frame()
    executor, stats = arms._make_tool_executor(record, _base_q(), frame)
    result = json.loads(executor("history_window", {"hours": 24}))
    assert len(result["prices"]) == 24
    # available_at of every returned row must be <= origin
    origin = pd.Timestamp(record["origin_local"])
    visible = frame[frame["available_at_aest"] <= origin]
    expected = visible.tail(24)["rrp_aud_per_mwh"].round(2).tolist()
    assert result["prices"] == expected
    bad = json.loads(executor("history_window", {"hours": 3}))
    assert "error" in bad
    assert stats["tool_calls"] == 2


# ---------- real-CSV grid integration ----------

@pytest.mark.skipif(not DATA_ROOT.exists(), reason="real holdout CSV root absent")
def test_grid_strict_asof_and_2022_containment():
    records, info = data.build_origin_grid("NSW1")
    assert info["n_scored_origins"] == len(records)
    assert len(records) == 53
    assert info["dropped"] == []
    dates = [pd.Timestamp(r.origin_local).date() for r in records]
    assert all((dates[i + 1] - dates[i]).days == 7 for i in range(len(dates) - 1))
    frame, _ = data.load_region_frame("NSW1")
    for record in records:
        origin = pd.Timestamp(record.origin_local)
        visible = frame[frame["available_at_aest"] <= origin]
        assert len(visible) == record.visible_rows
        assert record.visible_rows >= 72
        history = visible.tail(72)
        assert (history["delivery_start_aest"] < origin).all()
        after = frame[frame["delivery_start_aest"] >= origin].head(24)
        assert len(after) == 24
        assert (after["split"] == "validation").all()
        assert (after["_local"].dt.year == 2022).all()
        assert np.allclose(record.actual, after["rrp_aud_per_mwh"].to_numpy())
        assert record.half == ("H1" if origin.month <= 6 else "H2")


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="real holdout CSV root absent")
def test_grid_all_three_regions_and_aest_continuity():
    for region in protocol.REGIONS:
        records, info = data.build_origin_grid(region)
        assert len(records) == 53, info
        halves = pd.Series([r.half for r in records]).value_counts().to_dict()
        assert halves["H1"] == 26 and halves["H2"] == 27
        frame, _ = data.load_region_frame(region)
        # The CSV axis is fixed AEST with no DST gaps: rows are hourly and every
        # label window spans exactly 23 wall hours (origin hour + 23 more).
        gaps = frame["delivery_start_aest"].diff().dropna().unique()
        assert all(gap == pd.Timedelta(hours=1) for gap in gaps)
        for record in records:
            span_hours = (record.target_time_unix[-1] - record.origin_unix) / 3600.0
            assert span_hours == 23.0
        assert records[0].origin_local.startswith("2022-01-01 00:00:00+10:00")
        assert records[-1].origin_local.startswith("2022-12-31 00:00:00+10:00")
