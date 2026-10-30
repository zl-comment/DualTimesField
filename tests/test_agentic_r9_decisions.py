"""r9 decision regressions; synthetic records and fake HTTP only.

These tests first run against r8: its final turn can exceed the declared
three-call limit, and selected seeds do not reach the request payload.
"""
from __future__ import annotations

import json
import urllib.error

import numpy as np
import pytest

from forecasting.agentic_dev_eval import arms, llm_transport, protocol


def _inputs():
    record = {
        "region": "NSW1", "origin_local": "2022-05-07 00:00:00+10:00",
        "origin_dow": "Saturday", "origin_month": 5,
        "history_prices": np.linspace(75.0, 125.0, 72),
        "history_demand": np.full(72, 8000.0),
    }
    base = np.tile([-20.0, 10.0, 70.0, 130.0, 180.0], (24, 1))
    return record, base


def _response(content='{"action":"KEEP","reason":"unchanged"}', calls=None):
    return {
        "content": content, "tool_calls": calls or [], "finish_reason": "stop",
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


def _tool(name="base_quantiles", arguments=None):
    return {"id": "call-1", "name": name,
            "arguments": {} if arguments is None else arguments}


def test_c_binds_existing_final_schema_without_changing_prompt_or_tokens():
    captured = []

    def fake(messages, **kwargs):
        captured.append((messages, kwargs))
        return _response()

    record, base = _inputs()
    result = arms.decide_c(record, base, transport=fake)
    assert result["exit_code"] == 0
    assert len(captured) == 1
    assert captured[0][1].get("response_format") == arms.FINAL_RESPONSE_FORMAT
    assert captured[0][1]["max_tokens"] == 600
    assert captured[0][0][0]["content"] == arms._system_prompt(agent=False)


@pytest.mark.parametrize("seed", [2026, 2027, 2028])
def test_selected_seed_reaches_c_and_every_d_turn(seed):
    record, base = _inputs()
    received = []

    def fake(messages, **kwargs):
        received.append(kwargs)
        if kwargs["tag"].startswith("D") and kwargs["tag"] != "D-final":
            return _response("", [_tool()])
        return _response()

    c = arms.decide_c(record, base, transport=fake, seed=seed)
    d = arms.decide_d(record, base, None, transport=fake, seed=seed)
    assert c["exit_code"] == d["exit_code"] == 0
    assert all(call["seed"] == seed for call in received)
    assert protocol.LLM_TEMPERATURE == 0.0


def test_d_final_decision_is_inside_three_call_limit():
    record, base = _inputs()
    captured = []

    def fake(messages, **kwargs):
        captured.append(kwargs)
        if kwargs["tag"] == "D-final":
            return _response()
        return _response("", [_tool()])

    result = arms.decide_d(record, base, None, transport=fake)
    assert result["exit_code"] == 0
    assert result["llm_calls"] == len(captured) == 3
    assert [call["tag"] for call in captured] == ["D1", "D2", "D-final"]
    assert captured[-1].get("tools") is None
    assert captured[-1]["response_format"] == arms.FINAL_RESPONSE_FORMAT
    assert result["tool_calls"] == 2 <= protocol.D_MAX_TOOL_CALLS


def test_d_four_tools_still_leaves_only_one_final_turn():
    record, base = _inputs()
    captured = []

    def fake(messages, **kwargs):
        captured.append(kwargs)
        if kwargs["tag"] == "D-final":
            return _response()
        return _response("", [dict(_tool(), id=f"call-{i}") for i in range(4)])

    result = arms.decide_d(record, base, None, transport=fake)
    assert result["exit_code"] == 0
    assert result["tool_calls"] == 4
    assert result["llm_calls"] <= 3
    assert captured[-1]["tag"] == "D-final"
    assert len([c for c in captured if c["tag"] == "D-final"]) == 1


@pytest.mark.parametrize("raw", [
    '{"action":"KEEP","action":"WIDEN","reason":"duplicate"}',
    '{"action":"KEEP","reason":"first","reason":"second"}',
    '{"action":"KEEP","reason":"ok","price":123}',
    '{"action":"KEEP"}',
    '{"action":"KEEP","reason":null}',
    '{"action":"KEEP","reason":7}',
])
def test_decision_schema_rejects_duplicate_unknown_missing_or_wrong_type(raw):
    with pytest.raises(protocol.DecisionParseError):
        protocol.parse_decision(raw)


@pytest.mark.parametrize("raw", [
    '{"action":"KEEP","action":"WIDEN","reason":"duplicate"}',
    '{"action":"KEEP","reason":"ok","price":123}',
    '{"action":"UNKNOWN","reason":"outside action space"}',
])
def test_d_schema_violation_is_not_rescued_by_an_extra_final_reply(raw):
    record, base = _inputs()
    attempts = []

    def fake(messages, **kwargs):
        attempts.append(kwargs)
        return _response(raw if len(attempts) == 1 else
                         '{"action":"KEEP","reason":"later valid reply"}')

    result = arms.decide_d(record, base, None, transport=fake)
    assert result["exit_code"] != 0 and result["action"] == ""
    assert result["llm_calls"] == len(attempts) == 1


def test_d_failed_transport_attempt_counts_against_call_budget():
    record, base = _inputs()
    attempts = []

    def fake(messages, **kwargs):
        attempts.append(kwargs)
        raise llm_transport.TransportError("CONNECTION_ERROR", "synthetic")

    result = arms.decide_d(record, base, None, transport=fake)
    assert result["exit_code"] != 0 and result["action"] == ""
    assert result["llm_calls"] == len(attempts) == 1


@pytest.mark.parametrize("name,arguments", [
    ("base_quantiles", {"future_labels": True}),
    ("base_quantiles", []),
    ("history_window", {"hours": 3}),
    ("history_window", {"hours": 24.0}),
    ("hourly_profile", {"days_back": 7, "region": "QLD1"}),
    ("fetch_labels", {}),
])
def test_illegal_tool_request_fails_without_execution_or_final_rescue(name, arguments):
    record, base = _inputs()
    attempts = []

    def fake(messages, **kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            return _response("", [_tool(name, arguments)])
        return _response()

    result = arms.decide_d(record, base, None, transport=fake)
    assert result["exit_code"] != 0 and result["action"] == ""
    assert result["tool_calls"] == 0
    assert result["llm_calls"] == len(attempts) == 1


class _HTTPResponse:
    status = 200
    headers = {}

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


def _wire_body(*, arguments=None, response_model="qwen3-server-alias"):
    message = {"content": '{"action":"KEEP","reason":"unchanged"}'}
    if arguments is not None:
        message["tool_calls"] = [{"id": "call-1", "type": "function", "function": {
            "name": "history_window", "arguments": arguments}}]
    return json.dumps({
        "model": response_model,
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }).encode()


def test_transport_records_actual_seed_request_response_identity_and_usage(monkeypatch):
    received = []
    raw = _wire_body()

    def open_fake(request, timeout):
        received.append(json.loads(request.data))
        return _HTTPResponse(raw)

    monkeypatch.setattr(llm_transport.urllib.request, "urlopen", open_fake)
    trace = []
    result = llm_transport.chat([{"role": "user", "content": "synthetic"}],
                                max_tokens=600, seed=2028, tag="C", trace=trace.append)
    assert received[0]["seed"] == 2028
    assert received[0]["temperature"] == 0.0
    assert received[0]["model"] == "qwen3-30b3a"
    assert len(trace) == 1
    assert json.loads(trace[0]["raw_request_json"]) == received[0]
    assert trace[0]["raw_response_text"] == raw.decode()
    assert trace[0]["requested_model"] == "qwen3-30b3a"
    assert trace[0]["response_model"] == "qwen3-server-alias"
    assert trace[0]["seed"] == 2028
    assert trace[0]["usage"] == result["usage"]


def test_failed_http_attempt_is_traced_with_seed_without_invented_response_model(monkeypatch):
    def open_fake(request, timeout):
        raise urllib.error.URLError("synthetic connection failure")

    monkeypatch.setattr(llm_transport.urllib.request, "urlopen", open_fake)
    trace = []
    with pytest.raises(llm_transport.TransportError):
        llm_transport.chat([], max_tokens=600, seed=2027, tag="C", trace=trace.append)
    assert len(trace) == 1
    assert trace[0]["seed"] == json.loads(trace[0]["raw_request_json"])["seed"] == 2027
    assert trace[0]["response_model"] is None
    assert trace[0]["error"]


def test_duplicate_tool_argument_key_is_rejected_at_wire_boundary(monkeypatch):
    raw = _wire_body(arguments='{"hours":24,"hours":72}')
    monkeypatch.setattr(llm_transport.urllib.request, "urlopen",
                        lambda request, timeout: _HTTPResponse(raw))
    with pytest.raises(llm_transport.TransportError):
        llm_transport.chat([], max_tokens=700, tag="D1")
