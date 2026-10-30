"""Stdlib-only OpenAI-compatible client for the fixed local qwen3-30b3a endpoint."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Callable

from forecasting.agent_forecasting.execution import DeadlineExceeded, call_with_deadline

from . import protocol


class TransportError(Exception):
    def __init__(self, reason_code: str, detail: str = ""):
        super().__init__(f"llm transport error: {reason_code} {detail}")
        self.reason_code = reason_code


def chat(messages: list[dict], *, tools: list[dict] | None = None,
         response_format: dict | None = None, max_tokens: int, tag: str = "",
         seed: int = protocol.SEED, trace: Callable[[dict], None] | None = None) -> dict:
    """One bounded HTTP attempt, with optional experiment-local raw trace.

    Sending a seed at temperature zero does not establish server randomness or
    reproducibility. A returned model ID is recorded as received, including None.
    """
    payload = {
        "model": protocol.LLM_MODEL,
        "messages": messages,
        "temperature": protocol.LLM_TEMPERATURE,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
    if response_format:
        payload["response_format"] = response_format
    raw_request = json.dumps(payload)
    request = urllib.request.Request(
        f"{protocol.LLM_BASE_URL}/chat/completions",
        data=raw_request.encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    recorded = {
        "tag": tag, "seed": seed, "temperature": protocol.LLM_TEMPERATURE,
        "requested_model": protocol.LLM_MODEL, "response_model": None,
        "url": request.full_url, "raw_request_json": raw_request,
        "raw_response_text": None, "usage": None, "http_status": None,
        "error": None,
    }
    started = time.perf_counter()

    def request_once():
        try:
            with urllib.request.urlopen(request, timeout=protocol.LLM_TIMEOUT_SECONDS) as response:
                recorded["http_status"] = response.status
                raw = response.read()
        except urllib.error.HTTPError as exc:
            recorded["http_status"] = exc.code
            recorded["raw_response_text"] = exc.read().decode("utf-8", errors="replace")
            raise TransportError("HTTP_ERROR", f"status={exc.code} tag={tag}") from exc
        recorded["raw_response_text"] = raw.decode("utf-8", errors="replace")
        return protocol.json_loads_unique(raw)

    try:
        try:
            body = call_with_deadline(request_once, protocol.LLM_TIMEOUT_SECONDS)
        except DeadlineExceeded as exc:
            raise TransportError("DEADLINE_EXCEEDED", f"tag={tag}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TransportError("CONNECTION_ERROR", f"{type(exc).__name__} tag={tag}") from exc
        except ValueError as exc:
            raise TransportError("INVALID_RESPONSE_BODY", f"{exc} tag={tag}") from exc

        recorded["response_model"] = body.get("model")
        recorded["usage"] = body.get("usage")
        choice = body["choices"][0]
        message = choice["message"]
        usage = body.get("usage") or {}
        tool_calls = []
        for call in message.get("tool_calls") or []:
            fn = call["function"]
            try:
                arguments = protocol.json_loads_unique(fn["arguments"])
            except (ValueError, TypeError) as exc:
                raise TransportError("INVALID_TOOL_ARGUMENTS", f"{exc} tag={tag}") from exc
            tool_calls.append({"id": call.get("id"), "name": fn.get("name"), "arguments": arguments})
        return {
            "content": message.get("content") or "",
            "tool_calls": tool_calls,
            "finish_reason": choice.get("finish_reason"),
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            },
        }
    except TransportError as exc:
        recorded["error"] = {"reason_code": exc.reason_code, "detail": str(exc)}
        raise
    except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
        recorded["error"] = {"reason_code": "INVALID_RESPONSE_BODY", "detail": str(exc)}
        raise TransportError("INVALID_RESPONSE_BODY", f"{exc} tag={tag}") from exc
    finally:
        recorded["elapsed_seconds"] = time.perf_counter() - started
        if trace is not None:
            trace(recorded)
