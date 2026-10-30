"""Stdlib-only OpenAI-compatible client for the fixed local qwen3-30b3a endpoint."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import protocol


class TransportError(Exception):
    def __init__(self, reason_code: str, detail: str = ""):
        super().__init__(f"llm transport error: {reason_code} {detail}")
        self.reason_code = reason_code


def chat(messages: list[dict], *, tools: list[dict] | None = None,
         max_tokens: int, tag: str = "") -> dict:
    """One real HTTP call; returns content, tool_calls, usage and finish_reason."""
    payload = {
        "model": protocol.LLM_MODEL,
        "messages": messages,
        "temperature": protocol.LLM_TEMPERATURE,
        "seed": protocol.SEED,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
    request = urllib.request.Request(
        f"{protocol.LLM_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=protocol.LLM_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise TransportError("HTTP_ERROR", f"status={exc.code} tag={tag}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TransportError("CONNECTION_ERROR", f"{type(exc).__name__} tag={tag}") from exc
    except ValueError as exc:
        raise TransportError("INVALID_RESPONSE_BODY", f"{exc} tag={tag}") from exc
    choice = body.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage = body.get("usage", {}) or {}
    tool_calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        try:
            arguments = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            arguments = None  # malformed arguments become a tool-level error, not a crash
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
