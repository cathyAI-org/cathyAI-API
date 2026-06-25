from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .tools import build_tool_policy, execute_tool_call, tool_specs_for_policy


UI_ONLY_BODY_KEYS = {
    "mode",
    "settings",
}


def strip_ui_only_fields(body: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in body.items() if k not in UI_ONLY_BODY_KEYS}


async def prepare_ollama_chat_request(
    body: dict[str, Any],
    ai_backend_url: str,
    timeout: httpx.Timeout,
    max_tool_rounds: int = 3,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ollama_body = strip_ui_only_fields(body)
    mode = str(body.get("mode") or "assistant")
    settings = body.get("settings") if isinstance(body.get("settings"), dict) else {}

    policy = build_tool_policy(mode, settings)
    tools = tool_specs_for_policy(policy)

    if not policy.internet_enabled:
        messages = ollama_body.get("messages")
        if isinstance(messages, list):
            ollama_body["messages"] = [
                {
                    "role": "system",
                    "content": (
                        "Internet access is disabled for this request. "
                        "Do not claim to search, browse, check current news, look up prices, "
                        "or access live/current external information. "
                        "If the user asks for current or web-only information, say that internet access is disabled "
                        "and answer only from existing context if possible."
                    ),
                },
                *messages,
            ]

    if not tools:
        return ollama_body, None

    messages = ollama_body.get("messages")
    model = ollama_body.get("model")

    if not isinstance(messages, list) or not messages or not model:
        return ollama_body, None

    working_messages = list(messages)
    used_tools = False

    async with httpx.AsyncClient(timeout=timeout) as client:
        for _ in range(max_tool_rounds):
            planning_body = dict(ollama_body)
            planning_body["messages"] = working_messages
            planning_body["stream"] = False
            planning_body["tools"] = tools

            r = await client.post(f"{ai_backend_url}/api/chat", json=planning_body)
            r.raise_for_status()
            data = r.json()

            assistant_msg = data.get("message") or {}
            tool_calls = assistant_msg.get("tool_calls") or []

            if not tool_calls:
                if used_tools:
                    return ollama_body, data
                return ollama_body, None

            used_tools = True

            tool_request_msg = {
                "role": "assistant",
                "tool_calls": tool_calls,
            }
            if assistant_msg.get("content"):
                tool_request_msg["content"] = assistant_msg.get("content")
            if assistant_msg.get("thinking"):
                tool_request_msg["thinking"] = assistant_msg.get("thinking")

            working_messages.append(tool_request_msg)

            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name")
                arguments = fn.get("arguments") or {}

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except Exception:
                        arguments = {"raw": arguments}

                if not isinstance(arguments, dict):
                    arguments = {"value": arguments}

                result = await execute_tool_call(str(name), arguments, policy)

                working_messages.append({
                    "role": "tool",
                    "tool_name": str(name),
                    "content": json.dumps(result, ensure_ascii=False),
                })

    return ollama_body, _synthetic_ollama_response(
        model=str(model),
        content="I stopped because the model requested too many tool calls in a row.",
    )


def _synthetic_ollama_response(model: str, content: str) -> dict[str, Any]:
    return {
        "model": model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {
            "role": "assistant",
            "content": content,
        },
        "done": True,
        "done_reason": "stop",
    }


async def iter_precomputed_ollama_ndjson(data: dict[str, Any]):
    yield (json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8")
