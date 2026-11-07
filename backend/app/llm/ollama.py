"""Streaming chat through Ollama."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import Settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Delta:
    """One piece of the model's reply. The last one has `done` set."""

    thinking: str = ""
    text: str = ""
    done: bool = False
    done_reason: str | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str | None = None        # which model produced this, where that can differ
    provider: str | None = None     # who served it, for a hosted model reached through a router
    restart: bool = False           # drop everything so far: `model` takes over
    reason: str | None = None       # why it did
    tool_calls: list[ToolCall] = field(default_factory=list)    # on the last one
    raw: Any = None                 # the reply as the API gave it, to send back unchanged


def chat_payload(messages: list[dict], settings: Settings, tools: list[dict] | None = None) -> dict:
    payload = {
        "model": settings.chat_model,
        "messages": messages,
        "stream": True,
        "think": settings.chat_think,
        "keep_alive": settings.chat_keep_alive,
        # Stop at the end of the context ("length") instead of silently dropping the start of
        # the conversation, sources included, to keep writing.
        "shift": False,
        "options": {
            "num_ctx": settings.chat_num_ctx,
            "num_predict": settings.chat_num_predict,
        },
    }
    if tools:
        payload["tools"] = tools
    return payload


def tool_call_of(raw: dict, number: int) -> ToolCall:
    """Arguments normally arrive as an object; a model can also send them as a JSON string."""
    function = raw.get("function", {})
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {"invalid_json": arguments}
    return ToolCall(raw.get("id") or f"call-{number}", function.get("name", ""), arguments)


async def stream_chat(
    messages: list[dict], settings: Settings, client: httpx.AsyncClient, tools: list[dict] | None = None
) -> AsyncIterator[Delta]:
    """Yield the reply as it's generated. Leaving early stops the model."""
    url = f"{settings.ollama_url}/api/chat"
    calls: list[ToolCall] = []
    async with client.stream("POST", url, json=chat_payload(messages, settings, tools)) as response:
        if response.status_code != 200:
            detail = (await response.aread()).decode("utf-8", "replace")
            raise RuntimeError(f"Ollama returned {response.status_code}: {detail[:300]}")
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            event = json.loads(line)
            if "error" in event:
                raise RuntimeError(f"Ollama: {event['error']}")
            message = event.get("message", {})
            calls += [tool_call_of(raw, len(calls) + 1) for raw in message.get("tool_calls") or []]
            yield Delta(
                thinking=message.get("thinking") or "",
                text=message.get("content") or "",
                done=bool(event.get("done")),
                done_reason=event.get("done_reason"),
                prompt_tokens=event.get("prompt_eval_count") or 0,
                output_tokens=event.get("eval_count") or 0,
                tool_calls=calls if event.get("done") else [],
            )
