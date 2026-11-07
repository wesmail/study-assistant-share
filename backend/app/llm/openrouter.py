"""Streaming chat through OpenRouter, for the answer option that uses it. Logged like hosted calls.

OpenRouter speaks the OpenAI chat completions protocol, over httpx, which the backend
already pins. The key is read only from STUDY_ASSISTANT_OPENROUTER_KEY in .env, through
openrouter_api_key. It goes into the Authorization header and nowhere else: not into
Settings, a log, the hosted_calls table or an error message.

Each request is pinned to one provider, with fallbacks off and only providers that
neither keep nor train on prompts, because the prompt holds book text. A reply served
by anyone else is stopped at its first event, before any of it is shown.

Every call is logged in hosted_calls with its tokens and cost, failures too. The cost is
what OpenRouter billed when it says, and otherwise the tokens at the configured price.
"""
from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import httpx

from app.config import OPENROUTER_KEY, Settings, openrouter_api_key, price_of
from app.llm.hosted import MILLION, Tokens, estimated, record_call
from app.llm.ollama import Delta, tool_call_of

API = "https://openrouter.ai/api/v1"
CLIENT_NAME = "study-assistant"
SERVICE = "OpenRouter"


class OpenRouterFailure(RuntimeError):
    pass


def cost_usd(tokens: Tokens, prices) -> float:
    """Reasoning tokens are billed as output, and OpenRouter counts them in completion_tokens."""
    price = price_of(prices)
    return (tokens.input * price.input + tokens.output * price.output) / MILLION


def describe(status: int, body: bytes) -> str:
    """The status and OpenRouter's error, cut short. Neither ever holds the key."""
    try:
        error = json.loads(body).get("error") or {}
    except (ValueError, AttributeError):
        error = {}
    message = str(error.get("message", ""))[:200] if isinstance(error, dict) else ""
    return f"HTTP {status}" + (f", {message}" if message else "")


def sse_events(lines):
    """The data of each server-sent event. OpenRouter interleaves ': OPENROUTER PROCESSING' comments."""
    async def events():
        async for line in lines:
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            yield json.loads(data)
    return events()


def provider_rules(settings: Settings) -> dict:
    rules = {"order": [settings.openrouter_route], "allow_fallbacks": False, "data_collection": "deny"}
    if settings.openrouter_quantization:
        rules["quantizations"] = [settings.openrouter_quantization]
    return rules


def request_for(messages: list[dict], settings: Settings, tools: list[dict] | None = None,
                allow_tools: bool = True) -> dict:
    """Tools stay listed while earlier turns hold calls to them; "none" stops new ones."""
    request = {
        "model": settings.openrouter_model,
        "messages": messages,
        "stream": True,
        "max_tokens": settings.openrouter_max_tokens,
        "reasoning": settings.openrouter_reasoning,
        "usage": {"include": True},
        "provider": provider_rules(settings),
        **settings.openrouter_sampling,
    }
    if tools:
        request["tools"] = tools
        if not allow_tools:
            request["tool_choice"] = "none"
    return request


class CallPieces:
    """Streamed tool calls arrive in pieces, keyed by index; arguments as a growing string."""

    def __init__(self) -> None:
        self.calls: dict[int, dict] = {}

    def feed(self, pieces: list[dict]) -> None:
        for piece in pieces:
            call = self.calls.setdefault(piece.get("index", len(self.calls)),
                                         {"id": None, "function": {"name": "", "arguments": ""}})
            call["id"] = piece.get("id") or call["id"]
            function = piece.get("function") or {}
            call["function"]["name"] += function.get("name") or ""
            call["function"]["arguments"] += function.get("arguments") or ""

    def finished(self):
        return [tool_call_of(raw, number) for number, raw in enumerate(
            (self.calls[index] for index in sorted(self.calls)), 1)]


async def stream_chat(
    messages: list[dict], settings: Settings, client: httpx.AsyncClient | None = None,
    purpose: str = "chat", tools: list[dict] | None = None, allow_tools: bool = True,
) -> AsyncIterator[Delta]:
    """Yield the reply as it's generated, like the other clients. Leaving early stops it."""
    model = settings.openrouter_model
    if model not in settings.hosted_prices:
        raise RuntimeError(f"No price for {model}, so its cost couldn't be logged")
    if not settings.openrouter_route:
        raise RuntimeError(f"No provider to pin for {model}")
    key = openrouter_api_key()
    if not key:
        raise RuntimeError(f"{OPENROUTER_KEY} is not set in .env")
    owned = client is None
    if owned:
        client = httpx.AsyncClient(timeout=httpx.Timeout(settings.chat_timeout_seconds, connect=15))
    prices = settings.hosted_prices[model]
    started = time.monotonic()
    tokens = Tokens()
    billed = finish = provider = served_model = error = None
    calls = CallPieces()
    streamed: list[str] = []
    accepted = False
    try:
        headers = {"Authorization": f"Bearer {key}", "X-Title": CLIENT_NAME, "User-Agent": CLIENT_NAME}
        async with client.stream("POST", f"{API}/chat/completions", headers=headers,
                                 json=request_for(messages, settings, tools, allow_tools)) as response:
            if response.status_code != 200:
                raise OpenRouterFailure(describe(response.status_code, await response.aread()))
            accepted = True     # from here the provider works on it, and bills it
            async for event in sse_events(response.aiter_lines()):
                if "error" in event:
                    failure = event["error"] if isinstance(event["error"], dict) else {}
                    raise OpenRouterFailure(describe(failure.get("code", 0), json.dumps(event).encode()))
                if (served := event.get("provider")) and served != settings.openrouter_route:
                    raise OpenRouterFailure(f"served by {served}, not the pinned {settings.openrouter_route}")
                provider = event.get("provider") or provider
                served_model = event.get("model") or served_model
                if usage := event.get("usage"):
                    tokens = Tokens(input=usage.get("prompt_tokens") or 0, output=usage.get("completion_tokens") or 0)
                    billed = usage.get("cost")
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    calls.feed(delta.get("tool_calls") or [])
                    streamed += [delta.get("reasoning") or "", delta.get("content") or ""]
                    streamed += [(call.get("function") or {}).get("arguments") or "" for call in delta.get("tool_calls") or []]
                    if delta.get("reasoning") or delta.get("content"):
                        yield Delta(thinking=delta.get("reasoning") or "", text=delta.get("content") or "")
                    finish = choice.get("finish_reason") or finish
        cost = billed if billed is not None else cost_usd(tokens, prices)
        yield Delta(done=True, done_reason=finish, prompt_tokens=tokens.input, output_tokens=tokens.output,
                    cost_usd=cost, model=served_model or model, provider=provider, tool_calls=calls.finished())
    except (httpx.HTTPError, ValueError) as failure:
        error = type(failure).__name__
        raise OpenRouterFailure(f"{SERVICE}: {error}") from None
    except OpenRouterFailure as failure:
        error = str(failure)
        raise OpenRouterFailure(f"{SERVICE}: {error}") from None
    finally:
        if accepted and billed is None and not tokens.input:    # cut off before the usage, which comes last
            tokens, error = estimated(tokens, streamed, error, json.dumps(messages, ensure_ascii=False))
        cost = billed if billed is not None else cost_usd(tokens, prices)
        record_call(settings, purpose, tokens, cost, time.monotonic() - started, finish, error, model=model)
        if owned:
            await client.aclose()
