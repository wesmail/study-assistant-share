"""Streaming chat through the Anthropic API. Every call is logged with its tokens and cost.

The log is the hosted_calls table. It holds no prompt or answer text, and the key is
never stored or put in an error message. Failed and interrupted calls are logged too,
because they can still cost money.

If the main model fails, `answer_stream` tells the caller and starts again on the
fallback model. This is done here and not with the API's server-side fallback, which
covers refusals only, not rate limits, overloads or server errors. A failed attempt
restarts the answer, so no reply is ever part one model and part another.
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace

import anthropic

from app.config import HOSTED_KEY, Price, Settings, hosted_api_key, price_of
from app.db.connection import connect
from app.ingest.chunker import estimate_tokens
from app.llm.ollama import Delta, ToolCall

MILLION = 1_000_000
CACHE_WRITE_RATE = 1.25         # of the input price, when a model lists no cache prices
KEY_REJECTED = (401, 403)       # the fallback uses the same key, so it would fail too


class HostedFailure(RuntimeError):
    def __init__(self, message: str, key_rejected: bool = False) -> None:
        super().__init__(message)
        self.key_rejected = key_rejected


@dataclass
class Tokens:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @classmethod
    def of(cls, usage) -> Tokens:
        return cls(
            input=usage.input_tokens or 0,
            output=usage.output_tokens or 0,
            cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )


def cost_usd(tokens: Tokens, prices) -> float:
    """Dollars per million tokens, each kind at its own rate. Thinking counts as output."""
    price: Price = price_of(prices)
    read = price.input if price.cache_read is None else price.cache_read
    write = price.input * CACHE_WRITE_RATE if price.cache_write is None else price.cache_write
    return (tokens.input * price.input + tokens.cache_read * read + tokens.cache_write * write
            + tokens.output * price.output) / MILLION


def provider_of(model: str) -> str:
    """Who bills a logged call: OpenRouter's ids are vendor/model, Anthropic's aren't."""
    return "openrouter" if "/" in model else "anthropic"


def month_spend(settings: Settings) -> dict[str, float]:
    """Every hosted call this calendar month (UTC), eval runs included, by who bills it."""
    conn = connect(settings.db_path, apply_schema=False)    # polled by /api/options: no schema each time (X4)
    try:
        rows = conn.execute(
            "SELECT model, SUM(cost_usd) AS cost FROM hosted_calls "
            "WHERE created_at >= strftime('%Y-%m-01 00:00:00', 'now') GROUP BY model").fetchall()
    finally:
        conn.close()
    spend = {"anthropic": 0.0, "openrouter": 0.0}
    for row in rows:
        spend[provider_of(row["model"])] += row["cost"] or 0.0
    return spend


def turn_cost(settings: Settings, turn_id: str) -> dict:
    """What one question's hosted calls cost, from the log: failed attempts included."""
    conn = connect(settings.db_path, apply_schema=False)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS calls, COALESCE(SUM(cost_usd), 0) AS cost, COALESCE(SUM(input_tokens), 0) AS input, "
            "COALESCE(SUM(output_tokens), 0) AS output, COALESCE(SUM(cache_read_tokens), 0) AS cache_read, "
            "COALESCE(SUM(cache_write_tokens), 0) AS cache_write, "
            "SUM(error IS NOT NULL AND error NOT LIKE 'estimated tokens%') AS failed "    # a stop isn't a failure
            "FROM hosted_calls WHERE turn = ?", (turn_id,)).fetchone()
    finally:
        conn.close()
    # Every input token, cached or not: a tool turn reads most of its prompt from the cache.
    return {"calls": row["calls"], "cost_usd": row["cost"],
            "input_tokens": row["input"] + row["cache_read"] + row["cache_write"],
            "output_tokens": row["output"], "cache_read_tokens": row["cache_read"],
            "cache_write_tokens": row["cache_write"], "failed_calls": row["failed"] or 0}


def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """The API takes the system prompt apart from the turns."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


def request_for(messages: list[dict], settings: Settings, tools: list[dict] | None = None,
                allow_tools: bool = True) -> dict:
    """Thinking is adaptive. Current models refuse any other setting, or sampling options.

    A tool turn sends the same growing prefix several times, so it's cached. These models
    refuse a forced tool_choice, but "none" is allowed, and the tools must stay listed
    while earlier replies contain calls to them.
    """
    system, turns = split_system(messages)
    request = {
        "model": settings.hosted_model,
        "max_tokens": settings.hosted_max_tokens,
        "system": system,
        "messages": turns,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": settings.hosted_effort},
    }
    if tools:
        request["tools"] = tools
        request["cache_control"] = {"type": "ephemeral"}
        if not allow_tools:
            request["tool_choice"] = {"type": "none"}
    return request


def record_call(settings: Settings, purpose: str, tokens: Tokens, cost: float, seconds: float,
                stop_reason: str | None, error: str | None, model: str | None = None) -> None:
    conn = connect(settings.db_path)
    try:
        conn.execute(
            "INSERT INTO hosted_calls (purpose, model, input_tokens, output_tokens, cache_read_tokens, "
            "cache_write_tokens, cost_usd, seconds, stop_reason, error, turn) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (purpose, model or settings.hosted_model, tokens.input, tokens.output, tokens.cache_read,
             tokens.cache_write, cost, seconds, stop_reason, error, settings.turn_id),
        )
        conn.commit()
    finally:
        conn.close()


def describe(failure: anthropic.APIError) -> str:
    """The status and error type only, so that nothing from the request reaches a log."""
    if isinstance(failure, anthropic.APIStatusError):
        body = failure.body if isinstance(failure.body, dict) else {}
        kind = (body.get("error") or {}).get("type", "unknown") if isinstance(body.get("error"), dict) else "unknown"
        return f"HTTP {failure.status_code}, {kind}"
    return type(failure).__name__


async def stream_chat(
    messages: list[dict], settings: Settings, client=None, purpose: str = "chat",
    tools: list[dict] | None = None, allow_tools: bool = True,
) -> AsyncIterator[Delta]:
    """Yield the reply as it's generated, like the Ollama client. Leaving early stops it."""
    if settings.hosted_model not in settings.hosted_prices:
        raise RuntimeError(f"No price for {settings.hosted_model}, so its cost couldn't be logged")
    owned = client is None
    if owned:
        key = hosted_api_key()
        if not key:
            raise RuntimeError(f"{HOSTED_KEY} is not set in .env")
        client = anthropic.AsyncAnthropic(api_key=key, timeout=settings.chat_timeout_seconds)
    prices = settings.hosted_prices[settings.hosted_model]
    started = time.monotonic()
    tokens = Tokens()
    stop_reason = error = None
    streamed: list[str] = []        # what came back, for an estimate if the usage never does
    try:
        async with client.messages.stream(**request_for(messages, settings, tools, allow_tools)) as stream:
            async for event in stream:
                if event.type == "message_start":
                    tokens = Tokens.of(event.message.usage)
                elif event.type == "message_delta" and event.usage:
                    tokens.output = event.usage.output_tokens or tokens.output
                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        streamed.append(event.delta.text)
                        yield Delta(text=event.delta.text)
                    elif event.delta.type == "thinking_delta":
                        streamed.append(event.delta.thinking)
                        yield Delta(thinking=event.delta.thinking)
                    elif event.delta.type == "input_json_delta":
                        streamed.append(getattr(event.delta, "partial_json", "") or "")
            final = await stream.get_final_message()
        tokens = Tokens.of(final.usage)
        stop_reason = final.stop_reason
        blocks = getattr(final, "content", None) or []
        yield Delta(done=True, done_reason=stop_reason, prompt_tokens=tokens.input,
                    output_tokens=tokens.output, cost_usd=cost_usd(tokens, prices),
                    model=settings.hosted_model, raw=blocks,
                    tool_calls=[ToolCall(block.id, block.name, block.input)
                                for block in blocks if block.type == "tool_use"])
    except anthropic.APIError as failure:
        error = describe(failure)
        rejected = getattr(failure, "status_code", None) in KEY_REJECTED
        raise HostedFailure(f"Hosted model: {error}", rejected) from None
    finally:
        if stop_reason is None:     # cut off (Stop, a dropped connection): the final usage never came
            tokens, error = estimated(tokens, streamed, error)
        record_call(settings, purpose, tokens, cost_usd(tokens, prices),
                    time.monotonic() - started, stop_reason, error)
        if owned:
            await client.close()


def estimated(tokens: Tokens, streamed: list[str], error: str | None, prompt: str = "") -> tuple[Tokens, str | None]:
    """Tokens for a call whose usage never arrived: at least what was seen streaming back, and
    for a provider that reports input only at the end, the request's own estimate. The provider
    still bills them; a cut-off answer was logged near zero (review C5). The log says so."""
    seen = estimate_tokens("".join(streamed)) if streamed else 0
    output = seen > tokens.output
    guessed_input = not tokens.input and bool(prompt)
    if not output and not guessed_input:
        return tokens, error
    tokens = replace(tokens, output=max(tokens.output, seen),
                     input=estimate_tokens(prompt) if guessed_input else tokens.input)
    parts = [part for part, on in (("output from the streamed text", output), ("input from the request", guessed_input))
             if on]
    note = "estimated tokens: " + ", ".join(parts)
    return tokens, f"{error}; {note}" if error else note


async def answer_stream(
    messages: list[dict], settings: Settings, client=None, purpose: str = "chat", fallback: bool = True
) -> AsyncIterator[Delta]:
    """The main model's reply, or a restart on the fallback model if it fails.

    Failing means an API error, a refusal, or finishing with no answer. The restart is
    a Delta of its own, so the caller can drop what it has and say what happened.
    """
    spare = settings.hosted_fallback_model if fallback else None
    if spare == settings.hosted_model:
        spare = None
    wrote = False
    reason = None
    try:
        async for delta in stream_chat(messages, settings, client, purpose):
            if spare and delta.done and (delta.done_reason == "refusal" or not wrote):
                reason = "it refused" if delta.done_reason == "refusal" else "it returned no answer"
                continue
            wrote = wrote or bool(delta.text)
            yield delta
    except HostedFailure as failure:
        if not spare or failure.key_rejected:
            raise
        reason = str(failure).removeprefix("Hosted model: ")
    if reason is None:
        return
    yield Delta(restart=True, model=spare, reason=reason)
    try:
        async for delta in stream_chat(messages, replace(settings, hosted_model=spare), client, purpose):
            yield delta
    except HostedFailure as failure:
        raise RuntimeError(f"{settings.hosted_model} failed ({reason}), and so did {spare}: {failure}") from None
