"""One turn with tools: the model replies, tools run, and it goes on until it answers.

A file takes three replies, because neither model can reliably write a document and call
a tool in the same one (see schemas.py):

    1. the model calls make_pdf with a file name, which only declares the file
    2. tools are off, and its whole next reply is the document, shown as it's written
    3. the app builds the file and tells the model how that went

The models differ in how a tool call and its result are written down, so each has a small
adapter here and the loop itself doesn't know which one it's talking to.
"""
from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from app.config import Settings
from app.llm import hosted, ollama, openrouter
from app.llm.citations import Citations
from app.llm.ollama import Delta, ToolCall
from app.tools.executor import Outcome, Pending, ToolBox
from app.tools.schemas import anthropic_tools, ollama_tools

Event = tuple[str, dict]
CUT_OFF = ("length", "max_tokens")
BAD_CALL = "bad_tool_call"
UNPARSED_CALL = "error parsing tool call"   # Ollama gives up on malformed JSON from the model
SHORT_PREAMBLE = 200                    # characters of "Here is your document:"
FIRST_HEADING = re.compile(r"^#{1,6}\s", re.M)
OUTER_FENCE = re.compile(r"\A\s*```[a-z]*\n(.*?)\n```\s*\Z", re.S)


class ModelFailed(Exception):
    """The model gave nothing usable, so a fallback may try. Tools from that reply don't run."""


@dataclass
class Finished:
    """The turn's totals, for the "done" event."""

    answer: str = ""
    done_reason: str | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str | None = None
    provider: str | None = None
    files: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)


class LocalModel:
    """gpt-oss through Ollama."""

    tool_flags: dict = {}               # web, propose: set by run_turn from the toolbox

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None) -> None:
        self.settings = settings
        self.transport = transport

    async def stream(self, messages: list[dict], allow_tools: bool) -> AsyncIterator[Delta]:
        timeout = httpx.Timeout(self.settings.chat_timeout_seconds, connect=10)
        async with httpx.AsyncClient(timeout=timeout, transport=self.transport) as client:
            tools = ollama_tools(**self.tool_flags) if allow_tools else None
            try:
                async for delta in ollama.stream_chat(messages, self.settings, client, tools):
                    yield delta
            except RuntimeError as error:
                if UNPARSED_CALL not in str(error):
                    raise
                yield Delta(done=True, done_reason=BAD_CALL)

    def assistant_turn(self, text: str, thinking: str, done: Delta) -> dict:
        turn: dict = {"role": "assistant", "content": text}
        if thinking:
            turn["thinking"] = thinking
        if done.tool_calls:
            turn["tool_calls"] = [{"id": call.id, "function": {"name": call.name, "arguments": call.arguments}}
                                  for call in done.tool_calls]
        return turn

    def tool_results(self, calls: list[ToolCall], outcomes: list[Outcome]) -> list[dict]:
        return [{"role": "tool", "tool_name": call.name, "content": outcome.text}
                for call, outcome in zip(calls, outcomes)]


class HostedModel:
    """Claude through the Anthropic API. In deep mode, every tool result goes to Anthropic."""

    tool_flags: dict = {}

    def __init__(self, settings: Settings, purpose: str, client=None) -> None:
        self.settings = settings
        self.purpose = purpose
        self.client = client

    async def stream(self, messages: list[dict], allow_tools: bool) -> AsyncIterator[Delta]:
        async for delta in hosted.stream_chat(messages, self.settings, self.client, self.purpose,
                                              anthropic_tools(**self.tool_flags), allow_tools):
            yield delta

    def assistant_turn(self, text: str, thinking: str, done: Delta) -> dict:
        """The content blocks exactly as they came: thinking must go back unchanged."""
        return {"role": "assistant", "content": done.raw}

    def tool_results(self, calls: list[ToolCall], outcomes: list[Outcome]) -> list[dict]:
        return [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call.id, "content": outcome.text, "is_error": not outcome.ok}
            for call, outcome in zip(calls, outcomes)
        ]}]


class OpenRouterModel:
    """A hosted model through OpenRouter, in the OpenAI format. Every tool result goes out too."""

    tool_flags: dict = {}

    def __init__(self, settings: Settings, purpose: str, client=None) -> None:
        self.settings = settings
        self.purpose = purpose
        self.client = client

    async def stream(self, messages: list[dict], allow_tools: bool) -> AsyncIterator[Delta]:
        async for delta in openrouter.stream_chat(messages, self.settings, self.client, self.purpose,
                                                  ollama_tools(**self.tool_flags), allow_tools):
            yield delta

    def assistant_turn(self, text: str, thinking: str, done: Delta) -> dict:
        turn: dict = {"role": "assistant", "content": text}
        if done.tool_calls:
            turn["tool_calls"] = [{"id": call.id, "type": "function", "function": {
                "name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}}
                for call in done.tool_calls]
        return turn

    def tool_results(self, calls: list[ToolCall], outcomes: list[Outcome]) -> list[dict]:
        return [{"role": "tool", "tool_call_id": call.id, "content": outcome.text}
                for call, outcome in zip(calls, outcomes)]


def document_body(text: str) -> str:
    """What the model wrote when told to write only the document, minus habits."""
    text = text.strip()
    fenced = OUTER_FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    heading = FIRST_HEADING.search(text)
    if heading and 0 < heading.start() <= SHORT_PREAMBLE and "$" not in text[:heading.start()]:
        text = text[heading.start():]
    return text


async def run_turn(model, messages: list[dict], box: ToolBox, citations: Citations,
                   settings: Settings, strict: bool) -> AsyncIterator[Event]:
    """Yield chat events, ending with ("finished", Finished). `messages` grows as it goes.

    With `strict`, a refusal or an empty reply raises ModelFailed, so a fallback can start
    the turn again.
    """
    model.tool_flags = {"web": box.web_tools, "propose": box.propose, "annotate": box.annotate}
    finished = Finished()
    pending: Pending | None = None
    draft = ""                          # text written alongside a file tool's call
    document_start = 0                  # where the file's text began in the answer
    withdrawn: tuple[int, str] | None = None    # a failed file's text, taken out of the answer
    shown_documents: dict[str, tuple[int, int]] = {}    # each file tool's latest built document, in the answer
    asked_again = False                 # an empty document gets one more try
    calls_seen = 0

    for round_number in range(settings.tool_rounds):
        # After a proposed search the model only finishes its reply: the reader decides.
        last_round = round_number == settings.tool_rounds - 1
        allow_tools = pending is None and not last_round and box.proposal is None
        if last_round and pending is None and messages and messages[-1].get("role") == "tool":
            # gpt-oss, after a long run of tool calls, wrote nothing once tools were taken away.
            messages.append({"role": "user", "content": (
                "(From the app, not the reader.) No more tools: answer the reader's question now, "
                "with what you have, citing it as you were told.")})
        writing = pending is not None
        round_start = len(finished.answer)
        text = thinking = ""
        done: Delta | None = None
        separator = "\n\n" if finished.answer.strip() else ""

        async for delta in model.stream(messages, allow_tools):
            if delta.thinking:
                thinking += delta.thinking
                yield "thinking", {"text": delta.thinking}
            if delta.text and (shown := citations.feed(delta.text)):
                if not text and separator:
                    finished.answer += separator
                    yield "token", {"text": separator}
                text += shown
                finished.answer += shown
                yield "token", {"text": shown}
            if delta.done:
                done = delta
        if rest := citations.flush():
            text += rest
            finished.answer += rest
            yield "token", {"text": rest}

        if done is None:
            raise ModelFailed("the stream ended early")
        finished.done_reason = done.done_reason
        finished.prompt_tokens += done.prompt_tokens
        finished.output_tokens += done.output_tokens
        finished.cost_usd += done.cost_usd
        finished.model = done.model or finished.model
        finished.provider = done.provider or finished.provider
        if strict and done.done_reason == "refusal":
            raise ModelFailed("it refused")
        if strict and not text.strip() and not done.tool_calls and not finished.tools:
            raise ModelFailed("it returned no answer")

        messages.append(model.assistant_turn(text, thinking, done))

        if writing and not asked_again and not document_body(text) and not document_body(draft):
            asked_again = True
            messages.append({"role": "user", "content": (
                "(From the app, not the reader.) That reply was empty. Write the document now.")})
            continue

        if writing:
            pending_now, pending = pending, None
            calls_seen += 1
            call_id = f"build-{calls_seen}"
            yield "tool_start", {"id": call_id, "name": pending_now.tool, "arguments": {}}
            if done.done_reason in CUT_OFF:
                outcome = Outcome(False, f"{pending_now.tool} failed: the document hit the length limit. "
                                  "Call the tool again and write something shorter.", "The document was too long")
            elif not (body := document_body(text) or document_body(draft)):
                outcome = Outcome(False, f"{pending_now.tool} failed: you wrote nothing.", "Nothing was written")
            else:
                outcome = await box.build(pending_now, body)
            for event in tool_events(call_id, pending_now.tool, outcome, finished):
                yield event
            report = outcome.text
            if not outcome.ok and len(finished.answer) > document_start:
                # The failed document leaves the answer, or a retry would show it twice.
                withdrawn = (document_start, finished.answer[document_start:])
                finished.answer = finished.answer[:document_start]
                yield "withdraw", {"keep": document_start}
                report += (f" Your document was taken out of the chat, so if you call {pending_now.tool} "
                           "again, write it in full.")
            elif outcome.ok:
                # A rebuild (shorter, fixed) replaces its file, so its earlier text leaves the answer too.
                start = document_start
                if earlier := shown_documents.get(pending_now.tool):
                    finished.answer = finished.answer[:earlier[0]] + finished.answer[earlier[1]:]
                    yield "withdraw", {"keep": earlier[0], "end": earlier[1]}
                    start -= earlier[1] - earlier[0]
                shown_documents[pending_now.tool] = (start, len(finished.answer))
            messages.append({"role": "user", "content": f"(From the app, not the reader.) {report}"})
            continue

        if done.done_reason == BAD_CALL:
            messages.append({"role": "user", "content": (
                "(From the app, not the reader.) Your tool call couldn't be read. Call the tool "
                "again with only a file name, or short arguments.")})
            continue

        if not done.tool_calls or done.done_reason in CUT_OFF:
            break

        outcomes = []
        for call in done.tool_calls:
            calls_seen += 1
            call_id = f"call-{calls_seen}"
            yield "tool_start", {"id": call_id, "name": call.name, "arguments": call.arguments}
            if pending is not None:
                outcome = Outcome(False, "Finish the file you've started first.", "One file at a time")
            else:
                outcome = await box.call(call.name, call.arguments)
            outcomes.append(outcome)
            if outcome.pending:
                pending, draft, document_start = outcome.pending, text, round_start
            if outcome.hits:
                citations.provided.update(range(1, len(box.hits) + 1))
                yield "more_sources", {"sources": outcome.detail["sources"]}
            if "proposal" in outcome.detail:
                yield "web_proposal", outcome.detail["proposal"]
            if "web" in outcome.detail:
                yield "web_source", web_source(box, outcome.detail["web"])
            for event in tool_events(call_id, call.name, outcome, finished):
                yield event
        messages.extend(model.tool_results(done.tool_calls, outcomes))

    # Nothing came after a failed document: it goes back, rather than leave the reader without it.
    if withdrawn and not finished.answer[withdrawn[0]:].strip():
        finished.answer = finished.answer[:withdrawn[0]] + withdrawn[1]
        yield "withdraw", {"keep": withdrawn[0]}
        yield "token", {"text": withdrawn[1]}

    # A turn that only made a file can stand on the tool lines ("Wrote notes.md"). Any other
    # turn with no text has no answer: "Found 2 more passages" isn't one (Phase 5d,
    # deriv-006), so it ends as an error, which offers Ask again in Deep.
    if not finished.answer.strip() and finished.files and finished.done_reason not in CUT_OFF:
        finished.answer = "\n".join(item["summary"] for item in finished.tools if item["summary"])
        if finished.answer:
            yield "token", {"text": finished.answer}
    yield "finished", {"result": finished}


def web_source(box: ToolBox, number: int) -> dict:
    page = box.web.pages[number - 1]
    return {"number": page.number, "title": page.title, "url": page.final_url, "fetched": page.fetched}


def tool_events(call_id: str, name: str, outcome: Outcome, finished: Finished) -> list[Event]:
    finished.tools.append({"name": name, "ok": outcome.ok, "summary": outcome.summary})
    replaced = {card["name"] for card in outcome.files}
    finished.files = [card for card in finished.files if card["name"] not in replaced] + outcome.files
    return [("tool_end", {"id": call_id, "name": name, "ok": outcome.ok, "summary": outcome.summary,
                          "result": outcome.text, "detail": outcome.detail}),
            *(("file", card) for card in outcome.files)]
