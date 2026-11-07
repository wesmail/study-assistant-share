"""The tool loop with canned models: what each reply can do, and what goes back."""
import asyncio
from dataclasses import replace
import json
from types import SimpleNamespace as Fake

import httpx
import pytest
from fastapi.testclient import TestClient

from app.chat import answer_events, exportable_answer
from app.config import Settings
from app.db.connection import connect
from app.ingest.embed import serialize
from app.llm import hosted
from app.llm.citations import CitationFilter
from app.llm.hosted import request_for
from app.llm.ollama import Delta, ToolCall, chat_payload, tool_call_of
from app.llm.prompts import SYSTEM_PROMPT, TOOLS_PROMPT
from app.main import create_app
from app.tools.executor import Outcome, ToolBox
from app.tools.loop import (
    Finished, HostedModel, LocalModel, ModelFailed, document_body, run_turn,
)
from app.tools.schemas import anthropic_tools, ollama_tools


@pytest.fixture
def settings(tmp_path):
    made = Settings(data_dir=tmp_path / "data", library_dir=tmp_path / "library",
                    outputs_dir=tmp_path / "outputs")
    connect(made.db_path).close()
    return made


class Scripted:
    """A model that plays one list of deltas per reply, and notes what it was asked."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.allowed = []
        self.seen = []

    async def stream(self, messages, allow_tools):
        self.allowed.append(allow_tools)
        self.seen.append(list(messages))
        for delta in self.replies.pop(0):
            yield delta

    def assistant_turn(self, text, thinking, done):
        return {"role": "assistant", "content": text}

    def tool_results(self, calls, outcomes):
        return [{"role": "tool", "content": outcome.text} for outcome in outcomes]


def says(text, reason="stop"):
    return [Delta(text=text), Delta(done=True, done_reason=reason, output_tokens=5, cost_usd=0.01)]


def calls(*pairs, reason="stop"):
    return [Delta(done=True, done_reason=reason, tool_calls=[
        ToolCall(f"id-{number}", name, arguments) for number, (name, arguments) in enumerate(pairs)])]


def turn(settings, model, sources=None, previous=None, strict=False, provided=(), hits=None):
    box = ToolBox(settings, lambda query: [], sources or {}, hits or [], previous)
    citations = CitationFilter(set(provided))
    messages = [{"role": "user", "content": "Question"}]

    async def run():
        return [item async for item in run_turn(model, messages, box, citations, settings, strict)]
    return asyncio.run(run()), messages


def replayed(events) -> str:
    """The answer as the browser builds it from the stream."""
    shown = ""
    for name, data in events:
        if name == "token":
            shown += data["text"]
        elif name == "withdraw":
            shown = shown[:data["keep"]] + (shown[data["end"]:] if "end" in data else "")
    return shown


def finished_of(events) -> Finished:
    return dict(events)["finished"]["result"]


class TestAFile:
    def script(self):
        return Scripted(calls(("make_markdown", {"filename": "notes"})), says("# Notes\n\nWords."), says("Done."))

    def test_it_takes_three_replies_and_the_second_has_tools_off(self, settings):
        model = self.script()
        turn(settings, model)
        assert model.allowed == [True, False, True]

    def test_the_file_is_the_second_reply_exactly(self, settings):
        turn(settings, self.script())
        assert (settings.outputs_dir / "notes.md").read_text() == "# Notes\n\nWords.\n"

    def test_the_model_is_told_to_write_then_told_it_worked(self, settings):
        model = self.script()
        turn(settings, model)
        assert "Write the document now" in model.seen[1][-1]["content"]
        assert "saved notes.md" in model.seen[2][-1]["content"]

    def test_the_reader_sees_the_document_a_card_and_the_closing_note(self, settings):
        events, _ = turn(settings, self.script())
        finished = finished_of(events)
        assert finished.answer == "# Notes\n\nWords.\n\nDone."
        assert [card["name"] for name, card in events if name == "file"] == ["notes.md"]
        assert [item["ok"] for item in finished.tools] == [True, True]

    def test_a_failed_build_goes_back_to_the_model_and_it_can_try_again(self, settings):
        model = Scripted(calls(("make_pptx", {"filename": "d"})), says("No headings here."),
                         calls(("make_markdown", {"filename": "d"})), says("# Fixed"), says("Done."))
        events, _ = turn(settings, model)
        assert "no slides" in model.seen[2][-1]["content"]
        assert (settings.outputs_dir / "d.md").exists()

    def test_a_failed_document_leaves_the_answer_so_a_retry_shows_once(self, settings):
        """The bug report of Nov 7: the summary appeared twice, the failed draft and the retry."""
        model = Scripted(calls(("make_pptx", {"filename": "d"})), says("Draft one, with no slides."),
                         calls(("make_markdown", {"filename": "d"})), says("# Summary\n\nDraft two."), says("Done."))
        events, _ = turn(settings, model)
        finished = finished_of(events)
        assert finished.answer == "# Summary\n\nDraft two.\n\nDone."
        assert [data for name, data in events if name == "withdraw"] == [{"keep": 0}]
        assert "write it in full" in model.seen[2][-1]["content"]

    def test_text_before_the_failed_document_stays(self, settings):
        model = Scripted([Delta(text="I'll make it."), *calls(("make_pptx", {"filename": "d"}))],
                         says("No slides here."), says("It failed, sorry."))
        events, _ = turn(settings, model)
        assert finished_of(events).answer == "It failed, sorry."
        shown = ""
        for name, data in events:
            shown = shown + data["text"] if name == "token" else shown[:data["keep"]] if name == "withdraw" else shown
        assert shown == finished_of(events).answer

    def test_a_failed_document_comes_back_if_nothing_follows_it(self, settings):
        model = Scripted(calls(("make_pptx", {"filename": "d"})), says("No slides here."), says(""))
        events, _ = turn(settings, model)
        assert finished_of(events).answer == "No slides here."
        shown = ""
        for name, data in events:
            shown = shown + data["text"] if name == "token" else shown[:data["keep"]] if name == "withdraw" else shown
        assert shown == "No slides here."

    def test_a_rebuilt_file_shows_only_its_latest_document(self, settings):
        """Run 2 of pdf-paper-figure built four times to get to one page: the answer held every draft."""
        model = Scripted(calls(("make_markdown", {"filename": "s"})), says("# S\n\nLong draft."),
                         [Delta(text="Too long, shortening."), *calls(("make_markdown", {"filename": "s"}))],
                         says("# S\n\nShort."), says("Done."))
        events, _ = turn(settings, model)
        answer = finished_of(events).answer
        assert answer.count("# S") == 1 and "Long draft" not in answer and "Short." in answer
        assert replayed(events) == answer

    def test_two_different_files_keep_both_documents(self, settings):
        model = Scripted(calls(("make_markdown", {"filename": "a"})), says("# A doc"),
                         calls(("make_pptx", {"filename": "b"})), says("No slides here."), says("Done."))
        events, _ = turn(settings, model)
        assert "# A doc" in finished_of(events).answer
        assert replayed(events) == finished_of(events).answer

    def test_a_document_cut_off_at_the_limit_is_not_built(self, settings):
        model = Scripted(calls(("make_markdown", {"filename": "d"})), says("# Half", "length"), says("Sorry."))
        turn(settings, model)
        assert not (settings.outputs_dir / "d.md").exists()
        assert "length limit" in model.seen[2][-1]["content"]

    def test_an_empty_document_is_not_built(self, settings):
        model = Scripted(calls(("make_markdown", {"filename": "d"})), says(""), says(""), says("Sorry."))
        turn(settings, model)
        assert not (settings.outputs_dir / "d.md").exists()

    def test_an_empty_document_gets_one_more_try_with_tools_still_off(self, settings):
        model = Scripted(calls(("make_markdown", {"filename": "d"})), says(""), says("# Second try"), says("Done."))
        turn(settings, model)
        assert model.allowed == [True, False, False, True]
        assert (settings.outputs_dir / "d.md").read_text() == "# Second try\n"

    @pytest.mark.parametrize("replies", [[says("# Half", "length")], [says(""), says("")]], ids=["cut off", "empty"])
    def test_a_document_that_cant_be_built_is_reported_not_silently_dropped(self, settings, replies):
        model = Scripted(calls(("make_markdown", {"filename": "d"})), *replies, says("Sorry."))
        events, _ = turn(settings, model)
        assert [item["ok"] for item in finished_of(events).tools] == [True, False]
        assert [name for name, _ in events].count("tool_end") == 2

    def test_a_document_written_alongside_the_call_is_used_if_the_next_reply_is_empty(self, settings):
        """gpt-oss sometimes does both at once, against the usual pattern."""
        both = [Delta(text="# Early"), *calls(("make_markdown", {"filename": "d"}))]
        model = Scripted(both, says(""), says("Done."))
        turn(settings, model)
        assert (settings.outputs_dir / "d.md").read_text() == "# Early\n"

    def test_the_reply_written_for_the_purpose_wins_over_an_earlier_one(self, settings):
        both = [Delta(text="Let me write it."), *calls(("make_markdown", {"filename": "d"}))]
        model = Scripted(both, says("# Real"), says("Done."))
        turn(settings, model)
        assert (settings.outputs_dir / "d.md").read_text() == "# Real\n"

    def test_a_retry_leaves_one_card_in_the_turns_total(self, settings):
        model = Scripted(calls(("make_markdown", {"filename": "a"})), says("# Long"),
                         calls(("make_markdown", {"filename": "a"})), says("# Short"), says("Done."))
        events, _ = turn(settings, model)
        assert [card["name"] for card in finished_of(events).files] == ["a.md"]

    def test_two_files_in_one_reply_are_refused_one_at_a_time(self, settings):
        model = Scripted(calls(("make_markdown", {"filename": "a"}), ("make_pdf", {"filename": "b"})),
                         says("# A"), says("Done."))
        events, _ = turn(settings, model)
        assert [item["ok"] for item in finished_of(events).tools][:2] == [True, False]

    def test_an_export_builds_at_once_with_no_writing_step(self, settings):
        model = Scripted(calls(("make_markdown", {"filename": "a", "export_previous_answer": True})), says("Done."))
        turn(settings, model, previous="# Old answer")
        assert model.allowed == [True, True]
        assert (settings.outputs_dir / "a.md").read_text() == "# Old answer\n"

    def test_invented_citations_dont_reach_the_file(self, settings):
        model = Scripted(calls(("make_markdown", {"filename": "a"})), says("Real [S1] and fake [S9]."), says("Done."))
        turn(settings, model, provided=[1], sources={1: {"doc_title": "Book", "page_start": 1, "page_end": 1}})
        text = (settings.outputs_dir / "a.md").read_text()
        assert "[S1]" in text and "[S9]" not in text


class TestUnreadableCalls:
    def model(self, settings, *bodies):
        replies = iter(bodies)
        return LocalModel(settings, httpx.MockTransport(lambda request: httpx.Response(200, content=next(replies))))

    def test_a_malformed_call_goes_back_to_the_model_instead_of_ending_the_turn(self, settings):
        broken = b'{"error": "error parsing tool call: raw=\'{bad\'"}\n'
        fine = b'{"message": {"content": "Answered."}, "done": true, "done_reason": "stop"}\n'
        events, messages = turn(settings, self.model(settings, broken, fine))
        assert finished_of(events).answer == "Answered."
        assert any("couldn't be read" in str(message["content"]) for message in messages)

    def test_any_other_ollama_error_still_stops_the_turn(self, settings):
        with pytest.raises(RuntimeError, match="out of memory"):
            turn(settings, self.model(settings, b'{"error": "out of memory"}\n'))


class TestDocumentBody:
    def test_a_short_preamble_goes(self):
        assert document_body("Here you go:\n\n# Title\n\nText") == "# Title\n\nText"

    def test_a_fence_around_everything_goes(self):
        assert document_body("```markdown\n# Title\n```") == "# Title"

    def test_real_text_before_the_first_heading_stays(self):
        text = "Intro with $x$ in it.\n\n# Title"
        assert document_body(text) == text

    def test_a_document_with_no_headings_stays(self):
        assert document_body("Just text.") == "Just text."


class TestTheLoopEnds:
    def test_a_reply_with_no_tool_call_ends_the_turn(self, settings):
        model = Scripted(says("An answer."))
        events, _ = turn(settings, model)
        assert finished_of(events).answer == "An answer."
        assert model.replies == []

    def test_a_model_that_keeps_calling_tools_is_stopped(self, settings):
        settings.tool_rounds = 3
        model = Scripted(calls(("search_library", {"query": "a"})), calls(("search_library", {"query": "b"})),
                         says("Stopped."))
        events, _ = turn(settings, model)
        assert model.allowed == [True, True, False]
        assert finished_of(events).answer == "Stopped."

    def test_a_reply_cut_off_mid_call_runs_no_tool(self, settings):
        model = Scripted(calls(("make_markdown", {"filename": "a"}), reason="max_tokens"))
        events, _ = turn(settings, model)
        assert finished_of(events).tools == []

    def test_an_empty_last_reply_still_tells_the_reader_what_happened(self, settings):
        model = Scripted(calls(("make_markdown", {"filename": "a"})), says("# Doc"), says(""))
        model.replies[1].insert(0, Delta(text=""))
        events, _ = turn(settings, model)
        assert "# Doc" in finished_of(events).answer

    def test_tool_lines_alone_are_not_an_answer(self, settings):
        # Phase 5d, deriv-006: two library searches, then thinking ran out; the reply was
        # "Found 1 more passage / Found 2 more passages".
        for last in (says("", reason="length"), says("")):
            model = Scripted(calls(("search_library", {"query": "sampling rate"})), last)
            events, _ = turn(settings, model)
            assert finished_of(events).answer == ""
            assert "token" not in dict(events)

    def test_totals_add_up_over_rounds(self, settings):
        model = Scripted(calls(("make_markdown", {"filename": "a"})), says("# Doc"), says("Done."))
        events, _ = turn(settings, model)
        assert finished_of(events).cost_usd == pytest.approx(0.02)


class TestStrictMode:
    def test_a_refusal_runs_no_tools_and_raises(self, settings):
        model = Scripted(calls(("make_markdown", {"filename": "a"}), reason="refusal"))
        with pytest.raises(ModelFailed, match="refused"):
            turn(settings, model, strict=True)
        assert not settings.outputs_dir.exists()

    def test_nothing_at_all_raises(self, settings):
        with pytest.raises(ModelFailed, match="no answer"):
            turn(settings, Scripted(says("")), strict=True)

    def test_a_tool_call_with_no_text_is_a_real_reply(self, settings):
        model = Scripted(calls(("search_library", {"query": "a"})), says("Fine."))
        events, _ = turn(settings, model, strict=True)
        assert finished_of(events).answer == "Fine."


class TestOllama:
    def test_a_full_context_stops_the_reply_instead_of_dropping_the_sources(self, settings):
        assert chat_payload([], settings)["shift"] is False

    def test_tools_go_in_the_payload_only_when_allowed(self, settings):
        assert "tools" not in chat_payload([], settings)
        assert chat_payload([], settings, ollama_tools())["tools"][0]["type"] == "function"

    def test_arguments_can_be_a_json_string(self):
        call = tool_call_of({"function": {"name": "x", "arguments": '{"a": 1}'}}, 1)
        assert (call.id, call.arguments) == ("call-1", {"a": 1})

    def test_broken_arguments_become_a_validation_failure_not_a_crash(self):
        assert tool_call_of({"function": {"name": "x", "arguments": "{broken"}}, 1).arguments == {"invalid_json": "{broken"}

    def test_a_streamed_tool_call_reaches_the_last_delta(self, settings):
        lines = [{"message": {"tool_calls": [{"function": {"name": "run_python", "arguments": {"code": "1"}}}]}, "done": False},
                 {"message": {"content": ""}, "done": True, "done_reason": "stop"}]
        body = b"".join(json.dumps(line).encode() + b"\n" for line in lines)
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))
        model = LocalModel(settings, transport)

        async def run():
            return [delta async for delta in model.stream([], True)]
        deltas = asyncio.run(run())
        assert [call.name for call in deltas[-1].tool_calls] == ["run_python"]

    def test_a_result_is_a_tool_message_with_the_tools_name(self, settings):
        results = LocalModel(settings, None).tool_results([ToolCall("1", "run_python", {})], [Outcome(True, "out")])
        assert results == [{"role": "tool", "tool_name": "run_python", "content": "out"}]

    def test_thinking_and_calls_are_sent_back(self, settings):
        turn_ = LocalModel(settings, None).assistant_turn("", "hm", Delta(tool_calls=[ToolCall("1", "x", {"a": 1})]))
        assert turn_["thinking"] == "hm"
        assert turn_["tool_calls"][0]["function"] == {"name": "x", "arguments": {"a": 1}}


class TestAnthropic:
    MESSAGES = [{"role": "system", "content": "Rules"}, {"role": "user", "content": "Q"}]

    def test_plain_requests_are_unchanged(self, settings):
        request = request_for(self.MESSAGES, settings)
        assert "tools" not in request and "cache_control" not in request and "tool_choice" not in request

    def test_a_tool_turn_is_cached(self, settings):
        assert request_for(self.MESSAGES, settings, anthropic_tools())["cache_control"] == {"type": "ephemeral"}

    def test_the_writing_step_keeps_tools_listed_but_switched_off(self, settings):
        request = request_for(self.MESSAGES, settings, anthropic_tools(), allow_tools=False)
        assert request["tool_choice"] == {"type": "none"} and request["tools"]

    def test_forced_tool_choice_is_never_sent(self, settings):
        for allow in (True, False):
            choice = request_for(self.MESSAGES, settings, anthropic_tools(), allow).get("tool_choice", {})
            assert choice.get("type") not in ("any", "tool")

    def test_thinking_blocks_go_back_unchanged(self, settings):
        blocks = [Fake(type="thinking"), Fake(type="tool_use", id="t", name="x", input={})]
        turn_ = HostedModel(settings, "chat").assistant_turn("", "", Delta(raw=blocks))
        assert turn_["content"] is blocks

    def test_results_share_one_user_message_and_failures_are_marked(self, settings):
        calls_ = [ToolCall("a", "x", {}), ToolCall("b", "y", {})]
        results = HostedModel(settings, "chat").tool_results(calls_, [Outcome(True, "ok"), Outcome(False, "bad")])
        assert len(results) == 1 and results[0]["role"] == "user"
        assert [(r["tool_use_id"], r["is_error"]) for r in results[0]["content"]] == [("a", False), ("b", True)]


def usage():
    return Fake(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0)


class FakeHosted:
    """A hosted client whose streams play in turn, one for each request."""

    def __init__(self, *finals):
        self.finals = list(finals)
        self.requests = []
        self.messages = self

    def stream(self, **request):
        self.requests.append(request)
        final = self.finals.pop(0)

        class Stream:
            async def __aenter__(stream):
                if isinstance(final, Exception):
                    raise final
                return stream

            async def __aexit__(stream, *details):
                return False

            async def __aiter__(stream):
                for block in final.content:
                    if block.type == "text":
                        yield Fake(type="content_block_delta", delta=Fake(type="text_delta", text=block.text))

            async def get_final_message(stream):
                return final
        return Stream()


def message(*blocks, stop="end_turn"):
    return Fake(content=list(blocks), stop_reason=stop, usage=usage())


def text_block(words):
    return Fake(type="text", text=words)


def use_block(name, arguments):
    return Fake(type="tool_use", id=f"toolu-{name}", name=name, input=arguments)


def embed_nothing(texts, settings):
    return [[1.0] + [0.0] * 1023 for _ in texts]


@pytest.fixture
def library(settings):
    settings.library_dir.mkdir(parents=True, exist_ok=True)
    (settings.library_dir / "metadata.yaml").write_text("a.pdf:\n  title: Book\n")
    conn = connect(settings.db_path)
    conn.execute("INSERT INTO documents (id, path, sha256, title, kind, pages, status) "
                 "VALUES (1, 'a.pdf', 'x', 'Book', 'pdf', 9, 'indexed')")
    conn.execute("INSERT INTO chunks (id, doc_id, kind, seq, heading_path, text, token_count, block_ids, "
                 "page_start, page_end) VALUES (1, 1, 'text', 1, '[]', 'The Cholesky decomposition', 4, '[]', 2, 2)")
    conn.execute("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (1, ?)", (serialize([1.0] + [0.0] * 1023, 1024),))
    conn.commit()
    conn.close()
    return settings


def answer(settings, client, question="Cholesky?", history=(), cited=()):
    async def run():
        return [item async for item in answer_events(
            question, list(history), settings, embed=embed_nothing, provider="anthropic", tools=True,
            client=client, cited=cited)]
    return asyncio.run(run())


def rewritten(request):
    """The helper that rewrites follow-ups, on the local model even in deep mode."""
    return httpx.Response(200, content=b'{"message": {"content": "Next?"}, "done": true}\n')


class TestDeepTurns:
    def test_a_plain_answer_passes_straight_through(self, library):
        events = answer(library, FakeHosted(message(text_block("Yes [S1]."))))
        done = dict(events)["done"]
        assert (done["answer"], done["citations"], done["files"]) == ("Yes [S1].", [1], [])

    def test_the_system_prompt_gains_the_tool_rules_only_with_tools(self, library):
        client = FakeHosted(message(text_block("Yes.")))
        answer(library, client)
        assert client.requests[0]["system"] == f"{SYSTEM_PROMPT}\n\n{TOOLS_PROMPT}"

    def test_the_last_turn_goes_with_the_question(self, library):
        client = FakeHosted(message(text_block("Yes.")))
        answer(library, client, history=[{"role": "user", "content": "Before"},
                                         {"role": "assistant", "content": "Earlier reply"}])
        assert [turn_["role"] for turn_ in client.requests[0]["messages"]] == ["user", "assistant", "user"]

    def test_the_api_sends_nothing_older_than_the_last_turn_with_tools_on(self, library, monkeypatch):
        """Through the real endpoint, with three turns stored."""
        seen = []

        async def capture(messages, settings, client=None, purpose="chat", tools=None, allow_tools=True):
            seen.append(list(messages))
            yield Delta(text="Yes.")
            yield Delta(done=True, done_reason="end_turn", model="claude-opus-5-5", raw=[])
        monkeypatch.setattr(hosted, "stream_chat", capture)
        conn = connect(library.db_path)
        conn.execute("INSERT INTO conversations (id, title) VALUES (1, 'x')")
        for role, words in [("user", "OLDEST question"), ("assistant", "OLDEST answer"),
                            ("user", "Recent question"), ("assistant", "Recent answer")]:
            conn.execute("INSERT INTO messages (conversation_id, role, content) VALUES (1, ?, ?)", (role, words))
        conn.commit()
        conn.close()
        app = create_app(library, transport=httpx.MockTransport(rewritten), embed=embed_nothing)
        TestClient(app, headers={"host": "localhost"}).post(
            "/api/chat", json={"message": "Next?", "conversation_id": 1, "mode": "deep"})
        sent = json.dumps(seen[0])
        assert "Recent answer" in sent and "OLDEST" not in sent

    def test_tool_results_go_back_to_the_model_in_deep_mode(self, library):
        client = FakeHosted(message(use_block("search_library", {"query": "z"}), stop="tool_use"),
                            message(text_block("Done.")))
        answer(library, client)
        last = client.requests[1]["messages"][-1]
        assert last["role"] == "user" and last["content"][0]["type"] == "tool_result"

    def test_the_cost_of_every_round_is_logged(self, library):
        client = FakeHosted(message(use_block("search_library", {"query": "z"}), stop="tool_use"),
                            message(text_block("Done.")))
        done = dict(answer(library, client))["done"]
        conn = connect(library.db_path)
        logged = conn.execute("SELECT count(*), sum(cost_usd) FROM hosted_calls").fetchone()
        conn.close()
        assert logged[0] == 2 and done["cost_usd"] == pytest.approx(logged[1])

    def test_a_refusal_restarts_the_whole_turn_on_the_fallback(self, library):
        client = FakeHosted(message(stop="refusal"), message(text_block("From Sonnet.")))
        events = answer(library, client)
        names = [name for name, _ in events]
        assert names.index("fallback") < names.index("done")
        assert dict(events)["fallback"]["reason"] == "it refused"
        done = dict(events)["done"]
        assert (done["model"], done["answer"], done["fallback_from"]) == (
            "claude-sonnet-5-5", "From Sonnet.", "claude-opus-5-5")
        assert [request["model"] for request in client.requests] == ["claude-opus-5-5", "claude-sonnet-5-5"]

    def test_a_failure_after_a_tool_starts_again_from_the_first_step(self, library):
        client = FakeHosted(message(use_block("search_library", {"query": "z"}), stop="tool_use"),
                            hosted.HostedFailure("Hosted model: HTTP 529, overloaded_error"),
                            message(text_block("Start over.")))
        client.messages = client
        original = hosted.stream_chat

        async def failing(*args, **kwargs):
            if isinstance(client.finals[0], Exception):
                client.finals.pop(0)
                raise hosted.HostedFailure("Hosted model: HTTP 529, overloaded_error")
            async for delta in original(*args, **kwargs):
                yield delta
        hosted.stream_chat, saved = failing, hosted.stream_chat
        try:
            done = dict(answer(library, client))["done"]
        finally:
            hosted.stream_chat = saved
        assert done["answer"] == "Start over."
        assert client.requests[-1]["messages"][-1]["role"] == "user"
        assert isinstance(client.requests[-1]["messages"][-1]["content"], str)     # the question, not a tool result

    def test_a_rejected_key_doesnt_try_the_fallback(self, library):
        async def rejected(*args, **kwargs):
            raise hosted.HostedFailure("Hosted model: HTTP 401, authentication_error", key_rejected=True)
            yield
        saved, hosted.stream_chat = hosted.stream_chat, rejected
        try:
            events = answer(library, FakeHosted())
        finally:
            hosted.stream_chat = saved
        names = [name for name, _ in events]
        assert "fallback" not in names and "error" in names


class TestFastTurns:
    def test_the_local_model_is_offered_tools_and_answers_normally(self, library):
        seen = []

        def reply(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, content=json.dumps(
                {"message": {"content": "Yes."}, "done": True, "done_reason": "stop"}).encode() + b"\n")

        async def run():
            return [item async for item in answer_events(
                "Cholesky?", [], library, embed=embed_nothing, tools=True,
                transport=httpx.MockTransport(reply))]
        done = dict(asyncio.run(run()))["done"]
        assert done["answer"] == "Yes." and done["hosted"] is False
        assert len(seen[0]["tools"]) == 7


    def test_a_reply_of_only_tool_lines_is_an_error_that_offers_deep(self, library):
        replies = [{"message": {"tool_calls": [{"function": {"name": "search_library",
                                                             "arguments": {"query": "Hz"}}}]},
                    "done": True, "done_reason": "stop"},
                   {"message": {"content": "", "thinking": "Let me think. " * 20}, "done": True,
                    "done_reason": "length"}]

        def reply(request):
            return httpx.Response(200, content=json.dumps(replies.pop(0)).encode() + b"\n")

        async def run():
            return [item async for item in answer_events(
                "What is the sampling rate?", [], replace(library, chat_num_predict=16384), embed=embed_nothing,
                tools=True, transport=httpx.MockTransport(reply))]
        events = asyncio.run(run())
        names = [name for name, _ in events]
        assert "done" not in names and "token" not in names
        error = dict(events)["error"]
        assert error["cut_off"] and error["no_answer"] and "16,384-token limit" in error["message"]


class TestOutputFiles:
    @pytest.fixture
    def client(self, library):
        library.outputs_dir.mkdir()
        (library.outputs_dir / "notes.pdf").write_bytes(b"%PDF-1.7")
        (library.outputs_dir / "page.html").write_text("<script>alert(1)</script>")
        return TestClient(create_app(library, embed=embed_nothing), headers={"host": "localhost"})

    def test_a_file_is_served_inline_with_its_type(self, client):
        response = client.get("/api/outputs/notes.pdf")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.headers["content-disposition"].startswith("inline")
        assert response.headers["x-content-type-options"] == "nosniff"

    def test_download_asks_the_browser_to_save_it(self, client):
        assert client.get("/api/outputs/notes.pdf?download=true").headers["content-disposition"].startswith("attachment")

    @pytest.mark.parametrize("name", [
        "missing.pdf", "..%2F..%2F.env", "..%2fsecret.pdf", ".env", "page.html",
    ])
    def test_anything_else_is_a_404(self, client, name):
        assert client.get(f"/api/outputs/{name}").status_code == 404

    def test_a_link_in_outputs_isnt_followed(self, client, library, tmp_path):
        secret = tmp_path / "secret.txt"
        secret.write_text("KEY=value")
        (library.outputs_dir / "linked.txt").symlink_to(secret)
        assert client.get("/api/outputs/linked.txt").status_code == 404


class TestRequestsThatNeedNoLibrary:
    def run(self, library, question, history=(), exportable=None, cited=()):
        client = FakeHosted(message(text_block("Done.")))

        async def collect():
            return [item async for item in answer_events(
                question, list(history), library, embed=embed_nothing, provider="anthropic", tools=True, client=client,
                cited=cited, exportable=exportable)]
        return asyncio.run(collect()), client

    TURN = [{"role": "user", "content": "What is Cholesky?"}, {"role": "assistant", "content": "A factorization."}]

    @pytest.mark.parametrize("question, history", [
        ("Export this answer as a PDF.", TURN),
        ("Run this code.\n\n```python\nprint(1)\n```", []),
        ("Check with SymPy that x + x = 2*x.", []),
    ])
    def test_no_book_text_is_sent_for_them(self, library, question, history):
        events, client = self.run(library, question, history)
        sent = json.dumps(client.requests[0]["messages"], default=str)
        assert "The Cholesky decomposition" not in sent
        assert "wasn't searched" in sent
        assert dict(events)["sources"]["sources"] == []

    def test_a_real_question_is_still_searched(self, library):
        _, client = self.run(library, "Tell me about Cholesky.")
        assert "The Cholesky decomposition" in json.dumps(client.requests[0]["messages"], default=str)

    def test_a_deck_with_code_still_needs_the_library(self, library):
        _, client = self.run(library, "Make a deck on Cholesky. Run code that plots it.")
        assert "The Cholesky decomposition" in json.dumps(client.requests[0]["messages"], default=str)

    def test_the_model_can_still_search_when_it_decides_to(self, library):
        client = FakeHosted(message(use_block("search_library", {"query": "z"}), stop="tool_use"),
                            message(text_block("Done.")))

        async def collect():
            return [item async for item in answer_events(
                "Check with SymPy that the book's formula holds.", [], library, embed=embed_nothing,
                provider="anthropic", tools=True, client=client)]
        events = asyncio.run(collect())
        assert "more_sources" in [name for name, _ in events]


class TestExportingTwice:
    ROWS = [
        {"role": "user", "content": "What is Cholesky?", "citations": None},
        {"role": "assistant", "content": "The real answer [S1].", "citations": '[{"id": 1, "doc_title": "Book"}]'},
        {"role": "user", "content": "Export this answer as a PDF.", "citations": None},
        {"role": "assistant", "content": "I saved it as a PDF.", "citations": "[]"},
    ]

    def test_the_second_export_finds_the_real_answer_not_the_note(self):
        assert exportable_answer(self.ROWS) == ("The real answer [S1].", [{"id": 1, "doc_title": "Book"}])

    def test_with_no_export_it_is_simply_the_last_answer(self):
        assert exportable_answer(self.ROWS[:2])[0] == "The real answer [S1]."

    def test_nothing_to_export_is_none(self):
        assert exportable_answer([]) is None

    def test_through_the_api_the_word_file_holds_the_answer(self, library, monkeypatch):
        seen = []

        async def replies(messages, settings, client=None, purpose="chat", tools=None, allow_tools=True):
            seen.append(1)
            if len(seen) == 1:
                yield Delta(done=True, done_reason="tool_use", model="m", raw=[],
                            tool_calls=[ToolCall("1", "make_markdown", {"filename": "notes"})])
                return
            yield Delta(text="Saved.")
            yield Delta(done=True, done_reason="end_turn", model="m", raw=[])
        monkeypatch.setattr(hosted, "stream_chat", replies)
        conn = connect(library.db_path)
        conn.execute("INSERT INTO conversations (id, title) VALUES (1, 'x')")
        for row in self.ROWS:
            conn.execute("INSERT INTO messages (conversation_id, role, content, citations) VALUES (1, ?, ?, ?)",
                         (row["role"], row["content"], row["citations"]))
        conn.commit()
        conn.close()
        app = create_app(library, embed=embed_nothing)
        TestClient(app, headers={"host": "localhost"}).post(
            "/api/chat", json={"message": "Export this answer as Markdown.", "conversation_id": 1, "mode": "deep"})
        assert (library.outputs_dir / "notes.md").read_text().startswith("The real answer [S1].")
