"""The OpenRouter client the Qwen option uses, with the network replaced by canned streams."""
import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from app.chat import answer_events
from app.config import AnswerOption, Settings
from app.db.connection import connect
from app.ingest.embed import serialize
from app.llm import openrouter
from app.llm.hosted import Tokens, record_call
from app.llm.ollama import Delta
from app.main import create_app

KEY = "sk-or-v1-test-secret-0123456789"
QWEN = AnswerOption("qwen", "Qwen", "openrouter", "qwen/qwen3.8-27b", route="DeepInfra", quantization="bf16",
                    sampling={"temperature": 1, "top_k": 20}, reasoning={"enabled": True})
OPTIONS = [AnswerOption("fast", "Fast", "ollama", "gpt-oss:20b", think="medium"), QWEN,
           AnswerOption("deep", "Deep", "anthropic", "claude-opus-5-5", fallback="claude-sonnet-5-5")]


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(openrouter, "openrouter_api_key", lambda: KEY)
    made = Settings(data_dir=tmp_path, library_dir=tmp_path, tools_enabled=False, answer_options=list(OPTIONS))
    connect(made.db_path).close()
    return QWEN.apply(made)


def sse(*events, provider="DeepInfra"):
    lines = [": OPENROUTER PROCESSING"]
    lines += [f"data: {json.dumps({'provider': provider, 'model': 'qwen/qwen3.8-27b', **event})}" for event in events]
    return ("\n\n".join(lines + ["data: [DONE]"]) + "\n\n").encode()


def delta(content=None, reasoning=None, finish=None, tool_calls=None):
    piece = {k: v for k, v in (("content", content), ("reasoning", reasoning), ("tool_calls", tool_calls))
             if v is not None}
    return {"choices": [{"delta": piece, "finish_reason": finish}]}


def usage(prompt=1000, completion=400, cost=0.00123):
    return {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "cost": cost}}


class Fake:
    """Answers each request with the next body; keeps what was sent."""

    def __init__(self, *bodies: bytes, status: int = 200):
        self.bodies = list(bodies)
        self.status = status
        self.requests: list[httpx.Request] = []

    def client(self) -> httpx.AsyncClient:
        def handler(request):
            self.requests.append(request)
            return httpx.Response(self.status, content=self.bodies.pop(0))
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def sent(self, number=0) -> dict:
        return json.loads(self.requests[number].content)


def collect(settings, fake, messages=None, **options):
    async def run():
        async with fake.client() as client:
            return [d async for d in openrouter.stream_chat(
                messages or [{"role": "user", "content": "q"}], settings, client, **options)]
    return asyncio.run(run())


def calls(settings):
    conn = connect(settings.db_path)
    try:
        return [dict(row) for row in conn.execute("SELECT * FROM hosted_calls ORDER BY id")]
    finally:
        conn.close()


GOOD = sse(delta(reasoning="Hmm."), delta(content="Proper "), delta(content="time.", finish="stop"), usage())


class TestRequest:
    def test_one_provider_no_fallbacks_and_no_provider_that_keeps_prompts(self, settings):
        request = openrouter.request_for([], settings)
        assert request["provider"] == {"order": ["DeepInfra"], "allow_fallbacks": False,
                                       "data_collection": "deny", "quantizations": ["bf16"]}
        assert request["model"] == "qwen/qwen3.8-27b"
        assert request["reasoning"] == {"enabled": True}
        assert request["temperature"] == 1 and request["top_k"] == 20
        assert request["max_tokens"] == settings.openrouter_max_tokens

    def test_tools_stay_listed_but_none_can_be_called_while_a_document_is_written(self, settings):
        tools = [{"type": "function", "function": {"name": "make_pdf"}}]
        assert "tool_choice" not in openrouter.request_for([], settings, tools)
        assert openrouter.request_for([], settings, tools, allow_tools=False)["tool_choice"] == "none"

    def test_the_key_goes_only_in_the_authorization_header(self, settings):
        fake = Fake(GOOD)
        collect(settings, fake)
        request = fake.requests[0]
        assert request.headers["authorization"] == f"Bearer {KEY}"
        assert KEY not in request.content.decode() and KEY not in str(request.url)

    def test_a_missing_key_says_what_to_set_and_calls_nothing(self, settings, monkeypatch):
        monkeypatch.setattr(openrouter, "openrouter_api_key", lambda: None)
        fake = Fake(GOOD)
        with pytest.raises(RuntimeError, match="STUDY_ASSISTANT_OPENROUTER_KEY"):
            collect(settings, fake)
        assert fake.requests == []

    def test_a_model_with_no_price_is_refused_before_any_call(self, settings):
        fake = Fake(GOOD)
        with pytest.raises(RuntimeError, match="No price"):
            collect(replace(settings, openrouter_model="someone/else"), fake)
        assert fake.requests == []


class TestStream:
    def test_thinking_and_text_stream_and_the_last_delta_has_the_totals(self, settings):
        deltas = collect(settings, Fake(GOOD))
        assert [d.thinking for d in deltas[:-1]] == ["Hmm.", "", ""]
        assert "".join(d.text for d in deltas) == "Proper time."
        last = deltas[-1]
        assert last.done and last.done_reason == "stop"
        assert (last.prompt_tokens, last.output_tokens, last.cost_usd) == (1000, 400, 0.00123)
        assert (last.model, last.provider) == ("qwen/qwen3.8-27b", "DeepInfra")

    def test_every_call_is_logged_with_model_tokens_and_billed_cost(self, settings):
        collect(settings, Fake(GOOD), purpose="chat")
        [row] = calls(settings)
        assert (row["purpose"], row["model"], row["input_tokens"], row["output_tokens"]) == (
            "chat", "qwen/qwen3.8-27b", 1000, 400)
        assert row["cost_usd"] == 0.00123 and row["stop_reason"] == "stop" and row["error"] is None

    def test_without_a_billed_cost_the_tokens_are_priced(self, settings):
        body = sse(delta(content="x", finish="stop"), {"choices": [], "usage": {"prompt_tokens": 1_000_000,
                                                                                "completion_tokens": 1_000_000}})
        assert collect(settings, Fake(body))[-1].cost_usd == pytest.approx(0.15 + 1.875)

    def test_another_provider_is_stopped_before_any_text_is_shown(self, settings):
        body = sse(delta(content="Leaked?"), usage(), provider="SomeoneElse")
        shown = []

        async def run():
            async with Fake(body).client() as client:
                async for d in openrouter.stream_chat([{"role": "user", "content": "q"}], settings, client):
                    shown.append(d)
        with pytest.raises(openrouter.OpenRouterFailure, match="served by SomeoneElse, not the pinned DeepInfra"):
            asyncio.run(run())
        assert shown == []
        assert "SomeoneElse" in calls(settings)[0]["error"]

    def test_an_http_failure_says_the_status_and_is_logged_without_the_key(self, settings):
        body = json.dumps({"error": {"message": "No endpoints found"}}).encode()
        with pytest.raises(openrouter.OpenRouterFailure) as raised:
            collect(settings, Fake(body, status=404))
        assert str(raised.value) == "OpenRouter: HTTP 404, No endpoints found"
        [row] = calls(settings)
        assert row["error"] == "HTTP 404, No endpoints found" and row["model"] == "qwen/qwen3.8-27b"
        assert KEY not in json.dumps(row)

    def test_an_error_inside_the_stream_is_a_failure(self, settings):
        body = sse(delta(reasoning="..."), {"error": {"code": 502, "message": "Provider returned error"}})
        with pytest.raises(openrouter.OpenRouterFailure, match="HTTP 502"):
            collect(settings, Fake(body))

    def test_streamed_tool_calls_are_put_together(self, settings):
        body = sse(
            delta(tool_calls=[{"index": 0, "id": "call_1", "type": "function",
                               "function": {"name": "run_python", "arguments": '{"co'}}]),
            delta(tool_calls=[{"index": 0, "function": {"arguments": 'de": "print(1)"}'}}]),
            delta(finish="tool_calls"), usage())
        [call] = collect(settings, Fake(body))[-1].tool_calls
        assert (call.id, call.name, call.arguments) == ("call_1", "run_python", {"code": "print(1)"})


def answers(settings, fake, **options):
    async def run():
        async with fake.client() as client:
            return [e async for e in answer_events("Proper time?", [], settings, embed=lambda t, s: [[1.0] * 1024],
                                                   provider="openrouter", client=client, **options)]
    return asyncio.run(run())


class TestAnswers:
    def test_the_answer_is_marked_hosted_with_who_served_it(self, settings):
        name, done = answers(settings, Fake(GOOD))[-1]
        assert name == "done" and done["answer"] == "Proper time."
        assert done["hosted"] is True and done["model"] == "qwen/qwen3.8-27b" and done["served_by"] == "DeepInfra"
        assert done["cost_usd"] == 0.00123

    def test_stopping_at_the_limit_while_thinking_says_so_and_never_falls_back(self, settings):
        body = sse(delta(reasoning="still thinking"), delta(finish="length"), usage(completion=16000))
        events = answers(settings, Fake(body))
        assert "fallback" not in [name for name, _ in events]
        name, error = events[-1]
        assert name == "error" and error["cut_off"] is True
        assert "qwen/qwen3.8-27b stopped at its 16,000-token limit" in error["message"]

    def test_a_failure_is_an_error_and_never_a_fallback(self, settings):
        events = answers(settings, Fake(b'{"error": {"message": "overloaded"}}', status=503))
        assert "fallback" not in [name for name, _ in events]
        assert events[-1][0] == "error" and "HTTP 503" in events[-1][1]["message"]


@pytest.fixture
def library(settings):
    conn = connect(settings.db_path)
    conn.execute("INSERT INTO documents (id, path, sha256, title, kind, pages, status) "
                 "VALUES (1, 'a.pdf', 'x', 'Book', 'pdf', 9, 'indexed')")
    conn.execute("INSERT INTO chunks (id, doc_id, kind, seq, heading_path, text, token_count, block_ids, "
                 "page_start, page_end) VALUES (1, 1, 'text', 1, '[]', 'The Cholesky decomposition', 4, '[]', 2, 2)")
    conn.execute("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (1, ?)", (serialize([1.0] + [0.0] * 1023, 1024),))
    conn.commit()
    conn.close()
    return replace(settings, tools_enabled=True)


class TestTools:
    def test_a_tool_call_runs_and_its_result_goes_back_in_the_openai_format(self, library):
        first = sse(delta(tool_calls=[{"index": 0, "id": "call_7", "type": "function", "function": {
            "name": "search_library", "arguments": json.dumps({"query": "Cholesky"})}}]),
            delta(finish="tool_calls"), usage(cost=0.001))
        second = sse(delta(content="It factors A = LL^T [S1].", finish="stop"), usage(cost=0.002))
        fake = Fake(first, second)
        events = answers(library, fake, tools=True)
        assert [name for name, _ in events if name.startswith("tool")] == ["tool_start", "tool_end"]
        done = events[-1][1]
        assert done["answer"] == "It factors A = LL^T [S1]." and done["cost_usd"] == pytest.approx(0.003)
        followed = fake.sent(1)["messages"]
        assistant, result = followed[-2], followed[-1]
        assert assistant["tool_calls"][0]["id"] == "call_7"
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"query": "Cholesky"}
        assert result["role"] == "tool" and result["tool_call_id"] == "call_7"
        assert fake.sent(0)["tools"][0]["type"] == "function"
        assert len(calls(library)) == 2


def fake_stream(seen):
    async def stream(messages, settings, client=None, purpose="chat", tools=None, allow_tools=True):
        seen.append({"messages": messages, "settings": settings})
        yield Delta(text="From Qwen.")
        yield Delta(done=True, done_reason="stop", model="qwen/qwen3.8-27b", provider="DeepInfra", cost_usd=0.01)
    return stream


def ollama_reply(*words):
    events = [{"message": {"content": word}, "done": False} for word in words]
    events.append({"message": {"content": ""}, "done": True, "done_reason": "stop"})
    body = b"".join(json.dumps(event).encode() + b"\n" for event in events)

    def handler(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "gpt-oss:20b"}]})
        return httpx.Response(200, content=body)
    return httpx.MockTransport(handler)


def app_client(settings, *words):
    base = replace(settings, answer_options=list(OPTIONS))
    return TestClient(create_app(base, transport=ollama_reply(*words), embed=lambda t, s: [[1.0] * 1024 for _ in t]),
                      base_url="http://localhost")


def stream_events(response):
    found = []
    for block in response.text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        found.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return found


class TestApi:
    def test_the_options_come_in_order_with_where_each_runs(self, settings):
        body = app_client(settings).get("/api/options").json()
        assert [option["id"] for option in body["options"]] == ["fast", "qwen", "deep"]
        assert body["default"] == "fast"
        fast, qwen, deep = body["options"]
        assert (fast["hosted"], fast["loaded"]) == (False, True)
        assert (qwen["hosted"], qwen["route"], qwen["loaded"]) == (True, "DeepInfra", None)
        assert deep["fallback"] == "claude-sonnet-5-5"

    def test_a_question_without_a_mode_uses_the_first_option(self, settings, monkeypatch):
        monkeypatch.setattr(openrouter, "stream_chat", lambda *a, **k: pytest.fail("went to OpenRouter"))
        events = stream_events(app_client(settings, "Local.").post("/api/chat", json={"message": "Hi?"}))
        assert events[0][1]["option"] == "fast" and events[-1][1]["hosted"] is False

    def test_qwen_sends_only_the_last_turn_and_says_where_it_went(self, settings, monkeypatch):
        client = app_client(settings, "Local secret.")
        first = stream_events(client.post("/api/chat", json={"message": "Older question?"}))
        conversation = first[0][1]["conversation_id"]
        client = app_client(settings, "Latest answer.")
        client.post("/api/chat", json={"message": "Latest question?", "conversation_id": conversation})
        seen = []
        monkeypatch.setattr(openrouter, "stream_chat", fake_stream(seen))
        events = stream_events(client.post("/api/chat", json={
            "message": "Follow-up?", "mode": "qwen", "conversation_id": conversation}))
        meta, done = events[0][1], events[-1][1]
        assert (meta["option"], meta["provider"], meta["route"], meta["hosted"]) == ("qwen", "openrouter", "DeepInfra", True)
        assert (done["model"], done["served_by"], done["hosted"]) == ("qwen/qwen3.8-27b", "DeepInfra", True)
        sent = json.dumps(seen[0]["messages"])
        assert "Latest question?" in sent and "Latest answer." in sent and "Follow-up?" in sent
        assert "Older question" not in sent and "Local secret" not in sent
        assert seen[0]["settings"].openrouter_route == "DeepInfra"

    def test_an_unknown_option_is_refused_before_anything_is_stored(self, settings):
        response = app_client(settings).post("/api/chat", json={"message": "Hi?", "mode": "turbo"})
        assert response.status_code == 422 and "fast, qwen, deep" in response.json()["detail"]
        conn = connect(settings.db_path)
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 0

    def test_a_local_option_runs_its_own_model(self, settings):
        balanced = AnswerOption("balanced", "Balanced", "ollama", "gemma4:26b-a4b-it-qat", think=True)
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, content=b'{"message": {"content": "G."}, "done": true, "done_reason": "stop"}\n')
        base = replace(settings, answer_options=[OPTIONS[0], balanced])
        client = TestClient(create_app(base, transport=httpx.MockTransport(handler),
                                       embed=lambda t, s: [[1.0] * 1024 for _ in t]), base_url="http://localhost")
        events = stream_events(client.post("/api/chat", json={"message": "Hi?", "mode": "balanced"}))
        assert (seen[-1]["model"], seen[-1]["think"]) == ("gemma4:26b-a4b-it-qat", True)
        assert events[-1][1]["model"] == "gemma4:26b-a4b-it-qat" and events[-1][1]["hosted"] is False


class TestCosts:
    def test_hosted_options_carry_their_expected_cost_and_the_months_spend(self, settings):
        priced = [replace(option, expected_tokens=(1_000_000, 1_000_000)) for option in OPTIONS]
        client = TestClient(create_app(replace(settings, answer_options=priced), transport=ollama_reply(),
                                       embed=lambda t, s: [[1.0] * 1024 for _ in t]), base_url="http://localhost")
        record_call(settings, "chat", Tokens(input=10, output=10), 0.25, 1.0, "stop", None, model="claude-opus-5-5")
        record_call(settings, "chat", Tokens(input=10, output=10), 0.05, 1.0, "stop", None, model="qwen/qwen3.8-27b")
        body = client.get("/api/options").json()
        fast, qwen, deep = body["options"]
        assert fast["expected_usd"] is None and fast["price"] is None
        assert qwen["expected_usd"] == pytest.approx(0.15 + 1.875) and deep["expected_usd"] == pytest.approx(24.0)
        assert body["month_spend"] == [{"provider": "anthropic", "usd": 0.25, "limit_usd": 80.0},
                                       {"provider": "openrouter", "usd": 0.05, "limit_usd": None}]

    def test_a_hosted_answer_reports_what_the_log_says_it_cost(self, settings, monkeypatch):
        async def stream(messages, settings, client=None, purpose="chat", tools=None, allow_tools=True):
            # Two logged calls under this question, as a retry inside one answer would make.
            record_call(settings, purpose, Tokens(input=100, output=10), 0.004, 1.0, None, "HTTP 502", model=settings.openrouter_model)
            record_call(settings, purpose, Tokens(input=200, output=20), 0.006, 1.0, "stop", None, model=settings.openrouter_model)
            yield Delta(text="Answer.")
            yield Delta(done=True, done_reason="stop", model="qwen/qwen3.8-27b", provider="DeepInfra", cost_usd=0.006)
        monkeypatch.setattr(openrouter, "stream_chat", stream)
        events = stream_events(app_client(settings).post("/api/chat", json={"message": "Q?", "mode": "qwen"}))
        spend = events[-1][1]["spend"]
        assert spend["calls"] == 2 and spend["failed_calls"] == 1
        assert spend["cost_usd"] == pytest.approx(0.010) and (spend["input_tokens"], spend["output_tokens"]) == (300, 30)

    def test_a_local_answer_has_no_spend(self, settings):
        events = stream_events(app_client(settings, "Local.").post("/api/chat", json={"message": "Q?"}))
        assert "spend" not in events[-1][1]


def test_a_stream_cut_off_before_its_usage_is_logged_with_an_estimate(settings):
    """Review C5: OpenRouter sends the usage, input included, only at the end, so a cut-off
    answer was logged as 0 in and 0 out though the provider billed it."""
    body = sse(*[delta(content="word " * 50) for _ in range(20)])           # no usage, no finish
    collect(settings, Fake(body))
    [row] = calls(settings)
    assert row["input_tokens"] > 0 and row["output_tokens"] > 1000 and row["cost_usd"] > 0
    assert "estimated tokens" in row["error"]
