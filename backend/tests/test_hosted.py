"""The hosted client, with the Anthropic SDK's stream replaced by a canned one."""
import asyncio
from types import SimpleNamespace as Fake

import anthropic
import httpx2
import pytest

from app.chat import answer_events
from dataclasses import replace

from app.config import HOSTED_KEY, Price, Settings, load_settings
from app.db.connection import connect
from app.llm import hosted
from app.llm.hosted import Tokens, cost_usd, request_for, split_system, stream_chat
from app.llm.ollama import Delta

MESSAGES = [{"role": "system", "content": "Rules"}, {"role": "user", "content": "Question"}]
PRICES = [4.0, 20.0]


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", library_dir=tmp_path / "library",
                    hosted_model="claude-opus-5-5")


def usage(input=0, output=0, cache_read=0, cache_write=0):
    return Fake(input_tokens=input, output_tokens=output,
                cache_read_input_tokens=cache_read, cache_creation_input_tokens=cache_write)


def text(words):
    return Fake(type="content_block_delta", delta=Fake(type="text_delta", text=words))


def thinking(words):
    return Fake(type="content_block_delta", delta=Fake(type="thinking_delta", thinking=words))


class FakeStream:
    def __init__(self, events, final, failure=None):
        self.events, self.final, self.failure = events, final, failure

    async def __aenter__(self):
        return self

    async def __aexit__(self, *details):
        return False

    async def __aiter__(self):
        for event in self.events:
            yield event
        if self.failure:
            raise self.failure

    async def get_final_message(self):
        return self.final


class FakeClient:
    def __init__(self, events=(), final=None, failure=None):
        self.stream_object = FakeStream(events, final, failure)
        self.messages = self
        self.request = None

    def stream(self, **request):
        self.request = request
        return self.stream_object


def collect(generator):
    async def run():
        return [item async for item in generator]
    return asyncio.run(run())


def calls(settings):
    conn = connect(settings.db_path)
    try:
        return [dict(row) for row in conn.execute("SELECT * FROM hosted_calls ORDER BY id")]
    finally:
        conn.close()


def finished_client():
    return FakeClient(
        [Fake(type="message_start", message=Fake(usage=usage(input=100, output=1))),
         thinking("hm"), text("Hello "), text("there"), Fake(type="message_delta", usage=usage(output=50))],
        Fake(usage=usage(input=100, output=50), stop_reason="end_turn"),
    )


def status_error(status, kind):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    return anthropic.APIStatusError("failed", response=response, body={"error": {"type": kind}})


def test_cost_counts_input_and_output_per_million():
    assert cost_usd(Tokens(input=1_000_000, output=1_000_000), PRICES) == pytest.approx(24.0)
    assert cost_usd(Tokens(input=4000, output=2000), PRICES) == pytest.approx(0.056)


def test_cached_tokens_cost_their_own_rates():
    opus = Price(4.0, 20.0, cache_read=0.20, cache_write=5.0)
    assert cost_usd(Tokens(cache_read=1_000_000), opus) == pytest.approx(0.20)
    assert cost_usd(Tokens(cache_write=1_000_000), opus) == pytest.approx(5.0)


def test_without_cache_prices_a_bill_is_never_understated():
    assert cost_usd(Tokens(cache_read=1_000_000), PRICES) == pytest.approx(4.0)
    assert cost_usd(Tokens(cache_write=1_000_000), PRICES) == pytest.approx(5.0)


def test_the_configured_prices_carry_the_date_they_were_checked():
    settings = load_settings()
    for model in ("claude-opus-5-5", "claude-sonnet-5-5", "qwen/qwen3.8-27b"):
        price = settings.hosted_prices[model]
        assert price.checked and price.source, model
    assert settings.hosted_prices["claude-opus-5-5"].cache_read == pytest.approx(0.05 * 4.0)


def log(settings, model, cost, turn=None, error=None, when=None):
    conn = connect(settings.db_path)
    conn.execute("INSERT INTO hosted_calls (purpose, model, input_tokens, output_tokens, cache_read_tokens, "
                 "cache_write_tokens, cost_usd, seconds, error, turn, created_at) "
                 "VALUES ('chat', ?, 100, 50, 0, 0, ?, 1, ?, ?, COALESCE(?, datetime('now')))",
                 (model, cost, error, turn, when))
    conn.commit()
    conn.close()


def test_a_questions_cost_adds_up_every_call_it_made_failed_ones_too(settings):
    log(settings, "claude-opus-5-5", 0.01, turn="t1", error="HTTP 529, overloaded_error")
    log(settings, "claude-sonnet-5-5", 0.02, turn="t1")
    log(settings, "claude-opus-5-5", 0.50, turn="other")
    spent = hosted.turn_cost(settings, "t1")
    assert spent["calls"] == 2 and spent["failed_calls"] == 1
    assert spent["cost_usd"] == pytest.approx(0.03) and spent["input_tokens"] == 200


def test_the_months_spend_is_split_by_who_bills_it_and_leaves_out_last_month(settings):
    log(settings, "claude-opus-5-5", 0.10)
    log(settings, "qwen/qwen3.8-27b", 0.02)
    log(settings, "claude-opus-5-5", 9.00, when="2000-01-31 23:59:59")
    assert hosted.month_spend(settings) == pytest.approx({"anthropic": 0.10, "openrouter": 0.02})


def test_each_logged_call_names_its_question(settings):
    collect(hosted.stream_chat(MESSAGES, replace(settings, turn_id="q-7"), finished_client()))
    assert calls(settings)[0]["turn"] == "q-7"


def test_the_system_prompt_goes_apart_from_the_turns():
    assert split_system(MESSAGES) == ("Rules", [{"role": "user", "content": "Question"}])


def test_the_request_uses_settings_and_adaptive_thinking(settings):
    request = request_for(MESSAGES, settings)
    assert request["model"] == "claude-opus-5-5"
    assert request["thinking"]["type"] == "adaptive"
    assert request["output_config"] == {"effort": settings.hosted_effort}
    assert request["max_tokens"] == settings.hosted_max_tokens


def test_options_these_models_refuse_are_never_sent(settings):
    request = request_for(MESSAGES, settings)
    assert not {"temperature", "top_p", "top_k", "fallbacks"} & set(request)
    assert "budget_tokens" not in request["thinking"]


def test_text_and_thinking_stream_in_order_and_the_last_delta_has_the_totals(settings):
    deltas = collect(stream_chat(MESSAGES, settings, finished_client()))
    assert [(d.thinking, d.text) for d in deltas[:-1]] == [("hm", ""), ("", "Hello "), ("", "there")]
    last = deltas[-1]
    assert (last.done, last.done_reason, last.prompt_tokens, last.output_tokens) == (True, "end_turn", 100, 50)
    assert last.cost_usd == pytest.approx(0.0014)


def test_every_call_is_logged_with_model_tokens_and_cost(settings):
    collect(stream_chat(MESSAGES, settings, finished_client(), purpose="eval:opus"))
    [call] = calls(settings)
    assert (call["purpose"], call["model"], call["input_tokens"], call["output_tokens"]) == (
        "eval:opus", "claude-opus-5-5", 100, 50)
    assert call["cost_usd"] == pytest.approx(0.0014)
    assert call["stop_reason"] == "end_turn" and call["error"] is None and call["seconds"] >= 0


def test_the_log_has_no_column_for_text_or_a_key(settings):
    collect(stream_chat(MESSAGES, settings, finished_client()))
    columns = set(calls(settings)[0])
    assert not {"prompt", "content", "answer", "text", "key", "api_key"} & columns


def test_a_failed_call_is_logged_and_says_only_the_status(settings):
    client = FakeClient([Fake(type="message_start", message=Fake(usage=usage(input=100)))],
                        failure=status_error(429, "rate_limit_error"))
    with pytest.raises(RuntimeError, match="HTTP 429, rate_limit_error") as caught:
        collect(stream_chat(MESSAGES, settings, client))
    assert caught.value.__cause__ is None
    [call] = calls(settings)
    assert call["error"] == "HTTP 429, rate_limit_error" and call["input_tokens"] == 100


def test_an_interrupted_call_is_logged_with_the_tokens_so_far(settings):
    async def run():
        replies = stream_chat(MESSAGES, settings, finished_client())
        async for delta in replies:
            if delta.text:
                await replies.aclose()
                return
    asyncio.run(run())
    [call] = calls(settings)
    assert call["input_tokens"] == 100 and call["stop_reason"] is None


def test_a_model_with_no_price_is_refused_before_any_call(settings):
    settings.hosted_model = "claude-unknown"
    client = FakeClient()
    with pytest.raises(RuntimeError, match="No price"):
        collect(stream_chat(MESSAGES, settings, client))
    assert client.request is None


def test_a_missing_key_says_what_to_set(settings, monkeypatch):
    monkeypatch.setattr(hosted, "hosted_api_key", lambda: None)
    with pytest.raises(RuntimeError, match=HOSTED_KEY):
        collect(stream_chat(MESSAGES, settings))


def test_the_key_goes_straight_to_the_client(settings, monkeypatch):
    seen = {}

    class Client(FakeClient):
        def __init__(self, api_key, timeout):
            super().__init__(final=Fake(usage=usage(), stop_reason="end_turn"))
            seen["key"] = api_key

        async def close(self):
            seen["closed"] = True

    monkeypatch.setattr(hosted, "hosted_api_key", lambda: "not-a-real-key")
    monkeypatch.setattr(hosted.anthropic, "AsyncAnthropic", Client)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    collect(stream_chat(MESSAGES, settings))
    assert seen == {"key": "not-a-real-key", "closed": True}
    assert "ANTHROPIC_API_KEY" not in __import__("os").environ


def empty_search(settings, question, doc_ids, embed, cited=()):
    return []


def test_deep_mode_answers_through_the_hosted_client(settings, monkeypatch):
    async def fake_stream(messages, settings, client=None, purpose="chat"):
        assert purpose == "eval:x"
        yield Delta(text="An answer.")
        yield Delta(done=True, done_reason="end_turn", prompt_tokens=7, output_tokens=3, cost_usd=0.25)

    monkeypatch.setattr("app.chat.find_sources", empty_search)
    monkeypatch.setattr(hosted, "stream_chat", fake_stream)
    events = collect(answer_events("Q?", [], settings, provider="anthropic", purpose="eval:x"))
    done = events[-1][1]
    assert events[-1][0] == "done"
    assert (done["answer"], done["hosted"], done["model"], done["cost_usd"]) == (
        "An answer.", True, "claude-opus-5-5", 0.25)


def test_a_refusal_is_reported_and_not_passed_off_as_an_answer(settings, monkeypatch):
    async def fake_stream(messages, settings, client=None, purpose="chat"):
        yield Delta(done=True, done_reason="refusal")

    monkeypatch.setattr("app.chat.find_sources", empty_search)
    monkeypatch.setattr(hosted, "stream_chat", fake_stream)
    event, data = collect(answer_events("Q?", [], settings, provider="anthropic", fallback=False))[-1]
    assert event == "error" and "refusal" in data["message"]


def test_a_hosted_failure_reaches_the_reader_as_an_error(settings, monkeypatch):
    async def fake_stream(messages, settings, client=None, purpose="chat"):
        raise RuntimeError("Hosted model: HTTP 429, rate_limit_error")
        yield

    monkeypatch.setattr("app.chat.find_sources", empty_search)
    monkeypatch.setattr(hosted, "stream_chat", fake_stream)
    event, data = collect(answer_events("Q?", [], settings, provider="anthropic"))[-1]
    assert event == "error" and "429" in data["message"]


class Attempts:
    """A client whose streams play in turn, one for each model tried."""

    def __init__(self, *streams):
        self.streams = list(streams)
        self.models = []
        self.messages = self

    def stream(self, **request):
        self.models.append(request["model"])
        return self.streams.pop(0)


def good_stream(words="Sonnet says", output=10):
    return FakeStream([text(words)], Fake(usage=usage(input=100, output=output), stop_reason="end_turn"))


def failed_stream(status=529, kind="overloaded_error", events=()):
    return FakeStream(list(events), None, failure=status_error(status, kind))


def replies(settings, client, **options):
    return collect(hosted.answer_stream(MESSAGES, settings, client, **options))


def test_a_healthy_main_model_never_touches_the_fallback(settings):
    client = Attempts(good_stream("Opus says"))
    deltas = replies(settings, client)
    assert client.models == ["claude-opus-5-5"]
    assert not any(d.restart for d in deltas) and deltas[-1].model == "claude-opus-5-5"


def test_an_api_failure_restarts_on_the_fallback_and_says_why(settings):
    client = Attempts(failed_stream(), good_stream())
    deltas = replies(settings, client)
    assert client.models == ["claude-opus-5-5", "claude-sonnet-5-5"]
    restart = next(d for d in deltas if d.restart)
    assert (restart.model, restart.reason) == ("claude-sonnet-5-5", "HTTP 529, overloaded_error")
    assert deltas[-1].model == "claude-sonnet-5-5" and deltas[-1].done


def test_a_failure_part_way_restarts_after_what_was_already_sent(settings):
    client = Attempts(failed_stream(events=[text("half an ans")]), good_stream())
    order = [("restart" if d.restart else d.text) for d in replies(settings, client) if d.restart or d.text]
    assert order == ["half an ans", "restart", "Sonnet says"]


def test_a_refusal_goes_to_the_fallback(settings):
    refused = FakeStream([], Fake(usage=usage(input=100), stop_reason="refusal"))
    deltas = replies(settings, Attempts(refused, good_stream()))
    assert next(d for d in deltas if d.restart).reason == "it refused"
    assert deltas[-1].model == "claude-sonnet-5-5"


def test_a_refusal_after_some_text_still_goes_to_the_fallback(settings):
    refused = FakeStream([text("Some")], Fake(usage=usage(), stop_reason="refusal"))
    assert any(d.restart for d in replies(settings, Attempts(refused, good_stream())))


def test_an_empty_answer_goes_to_the_fallback(settings):
    empty = FakeStream([thinking("hm")], Fake(usage=usage(output=16000), stop_reason="max_tokens"))
    assert next(d for d in replies(settings, Attempts(empty, good_stream())) if d.restart).reason == (
        "it returned no answer")


def test_an_answer_cut_off_by_the_length_limit_is_kept(settings):
    cut = FakeStream([text("Long answer")], Fake(usage=usage(), stop_reason="max_tokens"))
    client = Attempts(cut)
    deltas = replies(settings, client)
    assert client.models == ["claude-opus-5-5"] and deltas[-1].done_reason == "max_tokens"


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_skips_the_fallback_because_it_shares_the_key(settings, status):
    client = Attempts(failed_stream(status, "authentication_error"))
    with pytest.raises(RuntimeError, match=str(status)):
        replies(settings, client)
    assert client.models == ["claude-opus-5-5"]


def test_when_both_models_fail_the_error_names_both(settings):
    client = Attempts(failed_stream(), failed_stream(500, "api_error"))
    with pytest.raises(RuntimeError) as caught:
        replies(settings, client)
    for part in ("claude-opus-5-5", "claude-sonnet-5-5", "529", "500"):
        assert part in str(caught.value)


def test_both_attempts_are_logged_under_their_own_model_and_price(settings):
    replies(settings, Attempts(failed_stream(events=[Fake(type="message_start", message=Fake(usage=usage(input=1000)))]),
                               good_stream(output=1000)))
    first, second = calls(settings)
    assert (first["model"], first["error"]) == ("claude-opus-5-5", "HTTP 529, overloaded_error")
    assert first["cost_usd"] == pytest.approx(0.004)
    assert (second["model"], second["error"]) == ("claude-sonnet-5-5", None)
    assert second["cost_usd"] == pytest.approx(0.0102)


def test_without_a_fallback_a_failure_is_just_a_failure(settings):
    client = Attempts(failed_stream())
    with pytest.raises(RuntimeError, match="529"):
        replies(settings, client, fallback=False)
    assert client.models == ["claude-opus-5-5"]


def test_a_fallback_equal_to_the_main_model_is_ignored(settings):
    settings.hosted_fallback_model = "claude-opus-5-5"
    client = Attempts(failed_stream())
    with pytest.raises(RuntimeError):
        replies(settings, client)
    assert len(client.models) == 1


def test_the_reader_is_told_about_a_fallback_and_it_replaces_the_partial_answer(settings, monkeypatch):
    async def fake(messages, settings, purpose="chat", fallback=True):
        yield Delta(text="Half [S1")
        yield Delta(restart=True, model="claude-sonnet-5-5", reason="HTTP 529, overloaded_error")
        yield Delta(text="Whole answer.")
        yield Delta(done=True, done_reason="end_turn", model="claude-sonnet-5-5", cost_usd=0.01)

    monkeypatch.setattr("app.chat.find_sources", empty_search)
    monkeypatch.setattr(hosted, "answer_stream", fake)
    events = collect(answer_events("Q?", [], settings, provider="anthropic"))
    assert ("fallback", {"model": "claude-sonnet-5-5", "from": "claude-opus-5-5",
                         "reason": "HTTP 529, overloaded_error"}) in events
    done = events[-1][1]
    assert done["answer"] == "Whole answer." and done["model"] == "claude-sonnet-5-5"
    assert done["fallback_from"] == "claude-opus-5-5" and done["fallback_reason"] == "HTTP 529, overloaded_error"


def test_the_eval_turns_the_fallback_off_so_runs_cannot_mix_models():
    source = (__import__("pathlib").Path(__file__).parents[2] / "eval" / "run_eval.py").read_text(encoding="utf-8")
    assert "fallback=False" in source


def test_an_answer_stopped_midway_logs_what_streamed_not_zero(settings):
    """Review C5: Stop (or a dropped connection) ends the stream before message_delta, so the
    log kept message_start's output of 1 token though a long reply had been billed."""
    long_reply = [text("word " * 50) for _ in range(20)]       # about 1000 words
    client = FakeClient([Fake(type="message_start", message=Fake(usage=usage(input=100, output=1))), *long_reply],
                        Fake(usage=usage(input=100, output=1300), stop_reason="end_turn"))

    async def stop_after_some():
        stream = stream_chat(MESSAGES, settings, client)
        async for delta in stream:
            if delta.text:
                break               # the reader pressed Stop
        await stream.aclose()
    asyncio.run(stop_after_some())

    async def read_all_then_drop():
        async for _ in stream_chat(MESSAGES, settings, FakeClient(
                [Fake(type="message_start", message=Fake(usage=usage(input=100, output=1))), *long_reply],
                None, failure=anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x")))):
            pass
    with pytest.raises(hosted.HostedFailure):
        asyncio.run(read_all_then_drop())
    stopped, dropped = calls(settings)
    assert stopped["input_tokens"] == 100 and stopped["output_tokens"] > 1 and "estimated" in stopped["error"]
    assert stopped["error"].startswith("estimated tokens")      # not counted as a failed call
    assert dropped["error"].startswith("APIConnectionError") or "estimated" in dropped["error"]
    assert dropped["output_tokens"] > 1000 and "estimated" in dropped["error"]
