"""The model comparison bench's OpenRouter client, with the network replaced by canned streams."""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.db.connection import connect
from app.llm.hosted import Tokens

EVAL = Path(__file__).resolve().parents[2] / "eval"
spec = importlib.util.spec_from_file_location("api_bench", EVAL / "api_bench.py")
api_bench = importlib.util.module_from_spec(spec)
sys.modules["api_bench"] = api_bench        # its dataclasses look themselves up there
spec.loader.exec_module(api_bench)

KEY = "sk-or-v1-test-secret-0123456789"
MESSAGES = [{"role": "system", "content": "Rules"}, {"role": "user", "content": "Sources: [S1] x\n\nQuestion: q"}]
MODEL = api_bench.MODELS[0]


@pytest.fixture
def settings(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", library_dir=tmp_path / "library")
    connect(settings.db_path).close()
    return settings


def calls(settings):
    conn = connect(settings.db_path)
    try:
        return [dict(row) for row in conn.execute("SELECT * FROM hosted_calls ORDER BY id")]
    finally:
        conn.close()


def sse(*events, provider=MODEL.provider):
    lines = [": OPENROUTER PROCESSING"]
    lines += [f"data: {json.dumps({'provider': provider, **event})}" for event in events]
    return ("\n\n".join(lines + ["data: [DONE]"]) + "\n\n").encode()


def delta(content=None, reasoning=None, finish=None):
    piece = {k: v for k, v in (("content", content), ("reasoning", reasoning)) if v is not None}
    return {"choices": [{"delta": piece, "finish_reason": finish}]}


USAGE = {"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 400, "cost": 0.00123}}


def run(settings, body: bytes, status: int = 200, seen: list | None = None) -> dict:
    def handler(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, content=body)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await api_bench.answer(MODEL, MESSAGES, {1}, client, KEY, settings)
    return asyncio.run(go())


def test_the_key_comes_from_the_file_and_not_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(api_bench.OPENROUTER_KEY, "from-the-environment")
    env = tmp_path / ".env"
    assert api_bench.api_key(env) is None
    env.write_text(f"{api_bench.OPENROUTER_KEY}=from-the-file\n")
    assert api_bench.api_key(env) == "from-the-file"


def test_every_model_is_pinned_to_one_provider_that_keeps_no_prompts():
    for model in api_bench.MODELS:
        request = api_bench.request_for(model, MESSAGES)
        assert request["provider"] == {"order": [model.provider], "allow_fallbacks": False,
                                       "data_collection": "deny", "quantizations": [model.quantization]}
        assert request["messages"] is MESSAGES
        assert request["model"] == model.id and request["max_tokens"] == api_bench.MAX_TOKENS
        assert request["usage"] == {"include": True}


def test_the_three_models_are_the_ones_planned():
    assert [m.id for m in api_bench.MODELS] == ["qwen/qwen3.8-27b", "google/gemma-4-26b-a4b-it", "openai/gpt-oss-20b"]
    assert all(":" not in m.id for m in api_bench.MODELS)      # no :free or :batch variant


def test_cost_and_worst_case():
    assert api_bench.cost_usd(Tokens(input=1_000_000, output=1_000_000), (0.15, 1.875)) == pytest.approx(2.025)
    worst = api_bench.worst_case_usd(MODEL, [{"role": "user", "content": "x" * 30_000}])
    expected = (10_001 * MODEL.prices[0] + api_bench.MAX_TOKENS * MODEL.prices[1]) / 1_000_000
    assert worst == pytest.approx(expected)


def test_an_answer_is_recorded_like_run_eval_and_logged(settings):
    seen = []
    body = sse(delta(reasoning="thinking"), delta(content="The answer [S1] and [S"), delta(content="9]."),
               delta(finish="stop"), USAGE)
    record = run(settings, body, seen=seen)
    assert record["answer"] == "The answer [S1] and."
    assert record["citations"] == [1] and record["dropped_citations"] == [9]
    assert record["done_reason"] == "stop" and record["provider"] == MODEL.provider
    assert record["prompt_tokens"] == 1000 and record["output_tokens"] == 400
    assert record["cost_usd"] == 0.00123 and record["cost_billed"] is True
    assert record["first_output_seconds"] <= record["first_token_seconds"] <= record["total_seconds"]
    assert seen[0].headers["Authorization"] == f"Bearer {KEY}"
    assert json.loads(seen[0].content)["messages"] == MESSAGES
    [logged] = calls(settings)
    assert logged["purpose"] == api_bench.PURPOSE and logged["model"] == MODEL.id
    assert logged["cost_usd"] == 0.00123 and logged["error"] is None
    assert KEY not in json.dumps(record) and KEY not in json.dumps(logged)


def test_without_a_billed_cost_the_price_is_used(settings):
    usage = {"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 400}}
    record = run(settings, sse(delta(content="Yes [S1]."), delta(finish="stop"), usage))
    assert record["cost_usd"] == pytest.approx(api_bench.cost_usd(Tokens(input=1000, output=400), MODEL.prices))
    assert record["cost_billed"] is False


def test_a_rejected_request_is_logged_and_keeps_the_key_out(settings):
    body = json.dumps({"error": {"code": 401, "message": "No auth credentials found"}}).encode()
    record = run(settings, body, status=401)
    assert record["error"] == "HTTP 401, No auth credentials found"
    [logged] = calls(settings)
    assert logged["error"] == record["error"] and KEY not in logged["error"]


def test_another_provider_or_an_empty_answer_is_a_failure(settings):
    wrong = run(settings, sse(delta(content="Hi."), delta(finish="stop"), USAGE, provider="Elsewhere"))
    assert "not the pinned" in wrong["error"]
    empty = run(settings, sse(delta(reasoning="hmm"), delta(finish="length"), USAGE))
    assert empty["error"] == "no answer (length)"
    assert [row["cost_usd"] for row in calls(settings)] == [0.00123, 0.00123]     # failures still cost


def test_an_error_inside_the_stream_is_a_failure(settings):
    record = run(settings, sse({"error": {"code": 502, "message": "Provider returned error"}}))
    assert record["error"] == "HTTP 502, Provider returned error"


def test_spent_counts_only_this_bench(settings):
    conn = connect(settings.db_path)
    conn.executemany(
        "INSERT INTO hosted_calls (purpose, model, input_tokens, output_tokens, cache_read_tokens, "
        "cache_write_tokens, cost_usd, seconds) VALUES (?, ?, 0, 0, 0, 0, ?, 0)",
        [(api_bench.PURPOSE, "a", 0.5), (api_bench.PURPOSE, "b", 0.25), ("chat", "claude-opus-5-5", 9.0)])
    conn.commit()
    conn.close()
    assert api_bench.spent_usd(settings.db_path) == pytest.approx(0.75)
