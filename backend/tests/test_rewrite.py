"""Rewriting follow-ups, with Ollama replaced by a canned reply."""
import asyncio
import json

import httpx
import pytest

from app.chat import answer_events
from app.config import Settings
from app.ingest.embed import embed_texts
from app.llm import hosted
from app.llm.ollama import Delta
from app.retrieval.rewrite import (adds_context, points_back, rewrite_payload, standalone_question, usable,
                                   warm_up_helper)

TURN = [{"role": "user", "content": "What is SeismoGPT?"},
        {"role": "assistant", "content": "A transformer for seismic waves [S1]."}]


def helper(content="What does SeismoGPT do?", status=200, failure=None):
    seen = []

    def answer(request):
        seen.append(json.loads(request.content))
        if failure:
            raise failure
        return httpx.Response(status, json={"message": {"content": content}})

    transport = httpx.MockTransport(answer)
    transport.seen = seen
    return transport


def rewrite(question, transport):
    return asyncio.run(standalone_question(question, TURN, Settings(), transport))


def test_a_follow_up_becomes_the_helpers_question():
    assert rewrite("I don't understand!", helper()) == "What does SeismoGPT do?"


def test_the_helper_sees_the_previous_turn_without_its_citations():
    transport = helper()
    rewrite("I don't understand!", transport)
    prompt = transport.seen[0]["messages"][1]["content"]
    assert "What is SeismoGPT?" in prompt and "A transformer for seismic waves." in prompt
    assert "I don't understand!" in prompt and "[S1]" not in prompt


def test_the_request_uses_the_small_helper_without_thinking():
    payload = rewrite_payload("Why?", TURN, Settings())
    assert payload["model"] == Settings().helper_model
    assert payload["think"] is False and payload["stream"] is False
    assert payload["options"]["temperature"] == 0
    assert payload["options"]["num_gpu"] == 0, "the reranker needs that GPU memory"


def test_a_long_answer_is_cut_to_fit_the_helpers_small_context():
    settings = Settings(rewrite_answer_chars=10)
    turn = [TURN[0], {"role": "assistant", "content": "x" * 5000}]
    assert "x" * 11 not in rewrite_payload("Why?", turn, settings)["messages"][1]["content"]


@pytest.mark.parametrize("output, expected", [
    ("What is it?", "What is it?"),
    ('"What is it?"', "What is it?"),
    ("  What is it?  \nSome explanation", "What is it?"),
    ("", None),
    ("   ", None),
    ("x" * 301, None),
])
def test_only_a_usable_line_counts(output, expected):
    assert usable(output) == expected


@pytest.mark.parametrize("transport", [
    helper(status=500),
    helper(content=""),
    helper(content="x" * 400),
    helper(failure=httpx.ConnectError("refused")),
    helper(failure=httpx.ReadTimeout("slow")),
    httpx.MockTransport(lambda request: httpx.Response(200, text="not json")),
    httpx.MockTransport(lambda request: httpx.Response(200, json={"unexpected": True})),
], ids=["error", "empty", "too long", "refused", "timeout", "not json", "wrong shape"])
def test_any_failure_searches_with_the_message_as_written(transport):
    assert rewrite("I don't understand!", transport) == "I don't understand!"


def run_answer(monkeypatch, history, transport, **options):
    searched = []
    carried = options.pop("carried", [])
    question = options.pop("question", "I don't understand!")

    def fake_search(settings, query, doc_ids, embed, cited=()):
        searched.append(query)
        carried.append(list(cited))
        return []

    seen = []

    async def fake_reply(messages, settings, purpose="chat", fallback=True):
        seen.append(messages)
        yield Delta(text="Answer.")
        yield Delta(done=True, done_reason="end_turn", model="m")

    monkeypatch.setattr("app.chat.find_sources", fake_search)
    monkeypatch.setattr(hosted, "answer_stream", fake_reply)

    async def collect():
        return [item async for item in answer_events(
            question, history, Settings(), None, embed_texts, transport, provider="anthropic", **options)]

    return asyncio.run(collect()), searched, seen


def test_the_search_uses_the_rewrite_and_the_model_sees_the_readers_words(monkeypatch):
    transport = helper()
    events, searched, seen = run_answer(monkeypatch, TURN, transport)
    assert searched == ["What does SeismoGPT do?"]
    assert "I don't understand!" in seen[0][-1]["content"]
    assert "What does SeismoGPT do?" not in seen[0][-1]["content"]


def test_the_reader_is_shown_what_was_searched(monkeypatch):
    events, _, _ = run_answer(monkeypatch, TURN, helper())
    assert dict(events)["sources"]["query"] == "What does SeismoGPT do?"


def test_a_first_question_never_calls_the_helper(monkeypatch):
    transport = helper()
    events, searched, _ = run_answer(monkeypatch, [], transport)
    assert transport.seen == [] and searched == ["I don't understand!"]
    assert dict(events)["sources"]["query"] is None


def test_an_unchanged_question_shows_no_search_line(monkeypatch):
    events, _, _ = run_answer(monkeypatch, TURN, helper(content="I don't understand!"))
    assert dict(events)["sources"]["query"] is None


CITED = [{"chunk_id": 7, "chunk_ids": [7]}]


def test_a_follow_up_carries_what_the_previous_answer_cited(monkeypatch):
    carried = []
    run_answer(monkeypatch, TURN, helper(), cited=CITED, carried=carried)
    assert carried == [CITED]


def test_a_new_topic_leaves_the_old_sources_behind(monkeypatch):
    carried = []
    run_answer(monkeypatch, TURN, helper(content="I don't understand!"), cited=CITED, carried=carried)
    assert carried == [[]]


@pytest.mark.parametrize("query, question", [
    ("What does SeismoGPT do?", "I don't understand!"),
    ("Is the Cholesky decomposition unique?", "is it unique?"),
    ("What is the Z mass?", "and the Z?"),
])
def test_a_rewrite_that_adds_words_is_a_follow_up(query, question):
    assert adds_context(query, question)


@pytest.mark.parametrize("query, question", [
    ("What is double descent?", "Thanks. What is double descent?"),
    ("What does Cholesky factor a matrix into?", "In one sentence, what does Cholesky factor a matrix into?"),
    ("What is entropy?", "What is entropy?"),
    ("what is ENTROPY", "What is entropy?"),
])
def test_a_tidied_new_question_is_not_a_follow_up(query, question):
    assert not adds_context(query, question)


def test_a_tidied_new_topic_leaves_the_old_sources_behind(monkeypatch):
    carried = []
    monkeypatch.setattr("app.chat.standalone_question", tidied)
    run_answer(monkeypatch, TURN, helper(), cited=CITED, carried=carried, question="Thanks. What is entropy?")
    assert carried == [[]]


async def tidied(question, turn, settings, transport=None):
    return "What is entropy?"


def test_warming_up_uses_the_same_context_as_a_rewrite_or_ollama_would_reload(monkeypatch):
    seen = {}

    class Answer:
        def raise_for_status(self):
            pass

    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: seen.update(json) or Answer())
    settings = Settings()
    warm_up_helper(settings)
    sent = rewrite_payload("Why?", TURN, settings)
    assert seen["options"]["num_ctx"] == sent["options"]["num_ctx"]
    assert seen["options"]["num_gpu"] == sent["options"]["num_gpu"]
    assert seen["keep_alive"] == sent["keep_alive"]



# Regression: "So, how can I translate this to code?" after a question about the SeismoGPT
# loss was searched word for word (Nov 2026). The helper had answered "How can I translate
# this into code?", and the new word "into" made it look like a follow-up with a topic.
LOSS_TURN = [{"role": "user", "content": "What loss function does SeismoGPT use, and why?"},
             {"role": "assistant", "content": "SeismoGPT trains with a log-cosh loss [S1]."}]
FIGURE_TURN = [{"role": "user", "content": "What does Figure 4 of the SeismoGPT paper show about the context ratio?"},
               {"role": "assistant", "content": "It plots NCC against the context ratio [S1]."}]
POINTING_CASES = [
    ("So, how can I translate this to code?", LOSS_TURN, "How can I translate this into code?"),
    ("explain this figure", FIGURE_TURN, "Explain this figure."),
    ("what does that curve show?", FIGURE_TURN, "What does that curve show?"),
]


def follow_up(question, turn, transport):
    return asyncio.run(standalone_question(question, turn, Settings(), transport))


@pytest.mark.parametrize("question, turn, unhelpful", POINTING_CASES, ids=["to code", "this figure", "that curve"])
def test_a_follow_up_that_points_back_is_never_searched_without_its_topic(question, turn, unhelpful):
    query = follow_up(question, turn, helper(content=unhelpful))
    assert query == f"{question} (about: {turn[0]['content']})"
    assert adds_context(query, question), "so it also carries what the previous answer cited"


@pytest.mark.parametrize("question, turn, unhelpful", POINTING_CASES, ids=["to code", "this figure", "that curve"])
def test_when_the_helper_fails_a_follow_up_still_gets_its_topic(question, turn, unhelpful):
    assert follow_up(question, turn, helper(status=500)) == f"{question} (about: {turn[0]['content']})"
    assert follow_up(question, turn, helper(failure=httpx.ReadTimeout("slow"))).endswith(f"(about: {turn[0]['content']})")


@pytest.mark.parametrize("question, turn, good", [
    ("So, how can I translate this to code?", LOSS_TURN, "How can I implement the SeismoGPT loss function in code?"),
    ("explain this figure", FIGURE_TURN, "Explain Figure 4 of the SeismoGPT paper."),
    ("what does that curve show?", FIGURE_TURN, "What does the curve in Figure 4 show about the context ratio?"),
], ids=["to code", "this figure", "that curve"])
def test_a_rewrite_that_names_the_topic_is_kept_as_it_is(question, turn, good):
    assert follow_up(question, turn, helper(content=good)) == good


def test_the_exact_case_searches_for_seismogpt_and_carries_the_cited_passages(monkeypatch):
    carried = []
    events, searched, seen = run_answer(monkeypatch, LOSS_TURN, helper(content="How can I translate this into code?"),
                                        question="So, how can I translate this to code?", cited=CITED, carried=carried)
    assert "SeismoGPT" in searched[0] and "loss" in searched[0]
    assert dict(events)["sources"]["query"] == searched[0], "the reader sees what was searched"
    assert carried == [CITED]
    assert "So, how can I translate this to code?" in seen[0][-1]["content"], "the model still gets the reader's words"


@pytest.mark.parametrize("question", [
    "So, how can I translate this to code?", "explain this figure", "what does that curve show?",
    "Why does it diverge?", "Can you plot these?", "And those two terms?", "What do they mean by tokens?",
])
def test_these_lean_on_what_came_before(question):
    assert points_back(question)


@pytest.mark.parametrize("question", [
    "What is a Fourier transform?",
    "Prove that the Cholesky factor is unique.",
    "Show that the determinant of a product is the product of the determinants.",
    "Find x such that Ax = b.",
    "Write a Python function for the Gram-Schmidt process.",
])
def test_a_standalone_question_doesnt_lean_on_anything(question):
    assert not points_back(question)


def test_a_new_question_with_that_as_a_conjunction_is_searched_as_the_helper_returns_it():
    question = "Prove that the Cholesky factor is unique."
    assert follow_up(question, LOSS_TURN, helper(content=question)) == question


@pytest.mark.parametrize("query, question", [
    ("How can I translate this into code?", "So, how can I translate this to code?"),
    ("What is the entropy in one sentence?", "In one sentence, what is entropy?"),
])
def test_words_that_carry_no_topic_dont_make_a_follow_up(query, question):
    assert not adds_context(query, question)
