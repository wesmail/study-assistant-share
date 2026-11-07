"""Explain, line by line (roadmap item 4, step 6)."""
from __future__ import annotations

import asyncio
import json

import pytest

from app import explain
from app.annotations import Refused
from tests.test_anchors import setup  # noqa: F401  (fixture)
from tests.test_suggest import DEEP, fake_model

ALIGNED = r"""\begin{aligned} a &= b + c \\ d &= e f \end{aligned}"""


def reply(*spans):
    return json.dumps({"items": [{"span": s, "explanation": f"About {s}"} for s in spans]})


class TestChecks:
    def test_spans_must_cover_the_text_in_order(self):
        text = "x = 1\ny = x + 2\nprint(y)"
        items = lambda *spans: [explain.Item(span=s, explanation="e") for s in spans]  # noqa: E731
        assert explain.covers(items("x = 1", "y = x + 2", "print(y)"), text)
        assert explain.covers(items("x=1", "y = x+2\nprint(y)"), text)            # spaces don't matter
        assert explain.covers(items("a = b,", "c = d"), "$$ a = b, c = d. $$")      # fences and punctuation
        assert not explain.covers(items("x = 1", "print(y)"), text)                 # a line left out
        assert not explain.covers(items("y = x + 2", "x = 1", "print(y)"), text)    # out of order
        assert not explain.covers(items("x = 1", "y = x + 3", "print(y)"), text)    # changed

    def test_parse(self):
        assert explain.parse("```json\n" + reply("x = 1", "y = 2") + "\n```", "x = 1\ny = 2")[1].span == "y = 2"
        assert explain.parse(reply("x = 1"), "x = 1\ny = 2") is None
        assert explain.parse("no json", "x") is None

    @pytest.mark.parametrize("text, kind", [
        ("for i in range(3):\n    total += i", "code"), ("import numpy as np", "code"),
        (r"\frac{\partial \ell}{\partial \beta_0} = -2(f - y)", "equation"), ("E = mc^2", "equation"),
    ])
    def test_kind_from_the_text(self, text, kind):
        assert explain.kind_of(text) == kind


class TestExplain:
    def test_asks_again_when_the_spans_dont_cover_the_text(self, monkeypatch):
        model = fake_model(reply("x = 1"), reply("x = 1", "print(x + 2)"))
        monkeypatch.setattr("app.chat.model_reply", model)
        result = asyncio.run(explain.explain(None, DEEP, "x = 1\nprint(x + 2)", depth="intro", context="from p. 3"))
        assert [item["span"] for item in result["items"]] == ["x = 1", "print(x + 2)"]
        assert result["kind"] == "code" and result["cost_usd"] == 0.02 and len(model.calls) == 2
        assert "first-year student" in model.calls[0][0]["content"] and "from p. 3" in model.calls[0][1]["content"]
        assert "consecutive pieces" in model.calls[1][-1]["content"]

    def test_the_retry_quotes_a_changed_span(self, monkeypatch):
        model = fake_model(reply("x = 1", "print(x + 3)"), reply("x = 1", "print(x + 2)"))
        monkeypatch.setattr("app.chat.model_reply", model)
        asyncio.run(explain.explain(None, DEEP, "x = 1\nprint(x + 2)"))
        assert "not the next piece of the given text: print(x + 3)" in model.calls[1][-1]["content"]

    def test_refused(self):
        with pytest.raises(Refused, match="Select"):
            asyncio.run(explain.explain(None, DEEP, "  "))
        with pytest.raises(Refused, match="depth"):
            asyncio.run(explain.explain(None, DEEP, "x", depth="phd"))
        with pytest.raises(Refused, match="At most"):
            asyncio.run(explain.explain(None, DEEP, "x" * 5000))


def test_the_books_latex_under_a_selection(setup):  # noqa: F811
    settings, conn, _ = setup
    conn.execute("INSERT INTO blocks (doc_id, \"order\", kind, page, bbox, text) VALUES (1, 1, 'equation', 2, ?, ?)",
                 (json.dumps([30, 60, 300, 80]), "$$\nE^2 = p^2 c^2 + m^2 c^4\n$$"))
    conn.execute("INSERT INTO blocks (doc_id, \"order\", kind, page, bbox, text) VALUES (1, 2, 'equation', 2, ?, ?)",
                 (json.dumps([30, 200, 300, 220]), "$$\nx = 1\n$$"))
    assert explain.book_latex(conn, 1, 2, [[40, 62, 200, 78]]) == "E^2 = p^2 c^2 + m^2 c^4"
    assert explain.book_latex(conn, 1, 2, [[40, 150, 200, 160]]) is None


def test_through_the_api(setup, monkeypatch):  # noqa: F811
    from tests.test_chat import FakeOllama, client_for

    settings, conn, _ = setup
    conn.execute("INSERT INTO blocks (doc_id, \"order\", kind, page, bbox, text) VALUES (1, 1, 'equation', 2, ?, ?)",
                 (json.dumps([30, 60, 300, 80]), "$$\nE^2 = p^2 c^2 + m^2 c^4\n$$"))
    conn.commit()
    model = fake_model(reply("E^2 =", "p^2 c^2 + m^2 c^4"))
    monkeypatch.setattr("app.chat.model_reply", model)
    client = client_for(settings, FakeOllama())
    made = client.post("/api/explain", json={"option": "deep", "passage": {"doc_id": 1, "page": 2,
                                             "rects": [[40, 62, 200, 78]], "quote": "E2 = p2c2 + m2c4"}})
    assert made.status_code == 200, made.text
    body = made.json()
    assert body["from_book"] is True and body["kind"] == "equation" and body["text"] == "E^2 = p^2 c^2 + m^2 c^4"
    assert "Energy and momentum" in model.calls[0][1]["content"]      # the page's words, as context
    assert client.post("/api/explain", json={"text": "x", "option": "nope"}).status_code == 404
    assert client.post("/api/explain", json={"text": " ", "option": "deep"}).status_code == 409


def test_the_option_follows_the_kind(setup, monkeypatch):  # noqa: F811
    """Equations always from Deep; code from the reader's mode if Balanced or Deep, else Deep."""
    from tests.test_chat import FakeOllama, client_for

    from dataclasses import replace

    from app.config import AnswerOption

    settings, _, _ = setup
    settings = replace(settings, answer_options=[*settings.answer_options, AnswerOption(
        id="balanced", label="Balanced", provider="openrouter", model="qwen")])
    code, equation = "x = 1\nprint(x)", r"\frac{a}{b} = c"
    cases = [(code, "balanced", "balanced"), (code, "deep", "deep"), (code, "fast", "deep"),
             (equation, "balanced", "deep"), (equation, "fast", "deep"), (equation, None, "deep")]
    monkeypatch.setattr("app.chat.model_reply", fake_model(*[reply(*text.split("\n")) if text == code
                                                             else reply(text) for text, _, _ in cases]))
    client = client_for(settings, FakeOllama())
    for text, asked, used in cases:
        made = client.post("/api/explain", json={"text": text, **({"option": asked} if asked else {})})
        assert made.status_code == 200, made.text
        assert made.json()["option"] == used, (text, asked)
    assert client.get("/api/options").json()["explain_options"] == {"equation": ["deep"], "code": ["deep", "balanced"]}


def test_latex_with_lone_backslashes_is_still_read():
    from app import jsonreply

    # Balanced's reply for UDL eq. (7.6): \phi with one backslash is an invalid JSON escape.
    reply = '```json\n{"items": [{"span": "\\ell_{i}", "explanation": "loss"}, {"span": "= (f[x_{i}, \\boldsymbol{\\phi}]- y_{i})^{2}", "explanation": "squared error $\\phi$"}]}\n```'
    data = jsonreply.loads(reply)
    assert data["items"][1]["span"] == r"= (f[x_{i}, \boldsymbol{\phi}]- y_{i})^{2}"
    assert data["items"][1]["explanation"] == r"squared error $\phi$"
    assert jsonreply.loads('{"a": "line\\nbreak"}') == {"a": "line\nbreak"}   # valid JSON untouched
    assert jsonreply.loads("no json") is None
    items = explain.parse(reply, r"\ell_{i}= (f[x_{i}, \boldsymbol{\phi}]- y_{i})^{2}")
    assert items is not None and len(items) == 2


def test_an_equation_number_is_left_out(monkeypatch):
    model = fake_model(reply(r"\ell_{i}=", r"(f - y_{i})^{2}"))
    monkeypatch.setattr("app.chat.model_reply", model)
    result = asyncio.run(explain.explain(None, DEEP, r"\ell_{i}=(f - y_{i})^{2}, \tag{7.6}", "equation"))
    assert result["text"] == r"\ell_{i}=(f - y_{i})^{2}" and "tag" not in model.calls[0][1]["content"]


def test_latex_symbols_after_a_lone_backslash_are_read():
    from app import jsonreply

    # \, \% \{ and "\ " are no JSON escapes; Balanced writes them (5 of 10 Explain items failed on them).
    assert jsonreply.loads(r'{"a": "\beta^{h}\, \ell_{h}", "b": "95\% and \{x\}", "c": "\ \lambda"}') == {
        "a": r"\beta^{h}\, \ell_{h}", "b": r"95\% and \{x\}", "c": r"\ \lambda"}
    # An escaped backslash pair is kept whole, so "\\," stays one backslash and a comma; and \frac
    # in otherwise valid JSON is a fraction, not a form feed and "rac".
    assert jsonreply.loads(r'{"a": "x\\, \frac{1}{2}"}') == {"a": r"x\, \frac{1}{2}"}
    # A JSON newline or tab before ordinary text stays one; one before a known command is LaTeX.
    assert jsonreply.loads(r'{"a": "x = 1\nprint(x)\tdone", "b": "\nabla \theta \times \rho \frac{1}{2} \beta"}') == {
        "a": "x = 1\nprint(x)\tdone", "b": r"\nabla \theta \times \rho \frac{1}{2} \beta"}
    assert jsonreply.loads(r'{"a": "caf\u00e9 \underbrace{x}"}') == {"a": r"café \underbrace{x}"}


def test_latex_that_starts_like_a_json_escape_is_read():
    """Review R2-C6: \begin, \bigoplus and \nexists became a backspace, a backspace and a
    newline. \b and \f are never meant as control characters; \n \r \t only before a word."""
    from app import jsonreply

    written = r"\begin{pmatrix} a \end{pmatrix} \bigoplus_i V_i \nexists x \boxtimes \flat \rightleftharpoons \textsf{A}"
    assert jsonreply.loads('{"a": "' + written + '"}') == {"a": written}
    assert jsonreply.loads(r'{"a": "one\ntwo\rthree\tfour\nwhere"}') == {"a": "one\ntwo\rthree\tfour\nwhere"}
    assert not any(ord(c) < 32 for c in jsonreply.loads('{"a": "' + written + '"}')["a"])


def test_balanced_e01_replies_are_json_but_miscopied():
    import json
    import re
    from pathlib import Path

    fixture = json.loads((Path(__file__).parent / "fixtures" / "explain_balanced_e01.json").read_text())
    text = re.sub(r"\\tag\*?\{[^}]*\}", "", fixture["text"]).strip().rstrip(",").strip()
    from app import jsonreply

    # Both are JSON now (they failed on "\,"). Both are still refused, rightly: they copy
    # \underbrace{...}_{\mathcal{L}_{horizon}} with ^ for _, so the spans aren't the text.
    assert all(len(jsonreply.loads(reply)["items"]) >= 5 for reply in fixture["replies"])
    assert all(explain.parse(reply, text) is None for reply in fixture["replies"])


def test_the_retry_names_the_miscopied_span():
    import json
    import re
    from pathlib import Path

    fixture = json.loads((Path(__file__).parent / "fixtures" / "explain_balanced_e01.json").read_text())
    text = re.sub(r"\\tag\*?\{[^}]*\}", "", fixture["text"]).strip().rstrip(",").strip()
    assert explain.first_miss(fixture["replies"][0], text).startswith(r"\underbrace{")
    assert explain.first_miss("not json", text) is None


def test_a_fallback_restart_drops_the_first_models_words(monkeypatch):
    """Review C6, as for Suggest: the half reply before a restart is not part of the answer."""
    from app.llm.ollama import Delta

    async def model_reply(messages, settings, provider, transport, purpose, fallback, client=None):
        yield Delta(text='{"items": [{"span": "x =')
        yield Delta(restart=True)
        yield Delta(text=reply("x = 1", "y = 2"))
        yield Delta(done=True, cost_usd=0.01)

    monkeypatch.setattr("app.chat.model_reply", model_reply)
    result = asyncio.run(explain.explain(None, DEEP, "x = 1\ny = 2", "code"))
    assert [item["span"] for item in result["items"]] == ["x = 1", "y = 2"]


def test_a_local_explain_holds_the_figure_job_off_the_gpu(setup, monkeypatch):  # noqa: F811
    """Review C7: chat marks itself running so the figure job waits; a local Explain didn't,
    and the two models shared the GPU until the local one spilled to the CPU."""
    from dataclasses import replace

    from app.config import AnswerOption
    from app.ingest import figures
    from app.llm.ollama import Delta
    from tests.test_chat import FakeOllama, client_for

    settings, _, _ = setup
    settings = replace(settings, answer_options=[*settings.answer_options, AnswerOption(
        id="balanced", label="Balanced", provider="ollama", model="gemma")])
    busy = []

    async def model_reply(messages, settings_, provider, transport, purpose, fallback, client=None):
        busy.append(figures.chat_busy(settings))
        yield Delta(text=reply("x = 1", "print(x)"))
        yield Delta(done=True)

    monkeypatch.setattr("app.chat.model_reply", model_reply)
    made = client_for(settings, FakeOllama()).post("/api/explain", json={"text": "x = 1\nprint(x)", "option": "balanced"})
    assert made.status_code == 200, made.text
    assert busy == [True] and not figures.chat_busy(settings)
