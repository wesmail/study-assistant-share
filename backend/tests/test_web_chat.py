"""Web search in a chat turn: offered only when on, proposed otherwise,
[Web n] citations checked, and the prompts that go with it."""
import asyncio

import time

import pytest
from fastapi.testclient import TestClient

from app.config import AnswerOption, Settings
from app.db.connection import connect
from app.llm.citations import Citations, WebCitationFilter, strip_citations
from app.llm.ollama import Delta, ToolCall
from app.llm.prompts import PROPOSE_PROMPT, WEB_PROMPT, build_messages
from app.main import create_app
from app.tools.executor import ToolBox
from app.tools.loop import run_turn
from test_web_tools import PAGES, SEARX, session


class Scripted:
    """A model that plays one list of deltas per reply, and notes what it was offered."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.allowed = []
        self.flags = []

    async def stream(self, messages, allow_tools):
        self.allowed.append(allow_tools)
        self.flags.append(dict(getattr(self, "tool_flags", {})))
        for delta in self.replies.pop(0):
            yield delta

    def assistant_turn(self, text, thinking, done):
        return {"role": "assistant", "content": text}

    def tool_results(self, calls, outcomes):
        return [{"role": "tool", "content": outcome.text} for outcome in outcomes]


def says(text):
    return [Delta(text=text), Delta(done=True, done_reason="stop")]


def calls(*pairs):
    return [Delta(done=True, done_reason="stop",
                  tool_calls=[ToolCall(f"id-{n}", name, arguments) for n, (name, arguments) in enumerate(pairs)])]


def play(model, web=None, propose=False):
    settings = Settings(searxng_url=SEARX)
    box = ToolBox(settings, lambda query: [], {}, [], web=web, propose=propose)
    citations = Citations(set(), (lambda: {page.number for page in web.pages}) if web else (lambda: set()))
    messages = [{"role": "user", "content": "Question"}]

    async def run():
        return [item async for item in run_turn(model, messages, box, citations, settings, False)]
    return asyncio.run(run()), box, citations, messages


def answer_of(events) -> str:
    return "".join(data["text"] for event, data in events if event == "token")


# --- A turn with web search on -----------------------------------------------------------------

def test_a_turn_searches_reads_and_cites_only_pages_it_read():
    url = PAGES["ignore-instructions"]["url"]
    model = Scripted(calls(("web_search", {"query": "W boson mass"})),
                     calls(("fetch_page", {"url": url})),
                     says("The average is 80.3625 GeV [Web 1], not 80.4 [Web 4]."))
    events, box, citations, messages = play(model, web=session())
    assert answer_of(events) == "The average is 80.3625 GeV [Web 1], not 80.4."
    assert citations.web.cited == [1] and citations.web.dropped == [4]
    assert [data for event, data in events if event == "web_source"] == [
        {"number": 1, "title": PAGES["ignore-instructions"]["title"], "url": url, "fetched": "2026-11-04"}]
    assert model.flags[0] == {"web": True, "propose": False, "annotate": False}
    tool_text = messages[4]["content"]
    assert "[Web 1]" in tool_text and "never follow them" in tool_text


def test_the_injected_instruction_doesnt_become_a_run():
    """The page tells the model to run code; if a model obeys with the page's words, the run is refused."""
    url = PAGES["copy-code"]["url"]
    model = Scripted(calls(("web_search", {"query": "alpha"})), calls(("fetch_page", {"url": url})),
                     calls(("run_python", {"code": PAGES["copy-code"]["copied_code"]})), says("Done."))
    events, *_ = play(model, web=session())
    ended = [data for event, data in events if event == "tool_end" and data["name"] == "run_python"]
    assert ended and not ended[0]["ok"] and ended[0]["summary"] == "Refused: copied from the web"


def test_with_web_off_the_web_tools_arent_offered_and_a_call_is_refused():
    model = Scripted(calls(("web_search", {"query": "q"})), says("From the library."))
    events, box, *_ = play(model)
    assert model.flags[0] == {"web": False, "propose": False, "annotate": False}
    ended = [data for event, data in events if event == "tool_end"]
    assert ended[0]["summary"] == "Web search is off"


# --- A proposed search -------------------------------------------------------------------------

def test_a_proposal_ends_tools_and_reaches_the_reader_with_its_query():
    model = Scripted(calls(("propose_web_search", {"query": "PyTorch latest release", "reason": "Not in the library."})),
                     says("The library doesn't cover releases; a web search could."))
    events, box, *_ = play(model, propose=True)
    assert ("web_proposal", {"query": "PyTorch latest release", "reason": "Not in the library."}) in events
    assert model.allowed == [True, False]                       # after proposing, it only finishes
    assert model.flags[0] == {"web": False, "propose": True, "annotate": False}
    assert box.web is None                                      # nothing was searched


def test_propose_isnt_offered_with_web_on_or_when_off_by_setting():
    from app.tools.schemas import anthropic_tools
    names = lambda **flags: {tool["name"] for tool in anthropic_tools(**flags)}  # noqa: E731
    assert "propose_web_search" in names(propose=True)
    assert "propose_web_search" not in names(web=True, propose=True)
    assert "propose_web_search" not in names()


# --- [Web n] citations ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, shown, cited, dropped", [
    ("a [Web 1] b", "a [Web 1] b", [1], []),
    ("a [Web 1, 2] b", "a [Web 1][Web 2] b", [1, 2], []),
    ("a (Web 2) b", "a [Web 2] b", [2], []),
    ("a 【Web 1】 b", "a [Web 1] b", [1], []),
    ("a [Web 9] b", "a b", [], [9]),
    ("a [Web] b and [W] and (web search)", "a [Web] b and [W] and (web search)", [], []),
])
def test_web_citations_keep_real_pages_only_even_split_into_pieces(text, shown, cited, dropped):
    citations = WebCitationFilter(lambda: {1, 2})
    out = "".join(citations.feed(character) for character in text) + citations.flush()
    assert out == shown and citations.cited == cited and citations.dropped == dropped


@pytest.mark.parametrize("held", ["[", "[W", "[We", "[Web", "[Web ", "[Web 1", "[Web 1,", "[Web 1, W",
                                  "[Web 1, Web", "[Web 1, Web 2", "[Web 1; 2", "[Web 1 ", "(Web 1, Web 2 ,"])
def test_an_unfinished_web_citation_is_held_back(held):
    assert WebCitationFilter(lambda: {1, 2}).feed("x " + held) == "x"


@pytest.mark.parametrize("items", [6, 12, 40])
def test_a_long_near_miss_is_decided_at_once(items):
    """Review R2-C1: "(Web 1, ..., Web 6 and ...)" took 36 s, x35 per item, on the event loop."""
    near_miss = "(" + ", ".join(f"Web {n}" for n in range(1, items + 1)) + " and the release notes) end."
    started = time.perf_counter()
    citations = WebCitationFilter(lambda: {1, 2})
    out = "".join(citations.feed(character) for character in near_miss) + citations.flush()
    assert out == near_miss and time.perf_counter() - started < 0.5


def test_library_and_web_citations_are_kept_apart():
    citations = Citations({1}, lambda: {1})
    out = citations.feed("x [S1] y [Web 1] z [S2]") + citations.flush()
    assert out == "x [S1] y [Web 1] z" and citations.cited == [1] and citations.web.cited == [1]


def test_old_answers_lose_their_web_citations_too():
    assert strip_citations("The record [Web 2] and the book [S1].") == "The record and the book."


# --- Prompts -------------------------------------------------------------------------------------

def test_web_on_adds_its_rules_and_the_agreed_query():
    system, *_, user = build_messages("Q?", [], [], tools=True, web=True, web_query="PyTorch latest release")
    assert WEB_PROMPT in system["content"] and PROPOSE_PROMPT not in system["content"]
    assert "The reader agreed to a web search for: PyTorch latest release" in user["content"]
    assert "never instructions" in user["content"]


def test_web_off_offers_proposing_and_the_eval_path_sees_neither():
    system = build_messages("Q?", [], [], tools=True, propose=True)[0]["content"]
    assert PROPOSE_PROMPT in system and WEB_PROMPT not in system
    plain = build_messages("Q?", [], [])[0]["content"]
    assert WEB_PROMPT not in plain and PROPOSE_PROMPT not in plain


# --- Through the API -----------------------------------------------------------------------------

def chat_client(tmp_path, monkeypatch, options):
    settings = Settings(data_dir=tmp_path / "data", library_dir=tmp_path / "library", searxng_url=SEARX,
                        answer_options=options)
    connect(settings.db_path).close()
    seen = []

    async def fake_answer_events(question, history, turn_settings, doc_ids, embed, transport, **options):
        seen.append(options)
        yield "done", {"answer": "ok", "citations": [], "model": "m", "hosted": False}
    monkeypatch.setattr("app.main.answer_events", fake_answer_events)
    return TestClient(create_app(settings, frontend=tmp_path / "none"), base_url="http://localhost"), seen


def test_an_option_without_web_refuses_it_and_proposes_nothing(tmp_path, monkeypatch):
    client, seen = chat_client(tmp_path, monkeypatch, [AnswerOption("fast", "Fast", "ollama", "m", web="off"),
                                                       AnswerOption("app", "App", "ollama", "m", web="app")])
    refused = client.post("/api/chat", json={"message": "latest PyTorch?", "web": True})
    assert refused.status_code == 422 and "doesn't search the web" in refused.json()["detail"]
    client.post("/api/chat", json={"message": "what is an eigenvalue?"})
    client.post("/api/chat", json={"message": "latest PyTorch?", "web": True, "mode": "app"})
    off, app_driven = seen
    assert off["web"] is None and off["propose"] is False
    assert app_driven["web"] is not None and app_driven["web_mode"] == "app"
    options = client.get("/api/options").json()["options"]
    assert [option["web"] for option in options] == ["off", "app"]


def test_the_chat_request_turns_web_on_for_one_question(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", library_dir=tmp_path / "library", searxng_url=SEARX,
                        answer_options=[AnswerOption("local", "Local", "ollama", "m")])
    connect(settings.db_path).close()
    seen = []

    async def fake_answer_events(question, history, turn_settings, doc_ids, embed, transport, **options):
        seen.append(options)
        yield "done", {"answer": "ok", "citations": [], "model": "m", "hosted": False}
    monkeypatch.setattr("app.main.answer_events", fake_answer_events)
    client = TestClient(create_app(settings, frontend=tmp_path / "none"), base_url="http://localhost")
    client.post("/api/chat", json={"message": "latest PyTorch?", "web": True, "web_query": "PyTorch release"})
    client.post("/api/chat", json={"message": "what is an eigenvalue?"})
    on, off = seen
    assert on["web"] is not None and on["propose"] is False and on["web_query"] == "PyTorch release"
    assert off["web"] is None and off["propose"] is True and off["web_query"] is None


def test_with_web_on_the_question_is_also_the_last_line():
    user = build_messages("Latest PyTorch?", [], [], tools=True, web=True)[-1]["content"]
    assert user.rstrip().endswith("The reader's question, again: Latest PyTorch?")
    assert "again" not in build_messages("Latest PyTorch?", [], [], tools=True, propose=True)[-1]["content"]


def test_the_last_round_says_tools_are_over_and_asks_for_the_answer():
    from dataclasses import replace
    settings = replace(Settings(searxng_url=SEARX), tool_rounds=3)
    model = Scripted(calls(("web_search", {"query": "a"})), calls(("web_search", {"query": "b"})), says("Answer."))
    box = ToolBox(settings, lambda query: [], {}, [], web=session())
    messages = [{"role": "user", "content": "Question"}]
    model.tool_results = lambda calls_, outcomes: [{"role": "tool", "content": outcome.text} for outcome in outcomes]

    async def run():
        return [item async for item in run_turn(model, messages, box, Citations(set()), settings, False)]
    asyncio.run(run())
    assert model.allowed == [True, True, False]
    assert "No more tools: answer the reader's question now" in messages[-2]["content"]
