"""App-driven web search (web: app): the app searches, reads the top pages and chooses their
passages; the model only answers. No network: SearXNG and the pages are a mock transport."""
import asyncio
import datetime
import json

import httpx
import pytest

from app.chat import answer_events
from app.config import Settings
from app.llm.prompts import APP_WEB_PROMPT, WEB_PROMPT, build_messages
from app.tools.executor import ToolBox
from app.tools.web import Page, Result, WebSession
from app.tools.web_gather import (GAP, MIN_PAGE_CHARS, bm25, choose_passages, excerpt, gather, pages_block,
                                  ranked_results, search_query, windows)
from test_tool_loop import embed_nothing, library, settings  # noqa: F401  (fixtures)
from test_web_tools import SEARX, fake_resolve, html_response

PRIZE = "The 2026 Nobel Prize in Physics was awarded to Francis Halzen for IceCube. " * 12


class FakeWeb:
    """SearXNG returns `order`; each page is `pages[url]` (None: 404)."""

    def __init__(self, order, pages):
        self.order = order
        self.pages = pages
        self.requests: list[str] = []
        self.search_status = 200

    def __call__(self, request):
        url = str(request.url)
        self.requests.append(url)
        if url.startswith(SEARX):
            if self.search_status != 200:
                return httpx.Response(self.search_status)
            return httpx.Response(200, json={"results": [
                {"url": u, "title": title, "content": snippet, "engines": ["yahoo"]} for title, u, snippet in self.order]})
        body = self.pages.get(url)
        if body is None:
            return html_response("<p>Not found</p>", status=404)
        return html_response(body)


def web_session(fake, message="Who won the 2026 Nobel Prize in Physics?"):
    return WebSession(Settings(searxng_url=SEARX), message, transport=httpx.MockTransport(fake),
                      resolve=fake_resolve, today=datetime.date(2026, 11, 5))


def page(paragraphs):
    return "<html><head><title>T</title></head><body>" + "".join(f"<p>{p}</p>" for p in paragraphs) + "</body></html>"


# --- Passages: windows, BM25, the excerpt ---------------------------------------------------------

def test_windows_keep_paragraphs_whole_and_cut_long_ones_at_sentences():
    text = "Short one.\n\nShort two.\n\n" + "A sentence that goes on. " * 60
    pieces = windows(text, size=200)
    assert pieces[0].startswith("Short one.\nShort two.\n")   # small paragraphs share a window
    assert all(len(piece) <= 200 for piece in pieces)
    assert " ".join(pieces).split() == text.split()                     # nothing lost, nothing added


def test_bm25_ranks_the_matching_window_first_and_a_common_word_counts_less():
    docs = ["the W boson mass is 80.3625 GeV", "the Z boson is neutral", "boson boson boson", "cats and dogs"]
    scores = bm25("W boson mass", docs)
    assert scores.index(max(scores)) == 0 and scores[3] == 0
    # "boson" is in three of four windows, "mass" in one: "mass" alone outscores "boson" alone.
    assert bm25("mass", docs)[0] > bm25("boson", docs)[1]


def test_the_excerpt_keeps_the_opening_and_the_best_windows_in_page_order():
    pieces = [f"piece {i} " + "x" * 90 for i in range(10)]
    scores = [0.0] * 10
    scores[7], scores[3] = 2.0, 1.0
    shown = excerpt(pieces, scores, budget=350)
    assert shown.split("\n") == [pieces[0], GAP, pieces[3], GAP, pieces[7], GAP]
    assert excerpt(pieces[:2], [0, 0], budget=1000) == "\n".join(pieces[:2])   # it fits: all of it


def test_passages_come_from_embeddings_and_words_and_survive_a_failed_embedder():
    text = "\n\n".join(["Intro to the prize."] + [f"Filler paragraph number {i} about weather. " * 5 for i in range(30)]
                       + ["Francis Halzen won the 2026 Nobel Prize in Physics."])
    made = Page(1, "u", "u", "t", text, False, "2026-11-05")

    def broken(texts):
        raise OSError("Ollama is down")
    choose_passages([made], "Who won the 2026 Nobel Prize in Physics?", broken, budget=2000)
    assert made.excerpt.startswith("Intro to the prize.") and "Francis Halzen won" in made.excerpt


# --- Which results are opened ----------------------------------------------------------------------

def test_results_matching_the_question_go_first_and_pdfs_and_videos_last():
    results = [Result("Paper (PDF)", "https://x.org/paper.pdf", "W boson mass"),
               Result("Video", "https://www.youtube.com/watch?v=1", "W boson mass"),
               Result("Dictionary: current", "https://dict.example/current", "current, Strom"),
               Result("W boson mass world average", "https://pdg.example/w", "PDG average of the W boson mass")]
    order = ranked_results(results, "current world average W boson mass")
    assert order[0] == "https://pdg.example/w"
    assert order[-2:] == ["https://x.org/paper.pdf", "https://www.youtube.com/watch?v=1"]


def test_a_long_question_is_cut_for_the_search_engine_on_a_word():
    query = search_query("word " * 200)
    assert len(query) <= 300 and query.endswith("word")


# --- gather -----------------------------------------------------------------------------------------

def nobel_web():
    order = [("Prize PDF", "https://n.example/prize.pdf", "Nobel Prize Physics 2026"),
             ("Tiny", "https://n.example/tiny", "Nobel Prize Physics 2026 Halzen"),
             ("Press release Nobel Prize Physics 2026", "https://n.example/press", "Halzen IceCube"),
             ("Gone", "https://n.example/gone", "Nobel Prize Physics 2026"),
             ("Nobel 2026 physics summary", "https://n.example/summary", "Physics 2026 Halzen"),
             ("Nobel 2026 physics news", "https://n.example/news", "Physics prize 2026"),
             ("Unrelated", "https://n.example/other", "cooking")]
    pages = {"https://n.example/tiny": page(["Nobel."]),
             "https://n.example/press": page([PRIZE]),
             "https://n.example/summary": page(["Summary: " + PRIZE]),
             "https://n.example/news": page(["News: " + PRIZE]),
             "https://n.example/other": page(["Recipes. " * 80])}
    return FakeWeb(order, pages)


def test_gather_reads_the_top_pages_numbered_in_order_skipping_what_cant_be_used():
    fake = nobel_web()
    web = web_session(fake)
    found = asyncio.run(gather(web, "Who won the 2026 Nobel Prize in Physics?"))
    readable = {"https://n.example/press", "https://n.example/summary", "https://n.example/news"}
    order = [url for url in ranked_results(web.results, "Who won the 2026 Nobel Prize in Physics?") if url in readable]
    assert [p.url for p in found.pages] == order and set(order) == readable
    assert [p.number for p in found.pages] == [1, 2, 3] and web.pages == found.pages
    failed = dict(found.failed)
    assert failed["https://n.example/tiny"] == "too little text to use"
    assert "404" in failed["https://n.example/gone"]
    assert not any(url.endswith(".pdf") for url in fake.requests)       # never fetched: tried last
    assert web.searches == 1 and web.fetches <= 5
    assert all(p.excerpt is not None for p in found.pages)
    assert len(page([PRIZE])) > MIN_PAGE_CHARS


def test_the_readers_own_link_is_read_first():
    fake = nobel_web()
    fake.pages["https://reader.example/notes"] = page(["My notes on the prize. " * 30])
    web = web_session(fake, "Summarise https://reader.example/notes and who won the 2026 Nobel Prize in Physics")
    found = asyncio.run(gather(web, web.reader_message))
    assert found.pages[0].url == "https://reader.example/notes" and len(found.pages) == 3


def test_a_failed_search_says_why_and_the_prompt_says_nothing_was_found():
    fake = nobel_web()
    fake.search_status = 503
    web = web_session(fake)
    found = asyncio.run(gather(web, "Who won?"))
    assert found.pages == [] and "SearXNG" in found.search_error
    block = pages_block(web, found)
    assert block.startswith("Web pages: none.") and "found nothing usable" in block


def test_the_pages_reach_the_model_between_markers_with_the_app_web_rules():
    web = web_session(nobel_web())
    found = asyncio.run(gather(web, "Who won the 2026 Nobel Prize in Physics?"))
    system, user = build_messages("Who won?", [], [], tools=True, web=True, web_pages=pages_block(web, found))
    assert APP_WEB_PROMPT in system["content"] and WEB_PROMPT not in system["content"]
    content = user["content"]
    assert f"<<<WEB 1 {web.tag}>>>" in content and f"<<<END WEB 3 {web.tag}>>>" in content
    assert "The app chose these passages" in content
    assert content.index("Web pages, found by the app") < content.index("Question: Who won?")
    assert content.rstrip().endswith("The reader's question, again: Who won?")


# --- A whole turn ------------------------------------------------------------------------------------

def test_an_app_web_turn_shows_the_search_offers_no_web_tools_and_keeps_real_citations(library):  # noqa: F811
    web = web_session(nobel_web())
    seen = []

    def ollama(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=json.dumps({
            "message": {"content": "Francis Halzen won it [Web 1], for IceCube [Web 2][Web 7]."},
            "done": True, "done_reason": "stop"}).encode() + b"\n")

    async def run():
        return [item async for item in answer_events(
            "Who won the 2026 Nobel Prize in Physics?", [], library, embed=embed_nothing, tools=True,
            transport=httpx.MockTransport(ollama), web=web, web_mode="app")]
    events = asyncio.run(run())
    names = [name for name, _ in events]
    assert names.index("tool_start") < names.index("web_source") < names.index("done")
    starts = [data for name, data in events if name == "tool_start"]
    assert starts[0] == {"id": "app-search", "name": "web_search",
                         "arguments": {"query": "Who won the 2026 Nobel Prize in Physics?"}}
    offered = {tool["function"]["name"] for tool in seen[0]["tools"]}
    assert "web_search" not in offered and "fetch_page" not in offered and "propose_web_search" not in offered
    assert "<<<WEB 1" in seen[0]["messages"][-1]["content"]
    done = dict(events)["done"]
    assert done["answer"] == "Francis Halzen won it [Web 1], for IceCube [Web 2]."
    assert done["web_citations"] == [1, 2] and done["dropped_web_citations"] == [7]
    assert [source["number"] for source in done["web_sources"]] == [1, 2, 3]


def test_a_web_tool_call_in_app_mode_is_refused_with_a_pointer_to_the_pages():
    box = ToolBox(Settings(searxng_url=SEARX), lambda query: [], {}, [], web=web_session(nobel_web()),
                  web_tools=False)
    outcome = asyncio.run(box.call("web_search", {"query": "again"}))
    assert not outcome.ok and outcome.summary == "The app searched already"
