"""Phase 5.1, step 3: figures in sources, their images to vision options, and the figure endpoint."""
import asyncio
import base64
import json
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from app.chat import answer_events, figure_images, figures_of, source_record
from app.ingest.embed import serialize
from app.config import AnswerOption, Settings
from app.db.connection import connect
from app.llm import hosted
from app.llm.ollama import Delta
from app.llm.prompts import SYSTEM_PROMPT, with_images
from app.main import create_app
from app.retrieval.search import Hit

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


@pytest.fixture
def library(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", library_dir=tmp_path / "library", tools_enabled=False)
    conn = connect(settings.db_path)
    folder = settings.parsed_dir / "doc-sha" / "figures"
    folder.mkdir(parents=True)
    for n in range(1, 7):
        (folder / f"figure_00{n}_page{n}.png").write_bytes(PNG)
    conn.execute("INSERT INTO documents (id, path, sha256, title, kind, pages, status) "
                 "VALUES (1, 'a.pdf', 'doc-sha', 'Book', 'pdf', 9, 'indexed')")
    conn.execute("INSERT INTO chunks (id, doc_id, kind, seq, text, token_count, page_start, page_end, meta) "
                 "VALUES (1, 1, 'text', 0, 'A passage.', 3, 1, 1, '{}')")
    for n in range(1, 7):
        name = f"figures/figure_00{n}_page{n}.png"
        conn.execute("INSERT INTO figures (id, doc_id, page, image_path, caption) VALUES (?, 1, ?, ?, ?)",
                     (n, n, name, f"Figure 8.{n} A curve."))
        conn.execute("INSERT INTO chunks (id, doc_id, kind, seq, text, token_count, page_start, page_end, meta) "
                     "VALUES (?, 1, 'figure', ?, 'Figure caption.', 3, ?, ?, ?)",
                     (n + 1, n, n, n, json.dumps({"image_path": name})))
    conn.commit()
    return settings, conn


def hit(chunk_ids, number=1):
    return Hit(chunk_id=chunk_ids[0], doc_id=1, doc_path="a.pdf", doc_title="Book", heading_path=[], text="Text",
               page_start=1, page_end=1, block_ids=[], meta={}, score=1.0, chunk_ids=list(chunk_ids))


class TestSources:
    def test_a_passage_knows_the_figures_inside_it_merged_neighbours_too(self, library):
        settings, conn = library
        found = figures_of(conn, hit([1, 2, 3]))
        assert [figure["figure_id"] for figure in found] == [1, 2]
        assert found[0]["url"] == "/api/documents/1/figures/figure_001_page1.png"
        assert found[0]["caption"] == "Figure 8.1 A curve."

    def test_a_text_only_passage_has_no_figures(self, library):
        assert figures_of(library[1], hit([1])) == []

    def test_the_source_record_gives_the_ui_each_figures_image(self, library):
        settings, conn = library
        passage = hit([1, 2])
        passage.figures = figures_of(conn, passage)
        record = source_record(1, passage)
        assert record["figures"] == [{"url": "/api/documents/1/figures/figure_001_page1.png", "page": 1,
                                      "caption": "Figure 8.1 A curve."}]
        assert "sha256" not in json.dumps(record), "no file paths leave the server"


class TestImages:
    def test_images_follow_the_source_order_and_stop_at_the_limit(self, library):
        settings, conn = library
        passages = [hit([5, 6], 1), hit([2, 3, 4], 2)]
        for passage in passages:
            passage.figures = figures_of(conn, passage)
        images = figure_images(passages, replace(settings, figure_images_max=4))
        # Each source's nearest figure first, then each one's next.
        assert [(image["source"], image["page"]) for image in images] == [(1, 4), (2, 1), (1, 5), (2, 2)]
        assert images[0]["media_type"] == "image/png" and base64.b64decode(images[0]["data"]) == PNG

    def test_a_missing_image_file_is_skipped(self, library):
        settings, conn = library
        (settings.parsed_dir / "doc-sha" / "figures" / "figure_001_page1.png").unlink()
        passage = hit([2, 3])
        passage.figures = figures_of(conn, passage)
        assert [image["page"] for image in figure_images([passage], settings)] == [2]

    def test_anthropic_gets_each_image_with_its_source_before_the_question(self):
        messages = [{"role": "system", "content": "Rules"}, {"role": "user", "content": "Sources: ...\n\nQuestion: q"}]
        images = [{"source": 3, "page": 90, "caption": "Figure 8.10 Double descent.", "media_type": "image/png", "data": "AAA"}]
        content = with_images(messages, images, "anthropic")[-1]["content"]
        assert content[0] == {"type": "text", "text": "[S3] Figure 8.10, page 90"}
        assert content[1]["type"] == "image" and content[1]["source"]["data"] == "AAA"
        assert content[2]["text"].endswith("Sources: ...\n\nQuestion: q")

    def test_ollama_gets_the_images_listed_in_order(self):
        messages = [{"role": "user", "content": "Sources: ...\n\nQuestion: q"}]
        images = [{"source": 1, "page": 5, "caption": "", "media_type": "image/png", "data": "AAA"},
                  {"source": 2, "page": 7, "caption": "Figure 2.1 Curve.", "media_type": "image/png", "data": "BBB"}]
        last = with_images(messages, images, "ollama")[-1]
        assert last["images"] == ["AAA", "BBB"]
        assert last["content"].startswith("Images attached, in this order: "
                                          "1. [S1] A figure, page 5; 2. [S2] Figure 2.1, page 7.")

    def test_no_images_leaves_the_messages_alone(self):
        messages = [{"role": "user", "content": "q"}]
        assert with_images(messages, [], "anthropic") is messages

    def test_the_prompt_says_descriptions_are_approximate_and_the_image_wins(self):
        assert "values were read by eye and are approximate" in SYSTEM_PROMPT
        assert "read values from the image itself, and trust it over the description" in SYSTEM_PROMPT


def answers(settings, monkeypatch, provider, vision):
    sent = []

    def fake_sources(settings_, query, doc_ids, embed, cited=()):
        passage = hit([1, 2])
        passage.figures = figures_of(connect(settings.db_path), passage)
        return [passage]

    async def deep(messages, settings_, purpose="chat", fallback=True, **kw):
        sent.append(messages)
        yield Delta(text="It rises [S1].")
        yield Delta(done=True, done_reason="end_turn", model="claude-opus-5-5")
    monkeypatch.setattr("app.chat.find_sources", fake_sources)
    monkeypatch.setattr(hosted, "answer_stream", deep)

    def local(request):
        sent.append(json.loads(request.content)["messages"])
        return httpx.Response(200, content=b'{"message": {"content": "It rises [S1]."}, "done": true, "done_reason": "stop"}\n')

    async def run():
        return [e async for e in answer_events("What does Figure 8.1 show?", [], settings, embed=None,
                                               transport=httpx.MockTransport(local), provider=provider, vision=vision)]
    events = asyncio.run(run())
    return events, sent[-1]


class TestAnswers:
    def test_deep_sends_the_figure_image(self, library, monkeypatch):
        events, sent = answers(library[0], monkeypatch, "anthropic", vision=True)
        assert any(block.get("type") == "image" for block in sent[-1]["content"])
        assert events[-1][0] == "done"

    def test_balanced_sends_it_to_ollama(self, library, monkeypatch):
        _, sent = answers(library[0], monkeypatch, "ollama", vision=True)
        assert len(sent[-1]["images"]) == 1

    def test_fast_sends_no_image_only_the_text(self, library, monkeypatch):
        _, sent = answers(library[0], monkeypatch, "ollama", vision=False)
        assert "images" not in sent[-1] and isinstance(sent[-1]["content"], str)

    def test_the_sources_event_carries_the_figure(self, library, monkeypatch):
        events, _ = answers(library[0], monkeypatch, "ollama", vision=False)
        sources = dict(events)["sources"]["sources"]
        assert sources[0]["figures"][0]["url"] == "/api/documents/1/figures/figure_001_page1.png"


class TestEndpoint:
    @pytest.fixture
    def client(self, library):
        settings, _ = library
        return TestClient(create_app(settings, embed=lambda t, s: [[1.0] * 1024 for _ in t]), base_url="http://localhost")

    def test_a_figure_is_served_as_a_png(self, client):
        response = client.get("/api/documents/1/figures/figure_001_page1.png")
        assert response.status_code == 200 and response.content == PNG
        assert response.headers["content-type"] == "image/png"
        assert response.headers["x-content-type-options"] == "nosniff"

    @pytest.mark.parametrize("name", ["..%2F..%2Fapp.sqlite", "figure_001_page1.svg", "blocks.jsonl", "figure_999_page9.png"])
    def test_anything_else_is_refused(self, client, name):
        assert client.get(f"/api/documents/1/figures/{name}").status_code == 404

    def test_an_unknown_document_is_refused(self, client):
        assert client.get("/api/documents/9/figures/figure_001_page1.png").status_code == 404

    def test_the_options_say_which_read_images(self, library):
        settings, _ = library
        options = [AnswerOption("fast", "Fast", "ollama", "gpt-oss:20b"),
                   AnswerOption("deep", "Deep", "anthropic", "claude-opus-5-5", vision=True)]
        client = TestClient(create_app(replace(settings, answer_options=options), embed=lambda t, s: [[1.0] * 1024 for _ in t]),
                            base_url="http://localhost")
        assert [(o["id"], o["vision"]) for o in client.get("/api/options").json()["options"]] == [("fast", False), ("deep", True)]



class TestReaderImages:
    @pytest.fixture
    def client(self, library):
        settings, _ = library
        options = [AnswerOption("fast", "Fast", "ollama", "gpt-oss:20b"),
                   AnswerOption("balanced", "Balanced", "ollama", "gemma4", vision=True)]
        sent = []

        def ollama(request):
            if request.url.path == "/api/chat":
                sent.append(json.loads(request.content))
            return httpx.Response(200, content=b'{"message": {"content": "A plot."}, "done": true, "done_reason": "stop"}\n')
        client = TestClient(create_app(replace(settings, answer_options=options), transport=httpx.MockTransport(ollama),
                                       embed=lambda t, s: [[1.0] * 1024 for _ in t]), base_url="http://localhost")
        client.sent = sent
        return client

    IMAGE = {"media_type": "image/png", "data": base64.b64encode(PNG).decode()}

    def test_a_vision_option_gets_the_readers_image_first(self, client):
        response = client.post("/api/chat", json={"message": "What is this plot?", "mode": "balanced", "images": [self.IMAGE]})
        assert response.status_code == 200
        last = client.sent[-1]["messages"][-1]
        assert last["images"][0] == self.IMAGE["data"]
        assert last["content"].startswith("Images attached, in this order: 1. The reader's own image")

    def test_fast_refuses_an_image_before_anything_is_stored(self, client, library):
        response = client.post("/api/chat", json={"message": "What is this?", "mode": "fast", "images": [self.IMAGE]})
        assert response.status_code == 422 and "can't read images" in response.json()["detail"]
        assert client.sent == []
        assert library[1].execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0

    @pytest.mark.parametrize("image, message", [
        ({"media_type": "image/svg+xml", "data": "PHN2Zz4="}, "PNG, JPEG, WebP or GIF"),
        ({"media_type": "image/png", "data": "not base64!"}, "valid base64"),
        ({"media_type": "image/png", "data": base64.b64encode(b"x" * (5 * 1024 * 1024 + 1)).decode()}, "the limit is 5 MB"),
    ])
    def test_a_bad_image_is_refused(self, client, image, message):
        response = client.post("/api/chat", json={"message": "What?", "mode": "balanced", "images": [image]})
        assert response.status_code == 422 and message in response.json()["detail"]

    def test_at_most_three_images(self, client):
        response = client.post("/api/chat", json={"message": "What?", "mode": "balanced", "images": [self.IMAGE] * 4})
        assert response.status_code == 422

    def test_the_reader_image_goes_to_anthropic_as_a_block_before_any_figure(self):
        messages = [{"role": "user", "content": "Sources\n\nQuestion: q"}]
        images = [{"reader": True, "media_type": "image/png", "data": "R"},
                  {"source": 1, "page": 5, "caption": "Figure 1.1 A.", "media_type": "image/png", "data": "F"}]
        content = with_images(messages, images, "anthropic")[-1]["content"]
        assert content[0]["text"] == "The reader's own image, sent with this question" and content[1]["source"]["data"] == "R"
        assert content[2]["text"] == "[S1] Figure 1.1, page 5" and content[3]["source"]["data"] == "F"



class TestRelevance:
    def test_a_merged_passage_puts_the_figure_it_was_found_by_first(self, library):
        settings, conn = library
        # Found by chunk 6 (figure 5), with neighbours on both sides merged in, as with Figure 8.10.
        passage = hit([2, 3, 4, 5, 6])
        passage.chunk_id = 6
        found = figures_of(conn, passage)
        assert found[0]["figure_id"] == 5
        assert [figure["figure_id"] for figure in found[1:3]] == [4, 3], "then the nearest neighbours"

    def test_the_figure_it_was_found_by_is_sent_even_in_a_crowded_passage(self, library):
        settings, conn = library
        passage = hit([2, 3, 4, 5, 6, 7])
        passage.chunk_id = 7
        passage.figures = figures_of(conn, passage)
        sent = figure_images([passage], replace(settings, figure_images_max=4))
        assert sent[0]["page"] == 6 and len(sent) == 4

    def test_an_attached_image_means_this_is_the_image_not_the_previous_turn(self, library, monkeypatch):
        settings, _ = library
        searched = []
        monkeypatch.setattr("app.chat.find_sources",
                            lambda s, query, *a, **k: searched.append(query) or [])
        async def rewrite(*a, **k):
            raise AssertionError("an attached image skips the follow-up rewrite")
        monkeypatch.setattr("app.chat.standalone_question", rewrite)
        history = [{"role": "user", "content": "What is double descent?"}, {"role": "assistant", "content": "A curve."}]

        async def run():
            return [e async for e in answer_events("What does this plot show?", history, settings, embed=None,
                                                   transport=httpx.MockTransport(lambda r: httpx.Response(
                                                       200, content=b'{"message": {"content": "A plot."}, "done": true}\n')),
                                                   provider="ollama", vision=True,
                                                   reader_images=[{"media_type": "image/png", "data": "AAA"}])]
        asyncio.run(run())
        assert searched == ["What does this plot show?"]


def test_an_answers_cost_counts_cached_input_too(tmp_path):
    settings = Settings(data_dir=tmp_path)
    conn = connect(settings.db_path)
    conn.execute("INSERT INTO hosted_calls (purpose, model, input_tokens, output_tokens, cache_read_tokens, "
                 "cache_write_tokens, cost_usd, seconds, turn) VALUES ('chat', 'claude-opus-5-5', 4, 1510, 9000, 2000, 0.1, 1, 't')")
    conn.commit()
    assert hosted.turn_cost(settings, "t")["input_tokens"] == 11004



def test_the_page_always_revalidates_and_hashed_scripts_are_kept(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", library_dir=tmp_path)
    connect(settings.db_path).close()
    built = tmp_path / "dist"
    (built / "assets").mkdir(parents=True)
    (built / "index.html").write_text("<html></html>")
    (built / "assets" / "index-abc123.js").write_text("// app")
    client = TestClient(create_app(settings, frontend=built, embed=lambda t, s: []), base_url="http://localhost")
    assert client.get("/").headers["cache-control"] == "no-cache"
    assert client.get("/index.html").headers["cache-control"] == "no-cache"
    assert "immutable" in client.get("/assets/index-abc123.js").headers["cache-control"]
    assert "cache-control" not in client.get("/api/options").headers



def vectors(conn, chosen_chunk):
    """Figure chunks pointing away from the question, except one pointing at it."""
    question = [1.0] + [0.0] * 1023
    for chunk_id in range(2, 8):
        vector = question if chunk_id == chosen_chunk else [0.0, 1.0] + [0.0] * 1022
        conn.execute("INSERT OR REPLACE INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)", (chunk_id, serialize(vector, 1024)))
    conn.commit()
    return question


class TestSimilarity:
    def test_the_figure_closest_to_the_question_comes_first_however_far_from_the_match(self, library):
        settings, conn = library
        question = vectors(conn, chosen_chunk=7)        # figure 6, at the far end of the passage
        passage = hit([2, 3, 4, 5, 6, 7])
        passage.chunk_id = 2                            # matched on the first figure, as with Figure 8.7
        found = figures_of(conn, passage, question)
        assert found[0]["figure_id"] == 6 and found[0]["score"] == pytest.approx(1.0)

    def test_the_images_sent_are_the_closest_across_all_sources(self, library):
        settings, conn = library
        question = vectors(conn, chosen_chunk=6)        # figure 5, in the second source
        first, second = hit([2, 3, 4]), hit([5, 6, 7])
        first.chunk_id, second.chunk_id = 2, 5
        first.figures, second.figures = figures_of(conn, first, question), figures_of(conn, second, question)
        sent = figure_images([first, second], replace(settings, figure_images_max=1))
        assert [(image["source"], image["page"]) for image in sent] == [(2, 5)]

    def test_find_sources_ranks_the_figures_with_the_question_it_was_asked(self, library, monkeypatch):
        from app import chat
        settings, conn = library
        vectors(conn, chosen_chunk=4)
        passage = hit([2, 3, 4])
        monkeypatch.setattr(chat, "search_chunks", lambda *a, **k: [passage])
        monkeypatch.setattr(chat, "expand_hits", lambda conn, hits, *a, **k: hits)
        monkeypatch.setattr(chat, "carried_hits", lambda *a: [])
        asked = []
        found = chat.find_sources(settings, "What does Figure 8.3 show?", None,
                                  lambda texts, s: asked.extend(texts) or [[1.0] + [0.0] * 1023 for _ in texts])
        assert "What does Figure 8.3 show?" in asked
        assert found[0].figures[0]["figure_id"] == 3
