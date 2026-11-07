"""Figure descriptions: cached by image, put into search, never run without asking."""
import asyncio
import base64
import json
import os
import time
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from app.config import FigureDescriber, Settings
from app.db.connection import connect
from app.ingest import figures
from app.ingest.embed import serialize
from app.llm.ollama import Delta

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
OTHER_PNG = PNG + b"\x00"     # bytes after the end marker: a different file, the same picture
LOCAL = FigureDescriber("ollama", "vision-model")


@pytest.fixture
def library(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", library_dir=tmp_path / "library", figure_describer=LOCAL)
    conn = connect(settings.db_path)
    parsed = settings.parsed_dir / "doc-sha" / "figures"
    parsed.mkdir(parents=True)
    (parsed / "one.png").write_bytes(PNG)
    (parsed / "copy.png").write_bytes(PNG)          # the same image again, elsewhere in the book
    (parsed / "bare.png").write_bytes(OTHER_PNG)    # a figure the parser found no caption for
    conn.execute("INSERT INTO documents (id, path, sha256, title, kind, pages, status) "
                 "VALUES (1, 'a.pdf', 'doc-sha', 'Book', 'pdf', 9, 'indexed')")
    conn.execute('INSERT INTO blocks (id, doc_id, "order", kind, page, text) VALUES (1, 1, 1, "paragraph", 3, "The curve rises.")')
    for name, page, caption in (("one", 3, "Figure 1.1 A curve."), ("copy", 5, "Figure 2.1 The same curve."), ("bare", 7, None)):
        conn.execute("INSERT INTO figures (doc_id, page, image_path, caption) VALUES (1, ?, ?, ?)",
                     (page, f"figures/{name}.png", caption))
    conn.execute("INSERT INTO chunks (id, doc_id, kind, seq, heading_path, text, token_count, block_ids, page_start, "
                 "page_end, meta) VALUES (1, 1, 'text', 0, '[\"Ch 1\"]', 'Ch 1 ›\n\nThe curve rises.', 5, '[1]', 3, 3, '{}')")
    conn.execute("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (1, ?)", (serialize([1.0] + [0.0] * 1023, 1024),))
    for chunk_id, seq, name, caption, page in ((2, 1, "one", "Figure 1.1 A curve.", 3), (3, 2, "copy", "Figure 2.1 The same curve.", 5)):
        conn.execute("INSERT INTO chunks (id, doc_id, kind, seq, heading_path, text, token_count, block_ids, page_start, "
                     "page_end, meta) VALUES (?, 1, 'figure', ?, '[\"Ch 1\"]', ?, 5, '[]', ?, ?, ?)",
                     (chunk_id, seq, f"Ch 1 ›\n\n{caption}", page, page, json.dumps({"image_path": f"figures/{name}.png"})))
        conn.execute("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)", (chunk_id, serialize([1.0] + [0.0] * 1023, 1024)))
    conn.commit()
    figures.fill_image_shas(conn, settings)
    return settings, conn


def vision(text="A rising curve, from about 0 to about 5.", seen=None, status=200):
    def answer(request):
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(status, json={"message": {"content": text}, "done_reason": "stop"})
    return httpx.MockTransport(answer)


def embed_ones(texts, settings):
    return [[1.0] + [0.0] * 1023 for _ in texts]


def run(settings, conn, rows, **options):
    lines = []
    result = asyncio.run(figures.describe_pending(conn, settings, rows, embed=embed_ones, report=lines.append, **options))
    return result, lines


class TestCache:
    def test_the_same_image_twice_is_described_once(self, library):
        settings, conn = library
        waiting = figures.pending(conn, settings)
        assert len(waiting) == 2, "one.png and copy.png are the same image"
        seen = []
        run(settings, conn, waiting, transport=vision(seen=seen))
        assert len(seen) == 2 and figures.pending(conn, settings) == []

    def test_a_new_describer_or_prompt_starts_again(self, library, monkeypatch):
        settings, conn = library
        run(settings, conn, figures.pending(conn, settings), transport=vision())
        assert figures.pending(conn, replace(settings, figure_describer=FigureDescriber("ollama", "another")))
        monkeypatch.setattr(figures, "PROMPT_VERSION", figures.PROMPT_VERSION + 1)
        assert figures.pending(conn, settings)

    def test_a_failure_is_not_cached_and_is_tried_next_time(self, library):
        settings, conn = library
        result, lines = run(settings, conn, figures.pending(conn, settings), transport=vision(status=500))
        assert result["failed"] == 2 and result["described"] == 0
        assert any("FAILED, will be tried next time" in line for line in lines)
        assert len(figures.pending(conn, settings)) == 2

    def test_an_empty_reply_is_a_failure_not_a_description(self, library):
        settings, conn = library
        result, _ = run(settings, conn, figures.pending(conn, settings), transport=vision(text="  "))
        assert result["failed"] == 2 and conn.execute("SELECT COUNT(*) FROM figure_descriptions").fetchone()[0] == 0


class TestIntoSearch:
    def test_a_captioned_figure_gets_its_description_under_a_label(self, library):
        settings, conn = library
        run(settings, conn, figures.pending(conn, settings), transport=vision())
        text = conn.execute("SELECT text FROM chunks WHERE id = 2").fetchone()[0]
        assert text.startswith("Ch 1 ›\n\nFigure 1.1 A curve.\n\n" + figures.MARKER)
        assert text.endswith("A rising curve, from about 0 to about 5.")
        assert "approximate" in figures.MARKER and "AI-written" in figures.MARKER

    def test_both_places_of_the_same_image_get_it(self, library):
        settings, conn = library
        run(settings, conn, figures.pending(conn, settings), transport=vision())
        assert figures.MARKER in conn.execute("SELECT text FROM chunks WHERE id = 3").fetchone()[0]

    def test_applying_again_changes_nothing(self, library):
        settings, conn = library
        run(settings, conn, figures.pending(conn, settings), transport=vision())
        before = [row[0] for row in conn.execute("SELECT text FROM chunks ORDER BY id")]
        assert figures.apply_descriptions(conn, settings) == 0
        assert [row[0] for row in conn.execute("SELECT text FROM chunks ORDER BY id")] == before

    def test_an_uncaptioned_figure_gets_a_chunk_outside_the_reading_order(self, library):
        settings, conn = library
        run(settings, conn, figures.pending(conn, settings), transport=vision())
        row = conn.execute("SELECT * FROM chunks WHERE json_extract(meta, '$.standalone')").fetchone()
        assert row["kind"] == "figure" and row["page_start"] == 7 and row["seq"] < 0
        assert "no caption found" in row["text"] and figures.MARKER in row["text"]
        seqs = [seq for (seq,) in conn.execute("SELECT seq FROM chunks WHERE id != ?", (row["id"],))]
        assert all(abs(seq - row["seq"]) > 2 for seq in seqs), "so neighbour expansion never reaches it"

    def test_changed_chunks_are_embedded_again_and_found_by_keyword(self, library):
        settings, conn = library
        result, _ = run(settings, conn, figures.pending(conn, settings), transport=vision(text="A rising sigmoid curve."))
        assert result["embedded"] == 3
        missing = conn.execute("SELECT COUNT(*) FROM chunks c LEFT JOIN chunk_vec v ON v.chunk_id = c.id "
                               "WHERE v.chunk_id IS NULL").fetchone()[0]
        assert missing == 0
        hits = conn.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'sigmoid'").fetchall()
        assert len(hits) == 3

    def test_a_rechunked_book_gets_its_cached_descriptions_back(self, library):
        settings, conn = library
        run(settings, conn, figures.pending(conn, settings), transport=vision())
        conn.execute("UPDATE chunks SET text = 'Ch 1 ›\n\nFigure 1.1 A curve.' WHERE id = 2")
        conn.commit()
        assert figures.apply_descriptions(conn, settings, [1]) == 1
        assert figures.MARKER in conn.execute("SELECT text FROM chunks WHERE id = 2").fetchone()[0]


class TestDescribers:
    def test_a_hosted_describer_sends_the_image_and_logs_the_call(self, library, monkeypatch):
        settings, conn = library
        settings = replace(settings, figure_describer=FigureDescriber("anthropic", "claude-sonnet-5-5", effort="high"))
        seen = []

        async def stream(messages, turn, client=None, purpose="chat", tools=None, allow_tools=True):
            seen.append((messages, turn, purpose))
            yield Delta(text="A curve.")
            yield Delta(done=True, done_reason="end_turn", cost_usd=0.01)
        monkeypatch.setattr(figures.hosted, "stream_chat", stream)
        result, _ = run(settings, conn, figures.pending(conn, settings)[:1])
        messages, turn, purpose = seen[0]
        image, text = messages[1]["content"]
        assert image["type"] == "image" and image["source"]["media_type"] == "image/png"
        assert "Figure 1.1 A curve." in text["text"] and "The curve rises." in text["text"]
        assert turn.hosted_model == "claude-sonnet-5-5" and purpose == "figures"
        assert result["spent_usd"] == pytest.approx(0.01)

    def test_a_local_describer_sends_the_image_to_ollama(self, library):
        settings, conn = library
        seen = []
        run(settings, conn, figures.pending(conn, settings)[:1], transport=vision(seen=seen))
        assert seen[0]["model"] == "vision-model" and seen[0]["messages"][1]["images"]
        assert "think" not in seen[0]

    def test_the_estimate_prices_the_measured_tokens(self):
        settings = Settings(figure_describer=FigureDescriber("anthropic", "claude-sonnet-5-5", expected_tokens=(1302, 770)))
        assert figures.estimate_usd(settings, 397) == pytest.approx(397 * (1302 * 2 + 770 * 10) / 1e6)
        assert figures.estimate_usd(Settings(figure_describer=LOCAL), 397) == 0.0


class TestWaitingForChat:
    def test_the_job_waits_while_a_chat_request_runs(self, library):
        settings, conn = library
        pauses = []
        with figures.chat_running(settings):
            async def pause(seconds):
                pauses.append(seconds)
                for marker in figures.busy_dir(settings).iterdir():
                    marker.unlink()         # the chat finishes during the pause
            result, lines = run(settings, conn, figures.pending(conn, settings)[:1], transport=vision(), pause=pause)
        assert pauses and lines[0] == "Paused: a chat request is running." and lines[1] == "Going on."
        assert result["described"] == 1

    def test_a_marker_left_over_from_a_crash_is_ignored(self, library):
        settings, conn = library
        marker = figures.busy_dir(settings) / "left-over"
        marker.parent.mkdir(parents=True)
        marker.touch()
        old = time.time() - settings.figure_busy_seconds - 5
        os.utime(marker, (old, old))
        assert not figures.chat_busy(settings)

    def test_a_marker_gone_between_listing_and_reading_is_skipped(self, library, monkeypatch):
        """Review C3: a chat ending at that moment raised FileNotFoundError in the worker."""
        settings, _ = library
        folder = figures.busy_dir(settings)
        folder.mkdir(parents=True)
        (folder / "live").touch()
        monkeypatch.setattr(type(folder), "iterdir", lambda self: iter([self / "just-ended", self / "live"]))
        assert figures.chat_busy(settings)
        (folder / "live").unlink()
        assert not figures.chat_busy(settings)

    def test_the_marker_goes_when_the_request_ends_even_on_an_error(self, library):
        settings, _ = library
        with pytest.raises(ValueError):
            with figures.chat_running(settings):
                assert figures.chat_busy(settings)
                raise ValueError
        assert not figures.chat_busy(settings)


class TestAsking:
    def test_a_hosted_describer_asks_and_a_no_describes_nothing(self, library, monkeypatch):
        settings, conn = library
        hosted = replace(settings, figure_describer=FigureDescriber("anthropic", "claude-sonnet-5-5",
                                                                    expected_tokens=(1302, 770)))
        monkeypatch.setattr(figures, "load_settings", lambda: hosted)
        asked = []
        monkeypatch.setattr("builtins.input", lambda question: asked.append(question) or "n")
        monkeypatch.setattr(figures, "describe_pending", lambda *a, **k: pytest.fail("described without a yes"))
        asyncio.run(figures.cmd_describe(SimpleNamespace(doc=None, limit=None, yes=False)))
        assert "Describe 2 figures with claude-sonnet-5-5" in asked[0]
        assert "sending each image, its caption and its page's text to Anthropic" in asked[0]
        assert "about $0.02" in asked[0]

    def test_the_status_names_what_each_book_still_needs(self, library):
        settings, conn = library
        lines = figures.status_lines(conn, settings)
        assert lines[0] == "Describer: ollama:vision-model (local)"
        assert "Book: 3 figures, 2 to describe" in lines[1]


def test_a_chat_request_marks_itself_busy_while_it_runs(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    settings = Settings(data_dir=tmp_path, library_dir=tmp_path, tools_enabled=False)
    connect(settings.db_path).close()
    busy_during = []

    def ollama(request):
        busy_during.append(figures.chat_busy(settings))
        return httpx.Response(200, content=b'{"message": {"content": "Hi."}, "done": true, "done_reason": "stop"}\n')
    client = TestClient(create_app(settings, transport=httpx.MockTransport(ollama),
                                   embed=lambda texts, s: [[1.0] * 1024 for _ in texts]), base_url="http://localhost")
    client.post("/api/chat", json={"message": "Hello?"})
    assert busy_during == [True]
    assert not figures.chat_busy(settings), "and the marker is gone afterwards"
