"""The summary tree (app.ingest.summarize), with a stand-in for the model."""
from __future__ import annotations

import json

import fitz
import pytest

from app.config import Settings
from app.db.connection import connect
from app.ingest import summarize as tree_module
from app.ingest.chunker import _prefix
from app.ingest.summarize import build_tree, cached, contents, marker, store, summarize


@pytest.fixture
def conn(tmp_path):
    db = connect(tmp_path / "a.sqlite")
    db.execute("INSERT INTO documents (id, path, sha256, title, kind) VALUES (1, 'b.pdf', 'x', 'Book', 'pdf')")
    return db


def add(db, order, page, path, body, kind="paragraph", chunk=True):
    block = db.execute('INSERT INTO blocks (doc_id, "order", kind, page, text) VALUES (1, ?, ?, ?, ?)',
                       (order, kind, page, body)).lastrowid
    if chunk:
        db.execute("INSERT INTO chunks (doc_id, kind, seq, heading_path, text, token_count, block_ids, page_start, "
                   "page_end) VALUES (1, 'text', ?, ?, ?, 10, ?, ?, ?)",
                   (order, json.dumps(path), _prefix(path) + body, json.dumps([block]), page, page))


def book(db):
    add(db, 0, 1, ["Contents"], "1 Algebra ... 3; 9 Learning ... 40")
    add(db, 1, 2, ["Algebra"], "Algebra", kind="heading", chunk=False)
    add(db, 2, 2, ["Algebra"], "Algebra is about structure.")
    add(db, 3, 3, ["Algebra", "1.1 Matrices"], "1.1 Matrices", kind="heading", chunk=False)
    add(db, 4, 3, ["Algebra", "1.1 Matrices"], "A matrix is a grid.")
    add(db, 5, 4, ["Algebra", "1.2 Ranks"], "1.2 Ranks", kind="heading", chunk=False)
    add(db, 6, 4, ["Algebra", "1.2 Ranks"], "Rank counts independent columns.")
    add(db, 7, 5, ["Geometry"], "Geometry", kind="heading", chunk=False)
    add(db, 8, 5, ["Geometry"], "Norms measure length.")


def outline_pdf(tmp_path, toc):
    document = fitz.open()
    for _ in range(6):
        document.new_page()
    document.set_toc(toc)
    path = tmp_path / "b.pdf"
    document.save(path)
    return path


def names(node):
    return [n.name() for n in node.walk()]


def test_the_tree_follows_the_outline_and_skips_contents_pages(conn, tmp_path):
    book(conn)
    pdf = outline_pdf(tmp_path, [[1, "1 Algebra", 2], [2, "1.1 Matrices", 3], [2, "1.2 Ranks", 4],
                                 [1, "2 Geometry", 5], [1, "9 Learning", 6]])
    root = build_tree(conn, 1, pdf, "Book")
    # Paths as the chunks spell them; "9 Learning" has no text in this copy and is dropped.
    assert names(root) == ["Book", "Algebra", "Algebra › 1.1 Matrices", "Algebra › 1.2 Ranks", "Geometry"]
    assert [len(n.chunks) for n in root.walk()] == [0, 1, 1, 1, 1]
    assert root.pages() == (2, 5)


def test_without_an_outline_the_heading_paths_make_the_tree(conn):
    book(conn)
    root = build_tree(conn, 1, None, "Book")
    assert names(root) == ["Book", "Algebra", "Algebra › 1.1 Matrices", "Algebra › 1.2 Ranks", "Geometry"]


def test_summaries_go_bottom_up_and_a_parent_sees_its_parts(conn):
    book(conn)
    root = build_tree(conn, 1, None, "Book")
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        return f"summary {len(prompts)}"
    summarize(root, "Book", ask)
    assert "A matrix is a grid." in prompts[0] and "Summarize this section" in prompts[0]
    chapter = prompts[2]
    assert "Algebra is about structure." in chapter and "summary 1" in chapter and "summary 2" in chapter
    assert prompts[-1].startswith("Summarize the whole of Book")
    assert root.summary == f"summary {len(prompts)}"


def test_a_long_section_is_summarized_from_notes_on_its_parts(conn, monkeypatch):
    for i in range(6):
        add(conn, i, i + 1, ["Long"], f"Paragraph {i}. " + "word " * 40)
    monkeypatch.setattr(tree_module, "PART_TOKENS", 100)
    root = build_tree(conn, 1, None, "Book")
    prompts = []
    summarize(root, "Book", lambda p: prompts.append(p) or "note")
    notes = [p for p in prompts if p.startswith("This is part")]
    assert len(notes) > 1
    assert "NOTES ON THE TEXT" in prompts[len(notes)]


def test_contents_list_the_copys_chapters_with_pages(conn):
    book(conn)
    root = build_tree(conn, 1, None, "Book")
    listed = contents(root)
    assert "- Algebra (pages 2–4)" in listed and "  - 1.1 Matrices (page 3)" in listed
    assert "Learning" not in listed


def test_replies_are_cached_by_their_whole_input(conn):
    calls = []
    ask = cached(conn, "test:model", lambda p: calls.append(p) or p.upper())
    assert ask("abc") == "ABC" and ask("abc") == "ABC" and ask("abd") == "ABD"
    assert calls == ["abc", "abd"]


def test_stored_summaries_are_labelled_linked_and_embedded(conn):
    book(conn)
    root = build_tree(conn, 1, None, "Book")
    summarize(root, "Book", lambda p: "S.")
    count = store(conn, Settings(), 1, root, "Book", "test:model", lambda texts: [[0.1] * 1024 for _ in texts])
    rows = conn.execute("SELECT * FROM chunks WHERE kind = 'summary' ORDER BY id").fetchall()
    assert count == len(rows) == 5
    top = rows[-1]
    assert top["level"] == 2 and "Contents of this copy" in top["text"]
    assert top["text"].startswith(marker(root, "Book"))
    chapter = next(r for r in rows if json.loads(r["heading_path"]) == ["Algebra"])
    assert chapter["text"].startswith(_prefix(["Algebra"]) + '[AI-written summary of "Algebra", pages 2–4 of Book.')
    assert len(json.loads(chapter["meta"])["children"]) == 2
    assert conn.execute("SELECT count(*) FROM chunk_vec").fetchone()[0] == 5
    store(conn, Settings(), 1, root, "Book", "test:model", lambda texts: [[0.1] * 1024 for _ in texts])
    assert conn.execute("SELECT count(*) FROM chunks WHERE kind = 'summary'").fetchone()[0] == 5


# --- Streamed calls that a chat request can interrupt (Phase 5c) ------------------------------

import httpx as _httpx  # noqa: E402

from app.config import Settings as _Settings  # noqa: E402
from app.ingest.summarize import Preempted, ollama_stream_ask  # noqa: E402


def _stream(*bodies):
    lines = "\n".join(json.dumps(body) for body in bodies) + "\n"
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return _httpx.Response(200, content=lines.encode())
    return _httpx.MockTransport(handler), seen


def test_a_streamed_reply_is_joined_and_sent_like_the_cli():
    transport, seen = _stream({"message": {"content": "Sec"}}, {"message": {"content": "tion one."}},
                              {"message": {"content": ""}, "done": True, "done_reason": "stop"})
    ask = ollama_stream_ask(_Settings(), "qwen3.8:27b", transport=transport)
    assert ask("Summarize") == "Section one."
    assert seen[0]["stream"] is True and seen[0]["options"]["temperature"] == 0.3 and seen[0]["think"] is False


def test_a_chat_request_drops_the_call():
    transport, _ = _stream({"message": {"content": "Sec"}}, {"message": {"content": "tion"}}, {"done": True})
    ask = ollama_stream_ask(_Settings(), "m", busy=lambda: True, transport=transport, check_every=0)
    with pytest.raises(Preempted):
        ask("Summarize")


def test_a_streamed_reply_cut_at_the_limit_is_an_error():
    transport, _ = _stream({"message": {"content": "Half"}}, {"done": True, "done_reason": "length"})
    with pytest.raises(RuntimeError, match="token limit"):
        ollama_stream_ask(_Settings(), "m", transport=transport)("Summarize")
