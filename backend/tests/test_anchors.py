"""Re-anchoring after a book's file changes (roadmap item 4, step 4)."""
from __future__ import annotations

import hashlib
import json

import fitz
import pytest

from app import anchors, annotations, notes
from app.config import Settings
from app.db.connection import connect

PAGES = [
    ["Chapter one opens here.", "The proper time is measured by a clock moving with the particle.", "Nothing else."],
    ["Chapter two is about energy.", "Energy and momentum form a four-vector in special relativity.", "The end."],
    ["Chapter three repeats a line.", "A repeated sentence lives here.", "Between them, filler text.",
     "A repeated sentence lives here.", "After the second copy, other words."],
]


def write_pdf(path, pages, width=400, size=11):
    pdf = fitz.open()
    for lines in pages:
        page = pdf.new_page(width=width, height=300)
        for number, line in enumerate(lines):
            page.insert_text((30, 50 + 22 * number), line, fontsize=size)
    pdf.save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    sha = write_pdf(library / "book.pdf", PAGES)
    settings = Settings(data_dir=tmp_path / "data", library_dir=library, outputs_dir=tmp_path / "outputs")
    conn = connect(settings.db_path)
    conn.execute("INSERT INTO documents (id, path, sha256, title, kind, pages, status) "
                 "VALUES (1, 'book.pdf', ?, 'Book', 'pdf', 3, 'indexed')", (sha,))
    conn.commit()
    yield settings, conn, library / "book.pdf"
    conn.close()


def anchor_on(conn, settings, path, page, first, last, occurrence=0, note=False):
    """A highlight (or a note) on the words from `first` to `last`, as a selection would make it."""
    with fitz.open(path) as pdf:
        words = pdf[page - 1].get_text("words", sort=False)
    starts = [i for i, w in enumerate(words) if w[4] == first]
    start = starts[occurrence]
    end = next(i for i, w in enumerate(words) if w[4] == last and i >= start)
    chosen = words[start:end + 1]
    rects = [[min(w[0] for w in chosen), min(w[1] for w in chosen), max(w[2] for w in chosen), max(w[3] for w in chosen)]]
    quote = " ".join(w[4] for w in chosen)
    if note:
        return notes.create(conn, settings, "A note", [], {"doc_id": 1, "page": page, "rects": rects, "quote": quote})
    return annotations.create(conn, settings, 1, page, rects, quote)


def change_file(conn, path, pages, **kwargs):
    sha = write_pdf(path, pages, **kwargs)
    conn.execute("UPDATE documents SET sha256 = ? WHERE id = 1", (sha,))
    conn.commit()


def row(conn, table, row_id):
    return conn.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone()


class TestReanchor:
    def test_an_inserted_page_moves_the_anchors_after_it(self, setup):
        settings, conn, path = setup
        before = anchor_on(conn, settings, path, 1, "proper", "clock")
        after = anchor_on(conn, settings, path, 2, "Energy", "four-vector", note=True)
        assert before["prefix"] and after["quote"] == "Energy and momentum form a four-vector"
        change_file(conn, path, [PAGES[0], ["An inserted errata page.", "Nothing to see."], PAGES[1], PAGES[2]])
        counts = anchors.reanchor_document(conn, settings, 1)
        assert counts == {"ok": 1, "moved": 1, "unanchored": 0}
        assert row(conn, "annotations", before["id"])["anchor_status"] == "ok"
        moved = row(conn, "notes", after["id"])
        assert moved["anchor_status"] == "moved" and moved["page"] == 3
        assert json.loads(moved["rects"]) == after["rects"]       # same words, same place on the page
        assert anchors.reanchor_document(conn, settings, 1) == {"ok": 0, "moved": 0, "unanchored": 0}  # done once

    def test_retypeset_lines_and_a_small_edit_are_found(self, setup):
        settings, conn, path = setup
        made = anchor_on(conn, settings, path, 2, "Energy", "relativity.")
        edited = [PAGES[0], ["Chapter two is about energy.", "Energy and momentum form a 4-vector in special",
                             "relativity.", "The end."], PAGES[2]]
        change_file(conn, path, edited, width=420, size=12)
        anchors.reanchor_document(conn, settings, 1)
        found = row(conn, "annotations", made["id"])
        assert found["anchor_status"] == "moved" and found["page"] == 2
        assert found["quote"] == "Energy and momentum form a 4-vector in special relativity."
        assert len(json.loads(found["rects"])) == 2          # now over two lines

    def test_a_removed_passage_is_unanchored_and_keeps_what_it_was(self, setup):
        settings, conn, path = setup
        made = anchor_on(conn, settings, path, 1, "proper", "clock")
        change_file(conn, path, [["Chapter one was rewritten entirely.", "Different words now."], PAGES[1], PAGES[2]])
        assert anchors.reanchor_document(conn, settings, 1)["unanchored"] == 1
        lost = row(conn, "annotations", made["id"])
        assert lost["anchor_status"] == "unanchored" and lost["page"] == 1 and lost["quote"] == made["quote"]
        assert annotations.listing(conn, 1)[0]["anchor_status"] == "unanchored"

    def test_a_repeated_quote_is_told_apart_by_its_context(self, setup):
        settings, conn, path = setup
        second = anchor_on(conn, settings, path, 3, "A", "here.", occurrence=1)
        assert second["suffix"].startswith("After the second copy")
        change_file(conn, path, [PAGES[0], ["Inserted."], PAGES[1], PAGES[2]])
        anchors.reanchor_document(conn, settings, 1)
        found = row(conn, "annotations", second["id"])
        assert found["page"] == 4 and json.loads(found["rects"]) == second["rects"]

    def test_a_repeat_with_no_context_to_tell_them_apart_is_unanchored(self, setup):
        settings, conn, path = setup
        made = anchor_on(conn, settings, path, 3, "A", "here.", occurrence=1)
        conn.execute("UPDATE annotations SET prefix = '', suffix = '' WHERE id = ?", (made["id"],))
        change_file(conn, path, [PAGES[0], PAGES[1], ["Chapter three repeats a line.", "A repeated sentence lives here.",
                                                       "A repeated sentence lives here."]])
        anchors.reanchor_document(conn, settings, 1)
        assert row(conn, "annotations", made["id"])["anchor_status"] == "unanchored"

    def test_old_anchors_without_a_file_hash_are_taken_as_they_are(self, setup):
        settings, conn, path = setup
        made = anchor_on(conn, settings, path, 1, "proper", "clock")
        conn.execute("UPDATE annotations SET anchor_sha = NULL")
        assert anchors.reanchor_stale(conn, settings) == {1: {"ok": 0, "moved": 0, "unanchored": 0}}
        assert row(conn, "annotations", made["id"])["anchor_sha"] is not None

    def test_free_notes_are_left_alone(self, setup):
        settings, conn, path = setup
        free = notes.create(conn, settings, "No passage")
        change_file(conn, path, [["All new."]])
        anchors.reanchor_stale(conn, settings)
        assert notes.get(conn, free["id"])["anchor_status"] is None

    def test_a_lost_note_can_be_attached_again(self, setup):
        settings, conn, path = setup
        made = anchor_on(conn, settings, path, 1, "proper", "clock", note=True)
        change_file(conn, path, [["Rewritten."], PAGES[1], PAGES[2]])
        anchors.reanchor_document(conn, settings, 1)
        assert notes.get(conn, made["id"])["anchor_status"] == "unanchored"
        with fitz.open(path) as pdf:
            words = pdf[1].get_text("words", sort=False)
        energy = [w for w in words if w[4] in ("Energy", "and", "momentum")][:3]
        passage = {"doc_id": 1, "page": 2, "rects": [[min(w[0] for w in energy), min(w[1] for w in energy),
                                                       max(w[2] for w in energy), max(w[3] for w in energy)]],
                   "quote": "Energy and momentum"}
        attached = notes.update(conn, made["id"], passage=passage, settings=settings)
        assert attached["anchor_status"] == "ok" and attached["page"] == 2 and attached["quote"] == "Energy and momentum"


def test_the_worker_reanchors_once_the_document_is_searchable_again(setup, monkeypatch):
    from app.ingest import worker as worker_module

    settings, conn, path = setup
    made = anchor_on(conn, settings, path, 2, "Energy", "four-vector")
    change_file(conn, path, [PAGES[0], ["Inserted."], PAGES[1], PAGES[2]])
    sha = conn.execute("SELECT sha256 FROM documents WHERE id = 1").fetchone()[0]
    conn.execute("INSERT INTO jobs (doc_id, sha256, stage, stage_order) VALUES (1, ?, ?, ?)",
                 (sha, worker_module.SEARCHABLE_AFTER, worker_module.STAGES.index(worker_module.SEARCHABLE_AFTER)))
    conn.commit()
    monkeypatch.setitem(worker_module.STAGE_FUNCTIONS, worker_module.SEARCHABLE_AFTER, lambda ctx: None)
    monkeypatch.setattr(worker_module.Worker, "next_job", lambda self: self.conn.execute(
        "SELECT * FROM jobs WHERE status = 'pending' LIMIT 1").fetchone())
    worker = worker_module.Worker(conn, settings, embedder=lambda texts: [], formula_reader=object())
    monkeypatch.setattr(worker, "wait_for_chat", lambda: None)
    assert worker.run_one() is True
    found = row(conn, "annotations", made["id"])
    assert found["page"] == 3 and found["anchor_status"] == "moved"


def test_listing_through_the_api_reanchors(setup):
    from tests.test_chat import FakeOllama, client_for

    settings, conn, path = setup
    made = anchor_on(conn, settings, path, 2, "Energy", "four-vector", note=True)
    change_file(conn, path, [PAGES[0], ["Inserted."], PAGES[1], PAGES[2]])
    client = client_for(settings, FakeOllama())
    listed = client.get("/api/notes").json()["notes"]
    assert listed[0]["id"] == made["id"] and listed[0]["page"] == 3 and listed[0]["anchor_status"] == "moved"


def test_page_map_aligns_slides_in_order_even_identical_twins():
    """Roadmap item 7: a slide put in front of a deck with two identical slides (a build-up)
    and a footer on every slide. Each old slide maps to its own new slide."""
    from app.anchors import Page, page_map

    def page(text):
        return Page.of([(0, 0, 1, 1, word, 0, 0, i) for i, word in enumerate(text.split())])
    footer = "Talk footer 2026"
    old = [page(f"Title slide {footer}"), page(f"t0 reconstruction 700 ps {footer}"),
           page(f"t0 reconstruction 700 ps {footer}"), page(f"Physics results {footer}")]
    new = [page("A slide added in front")] + old
    assert page_map(old, new) == {1: 2, 2: 3, 3: 4, 4: 5}
    assert page_map(old, old) == {1: 1, 2: 2, 3: 3, 4: 4}
    assert page_map(old, [page("nothing alike at all")]) == {}


def test_no_write_lock_is_held_while_pages_are_read(setup, monkeypatch):
    """Review C1: an UPDATE opened the write lock before read_pages and every search, so the app
    couldn't save a highlight or a chat message until a big book had been searched."""
    import sqlite3

    settings, conn, path = setup
    highlight = anchor_on(conn, settings, path, 2, "Energy", "four-vector")
    change_file(conn, path, [PAGES[0], ["An inserted errata page.", "Nothing to see."], PAGES[1], PAGES[2]])
    other = sqlite3.connect(settings.db_path, timeout=0)
    real = anchors.read_pages
    seen = []

    def reading(pdf):
        seen.append(conn.in_transaction)
        other.execute("UPDATE documents SET title = 'Saved meanwhile' WHERE id = 1")     # timeout=0: fails if locked
        other.commit()
        return real(pdf)

    monkeypatch.setattr(anchors, "read_pages", reading)
    assert anchors.reanchor_document(conn, settings, 1)["moved"] == 1
    assert seen == [False] and row(conn, "annotations", highlight["id"])["page"] == 3
    other.close()
