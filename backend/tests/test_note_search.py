"""The reader's notes in search (roadmap item 4, step 3): their own index, merged with the books."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from app import notes
from app.chat import source_record
from app.llm.prompts import SYSTEM_PROMPT, format_sources
from app.retrieval.context import expand_hits
from app.retrieval.search import hybrid_search, search_chunks
from tests.test_search import axis, conn, embedder, settings  # noqa: F401  (fixtures)


def add_note(db, note_id, body, vector_axis=None, doc_id=None, page=None, quote=None, tags=(), doc_title=None):
    db.execute("INSERT INTO notes (id, doc_id, doc_title, page, rects, bbox, quote, body_md, tags)"
               " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
               (note_id, doc_id, doc_title, page, json.dumps([[1, 2, 3, 4]]) if page else None,
                json.dumps([1, 2, 3, 4]) if page else None, quote, body, json.dumps(list(tags))))
    db.commit()


def embed_by_text(table: dict[str, list[float]], default):
    """An embedder that picks a vector by a word in the text, so notes and the query differ."""
    def embed(texts, settings):
        return [next((vector for word, vector in table.items() if word in text), default) for text in texts]
    return embed


class TestIndex:
    def test_written_notes_are_indexed_and_embedded_once(self, conn, settings):  # noqa: F811
        add_note(conn, 1, "Lagrangian mnemonic: T minus V", tags=["mechanics"])
        made = []

        def embed(texts, s):
            made.extend(texts)
            return [axis(5) for _ in texts]

        assert notes.index_notes(conn, settings, embed) == 1
        assert notes.index_notes(conn, settings, embed) == 0
        assert "Tags: mechanics" in made[0]
        assert [r[0] for r in conn.execute("SELECT rowid FROM note_fts WHERE note_fts MATCH 'mnemonic'").fetchall()] == [1]

    def test_no_write_lock_is_held_while_the_embedder_runs(self, conn, settings):  # noqa: F811
        """Review C2: the keyword insert opened the write lock, then the embedder's HTTP call ran
        (up to 300 s) before the commit, so nothing else could be saved meanwhile."""
        import sqlite3

        add_note(conn, 1, "Lagrangian mnemonic")
        conn.execute("DELETE FROM note_fts WHERE rowid = 1")
        conn.commit()
        other = sqlite3.connect(conn.execute("PRAGMA database_list").fetchone()[2], timeout=0)
        seen = []

        def embed(texts, s):
            seen.append(conn.in_transaction)
            other.execute("INSERT INTO app_settings (key, value) VALUES ('meanwhile', '1')")  # fails if locked
            other.commit()
            return [axis(5) for _ in texts]

        assert notes.index_notes(conn, settings, embed) == 1 and seen == [False]
        assert [r[0] for r in conn.execute("SELECT rowid FROM note_fts WHERE note_fts MATCH 'mnemonic'").fetchall()] == [1]
        other.close()

    def test_an_edit_reindexes_and_a_delete_unindexes(self, conn, settings):  # noqa: F811
        add_note(conn, 1, "first words")
        notes.index_notes(conn, settings, embedder(axis(5)))
        conn.execute("UPDATE notes SET body_md = 'second words' WHERE id = 1")
        assert conn.execute("SELECT count(*) FROM note_vec").fetchone()[0] == 0
        assert [r[0] for r in conn.execute("SELECT rowid FROM note_fts WHERE note_fts MATCH 'second'").fetchall()] == [1]
        assert conn.execute("SELECT rowid FROM note_fts WHERE note_fts MATCH 'first'").fetchall() == []
        notes.index_notes(conn, settings, embedder(axis(5)))
        conn.execute("DELETE FROM notes WHERE id = 1")
        assert conn.execute("SELECT count(*) FROM note_vec").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM note_fts").fetchone()[0] == 0


class TestSearch:
    def test_a_note_joins_the_chunks_without_pushing_one_out(self, conn, settings):  # noqa: F811
        add_note(conn, 7, "My exam trick for Noether: every symmetry gives a conserved charge", doc_id=1,
                 page=3, quote="Noether's theorem", doc_title="Book A")
        embed = embed_by_text({"exam trick": axis(0, 0.1)}, axis(0))
        without = hybrid_search(conn, "Noether symmetry", replace(settings, search_notes="off"), top_k=3, embed=embed)
        found = hybrid_search(conn, "Noether symmetry", settings, top_k=3, embed=embed)
        assert [h.chunk_id for h in found if h.kind != "note"] == [h.chunk_id for h in without]
        note = next(h for h in found if h.kind == "note")
        assert note.chunk_id == -7 and note.chunk_ids == [-7] and note.meta == {"note_id": 7}
        assert note.doc_title == "Your note on Book A" and note.page_start == 3
        assert note.regions == [{"page": 3, "bbox": [1, 2, 3, 4]}]
        assert "every symmetry gives" in note.text and 'Attached to the passage: "Noether' in note.text

    def test_notes_take_slots_on_top_of_the_text(self, conn, settings):  # noqa: F811
        for note_id in (1, 2, 3):
            add_note(conn, note_id, f"Noether symmetry note {note_id}")
        chosen = search_chunks(conn, "Noether symmetry", replace(settings, search_note_slots=2), top_k=2,
                               embed=embedder(axis(0)))
        assert len([h for h in chosen if h.kind != "note"]) == 2
        assert len([h for h in chosen if h.kind == "note"]) <= 2

    def test_the_notes_scope_searches_notes_only(self, conn, settings):  # noqa: F811
        add_note(conn, 1, "Noether, in my words")
        add_note(conn, 2, "Clocks run slow, in my words")
        found = search_chunks(conn, "Noether", replace(settings, search_notes="only"), top_k=6, embed=embedder(axis(0)))
        assert found and all(h.kind == "note" for h in found)

    def test_document_scope_and_hidden_collections(self, conn, settings):  # noqa: F811
        add_note(conn, 1, "Noether on book A", doc_id=1, page=3)
        add_note(conn, 2, "Noether on book B", doc_id=2, page=3)
        add_note(conn, 3, "Noether, free")
        ids = lambda hits: sorted(h.meta["note_id"] for h in hits if h.kind == "note")  # noqa: E731
        assert ids(hybrid_search(conn, "Noether", settings, 10, doc_ids=[1], embed=embedder(axis(0)))) == [1]
        assert ids(hybrid_search(conn, "Noether", settings, 10, embed=embedder(axis(0)))) == [1, 2, 3]
        conn.execute("UPDATE documents SET collection = 'test' WHERE id = 2")
        hidden = replace(settings, hidden_collections=("test",))
        assert ids(hybrid_search(conn, "Noether", hidden, 10, embed=embedder(axis(0)))) == [1, 3]

    def test_a_removed_books_note_is_still_found(self, conn, settings):  # noqa: F811
        add_note(conn, 1, "Noether, kept", doc_id=1, page=3, doc_title="Book A")
        conn.execute("DELETE FROM chunks WHERE doc_id = 1")
        conn.execute("DELETE FROM documents WHERE id = 1")
        found = [h for h in hybrid_search(conn, "Noether", settings, 10, embed=embedder(axis(0))) if h.kind == "note"]
        assert found[0].doc_id == 0 and found[0].doc_title == "Your note on Book A"

    def test_no_notes_changes_nothing(self, conn, settings):  # noqa: F811
        found = hybrid_search(conn, "Noether symmetry", settings, top_k=3, embed=embedder(axis(0)))
        off = hybrid_search(conn, "Noether symmetry", replace(settings, search_notes="off"), top_k=3, embed=embedder(axis(0)))
        assert [(h.chunk_id, h.score) for h in found] == [(h.chunk_id, h.score) for h in off]


class TestForTheModel:
    def test_a_note_stands_alone_in_the_context(self, conn, settings):  # noqa: F811
        add_note(conn, 7, "Noether note", doc_id=1, page=3)
        hits = hybrid_search(conn, "Noether", settings, top_k=4, embed=embedder(axis(0)))
        expanded = expand_hits(conn, hits, top=3, radius=1, budget=10_000)
        note = [h for h in expanded if h.kind == "note"]
        assert len(note) == 1 and note[0].text.startswith("Noether note")

    def test_labelled_and_recorded_as_the_readers_note(self, conn, settings):  # noqa: F811
        add_note(conn, 7, "Noether note", doc_id=1, page=3, doc_title="Book A")
        add_note(conn, 8, "Free Noether thought")
        hits = [h for h in hybrid_search(conn, "Noether", settings, 10, embed=embedder(axis(0))) if h.kind == "note"]
        text = format_sources(hits)
        assert "[S1] Your note on Book A, p. 3\nNoether note" in text or "[S2] Your note on Book A, p. 3\nNoether note" in text
        assert "] Your note\nFree Noether thought" in text
        record = source_record(1, hits[0])
        assert record["kind"] == "note" and record["note_id"] in (7, 8)
        assert "Your note" in SYSTEM_PROMPT and "not the book" in SYSTEM_PROMPT


def test_the_chat_request_takes_the_notes_scope():
    from app.main import ChatRequest

    assert ChatRequest(message="x", notes_only=True).notes_only is True
    assert ChatRequest(message="x").notes_only is False


@pytest.fixture(autouse=True)
def _no_rerank(settings):  # noqa: F811
    assert settings.rerank_enabled is False
