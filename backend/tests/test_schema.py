"""The schema's promises: both indexes follow `chunks`, and re-applying is safe."""
import struct

import pytest

from app.db.connection import connect

DIM = 1024


def vec(axis: int) -> bytes:
    values = [0.0] * DIM
    values[axis] = 1.0
    return struct.pack(f"{DIM}f", *values)


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    conn.execute("INSERT INTO documents (id, path, sha256, kind) VALUES (1, 'a.pdf', 'x', 'pdf')")
    conn.commit()
    yield conn
    conn.close()


def add_chunk(conn, chunk_id, text):
    conn.execute(
        "INSERT INTO chunks (id, doc_id, kind, seq, text, token_count) VALUES (?, 1, 'text', ?, ?, 5)",
        (chunk_id, chunk_id, text),
    )
    conn.execute("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)", (chunk_id, vec(chunk_id)))


def fts_ids(conn, query):
    return [r[0] for r in conn.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ?", (query,))]


def vec_count(conn):
    return conn.execute("SELECT count(*) FROM chunk_vec").fetchone()[0]


def test_schema_can_be_applied_twice(tmp_path):
    path = tmp_path / "t.sqlite"
    connect(path).close()
    connect(path).close()


def test_all_planned_tables_exist(db):
    names = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"documents", "blocks", "chunks", "figures", "annotations", "notes",
            "conversations", "messages", "jobs", "chunk_fts", "chunk_vec"} <= names


def test_fts_finds_exact_names_and_stems(db):
    add_chunk(db, 1, "The Christoffel symbols of the metric")
    add_chunk(db, 2, "Lorentz transformations in two dimensions")
    assert fts_ids(db, "christoffel") == [1]
    assert fts_ids(db, "transformation") == [2]


def test_fts_folds_german_umlauts(db):
    add_chunk(db, 1, "Die Lösung der Gleichung")
    assert fts_ids(db, "losung") == [1]


def test_vector_search_returns_nearest(db):
    add_chunk(db, 1, "a")
    add_chunk(db, 2, "b")
    rows = db.execute(
        "SELECT chunk_id, distance FROM chunk_vec WHERE embedding MATCH ? AND k = 2 ORDER BY distance",
        (vec(2),),
    ).fetchall()
    assert rows[0]["chunk_id"] == 2
    assert rows[0]["distance"] == pytest.approx(0.0, abs=1e-6)
    assert rows[1]["distance"] == pytest.approx(1.0, abs=1e-6)


def test_deleting_a_chunk_cleans_both_indexes(db):
    add_chunk(db, 1, "Noether theorem")
    db.execute("DELETE FROM chunks WHERE id = 1")
    assert fts_ids(db, "noether") == []
    assert vec_count(db) == 0


def test_editing_a_chunk_reindexes_and_drops_its_stale_vector(db):
    add_chunk(db, 1, "old wording")
    db.execute("UPDATE chunks SET text = 'new phrasing' WHERE id = 1")
    assert fts_ids(db, "old") == []
    assert fts_ids(db, "phrasing") == [1]
    assert vec_count(db) == 0


def test_deleting_a_document_cascades_through_chunks_to_both_indexes(db):
    add_chunk(db, 1, "Noether theorem")
    db.execute("INSERT INTO blocks (doc_id, \"order\", kind, text) VALUES (1, 0, 'paragraph', 'x')")
    db.execute("DELETE FROM documents WHERE id = 1")
    assert db.execute("SELECT count(*) FROM blocks").fetchone()[0] == 0
    assert fts_ids(db, "noether") == []
    assert vec_count(db) == 0


def test_fts_stays_consistent_after_churn(db):
    for i in range(1, 6):
        add_chunk(db, i, f"text number {i}")
    db.execute("DELETE FROM chunks WHERE id IN (2, 4)")
    db.execute("INSERT INTO chunk_fts (chunk_fts) VALUES ('integrity-check')")
