"""Hybrid search on a small index, with vectors chosen by hand."""
import json
import math

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db.connection import connect
from app.ingest.embed import serialize
from app.main import create_app
from app.retrieval.search import fts_query, hybrid_search, keyword_search, rrf, vector_search

DIMENSIONS = 1024


def axis(index: int, tilt: float = 0.0) -> list[float]:
    """A unit vector along `index`, leaning towards the next axis by `tilt`."""
    vector = [0.0] * DIMENSIONS
    vector[index], vector[index + 1] = math.cos(tilt), math.sin(tilt)
    return vector


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path, library_dir=tmp_path, search_depth=10)


@pytest.fixture
def conn(settings):
    db = connect(settings.db_path)
    db.execute("INSERT INTO documents (id, path, sha256, title, kind) VALUES (1, 'a.pdf', 'x', 'Book A', 'pdf')")
    db.execute("INSERT INTO documents (id, path, sha256, title, kind) VALUES (2, 'b.pdf', 'y', NULL, 'pdf')")
    rows = [
        (1, 1, "Noether's theorem links symmetry to conservation", axis(0)),
        (2, 1, "Eigenvalues of a symmetric matrix are real", axis(1)),
        (3, 2, "Time dilation for moving clocks", axis(2)),
        (4, 2, "Symmetry breaking gives the W boson its mass", axis(3)),
    ]
    for chunk_id, doc_id, text, vector in rows:
        db.execute(
            "INSERT INTO chunks (id, doc_id, kind, seq, heading_path, text, token_count, block_ids, "
            "page_start, page_end, meta) VALUES (?, ?, 'text', ?, ?, ?, 10, '[7]', 3, 4, ?)",
            (chunk_id, doc_id, chunk_id, '["Ch. 1"]', text, '{"k": 1}'),
        )
        db.execute("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)",
                   (chunk_id, serialize(vector, DIMENSIONS)))
    db.commit()
    return db


def embedder(vector):
    return lambda texts, settings: [vector for _ in texts]


@pytest.mark.parametrize("text, expected", [
    ("proper time", '"proper" OR "time"'),
    ("C++ (a) AND b", '"C" OR "a" OR "AND" OR "b"'),
    ("Lorénz", '"Lorénz"'),
])
def test_user_text_becomes_quoted_words(text, expected):
    assert fts_query(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "?!(+)"])
def test_text_without_words_is_no_query(text):
    assert fts_query(text) is None


@pytest.mark.parametrize("text", ['"', "(", "AND", "NEAR(", "a:b", "x*", "-"])
def test_query_syntax_in_user_text_is_harmless(conn, text):
    keyword_search(conn, text, 5)


def test_keyword_search_finds_an_exact_name(conn):
    assert keyword_search(conn, "Noether", 5) == [1]


def test_keyword_search_stems_words(conn):
    assert keyword_search(conn, "eigenvalue", 5) == [2]


def test_one_absent_word_doesnt_remove_a_chunk(conn):
    assert keyword_search(conn, "Noether quaternion", 5) == [1]


def test_keyword_search_keeps_to_the_scope(conn):
    assert keyword_search(conn, "symmetry", 5, doc_ids=[2]) == [4]


def test_vector_search_orders_by_closeness(conn):
    vector = serialize(axis(1, tilt=0.3), DIMENSIONS)
    assert vector_search(conn, vector, 2) == [2, 3]


def test_vector_search_keeps_to_the_scope(conn):
    vector = serialize(axis(1), DIMENSIONS)
    assert 2 not in vector_search(conn, vector, 4, doc_ids=[2])


def test_a_scope_with_no_chunks_finds_nothing(conn):
    assert vector_search(conn, serialize(axis(1), DIMENSIONS), 4, doc_ids=[99]) == []


def gone(conn, doc_id):
    """The worker's mark for a document whose file left library/."""
    conn.execute("UPDATE documents SET status = 'missing' WHERE id = ?", (doc_id,))
    conn.commit()


def test_keyword_search_skips_a_missing_document(conn):
    gone(conn, 1)
    assert keyword_search(conn, "symmetry", 5) == [4]


def test_vector_search_skips_a_missing_document(conn):
    gone(conn, 1)
    assert set(vector_search(conn, serialize(axis(1), DIMENSIONS), 4)) == {3, 4}     # 3 and 4 tie


def test_a_scope_naming_a_missing_document_finds_nothing_there(conn):
    gone(conn, 1)
    assert keyword_search(conn, "symmetry", 5, doc_ids=[1, 2]) == [4]
    assert vector_search(conn, serialize(axis(0), DIMENSIONS), 4, doc_ids=[1]) == []


def test_hybrid_search_never_returns_a_missing_documents_chunk(conn, settings):
    gone(conn, 1)
    hits = hybrid_search(conn, "Noether symmetry eigenvalues", settings, embed=embedder(axis(0)), top_k=4)
    assert hits and {hit.doc_id for hit in hits} == {2}


def test_a_document_that_comes_back_is_searched_again(conn):
    gone(conn, 1)
    conn.execute("UPDATE documents SET status = 'indexed' WHERE id = 1")
    assert keyword_search(conn, "Noether", 5) == [1]


def test_rrf_formula():
    scores = rrf([[10, 20]], k=60)
    assert scores == {10: pytest.approx(1 / 61), 20: pytest.approx(1 / 62)}


def test_rrf_adds_across_lists():
    assert rrf([[1], [1]], k=60)[1] == pytest.approx(2 / 61)


def test_agreement_beats_a_single_first_place():
    scores = rrf([[1] + list(range(100, 108)) + [7], [2] + list(range(200, 208)) + [7]], k=60)
    assert list(scores)[0] == 7


def test_a_chunk_in_one_list_still_scores():
    assert set(rrf([[1], [2]], k=60)) == {1, 2}


def test_hybrid_merges_both_lists(conn, settings):
    hits = hybrid_search(conn, "Noether", settings, embed=embedder(axis(2)), top_k=4)
    assert {hit.chunk_id for hit in hits} >= {1, 3}


def test_a_chunk_found_both_ways_comes_first(conn, settings):
    hits = hybrid_search(conn, "time dilation", settings, embed=embedder(axis(2)), top_k=4)
    assert hits[0].chunk_id == 3
    assert (hits[0].keyword_rank, hits[0].vector_rank) == (1, 1)


def test_top_k_limits_the_result(conn, settings):
    assert len(hybrid_search(conn, "symmetry", settings, embed=embedder(axis(0)), top_k=2)) == 2


def test_top_k_comes_from_settings(conn):
    narrow = Settings(search_top_k=1, search_depth=10)
    assert len(hybrid_search(conn, "symmetry", narrow, embed=embedder(axis(0)))) == 1


def test_a_hit_carries_what_a_citation_needs(conn, settings):
    hit = hybrid_search(conn, "Noether", settings, embed=embedder(axis(0)))[0]
    assert (hit.doc_path, hit.doc_title, hit.page_start, hit.page_end) == ("a.pdf", "Book A", 3, 4)
    assert (hit.heading_path, hit.block_ids, hit.meta) == (["Ch. 1"], [7], {"k": 1})


def test_a_document_without_a_title_is_untitled_never_its_file_name(conn, settings):
    hit = hybrid_search(conn, "dilation", settings, embed=embedder(axis(2)))[0]
    assert hit.doc_title == "Untitled document"


def test_scope_applies_to_both_lists(conn, settings):
    hits = hybrid_search(conn, "symmetry", settings, doc_ids=[1], embed=embedder(axis(3)), top_k=4)
    assert {hit.doc_id for hit in hits} == {1}


def test_an_empty_index_finds_nothing(tmp_path):
    made = Settings(data_dir=tmp_path / "empty")
    assert hybrid_search(connect(made.db_path), "x", made, embed=embedder(axis(0))) == []


def client(settings, conn, embed):
    return TestClient(create_app(settings, embed=embed), base_url="http://localhost")


def test_the_debug_endpoint_shows_both_ranks(settings, conn):
    body = client(settings, conn, embedder(axis(2))).get("/api/search", params={"q": "time dilation"}).json()
    assert body["query"] == "time dilation"
    first = body["hits"][0]
    assert (first["keyword_rank"], first["vector_rank"], first["pages"]) == (1, 1, [3, 4])


def test_the_debug_endpoint_takes_a_scope_and_k(settings, conn):
    body = client(settings, conn, embedder(axis(0))).get(
        "/api/search", params={"q": "symmetry", "doc_id": [2], "k": 1}).json()
    assert [hit["doc_id"] for hit in body["hits"]] == [2]


def test_an_empty_query_is_refused(settings, conn):
    assert client(settings, conn, embedder(axis(0))).get("/api/search", params={"q": ""}).status_code == 422


def test_a_dead_embedder_is_a_503(settings, conn):
    def dead(texts, settings):
        raise OSError("connection refused")

    assert client(settings, conn, dead).get("/api/search", params={"q": "x"}).status_code == 503


# --- The test collection, hidden from whole-library search (Phase 5c step 6) -------------------

def test_a_hidden_collection_is_left_out_of_whole_library_search(conn, settings):
    from dataclasses import replace
    conn.execute("UPDATE documents SET collection = 'test' WHERE id = 1")
    assert keyword_search(conn, "symmetry", 5, hidden=("test",)) == [4]
    assert set(vector_search(conn, serialize(axis(0), DIMENSIONS), 4, hidden=("test",))) == {3, 4}
    hits = hybrid_search(conn, "Noether symmetry", replace(settings, hidden_collections=("test",)),
                         embed=embedder(axis(0)), top_k=4)
    assert {hit.doc_id for hit in hits} == {2}


def test_a_hidden_collection_is_still_searched_when_picked(conn):
    conn.execute("UPDATE documents SET collection = 'test' WHERE id = 1")
    assert keyword_search(conn, "Noether", 5, doc_ids=[1], hidden=("test",)) == [1]


def test_nothing_is_hidden_by_default_so_the_evaluations_see_every_book(conn, settings):
    from app.config import load_settings
    conn.execute("UPDATE documents SET collection = 'test' WHERE id = 1")
    assert load_settings().hidden_collections == ()
    assert keyword_search(conn, "Noether", 5) == [1]


def test_the_library_setting_hides_the_collection_from_the_search_api(conn, settings):
    conn.execute("UPDATE documents SET collection = 'test' WHERE id = 1")
    conn.commit()
    client = TestClient(create_app(settings, embed=embedder(axis(0))), base_url="http://localhost")
    # create_app rereads metadata.yaml, which this library doesn't have: mark the collection again.
    conn.execute("UPDATE documents SET collection = 'test' WHERE id = 1")
    conn.commit()
    before = {hit["doc_id"] for hit in client.get("/api/search", params={"q": "Noether symmetry"}).json()["hits"]}
    assert client.put("/api/library/settings", json={"hide_test_collection": True}).json() == {"hide_test_collection": True}
    after = {hit["doc_id"] for hit in client.get("/api/search", params={"q": "Noether symmetry"}).json()["hits"]}
    assert before == {1, 2} and after == {2}
    assert client.get("/api/documents").json()["settings"] == {"hide_test_collection": True}
    assert client.put("/api/library/settings", json={"hide_everything": 1}).status_code == 409
