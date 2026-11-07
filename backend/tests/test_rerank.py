"""Reranking search results, with the cross-encoder replaced by a canned scorer."""
import pytest

from app.chat import find_sources
from app.config import Settings
from app.db.connection import connect
from app.ingest.embed import serialize
from app.retrieval import rerank as module
from app.retrieval.rerank import rerank
from app.retrieval.search import Hit, search_chunks

ON = Settings(rerank_enabled=True)


def hit(number):
    return Hit(chunk_id=number, doc_id=1, doc_path="a.pdf", doc_title="A", heading_path=[], text=f"text {number}",
               page_start=1, page_end=1, block_ids=[], meta={}, score=0.0, seq=number)


def by_number(*scores):
    return lambda query, texts, settings: list(scores)


def test_the_best_score_comes_first():
    ranked = rerank("q", [hit(1), hit(2), hit(3)], ON, by_number(0.1, 5.0, -2.0))
    assert [h.chunk_id for h in ranked] == [2, 1, 3]


def test_the_score_is_kept_on_each_hit():
    assert rerank("q", [hit(1)], ON, by_number(1.5))[0].score == 1.5


def test_equal_scores_keep_the_search_order():
    assert [h.chunk_id for h in rerank("q", [hit(1), hit(2)], ON, by_number(1.0, 1.0))] == [1, 2]


def test_the_scorer_sees_the_question_and_each_passage():
    seen = {}

    def scorer(query, texts, settings):
        seen.update(query=query, texts=texts)
        return [0.0, 0.0]

    rerank("Why?", [hit(1), hit(2)], ON, scorer)
    assert seen == {"query": "Why?", "texts": ["text 1", "text 2"]}


@pytest.mark.parametrize("failure", [
    RuntimeError("CUDA out of memory"), OSError("model not found"), ValueError("bad"), ImportError("no torch"),
])
def test_a_reranker_failure_keeps_the_search_order(failure, caplog):
    def scorer(query, texts, settings):
        raise failure

    ranked = rerank("q", [hit(1), hit(2)], ON, scorer)
    assert [h.chunk_id for h in ranked] == [1, 2]
    assert "Keeping the search order" in caplog.text


def test_when_turned_off_it_never_calls_the_scorer():
    ranked = rerank("q", [hit(1)], Settings(rerank_enabled=False), lambda *a: pytest.fail("called"))
    assert [h.chunk_id for h in ranked] == [1]


def test_no_candidates_means_no_call():
    assert rerank("q", [], ON, lambda *a: pytest.fail("called")) == []


def vector(second):
    return [1.0, second] + [0.0] * (Settings().embed_dimensions - 2)


def library(tmp_path, chunks):
    settings = Settings(data_dir=tmp_path, library_dir=tmp_path, rerank_enabled=True,
                        rerank_candidates=4, search_top_k=2)
    db = connect(settings.db_path)
    db.execute("INSERT INTO documents (id, path, sha256, title, kind) VALUES (1, 'a.pdf', 's', 'A', 'pdf')")
    for seq in range(chunks):
        row = db.execute(
            "INSERT INTO chunks (doc_id, kind, seq, text, token_count, block_ids, page_start, page_end) "
            "VALUES (1, 'text', ?, ?, 10, '[]', 1, 1)", (seq, f"apple chunk {seq}")).lastrowid
        db.execute("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)",
                   (row, serialize(vector(seq / 10), settings.embed_dimensions)))
    db.commit()
    return settings, db


def embed(texts, settings):
    return [vector(0.0) for _ in texts]


def fake_scores(query, texts, settings):
    return [float(text.rsplit(" ", 1)[1]) for text in texts]


def test_only_the_candidates_are_scored_and_the_best_are_kept(tmp_path, monkeypatch):
    settings, db = library(tmp_path, 8)
    scored = []

    def scorer(query, texts, settings):
        scored.append(len(texts))
        return fake_scores(query, texts, settings)

    hits = search_chunks(db, "apple", settings, embed=embed, scorer=scorer)
    assert scored == [4] and len(hits) == 2
    assert hits[0].score >= hits[1].score


def test_turned_off_it_is_the_plain_search(tmp_path):
    settings, db = library(tmp_path, 8)
    settings.rerank_enabled = False
    hits = search_chunks(db, "apple", settings, embed=embed, scorer=lambda *a: pytest.fail("called"))
    assert len(hits) == 2


def test_the_chat_pipeline_reranks(tmp_path, monkeypatch):
    settings, _ = library(tmp_path, 8)
    monkeypatch.setattr(module.cross_encoder, "scores", fake_scores)
    hits = find_sources(settings, "apple", None, embed)
    assert hits and hits[0].chunk_ids
