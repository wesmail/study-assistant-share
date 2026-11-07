"""Hybrid search: FTS5 keywords + sqlite-vec vectors, merged with Reciprocal Rank Fusion."""
from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from app.ingest.metadata import shown_title
from app.config import Settings
from app.ingest.embed import embed_texts, serialize


@dataclass
class Hit:
    chunk_id: int
    doc_id: int
    doc_path: str
    doc_title: str
    heading_path: list[str]
    text: str
    page_start: int | None
    page_end: int | None
    block_ids: list[int]
    meta: dict
    score: float
    keyword_rank: int | None = None     # 1-based place in each list, None if absent
    vector_rank: int | None = None
    regions: list[dict] = field(default_factory=list)
    seq: int = 0
    chunk_ids: list[int] = field(default_factory=list)      # more than one once merged
    focus_block_ids: list[int] = field(default_factory=list)  # the hit itself, not its neighbours
    focus_page: int | None = None
    doc_author: str | None = None
    doc_year: int | None = None
    doc_publisher: str | None = None
    # The figures inside the passage: page, caption, image file. Filled by find_sources.
    figures: list[dict] = field(default_factory=list)
    kind: str = "text"      # the chunk's kind: text, figure, ...


def fts_query(text: str) -> str | None:
    """User text as an FTS5 query: each word quoted, so `C++` or `(a)` isn't syntax.

    OR, not AND: one word missing from a chunk mustn't remove it, and BM25 already
    ranks chunks with more of the rarer words higher.
    """
    words = re.findall(r"\w+", text)
    return " OR ".join(f'"{word}"' for word in words) if words else None


def searchable_chunks(conn: sqlite3.Connection, doc_ids: Sequence[int] | None,
                      hidden: Sequence[str] = ()) -> str | None:
    """SQL selecting the ids of the chunks search may return, or None when that is every chunk.

    A document whose file has left the library ('missing') is never searched, so chat
    can't cite a book that's gone; its rows stay until the reader confirms removing them.
    `hidden` collections are left out of whole-library search only: documents the reader
    picked are searched whatever their collection.
    """
    conditions = []
    if doc_ids is not None:
        conditions.append(f"doc_id IN ({','.join(str(int(doc_id)) for doc_id in doc_ids)})")
    elif hidden:
        names = ",".join("'" + name.replace("'", "''") + "'" for name in hidden)
        conditions.append(f"doc_id NOT IN (SELECT id FROM documents WHERE collection IN ({names}))")
    if conn.execute("SELECT 1 FROM documents WHERE status = 'missing' LIMIT 1").fetchone():
        conditions.append("doc_id NOT IN (SELECT id FROM documents WHERE status = 'missing')")
    if not conditions:
        return None
    return "SELECT id FROM chunks WHERE " + " AND ".join(conditions)


def keyword_search(
    conn: sqlite3.Connection, query: str, limit: int, doc_ids: Sequence[int] | None = None,
    hidden: Sequence[str] = (),
) -> list[int]:
    expression = fts_query(query)
    if expression is None:
        return []
    scope = searchable_chunks(conn, doc_ids, hidden)
    rows = conn.execute(
        "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ?"
        + (f" AND rowid IN ({scope})" if scope else "") + " ORDER BY rank LIMIT ?",
        (expression, limit),
    ).fetchall()
    return [row[0] for row in rows]


def vector_search(
    conn: sqlite3.Connection, vector: bytes, limit: int, doc_ids: Sequence[int] | None = None,
    hidden: Sequence[str] = (),
) -> list[int]:
    scope = searchable_chunks(conn, doc_ids, hidden)
    if scope is None:
        allowed = ""
    else:
        # vec0 takes a literal list here, not a subquery.
        ids = [row[0] for row in conn.execute(scope)]
        if not ids:
            return []
        allowed = f" AND chunk_id IN ({','.join(map(str, ids))})"
    rows = conn.execute(
        "SELECT chunk_id FROM chunk_vec WHERE embedding MATCH ? AND k = ?" + allowed
        + " ORDER BY distance",
        (vector, limit),
    ).fetchall()
    return [row[0] for row in rows]


def rrf(ranked_lists: Sequence[Sequence[int]], k: int) -> dict[int, float]:
    """Sum of 1 / (k + rank) over the lists, highest first; ties keep first-seen order."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, 1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return dict(sorted(scores.items(), key=lambda pair: pair[1], reverse=True))


SLOTTED = ("figure", "summary", "note")


def with_slots(ranked: Sequence[Hit], top_k: int, slots: dict[str, int]) -> list[Hit]:
    """The best `top_k` text hits, plus up to `slots[kind]` figure and summary hits from the first `top_k` overall.

    A figure description that ranks high used to take a text passage's place, and its
    place among the top hits that bring in their neighbours: on book-mur-01 one pushed
    the needed passage from 3rd to 4th and two of its facts were lost. Figures and
    summaries now come on top of the text, and only those that would have made the top
    `top_k` anyway.
    """
    chosen = [hit for hit in ranked if hit.kind not in SLOTTED][:top_k]
    for kind in SLOTTED:
        chosen += [hit for hit in ranked[:top_k] if hit.kind == kind][:slots.get(kind, 0)]
    keep = {id(hit) for hit in chosen}
    return [hit for hit in ranked if id(hit) in keep]


def search_chunks(
    conn: sqlite3.Connection,
    query: str,
    settings: Settings,
    top_k: int | None = None,
    doc_ids: Sequence[int] | None = None,
    embed: Callable[[list[str], Settings], list[list[float]]] = embed_texts,
    scorer=None,
) -> list[Hit]:
    """What the app searches with: hybrid candidates, reranked, best first."""
    from app.ingest.deck import apply_exclusions
    from app.retrieval.rerank import rerank     # it needs Hit from this module

    from app.ingest.metadata import MetadataError
    try:
        apply_exclusions(conn, settings, only_if_changed=True)    # metadata.yaml edited while the app runs
    except MetadataError:
        pass        # what is stored already leaves out the pages excluded before; the pane shows the error
    top_k = top_k or settings.search_top_k
    candidates = hybrid_search(conn, query, settings, max(top_k, settings.rerank_candidates), doc_ids, embed)
    if settings.rerank_enabled:
        candidates = rerank(query, candidates, settings, scorer)
    return with_slots(candidates, top_k, {"figure": settings.search_figure_slots,
                                          "summary": settings.search_summary_slots,
                                          "note": top_k if settings.search_notes == "only" else settings.search_note_slots})


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    settings: Settings,
    top_k: int | None = None,
    doc_ids: Sequence[int] | None = None,
    embed: Callable[[list[str], Settings], list[list[float]]] = embed_texts,
) -> list[Hit]:
    """The best `top_k` chunks by RRF of keywords and vectors, and with them the reader's best
    notes (their own index, app/notes.py), placed among the chunks by their RRF score. The
    chunks are chosen first and alone, so notes never push a chunk out of the candidates."""
    from app.notes import note_hits

    top_k = top_k or settings.search_top_k
    depth = settings.search_depth
    vector = serialize(embed([query], settings)[0], settings.embed_dimensions)
    notes = note_hits(conn, query, vector, settings, doc_ids, embed) if settings.search_notes != "off" else []
    if settings.search_notes == "only":
        return notes
    keyword = keyword_search(conn, query, depth, doc_ids, settings.hidden_collections)
    similar = vector_search(conn, vector, depth, doc_ids, settings.hidden_collections)
    scores = rrf([keyword, similar], settings.rrf_k)
    chosen = list(scores)[:top_k]
    if not chosen:
        return notes

    places = ",".join("?" * len(chosen))
    rows = {row["id"]: row for row in conn.execute(
        "SELECT c.*, d.path, d.title, d.author, d.year, d.publisher FROM chunks c JOIN documents d ON d.id = c.doc_id "
        f"WHERE c.id IN ({places})",
        chosen,
    )}
    keyword_ranks = {item: rank for rank, item in enumerate(keyword, 1)}
    vector_ranks = {item: rank for rank, item in enumerate(similar, 1)}
    chunks = [
        Hit(
            chunk_id=item,
            doc_id=rows[item]["doc_id"],
            doc_path=rows[item]["path"],
            doc_title=shown_title(rows[item]["title"]), doc_author=rows[item]["author"],
            doc_year=rows[item]["year"], doc_publisher=rows[item]["publisher"],
            heading_path=json.loads(rows[item]["heading_path"] or "[]"),
            text=rows[item]["text"],
            page_start=rows[item]["page_start"],
            page_end=rows[item]["page_end"],
            block_ids=json.loads(rows[item]["block_ids"] or "[]"),
            meta=json.loads(rows[item]["meta"] or "{}"),
            score=scores[item],
            keyword_rank=keyword_ranks.get(item),
            vector_rank=vector_ranks.get(item),
            seq=rows[item]["seq"],
            chunk_ids=[item],
            focus_block_ids=json.loads(rows[item]["block_ids"] or "[]"),
            kind=rows[item]["kind"],
        )
        for item in chosen
    ]
    return sorted(chunks + notes, key=lambda hit: -hit.score) if notes else chunks
