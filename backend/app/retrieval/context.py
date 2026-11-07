"""What the model reads: the best hits, widened by their neighbours and joined up.

A chunk boundary can fall between the sentence a search matches and the fact that
was wanted, so the top hits bring in the chunks around them. Chunks that end up next
to each other then become one source, so nothing is read twice.

Two chunks each way around the top 3 hits took chunk recall from 16/20 to 19/20. One
each way often stopped just short of the fact. More hits or a wider window added
text and no recall. A token budget keeps the prompt within the context, nearest
neighbours first, because a prompt that's too long is cut without any error.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import replace

from app.ingest.metadata import shown_title
from app.ingest.chunker import _prefix, estimate_tokens
from app.retrieval.search import Hit

MIN_OVERLAP_CHARS = 20
MAX_OVERLAP_CHARS = 1500        # chunks overlap by one sentence at most


def _without_overlap(previous: str, following: str) -> str:
    """Drop the start of `following` that repeats the end of `previous`."""
    longest = min(len(previous), len(following), MAX_OVERLAP_CHARS)
    for size in range(longest, MIN_OVERLAP_CHARS - 1, -1):
        if previous.endswith(following[:size]):
            return following[size:].lstrip()
    return following


def _joined(rows: list[sqlite3.Row]) -> str:
    """Chunks in reading order as one text, each heading path shown once."""
    text = rows[0]["text"]
    current = rows[0]["heading_path"]
    for row in rows[1:]:
        body = row["text"]
        if row["heading_path"] == current:
            body = body.removeprefix(_prefix(json.loads(current or "[]")))
        current = row["heading_path"]
        text += "\n\n" + _without_overlap(text, body)
    return text


def _merged(rows: list[sqlite3.Row], best: Hit) -> Hit:
    """One source from a run of chunks, ranked and focused on its best hit."""
    pages = [row[column] for row in rows for column in ("page_start", "page_end")
             if row[column] is not None]
    metas = [json.loads(row["meta"] or "{}") for row in rows]
    formulas = [formula for meta in metas for formula in meta.get("unverified_formulas", [])]
    return Hit(
        chunk_id=best.chunk_id,
        doc_id=best.doc_id,
        doc_path=best.doc_path,
        doc_title=best.doc_title, doc_author=best.doc_author, doc_year=best.doc_year,
        doc_publisher=best.doc_publisher,
        heading_path=best.heading_path,
        text=_joined(rows),
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        block_ids=[block for row in rows for block in json.loads(row["block_ids"] or "[]")],
        meta={"unverified_formulas": formulas} if formulas else {},
        score=best.score,
        keyword_rank=best.keyword_rank,
        vector_rank=best.vector_rank,
        seq=best.seq,
        chunk_ids=[row["id"] for row in rows],
        focus_block_ids=best.block_ids,
        kind=best.kind,
    )


def _token_count(conn: sqlite3.Connection, doc_id: int, seq: int) -> int | None:
    row = conn.execute(
        "SELECT token_count FROM chunks WHERE doc_id = ? AND seq = ? AND kind IN ('text', 'figure')",
        (doc_id, seq),
    ).fetchone()
    return row["token_count"] if row else None


def _with_neighbours(
    conn: sqlite3.Connection, hits: list[Hit], top: int, radius: int, budget: int | None
) -> dict[int, set[int]]:
    """The chunks to read, per document. Hits always count; neighbours fill the budget.

    Only text hits bring in neighbours: a figure hit has its own slot (search.with_figure_slots)
    and doesn't take the place of a text passage's context.
    """
    wanted: dict[int, set[int]] = {}
    spent = 0
    for hit in hits:
        wanted.setdefault(hit.doc_id, set()).add(hit.seq)
        spent += _token_count(conn, hit.doc_id, hit.seq) or 0
    for distance in range(1, radius + 1):
        for hit in [hit for hit in hits if hit.kind != "figure"][:top]:
            for seq, inner in ((hit.seq - distance, hit.seq - distance + 1), (hit.seq + distance, hit.seq + distance - 1)):
                if inner not in wanted[hit.doc_id]:     # only next to what is read: no island without a hit (R2-C3)
                    continue
                cost = _token_count(conn, hit.doc_id, seq)
                if cost is None or seq in wanted[hit.doc_id]:
                    continue
                if budget is None or spent + cost <= budget:
                    wanted[hit.doc_id].add(seq)
                    spent += cost
    return wanted


def with_parts(conn: sqlite3.Connection, hit: Hit, cap: int) -> Hit:
    """A summary hit followed by the summaries of its parts, in order, as far as `cap` tokens allow.

    A chapter's summary says what the chapter is for; its sections' summaries say what
    each covers. Together they are the overview, at a few thousand tokens.
    """
    ids, texts = [hit.chunk_id], [hit.text]
    spent = estimate_tokens(hit.text)
    for child in hit.meta.get("children", []):
        row = conn.execute("SELECT id, text, token_count FROM chunks WHERE id = ? AND kind = 'summary'",
                           (child,)).fetchone()
        if row is None or spent + row["token_count"] > cap:
            break
        ids.append(row["id"])
        texts.append(row["text"])
        spent += row["token_count"]
    return replace(hit, text="\n\n".join(texts), chunk_ids=ids)


def expand_hits(
    conn: sqlite3.Connection, hits: list[Hit], top: int, radius: int = 1, budget: int | None = None,
    summary_tokens: int = 3000,
) -> list[Hit]:
    """Add the chunks within `radius` of the first `top` text hits, and merge whatever is adjacent.

    Summary hits stand alone. Their own text counts first, like any hit; the summaries of
    their parts only fill what the text neighbours leave of the budget, up to
    `summary_tokens` each. Taken first, the parts pushed a needed passage out on book-mur-01.
    """
    summaries = [(rank, hit) for rank, hit in enumerate(hits) if hit.kind == "summary"]
    # The reader's notes stand alone too: they have no neighbours in the book.
    notes = [(rank, hit) for rank, hit in enumerate(hits) if hit.kind == "note"]
    if budget is not None:
        budget -= sum(estimate_tokens(hit.text) for _, hit in summaries + notes)
    passages = [hit for hit in hits if hit.kind not in ("summary", "note")]
    wanted = _with_neighbours(conn, passages, top, radius, budget)
    left = None
    if budget is not None:
        left = budget - sum(_token_count(conn, doc_id, seq) or 0 for doc_id, seqs in wanted.items() for seq in seqs)
    runs: list[tuple[int, Hit]] = list(notes)
    for rank, hit in summaries:
        own = estimate_tokens(hit.text)
        cap = summary_tokens if left is None else min(summary_tokens, own + max(left, 0))
        whole = with_parts(conn, hit, cap)
        if left is not None and len(whole.chunk_ids) > 1:
            places = ",".join("?" * (len(whole.chunk_ids) - 1))
            left -= conn.execute(f"SELECT sum(token_count) FROM chunks WHERE id IN ({places})",
                                 whole.chunk_ids[1:]).fetchone()[0]
        runs.append((rank, whole))
    best_by_place = {(hit.doc_id, hit.seq): (rank, hit) for rank, hit in enumerate(hits)
                     if hit.kind not in ("summary", "note")}
    for doc_id, seqs in wanted.items():
        places = ",".join("?" * len(seqs))
        rows = conn.execute(
            f"SELECT * FROM chunks WHERE doc_id = ? AND seq IN ({places}) "
            "AND kind IN ('text', 'figure') ORDER BY seq",
            (doc_id, *sorted(seqs)),
        ).fetchall()
        run: list[sqlite3.Row] = []
        for row in [*rows, None]:
            if run and (row is None or row["seq"] != run[-1]["seq"] + 1):
                finished = _finished(run, doc_id, best_by_place)
                if finished is not None:    # a neighbour cut off from its hit (a row between them gone) is dropped
                    runs.append(finished)
                run = []
            if row is not None:
                run.append(row)
    return [hit for _, hit in sorted(runs, key=lambda item: item[0])]


def carried_hits(
    conn: sqlite3.Connection,
    cited: Sequence[dict],
    present: set[int],
    budget: int,
    doc_ids: Sequence[int] | None = None,
    hidden: Sequence[str] = (),
) -> list[Hit]:
    """The sources the previous answer cited, as far as `budget` tokens allow.

    Each source's main chunk comes first, so a tight budget loses the context around
    a passage and not the passage itself. Chunks the search found again are skipped,
    and so is anything outside the reader's current scope.
    """
    order = [source["chunk_id"] for source in cited]
    order += [chunk for source in cited for chunk in source.get("chunk_ids", [])]
    chosen: dict[int, sqlite3.Row] = {}
    spent = 0
    for chunk_id in dict.fromkeys(order):
        row = conn.execute(
            "SELECT c.*, d.path, d.title, d.author, d.year, d.publisher, d.collection FROM chunks c JOIN documents d "
            "ON d.id = c.doc_id WHERE c.id = ? AND d.status != 'missing'",     # its file is gone: never cite it
            (chunk_id,),
        ).fetchone()
        if row is None or chunk_id in present or (doc_ids is not None and row["doc_id"] not in doc_ids):
            continue
        if doc_ids is None and row["collection"] in hidden:
            continue
        if spent + row["token_count"] <= budget:
            spent += row["token_count"]
            chosen[chunk_id] = row

    runs: list[list[sqlite3.Row]] = []
    for row in sorted(chosen.values(), key=lambda r: (r["doc_id"], r["seq"])):
        if runs and runs[-1][-1]["doc_id"] == row["doc_id"] and runs[-1][-1]["seq"] + 1 == row["seq"]:
            runs[-1].append(row)
        else:
            runs.append([row])
    place = {chunk_id: rank for rank, chunk_id in enumerate(chosen)}
    runs.sort(key=lambda run: min(place[row["id"]] for row in run))
    return [_merged(run, _as_hit(min(run, key=lambda row: place[row["id"]]))) for run in runs]


def _as_hit(row: sqlite3.Row) -> Hit:
    return Hit(
        chunk_id=row["id"], doc_id=row["doc_id"], doc_path=row["path"],
        doc_title=shown_title(row["title"]), doc_author=row["author"],
            doc_year=row["year"], doc_publisher=row["publisher"], heading_path=json.loads(row["heading_path"] or "[]"),
        text=row["text"], page_start=row["page_start"], page_end=row["page_end"],
        block_ids=json.loads(row["block_ids"] or "[]"), meta={}, score=0.0, seq=row["seq"],
    )


def _finished(run: list[sqlite3.Row], doc_id: int, best_by_place: dict) -> tuple[int, Hit] | None:
    found = [best_by_place[doc_id, row["seq"]] for row in run if (doc_id, row["seq"]) in best_by_place]
    if not found:
        return None
    rank, best = min(found, key=lambda item: item[0])
    return rank, _merged(run, best)
