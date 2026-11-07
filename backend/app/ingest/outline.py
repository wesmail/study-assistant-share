"""Heading paths from the PDF's own outline, where the parsed headings got the structure wrong.

Docling gives each heading a level from its Markdown, and for some books those levels are
wrong. In Mathematics for Machine Learning every chapter title came out at the third
level, under the previous chapter's "Further Reading", and the top level held whatever
Docling had last taken for a title ("Part II Is about Machine Learning", "2. Expansion
along row j"). A summary tree that groups chunks by chapter can't work from that.

The outline (the PDF's bookmarks) is the author's structure: level, title and page. Each
entry is placed at its heading in the text where one matches, else at the top of its page,
and a chunk's path is the entries in force where the chunk starts. Deeper headings that
the outline doesn't have, such as "4.5.3 ..." or "Example 4.11", are kept from the parsed
path when that path agrees with the outline's last entry.

Only a document whose parsed paths mostly miss its outline is changed: in the others the
parsed paths already carry the structure, sometimes better (Understanding Deep Learning's
say "Chapter 13", its outline doesn't).

    python -m app.ingest.outline            # agreement per document, changes nothing
    python -m app.ingest.outline --apply    # rewrite the documents below the threshold
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.ingest.chunker import _prefix, estimate_tokens

log = logging.getLogger(__name__)

# Below this share of outline titles found in the parsed paths, the outline wins. Measured
# Nov 4, 2026: MML 0.32; the paper 0.88, Murphy 0.99, Rubbia 0.99, UDL 1.00.
AGREEMENT_THRESHOLD = 0.5
PAGE_SLACK = 1      # a heading may sit a page from where its bookmark points

_NUMBERING = re.compile(r"^(?:(?:chapter|part)\s+)?(?:\d+(?:\.\d+)*|[ivxlc]+)\.?\s+")
_MARKUP = re.compile(r"\[(?:sup|sub)\(|\)\]")      # an outline's "W[sup(+)]" is the text's "W +"
_LABEL = re.compile(r"^(?:chapter|part)\s+(?:\d+|[ivxlc]+)$", re.IGNORECASE)


def normalized(title: str) -> str:
    """A title as letters and digits only, without its numbering: "4.3 Cholesky ..." matches "Cholesky ..."."""
    text = re.sub(r"\s+", " ", _MARKUP.sub("", title)).strip().lower()
    text = _NUMBERING.sub("", text)
    return re.sub(r"[\W_]+", "", text)


@dataclass
class Entry:
    level: int
    title: str
    page: int
    order: int | None = None    # the block where it starts; None if past the last block


def read_outline(pdf: Path) -> list[Entry]:
    import fitz     # PyMuPDF, already used to render pages
    with fitz.open(pdf) as document:
        return [Entry(level, re.sub(r"\s+", " ", title).strip(), page)
                for level, title, page in document.get_toc() if page >= 1]


def place(entries: Sequence[Entry], blocks: Sequence[tuple[int, int | None, str, str]]) -> list[Entry]:
    """Set each entry's starting block. `blocks` are (order, page, kind, text), in reading order.

    Entries are placed in outline order and never before the previous one, so a title
    repeated later in the book ("Exercises") can't pull an entry backwards.
    """
    blocks = sorted(blocks)
    placed: list[Entry] = []
    floor = -1
    for entry in entries:
        want = normalized(entry.title)
        near = {order for order, page, kind, text in blocks       # a set once per entry (review R2-X1)
                if order > floor and page is not None and abs(page - entry.page) <= PAGE_SLACK}
        heading = next((order for order, page, kind, text in blocks
                        if order in near and kind == "heading" and normalized(text) == want and want), None)
        start = heading if heading is not None else next(
            (order for order, page, kind, text in blocks if order > floor and page is not None and page >= entry.page),
            None)
        placed.append(Entry(entry.level, entry.title, entry.page, start))
        if start is not None:
            floor = start - 1 if heading is None else start
    return placed


def path_at(placed: Sequence[Entry], order: int) -> list[str]:
    """The outline titles in force at block `order`, outermost first."""
    by_level: dict[int, str] = {}
    for entry in placed:
        if entry.order is None or entry.order > order:
            break
        for level in [lv for lv in by_level if lv >= entry.level]:
            del by_level[level]
        by_level[entry.level] = entry.title
    return [by_level[level] for level in sorted(by_level)]


def merged(parsed: Sequence[str], outline: Sequence[str], titles: set[str]) -> list[str]:
    """The outline path, plus the parsed path's deeper headings when it agrees with the outline's last."""
    if not outline:
        return list(parsed)     # before the first entry: a contents page, a foreword
    keys = [normalized(part) for part in parsed]
    matched = [i for i, key in enumerate(keys) if key and key in titles]
    if not matched or keys[matched[-1]] != normalized(outline[-1]):
        return list(outline)
    tail = [part for part, key in zip(parsed[matched[-1] + 1:], keys[matched[-1] + 1:]) if key not in titles]
    return list(outline) + tail


def agreement(placed: Sequence[Entry], chunks: Sequence[tuple[int, list[str]]]) -> float:
    """Average share of a chunk's outline titles that its parsed path also has.

    A parsed "Chapter 16" stands for the outermost title: Understanding Deep Learning's
    paths name chapters by number, its outline by title, and both are right.
    """
    shares = []
    for first, parsed in chunks:
        outline = path_at(placed, first)
        if outline:
            have = {normalized(part) for part in parsed}
            found = [normalized(title) in have for title in outline]
            if not found[0] and any(_LABEL.match(part.strip()) for part in parsed):
                found[0] = True
            shares.append(sum(found) / len(outline))
    return sum(shares) / len(shares) if shares else 1.0


def new_paths(placed: Sequence[Entry], chunks: Sequence[tuple[int, list[str]]]) -> list[list[str]]:
    titles = {normalized(entry.title) for entry in placed} - {""}
    return [merged(parsed, path_at(placed, first), titles) for first, parsed in chunks]


def reprefixed(text: str, old: Sequence[str], new: Sequence[str]) -> str:
    before = _prefix(list(old))
    if not text.startswith(before):
        raise ValueError(f"chunk text doesn't start with its heading path {list(old)}")
    return _prefix(list(new)) + text[len(before):]


def outline_chunks(chunks: list, blocks: Sequence, pdf: Path, excluded: set[int] = frozenset()) -> list:
    """For the chunk stage: the chunks with outline paths, if the parsed ones mostly miss the outline.
    A bookmark on a kept-out page is dropped: its title would name the pages after it (review R2-S2)."""
    try:
        entries = [entry for entry in read_outline(pdf) if entry.page not in excluded]
    except Exception as error:      # not a PDF, or one PyMuPDF can't open: keep the parsed paths
        log.info("%s: no outline read (%s)", pdf.name, error)
        return chunks
    if not entries:
        return chunks
    placed = place(entries, [(b.order, b.page, b.kind, b.text) for b in blocks])
    pairs = [(min(c.block_orders), c.heading_path) for c in chunks if c.block_orders]
    score = agreement(placed, pairs)
    if score >= AGREEMENT_THRESHOLD:
        return chunks
    log.info("%s: parsed headings match %.0f%% of the outline; using the outline", pdf.name, 100 * score)
    for chunk, path in zip([c for c in chunks if c.block_orders], new_paths(placed, pairs)):
        chunk.text = reprefixed(chunk.text, chunk.heading_path, path)
        chunk.heading_path = path
        chunk.token_count = estimate_tokens(chunk.text)
    return chunks


def _stored(conn: sqlite3.Connection, doc_id: int):
    blocks = conn.execute('SELECT id, "order", page, kind, text FROM blocks WHERE doc_id = ?', (doc_id,)).fetchall()
    order_of = {row["id"]: row["order"] for row in blocks}
    chunks = conn.execute("SELECT id, heading_path, text, block_ids FROM chunks WHERE doc_id = ? "
                          "AND kind IN ('text', 'figure') ORDER BY seq", (doc_id,)).fetchall()
    chunks = [row for row in chunks if json.loads(row["block_ids"] or "[]")]
    pairs = [(min(order_of[i] for i in json.loads(row["block_ids"])), json.loads(row["heading_path"] or "[]"))
             for row in chunks]
    return [(row["order"], row["page"], row["kind"], row["text"]) for row in blocks], chunks, pairs


def rewrite_stored(conn: sqlite3.Connection, doc_id: int, pdf: Path, apply: bool,
                   embed: Callable[[list[str]], list[list[float]]] | None = None, dimensions: int = 0,
                   batch: int = 16, excluded: set[int] = frozenset()) -> dict:
    """Agreement for one stored document and, with `apply`, its chunks rewritten in place.

    In place, so chunk ids, which the gold questions name, stay the same. The keyword index
    follows through its trigger; the changed chunks are embedded again.
    """
    from app.ingest.embed import serialize
    entries = [entry for entry in read_outline(pdf) if entry.page not in excluded] if pdf.suffix.lower() == ".pdf" else []
    if not entries:
        return {"doc": doc_id, "entries": 0}
    blocks, chunks, pairs = _stored(conn, doc_id)
    placed = place(entries, blocks)
    score = agreement(placed, pairs)
    starts = {order for order, page, kind, text in blocks if kind == "heading"}
    report = {"doc": doc_id, "entries": len(entries), "agreement": round(score, 3),
              "placed_at_heading": sum(1 for e in placed if e.order in starts), "changed": 0}
    if score >= AGREEMENT_THRESHOLD:
        return report
    changes = [(row, path) for row, path in zip(chunks, new_paths(placed, pairs))
               if path != json.loads(row["heading_path"] or "[]")]
    report["changed"] = len(changes)
    report["examples"] = [(json.loads(row["heading_path"]), path) for row, path in changes[::max(1, len(changes) // 8)]]
    if not apply:
        return report
    texts = [reprefixed(row["text"], json.loads(row["heading_path"] or "[]"), path) for row, path in changes]
    vectors = [vector for start in range(0, len(texts), batch) for vector in embed(texts[start:start + batch])]
    with conn:
        for (row, path), text, vector in zip(changes, texts, vectors):
            conn.execute("UPDATE chunks SET heading_path = ?, text = ?, token_count = ? WHERE id = ?",
                         (json.dumps(path, ensure_ascii=False), text, estimate_tokens(text), row["id"]))
            conn.execute("DELETE FROM chunk_vec WHERE chunk_id = ?", (row["id"],))
            conn.execute("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)",
                         (row["id"], serialize(vector, dimensions)))
    return report


def main() -> None:
    from app.config import load_settings
    from app.db.connection import connect
    from app.ingest.embed import embed_texts

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="rewrite the documents below the threshold")
    parser.add_argument("--doc", type=int, action="append", help="only this document id")
    args = parser.parse_args()
    from app.ingest.deck import excluded_pages
    settings = load_settings()
    conn = connect(settings.db_path)
    rows = conn.execute("SELECT id, path FROM documents WHERE kind = 'pdf' ORDER BY id").fetchall()
    for row in rows:
        if args.doc and row["id"] not in args.doc:
            continue
        report = rewrite_stored(conn, row["id"], settings.library_dir / row["path"], args.apply,
                                lambda texts: embed_texts(texts, settings), settings.embed_dimensions,
                                settings.embed_batch, excluded_pages(settings, row["path"]))
        examples = report.pop("examples", [])
        print(row["path"][:50], report)
        for old, new in examples:
            print("   ", " › ".join(old), "\n      ->", " › ".join(new))


if __name__ == "__main__":
    main()
