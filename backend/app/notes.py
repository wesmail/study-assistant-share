"""The reader's notes (roadmap item 4, step 2): Markdown with LaTeX, tags, anchored to a passage.

A note is anchored like a highlight: document, page, boxes (one per line), the quote and the
words around it, snapped to the page's words by `annotations.anchor`. A note can also have no
passage (written from the notes panel). Notes outlive their book: removing a document sets
`doc_id` to null, and the note keeps the book's title (`doc_title`, saved when it's written),
its page and its quote.

Export for Obsidian: one Markdown file per document, with YAML frontmatter (title, authors,
year, source file, tags), a heading per note with its page, the quote as a block quote, the
note, and its tags as #tags. Maths stays in `$…$` and `$$…$$`, the delimiters Obsidian
renders; `\\(…\\)` and `\\[…\\]` are rewritten to them, outside code.
"""
from __future__ import annotations

import io
import json
import re
import sqlite3
import zipfile
from datetime import date

import yaml

from app import annotations
from app.errors import NotFound, Refused  # noqa: F401  (re-exported: notes.Refused)
from app.config import Settings
from app.tools.documents import dollar_math

BODY_CHARS = 50_000
MAX_TAGS = 20
TAG_CHARS = 40
NO_DOCUMENT = "Notes without a document"


def clean_tags(tags: list[str]) -> list[str]:
    """Tags as Obsidian takes them: no leading #, no spaces (made hyphens), no duplicates."""
    cleaned: list[str] = []
    for tag in tags:
        tag = re.sub(r"\s+", "-", tag.strip().lstrip("#").strip())
        tag = re.sub(r"[^\w\-/]", "", tag)[:TAG_CHARS]
        if tag and tag.lower() not in {t.lower() for t in cleaned}:
            cleaned.append(tag)
    if len(cleaned) > MAX_TAGS:
        raise Refused(f"At most {MAX_TAGS} tags")
    return cleaned


def clean_body(body: str) -> str:
    body = body.replace("\r\n", "\n").strip()
    if not body:
        raise Refused("A note needs some text")
    if len(body) > BODY_CHARS:
        raise Refused(f"At most {BODY_CHARS} characters")
    return body


def row_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"], "doc_id": row["doc_id"], "doc_title": row["doc_title"] or "", "page": row["page"],
        "rects": json.loads(row["rects"]) if row["rects"] else [], "bbox": json.loads(row["bbox"]) if row["bbox"] else None,
        "quote": row["quote"] or "", "prefix": row["prefix"] or "", "suffix": row["suffix"] or "",
        "body_md": row["body_md"], "tags": json.loads(row["tags"] or "[]"),
        "anchor_status": (row["anchor_status"] or "ok") if row["page"] is not None else None,
        "created_at": row["created_at"], "updated_at": row["updated_at"],
    }


def get(conn: sqlite3.Connection, note_id: int) -> dict:
    row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    if row is None:
        raise NotFound(f"No note {note_id}")
    return row_dict(row)


def listing(conn: sqlite3.Connection, doc_id: int | None = None, query: str = "") -> list[dict]:
    """Notes, newest first; of one document if asked; with every word of `query` in the note,
    its quote or its tags (the panel's filter; searching notes with the books is step 3)."""
    sql, params = "SELECT * FROM notes", []
    where = []
    if doc_id is not None:
        where.append("doc_id = ?")
        params.append(doc_id)
    for word in query.split()[:10]:
        where.append("(body_md LIKE ? ESCAPE '\\' OR quote LIKE ? ESCAPE '\\' OR tags LIKE ? ESCAPE '\\')")
        like = "%" + word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        params += [like, like, like]
    if where:
        sql += " WHERE " + " AND ".join(where)
    return [row_dict(row) for row in conn.execute(sql + " ORDER BY updated_at DESC, id DESC", params)]


def create(conn: sqlite3.Connection, settings: Settings, body_md: str, tags: list[str] | None = None,
           passage: dict | None = None) -> dict:
    """A new note; `passage` is a selection in the viewer ({doc_id, page, rects, quote}), or None."""
    body_md, tags = clean_body(body_md), clean_tags(tags or [])
    place: dict = {}
    title = None
    if passage is not None:
        place = annotations.anchor(conn, settings, passage["doc_id"], passage["page"], passage["rects"], passage["quote"])
        title = conn.execute("SELECT title FROM documents WHERE id = ?", (passage["doc_id"],)).fetchone()["title"]
    cursor = conn.execute(
        "INSERT INTO notes (doc_id, doc_title, page, bbox, rects, quote, prefix, suffix, anchor_sha, body_md, tags)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (place.get("doc_id"), title, place.get("page"), json.dumps(place["bbox"]) if place else None,
         json.dumps(place["rects"]) if place else None, place.get("quote"), place.get("prefix"),
         place.get("suffix"), place.get("sha"), body_md, json.dumps(tags)))
    conn.commit()
    return {**get(conn, cursor.lastrowid), "snapped": place.get("snapped")}


def update(conn: sqlite3.Connection, note_id: int, body_md: str | None = None, tags: list[str] | None = None,
           passage: dict | None = None, settings: Settings | None = None) -> dict:
    """Change a note's text or tags; with `passage`, attach it to a new passage (after its old
    one was lost when the book changed, or to move it)."""
    get(conn, note_id)
    changes: dict = {}
    if passage is not None:
        place = annotations.anchor(conn, settings, passage["doc_id"], passage["page"], passage["rects"], passage["quote"])
        title = conn.execute("SELECT title FROM documents WHERE id = ?", (passage["doc_id"],)).fetchone()["title"]
        changes.update({"doc_id": place["doc_id"], "doc_title": title, "page": place["page"],
                        "rects": json.dumps(place["rects"]), "bbox": json.dumps(place["bbox"]), "quote": place["quote"],
                        "prefix": place["prefix"], "suffix": place["suffix"], "anchor_sha": place["sha"],
                        "anchor_status": None})
    if body_md is not None:
        changes["body_md"] = clean_body(body_md)
    if tags is not None:
        changes["tags"] = json.dumps(clean_tags(tags))
    if changes:
        conn.execute(f"UPDATE notes SET {', '.join(f'{name} = ?' for name in changes)}, updated_at = datetime('now')"
                     " WHERE id = ?", (*changes.values(), note_id))
        conn.commit()
    return get(conn, note_id)


def delete(conn: sqlite3.Connection, note_id: int) -> None:
    if conn.execute("DELETE FROM notes WHERE id = ?", (note_id,)).rowcount == 0:
        raise NotFound(f"No note {note_id}")
    conn.commit()


# ---- Export for Obsidian ----

def file_name(title: str) -> str:
    """A file name Obsidian and Windows both take."""
    name = re.sub(r'[\\/:*?"<>|#^\[\]]', "-", title).strip(" .-") or "Untitled"
    return name[:120]


def note_block(note: dict) -> str:
    heading = f"p. {note['page']}" if note["page"] else "No page"
    lines = [f"## {heading} · note {note['id']}", ""]
    if note["quote"]:
        lines += [f"> {line}" for line in note["quote"].splitlines()] + [""]
    lines += [dollar_math(note["body_md"]), ""]     # the same conversion as made files (review X3)
    if note["tags"]:
        lines += [" ".join(f"#{tag}" for tag in note["tags"]), ""]
    return "\n".join(lines)


def document_file(doc: dict | None, notes: list[dict]) -> str:
    tags = sorted({tag for note in notes for tag in note["tags"]}, key=str.lower)
    if doc is not None:
        front = {"title": doc["title"] or doc["path"], "authors": doc.get("author") or None, "year": doc.get("year"),
                 "source": doc["path"]}
    else:
        front = {"title": NO_DOCUMENT}
    front.update({"tags": ["study-notes", *tags], "exported": date.today().isoformat(), "notes": len(notes)})
    front = {key: value for key, value in front.items() if value is not None}
    head = "---\n" + yaml.safe_dump(front, allow_unicode=True, sort_keys=False) + "---\n\n"
    title = front["title"]
    # By page, then down the page (a note without a box last on its page), then as written.
    order = sorted(notes, key=lambda n: (n["page"] or 0, (n["bbox"] or [0, 1e9])[1], (n["bbox"] or [0, 0])[0], n["id"]))
    body = "\n".join(note_block(note) for note in order)
    return f"{head}# {title}: notes\n\n{body}"


def export_files(conn: sqlite3.Connection, doc_id: int | None = None) -> dict[str, str]:
    """{file name: Markdown}, one file per document that has notes, and one for notes whose
    document is gone or that never had one (each titled with the book it came from)."""
    notes = listing(conn, doc_id)
    by_doc: dict[int | None, list[dict]] = {}
    for note in notes:
        by_doc.setdefault(note["doc_id"], []).append(note)
    files: dict[str, str] = {}
    for key, group in by_doc.items():
        doc = None
        if key is not None:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (key,)).fetchone()
            doc = dict(row) if row else None
        name = file_name(doc["title"] or doc["path"]) if doc else NO_DOCUMENT
        if f"{name}.md" in files:
            name = f"{name} ({key})"
        if doc is None:
            for note in group:
                if note["doc_title"]:
                    note["quote"] = note["quote"] and f"{note['quote']}\n— {note['doc_title']}"
        files[f"{name}.md"] = document_file(doc, group)
    return files


def export_zip(conn: sqlite3.Connection, doc_id: int | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in export_files(conn, doc_id).items():
            archive.writestr(f"Study notes/{name}", text)
    return buffer.getvalue()


# ---- Search (step 3): the reader's notes beside the books ----

def note_text(note: dict) -> str:
    """What a note says, as it is indexed and as the model reads it."""
    parts = [note["body_md"]]
    if note["quote"]:
        parts.append(f'(Attached to the passage: "{note["quote"]}")')
    if note["tags"]:
        parts.append("Tags: " + ", ".join(note["tags"]))
    return "\n\n".join(parts)


def index_notes(conn: sqlite3.Connection, settings: Settings, embed) -> int:
    """Give every note its keywords and vector if it lacks them (written before the index
    existed, or while the embedder was down). Notes are few, so this runs before each search.
    Returns how many vectors were made."""
    from app.ingest.embed import serialize

    # Embed first, write after: the keyword insert opened the write lock and held it through the
    # embedder's HTTP call, up to its 300 s timeout (review C2, Nov 2026).
    missing = [row_dict(row) for row in conn.execute(
        "SELECT * FROM notes WHERE id NOT IN (SELECT note_id FROM note_vec)")]
    conn.commit()       # end the read before the embedder is called
    vectors = embed([note_text(note) for note in missing], settings) if missing else []
    with conn:
        conn.execute("INSERT INTO note_fts (rowid, text) SELECT id, body_md || ' ' || coalesce(quote, '') || ' ' || tags"
                     " || ' ' || coalesce(doc_title, '') FROM notes WHERE id NOT IN (SELECT rowid FROM note_fts)")
        for note, vector in zip(missing, vectors):
            still = conn.execute("SELECT 1 FROM notes WHERE id = ?", (note["id"],)).fetchone()
            done = conn.execute("SELECT 1 FROM note_vec WHERE note_id = ?", (note["id"],)).fetchone()
            if still and not done:      # removed, or indexed by another search, while it was embedded
                conn.execute("INSERT INTO note_vec (note_id, embedding) VALUES (?, ?)",
                             (note["id"], serialize(vector, settings.embed_dimensions)))
    return len(missing)


def searchable_notes(conn: sqlite3.Connection, doc_ids, hidden) -> list[int]:
    """The notes a search may return. With documents chosen, their notes; over the whole
    library, every note (a free one, or one whose book was removed, too), except notes on
    documents in a hidden collection."""
    if doc_ids is not None:
        if not doc_ids:
            return []
        places = ",".join("?" * len(doc_ids))
        return [row[0] for row in conn.execute(f"SELECT id FROM notes WHERE doc_id IN ({places})", list(doc_ids))]
    rows = conn.execute("SELECT n.id, d.collection FROM notes n LEFT JOIN documents d ON d.id = n.doc_id")
    return [row[0] for row in rows if row[1] is None or row[1] not in hidden]


def ranked_notes(conn: sqlite3.Connection, query: str, vector: bytes, limit: int, allowed: list[int]) -> list[list[int]]:
    """The allowed notes by keywords and by vector, best first: two lists for RRF."""
    from app.retrieval.search import fts_query

    if not allowed:
        return [[], []]
    ids = ",".join(map(str, allowed))
    expression = fts_query(query)
    keyword = [row[0] for row in conn.execute(
        f"SELECT rowid FROM note_fts WHERE note_fts MATCH ? AND rowid IN ({ids}) ORDER BY rank LIMIT ?",
        (expression, limit))] if expression else []
    similar = [row[0] for row in conn.execute(
        f"SELECT note_id FROM note_vec WHERE embedding MATCH ? AND k = ? AND note_id IN ({ids}) ORDER BY distance",
        (vector, limit))]
    return [keyword, similar]


def as_hit(note: dict, score: float, keyword_rank: int | None, vector_rank: int | None, title: str | None):
    """A note as a search hit. Its id is the note's, negated, so it can't be taken for a chunk."""
    from app.retrieval.search import Hit

    book = title or note["doc_title"]
    label = f"Your note on {book}" if book else "Your note"
    return Hit(
        chunk_id=-note["id"], doc_id=note["doc_id"] or 0, doc_path="", doc_title=label, heading_path=[],
        text=note_text(note), page_start=note["page"], page_end=note["page"], block_ids=[],
        meta={"note_id": note["id"]}, score=score, keyword_rank=keyword_rank, vector_rank=vector_rank,
        regions=[{"page": note["page"], "bbox": rect} for rect in note["rects"]] if note["page"] else [],
        chunk_ids=[-note["id"]], focus_page=note["page"], kind="note",
    )


def note_hits(conn: sqlite3.Connection, query: str, vector: bytes, settings: Settings, doc_ids, embed) -> list:
    """The best notes for a query, as hits scored by RRF like the chunks."""
    from app.ingest.metadata import shown_title
    from app.retrieval.search import rrf

    if settings.search_note_candidates <= 0:
        return []
    if not conn.execute("SELECT 1 FROM notes LIMIT 1").fetchone():
        return []
    index_notes(conn, settings, embed)
    from app.ingest.deck import excluded_note_ids
    kept_out = excluded_note_ids(conn, settings)
    allowed = [n for n in searchable_notes(conn, doc_ids, settings.hidden_collections) if n not in kept_out]
    keyword, similar = ranked_notes(conn, query, vector, settings.search_depth, allowed)
    scores = rrf([keyword, similar], settings.rrf_k)
    hits = []
    for note_id in list(scores)[:settings.search_note_candidates]:
        note = get(conn, note_id)
        title_row = conn.execute("SELECT title FROM documents WHERE id = ?", (note["doc_id"],)).fetchone() \
            if note["doc_id"] else None
        title = shown_title(title_row["title"]) if title_row and title_row["title"] else None
        hits.append(as_hit(note, scores[note_id], keyword.index(note_id) + 1 if note_id in keyword else None,
                           similar.index(note_id) + 1 if note_id in similar else None, title))
    return hits
