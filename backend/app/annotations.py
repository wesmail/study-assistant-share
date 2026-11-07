"""Highlights in the viewer (roadmap item 4, step 1): saved, listed, changed and removed.

A highlight is anchored like every annotation: document, page, boxes and the quoted text.
Boxes are PDF points from the page's top left, the convention citations already use, so
the viewer only scales them.

The browser sends what the reader selected in the pdf.js text layer: one box per run of
text and the selected string. Those boxes follow pdf.js's font metrics, not the page's
words, so they are snapped to PyMuPDF's word boxes here: every word the selection mostly
covers, merged into one box per line. The snapped highlight is exactly what PyMuPDF will
write into an annotated copy (step 5), and its quote is PyMuPDF's text, the text a
re-ingested book is searched for (step 4). A snap is kept only if its words read like the
selection (`reads_like`: at least 90% of the selected text found in the words, spaces
ignored, and nothing more in them than the rest of a word cut at either edge), or, when
pdf.js reads some glyphs differently from PyMuPDF (it reads the floor brackets of
"⌊j/l⌋" in the Sparse Transformers paper as "bj/lc"), if at least 75% of the text is found
and the selection's boxes and the words' boxes overlap with an IoU of at least 0.6.
Otherwise the browser's boxes and text are kept as they came, and the reply says it wasn't
snapped.

The words just before and after the quote are kept too, to tell repeated quotes apart when
the anchor is looked for again.

Original files are only read here, never written.
"""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import fitz

from app.errors import NotFound, Refused  # noqa: F401  (re-exported: annotations.Refused)
from app.ingest.deck import render_path
from app.config import Settings

COLORS = ("yellow", "green", "blue", "pink")
KINDS = ("highlight",)
STATUSES = ("suggested", "accepted")
QUOTE_CHARS = 5000
BODY_CHARS = 20000
MAX_RECTS = 400
# A word belongs to the selection when the selection covers at least this share of its box.
WORD_COVER = 0.5
SNAP_SIMILARITY = 0.9
# When glyphs are read differently: less of the text, but the boxes must agree.
LOOSE_SIMILARITY = 0.75
LOOSE_IOU = 0.6
CONTEXT_WORDS = 8
TALL = 1.5      # a word this many times the line's median height is trimmed to the line (line_box)


Rect = tuple[float, float, float, float]


def squeeze(text: str) -> str:
    """Text as compared: compatibility forms unfolded (ligatures, wide forms), case and all
    whitespace dropped, since pdf.js and PyMuPDF space and break lines differently; accents
    and other marks dropped too (combining marks, and modifier symbols such as the tilde of
    B̃, which PyMuPDF can give as a word of its own on the line above), since the two place
    them differently."""
    unfolded = unicodedata.normalize("NFKD", text)
    kept = "".join(char for char in unfolded
                   if not unicodedata.combining(char) and unicodedata.category(char) != "Sk")
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", kept)).lower()


def similarity(a: str, b: str) -> float:
    a, b = squeeze(a), squeeze(b)
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def boxes_iou(a: list[Rect], b: list[Rect]) -> float:
    """IoU of two regions, each the union of its boxes (boxes in one region taken as not
    overlapping each other, as lines don't)."""
    common = sum(overlap(x, y) for x in a for y in b)
    total = sum(map(area, a)) + sum(map(area, b)) - common
    return common / total if total > 0 else 0.0


def reads_like(words: list[str], quote: str, share: float = SNAP_SIMILARITY) -> bool:
    """Whether the words a selection covers are that selection, taken to whole words: at
    least 90% of the selected text is found in them, in order, and outside it they hold only
    the rest of the first and last words, which a selection may cut (part of each edge word
    must be selected)."""
    words = [word for word in words if squeeze(word)]   # a lone accent is no edge
    text, wanted = squeeze("".join(words)), squeeze(quote)
    if not text or not wanted:
        return False
    blocks = [block for block in SequenceMatcher(None, wanted, text, autojunk=False).get_matching_blocks()
              if block.size]
    matched = sum(block.size for block in blocks)
    if matched < share * len(wanted):
        return False
    start, end = blocks[0].b, blocks[-1].b + blocks[-1].size
    inside_extra = (end - start) - matched
    return (start < len(squeeze(words[0])) and len(text) - end < len(squeeze(words[-1]))
            and inside_extra <= (1 - share) * len(wanted) + 2)


def tidy(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def area(rect: Rect) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def overlap(a: Rect, b: Rect) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def union(rects: list[Rect]) -> Rect:
    return (min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects), max(r[3] for r in rects))


def line_box(boxes: list[Rect]) -> Rect:
    """One line's box: the union of its words, after trimming any word much taller than the
    line's others (a root sign, a big bracket: over TALL x the median height) to the band most
    words share (the median top and bottom). A plain union reaches into the lines above and
    below on such a line, and a PDF highlight would cover words of those lines too; trimming
    every word to the band would cut a stacked fraction's numerators and denominators, which
    are words of ordinary height at other heights (gate 2's check, Nov 6)."""
    def median(values: list[float]) -> float:
        values = sorted(values)
        middle = len(values) // 2
        return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2

    height = median([b[3] - b[1] for b in boxes])
    top, bottom = median([b[1] for b in boxes]), median([b[3] for b in boxes])
    kept = [(b[0], max(b[1], top), b[2], min(b[3], bottom)) if b[3] - b[1] > TALL * height else b for b in boxes]
    return union(kept)


def page_words(path: Path, page: int) -> tuple[list[tuple], Rect]:
    """PyMuPDF's words on a page, in its reading order, and the page's size."""
    with fitz.open(path) as pdf:
        if not 1 <= page <= pdf.page_count:
            raise Refused(f"The document has no page {page}")
        fitz_page = pdf[page - 1]
        return fitz_page.get_text("words", sort=False), tuple(fitz_page.rect)


def snap(rects: list[Rect], quote: str, words: list[tuple]) -> dict | None:
    """The words the selection covers, one box per line, with their text and context, or
    None when they don't read like the selection."""
    chosen = [index for index, word in enumerate(words)
              if area(word[:4]) > 0 and sum(overlap(word[:4], rect) for rect in rects) >= WORD_COVER * area(word[:4])]
    if not chosen:
        return None
    lines: dict[tuple[int, int], list[Rect]] = {}
    for index in chosen:
        word = words[index]
        lines.setdefault((word[5], word[6]), []).append(word[:4])
    line_boxes = [line_box(boxes) for boxes in lines.values()]
    texts = [words[index][4] for index in chosen]
    if not (reads_like(texts, quote) or (reads_like(texts, quote, LOOSE_SIMILARITY)
                                         and boxes_iou(rects, line_boxes) >= LOOSE_IOU)):
        return None
    text = " ".join(texts)
    first, last = chosen[0], chosen[-1]
    return {
        "rects": line_boxes,
        "quote": text,
        "prefix": " ".join(word[4] for word in words[max(0, first - CONTEXT_WORDS):first]),
        "suffix": " ".join(word[4] for word in words[last + 1:last + 1 + CONTEXT_WORDS]),
    }


def clean_rects(rects: list[list[float]], page_rect: Rect) -> list[Rect]:
    if not rects:
        raise Refused("A highlight needs at least one box")
    if len(rects) > MAX_RECTS:
        raise Refused(f"At most {MAX_RECTS} boxes")
    cleaned = []
    for rect in rects:
        if len(rect) != 4:
            raise Refused("A box is four numbers: x0, y0, x1, y1")
        x0, y0, x1, y1 = (float(value) for value in rect)
        x0, x1 = sorted((max(page_rect[0], min(x0, page_rect[2])), max(page_rect[0], min(x1, page_rect[2]))))
        y0, y1 = sorted((max(page_rect[1], min(y0, page_rect[3])), max(page_rect[1], min(y1, page_rect[3]))))
        if x1 - x0 > 0 and y1 - y0 > 0:
            cleaned.append((round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)))
    if not cleaned:
        raise Refused("The boxes are outside the page")
    return cleaned


def rects_of(row) -> list[list[float]]:
    """A passage's boxes: its rects, or for rows from before rects were kept its bbox as one box."""
    if row["rects"]:
        return json.loads(row["rects"])
    return [json.loads(row["bbox"])] if row["bbox"] else []


def row_dict(row: sqlite3.Row) -> dict:
    rects = rects_of(row)
    return {
        "id": row["id"], "doc_id": row["doc_id"], "page": row["page"], "rects": rects,
        "bbox": json.loads(row["bbox"]), "quote": row["quote"], "prefix": row["prefix"] or "",
        "suffix": row["suffix"] or "", "kind": row["kind"], "color": row["color"], "body": row["body"],
        "origin": row["origin"], "status": row["status"], "created_at": row["created_at"],
        "anchor_status": row["anchor_status"] or "ok",
        "batch": row["batch"], "instruction": row["instruction"],
    }


def document_path(conn: sqlite3.Connection, settings: Settings, doc_id: int) -> Path:
    doc = conn.execute("SELECT path, kind, sha256 FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if doc is None:
        raise NotFound(f"No document {doc_id}")
    if doc["kind"] == "pptx":       # its LibreOffice render, page n = slide n
        path = render_path(settings, doc["sha256"])
        if not path.is_file():
            raise NotFound("The deck hasn't been rendered yet")
        return path
    if doc["kind"] != "pdf":
        raise Refused("Only PDFs and decks can be highlighted")
    library = settings.library_dir.resolve()
    path = (library / doc["path"]).resolve()
    if not path.is_relative_to(library) or not path.is_file():
        raise NotFound("The file isn't in the library")
    return path


def lost(conn: sqlite3.Connection) -> list[dict]:
    """Highlights whose passage wasn't found after their book's file changed, with the book's title."""
    rows = conn.execute("SELECT a.*, d.title AS doc_title FROM annotations a JOIN documents d ON d.id = a.doc_id"
                        " WHERE a.anchor_status = 'unanchored' ORDER BY a.doc_id, a.page, a.id")
    return [{**row_dict(row), "doc_title": row["doc_title"] or ""} for row in rows]


def listing(conn: sqlite3.Connection, doc_id: int) -> list[dict]:
    rows = conn.execute("SELECT * FROM annotations WHERE doc_id = ? ORDER BY page, id", (doc_id,))
    return [row_dict(row) for row in rows]


def anchor(conn: sqlite3.Connection, settings: Settings, doc_id: int, page: int, rects: list[list[float]],
           quote: str) -> dict:
    """A passage as selected in the viewer, made an anchor: boxes, quote and context, snapped to
    the page's words when they match the selection. Shared by highlights and notes."""
    quote = tidy(quote)
    if not quote:
        raise Refused("A passage needs the selected text")
    if len(quote) > QUOTE_CHARS:
        raise Refused(f"At most {QUOTE_CHARS} characters can be selected at once")
    words, page_rect = page_words(document_path(conn, settings, doc_id), page)
    boxes = clean_rects(rects, page_rect)
    snapped = snap(boxes, quote, words)
    if snapped:
        boxes, quote = snapped["rects"], snapped["quote"]
    sha = conn.execute("SELECT sha256 FROM documents WHERE id = ?", (doc_id,)).fetchone()[0]
    return {"doc_id": doc_id, "page": page, "sha": sha, "rects": [[round(value, 2) for value in box] for box in boxes],
            "bbox": [round(value, 2) for value in union(boxes)], "quote": quote,
            "prefix": snapped["prefix"] if snapped else "", "suffix": snapped["suffix"] if snapped else "",
            "snapped": snapped is not None}


def create(conn: sqlite3.Connection, settings: Settings, doc_id: int, page: int, rects: list[list[float]],
           quote: str, color: str = "yellow", kind: str = "highlight") -> dict:
    """Save the reader's highlight, snapped to the page's words when they match the selection."""
    if color not in COLORS:
        raise Refused(f"The colour is one of {', '.join(COLORS)}")
    if kind not in KINDS:
        raise Refused(f"The kind is one of {', '.join(KINDS)}")
    place = anchor(conn, settings, doc_id, page, rects, quote)
    cursor = conn.execute(
        "INSERT INTO annotations (doc_id, page, bbox, rects, quote, prefix, suffix, anchor_sha, kind, color, origin,"
        " status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'user', 'accepted')",
        (doc_id, page, json.dumps(place["bbox"]), json.dumps(place["rects"]), place["quote"], place["prefix"],
         place["suffix"], place["sha"], kind, color))
    conn.commit()
    row = conn.execute("SELECT * FROM annotations WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return {**row_dict(row), "snapped": place["snapped"]}


def update(conn: sqlite3.Connection, annotation_id: int, color: str | None = None, body: str | None = None,
           status: str | None = None) -> dict:
    row = conn.execute("SELECT * FROM annotations WHERE id = ?", (annotation_id,)).fetchone()
    if row is None:
        raise NotFound(f"No annotation {annotation_id}")
    changes: dict[str, str | None] = {}
    if color is not None:
        if color not in COLORS:
            raise Refused(f"The colour is one of {', '.join(COLORS)}")
        changes["color"] = color
    if body is not None:
        if len(body) > BODY_CHARS:
            raise Refused(f"At most {BODY_CHARS} characters")
        changes["body"] = body or None
    if status is not None:
        if status not in STATUSES:
            raise Refused(f"The status is one of {', '.join(STATUSES)}")
        changes["status"] = status
    if changes:
        conn.execute(f"UPDATE annotations SET {', '.join(f'{name} = ?' for name in changes)} WHERE id = ?",
                     (*changes.values(), annotation_id))
        conn.commit()
    return row_dict(conn.execute("SELECT * FROM annotations WHERE id = ?", (annotation_id,)).fetchone())


def delete(conn: sqlite3.Connection, annotation_id: int) -> None:
    if conn.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,)).rowcount == 0:
        raise NotFound(f"No annotation {annotation_id}")
    conn.commit()
