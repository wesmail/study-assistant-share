"""Re-anchoring highlights and notes after a book's file changes (roadmap item 4, step 4).

A changed file keeps its document id, but every block and chunk is made again, so an anchor
can't point at them. It holds what it needs to be found again instead: its page, its quote
(the file's own words, from snapping), the words just before and after it, and the sha256 of
the file it was made on (`anchor_sha`). When that differs from the document's, the quote is
looked for in the new file:

1. On the old page, then on the pages up to MAX_SHIFT before and after it, nearest first.
2. Exactly first, in the squeezed text (no spaces, case, accents: `annotations.squeeze`), so
   line breaks and hyphenation in a re-typeset page don't matter; then fuzzily, over runs of
   words about as long as the quote, with a similarity of at least THRESHOLD (0.9, the user's
   decision).
3. More than one place: the words around each are compared with the anchor's own, and the
   best wins only if it is clearly better (by CONTEXT_MARGIN); then the nearest page breaks a
   tie only if exactly one candidate sits there. Otherwise the anchor is unanchored: two
   equally likely places are never decided by guessing.

The result: `anchor_status` "ok" (same page, same boxes), "moved" (found elsewhere or with
changed words: new page, boxes, quote and context) or "unanchored" (not found, or not
decidable; the old page, boxes and quote are kept, for the reader to see what it was and
attach it again). Anchors made before this existed have no `anchor_sha` and are taken to be
on the file as it is.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import fitz

from app.ingest.deck import pdf_path, render_path
from app.annotations import CONTEXT_WORDS, line_box, squeeze, union
from app.config import Settings

log = logging.getLogger(__name__)

THRESHOLD = 0.9
MAX_SHIFT = 10
CONTEXT_MARGIN = 0.15
MIN_PAGE_MATCH = 0.5      # page_map: a page this unlike any new one keeps its number
TABLES = ("annotations", "notes")


@dataclass
class Page:
    words: list[tuple]
    text: str               # the squeezed words, joined
    starts: list[int]       # where each word starts in `text`

    @classmethod
    def of(cls, words: list[tuple]) -> "Page":
        text, starts = "", []
        for word in words:
            starts.append(len(text))
            text += squeeze(word[4])
        return cls(words, text, starts)

    def word_range(self, start: int, end: int) -> tuple[int, int]:
        """The words a span of `text` touches: first and last index."""
        first = max(i for i, s in enumerate(self.starts) if s <= start) if self.starts else 0
        last = max(i for i, s in enumerate(self.starts) if s < end) if self.starts else 0
        return first, last


@dataclass
class Candidate:
    page: int
    first: int
    last: int
    similarity: float
    context: float = 0.0


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio() if a or b else 1.0


def exact_on(page: Page, number: int, wanted: str) -> list[Candidate]:
    """Every place the squeezed quote is, exactly."""
    found, at = [], page.text.find(wanted)
    while at != -1 and wanted:
        first, last = page.word_range(at, at + len(wanted))
        found.append(Candidate(number, first, last, 1.0))
        at = page.text.find(wanted, at + 1)
    return found


def fuzzy_on(page: Page, number: int, wanted: str) -> list[Candidate]:
    """The best places with a similarity of at least THRESHOLD, one per cluster of runs."""
    fuzzy: list[Candidate] = []
    low, high = 0.8 * len(wanted), 1.2 * len(wanted)
    for first in range(len(page.words)):
        start = page.starts[first]
        for last in range(first, len(page.words)):
            end = page.starts[last + 1] if last + 1 < len(page.starts) else len(page.text)
            if end - start > high:
                break
            if end - start < low:
                continue
            matcher = SequenceMatcher(None, wanted, page.text[start:end], autojunk=False)
            if matcher.real_quick_ratio() < THRESHOLD or matcher.quick_ratio() < THRESHOLD:
                continue
            ratio = matcher.ratio()
            if ratio >= THRESHOLD:
                fuzzy.append(Candidate(number, first, last, ratio))
    # Overlapping runs around one place: keep the best of each cluster.
    fuzzy.sort(key=lambda c: -c.similarity)
    kept: list[Candidate] = []
    for candidate in fuzzy:
        if all(candidate.last < other.first or candidate.first > other.last for other in kept):
            kept.append(candidate)
    return kept


def context_of(page: Page, candidate: Candidate, prefix: str, suffix: str) -> float:
    """How well the words around a place match the anchor's own: the mean similarity of the
    text before and after it, as long as the anchor's context; 1 when the anchor has none."""
    scores = []
    if squeeze(prefix):
        want = squeeze(prefix)
        before = page.text[max(0, page.starts[candidate.first] - len(want)):page.starts[candidate.first]]
        scores.append(similarity(want, before))
    if squeeze(suffix):
        want = squeeze(suffix)
        after_start = page.starts[candidate.last + 1] if candidate.last + 1 < len(page.starts) else len(page.text)
        scores.append(similarity(want, page.text[after_start:after_start + len(want)]))
    return sum(scores) / len(scores) if scores else 1.0


def locate(pages: list[Page], old_page: int, quote: str, prefix: str, suffix: str,
           max_shift: int = MAX_SHIFT) -> tuple[Candidate | None, str]:
    """The anchor's place in the new file, and why: "found", "not found" or "ambiguous"."""
    wanted = squeeze(quote)
    if not wanted:
        return None, "not found"
    order = sorted(range(max(1, old_page - max_shift), min(len(pages), old_page + max_shift) + 1),
                   key=lambda number: (abs(number - old_page), number))
    # An exact place anywhere nearby beats a fuzzy one, so the slower fuzzy search runs only without one.
    found = [c for number in order for c in exact_on(pages[number - 1], number, wanted)]
    if not found:
        found = [c for number in order for c in fuzzy_on(pages[number - 1], number, wanted)]
    if not found:
        return None, "not found"
    if len(found) == 1:
        return found[0], "found"
    for candidate in found:
        candidate.context = context_of(pages[candidate.page - 1], candidate, prefix, suffix)
    found.sort(key=lambda c: (-(c.similarity + c.context), abs(c.page - old_page)))
    best, second = found[0], found[1]
    if (best.similarity + best.context) - (second.similarity + second.context) >= CONTEXT_MARGIN:
        return best, "found"
    level = [c for c in found if (best.similarity + best.context) - (c.similarity + c.context) < CONTEXT_MARGIN]
    nearest = min(abs(c.page - old_page) for c in level)
    closest = [c for c in level if abs(c.page - old_page) == nearest]
    return (closest[0], "found") if len(closest) == 1 else (None, "ambiguous")


def placed(page: Page, candidate: Candidate) -> dict:
    """Boxes (one per line), quote and context of a found place."""
    words = page.words[candidate.first:candidate.last + 1]
    lines: dict[tuple[int, int], list] = {}
    for word in words:
        lines.setdefault((word[5], word[6]), []).append(word[:4])
    rects = [[round(v, 2) for v in line_box(boxes)] for boxes in lines.values()]
    return {
        "page": candidate.page, "rects": rects, "bbox": [round(v, 2) for v in union([tuple(r) for r in rects])],
        "quote": " ".join(word[4] for word in words),
        "prefix": " ".join(w[4] for w in page.words[max(0, candidate.first - CONTEXT_WORDS):candidate.first]),
        "suffix": " ".join(w[4] for w in page.words[candidate.last + 1:candidate.last + 1 + CONTEXT_WORDS]),
    }


def page_map(old: list[Page], new: list[Page], max_shift: int = MAX_SHIFT) -> dict[int, int]:
    """Where each old page went in the new file, by whole-page text.

    For decks, whose old render is kept: slides repeat text (a footer on every slide, a slide
    built up over two), so after a slide is put in front, a highlight on repeated text would
    otherwise be found on the twin at its old number (5 of 40 on the two test decks, Nov 6,
    2026). Searching from the mapped page lets the nearest-page rule pick the right twin.

    The pages are aligned in order, one to one (the best total similarity, as in a diff, within
    max_shift of each other), so two identical slides map to two slides, not both to the
    first. Pages that match nothing at MIN_PAGE_MATCH or above keep their number."""
    n, m = len(old), len(new)
    sims: dict[tuple[int, int], float] = {}
    for i in range(1, n + 1):
        for j in range(max(1, i - max_shift), min(m, i + max_shift) + 1):
            sims[i, j] = similarity(old[i - 1].text, new[j - 1].text)
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best[i][j] = max(best[i - 1][j], best[i][j - 1], best[i - 1][j - 1] + sims.get((i, j), float("-inf")))
    mapped: dict[int, int] = {}
    i, j = n, m
    while i > 0 and j > 0:
        if (i, j) in sims and best[i][j] == best[i - 1][j - 1] + sims[i, j]:
            if sims[i, j] >= MIN_PAGE_MATCH:
                mapped[i] = j
            i, j = i - 1, j - 1
        elif best[i][j] == best[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return mapped


def read_pages(path: Path) -> list[Page]:
    with fitz.open(path) as pdf:
        return [Page.of(page.get_text("words", sort=False)) for page in pdf]


def reanchor_document(conn: sqlite3.Connection, settings: Settings, doc_id: int) -> dict[str, int]:
    """Find every stale anchor of a document in its file as it is now. Returns the counts by status."""
    doc = conn.execute("SELECT path, sha256, kind FROM documents WHERE id = ?", (doc_id,)).fetchone()
    counts = {"ok": 0, "moved": 0, "unanchored": 0}
    if doc is None or doc["kind"] not in ("pdf", "pptx"):
        return counts
    path = pdf_path(settings, doc).resolve()        # a deck: the render of its new version
    if not path.is_file():
        return counts       # the file is gone: anchors wait, as the document does
    # Read and search first, write after, in one short transaction: an UPDATE opened the write
    # lock and held it through read_pages and every search (review C1, Nov 2026).
    stale = [(table, row) for table in TABLES for row in conn.execute(
        f"SELECT * FROM {table} WHERE doc_id = ? AND page IS NOT NULL AND anchor_sha IS NOT NULL AND anchor_sha != ?",
        (doc_id, doc["sha256"]))]
    conn.commit()       # end the read, so nothing is held while pages are read
    changes: list[tuple[str, tuple]] = []
    if stale:
        pages = read_pages(path)
        maps: dict[str, dict[int, int]] = {}
        if doc["kind"] == "pptx":       # the old render is kept: start each search where its slide went
            for sha in {row["anchor_sha"] for _, row in stale if row["anchor_sha"]}:
                old = render_path(settings, sha)
                if old.is_file():
                    maps[sha] = page_map(read_pages(old), pages)
        for table, row in stale:
            start = maps.get(row["anchor_sha"], {}).get(row["page"], row["page"])
            candidate, why = locate(pages, min(start, len(pages)), row["quote"] or "", row["prefix"] or "",
                                    row["suffix"] or "")
            if candidate is None:
                status = "unanchored"
                changes.append((f"UPDATE {table} SET anchor_status = 'unanchored', anchor_sha = ? "
                                "WHERE id = ? AND anchor_sha = ?", (doc["sha256"], row["id"], row["anchor_sha"])))
                log.info("%s %s of document %s: %s after the file changed", table, row["id"], doc_id, why)
            else:
                new = placed(pages[candidate.page - 1], candidate)
                same = new["page"] == row["page"] and new["rects"] == json.loads(row["rects"] or "[]")
                status = "ok" if same and new["quote"] == row["quote"] else "moved"
                changes.append((
                    f"UPDATE {table} SET page = ?, rects = ?, bbox = ?, quote = ?, prefix = ?, suffix = ?,"
                    " anchor_status = ?, anchor_sha = ? WHERE id = ? AND anchor_sha = ?",
                    (new["page"], json.dumps(new["rects"]), json.dumps(new["bbox"]), new["quote"], new["prefix"],
                     new["suffix"], status, doc["sha256"], row["id"], row["anchor_sha"])))
            counts[status] += 1
    with conn:
        for table in TABLES:        # made before anchors were kept: they belong to the version there is now
            conn.execute(f"UPDATE {table} SET anchor_sha = ? WHERE doc_id = ? AND anchor_sha IS NULL AND page IS NOT NULL",
                         (doc["sha256"], doc_id))
        for statement, values in changes:      # a row changed meanwhile (its anchor_sha moved on) is left alone
            conn.execute(statement, values)
    if not stale:
        return counts
    log.info("Re-anchored document %s after its file changed: %s", doc_id, counts)
    return counts


def reanchor_stale(conn: sqlite3.Connection, settings: Settings) -> dict[int, dict[str, int]]:
    """Every document with an anchor made on another version of its file, re-anchored."""
    docs = {row[0] for table in TABLES for row in conn.execute(
        f"SELECT DISTINCT t.doc_id FROM {table} t JOIN documents d ON d.id = t.doc_id"
        " WHERE t.page IS NOT NULL AND (t.anchor_sha IS NULL OR t.anchor_sha != d.sha256)")}
    return {doc_id: reanchor_document(conn, settings, doc_id) for doc_id in sorted(docs)}
