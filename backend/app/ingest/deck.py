"""PowerPoint decks in the library (roadmap item 7).

A deck goes through the PDF pipeline as LibreOffice renders it (app/office.py): page n of the
render is slide n, so the viewer, citations, figures, formulas, highlights and notes all work
as for a book. The render is kept under data/renders, named by the deck's SHA-256; the
library is only read. What the render can't give, the deck's own text can: python-pptx's
slide text (text in groups and tables, equations as plain characters) and speaker notes,
read in the container into the parse folder as slides.json.

Chunks: one per slide, headed by the slide's title, whose text is the slide as the deck has
it plus any display formula read from the render; one per slide with speaker notes; figures
as for a book.

Pages can be kept out entirely (metadata.yaml, `exclude_pages`): no block, chunk, figure or
summary of them exists, so nothing of them is searched or sent to a hosted model, and the
annotation tools refuse them for hosted models. The viewer still shows them, locally. Pages
added to the list later are removed from what is stored (apply_exclusions), at start, after
the details form saves, and before the next search.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.config import Settings
from app.ingest.chunker import (DROPPED_EQUATION, PATH_SEPARATOR, Block, Chunk, _crops, _with_note, chunk_blocks,
                                 estimate_tokens)

DECK_SUFFIXES = {".pptx"}
SLIDES_FILE = "slides.json"
NOTES_HEADING = "Speaker notes"
WORD = re.compile(r"\w+", re.UNICODE)


def is_deck(path_or_kind: str) -> bool:
    text = str(path_or_kind).lower()
    return text == "pptx" or Path(text).suffix in DECK_SUFFIXES


def render_path(settings: Settings, sha256: str) -> Path:
    return settings.renders_dir / f"{sha256}.pdf"


def pdf_path(settings: Settings, doc) -> Path:
    """The PDF to open for a document: the file itself, or a deck's render."""
    if is_deck(doc["kind"] if "kind" in doc.keys() else doc["path"]):
        return render_path(settings, doc["sha256"])
    return settings.library_dir / doc["path"]


def excluded_pages(settings: Settings, path: str) -> set[int]:
    """Pages the reader keeps out of search and away from hosted models (metadata.yaml)."""
    from app.ingest.metadata import read_overrides

    entry = read_overrides(settings.library_dir).get(path, {})
    return {int(page) for page in entry.get("exclude_pages") or []}


def is_hosted(option) -> bool:
    return option.provider != "ollama"


def refuse_excluded(conn, settings: Settings, option, doc_id: int, first: int, last: int) -> str | None:
    """Why a hosted option can't be given these pages, or None. Kept-out pages never leave
    the machine: the reader set exclude_pages for exactly that."""
    from app.ingest.metadata import MetadataError

    if not is_hosted(option):
        return None
    row = conn.execute("SELECT path FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row is None:
        return None
    try:
        kept_out = excluded_pages(settings, row["path"])
    except MetadataError as error:      # which pages are kept out isn't known: none go
        return f"{error}. Until it is fixed, no page goes to a hosted model; choose a local option"
    hit = sorted(set(range(first, last + 1)) & kept_out)
    if not hit:
        return None
    pages = ", ".join(str(page) for page in hit)
    return (f"Page{'s' if len(hit) > 1 else ''} {pages} of this document {'are' if len(hit) > 1 else 'is'} "
            f"kept away from hosted models (exclude_pages in metadata.yaml); choose a local option")


def excluded_note_ids(conn, settings: Settings) -> set[int]:
    """Notes anchored on kept-out pages: never searched, like the pages themselves. While
    metadata.yaml can't be read, every note anchored on a page is left out."""
    from app.ingest.metadata import MetadataError, read_overrides

    try:
        overrides = read_overrides(settings.library_dir)
    except MetadataError:
        return {row["id"] for row in conn.execute("SELECT id FROM notes WHERE page IS NOT NULL")}
    pages = {path: {int(p) for p in entry.get("exclude_pages") or []} for path, entry in overrides.items()
             if entry.get("exclude_pages")}
    if not pages:
        return set()
    rows = conn.execute("SELECT n.id, n.page, d.path FROM notes n JOIN documents d ON d.id = n.doc_id "
                        "WHERE n.page IS NOT NULL")
    return {row["id"] for row in rows if row["page"] in pages.get(row["path"], set())}


def read_slides(parsed: Path) -> dict | None:
    path = parsed / SLIDES_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _words(text: str) -> set[str]:
    return {word.casefold() for word in WORD.findall(text)}


def deck_title(slides: dict, excluded: set[int] = frozenset()) -> tuple[str | None, bool]:
    """The deck's title, and whether its stored properties can be trusted.

    A deck's core title is often a template's ("PowerPoint Presentation", or a placeholder
    in another language), so it counts only if its words appear on the slides. Otherwise
    the first kept slide's title. When the core title is rejected, so is the core author:
    both came from the same template.
    """
    shown = [s for s in slides["slides"] if s["slide"] not in excluded]
    text = _words(" ".join(" ".join(s["lines"]) for s in shown))
    core = (slides.get("title") or "").strip()
    words = _words(core)
    if core and words and len(words & text) / len(words) >= 0.8:
        return core, True
    for slide in shown:
        if not slide.get("hidden") and slide.get("title"):
            return slide["title"], False
    return None, False


def slide_chunks(blocks: list[Block], slides: dict, title: str | None,
                 excluded: set[int] = frozenset()) -> list[Chunk]:
    """One chunk per slide (and one per slide with speaker notes), figures as for a book.

    The slide's text is python-pptx's, which keeps text the render draws as shapes; Docling's
    display formulas from the render follow it as LaTeX. A slide's chunk carries its blocks,
    so a citation highlights the slide's regions.
    """
    by_page: dict[int, list[Block]] = {}
    for block in blocks:
        if block.page is not None and block.page not in excluded:
            by_page.setdefault(block.page, []).append(block)
    chunks: list[Chunk] = []

    def emit(kind, path, text, members, page, meta=None):
        chunks.append(Chunk(kind=kind, seq=len(chunks), heading_path=list(path), text=text,
                            token_count=estimate_tokens(text), block_orders=sorted({b.order for b in members}),
                            page_start=page, page_end=page, meta=meta or {}))

    for slide in slides["slides"]:
        number = slide["slide"]
        if number in excluded:
            continue
        members = by_page.get(number, [])
        heading = slide.get("title") or f"Slide {number}"
        path = [part for part in (title, heading) if part]
        prefix = PATH_SEPARATOR.join(path) + " ›\n\n"
        lines = [line for line in slide["lines"] if line.strip()]
        # As in a book's chunk: the block keeps its $$...$$, and an unverified one gets the
        # warning and its crop, for the reader and the model (chunker._with_note, _crops).
        formulas = [_with_note(b) for b in members if b.kind == "equation" and b.text.strip()]
        body = "\n".join(lines)
        if formulas:
            body += "\n\n" + "\n\n".join(b.text.strip() for b in formulas)
        if body.strip():
            kept = [b for b in members if b.kind not in ("figure", DROPPED_EQUATION)]
            emit("text", path, f"{prefix}Slide {number}\n{body}", kept, number,
                 {"slide": number, **_crops(formulas)})
        notes = (slide.get("notes") or "").strip()
        if notes:
            notes_path = path + [NOTES_HEADING]
            emit("text", notes_path, PATH_SEPARATOR.join(notes_path) + f" ›\n\nSpeaker notes, slide {number}\n{notes}",
                 [], number, {"slide": number, "speaker_notes": True})
        for figure in (b for b in members if b.kind == "figure"):
            if figure.text.strip():         # caption-less figures get a chunk once described
                emit("figure", path, prefix + figure.text, [figure], number, dict(figure.meta))
    return chunks


def kept_blocks(blocks: list[Block], excluded: set[int]) -> list[Block]:
    return [block for block in blocks if block.page not in excluded]


_applied: dict[Path, tuple[int, int] | None] = {}      # metadata.yaml's (mtime, size) when last applied


def apply_exclusions(conn, settings: Settings, only_if_changed: bool = False) -> int:
    """Remove what is stored of kept-out pages: blocks, figures, and every chunk that touches
    one (a summary over a range takes its pages' content with it). Ingestion already leaves
    them out; this is for pages added to exclude_pages after a document was ingested, which
    stayed searchable and could reach Deep or Qwen (review R2-S1, Nov 2026). Triggers drop the
    chunks' keyword and vector rows. Returns how many chunks went.

    `only_if_changed`: skip unless metadata.yaml changed since the last run, so search can call
    this before every question for the cost of a stat."""
    from app.ingest.metadata import METADATA_FILE, read_overrides

    file = settings.library_dir / METADATA_FILE
    try:
        stat = file.stat()
        seen = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        seen = None
    if only_if_changed and file in _applied and _applied[file] == seen:
        return 0
    pages = {path: {int(page) for page in entry.get("exclude_pages") or []}
             for path, entry in read_overrides(settings.library_dir).items() if entry.get("exclude_pages")}
    removed = 0
    with conn:
        for path, kept_out in pages.items():
            row = conn.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()
            if row is None:
                continue
            for page in sorted(kept_out):
                conn.execute("DELETE FROM blocks WHERE doc_id = ? AND page = ?", (row["id"], page))
                conn.execute("DELETE FROM figures WHERE doc_id = ? AND page = ?", (row["id"], page))
                removed += conn.execute(
                    "DELETE FROM chunks WHERE doc_id = ? AND (page_start IS NULL "     # no range: it may hold any page
                    "OR (page_start <= ? AND coalesce(page_end, page_start) >= ?))",
                    (row["id"], page, page)).rowcount
    _applied[file] = seen
    return removed


__all__ = ["DECK_SUFFIXES", "apply_exclusions", "chunk_blocks", "deck_title", "excluded_pages", "is_deck",
           "kept_blocks", "pdf_path", "read_slides", "render_path", "slide_chunks"]
