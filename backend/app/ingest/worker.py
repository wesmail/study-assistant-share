"""Resumable ingestion: library scan, job queue and the Phase 1 stages.

Run it with `python -m app.ingest.worker` from backend/. It scans once, then
watches library/ until interrupted; `--once` drains the queue and exits.

Each job is one stage of one file version. A stage runs once the stages it needs
(REQUIRES) are done, and may die at any moment, so each is written to be repeated:

    parse     writes into a temporary folder, then renames it into place.
    formulas  saves each page Granite reads, renders a crop of every unverified
              formula, then rewrites the sidecar atomically.
    chunk     replaces the document's rows in a single transaction.
    index     commits after each batch and only embeds chunks without a vector.
              After it the document is searchable ('indexed').
    describe  figure descriptions, cached per image. A hosted describer waits for
              the reader's OK on the estimated cost (NeedsApproval).
    summarize the summary tree, every model reply cached by its input. Needs only
              index, so it doesn't wait for the figures' OK.

While a chat request runs, jobs pause at the next boundary: before a job starts,
and between Granite pages, embedding batches, figures and summary calls. Parse is
one Docling call and can't pause.

`attempts` goes up when a job is claimed, not when it fails, so a file that
kills the whole process (OOM, segfault) still runs out of attempts. Stopping
(Ctrl+C, the app shutting down) and waiting for an OK don't use one.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import shutil
import signal
import sqlite3
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.ingest import metadata
from app.config import Settings, load_settings
from app.db.connection import connect
from app.ingest import deck, figures
from app.ingest.chunker import chunk_blocks, load_blocks_jsonl
from app.ingest.outline import outline_chunks
from app.supervisor import LOCKED, die_with_parent
from app.ingest.embed import embed_texts, serialize
from app.ingest.formula_layer import page_words, printed_numbers
from app.ingest.formula_merge import (
    CompileCheck,
    FormulaReader,
    GraniteReader,
    katex_compiles,
    merge_document,
    save_crops,
)

log = logging.getLogger("ingest")

SUPPORTED_SUFFIXES = {".pdf", ".pptx"}     # a deck goes through as its LibreOffice render (app/ingest/deck.py)
EXCLUDED_FOLDERS = {"annotated"}  # our own output must never be re-ingested
TEMPORARY_SUFFIXES = {".part", ".crdownload", ".tmp", ".download"}

STAGES = ("parse", "formulas", "chunk", "index", "describe", "summarize")
REQUIRES = {"parse": (), "formulas": ("parse",), "chunk": ("formulas",), "index": ("chunk",),
            "describe": ("index",), "summarize": ("index",)}
SEARCHABLE_AFTER = "index"      # its end makes a document 'indexed'; a later stage failing doesn't undo that
LATE_STAGES = STAGES[STAGES.index(SEARCHABLE_AFTER) + 1:]
RUNNING_STATUS = {"parse": "parsing", "formulas": "parsing", "chunk": "chunking", "index": "indexing"}


class NeedsApproval(Exception):
    """A stage that would spend money and has no OK for this much yet. `detail` is shown to the reader."""

    def __init__(self, detail: dict):
        super().__init__(detail)
        self.detail = detail

Parser = Callable[[Path, Path], dict]
Embedder = Callable[[list[str]], list[list[float]]]


def is_ingestible(path: Path, library_dir: Path) -> bool:
    try:
        relative = path.relative_to(library_dir)
    except ValueError:
        return False
    if relative.parts and relative.parts[0] in EXCLUDED_FOLDERS:
        return False
    return (
        path.suffix.lower() in SUPPORTED_SUFFIXES
        and path.suffix.lower() not in TEMPORARY_SUFFIXES
        and not any(part.startswith((".", "~$")) for part in relative.parts)
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Detect: decide what each file needs
# --------------------------------------------------------------------------


def _create_jobs(conn: sqlite3.Connection, doc_id: int, sha256: str) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO jobs (doc_id, sha256, stage, stage_order) VALUES (?, ?, ?, ?)",
        [(doc_id, sha256, stage, i) for i, stage in enumerate(STAGES)],
    )


def _settled_status(conn: sqlite3.Connection, doc: sqlite3.Row) -> str:
    """The status a document returns to when its file reappears."""
    rows = conn.execute(
        "SELECT stage, status FROM jobs WHERE doc_id = ? AND sha256 = ?", (doc["id"], doc["sha256"])
    ).fetchall()
    states = {row["stage"]: row["status"] for row in rows}
    if any(status == "failed" for stage, status in states.items() if stage not in LATE_STAGES):
        return "failed"
    return "indexed" if states.get(SEARCHABLE_AFTER) == "done" else "queued"


def add_late_stages(conn: sqlite3.Connection, settings: Settings) -> int:
    """Give documents indexed before a stage existed that stage's job: done if its work is
    already there (descriptions cached, summaries stored, as for the five books made by hand),
    pending otherwise. Returns how many jobs were added."""
    rows = conn.execute(
        "SELECT d.id, d.sha256 FROM documents d JOIN jobs j ON j.doc_id = d.id AND j.sha256 = d.sha256 "
        "WHERE j.stage = ? AND j.status = 'done'", (SEARCHABLE_AFTER,)
    ).fetchall()
    added = 0
    for row in rows:
        have = {r[0] for r in conn.execute("SELECT stage FROM jobs WHERE doc_id = ? AND sha256 = ?",
                                           (row["id"], row["sha256"]))}
        for stage in LATE_STAGES:
            if stage in have:
                continue
            if stage == "describe":
                figures.fill_image_shas(conn, settings, [row["id"]])
                finished = not figures.pending(conn, settings, [row["id"]])
            else:
                finished = conn.execute("SELECT 1 FROM chunks WHERE doc_id = ? AND kind = 'summary' LIMIT 1",
                                        (row["id"],)).fetchone() is not None
            with conn:
                conn.execute(
                    "INSERT INTO jobs (doc_id, sha256, stage, stage_order, status, detail, finished_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, CASE WHEN ? THEN datetime('now') END)",
                    (row["id"], row["sha256"], stage, STAGES.index(stage), "done" if finished else "pending",
                     json.dumps({"found_done": True}) if finished else None, finished),
                )
            added += 1
    return added


def sync_file(conn: sqlite3.Connection, settings: Settings, path: Path) -> str:
    """Bring one path up to date. Returns what happened, for logging and tests.

    The hash decides: known means skip, changed means re-ingest. The document
    row is kept on a change, because notes and annotations hang off its id.
    """
    if not is_ingestible(path, settings.library_dir):
        return "ignored"
    relative = path.relative_to(settings.library_dir).as_posix()
    row = conn.execute("SELECT * FROM documents WHERE path = ?", (relative,)).fetchone()

    if not path.exists():
        if row is None or row["status"] == "missing":
            return "unknown"
        with conn:
            conn.execute(
                "UPDATE documents SET status = 'missing', updated_at = datetime('now') WHERE id = ?",
                (row["id"],),
            )
        return "missing"

    sha256 = sha256_file(path)
    with conn:
        if row is None:
            twin = conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
            if twin is not None:
                if (settings.library_dir / twin["path"]).exists():
                    return "duplicate"
                conn.execute(
                    "UPDATE documents SET path = ?, status = ?, updated_at = datetime('now') WHERE id = ?",
                    (relative, _settled_status(conn, twin), twin["id"]),
                )
                return "moved"
            cursor = conn.execute(
                "INSERT INTO documents (path, sha256, title, kind) VALUES (?, ?, NULL, ?)",
                (relative, sha256, path.suffix.lower().lstrip(".")),
            )
            _create_jobs(conn, cursor.lastrowid, sha256)
            return "new"

        if row["sha256"] == sha256:
            if row["status"] != "missing":
                return "known"
            conn.execute(
                "UPDATE documents SET status = ?, updated_at = datetime('now') WHERE id = ?",
                (_settled_status(conn, row), row["id"]),
            )
            return "restored"

        conn.execute(
            "DELETE FROM jobs WHERE doc_id = ? AND sha256 != ? AND status != 'done'",
            (row["id"], sha256),
        )
        conn.execute(
            "UPDATE documents SET sha256 = ?, status = 'queued', updated_at = datetime('now') WHERE id = ?",
            (sha256, row["id"]),
        )
        _create_jobs(conn, row["id"], sha256)
        return "changed"


def scan_library(conn: sqlite3.Connection, settings: Settings) -> dict[str, int]:
    """Catch up with everything that happened while the worker was off."""
    counts: dict[str, int] = {}

    def record(outcome: str) -> None:
        counts[outcome] = counts.get(outcome, 0) + 1

    for path in sorted(settings.library_dir.rglob("*")):
        if path.is_file() and is_ingestible(path, settings.library_dir):
            record(sync_file(conn, settings, path))
    for row in conn.execute("SELECT path FROM documents").fetchall():
        path = settings.library_dir / row["path"]
        if not path.exists():
            record(sync_file(conn, settings, path))
    return counts


def link_summary_caches(conn: sqlite3.Connection, settings: Settings) -> dict[int, int | None]:
    """Mark which summary-cache rows each document's tree came from, for documents summarized
    before rows carried a doc_id. Needs the PDF's outline, so only while the file is there;
    no model is called. None for a document whose tree can't be rebuilt from the cache."""
    from app.ingest import summarize as tree

    rows = conn.execute(
        "SELECT DISTINCT c.doc_id, json_extract(c.meta, '$.summarizer') AS summarizer FROM chunks c "
        "JOIN documents d ON d.id = c.doc_id WHERE c.kind = 'summary' AND d.status != 'missing' "
        "AND NOT EXISTS (SELECT 1 FROM summary_cache s WHERE s.doc_id = c.doc_id)"
    ).fetchall()
    return {row["doc_id"]: tree.link_cache(conn, settings, row["doc_id"], row["summarizer"]) for row in rows
            if (settings.library_dir / conn.execute("SELECT path FROM documents WHERE id = ?",
                                                     (row["doc_id"],)).fetchone()[0]).exists()}


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


@dataclass
class StageContext:
    conn: sqlite3.Connection
    settings: Settings
    doc: sqlite3.Row
    sha256: str
    job_id: int
    parser: Parser
    embedder: Embedder
    formula_reader: FormulaReader
    compile_check: CompileCheck
    wait_for_chat: Callable[[], None] = lambda: None
    approved_usd: float | None = None
    describe_figures: Callable | None = None        # (ctx, rows) -> describe_pending's result
    summarizer_ask: Callable[[str], str] | None = None

    @property
    def parsed(self) -> Path:
        return self.settings.parsed_dir / self.sha256

    def report(self, progress: int, total: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE jobs SET progress = ?, total = ? WHERE id = ?",
                (progress, total, self.job_id),
            )


def stage_parse(ctx: StageContext) -> None:
    final = ctx.parsed
    if not final.exists():
        temporary = final.with_name(final.name + ".tmp")
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        pdf = deck.pdf_path(ctx.settings, ctx.doc)
        slides = prepare_deck(ctx, pdf, temporary) if deck.is_deck(ctx.doc["kind"]) else None
        info = ctx.parser(pdf, temporary)
        if slides is not None:
            info["title"] = deck.deck_title(slides, deck.excluded_pages(ctx.settings, ctx.doc["path"]))[0]
            info["slides"] = len(slides["slides"])
        (temporary / "info.json").write_text(json.dumps(info), encoding="utf-8")
        temporary.rename(final)
    info = json.loads((final / "info.json").read_text(encoding="utf-8"))
    with ctx.conn:
        ctx.conn.execute(
            "UPDATE documents SET pages = ? WHERE id = ?", (info.get("pages"), ctx.doc["id"]),
        )
    metadata.refresh(ctx.conn, ctx.settings)        # the parsed title is one of its sources


def prepare_deck(ctx: StageContext, pdf: Path, parsed: Path) -> dict:
    """Render a deck to PDF (kept by its hash) and read its slides and notes, both in the
    office container. The render must have one page per slide, or the slides' text and
    notes would land on the wrong pages."""
    from app import office

    source = ctx.settings.library_dir / ctx.doc["path"]
    if not pdf.is_file():
        office.run_sync(office.render_pdf(ctx.settings, source, pdf))
    slides = office.run_sync(office.read_slides(ctx.settings, source))
    import fitz
    with fitz.open(pdf) as rendered:
        pages = rendered.page_count
    if pages != len(slides["slides"]):
        raise RuntimeError(f"The render has {pages} pages for {len(slides['slides'])} slides")
    (parsed / deck.SLIDES_FILE).write_text(json.dumps(slides, ensure_ascii=False), encoding="utf-8")
    return slides


def _write_atomically(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _read_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pages_to_read(records: list[dict], pdf_document) -> list[int]:
    """Pages with a display formula, or a printed number with no formula beside it."""
    pages = {r["page"] for r in records if r["kind"] == "equation" and r.get("page")}
    for number, page in enumerate(pdf_document, 1):
        if number not in pages and printed_numbers(page_words(page), page.rect.width):
            pages.add(number)
    return sorted(pages)


def stage_formulas(ctx: StageContext) -> None:
    """Read display formulas with Granite and merge them into the sidecar.

    blocks.docling.jsonl keeps what parse wrote, and granite_pages.json keeps every
    page already read, so a rerun repeats neither the parse nor the reading.
    """
    import fitz

    sidecar = ctx.parsed / "blocks.jsonl"
    original = ctx.parsed / "blocks.docling.jsonl"
    if not original.exists():
        _write_atomically(original, sidecar.read_text(encoding="utf-8"))
    records = _read_records(original)

    pdf = deck.pdf_path(ctx.settings, ctx.doc)
    try:
        document = fitz.open(pdf)
    except (fitz.FileDataError, fitz.EmptyFileError):
        log.warning("%s has no readable text layer; keeping Docling's formulas", pdf.name)
        return

    with document:
        pages = pages_to_read(records, document)
        checkpoint = ctx.parsed / "granite_pages.json"
        read: dict[str, list[dict]] = {}
        if checkpoint.exists():
            read = json.loads(checkpoint.read_text(encoding="utf-8"))
        for position, page in enumerate(pages, 1):
            if str(page) not in read:
                ctx.wait_for_chat()
                read[str(page)] = ctx.formula_reader(pdf, page)
                _write_atomically(checkpoint, json.dumps(read, ensure_ascii=False))
            ctx.report(position, len(pages))

        merged, stats = merge_document(
            records,
            lambda number: document[number - 1],
            {int(page): formulas for page, formulas in read.items()},
            ctx.compile_check,
        )
        save_crops(merged, lambda number: document[number - 1], ctx.parsed / "formula_crops")
    stats["pages_read"] = len(pages)
    _write_atomically(sidecar, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged))
    _write_atomically(ctx.parsed / "formula_stats.json", json.dumps(stats, indent=2))
    log.info("%s: %s", ctx.doc["path"], stats)


def stage_chunk(ctx: StageContext) -> None:
    doc_id = ctx.doc["id"]
    excluded = deck.excluded_pages(ctx.settings, ctx.doc["path"])     # kept out of everything
    blocks = deck.kept_blocks(load_blocks_jsonl(ctx.parsed / "blocks.jsonl"), excluded)
    slides = deck.read_slides(ctx.parsed) if deck.is_deck(ctx.doc["kind"]) else None
    if slides is not None:
        title = ctx.conn.execute("SELECT title FROM documents WHERE id = ?", (doc_id,)).fetchone()["title"]
        chunks = deck.slide_chunks(blocks, slides, metadata.shown_title(title), excluded)
    else:
        chunks = chunk_blocks(blocks)
    if ctx.doc["kind"] == "pdf":    # where the parsed headings mostly miss the PDF's outline, use it
        chunks = outline_chunks(chunks, blocks, ctx.settings.library_dir / ctx.doc["path"], excluded)
    with ctx.conn:
        # Summaries go too: they describe the old text. A rebuild reuses cached replies for unchanged parts.
        ctx.conn.execute("DELETE FROM chunks WHERE doc_id = ? AND kind IN ('text', 'figure', 'summary')", (doc_id,))
        ctx.conn.execute("DELETE FROM blocks WHERE doc_id = ?", (doc_id,))
        ctx.conn.execute("DELETE FROM figures WHERE doc_id = ?", (doc_id,))

        block_ids: dict[int, int] = {}
        for block in blocks:
            cursor = ctx.conn.execute(
                'INSERT INTO blocks (doc_id, "order", kind, page, bbox, bboxes_json, text, docling_ref) '
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id, block.order, block.kind, block.page,
                    json.dumps(block.meta.get("bbox")),
                    json.dumps(block.meta.get("bboxes")),
                    block.text,
                    block.meta.get("docling_ref"),
                ),
            )
            block_ids[block.order] = cursor.lastrowid
            if block.kind == "figure":
                ctx.conn.execute(
                    "INSERT INTO figures (doc_id, page, image_path, caption) VALUES (?, ?, ?, ?)",
                    (doc_id, block.page, block.meta.get("image_path"), block.text),
                )

        for chunk in chunks:
            ctx.conn.execute(
                "INSERT INTO chunks (doc_id, kind, seq, heading_path, text, token_count, "
                "block_ids, page_start, page_end, meta) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id, chunk.kind, chunk.seq,
                    json.dumps(chunk.heading_path, ensure_ascii=False),
                    chunk.text, chunk.token_count,
                    json.dumps([block_ids[order] for order in chunk.block_orders]),
                    chunk.page_start, chunk.page_end,
                    json.dumps(chunk.meta, ensure_ascii=False),
                ),
            )
    # Descriptions are cached by image hash, so a re-chunked book keeps them. New figures wait
    # for the describe stage, which asks the reader first if the describer is hosted.
    figures.fill_image_shas(ctx.conn, ctx.settings, [doc_id])
    figures.apply_descriptions(ctx.conn, ctx.settings, [doc_id])
    ctx.report(len(chunks), len(chunks))


def stage_index(ctx: StageContext) -> None:
    doc_id = ctx.doc["id"]
    rows = ctx.conn.execute(
        "SELECT id, text FROM chunks WHERE doc_id = ? AND kind IN ('text', 'figure') ORDER BY id",
        (doc_id,),
    ).fetchall()
    embedded = {row[0] for row in ctx.conn.execute("SELECT chunk_id FROM chunk_vec")}
    pending = [row for row in rows if row["id"] not in embedded]
    done = len(rows) - len(pending)
    ctx.report(done, len(rows))

    size = ctx.settings.embed_batch
    for start in range(0, len(pending), size):
        ctx.wait_for_chat()
        batch = pending[start:start + size]
        vectors = ctx.embedder([row["text"] for row in batch])
        with ctx.conn:
            ctx.conn.executemany(
                "INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)",
                [
                    (row["id"], serialize(vector, ctx.settings.embed_dimensions))
                    for row, vector in zip(batch, vectors)
                ],
            )
        done += len(batch)
        ctx.report(done, len(rows))


def approved(estimate: float | None, approved_usd: float | None) -> bool:
    """The reader's OK covers this run: up to 10% (or a cent) over the estimate they saw."""
    return approved_usd is not None and (estimate or 0) <= approved_usd * 1.1 + 0.01


def stage_describe(ctx: StageContext) -> None:
    doc_id = ctx.doc["id"]
    settings = ctx.settings
    figures.fill_image_shas(ctx.conn, settings, [doc_id])
    rows = figures.pending(ctx.conn, settings, [doc_id])
    describer = settings.figure_describer
    if rows:
        estimate = figures.estimate_usd(settings, len(rows))
        if describer.hosted and not approved(estimate, ctx.approved_usd):
            raise NeedsApproval({"figures": len(rows), "usd": estimate, "model": describer.model,
                                 "provider": describer.provider})
        log.info("%s: describing %d figures with %s", ctx.doc["path"], len(rows), describer.name)
        result = ctx.describe_figures(ctx, rows)
        log.info("%s: %s", ctx.doc["path"], result)
        if result["failed"]:
            raise RuntimeError(f"{result['failed']} of {len(rows)} figures weren't described; they're tried again")
    figures.apply_descriptions(ctx.conn, settings, [doc_id])
    figures.embed_missing(ctx.conn, settings, lambda texts, _settings: ctx.embedder(texts))
    ctx.report(len(rows), len(rows))


def default_describe_figures(ctx: StageContext, rows: list) -> dict:
    import asyncio

    def pause(seconds):         # describe_pending waits for chat itself, between figures
        return asyncio.sleep(seconds)
    return asyncio.run(figures.describe_pending(
        ctx.conn, ctx.settings, rows, lambda texts, _settings: ctx.embedder(texts), report=log.info, pause=pause))


def stage_summarize(ctx: StageContext) -> None:
    from app.ingest import summarize as tree

    doc_id = ctx.doc["id"]
    if not ctx.conn.execute("SELECT 1 FROM chunks WHERE doc_id = ? AND kind = 'text' LIMIT 1", (doc_id,)).fetchone():
        log.info("%s: no text to summarize", ctx.doc["path"])
        return
    title = metadata.shown_title(ctx.doc["title"]) or ctx.doc["path"]
    root = tree.build_tree(ctx.conn, doc_id, deck.pdf_path(ctx.settings, ctx.doc), title)
    summarizer = tree.summarizer_name(ctx.settings.summarizer_model, ctx.settings.summarizer_think)

    def ask(prompt: str) -> str:        # only on a cache miss: a hit needs no model, so no pause
        while True:
            ctx.wait_for_chat()
            try:
                return ctx.summarizer_ask(prompt)
            except tree.Preempted:
                log.info("%s: a summary call was dropped for a chat request; it's asked again after", ctx.doc["path"])

    nodes = sum(1 for _ in root.walk())
    written = 0

    def report(node) -> None:
        nonlocal written
        written += 1
        ctx.report(written, nodes)

    tree.summarize(root, title, tree.cached(ctx.conn, summarizer, ask, doc_id), report)
    count = tree.store(ctx.conn, ctx.settings, doc_id, root, title, summarizer, ctx.embedder)
    log.info("%s: %d summaries", ctx.doc["path"], count)


STAGE_FUNCTIONS = {
    "parse": stage_parse, "formulas": stage_formulas, "chunk": stage_chunk, "index": stage_index,
    "describe": stage_describe, "summarize": stage_summarize,
}


def default_parser(pdf: Path, out_dir: Path) -> dict:
    """Docling, imported late so that scanning needs no models or GPU libraries.

    Footnotes become endnotes because inline ones would land between a lead-in
    sentence and its equation, and shred paragraphs.
    """
    import fitz

    from app.ingest.docling_structured import convert_pdf_to_structured_output

    summary = convert_pdf_to_structured_output(str(pdf), str(out_dir), footnotes="endnotes")
    with fitz.open(pdf) as document:
        metadata_title = (document.metadata or {}).get("title")
    return {"title": summary["title"] or metadata_title or None, "pages": summary["pages"]}


# --------------------------------------------------------------------------
# Queue
# --------------------------------------------------------------------------


class Worker:
    def __init__(
        self,
        conn: sqlite3.Connection,
        settings: Settings,
        parser: Parser = default_parser,
        embedder: Embedder | None = None,
        formula_reader: FormulaReader | None = None,
        compile_check: CompileCheck = katex_compiles,
        describe_figures: Callable | None = None,
        summarizer_ask: Callable[[str], str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        pause_seconds: float = 2.0,
    ):
        self.conn = conn
        self.settings = settings
        self.parser = parser
        self.embedder = embedder or (lambda texts: embed_texts(texts, settings))
        self.formula_reader = formula_reader or GraniteReader()
        self.compile_check = compile_check
        self.describe_figures = describe_figures or default_describe_figures
        self._summarizer_ask = summarizer_ask
        self.sleep = sleep
        self.pause_seconds = pause_seconds

    def summarizer_ask(self, prompt: str) -> str:
        if self._summarizer_ask is None:
            from app.ingest.summarize import ollama_stream_ask
            self._summarizer_ask = ollama_stream_ask(self.settings, self.settings.summarizer_model,
                                                     self.settings.summarizer_think,
                                                     busy=lambda: figures.chat_busy(self.settings))
        return self._summarizer_ask(prompt)

    def wait_for_chat(self) -> None:
        """Hold the GPU work while a chat request runs (the server's chat-busy markers)."""
        if not figures.chat_busy(self.settings):
            return
        log.info("Paused: a chat request is running")
        while figures.chat_busy(self.settings):
            self.sleep(self.pause_seconds)
        log.info("Going on")

    def recover(self) -> int:
        """Requeue jobs left 'running' by a dead process. One worker at a time."""
        with self.conn:
            return self.conn.execute(
                "UPDATE jobs SET status = 'pending' WHERE status = 'running'"
            ).rowcount

    def next_job(self) -> sqlite3.Row | None:
        """The first runnable stage of a current file version: what it requires done, no OK awaited.

        Every document is made searchable before any is described or summarized, so a new
        book isn't kept out of search by an hour of summaries for the one before it.
        """
        rows = self.conn.execute(
            """
            SELECT j.* FROM jobs j
            JOIN documents d ON d.id = j.doc_id AND d.sha256 = j.sha256
            WHERE d.status NOT IN ('missing', 'failed')
            ORDER BY j.stage_order > ?, j.doc_id, j.stage_order
            """, (STAGES.index(SEARCHABLE_AFTER),)
        ).fetchall()
        done = {(row["doc_id"], row["stage"]) for row in rows if row["status"] == "done"}
        for row in rows:
            if (row["status"] == "pending" and row["waiting"] is None
                    and all((row["doc_id"], stage) in done for stage in REQUIRES[row["stage"]])):
                return row
        return None

    def _wait_for_approval(self, job: sqlite3.Row, detail: dict) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE jobs SET status = 'pending', waiting = 'approval', detail = ?, "
                "attempts = max(attempts - 1, 0) WHERE id = ?",
                (json.dumps(detail), job["id"]),
            )
        log.info("Document %s: %s waits for your OK in the library pane: %s", job["doc_id"], job["stage"], detail)

    def _fail(self, job: sqlite3.Row, error: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE jobs SET status = 'failed', error = ?, finished_at = datetime('now') WHERE id = ?",
                (error, job["id"]),
            )
            if job["stage"] not in LATE_STAGES:     # a late stage failing leaves the document searchable
                self.conn.execute(
                    "UPDATE documents SET status = 'failed', updated_at = datetime('now') WHERE id = ?",
                    (job["doc_id"],),
                )
        log.error("%s failed for document %s: %s", job["stage"], job["doc_id"], error.splitlines()[-1])

    def run_one(self) -> bool:
        """Run one job. Returns False when nothing is runnable."""
        self.wait_for_chat()        # before claiming: parse can't pause once it starts
        problem = metadata.problem(self.settings.library_dir)
        if problem:     # which pages to keep out isn't known: wait, don't burn attempts (review R2-C7)
            if problem != getattr(self, "_metadata_problem", None):
                log.error("%s; ingestion waits until it is fixed", problem)
            self._metadata_problem = problem
            return False
        self._metadata_problem = None
        job = self.next_job()
        if job is None:
            return False
        if job["attempts"] >= self.settings.max_attempts:
            self._fail(job, f"Gave up after {job['attempts']} attempts; the last run never finished")
            return True

        with self.conn:
            self.conn.execute(
                "UPDATE jobs SET status = 'running', attempts = attempts + 1, error = NULL, "
                "started_at = datetime('now') WHERE id = ?",
                (job["id"],),
            )
            if job["stage"] in RUNNING_STATUS:
                self.conn.execute(
                    "UPDATE documents SET status = ?, updated_at = datetime('now') WHERE id = ?",
                    (RUNNING_STATUS[job["stage"]], job["doc_id"]),
                )
        doc = self.conn.execute("SELECT * FROM documents WHERE id = ?", (job["doc_id"],)).fetchone()
        context = StageContext(
            self.conn, self.settings, doc, job["sha256"], job["id"], self.parser, self.embedder,
            self.formula_reader, self.compile_check, self.wait_for_chat, job["approved_usd"],
            self.describe_figures, self.summarizer_ask,
        )

        log.info("%s: %s (attempt %d)", doc["path"], job["stage"], job["attempts"] + 1)
        try:
            STAGE_FUNCTIONS[job["stage"]](context)
        except KeyboardInterrupt:
            # Stopped (Ctrl+C, the app shutting down), not crashed: back in the queue, attempt refunded.
            self.conn.rollback()
            with self.conn:
                self.conn.execute(
                    "UPDATE jobs SET status = 'pending', attempts = max(attempts - 1, 0) WHERE id = ?", (job["id"],),
                )
            log.info("%s: %s stopped; it will resume", doc["path"], job["stage"])
            raise
        except NeedsApproval as waiting:
            self.conn.rollback()
            self._wait_for_approval(job, waiting.detail)
            return True
        except Exception:
            self.conn.rollback()
            error = traceback.format_exc()
            if job["attempts"] + 1 >= self.settings.max_attempts:
                self._fail(job, error)
            else:
                with self.conn:
                    self.conn.execute(
                        "UPDATE jobs SET status = 'pending', error = ? WHERE id = ?",
                        (error, job["id"]),
                    )
            return True

        with self.conn:
            self.conn.execute(
                "UPDATE jobs SET status = 'done', finished_at = datetime('now') WHERE id = ?",
                (job["id"],),
            )
            if job["stage"] == SEARCHABLE_AFTER:
                self.conn.execute(
                    "UPDATE documents SET status = 'indexed', updated_at = datetime('now') WHERE id = ?",
                    (job["doc_id"],),
                )
        if job["stage"] == SEARCHABLE_AFTER:
            # A changed file: its highlights and notes find their passages again (app/anchors.py).
            try:
                from app.anchors import reanchor_document
                reanchor_document(self.conn, self.settings, job["doc_id"])
            except Exception:       # never fails the ingest; listing them tries again
                self.conn.rollback()
                log.exception("Re-anchoring document %s failed", job["doc_id"])
        return True

    def drain(self) -> int:
        ran = 0
        while self.run_one():
            ran += 1
        return ran


def reindex(conn: sqlite3.Connection, settings: Settings, from_stage: str = "parse") -> int:
    """Queue every document again from `from_stage`, keeping its id and its notes.

    From "formulas" the parse and the pages Granite already read are reused, so
    only the merge, the chunks and the vectors are repeated.
    """
    kept = STAGES[:STAGES.index(from_stage)]
    rows = conn.execute("SELECT id, sha256 FROM documents WHERE status != 'missing'").fetchall()
    with conn:
        for row in rows:
            conn.execute(
                "DELETE FROM jobs WHERE doc_id = ? AND (sha256 != ? OR stage NOT IN (%s))"
                % ",".join("?" * len(kept)),
                (row["id"], row["sha256"], *kept),
            )
            conn.execute("DELETE FROM chunks WHERE doc_id = ? AND kind IN ('text', 'figure', 'summary')", (row["id"],))
            conn.execute("DELETE FROM blocks WHERE doc_id = ?", (row["id"],))
            conn.execute("DELETE FROM figures WHERE doc_id = ?", (row["id"],))
            _create_jobs(conn, row["id"], row["sha256"])
            conn.execute(
                "UPDATE documents SET status = 'queued', updated_at = datetime('now') WHERE id = ?",
                (row["id"],),
            )
    if not kept:
        for row in rows:
            shutil.rmtree(settings.parsed_dir / row["sha256"], ignore_errors=True)
    return len(rows)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run(
    settings: Settings,
    watch: bool,
    stop: threading.Event | None = None,
    reindex_from: str | None = None,
) -> None:
    from app.ingest.watcher import LibraryWatcher

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    lock = open(settings.data_dir / "worker.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.warning("Another ingestion worker is already running")
        raise SystemExit(LOCKED) from None

    conn = connect(settings.db_path)
    worker = Worker(conn, settings)
    stop = stop or threading.Event()
    watcher = None
    try:
        recovered = worker.recover()
        if recovered:
            log.info("Requeued %d interrupted job(s)", recovered)
        added = add_late_stages(conn, settings)
        if added:
            log.info("Added %d describe or summarize job(s) for documents indexed before those stages", added)
        for doc_id, rows in link_summary_caches(conn, settings).items():
            log.info("Document %d: %s summary-cache rows marked as its own", doc_id,
                     rows if rows is not None else "no")
        if watch:
            # Start before the scan, so a file added during it isn't missed.
            settings.library_dir.mkdir(parents=True, exist_ok=True)
            watcher = LibraryWatcher(settings)
            watcher.start()
        if reindex_from:
            log.info("Requeued %d document(s) from %s", reindex(conn, settings, reindex_from), reindex_from)
        log.info("Library scan: %s", scan_library(conn, settings))

        while not stop.is_set():
            if watcher:
                for path in watcher.settled():
                    outcome = sync_file(conn, settings, path)
                    log.info("%s: %s", path.name, outcome)
            if worker.run_one():
                continue
            if not watch:
                break
            stop.wait(1.0)
    except KeyboardInterrupt:
        log.info("Stopping")
    finally:
        if watcher:
            watcher.stop()
        conn.close()
        lock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--once", action="store_true", help="drain the queue and exit")
    parser.add_argument("--reindex-from", choices=STAGES,
                        help="first queue every document again from this stage")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    signal.signal(signal.SIGTERM, _interrupt)     # stop like Ctrl+C: the current job goes back in the queue
    die_with_parent()
    run(load_settings(), watch=not args.once, reindex_from=args.reindex_from)


def _interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    main()
