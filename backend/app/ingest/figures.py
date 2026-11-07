"""Figure descriptions: written once per image, on demand, and used for search.

    python -m app.ingest.figures status
    python -m app.ingest.figures describe [--doc ID ...] [--limit N] [--yes]

The describer is `figure_describer` in config.yaml, chosen blind on ten held-out figures.
A description is cached by the image's hash, with the describer and
prompt version, so re-ingesting a book costs nothing. It joins the figure's chunk below a
label saying it was written by a model and that its values are approximate; a figure the
parser found no caption for gets a chunk of its own, outside the text's reading order, so
it never becomes a neighbour of a text passage.

Nothing here runs by itself. A hosted describer shows how many figures it would describe
and what that should cost, and asks first; so a newly added book is described only when
the reader says so. While a chat request runs the job waits, so the two never compete
for the GPU or the reader's attention.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import sqlite3
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import httpx

from app.config import Settings, load_settings
from app.db.connection import connect
from app.ingest.chunker import estimate_tokens
from app.ingest.embed import embed_texts, serialize
from app.llm import hosted

log = logging.getLogger("figures")
PURPOSE = "figures"
PROMPT_VERSION = 1
MARKER = "[AI-written description of this figure. Its values were read by eye and are approximate.]"
STANDALONE_SEQ = -1000      # times the figure's id: far from every other chunk, so no neighbours
LOCAL_LIMITS = {"num_ctx": 8192, "num_predict": 1024, "temperature": 0.3}

PROMPT = """You describe one figure from a textbook or a paper, so that someone who can't see it can answer questions about it. You get the image, its caption and text from the same page.

- Say what kind of figure it is and what each panel shows: axes with their labels, units and ranges, legend entries, curves, points, arrows and annotations.
- Describe the main features: trends, peaks, minima, crossings, where curves differ, what is marked.
- Give a number only if you can read it from the image: a printed value, an axis tick, or a position you can place between ticks, written as "about" a value. If you can't read a value, say it isn't legible. Never estimate a value you can't place, and never take one from your own knowledge.
- Use the caption and the page text for names and context, but describe what the image shows, and say so if the image seems to disagree with them.
- Write formulas in LaTeX between $ signs. Plain prose, no headings, 120 to 300 words."""


# --- Which figures, and what the describer gets ------------------------------------------

def image_path(settings: Settings, row) -> Path:
    return settings.parsed_dir / row["sha256"] / row["image_path"]


def image_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def figure_rows(conn: sqlite3.Connection, doc_ids: Sequence[int] | None = None) -> list[sqlite3.Row]:
    where = f"WHERE f.doc_id IN ({','.join('?' * len(doc_ids))})" if doc_ids else ""
    return conn.execute(
        "SELECT f.id, f.doc_id, f.page, f.image_path, f.caption, f.image_sha, d.sha256, d.title "
        f"FROM figures f JOIN documents d ON d.id = f.doc_id {where} ORDER BY f.doc_id, f.page, f.id",
        list(doc_ids or ())).fetchall()


def fill_image_shas(conn: sqlite3.Connection, settings: Settings, doc_ids: Sequence[int] | None = None) -> int:
    """The hash of each figure's image, which the cache is keyed by. Missing files are left out."""
    filled = 0
    with conn:
        for row in figure_rows(conn, doc_ids):
            if row["image_sha"] or not row["image_path"]:
                continue
            path = image_path(settings, row)
            if path.is_file():
                conn.execute("UPDATE figures SET image_sha = ? WHERE id = ?", (image_sha(path), row["id"]))
                filled += 1
    return filled


def page_text(conn: sqlite3.Connection, row, limit: int) -> str:
    paragraphs = [text for (text,) in conn.execute(
        'SELECT text FROM blocks WHERE doc_id = ? AND page = ? AND kind IN ("paragraph", "heading", "list_item") '
        'ORDER BY "order"', (row["doc_id"], row["page"]))]
    return "\n\n".join(paragraphs)[:limit]


def user_text(caption: str, text: str) -> str:
    return f"Caption: {caption or '(none)'}\n\nText from the same page:\n{text or '(none)'}\n\nDescribe the figure."


# --- Describing ----------------------------------------------------------------------------

async def describe(image: Path, caption: str, text: str, settings: Settings,
                   transport: httpx.AsyncBaseTransport | None = None, client=None) -> tuple[str, float]:
    """One description and what it cost. Raises on a failure or an empty reply."""
    describer = settings.figure_describer
    data = base64.b64encode(image.read_bytes()).decode()
    if describer.provider == "anthropic":
        messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
            {"type": "text", "text": user_text(caption, text)}]}]
        turn = replace(settings, hosted_model=describer.model, hosted_effort=describer.effort or settings.hosted_effort,
                       hosted_max_tokens=settings.figure_max_tokens, turn_id=None)
        written, cost, reason = "", 0.0, None
        async for delta in hosted.stream_chat(messages, turn, client, purpose=PURPOSE):
            written += delta.text
            if delta.done:
                cost, reason = delta.cost_usd, delta.done_reason
    else:
        payload = {"model": describer.model, "stream": False, "keep_alive": "30m", "options": LOCAL_LIMITS,
                   "messages": [{"role": "system", "content": PROMPT},
                                {"role": "user", "content": user_text(caption, text), "images": [data]}]}
        async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10), transport=transport) as local:
            response = await local.post(f"{settings.ollama_url}/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()
        written, cost, reason = (body.get("message") or {}).get("content") or "", 0.0, body.get("done_reason")
    written = written.strip()
    if not written:
        raise RuntimeError(f"{describer.model} wrote no description ({reason or 'stream ended early'})")
    return written, cost


def cached(conn: sqlite3.Connection, sha: str, settings: Settings) -> str | None:
    row = conn.execute("SELECT text FROM figure_descriptions WHERE image_sha = ? AND describer = ? AND prompt_version = ?",
                       (sha, settings.figure_describer.name, PROMPT_VERSION)).fetchone()
    return row[0] if row else None


def store(conn: sqlite3.Connection, sha: str, settings: Settings, text: str, cost: float) -> None:
    with conn:
        conn.execute("INSERT OR REPLACE INTO figure_descriptions (image_sha, describer, prompt_version, text, cost_usd) "
                     "VALUES (?, ?, ?, ?, ?)", (sha, settings.figure_describer.name, PROMPT_VERSION, text, cost))


def pending(conn: sqlite3.Connection, settings: Settings, doc_ids: Sequence[int] | None = None) -> list[sqlite3.Row]:
    """Figures whose image has no description yet from this describer and prompt. One per image."""
    seen, waiting = set(), []
    for row in figure_rows(conn, doc_ids):
        sha = row["image_sha"]
        if sha and sha not in seen and cached(conn, sha, settings) is None:
            seen.add(sha)
            waiting.append(row)
    return waiting


def estimate_usd(settings: Settings, count: int) -> float | None:
    describer = settings.figure_describer
    if not describer.hosted or not describer.expected_tokens:
        return None if describer.hosted else 0.0
    price = settings.hosted_prices[describer.model]
    tokens_in, tokens_out = describer.expected_tokens
    return count * (tokens_in * price.input + tokens_out * price.output) / hosted.MILLION


# --- Into the search index ---------------------------------------------------------------

def without_description(text: str) -> str:
    return text.split(f"\n\n{MARKER}", 1)[0]


def with_description(text: str, description: str) -> str:
    return f"{without_description(text)}\n\n{MARKER}\n{description}"


def figure_chunk(conn: sqlite3.Connection, row) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM chunks WHERE doc_id = ? AND kind = 'figure' AND json_extract(meta, '$.image_path') = ?",
                        (row["doc_id"], row["image_path"])).fetchone()


def nearest_heading(conn: sqlite3.Connection, row) -> list:
    found = conn.execute("SELECT heading_path FROM chunks WHERE doc_id = ? AND kind = 'text' AND page_start <= ? "
                         "ORDER BY page_start DESC, seq DESC LIMIT 1", (row["doc_id"], row["page"])).fetchone()
    return json.loads(found[0]) if found and found[0] else []


def apply_descriptions(conn: sqlite3.Connection, settings: Settings, doc_ids: Sequence[int] | None = None) -> int:
    """Put every cached description into its figure's chunk; returns how many chunks changed.

    An update drops the chunk's vector (a trigger), so `embed_missing` must follow.
    """
    changed = 0
    with conn:
        for row in figure_rows(conn, doc_ids):
            description = row["image_sha"] and cached(conn, row["image_sha"], settings)
            if not description:
                continue
            chunk = figure_chunk(conn, row)
            if chunk is not None:
                text = with_description(chunk["text"], description)
                if text != chunk["text"]:
                    conn.execute("UPDATE chunks SET text = ?, token_count = ? WHERE id = ?",
                                 (text, estimate_tokens(text), chunk["id"]))
                    changed += 1
                continue
            heading = nearest_heading(conn, row)
            prefix = (" › ".join(heading) + " ›\n\n") if heading else ""
            text = with_description(f"{prefix}Figure on page {row['page']} (no caption found)", description)
            block = conn.execute("SELECT id FROM blocks WHERE doc_id = ? AND page = ? AND kind = 'figure' AND text = ? "
                                 "ORDER BY \"order\" LIMIT 1", (row["doc_id"], row["page"], row["caption"] or "")).fetchone()
            conn.execute("INSERT INTO chunks (doc_id, kind, seq, heading_path, text, token_count, block_ids, page_start, "
                         "page_end, meta) VALUES (?, 'figure', ?, ?, ?, ?, ?, ?, ?, ?)",
                         (row["doc_id"], STANDALONE_SEQ * row["id"], json.dumps(heading, ensure_ascii=False), text,
                          estimate_tokens(text), json.dumps([block[0]] if block else []), row["page"], row["page"],
                          json.dumps({"image_path": row["image_path"], "standalone": True})))
            changed += 1
    return changed


def embed_missing(conn: sqlite3.Connection, settings: Settings, embed: Callable = embed_texts) -> int:
    rows = conn.execute("SELECT c.id, c.text FROM chunks c LEFT JOIN chunk_vec v ON v.chunk_id = c.id "
                        "WHERE c.kind IN ('text', 'figure') AND v.chunk_id IS NULL ORDER BY c.id").fetchall()
    for start in range(0, len(rows), settings.embed_batch):
        batch = rows[start:start + settings.embed_batch]
        vectors = embed([row["text"] for row in batch], settings)
        with conn:
            conn.executemany("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)",
                             [(row["id"], serialize(vector, settings.embed_dimensions)) for row, vector in zip(batch, vectors)])
    return len(rows)


# --- Waiting for chat ----------------------------------------------------------------------

def busy_dir(settings: Settings) -> Path:
    return settings.data_dir / "chat-busy"


@contextmanager
def chat_running(settings: Settings) -> Iterator[None]:
    """Marks a chat request as running, for the description job to wait on."""
    marker = busy_dir(settings) / uuid.uuid4().hex
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    try:
        yield
    finally:
        marker.unlink(missing_ok=True)


def chat_busy(settings: Settings, now: float | None = None) -> bool:
    """A marker counts while it's younger than the longest a chat could take; older ones are left over."""
    folder = busy_dir(settings)
    if not folder.is_dir():
        return False
    now = time.time() if now is None else now
    for marker in folder.iterdir():
        try:
            if now - marker.stat().st_mtime < settings.figure_busy_seconds:
                return True
        except FileNotFoundError:       # its chat ended between the listing and the stat (review C3)
            continue
    return False


async def wait_for_chat(settings: Settings, pause: Callable = asyncio.sleep, report: Callable = print) -> None:
    if not chat_busy(settings):
        return
    report("Paused: a chat request is running.")
    while chat_busy(settings):
        await pause(2)
    report("Going on.")


# --- The job -------------------------------------------------------------------------------

async def describe_pending(conn: sqlite3.Connection, settings: Settings, rows: Sequence, embed: Callable = embed_texts,
                           report: Callable = print, pause: Callable = asyncio.sleep, transport=None,
                           client=None) -> dict:
    """Describe each figure, caching as it goes, so stopping loses at most the one in progress."""
    described = failed = 0
    spent = 0.0
    for number, row in enumerate(rows, 1):
        await wait_for_chat(settings, pause, report)
        path = image_path(settings, row)
        try:
            text, cost = await describe(path, row["caption"] or "", page_text(conn, row, settings.figure_page_text_chars),
                                        settings, transport, client)
        except (httpx.HTTPError, RuntimeError, OSError) as error:
            failed += 1
            report(f"{number}/{len(rows)} figure {row['id']} (page {row['page']}): FAILED, will be tried next time: {error}")
            continue
        store(conn, row["image_sha"], settings, text, cost)
        described += 1
        spent += cost
        report(f"{number}/{len(rows)} figure {row['id']} (page {row['page']}): {len(text.split())} words, ${cost:.4f}")
    changed = apply_descriptions(conn, settings, sorted({row["doc_id"] for row in rows}) or None)
    embedded = embed_missing(conn, settings, embed)
    return {"described": described, "failed": failed, "spent_usd": spent, "chunks_changed": changed, "embedded": embedded}


def status_lines(conn: sqlite3.Connection, settings: Settings) -> list[str]:
    describer = settings.figure_describer
    lines = [f"Describer: {describer.name} ({'hosted' if describer.hosted else 'local'})"]
    for doc_id, title in conn.execute("SELECT id, title FROM documents ORDER BY id"):
        rows = figure_rows(conn, [doc_id])
        waiting = pending(conn, settings, [doc_id])
        estimate = estimate_usd(settings, len(waiting))
        cost = f", about ${estimate:.2f}" if estimate else ""
        lines.append(f"  {doc_id} {title[:50]}: {len(rows)} figures, {len(waiting)} to describe{cost}")
    return lines


def confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


async def cmd_describe(args) -> None:
    settings = load_settings()
    conn = connect(settings.db_path)
    fill_image_shas(conn, settings, args.doc)
    rows = pending(conn, settings, args.doc)
    if args.limit is not None:
        rows = rows[:args.limit]
    describer = settings.figure_describer
    if not rows:
        print(f"Nothing to describe with {describer.name}.")
        applied = apply_descriptions(conn, settings, args.doc)
        embedded = embed_missing(conn, settings)
        print(f"Chunks brought up to date: {applied}, embedded: {embedded}")
        return
    estimate = estimate_usd(settings, len(rows))
    if describer.hosted:
        question = (f"Describe {len(rows)} figures with {describer.model}, sending each image, its caption and its "
                    f"page's text to Anthropic, for about ${estimate:.2f}?")
        if not args.yes and not confirm(question):
            print("Nothing described.")
            return
    print(f"Describing {len(rows)} figures with {describer.name}.", flush=True)
    result = await describe_pending(conn, settings, rows, report=lambda line: print(line, flush=True))
    print(f"Described {result['described']}, failed {result['failed']}, spent ${result['spent_usd']:.2f}; "
          f"chunks changed {result['chunks_changed']}, embedded {result['embedded']}.")


def cmd_status(args) -> None:
    settings = load_settings()
    conn = connect(settings.db_path)
    fill_image_shas(conn, settings)
    print("\n".join(status_lines(conn, settings)))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status").set_defaults(run=cmd_status)
    describe_cmd = commands.add_parser("describe")
    describe_cmd.add_argument("--doc", type=int, nargs="*", help="only these documents")
    describe_cmd.add_argument("--limit", type=int, help="at most this many figures")
    describe_cmd.add_argument("--yes", action="store_true", help="don't ask, even for a hosted describer")
    describe_cmd.set_defaults(run=lambda a: asyncio.run(cmd_describe(a)))
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    sys.exit(main())
