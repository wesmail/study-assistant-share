"""The summary tree (ingestion stage 8): a summary of every section, chapter and document.

Search finds passages, and a broad question ("what does chapter 4 cover?", "map this book")
needs more than a few passages can hold. So each node of the document's outline gets a
summary, written bottom-up: a section from its own text; a chapter from its opening text
and its sections' summaries; the document from its chapters'. The summaries are stored as
chunks (kind 'summary') and searched with everything else; search.with_slots keeps them
from displacing text passages.

The tree is the PDF's outline, placed in the text as in app.ingest.outline; without one,
the first two levels of the parsed heading paths. Each text chunk belongs to the deepest
node in force where it starts. Contents pages are left out: a copy's contents can list
chapters it doesn't hold. Figure descriptions are left out too, since they are AI-written
already. The document node ends with a contents list made from the tree, not by the model.

A section too long for one call is first cut into parts, each turned into notes, and the
section summarized from those. Every model reply is cached by a hash of its whole input.

    python -m app.ingest.summarize build --doc 2 [--model gemma4:26b-a4b-it-qat]

The worker runs `build` for every new document (its summarize stage), with
summarizer_model from config.yaml.
    python -m app.ingest.summarize sample --doc 2 --node "4 Matrix Decompositions" --model M --out F
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.ingest.deck import pdf_path
from app.config import Settings
from app.ingest.chunker import _prefix, estimate_tokens
from app.ingest.metadata import shown_title
from app.ingest.outline import Entry, normalized, place, read_outline
from app.retrieval.context import _without_overlap

log = logging.getLogger(__name__)

PROMPT_VERSION = 3      # 2: keep the text's strength of claim, conditions and attributions;
                        # 3: never correct or reinterpret the text; a repeat stays a repeat (roadmap item 7)
SUMMARY_SEQ = 100_000_000       # summary chunks' seq: clear of text and figure chunks
PART_TOKENS = 7000              # of a node's own text per call
INPUT_TOKENS = 11000            # own text plus the children's summaries, before notes are needed
NUM_CTX = 16384
NUM_PREDICT = 1500
SKIPPED = {"contents", "tableofcontents", "briefcontents"}

SYSTEM = """You summarize parts of a book or paper for a study assistant's index. A reader asks broad questions ("what does chapter 4 cover?", "how do these chapters fit together?"), and your summary is what answers them.

Rules
- Use only the text you are given. Never add a fact, value, name, date, result or claim that isn't in it, even one you know to be true.
- Never add to, correct or reinterpret the text, even where it looks like a mistake: a line that repeats stays a repeat, and an unusual value, formula or reaction is given exactly as written.
- Keep the text's strength of claim. What it offers as an interpretation, a hypothesis or an association stays one ("the authors suggest", "is associated with"); never turn it into a cause or a fact.
- Attribute each result to what the text attributes it to, and give each number with the quantity and condition the text gives it.
- State conditions exactly as the text does: = or ≠, "if and only if", "necessary" or "sufficient", which case a statement covers.
- Be specific: name the definitions, results and methods, with the equation numbers, values and conditions the text states. Keep the text's notation and write math in LaTeX ($...$).
- Where the text says a formula couldn't be verified, describe it in words and don't write it out.
- If the text is cut off or seems incomplete, summarize what is there and don't guess the rest.
- Write plain paragraphs: no headings, no lists, no preamble, no closing remarks."""


@dataclass
class Node:
    title: str
    level: int                  # outline level; 0 for the document
    path: list[str]             # the heading path its summary chunk carries
    order: int | None
    children: list[Node] = field(default_factory=list)
    chunks: list[sqlite3.Row] = field(default_factory=list)    # its own text chunks
    summary: str | None = None
    chunk_id: int | None = None

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    def rows(self) -> list[sqlite3.Row]:
        return [row for node in self.walk() for row in node.chunks]

    def pages(self) -> tuple[int | None, int | None]:
        pages = [p for row in self.rows() for p in (row["page_start"], row["page_end"]) if p is not None]
        return (min(pages), max(pages)) if pages else (None, None)

    def name(self) -> str:
        return " › ".join(self.path) if self.path else self.title


def page_label(node: Node) -> str:
    start, end = node.pages()
    return f"page {start}" if start == end else f"pages {start}–{end}"


# --------------------------------------------------------------------------
# The tree
# --------------------------------------------------------------------------

def _text_rows(conn: sqlite3.Connection, doc_id: int) -> tuple[list[tuple], list[tuple[int, sqlite3.Row]]]:
    blocks = conn.execute('SELECT id, "order", page, kind, text FROM blocks WHERE doc_id = ?', (doc_id,)).fetchall()
    order_of = {row["id"]: row["order"] for row in blocks}
    rows = conn.execute("SELECT * FROM chunks WHERE doc_id = ? AND kind = 'text' ORDER BY seq", (doc_id,)).fetchall()
    chunks = []
    for row in rows:
        ids = json.loads(row["block_ids"] or "[]")
        path = json.loads(row["heading_path"] or "[]")
        if ids and not any(normalized(part) in SKIPPED for part in path):
            chunks.append((min(order_of[i] for i in ids), row))
    return [(r["order"], r["page"], r["kind"], r["text"]) for r in blocks], chunks


def _entries_from_paths(chunks: Sequence[tuple[int, sqlite3.Row]]) -> list[Entry]:
    """Without an outline: the first two levels of the parsed heading paths, where they change."""
    entries, previous = [], []
    for first, row in chunks:
        path = json.loads(row["heading_path"] or "[]")[:2]
        for depth in range(len(path)):
            if path[:depth + 1] != previous[:depth + 1]:
                entries.append(Entry(depth + 1, path[depth], row["page_start"] or 1, first))
        previous = path
    return entries


def _chunk_path(node: Node, fallback: list[str]) -> list[str]:
    """The node's path as its own chunks spell it ("Chapter 8 › Measuring performance"), else the outline's."""
    want = normalized(node.title)
    for row in node.rows()[:3]:
        path = json.loads(row["heading_path"] or "[]")
        for i, part in enumerate(path):
            if normalized(part) == want:
                return path[:i + 1]
    return fallback


def build_tree(conn: sqlite3.Connection, doc_id: int, pdf: Path | None, title: str) -> Node:
    blocks, chunks = _text_rows(conn, doc_id)
    entries = []
    if pdf is not None and pdf.suffix.lower() == ".pdf":
        try:
            entries = place(read_outline(pdf), blocks)
        except Exception as error:      # an unreadable outline: fall back to the parsed headings
            log.info("%s: no outline read (%s)", pdf.name, error)
    if not entries:
        entries = _entries_from_paths(chunks)
    root = Node(title, 0, [], None)
    stack, placed = [root], []
    for entry in entries:
        while stack[-1].level >= entry.level:
            stack.pop()
        node = Node(entry.title, entry.level, stack[-1].path + [entry.title], entry.order)
        stack[-1].children.append(node)
        stack.append(node)
        if entry.order is not None:
            placed.append(node)
    for first, row in chunks:
        owner = root
        for node in placed:             # in reading order: the last one started is the deepest in force
            if node.order > first:
                break
            owner = node
        owner.chunks.append(row)
    _prune(root)
    for node in root.walk():
        if node is not root:
            node.path = _chunk_path(node, node.path)
    return root


def _prune(node: Node) -> None:
    node.children = [child for child in node.children if child.rows()]
    for child in node.children:
        _prune(child)


# --------------------------------------------------------------------------
# What the model is given
# --------------------------------------------------------------------------

def _body(row: sqlite3.Row) -> str:
    from app.llm.prompts import text_for_model
    text = row["text"].removeprefix(_prefix(json.loads(row["heading_path"] or "[]")))
    meta = json.loads(row["meta"] or "{}")
    return text_for_model(text, meta.get("unverified_formulas", []))


def _joined(rows: Sequence[sqlite3.Row]) -> str:
    text = ""
    for row in rows:
        body = _body(row)
        text = body if not text else text + "\n\n" + _without_overlap(text, body)
    return text


def _parts(rows: Sequence[sqlite3.Row], budget: int | None = None) -> list[list[sqlite3.Row]]:
    budget = budget or PART_TOKENS
    parts, current, size = [], [], 0
    for row in rows:
        cost = estimate_tokens(row["text"])
        if current and size + cost > budget:
            parts.append(current)
            current, size = [], 0
        current.append(row)
        size += cost
    return parts + [current] if current else parts


def _children_text(node: Node) -> str:
    return "\n\n".join(f"### {child.name()} ({page_label(child)})\n{child.summary}" for child in node.children)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def prompts(node: Node, doc_title: str, own: str, notes: bool) -> str:
    where = f'"{node.name()}" ({page_label(node)}) of {doc_title}'
    opening = (f"NOTES ON ITS OWN TEXT:\n{own}" if notes else f"ITS OWN TEXT:\n{own}") if own else ""
    if not node.children:
        words = _clamp(estimate_tokens(own) // 15, 80, 250)
        source = "NOTES ON THE TEXT" if notes else "TEXT"
        return (f"Summarize this section, {where}. In at most {words} words, say what it covers and its main "
                f"results, in the order the text gives them.\n\n{source}:\n{own}")
    if node.level == 0:
        words = _clamp(150 + 40 * len(node.children), 300, 650)
        return (f"Summarize the whole of {doc_title} ({page_label(node)}) as it is in this copy. You get its "
                f"opening text, if any, and summaries of its parts. In at most {words} words: what it is about, "
                f"how its parts are organised and build on one another, and a sentence on each part. Describe "
                f"only the parts given here.\n\n{opening}\n\nPARTS:\n{_children_text(node)}")
    words = _clamp(100 + 40 * len(node.children), 200, 450)
    return (f"Summarize {where} as a whole. You get its own opening text, if any, and summaries of each of "
            f"its parts. In at most {words} words: what it is for, its main results, and how its parts build "
            f"on one another. Mention every part by name.\n\n{opening}\n\nPARTS:\n{_children_text(node)}")


def notes_prompt(node: Node, doc_title: str, part: str, number: int, total: int) -> str:
    return (f'This is part {number} of {total} of "{node.name()}" in {doc_title}. In at most 250 words, write '
            f"notes on what this part covers: its definitions, results and examples, with the values and "
            f"equation numbers it states.\n\nTEXT:\n{part}")


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

Ask = Callable[[str], str]


def ollama_ask(settings: Settings, model: str, think: bool | str = False, timeout: float = 900) -> Ask:
    def ask(prompt: str) -> str:
        response = httpx.post(f"{settings.ollama_url}/api/chat", timeout=timeout, json={
            "model": model, "stream": False, "think": think, "keep_alive": "30m",
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "options": {"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "temperature": 0.3},
        })
        response.raise_for_status()
        body = response.json()
        if body.get("done_reason") == "length":
            raise RuntimeError(f"{model} stopped at its {NUM_PREDICT}-token limit")
        text = (body.get("message") or {}).get("content", "").strip()
        if not text:
            raise RuntimeError(f"{model} returned nothing")
        return text
    return ask


class Preempted(Exception):
    """A chat request started while a summary was being written: the reply was dropped, to be asked again."""


def ollama_stream_ask(settings: Settings, model: str, think: bool | str = False, busy: Callable[[], bool] = lambda: False,
                      timeout: float = 900, transport: httpx.BaseTransport | None = None, check_every: float = 0.5) -> Ask:
    """As ollama_ask, but streamed, so a call can be dropped as soon as a chat request starts
    (`busy`): Ollama stops generating when the connection closes, and the chat doesn't wait
    for the rest of a reply that can take two minutes (p90 66 s, max 122 s in Phase 5b)."""
    def ask(prompt: str) -> str:
        payload = {"model": model, "stream": True, "think": think, "keep_alive": "30m",
                   "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                   "options": {"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "temperature": 0.3}}
        parts, reason, checked = [], None, time.monotonic()
        with httpx.Client(timeout=timeout, transport=transport) as client:
            with client.stream("POST", f"{settings.ollama_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if time.monotonic() - checked >= check_every:
                        checked = time.monotonic()
                        if busy():
                            raise Preempted
                    if not line.strip():
                        continue
                    body = json.loads(line)
                    if body.get("error"):
                        raise RuntimeError(f"{model}: {body['error']}")
                    parts.append((body.get("message") or {}).get("content") or "")
                    if body.get("done"):
                        reason = body.get("done_reason")
        if reason == "length":
            raise RuntimeError(f"{model} stopped at its {NUM_PREDICT}-token limit")
        text = "".join(parts).strip()
        if not text:
            raise RuntimeError(f"{model} returned nothing")
        return text
    return ask


def summarizer_name(model: str, think: bool | str) -> str:
    """Part of every cache key: the same model and setting must give the same name."""
    return f"ollama:{model}" + (f":think-{str(think).lower()}" if think else "")


def cache_key(summarizer: str, prompt: str) -> str:
    return hashlib.sha256(json.dumps([PROMPT_VERSION, summarizer, SYSTEM, prompt]).encode()).hexdigest()


def cached(conn: sqlite3.Connection, summarizer: str, ask: Ask, doc_id: int | None = None) -> Ask:
    """`ask` with every reply cached by its whole input. With `doc_id`, each row used is marked
    as that document's, so removing the document can remove its rows."""
    def lookup(prompt: str) -> str:
        key = cache_key(summarizer, prompt)
        row = conn.execute("SELECT text, doc_id FROM summary_cache WHERE input_sha = ?", (key,)).fetchone()
        if row:
            if doc_id is not None and row["doc_id"] is None:
                with conn:
                    conn.execute("UPDATE summary_cache SET doc_id = ? WHERE input_sha = ?", (doc_id, key))
            return row["text"]
        started = time.monotonic()
        text = ask(prompt)
        with conn:
            conn.execute("INSERT OR REPLACE INTO summary_cache (input_sha, summarizer, prompt_version, text, seconds, doc_id) "
                         "VALUES (?, ?, ?, ?, ?, ?)", (key, summarizer, PROMPT_VERSION, text, time.monotonic() - started, doc_id))
        return text
    return lookup


class NotCached(Exception):
    pass


def link_cache(conn: sqlite3.Connection, settings: Settings, doc_id: int, summarizer: str) -> int | None:
    """Mark the cache rows a document's stored tree came from, by building the tree again from the
    cache alone: no model is called. Returns how many rows it used, or None if one was missing
    (the tree was made with another title, model or text) and nothing was marked."""
    row = conn.execute("SELECT path, title, kind, sha256 FROM documents WHERE id = ?", (doc_id,)).fetchone()
    title = shown_title(row["title"]) or row["path"]
    keys: list[str] = []

    def from_cache(prompt: str) -> str:
        key = cache_key(summarizer, prompt)
        found = conn.execute("SELECT text FROM summary_cache WHERE input_sha = ?", (key,)).fetchone()
        if found is None:
            raise NotCached(key)
        keys.append(key)
        return found["text"]

    try:
        summarize(build_tree(conn, doc_id, pdf_path(settings, row), title), title, from_cache)
    except NotCached:
        return None
    with conn:
        conn.executemany("UPDATE summary_cache SET doc_id = ? WHERE input_sha = ? AND doc_id IS NULL",
                         [(doc_id, key) for key in keys])
    return len(set(keys))


def summarize(node: Node, doc_title: str, ask: Ask, report: Callable[[Node], None] = lambda node: None) -> None:
    """Every node's summary, children first."""
    for child in node.children:
        summarize(child, doc_title, ask, report)
    own = _joined(node.chunks)
    children = sum(estimate_tokens(child.summary or "") for child in node.children)
    notes = estimate_tokens(own) > PART_TOKENS or estimate_tokens(own) + children > INPUT_TOKENS
    if notes:
        parts = _parts(node.chunks)
        own = "\n\n".join(ask(notes_prompt(node, doc_title, _joined(part), i, len(parts)))
                          for i, part in enumerate(parts, 1))
    node.summary = ask(prompts(node, doc_title, own, notes))
    report(node)


def contents(root: Node) -> str:
    """The contents of this copy, from the tree: two levels, or three if two give fewer than five lines."""
    def lines(node: Node, depth: int, limit: int) -> list[str]:
        found = []
        for child in node.children:
            label = child.name() if depth == 1 else child.path[-1]
            found.append("  " * (depth - 1) + f"- {label} ({page_label(child)})")
            if depth < limit:
                found += lines(child, depth + 1, limit)
        return found
    listed = lines(root, 1, 2)
    if len(listed) < 5:
        listed = lines(root, 1, 3)
    return "Contents of this copy, from its outline and the text present:\n" + "\n".join(listed)


# --------------------------------------------------------------------------
# Storing
# --------------------------------------------------------------------------

def marker(node: Node, doc_title: str) -> str:
    what = "the whole document as it is in this copy" if node.level == 0 else f'"{node.name()}"'
    return f"[AI-written summary of {what}, {page_label(node)} of {doc_title}. Check details against the text passages.]"


def store(conn: sqlite3.Connection, settings: Settings, doc_id: int, root: Node, doc_title: str, summarizer: str,
          embed: Callable[[list[str]], list[list[float]]]) -> int:
    """Replace the document's summary chunks with the tree's, children before parents, and embed them."""
    from app.ingest.embed import serialize
    nodes = list(root.walk())[::-1]
    with conn:
        conn.execute("DELETE FROM chunks WHERE doc_id = ? AND kind = 'summary'", (doc_id,))
        for number, node in enumerate(nodes):
            body = node.summary + ("\n\n" + contents(root) if node is root else "")
            text = _prefix(node.path) + marker(node, doc_title) + "\n\n" + body
            start, end = node.pages()
            meta = {"node": node.title, "outline_level": node.level, "summarizer": summarizer,
                    "prompt_version": PROMPT_VERSION, "children": [child.chunk_id for child in node.children],
                    "source_chunks": len(node.rows())}
            cursor = conn.execute(
                "INSERT INTO chunks (doc_id, kind, level, seq, heading_path, text, token_count, block_ids, "
                "page_start, page_end, meta) VALUES (?, 'summary', ?, ?, ?, ?, ?, '[]', ?, ?, ?)",
                (doc_id, 2 if node is root else 1, SUMMARY_SEQ + number, json.dumps(node.path, ensure_ascii=False),
                 text, estimate_tokens(text), start, end, json.dumps(meta, ensure_ascii=False)))
            node.chunk_id = cursor.lastrowid
        rows = conn.execute("SELECT id, text FROM chunks WHERE doc_id = ? AND kind = 'summary' ORDER BY id",
                            (doc_id,)).fetchall()
    for start in range(0, len(rows), settings.embed_batch):
        batch = rows[start:start + settings.embed_batch]
        vectors = embed([row["text"] for row in batch])
        with conn:
            conn.executemany("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)",
                             [(row["id"], serialize(v, settings.embed_dimensions)) for row, v in zip(batch, vectors)])
    return len(nodes)


def _document(conn: sqlite3.Connection, settings: Settings, doc_id: int) -> tuple[Path, str]:
    row = conn.execute("SELECT path, title, kind, sha256 FROM documents WHERE id = ?", (doc_id,)).fetchone()
    return pdf_path(settings, row), shown_title(row["title"]) or row["path"]


def main() -> None:
    from app.config import load_settings
    from app.db.connection import connect
    from app.ingest.embed import embed_texts

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "sample", "tree"):
        command = commands.add_parser(name)
        command.add_argument("--doc", type=int, required=True)
        command.add_argument("--model", default=None, help="default: summarizer_model in config.yaml")
        command.add_argument("--think", default=None, help="false, true, low, medium or high; default: summarizer_think")
        if name == "sample":
            command.add_argument("--node", action="append", required=True, help="an outline title; repeatable")
            command.add_argument("--out", required=True, help="JSON lines: node, pages, source text, summary")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = load_settings()
    conn = connect(settings.db_path)
    pdf, title = _document(conn, settings, args.doc)
    root = build_tree(conn, args.doc, pdf, title)
    if args.command == "tree":
        for node in root.walk():
            own = sum(estimate_tokens(row["text"]) for row in node.chunks)
            print("  " * node.level + f"{node.name()} ({page_label(node)}): {len(node.chunks)} chunks, {own} tokens")
        return
    args.model = args.model or settings.summarizer_model
    think = settings.summarizer_think if args.think is None else {"false": False, "true": True}.get(args.think, args.think)
    summarizer = summarizer_name(args.model, think)
    ask = cached(conn, summarizer, ollama_ask(settings, args.model, think), args.doc)
    started = time.monotonic()
    report = lambda node: log.info("%5.0fs  %s", time.monotonic() - started, node.name())  # noqa: E731
    if args.command == "build":
        summarize(root, title, ask, report)
        count = store(conn, settings, args.doc, root, title, summarizer, lambda texts: embed_texts(texts, settings))
        log.info("%s: %d summaries in %.0f s", title, count, time.monotonic() - started)
        return
    wanted = {normalized(node) for node in args.node}
    with open(args.out, "a", encoding="utf-8") as out:
        for node in root.walk():
            if normalized(node.title) in wanted:
                summarize(node, title, ask, report)
                for part in node.walk():
                    out.write(json.dumps({"doc": args.doc, "node": part.name(), "pages": page_label(part),
                                          "model": summarizer, "children": [c.name() for c in part.children],
                                          "source": _joined(part.rows()), "summary": part.summary},
                                         ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
