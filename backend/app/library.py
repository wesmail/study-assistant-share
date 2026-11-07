"""The library pane's view of each document, and what the reader can do from it.

    state       queued, parsing, indexing, describing, summarizing, done, failed, or removed
                (the file has left library/), from the document's jobs
    waiting     a stage that waits for the reader's OK, with what it would cost
    needs_details  no entry in metadata.yaml yet: the pane asks for one

The server and the worker share only the database: an OK here clears the job's `waiting`
mark, and the worker picks the job up on its next pass.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path

import yaml

from app.errors import Refused  # noqa: F401  (re-exported: library.Refused)
from app.config import Settings
from app.ingest import metadata
from app.ingest.chunker import estimate_tokens
from app.ingest.embed import serialize
from app.ingest.worker import LATE_STAGES, STAGES

STAGE_STATE = {"parse": "parsing", "formulas": "parsing", "chunk": "indexing", "index": "indexing",
               "describe": "describing", "summarize": "summarizing"}
EARLY_STAGES = tuple(stage for stage in STAGES if stage not in LATE_STAGES)


def _last_line(error: str | None) -> str | None:
    lines = [line for line in (error or "").splitlines() if line.strip()]
    return lines[-1][:300] if lines else None


def document_state(doc: sqlite3.Row | dict, jobs: list) -> dict:
    """What the pane shows for one document, from its current file version's jobs."""
    jobs = [job for job in jobs if job["sha256"] == doc["sha256"]]
    by_stage = {job["stage"]: job for job in jobs}
    waiting = None
    describe = by_stage.get("describe")
    if describe is not None and describe["status"] == "pending" and describe["waiting"] == "approval":
        waiting = {"stage": "describe", **json.loads(describe["detail"] or "{}")}
    failed = [job for job in jobs if job["status"] == "failed"]
    running = next((job for job in jobs if job["status"] == "running"), None)
    progress = None
    if running is not None and running["total"]:
        progress = {"done": running["progress"], "total": running["total"]}

    if doc["status"] == "missing":
        state = "removed"
    elif any(job["stage"] in EARLY_STAGES for job in failed):
        state = "failed"
    elif running is not None:
        state = STAGE_STATE[running["stage"]]
    else:
        pending = [job for job in sorted(jobs, key=lambda job: job["stage_order"])
                   if job["status"] == "pending" and job["waiting"] is None]
        if not jobs or (pending and not any(job["status"] == "done" for job in jobs)):
            state = "queued"
        elif pending:
            state = STAGE_STATE[pending[0]["stage"]]
        elif failed:
            state = "failed"            # a late stage: the document is still searchable
        else:
            state = "done"
    return {
        "state": state,
        "searchable": doc["status"] == "indexed",
        "waiting": waiting,
        "progress": progress,
        "error": next((_last_line(job["error"]) for job in failed), None),
        "failed_stage": failed[0]["stage"] if failed else None,
    }


# --- The reader's library settings ---------------------------------------------------------

LIBRARY_SETTINGS = {"hide_test_collection": False}
TEST_COLLECTION = "test"


def library_settings(conn: sqlite3.Connection) -> dict:
    stored = {row["key"]: json.loads(row["value"]) for row in conn.execute("SELECT key, value FROM app_settings")}
    return {key: stored.get(key, default) for key, default in LIBRARY_SETTINGS.items()}


def set_library_settings(conn: sqlite3.Connection, values: dict) -> dict:
    with conn:
        for key, value in values.items():
            if key not in LIBRARY_SETTINGS or not isinstance(value, type(LIBRARY_SETTINGS[key])):
                raise Refused(f"No library setting {key!r} of that kind")
            conn.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)", (key, json.dumps(value)))
    return library_settings(conn)


def hidden_collections(conn: sqlite3.Connection) -> tuple[str, ...]:
    """What whole-library search leaves out, from the reader's setting."""
    return (TEST_COLLECTION,) if library_settings(conn)["hide_test_collection"] else ()


def library_rows(conn: sqlite3.Connection, settings: Settings) -> list[dict]:
    try:
        overrides = metadata.read_overrides(settings.library_dir)
    except metadata.MetadataError:      # the listing still shows; /api/documents says what is wrong
        overrides = {}
    docs = conn.execute(
        "SELECT d.id, d.path, d.sha256, d.title, d.author, d.year, d.publisher, d.kind, d.pages, d.status, d.collection, "
        "(SELECT count(*) FROM chunks c WHERE c.doc_id = d.id) AS chunks FROM documents d ORDER BY d.title"
    ).fetchall()
    jobs: dict[int, list] = {}
    for job in conn.execute("SELECT * FROM jobs ORDER BY stage_order"):
        jobs.setdefault(job["doc_id"], []).append(job)
    rows = []
    for doc in docs:
        shown = document_state(doc, jobs.get(doc["id"], []))
        rows.append({
            "id": doc["id"], "path": doc["path"], "title": metadata.shown_title(doc["title"]),
            "author": doc["author"], "year": doc["year"], "publisher": doc["publisher"], "kind": doc["kind"],
            "pages": doc["pages"], "status": doc["status"], "chunks": doc["chunks"], "collection": doc["collection"],
            **shown,
            "needs_details": doc["path"] not in overrides and doc["status"] != "missing"
                             and shown["state"] in ("done", "describing", "summarizing"),
        })
    return rows


# --- Figure descriptions: the reader's OK -------------------------------------------------


def _waiting_describe(conn: sqlite3.Connection, doc_id: int) -> sqlite3.Row:
    job = conn.execute(
        "SELECT j.* FROM jobs j JOIN documents d ON d.id = j.doc_id AND d.sha256 = j.sha256 "
        "WHERE j.doc_id = ? AND j.stage = 'describe'", (doc_id,)).fetchone()
    if job is None or job["status"] != "pending" or job["waiting"] != "approval":
        raise Refused("This document's figures aren't waiting for an OK")
    return job


def approve_figures(conn: sqlite3.Connection, doc_id: int, usd: float) -> None:
    """The reader agreed to `usd`, the estimate they were shown, which must be today's."""
    job = _waiting_describe(conn, doc_id)
    estimate = json.loads(job["detail"] or "{}").get("usd") or 0
    if abs(usd - estimate) > 0.005:
        raise Refused(f"The estimate is now ${estimate:.2f}; look again before agreeing")
    with conn:
        conn.execute("UPDATE jobs SET waiting = NULL, approved_usd = ? WHERE id = ?", (usd, job["id"]))


def skip_figures(conn: sqlite3.Connection, doc_id: int) -> None:
    """No descriptions for this version of the file: figures are found by their captions only."""
    job = _waiting_describe(conn, doc_id)
    detail = {**json.loads(job["detail"] or "{}"), "skipped": True}
    with conn:
        conn.execute("UPDATE jobs SET status = 'done', waiting = NULL, detail = ?, finished_at = datetime('now') "
                     "WHERE id = ?", (json.dumps(detail), job["id"]))


# --- metadata.yaml ---------------------------------------------------------------------------

def suggestion(settings: Settings, doc: sqlite3.Row) -> dict:
    """What the form starts with: the entry if there is one, else the PDF's own metadata and the parsed title."""
    entry = metadata.read_overrides(settings.library_dir).get(doc["path"])
    if entry:
        return {field: entry.get(field) for field in metadata.FIELDS}
    if doc["kind"] == "pptx":
        found = metadata.from_deck(settings, doc["path"], doc["sha256"])
        return {"title": found.get("title"), "author": found.get("author"), "year": None, "publisher": None}
    found = metadata.from_pdf(settings.library_dir / doc["path"])
    return {"title": found.get("title") or metadata.parsed_title(settings, doc["sha256"]),
            "author": found.get("author"), "year": None, "publisher": None}


def _entry_text(path: str, values: dict) -> str:
    """The entry's lines: the form's fields, then any other keys it had (such as collection), kept."""
    lines = [f"{json.dumps(path, ensure_ascii=False)}:"]
    for field, value in values.items():
        if value in (None, ""):
            continue
        if isinstance(value, list):     # such as exclude_pages: [1]; JSON is YAML's flow style
            lines.append(f"  {field}: {json.dumps(value, ensure_ascii=False)}")
            continue
        lines.append(f"  {field}: {value if isinstance(value, int) else json.dumps(str(value), ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def _clean(values: dict) -> dict:
    cleaned = {}
    for field in metadata.FIELDS:
        value = values.get(field)
        value = value.strip() if isinstance(value, str) else value
        if value in (None, ""):
            continue
        if field == "year":
            try:
                value = int(value)
            except (TypeError, ValueError):
                raise Refused("The year must be a number") from None
            if not 1000 <= value <= 2100:
                raise Refused("The year must be between 1000 and 2100")
        cleaned[field] = value
    if not cleaned.get("title"):
        raise Refused("A title is needed")
    return cleaned


def _replace_block(text: str, path: str, entry: str) -> str | None:
    """`text` with the top-level block for `path` replaced by `entry`, or None if there's no such block."""
    lines = text.splitlines(keepends=True)
    for start, line in enumerate(lines):
        if line[:1].isspace() or ":" not in line or line.lstrip().startswith("#"):
            continue
        try:
            key = yaml.safe_load(line.rstrip("\n") + " null")
        except yaml.YAMLError:
            continue
        if not (isinstance(key, dict) and path in key):
            continue
        end = start + 1
        while end < len(lines) and (lines[end][:1].isspace() or not lines[end].strip()):
            end += 1
        while end > start + 1 and not lines[end - 1].strip():      # keep the blank line before the next entry
            end -= 1
        return "".join(lines[:start]) + entry + "".join(lines[end:])
    return None


def save_details(conn: sqlite3.Connection, settings: Settings, doc: sqlite3.Row, values: dict,
                 embed: Callable[[list[str], Settings], list[list[float]]]) -> dict:
    """Write the document's entry into metadata.yaml, keeping every other line and comment, then
    reread it. Summaries name the document, so their labels follow a new title."""
    cleaned = _clean(values)
    file = settings.library_dir / metadata.METADATA_FILE
    text = file.read_text(encoding="utf-8") if file.is_file() else ""
    kept = {key: value for key, value in metadata.read_overrides(settings.library_dir).get(doc["path"], {}).items()
            if key not in metadata.FIELDS and value not in (None, "")}
    entry = _entry_text(doc["path"], {**cleaned, **kept})
    updated = _replace_block(text, doc["path"], entry)
    if updated is None:
        updated = text + ("" if not text or text.endswith("\n\n") else "\n" if text.endswith("\n") else "\n\n") + entry
    parsed = yaml.safe_load(updated) or {}
    if (parsed.get(doc["path"]) or {}) != {**cleaned, **kept}:
        raise Refused("metadata.yaml couldn't be updated safely; edit it by hand")
    temporary = file.with_name(file.name + ".tmp")
    temporary.write_text(updated, encoding="utf-8")
    os.replace(temporary, file)

    old_title = metadata.shown_title(doc["title"])
    metadata.refresh(conn, settings)
    new = conn.execute("SELECT title FROM documents WHERE id = ?", (doc["id"],)).fetchone()
    retitle_summaries(conn, settings, doc["id"], old_title, metadata.shown_title(new["title"]), embed)
    return cleaned


def retitle_summaries(conn: sqlite3.Connection, settings: Settings, doc_id: int, old: str, new: str,
                      embed: Callable[[list[str], Settings], list[list[float]]]) -> int:
    """Put the new title in the summaries' labels ("... of <title>. Check details ...").

    Changed text loses its vector (a trigger), and the index stage embeds only text and
    figure chunks, so the summaries are embedded again here.
    """
    if old == new:
        return 0
    rows = conn.execute("SELECT id, text FROM chunks WHERE doc_id = ? AND kind = 'summary'", (doc_id,)).fetchall()
    changed = [(row["id"], row["text"].replace(f" of {old}. Check details", f" of {new}. Check details", 1))
               for row in rows]
    changed = [(chunk_id, text) for (chunk_id, text), row in zip(changed, rows) if text != row["text"]]
    vectors = []
    for start in range(0, len(changed), settings.embed_batch):      # before writing, so a failure changes nothing
        vectors += embed([text for _, text in changed[start:start + settings.embed_batch]], settings)
    with conn:
        for (chunk_id, text), vector in zip(changed, vectors):
            conn.execute("UPDATE chunks SET text = ?, token_count = ? WHERE id = ?", (text, estimate_tokens(text), chunk_id))
            conn.execute("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)",
                         (chunk_id, serialize(vector, settings.embed_dimensions)))
    return len(changed)


# --- Removing a document whose file has gone --------------------------------------------------

def _versions(conn: sqlite3.Connection, doc: sqlite3.Row) -> set[str]:
    """Every file version the document had: each has its own folder under data/parsed/."""
    return {doc["sha256"]} | {row[0] for row in conn.execute("SELECT DISTINCT sha256 FROM jobs WHERE doc_id = ?", (doc["id"],))}


def _folders(conn: sqlite3.Connection, settings: Settings, doc: sqlite3.Row) -> list[Path]:
    taken = {row[0] for row in conn.execute("SELECT sha256 FROM documents WHERE id != ?", (doc["id"],))}
    names = [sha for sha in _versions(conn, doc) if sha not in taken]
    candidates = [settings.parsed_dir / name for name in names] + [settings.parsed_dir / f"{name}.tmp" for name in names]
    return sorted(path for path in candidates if path.is_dir() and not path.is_symlink())


def _own_descriptions(conn: sqlite3.Connection, doc_id: int) -> list[sqlite3.Row]:
    """Cached descriptions of this document's images that no other document's figures use."""
    return conn.execute(
        "SELECT * FROM figure_descriptions WHERE image_sha IN (SELECT image_sha FROM figures WHERE doc_id = ?) "
        "AND image_sha NOT IN (SELECT image_sha FROM figures WHERE doc_id != ? AND image_sha IS NOT NULL)",
        (doc_id, doc_id)).fetchall()


def _size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file() and not file.is_symlink())


def _check_removable(conn: sqlite3.Connection, settings: Settings, doc: sqlite3.Row) -> None:
    if doc["status"] != "missing" or (settings.library_dir / doc["path"]).exists():
        raise Refused("Only a document whose file has left library/ can be removed here")
    if conn.execute("SELECT 1 FROM jobs WHERE doc_id = ? AND status = 'running'", (doc["id"],)).fetchone():
        raise Refused("A job for this document is still running; try again when it stops")


def removal_preview(conn: sqlite3.Connection, settings: Settings, doc: sqlite3.Row) -> dict:
    """Everything removing the document would delete, for the reader to confirm."""
    _check_removable(conn, settings, doc)
    doc_id = doc["id"]
    chunks = dict(conn.execute("SELECT kind, count(*) FROM chunks WHERE doc_id = ? GROUP BY kind", (doc_id,)).fetchall())
    descriptions = _own_descriptions(conn, doc_id)
    cache = conn.execute("SELECT count(*), coalesce(sum(seconds), 0) FROM summary_cache WHERE doc_id = ?", (doc_id,)).fetchone()
    folders = _folders(conn, settings, doc)
    count = lambda table: conn.execute(f"SELECT count(*) FROM {table} WHERE doc_id = ?", (doc_id,)).fetchone()[0]  # noqa: E731
    return {
        "title": metadata.shown_title(doc["title"]),
        "chunks": {kind: chunks.get(kind, 0) for kind in ("text", "figure", "summary")},
        "blocks": count("blocks"), "figures": count("figures"), "jobs": count("jobs"),
        "annotations": count("annotations"), "notes_kept": count("notes"),
        "figure_descriptions": len(descriptions),
        "figure_descriptions_usd": round(sum(row["cost_usd"] for row in descriptions), 4),
        "summary_cache": cache[0], "summary_cache_seconds": round(cache[1]),
        "folders": [str(path.relative_to(settings.data_dir)) for path in folders],
        "folder_bytes": sum(_size(path) for path in folders),
    }


def remove_document(conn: sqlite3.Connection, settings: Settings, doc: sqlite3.Row) -> dict:
    """Delete the document and everything made from it, in one transaction, then its folders.

    Notes stay, detached from the document (ON DELETE SET NULL); the hosted-call log stays,
    since it records what was spent.
    """
    preview = removal_preview(conn, settings, doc)
    doc_id = doc["id"]
    folders = _folders(conn, settings, doc)
    descriptions = _own_descriptions(conn, doc_id)
    with conn:
        conn.executemany("DELETE FROM figure_descriptions WHERE image_sha = ? AND describer = ? AND prompt_version = ?",
                         [(row["image_sha"], row["describer"], row["prompt_version"]) for row in descriptions])
        conn.execute("DELETE FROM summary_cache WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))     # triggers drop keyword and vector rows
        for table in ("blocks", "figures", "jobs"):
            conn.execute(f"DELETE FROM {table} WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    root = settings.parsed_dir.resolve()
    for folder in folders:
        if folder.resolve().parent == root:     # never anything outside data/parsed/
            shutil.rmtree(folder)
    return preview
