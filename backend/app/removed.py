"""Removing a document from the library pane, reversibly (Nov 7, 2026).

Remove moves the file into library/.removed/<time>/<its path>. The worker never reads a dot
folder, so the document turns "missing": it leaves search at once, and its index, highlights,
notes and cached descriptions and summaries all stay. Restore moves the file back; the worker
sees the same content and restores the document as it was, with nothing parsed again.

Only "Empty removed files" deletes anything: the files in library/.removed/, then, through
library.remove_document, the index of each document whose file that was (the same content,
and still missing). Notes are kept, as with any removal.

Files move by a hard link and then an unlink, never a rename, so a move can't replace a file
that is already where it goes: a taken name is refused instead.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime
from pathlib import Path, PurePosixPath

from app import library
from app.config import Settings
from app.errors import NotFound, Refused
from app.ingest import metadata

FOLDER = ".removed"


def removed_dir(settings: Settings) -> Path:
    return settings.library_dir / FOLDER


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _move(source: Path, target: Path) -> None:
    """Move a file without ever replacing one: link, then unlink. FileExistsError if taken."""
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)
    os.unlink(source)


def _prune(folder: Path, stop: Path) -> None:
    """Remove the empty folders a move left behind, up to `stop`."""
    while folder != stop and folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()
        folder = folder.parent


def set_aside(conn: sqlite3.Connection, settings: Settings, doc: sqlite3.Row) -> dict:
    """Move a document's file into library/.removed/ and take it out of search."""
    if doc["status"] == "missing":
        raise Refused("This document's file isn't in the library")
    if conn.execute("SELECT 1 FROM jobs WHERE doc_id = ? AND status = 'running'", (doc["id"],)).fetchone():
        raise Refused("A job for this document is still running; try again when it stops")
    library_root = settings.library_dir.resolve()
    source = settings.library_dir / doc["path"]
    if (source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(library_root)
            or PurePosixPath(doc["path"]).parts[0] == FOLDER):
        raise Refused("This document's file isn't a plain file in the library")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    entry = f"{stamp}/{doc['path']}"
    for number in range(2, 100):
        target = removed_dir(settings) / entry
        try:
            _move(source, target)
            break
        except FileExistsError:
            entry = f"{stamp}-{number}/{doc['path']}"
    else:
        raise Refused("Couldn't find a free place in library/.removed/")
    with conn:      # out of search now, not when the watcher next looks
        conn.execute("UPDATE documents SET status = 'missing', updated_at = datetime('now') WHERE id = ?", (doc["id"],))
    return {"removed_as": entry}


def removed_files(settings: Settings) -> list[dict]:
    """Every file in library/.removed/: its entry, the path it came from, when, its size."""
    root = removed_dir(settings)
    if not root.is_dir():
        return []
    found = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        entry = path.relative_to(root).as_posix()
        stamp, _, original = entry.partition("/")
        if not original:
            continue
        found.append({"entry": entry, "path": original, "removed_at": stamp, "bytes": path.stat().st_size})
    return found


def entry_path(settings: Settings, entry: str) -> Path:
    """The file of one entry of library/.removed/, never anything outside it."""
    parts = PurePosixPath(entry or "").parts
    if (not entry or "\\" in entry or entry.startswith("/") or len(parts) < 2
            or any(part in ("", ".", "..") for part in parts)):
        raise Refused("That isn't an entry of library/.removed/")
    path = removed_dir(settings).joinpath(*parts)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(removed_dir(settings).resolve()):
        raise NotFound("There is no such removed file")
    return path


def restore(settings: Settings, entry: str) -> dict:
    """Move a removed file back to where it was. The worker then restores its document."""
    source = entry_path(settings, entry)
    original = PurePosixPath(*PurePosixPath(entry).parts[1:])
    target = settings.library_dir.joinpath(*original.parts)
    try:
        _move(source, target)
    except FileExistsError:
        raise Refused(f"There is a file called {original} in the library again; rename or remove it first") from None
    _prune(source.parent, removed_dir(settings))
    return {"restored": original.as_posix()}


def _documents_of(conn: sqlite3.Connection, files: list[tuple[dict, Path]]) -> list[sqlite3.Row]:
    """The missing documents whose file is one of these: the same content."""
    shas = {_sha256(path) for _, path in files}
    if not shas:
        return []
    places = ",".join("?" * len(shas))
    return conn.execute(f"SELECT * FROM documents WHERE status = 'missing' AND sha256 IN ({places})",
                        list(shas)).fetchall()


def empty_preview(conn: sqlite3.Connection, settings: Settings) -> dict:
    """What emptying library/.removed/ would delete, for the reader to confirm."""
    files = [(item, entry_path(settings, item["entry"])) for item in removed_files(settings)]
    docs = _documents_of(conn, files)
    return {
        "files": [item for item, _ in files],
        "bytes": sum(item["bytes"] for item, _ in files),
        "documents": [{"id": doc["id"], "title": metadata.shown_title(doc["title"])} for doc in docs],
        "notes_kept": sum(conn.execute("SELECT count(*) FROM notes WHERE doc_id = ?", (doc["id"],)).fetchone()[0]
                          for doc in docs),
    }


def empty(conn: sqlite3.Connection, settings: Settings) -> dict:
    """Delete the files in library/.removed/, then the index of the documents they were."""
    files = [(item, entry_path(settings, item["entry"])) for item in removed_files(settings)]
    docs = _documents_of(conn, files)
    root = removed_dir(settings)
    for _, path in files:
        path.unlink()
        _prune(path.parent, root)
    removed, kept = [], []
    for doc in docs:
        try:
            library.remove_document(conn, settings, doc)
            removed.append(doc["id"])
        except Refused:         # restored meanwhile, or a job still running: its index stays
            kept.append(doc["id"])
    return {"files": len(files), "documents_removed": removed, "documents_kept": kept}
