"""Open the app database with sqlite-vec loaded and the schema applied."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
# Columns added after a table first existed. CREATE TABLE IF NOT EXISTS leaves an old
# table as it was, so each is added here if it's missing.
ADDED_COLUMNS = {"annotations": {"rects": "TEXT", "prefix": "TEXT", "suffix": "TEXT", "anchor_sha": "TEXT",
                                 "anchor_status": "TEXT", "batch": "TEXT", "instruction": "TEXT"},
                 "notes": {"doc_title": "TEXT", "rects": "TEXT", "prefix": "TEXT", "suffix": "TEXT", "anchor_sha": "TEXT",
                           "anchor_status": "TEXT"},
                 "documents": {"author": "TEXT", "year": "INTEGER", "publisher": "TEXT", "collection": "TEXT"},
                 "hosted_calls": {"turn": "TEXT"},
                 "jobs": {"waiting": "TEXT", "detail": "TEXT", "approved_usd": "REAL"},
                 "messages": {"option": "TEXT", "details": "TEXT"},
                 "summary_cache": {"doc_id": "INTEGER"}}


def add_missing_columns(conn: sqlite3.Connection) -> None:
    for table, columns in ADDED_COLUMNS.items():
        present = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, kind in columns.items():
            if present and name not in present:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")


def connect(path: Path | str, apply_schema: bool = True) -> sqlite3.Connection:
    """Return a connection ready for use.

    WAL lets the folder watcher and the worker write from separate threads
    without blocking readers; busy_timeout makes them wait for each other
    instead of raising "database is locked". foreign_keys is per-connection in
    SQLite and off by default, so it has to be set here.
    """
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    if apply_schema:
        add_missing_columns(conn)
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn
