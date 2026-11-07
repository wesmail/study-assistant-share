"""What a document is called: title, authors, year and publisher, never its file name.

Taken from, in order:
  1. library/metadata.yaml, which the reader edits
  2. the PDF's own metadata
  3. the title the parser found on the pages
For a deck, 2 and 3 are its slides instead (from_deck).
and otherwise "Untitled document". File names are often a download's leftovers, such as
"Simon J. D. Prince - Understanding Deep Learning-The MIT Press (2024)-55-386", so the
app never shows one.

    python -m app.ingest.metadata        reread metadata.yaml and update the library
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import yaml

from app.config import Settings, load_settings
from app.db.connection import connect
from app.errors import Refused

log = logging.getLogger(__name__)

METADATA_FILE = "metadata.yaml"
UNTITLED = "Untitled document"
FIELDS = ("title", "author", "year", "publisher")


class MetadataError(Refused, ValueError):
    """metadata.yaml can't be read. It says which pages are kept away from hosted models, so
    nothing that depends on it goes ahead as if it were empty (review R2-C7, Nov 2026): the app
    still starts and says what is wrong, ingestion waits, hosted Explain and Suggest are refused."""


def _pages(name: str, value) -> list[int]:
    values = value if isinstance(value, list) else [value]
    if not values or not all(isinstance(page, int) and not isinstance(page, bool) and page >= 1 for page in values):
        raise MetadataError(f"metadata.yaml: {name}: exclude_pages must be page numbers, such as [1] or [1, 3]")
    return values


def read_overrides(library_dir: Path) -> dict[str, dict]:
    """The reader's entries, by file name. Raises MetadataError if the file can't be trusted."""
    path = library_dir / METADATA_FILE
    if not path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as error:
        place = getattr(error, "problem_mark", None)
        where = f" (line {place.line + 1})" if place is not None else ""
        raise MetadataError(f"metadata.yaml can't be read{where}: {getattr(error, 'problem', None) or error}") from None
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise MetadataError("metadata.yaml must list file names, each with its details under it")
    overrides: dict[str, dict] = {}
    for name, entry in loaded.items():
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise MetadataError(f"metadata.yaml: {name}: its details must go under it, one per line")
        entry = dict(entry)
        if entry.get("exclude_pages") is not None:
            entry["exclude_pages"] = _pages(str(name), entry["exclude_pages"])
        if entry.get("year") not in (None, "") and not str(entry["year"]).strip().isdigit():
            log.warning("metadata.yaml: %s: year %r isn't a year; left out", name, entry["year"])
            entry.pop("year")
        overrides[str(name)] = entry
    return overrides


def problem(library_dir: Path) -> str | None:
    """What is wrong with metadata.yaml, for the library pane, or None."""
    try:
        read_overrides(library_dir)
    except MetadataError as error:
        return str(error)
    return None


def from_pdf(path: Path) -> dict:
    """The PDF's own title and authors, which many files leave empty."""
    try:
        import fitz

        with fitz.open(path) as document:
            found = document.metadata or {}
    except Exception:
        return {}
    author = ", ".join(name.strip() for name in (found.get("author") or "").replace(";", ",").split(",") if name.strip())
    return {"title": (found.get("title") or "").strip() or None, "author": author or None}


def parsed_title(settings: Settings, sha256: str) -> str | None:
    info = settings.parsed_dir / sha256 / "info.json"
    if info.is_file():
        return json.loads(info.read_text(encoding="utf-8")).get("title") or None
    return None


def from_deck(settings: Settings, path: str, sha256: str) -> dict:
    """A deck's title and author, from its slides (app/ingest/deck.py): the stored properties
    only when the title in them appears on the slides, since templates fill them in."""
    from app.ingest import deck

    slides = deck.read_slides(settings.parsed_dir / sha256)
    if slides is None:
        return {}
    title, trusted = deck.deck_title(slides, deck.excluded_pages(settings, path))
    author = (slides.get("author") or "").strip() if trusted else ""
    return {"title": title, "author": author or None}


def resolve(settings: Settings, path: str, sha256: str, overrides: dict[str, dict]) -> dict:
    """The best value of each field, from the most trusted place that has one. A deck's render
    is a PDF too, but its properties are LibreOffice's copy of the deck's, so they're skipped."""
    from app.ingest import deck

    chosen = {field: None for field in FIELDS}
    if deck.is_deck(path):
        sources = [overrides.get(path, {}), from_deck(settings, path, sha256)]
    else:
        sources = [overrides.get(path, {}), from_pdf(settings.library_dir / path),
                   {"title": parsed_title(settings, sha256)}]
    for source in sources:
        for field in FIELDS:
            if chosen[field] is None and source.get(field) not in (None, ""):
                chosen[field] = source[field]
    if chosen["year"] is not None:
        chosen["year"] = int(chosen["year"])
    return chosen


def refresh(conn: sqlite3.Connection, settings: Settings) -> int:
    """Bring every document's title, authors and collection up to date. Returns how many changed.

    `collection` comes from metadata.yaml only: "test" marks the books the evaluation suites
    use, which a library setting can hide from whole-library search and the library pane.
    """
    overrides = read_overrides(settings.library_dir)
    changed = 0
    with conn:
        for row in conn.execute("SELECT id, path, sha256, title, author, year, publisher, collection FROM documents").fetchall():
            found = resolve(settings, row["path"], row["sha256"], overrides)
            found["collection"] = str(overrides.get(row["path"], {}).get("collection") or "") or None
            if any(found[field] != row[field] for field in (*FIELDS, "collection")):
                conn.execute("UPDATE documents SET title = ?, author = ?, year = ?, publisher = ?, collection = ? WHERE id = ?",
                             (found["title"], found["author"], found["year"], found["publisher"], found["collection"],
                              row["id"]))
                changed += 1
    from app.ingest.deck import apply_exclusions      # pages added to exclude_pages since ingestion
    apply_exclusions(conn, settings)
    return changed


def shown_title(title: str | None) -> str:
    return title or UNTITLED


def main() -> None:
    settings = load_settings()
    conn = connect(settings.db_path)
    try:
        print(f"Updated {refresh(conn, settings)} documents")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
