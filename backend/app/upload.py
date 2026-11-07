"""Adding a document from the browser: the library pane's Add button.

The file streams into a hidden temporary file in library/, which the worker never reads (a dot
name), is checked there, and then takes its own name through a hard link, which fails if the
name is taken. So nothing in the library is ever overwritten or changed, and the worker never
sees half a file. From there the watcher picks it up as if it had been copied in by hand.

Checks, before the file gets its name:
- the name: a plain file name, .pdf or .pptx, no path, not hidden, no control characters;
- the size: at most `upload_max_bytes`, counted as it streams, so a lie in Content-Length
  doesn't matter;
- the content: a PDF starts with %PDF- (within its first KB, as readers allow); a deck is a zip
  with ppt/presentation.xml, within the office image's limits (office.precheck);
- a document with the same content isn't already in the library.

A PDF is still parsed on the host by native code (ARCHITECTURE.md, "Not sandboxed, by
choice"), so the button changes how a file arrives, not what is trusted: add only files from
sources you trust.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import unicodedata
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path

from app.config import Settings
from app.db.connection import connect

UPLOADABLE = {".pdf", ".pptx"}
NAME_LIMIT = 180
PDF_MAGIC = b"%PDF-"
DECK_PART = "ppt/presentation.xml"


class UploadError(Exception):
    """Why a file wasn't added, for the reader, with the HTTP status to answer."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def upload_name(raw: str) -> str:
    """The name the file will have in library/, or UploadError. Never a path."""
    name = unicodedata.normalize("NFC", raw or "").strip()
    if not name or len(name) > NAME_LIMIT:
        raise UploadError(400, f"A file name must have 1 to {NAME_LIMIT} characters")
    if "/" in name or "\\" in name or name in (".", "..") or Path(name).name != name:
        raise UploadError(400, "Send a file name, not a path")
    if name.startswith((".", "~$")):
        raise UploadError(400, "A hidden or lock file can't be added")
    if any(unicodedata.category(character) in ("Cc", "Cf") for character in name):
        raise UploadError(400, "The file name has a control character in it")
    if Path(name).suffix.lower() not in UPLOADABLE:
        raise UploadError(400, "Only PDF files and PowerPoint decks (.pptx) can be added")
    return name


def check_content(path: Path, suffix: str, settings: Settings, name: str) -> None:
    """The file is what its name says, or UploadError."""
    if suffix == ".pdf":
        with open(path, "rb") as handle:
            if PDF_MAGIC not in handle.read(1024):
                raise UploadError(400, f"{name} isn't a PDF file")
        return
    from app import office

    try:
        with zipfile.ZipFile(path) as archive:
            parts = set(archive.namelist())
    except (zipfile.BadZipFile, OSError):
        raise UploadError(400, f"{name} isn't a PowerPoint deck") from None
    if DECK_PART not in parts:
        raise UploadError(400, f"{name} isn't a PowerPoint deck")
    try:
        office.precheck(path, settings)
    except office.OfficeError as error:
        raise UploadError(400, str(error).replace(path.name, name)) from None


def already_added(settings: Settings, sha256: str) -> str | None:
    """The title of a document in the library with this content, if there is one."""
    if not settings.db_path.exists():
        return None
    conn = connect(settings.db_path, apply_schema=False)
    try:
        row = conn.execute("SELECT title, path FROM documents WHERE sha256 = ? AND status != 'missing'",
                           (sha256,)).fetchone()
    finally:
        conn.close()
    return (row["title"] or row["path"]) if row else None


async def save_upload(settings: Settings, raw_name: str, chunks: AsyncIterator[bytes],
                      declared_length: str | None = None) -> dict:
    """Stream one file into library/ under its own name, checked, never overwriting anything."""
    name = upload_name(raw_name)
    suffix = Path(name).suffix.lower()
    limit = settings.upload_max_bytes
    final = settings.library_dir / name
    if final.exists():
        raise UploadError(409, f"There is already a file called {name} in the library")
    if declared_length and declared_length.isdigit() and int(declared_length) > limit:
        raise UploadError(413, f"{name} is larger than the {limit / 2**20:.0f} MB limit")
    settings.library_dir.mkdir(parents=True, exist_ok=True)
    temporary = settings.library_dir / f".upload-{secrets.token_hex(8)}{suffix}"
    digest, size = hashlib.sha256(), 0
    try:
        with open(temporary, "xb") as out:
            async for chunk in chunks:
                size += len(chunk)
                if size > limit:
                    raise UploadError(413, f"{name} is larger than the {limit / 2**20:.0f} MB limit")
                digest.update(chunk)
                out.write(chunk)
        if size == 0:
            raise UploadError(400, f"{name} is empty")
        check_content(temporary, suffix, settings, name)
        if title := already_added(settings, digest.hexdigest()):
            raise UploadError(409, f"{name} is already in the library, as {title}")
        try:
            os.link(temporary, final)       # fails if the name was taken meanwhile: never overwrites
        except FileExistsError:
            raise UploadError(409, f"There is already a file called {name} in the library") from None
    finally:
        temporary.unlink(missing_ok=True)
    return {"name": name, "size": size, "sha256": digest.hexdigest()}
