"""Word and PowerPoint through LibreOffice, in its own container (roadmap item 7).

Two uses: a deck in the library is rendered to PDF and read slide by slide (its text and
speaker notes) so it can go through the PDF pipeline; and a Word or PowerPoint file the
tools made is rendered for a real preview in the chat.

Every file is untrusted, so the host never parses a Word or PowerPoint file: it only reads
the zip's directory here, to refuse a file that is too big before anything starts.
LibreOffice and python-pptx run in `office_image` (office/) behind the sandbox's walls
(app/tools/sandbox.py): no network, a read-only root, one temporary folder, no capabilities,
CPU, memory, process and time limits. The reader's fonts are mounted read-only and never
copied anywhere. The PDF LibreOffice makes is then read on the host by PyMuPDF and Docling,
as every PDF in the library is: native parsers on files the reader chose, not sandboxed
(review S8, Nov 2026; recorded in ARCHITECTURE.md).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import zipfile
from pathlib import Path

from app.config import Settings
from app.tools.sandbox import (Image, SandboxUnavailable, copy_out, last_json_line, run_container, safe_copy,
                               work_folder)

SUFFIXES = {".pptx", ".docx"}
SCRIPT = "/opt/office/render.py"
FONTS_TARGET = "/opt/fonts/windows"
SLIDE_ENTRY = re.compile(r"ppt/slides/slide\d+\.xml")
MAX_ENTRIES = 20000
MAX_JSON_BYTES = 32 * 1024 * 1024


class OfficeError(RuntimeError):
    """The file couldn't be rendered or read. The message is fit to show the reader."""


def office_image(settings: Settings) -> Image:
    fonts = settings.office_fonts_dir
    mounts = ((fonts, FONTS_TARGET),) if fonts and Path(fonts).is_dir() else ()
    cache = font_cache_dir(settings)
    if mounts and (cache / STAMP).is_file():
        mounts += ((cache, CACHE_TARGET),)
    return Image(settings.office_image, settings.office_memory, settings.office_max_bytes,
                 tmp_size="1g", read_only=mounts)


CACHE_TARGET = "/opt/fontcache"
_cache_lock = asyncio.Lock()     # its own: a render waiting for its turn may need the cache
STAMP = "fonts-mtime"


def font_cache_dir(settings: Settings) -> Path:
    return settings.data_dir / "office-fontcache"


async def ensure_font_cache(settings: Settings) -> None:
    """Build fontconfig's cache of the reader's fonts once, and again when the folder changes.
    No document is open in that run: only fc-cache over the read-only fonts, so what it
    writes can be trusted, and later renders get it read-only."""
    fonts = settings.office_fonts_dir
    if not fonts or not Path(fonts).is_dir():
        return
    cache = font_cache_dir(settings)
    current = str(Path(fonts).stat().st_mtime_ns)
    stamp = cache / STAMP
    if stamp.is_file() and stamp.read_text() == current:
        return
    async with _cache_lock:
        if stamp.is_file() and stamp.read_text() == current:
            return
        with work_folder() as work:
            image = Image(settings.office_image, settings.office_memory, settings.office_max_bytes,
                          tmp_size="1g", read_only=((Path(fonts), FONTS_TARGET),))
            # Only the mounted folder: the image's own fonts were cached when it was built, and
            # its root is read-only. fc-cache's exit code reports those; the cache file decides.
            done = await run_container(settings, work, ["sh", "-c", f"fc-cache {FONTS_TARGET}; cp -r /tmp/fontconfig/. /work/"],
                                       settings.office_seconds, image)
            made = [path for path in work.iterdir() if path.name.endswith((".cache-9", ".cache-8"))]
            if done.timed_out or not made:
                raise OfficeError(f"Couldn't index the fonts in {fonts}: {done.stderr.strip()[-300:]}")
            cache.mkdir(parents=True, exist_ok=True)
            for path in made:
                target = cache / path.name
                target.unlink(missing_ok=True)
                safe_copy(path, target, 64 * 1024 * 1024)
        stamp.write_text(current)


def precheck(path: Path, settings: Settings) -> None:
    """Refuse a file before any container starts, from sizes alone. The zip's directory can
    lie about sizes; the container's memory, file and time limits are what hold then."""
    suffix = path.suffix.lower()
    if suffix not in SUFFIXES:
        raise OfficeError(f"Only .pptx and .docx files can be rendered, not {suffix or 'this file'}")
    size = path.stat().st_size
    if size > settings.office_max_bytes:
        raise OfficeError(f"{path.name} is {size / 2**20:.0f} MB; the limit is "
                          f"{settings.office_max_bytes / 2**20:.0f} MB")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
    except (zipfile.BadZipFile, OSError) as error:
        raise OfficeError(f"{path.name} isn't a valid Office file ({error})") from None
    if len(entries) > MAX_ENTRIES:
        raise OfficeError(f"{path.name} has {len(entries)} parts; the limit is {MAX_ENTRIES}")
    unpacked = sum(entry.file_size for entry in entries)
    if unpacked > settings.office_max_unpacked_bytes:
        raise OfficeError(f"{path.name} unpacks to {unpacked / 2**20:.0f} MB; the limit is "
                          f"{settings.office_max_unpacked_bytes / 2**20:.0f} MB")
    if suffix == ".pptx":
        slides = sum(1 for entry in entries if SLIDE_ENTRY.fullmatch(entry.filename))
        if slides > settings.office_max_slides:
            raise OfficeError(f"{path.name} has {slides} slides; the limit is {settings.office_max_slides}")


async def _run(settings: Settings, command: str, source: Path, wanted: str, target: Path,
               max_bytes: int) -> dict:
    """Run one render.py command on a copy of `source`; copy `wanted` out to `target`."""
    precheck(source, settings)
    if command == "render":
        await ensure_font_cache(settings)
    with work_folder() as work:
        name = "in" + source.suffix.lower()
        (work / name).write_bytes(source.read_bytes())
        (work / "out").mkdir()
        done = await run_container(settings, work, ["python", SCRIPT, command, name],
                                   settings.office_seconds, office_image(settings))
        if done.timed_out:
            raise OfficeError(f"Rendering {source.name} took longer than {settings.office_seconds} s")
        result = last_json_line(done.stdout)
        if result is None:
            reason = "it ran out of memory" if done.exit_code == 137 else f"exit code {done.exit_code}"
            raise OfficeError(f"LibreOffice stopped on {source.name} ({reason})")
        if not result.get("ok"):
            raise OfficeError(f"{source.name}: {result.get('error', 'failed')}")
        try:
            copy_out(work, wanted, target, max_bytes)
        except (ValueError, FileNotFoundError, FileExistsError) as error:
            raise OfficeError(f"{source.name}: no usable output ({error})") from None
        return result


async def render_pdf(settings: Settings, source: Path, target: Path) -> int:
    """Render `source` to the PDF `target`, written atomically. Returns its page count."""
    import pymupdf

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.part")
    partial.unlink(missing_ok=True)
    try:
        await _run(settings, "render", source, "render.pdf", partial, settings.office_max_bytes)
        try:
            with pymupdf.open(partial) as pdf:
                if not pdf.is_pdf:
                    raise OfficeError(f"{source.name}: the render isn't a PDF")
                pages = pdf.page_count
        except RuntimeError as error:
            raise OfficeError(f"{source.name}: the render can't be opened ({error})") from None
        os.replace(partial, target)
        return pages
    finally:
        partial.unlink(missing_ok=True)


async def _json_output(settings: Settings, command: str, source: Path, wanted: str):
    with work_folder() as scratch:
        target = scratch / wanted
        await _run(settings, command, source, wanted, target, MAX_JSON_BYTES)
        return json.loads(target.read_text(encoding="utf-8"))


async def read_slides(settings: Settings, source: Path) -> dict:
    """A deck's title, author and slides: each with its title, text lines (equations as plain
    characters), equations, speaker notes and whether it's hidden. Page n of the render is
    slide n, hidden slides included."""
    return await _json_output(settings, "slides", source, "slides.json")


async def source_words(settings: Settings, source: Path) -> list[str]:
    """The words a render should show: a deck's slide text (not notes), or a Word body."""
    return await _json_output(settings, "words", source, "words.json")


_render_turn = asyncio.Semaphore(1)     # one LibreOffice at a time: each takes up to 3 GB
_render_locks: dict[str, asyncio.Lock] = {}


async def cached_render(settings: Settings, source: Path) -> Path:
    """The PDF of `source`, rendered once and kept under renders_dir by its SHA-256.
    Two requests for the same file wait for one render."""
    import hashlib

    precheck(source, settings)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target = settings.renders_dir / f"{digest}.pdf"
    lock = _render_locks.setdefault(digest, asyncio.Lock())
    async with lock:
        if not target.is_file():
            async with _render_turn:
                await render_pdf(settings, source, target)
    _render_locks.pop(digest, None)
    return target


def run_sync(coroutine):
    """For the ingest worker, which isn't async."""
    return asyncio.run(coroutine)


__all__ = ["OfficeError", "SandboxUnavailable", "cached_render", "precheck", "read_slides", "render_pdf",
           "run_sync", "source_words"]
