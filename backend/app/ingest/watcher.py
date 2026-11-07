"""Folder watcher on library/.

The observer thread only records that a path changed; the worker's main thread
decides what to do with it. No database access happens on the observer thread.

A path is released once it has been quiet for `settle_seconds` and its size and
mtime have stopped moving, because inotify fires while a large book is still
being copied and hashing a half-written file would queue a bogus version. A
deleted path is released too, so the worker can mark the document missing.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from app.config import Settings
from app.ingest.worker import is_ingestible


def _fingerprint(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


class SettleQueue:
    """Debounce: hold each path until it stops changing."""

    def __init__(self, settle_seconds: float):
        self.settle_seconds = settle_seconds
        self._lock = threading.Lock()
        self._pending: dict[Path, tuple[float, tuple[int, int] | None]] = {}

    def touch(self, path: Path, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._pending[path] = (now, _fingerprint(path))

    def settled(self, now: float | None = None) -> list[Path]:
        now = time.monotonic() if now is None else now
        ready = []
        with self._lock:
            for path, (touched, fingerprint) in list(self._pending.items()):
                if now - touched < self.settle_seconds:
                    continue
                current = _fingerprint(path)
                if current != fingerprint:
                    self._pending[path] = (now, current)
                    continue
                del self._pending[path]
                ready.append(path)
        return ready


class _Handler(FileSystemEventHandler):
    def __init__(self, settings: Settings, queue: SettleQueue):
        self.settings = settings
        self.queue = queue

    def _note(self, raw_path: str | bytes) -> None:
        path = Path(raw_path.decode() if isinstance(raw_path, bytes) else raw_path)
        if is_ingestible(path, self.settings.library_dir):
            self.queue.touch(path)

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory or event.event_type in ("opened", "closed_no_write"):
            return
        self._note(event.src_path)
        if event.dest_path:
            self._note(event.dest_path)


class LibraryWatcher:
    def __init__(self, settings: Settings):
        self.queue = SettleQueue(settings.settle_seconds)
        self._observer = Observer()
        self._observer.schedule(_Handler(settings, self.queue), str(settings.library_dir), recursive=True)

    def start(self) -> None:
        self._observer.start()

    def settled(self) -> list[Path]:
        return self.queue.settled()

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join(timeout=5)
