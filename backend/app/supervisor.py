"""Runs the ingestion worker beside the web server, as a child process.

    python -m app.main      starts the server and, through this, the worker and its watcher

The worker is a separate process, not a thread, because it loads Docling and torch: an
out-of-memory kill or a crash in native code there must not take the server down. The
two share nothing but the SQLite database.

If the worker dies, it is started again after a pause that grows with each crash in a
row, and the job it was running uses up one of its attempts, so a file that kills the
process every time still ends as failed. If another worker already holds the lock
(started by hand), this one waits and tries again later.

On shutdown the worker gets SIGINT and puts its current job back in the queue; if it
hasn't exited after `grace` seconds it is killed, and the job is put back here instead.
Either way, a restart of the app doesn't count against the job's attempts.
"""
from __future__ import annotations

import ctypes
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from app.config import ROOT, Settings
from app.db.connection import connect

log = logging.getLogger("supervisor")

LOCKED = 75             # the worker's exit code when another worker holds the lock (EX_TEMPFAIL)
STEADY_SECONDS = 600    # a worker that ran this long before dying resets the crash count


def worker_command() -> list[str]:
    return [sys.executable, "-m", "app.ingest.worker"]


PARENT_ENV = "STUDY_ASSISTANT_SUPERVISOR_PID"


def die_with_parent() -> None:
    """Called by the worker at startup: SIGTERM it if the server dies without stopping it (kill -9,
    a crash), so no orphan keeps the worker lock. Done in the child: preexec_fn isn't safe in a
    process with threads."""
    parent = os.environ.get(PARENT_ENV)
    if not parent:
        return
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)    # PR_SET_PDEATHSIG
    except OSError:
        return
    if os.getppid() != int(parent):     # the server died before the line above
        os.kill(os.getpid(), signal.SIGTERM)


def release_interrupted(settings: Settings) -> int:
    """Put back jobs left 'running' by a worker that was stopped, not crashed, and refund the attempt."""
    conn = connect(settings.db_path, apply_schema=False)
    try:
        with conn:
            return conn.execute(
                "UPDATE jobs SET status = 'pending', attempts = max(attempts - 1, 0) WHERE status = 'running'"
            ).rowcount
    finally:
        conn.close()


class WorkerSupervisor:
    def __init__(
        self,
        settings: Settings,
        command: Sequence[str] | None = None,
        cwd: Path = ROOT / "backend",
        delays: Sequence[float] = (5, 30, 120, 600),
        locked_delay: float = 60,
        grace: float = 15,
        output: Callable[[str], None] = lambda line: print(line, flush=True),
    ):
        self.settings = settings
        self.command = list(command or worker_command())
        self.cwd = cwd
        self.delays = list(delays)
        self.locked_delay = locked_delay
        self.grace = grace
        self.output = output
        self.starts = 0
        self._stop = threading.Event()
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="worker-supervisor", daemon=True)
        self._thread.start()

    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _spawn(self) -> subprocess.Popen:
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1", PARENT_ENV: str(os.getpid())}
        self.starts += 1
        return subprocess.Popen(
            self.command, cwd=self.cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", start_new_session=True,    # the terminal's Ctrl+C reaches only the server
        )

    def _loop(self) -> None:
        crashes = 0
        while not self._stop.is_set():
            started = time.monotonic()
            self._process = self._spawn()
            for line in self._process.stdout:
                self.output("[worker] " + line.rstrip("\n"))
            code = self._process.wait()
            if self._stop.is_set():
                return
            if code == 0:
                self.output("[worker] exited")
                return
            if code == LOCKED:
                self.output(f"[worker] another worker is running; trying again in {self.locked_delay:.0f} s")
                self._stop.wait(self.locked_delay)
                continue
            crashes = 1 if time.monotonic() - started > STEADY_SECONDS else crashes + 1
            delay = self.delays[min(crashes, len(self.delays)) - 1]
            self.output(f"[worker] stopped unexpectedly (exit {code}); starting it again in {delay:.0f} s")
            self._stop.wait(delay)

    def stop(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(self.grace)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                released = release_interrupted(self.settings)
                if released:
                    log.info("Put back %d job(s) the worker was killed in", released)
        if self._thread is not None:
            self._thread.join(timeout=5)
