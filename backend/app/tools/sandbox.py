"""Runs things in a container that can reach nothing but one temporary folder.

Every model-written program, formula and document goes through `run_container`. The
limits are in `container_command` and nowhere else, so no caller can loosen them:

    no network, no project files, no .env, no library
    only a new temporary folder mounted, at /work
    read-only everywhere else, no capabilities, CPU, memory and process limits

`--sandbox` in pandoc adds a second wall, but it doesn't cover TeX, so this is the one
that counts. A file leaves the folder only through `safe_copy` (out/ through `out_folder`),
which matters as much: the container could have made `plot.png`, or out/ itself, a link
to anything, and an ordinary copy on this side would follow it.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import stat
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings

CONTAINER_WORK = "/work"
DOCKER_FAILED = (125, 126, 127)     # docker itself failed, not the command inside it
KILLED = 137
GRACE_SECONDS = 5                   # for the container to start and stop, past its own limit
READ_BYTES = 65536


class SandboxUnavailable(RuntimeError):
    """Docker or the image is missing. Nothing ran."""


@dataclass
class Completed:
    exit_code: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool
    too_big: bool = False       # stopped for writing too much into the work folder


@dataclass(frozen=True)
class Image:
    """Which image runs, with the limits that differ between images. The walls don't differ:
    they are in `container_command` for every image. Extra mounts are always read-only."""
    name: str
    memory: str
    file_bytes: int
    tmp_size: str = "256m"
    read_only: tuple[tuple[Path, str], ...] = ()


def sandbox_image(settings: Settings) -> Image:
    return Image(settings.sandbox_image, settings.sandbox_memory, settings.sandbox_file_bytes)


def container_command(settings: Settings, work: Path, name: str, argv: list[str],
                      image: Image | None = None) -> list[str]:
    image = image or sandbox_image(settings)
    mounts = [f"type=bind,source={source},target={target},readonly" for source, target in image.read_only]
    return [
        "docker", "run", "--rm", "--init", "--name", name,
        "--network", "none",
        "--read-only",
        "--tmpfs", f"/tmp:rw,nosuid,nodev,size={image.tmp_size}",
        "--mount", f"type=bind,source={work},target={CONTAINER_WORK}",
        *[part for mount in mounts for part in ("--mount", mount)],
        "--workdir", CONTAINER_WORK,
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", str(settings.sandbox_pids),
        "--memory", image.memory,
        "--memory-swap", image.memory,              # the same figure means no swap
        "--cpus", str(settings.sandbox_cpus),
        "--ulimit", f"fsize={image.file_bytes}",
        "--log-driver", "none",
        "--env", "HOME=/tmp",
        image.name, *argv,
    ]


def docker_environment() -> dict[str, str]:
    """Only what the docker client needs: nothing from this process reaches a container."""
    return {name: os.environ[name] for name in ("PATH", "HOME") if name in os.environ}


class CappedOutput:
    """Keeps the start and end of a stream, so endless printing can't fill the memory."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.head = bytearray()
        self.tail = bytearray()
        self.dropped = 0

    def add(self, chunk: bytes) -> None:
        room = self.limit // 2 - len(self.head)
        self.head += chunk[:max(room, 0)]
        self.tail += chunk[max(room, 0):]
        if len(self.tail) > self.limit // 2:
            cut = len(self.tail) - self.limit // 2
            self.dropped += cut
            del self.tail[:cut]

    def text(self) -> str:
        head, tail = self.head.decode("utf-8", "replace"), self.tail.decode("utf-8", "replace")
        if self.dropped:
            return f"{head}\n[... {self.dropped} bytes cut ...]\n{tail}"
        return head + tail


async def drain(stream: asyncio.StreamReader, into: CappedOutput) -> None:
    while chunk := await stream.read(READ_BYTES):
        into.add(chunk)


def kill(name: str) -> None:
    """Started, not awaited: this also runs while a cancelled request is unwinding."""
    subprocess.Popen(["docker", "kill", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     env=docker_environment())


WATCH_SECONDS = 0.2


def folder_size(folder: Path, stop_at_entries: int) -> tuple[int, int]:
    """(bytes, entries) under `folder`, links not followed; stops counting past `stop_at_entries`."""
    total = entries = 0
    pending = [str(folder)]
    while pending and entries <= stop_at_entries:
        try:
            with os.scandir(pending.pop()) as found:
                for entry in found:
                    entries += 1
                    try:
                        stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue        # gone while counting
                    total += stat.st_size
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(entry.path)
        except OSError:
            continue
    return total, entries


async def watch_size(settings: Settings, work: Path, name: str, done: asyncio.Event) -> bool:
    """Kill the container if its work folder, on the host, grows past the limits. The per-file
    limit (ulimit fsize) left the total open: many files just under it fill the host's /tmp
    (review S6, Nov 2026). Counted in a thread, so the event loop stays free."""
    while not done.is_set():
        size, entries = await asyncio.to_thread(folder_size, work, settings.sandbox_work_entries)
        if size > settings.sandbox_work_bytes or entries > settings.sandbox_work_entries:
            kill(name)
            return True
        try:
            await asyncio.wait_for(done.wait(), WATCH_SECONDS)
        except TimeoutError:
            pass
    return False


async def run_container(settings: Settings, work: Path, argv: list[str], seconds: int,
                        image: Image | None = None) -> Completed:
    """Run `argv` in the image. `seconds` is enforced here even if the container ignores it."""
    name = f"study-sandbox-{uuid.uuid4().hex[:12]}"
    started = time.monotonic()
    try:
        process = await asyncio.create_subprocess_exec(
            *container_command(settings, work, name, argv, image),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=docker_environment(),
        )
    except FileNotFoundError:
        raise SandboxUnavailable("Docker isn't installed") from None
    stdout, stderr = CappedOutput(settings.sandbox_output_bytes), CappedOutput(settings.sandbox_output_bytes)
    timed_out = False
    done = asyncio.Event()
    watcher = asyncio.create_task(watch_size(settings, work, name, done))
    try:
        try:
            await asyncio.wait_for(asyncio.gather(
                drain(process.stdout, stdout), drain(process.stderr, stderr), process.wait(),
            ), seconds + GRACE_SECONDS)
        except TimeoutError:
            timed_out = True
            kill(name)
            await process.wait()
    finally:
        done.set()
        if process.returncode is None:      # the request was cancelled
            kill(name)
    too_big = await watcher
    if process.returncode in DOCKER_FAILED and not stdout.text():
        raise SandboxUnavailable(f"Docker couldn't start the sandbox: {stderr.text().strip()[-300:]}")
    errors = stderr.text()
    if too_big:
        errors += (f"\n[stopped: the run wrote more than {settings.sandbox_work_bytes // 2**20} MB or "
                   f"{settings.sandbox_work_entries} files]")
    return Completed(process.returncode, stdout.text(), errors, time.monotonic() - started, timed_out, too_big)


def give_nonce(work: Path) -> str:
    """A secret the checker puts in its report (sandbox/safe_sympify.py). The checker deletes
    the file before it reads any claim, so evaluated text can't copy it into a forged report."""
    nonce = secrets.token_hex(16)
    (work / "nonce").write_text(nonce, encoding="ascii")
    return nonce


def last_json_line(stdout: str) -> dict | None:
    """The last line of `stdout` that is a JSON object: a build's or a render's report, after any
    noise its tools printed. Shared by executor and office (review X2)."""
    for line in reversed(stdout.strip().splitlines()):
        try:
            found = json.loads(line)
        except ValueError:
            continue
        if isinstance(found, dict):
            return found
    return None


def report_line(stdout: str, nonce: str) -> dict | None:
    """The last line of `stdout` that is a JSON object carrying `nonce`, without it."""
    for line in reversed(stdout.strip().splitlines()):
        try:
            found = json.loads(line)
        except ValueError:
            continue
        if isinstance(found, dict) and secrets.compare_digest(str(found.pop("nonce", "")), nonce):
            return found
    return None


@contextmanager
def work_folder() -> Iterator[Path]:
    """A new empty folder outside the project, gone afterwards. The only thing mounted."""
    folder = Path(tempfile.mkdtemp(prefix="study-sandbox-"))
    try:
        yield folder
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def safe_copy(source: Path | str, target: Path, max_bytes: int, dir_fd: int | None = None) -> int:
    """Copy one regular file out of a work folder. Returns its size, or raises ValueError.

    The folder was writable from inside the container, so nothing about it is trusted:
    a link is refused rather than followed, however it got there, and the file is
    checked after opening so that it can't be swapped for one in between. The open never
    waits: a pipe left there would otherwise block this process for good. With `dir_fd`
    (from `out_folder`), `source` is one name inside that folder, so no link on the way
    to it is followed either.
    """
    if dir_fd is not None and Path(source).name != str(source):
        raise ValueError(f"{source} is not a plain file name")
    try:
        handle = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
    except OSError as error:
        raise ValueError(f"{Path(source).name} is not a regular file") from error
    try:
        details = os.fstat(handle)
        if not stat.S_ISREG(details.st_mode):
            raise ValueError(f"{Path(source).name} is not a regular file")
        if details.st_size > max_bytes:
            raise ValueError(f"{Path(source).name} is too large")
        with open(target, "xb") as out, os.fdopen(os.dup(handle), "rb") as read:
            shutil.copyfileobj(read, out)
        return details.st_size
    finally:
        os.close(handle)


@contextmanager
def out_folder(work: Path) -> Iterator[int | None]:
    """The work folder's out/ as an open directory, or None if the run made none.

    The container could have replaced out/ with a link to any folder on this machine;
    opening it with O_NOFOLLOW refuses that, and everything read from it afterwards goes
    through this descriptor (`os.scandir(fd)`, `safe_copy(name, ..., dir_fd=fd)`).
    """
    try:
        handle = os.open(work / "out", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        yield None
        return
    except OSError as error:            # ELOOP for a link, ENOTDIR for anything else
        raise ValueError("the run's out folder isn't a folder") from error
    try:
        yield handle
    finally:
        os.close(handle)


def copy_out(work: Path, name: str, target: Path, max_bytes: int) -> int:
    """`safe_copy` of one file the run left in out/."""
    with out_folder(work) as folder:
        if folder is None:
            raise ValueError(f"{name} wasn't made")
        return safe_copy(name, target, max_bytes, dir_fd=folder)
