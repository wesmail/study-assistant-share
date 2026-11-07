"""Backups of everything git can't recreate, and restoring one into a fresh checkout.

    scripts/backup.sh create                # one archive now, then old ones are pruned
    scripts/backup.sh create --if-older     # only when the newest is older than backup_max_age_hours
    scripts/backup.sh create --pre-rework   # the one-off archive of data/pre-rework/
    scripts/backup.sh list
    scripts/backup.sh verify [ARCHIVE]      # the newest when left out
    scripts/backup.sh copy DIR              # the newest archive to DIR (an external drive), checked
    scripts/backup.sh restore ARCHIVE --to DIR [--env]

An archive is a .tar.gz in backup_dir (config.yaml): data/ (the database as a consistent
snapshot from SQLite's backup API, so it can be taken while the app writes), library/
(the PDFs, which git doesn't hold), outputs/ and the gitignored eval folders, with a
MANIFEST.json of every file's SHA-256 at the end. It is written as .partial and renamed
when complete; a .sha256 file beside it lets `sha256sum -c` check a copy anywhere.

.env never goes into an archive. Its copy goes to backup_local_dir/env/ (mode 600 in a
mode-700 folder, under /home, where Linux permissions hold), named by the archive's
timestamp and pruned with it.

Retention: the newest archive of each of the last backup_keep_daily days that have one,
plus the newest of each of the last backup_keep_weekly ISO weeks. The pre-rework archive
is never pruned.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath

import sqlite_vec

from app.config import ROOT, Settings, load_settings

PREFIX = "study-assistant-"
STAMP = "%Y%m%d-%H%M%S"
MANIFEST = "MANIFEST.json"
FORMAT = 1
DB_NAME = "data/app.sqlite"
DB_FILES = {"app.sqlite", "app.sqlite-wal", "app.sqlite-shm", "app.sqlite-journal"}
TRANSIENT = {"worker.lock", "chat-busy"}        # markers of a running app, meaningless in a copy
PRE_REWORK = "pre-rework"                       # backed up once, by itself (create --pre-rework)
EVAL_DIRS = ("eval/grading", "eval/spot_check", "eval/needed_facts")   # gitignored: blind grades
ALLOWED = {"data", "library", "outputs", *EVAL_DIRS}
COUNTED = ("documents", "chunks", "conversations", "messages", "notes", "annotations")   # the last two: roadmap item 4


class BackupError(Exception):
    """A backup or restore that can't go on; the message says why, for the reader."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def env_dir(settings: Settings) -> Path:
    return settings.backup_local_dir / "env"


def sidecar(archive: Path) -> Path:
    return archive.with_name(archive.name + ".sha256")


def stamp_of(archive: Path) -> datetime | None:
    """The time in a daily archive's name; None for anything else (the pre-rework one included)."""
    name = archive.name
    if not (name.startswith(PREFIX) and name.endswith(".tar.gz")):
        return None
    try:
        return datetime.strptime(name[len(PREFIX):-len(".tar.gz")], STAMP)
    except ValueError:
        return None


# --- the database ------------------------------------------------------------------------

def open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30)
    conn.enable_load_extension(True)        # integrity_check asks vec0 tables to check themselves
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def snapshot_db(source: Path, target: Path) -> None:
    """Copy the database as of one read transaction, WAL contents included, while the app may
    be writing. The copy is switched to a rollback journal so it is a single file; the app
    turns WAL back on when it opens it."""
    src = open_db(source)
    try:
        dst = sqlite3.connect(str(target))
        try:
            src.backup(dst)
            dst.execute("PRAGMA journal_mode = DELETE")
        finally:
            dst.close()
    finally:
        src.close()


def db_facts(path: Path) -> dict:
    conn = open_db(path)
    try:
        problems = [row[0] for row in conn.execute("PRAGMA integrity_check")]
        counts = {}
        for table in COUNTED:
            try:
                counts[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            except sqlite3.OperationalError:
                counts[table] = None
    finally:
        conn.close()
    return {"integrity": "ok" if problems == ["ok"] else "; ".join(problems[:5]), "counts": counts}


# --- what goes in ------------------------------------------------------------------------

def files_under(folder: Path, name: str, skip: set[str] = frozenset()):
    """(name in the archive, path) for each regular file under folder, in a stable order.
    Symbolic links are left out: a restore must never write outside its folder."""
    if not folder.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(folder):
        here = Path(dirpath)
        if here == folder:
            dirnames[:] = [d for d in dirnames if d not in skip]
            filenames = [f for f in filenames if f not in skip]
        dirnames[:] = sorted(d for d in dirnames if not (here / d).is_symlink())
        for filename in sorted(filenames):
            path = here / filename
            if path.is_symlink() or not path.is_file():
                continue
            yield f"{name}/{path.relative_to(folder).as_posix()}", path


def empty_dirs_under(folder: Path, name: str, skip: set[str] = frozenset()) -> list[str]:
    """Folders with nothing in them, which a tar of files alone would drop; a restore makes them
    again so the copy matches the original."""
    if not folder.is_dir():
        return []
    empty = []
    for dirpath, dirnames, filenames in os.walk(folder):
        here = Path(dirpath)
        if here == folder:
            dirnames[:] = [d for d in dirnames if d not in skip]
            continue
        dirnames[:] = sorted(d for d in dirnames if not (here / d).is_symlink())
        if not dirnames and not filenames:
            empty.append(f"{name}/{here.relative_to(folder).as_posix()}")
    return sorted(empty)


def trees(settings: Settings, root: Path, pre_rework: bool) -> list[tuple[str, Path, set[str]]]:
    if pre_rework:
        return [(f"data/{PRE_REWORK}", settings.data_dir / PRE_REWORK, set())]
    return [("data", settings.data_dir, DB_FILES | TRANSIENT | {PRE_REWORK}),
            ("library", settings.library_dir, set()),
            ("outputs", settings.outputs_dir, set()),
            *[(name, root / name, set()) for name in EVAL_DIRS]]


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


@contextmanager
def backup_lock(settings: Settings):
    """One backup at a time (the timer and start.sh can both start one). Yields False if
    another holds it."""
    settings.backup_local_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.backup_local_dir / ".lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def save_env(settings: Settings, env_file: Path, stamp: str) -> Path | None:
    """A copy of .env readable only by the user. Its contents are never read into a message."""
    if not env_file.is_file():
        return None
    folder = env_dir(settings)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(folder, 0o700)
    target = folder / f"{stamp}.env"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(env_file.read_bytes())
    return target


def create(settings: Settings, root: Path = ROOT, *, pre_rework: bool = False,
           now: datetime | None = None) -> dict:
    """Write one archive and, for a daily one, the .env copy; then prune. Returns the manifest
    with where everything went."""
    if settings.backup_dir is None:
        raise BackupError("Backups are off: set backup_dir in config.yaml.")
    now = now or datetime.now()
    stamp = now.strftime(STAMP)
    name = f"{PREFIX}{'pre-rework-' if pre_rework else ''}{stamp}.tar.gz"
    if pre_rework:
        if not (settings.data_dir / PRE_REWORK).is_dir():
            raise BackupError(f"There is no {settings.data_dir / PRE_REWORK} to back up.")
        earlier = sorted(settings.backup_dir.glob(f"{PREFIX}pre-rework-*.tar.gz"))
        if earlier:
            raise BackupError(f"data/{PRE_REWORK} is already backed up in {earlier[-1]}.")
    try:
        settings.backup_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise BackupError(f"Can't create {settings.backup_dir}: {error}. Is the drive there?") from None
    final = settings.backup_dir / name
    if final.exists():
        raise BackupError(f"{final} already exists.")
    partial = final.with_name(name + ".partial")
    entries, vanished, dirs, database = [], [], [], None

    try:
        with tempfile.TemporaryDirectory(prefix="study-backup-") as tmp, \
                tarfile.open(partial, "w:gz", compresslevel=6) as tar:
            def add(arcname: str, data: bytes, mtime: float) -> None:
                info = tarfile.TarInfo(arcname)
                info.size, info.mtime, info.mode = len(data), int(mtime), 0o644
                tar.addfile(info, io.BytesIO(data))
                entries.append({"path": arcname, "size": len(data), "sha256": sha256(data)})

            if not pre_rework and settings.db_path.exists():
                snapshot = Path(tmp) / "app.sqlite"
                snapshot_db(settings.db_path, snapshot)
                database = db_facts(snapshot)
                if database["integrity"] != "ok":
                    raise BackupError(f"The database copy fails its integrity check: {database['integrity']}")
                add(DB_NAME, snapshot.read_bytes(), time.time())
            for arcroot, folder, skip in trees(settings, root, pre_rework):
                dirs += empty_dirs_under(folder, arcroot, skip)
                for arcname, path in files_under(folder, arcroot, skip):
                    try:        # read once: what is hashed is what is archived
                        data, mtime = path.read_bytes(), path.stat().st_mtime
                    except FileNotFoundError:
                        vanished.append(arcname)        # removed by the app while we walked
                        continue
                    add(arcname, data, mtime)
            manifest = {
                "format": FORMAT,
                "kind": "pre-rework" if pre_rework else "daily",
                "stamp": stamp,
                "created": now.isoformat(timespec="seconds"),
                "git_commit": git_commit(root),
                "database": database,
                "files": list(entries),         # the manifest's own entry is added after it
                "dirs": dirs,
                "vanished": vanished,
            }
            add(MANIFEST, json.dumps(manifest, indent=1, ensure_ascii=False).encode(), time.time())
        with open(partial, "rb") as f:
            os.fsync(f.fileno())
        digest = file_sha256(partial)       # read back from the drive, not from memory
        partial.rename(final)
    except BaseException:
        partial.unlink(missing_ok=True)     # this run's own unfinished file
        raise
    sidecar(final).write_text(f"{digest}  {name}\n", encoding="utf-8")

    env_copy = None if pre_rework else save_env(settings, root / ".env", stamp)
    removed = [] if pre_rework else prune(settings)
    return {**manifest, "archive": str(final), "archive_bytes": final.stat().st_size,
            "env_copy": str(env_copy) if env_copy else None, "removed": [str(p) for p in removed]}


# --- retention ---------------------------------------------------------------------------

def daily_archives(settings: Settings) -> list[tuple[datetime, Path]]:
    if settings.backup_dir is None or not settings.backup_dir.is_dir():
        return []
    found = [(stamp_of(path), path) for path in settings.backup_dir.glob(f"{PREFIX}*.tar.gz")]
    return sorted((stamp, path) for stamp, path in found if stamp is not None)


def keep_set(stamps: list[datetime], daily: int, weekly: int) -> dict[datetime, str]:
    """Which archives to keep, and why: the newest of each of the last `daily` days that have
    one, and the newest of each of the last `weekly` ISO weeks."""
    newest_first = sorted(stamps, reverse=True)
    keep: dict[datetime, str] = {}
    days: list = []
    for stamp in newest_first:
        if stamp.date() not in days and len(days) < daily:
            days.append(stamp.date())
            keep[stamp] = "daily"
    weeks: list = []
    for stamp in newest_first:
        week = stamp.isocalendar()[:2]
        if week not in weeks and len(weeks) < weekly:
            weeks.append(week)
            keep.setdefault(stamp, "weekly")
    return keep


def prune(settings: Settings) -> list[Path]:
    found = daily_archives(settings)
    keep = keep_set([stamp for stamp, _ in found], settings.backup_keep_daily, settings.backup_keep_weekly)
    removed = []
    for stamp, path in found:
        if stamp in keep:
            continue
        for old in (path, sidecar(path), env_dir(settings) / f"{stamp.strftime(STAMP)}.env"):
            old.unlink(missing_ok=True)
        removed.append(path)
    return removed


def age_hours(settings: Settings, now: datetime | None = None) -> float | None:
    found = daily_archives(settings)
    if not found:
        return None
    return ((now or datetime.now()) - found[-1][0]).total_seconds() / 3600


# --- reading an archive back ------------------------------------------------------------

def checked_name(name: str) -> str:
    """An archive member's path, refused unless it is a plain relative path under a folder
    a backup writes."""
    path = PurePosixPath(name)
    parts = path.parts
    if path.is_absolute() or not parts or any(part in ("", ".", "..") for part in parts) or "\\" in name:
        raise BackupError(f"Refusing a member with an unsafe path: {name!r}")
    if parts[0] not in ALLOWED and "/".join(parts[:2]) not in ALLOWED:
        raise BackupError(f"Refusing a member outside the backed-up folders: {name!r}")
    if len(parts) < 2:
        raise BackupError(f"Refusing a member that is a top-level file: {name!r}")
    return path.as_posix()


def check_sidecar(archive: Path) -> bool:
    """True if the .sha256 beside it matches; False if there is none. A mismatch is an error."""
    side = sidecar(archive)
    if not side.is_file():
        return False
    expected = side.read_text(encoding="utf-8").split()[0]
    if file_sha256(archive) != expected:
        raise BackupError(f"{archive} doesn't match {side.name}: the copy is damaged or incomplete.")
    return True


def read_archive(archive: Path, target: Path | None = None) -> dict:
    """Read every member once, in order, checking it against the manifest; write each to
    target when one is given. Returns the manifest."""
    hashes: dict[str, str] = {}
    manifest = None
    with tempfile.TemporaryDirectory(prefix="study-restore-") as tmp:
        try:
            with tarfile.open(archive, "r|gz") as tar:     # a stream: in order, never seeking back
                for member in tar:
                    if not member.isfile():
                        raise BackupError(f"Refusing {member.name!r}: only regular files are restored.")
                    data = tar.extractfile(member).read()
                    if member.name == MANIFEST:
                        manifest = json.loads(data)
                        continue
                    name = checked_name(member.name)
                    if name in hashes:
                        raise BackupError(f"{name} is in the archive twice.")
                    hashes[name] = sha256(data)
                    if target is not None:
                        write_file(target / name, data)
                    elif name == DB_NAME:
                        (Path(tmp) / "app.sqlite").write_bytes(data)
        except (tarfile.TarError, EOFError, OSError, json.JSONDecodeError) as error:
            raise BackupError(f"Can't read {archive}: {error}") from None
        if manifest is None:
            raise BackupError(f"{archive} has no {MANIFEST}: it isn't one of these backups, or it is cut short.")
        if manifest.get("format") != FORMAT:
            raise BackupError(f"{archive} is backup format {manifest.get('format')}; this code reads {FORMAT}.")
        expected = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
        for name in [*expected, *manifest.get("dirs", [])]:
            checked_name(name)
        wrong = sorted(name for name in hashes if expected.get(name) != hashes[name])
        missing = sorted(set(expected) - set(hashes))
        if wrong or missing:
            raise BackupError(f"{archive} doesn't match its manifest: {len(wrong)} files differ "
                              f"({', '.join(wrong[:3])}), {len(missing)} missing ({', '.join(missing[:3])}).")
        if target is not None:
            for name in manifest.get("dirs", []):
                (target / name).mkdir(parents=True, exist_ok=True)
        db = (target / DB_NAME) if target is not None else (Path(tmp) / "app.sqlite")
        if manifest.get("database") is not None:
            facts = db_facts(db)
            if facts["integrity"] != "ok":
                raise BackupError(f"The database in {archive} fails its integrity check: {facts['integrity']}")
            if facts["counts"] != manifest["database"]["counts"]:
                raise BackupError(f"The database counts {facts['counts']} differ from the manifest's "
                                  f"{manifest['database']['counts']}.")
    return manifest


def write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".restoring")
    partial.write_bytes(data)
    os.replace(partial, path)


def verify(archive: Path) -> dict:
    had_sidecar = check_sidecar(archive)
    manifest = read_archive(archive)
    return {**manifest, "sidecar_checked": had_sidecar}


def restore(archive: Path, target: Path, settings: Settings, *, with_env: bool = False) -> dict:
    """Restore into a checkout that has no database yet. Nothing is written until the whole
    archive has been checked, and no existing file is ever overwritten with different content."""
    target = target.resolve()
    if not (target / "backend" / "app" / "config.py").is_file():
        raise BackupError(f"{target} isn't a checkout of the study assistant. Clone the repository there first.")
    if (target / DB_NAME).exists():
        raise BackupError(f"{target / DB_NAME} already exists. Restore into a fresh checkout, "
                          f"or move {target / 'data'} aside first.")
    manifest = verify(archive)      # first pass: nothing is written
    conflicts = [entry["path"] for entry in manifest["files"]
                 if (target / entry["path"]).exists() and file_sha256(target / entry["path"]) != entry["sha256"]]
    if conflicts:
        raise BackupError(f"{len(conflicts)} files in {target} differ from the backup and would be "
                          f"overwritten: {', '.join(conflicts[:5])}. Move them aside first.")
    read_archive(archive, target)   # second pass: writes, checking every file again
    env = None
    if with_env:
        env = restore_env(settings, manifest["stamp"], target / ".env")
    return {**manifest, "target": str(target), "env": env}


def restore_env(settings: Settings, stamp: str, target: Path) -> str:
    source = env_dir(settings) / f"{stamp}.env"
    if not source.is_file():
        raise BackupError(f"There is no .env copy for this backup in {env_dir(settings)}. "
                          "Use another copy there, or create new keys in the Anthropic and OpenRouter consoles.")
    data = source.read_bytes()
    if target.exists():
        if target.read_bytes() == data:
            return f"{target} already matches the copy"
        raise BackupError(f"{target} exists and differs from the copy; move it aside first.")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return f"{target} restored from {source} (mode 600)"


def copy_newest(settings: Settings, destination: Path) -> Path:
    """Copy the newest archive and its .sha256 to destination, then check the copy there."""
    found = daily_archives(settings)
    if not found:
        raise BackupError(f"There are no backups in {settings.backup_dir}.")
    archive = found[-1][1]
    check_sidecar(archive)
    destination.mkdir(parents=True, exist_ok=True)
    copied = destination / archive.name
    if copied.exists():
        raise BackupError(f"{copied} already exists.")
    shutil.copyfile(archive, copied)
    shutil.copyfile(sidecar(archive), sidecar(copied))
    if not check_sidecar(copied):
        raise BackupError(f"The .sha256 file didn't arrive in {destination}.")
    return copied


# --- command line ------------------------------------------------------------------------

def megabytes(size: int) -> str:
    return f"{size / 1e6:,.0f} MB"


def describe(manifest: dict) -> str:
    db = manifest.get("database")
    if not db:
        return f"{len(manifest['files']):,} files"
    counts = db["counts"]
    return (f"{len(manifest['files']):,} files; database: {counts['documents']} documents, "
            f"{counts['chunks']:,} chunks, {counts['conversations']} conversations, "
            f"{counts['messages']} messages"
            + (f", {counts['notes']} notes, {counts['annotations']} highlights" if counts.get("notes") is not None else "")
            + f", integrity {db['integrity']}")


def cmd_create(settings: Settings, args) -> int:
    if args.if_older:
        age = age_hours(settings)
        if age is not None and age < settings.backup_max_age_hours:
            print(f"The newest backup is {age:.1f} h old; nothing to do.")
            return 0
    with backup_lock(settings) as held:
        if not held:
            print("Another backup is running; leaving it to finish.")
            return 0 if args.if_older else 1
        result = create(settings, pre_rework=args.pre_rework)
    print(f"Backed up to {result['archive']} ({megabytes(result['archive_bytes'])}, {describe(result)}).")
    if result["vanished"]:
        print(f"Removed by the app while being copied, so left out: {', '.join(result['vanished'])}")
    if result["env_copy"]:
        print(f".env copy: {result['env_copy']} (mode 600)")
    elif not args.pre_rework:
        print("No .env to copy.")
    for path in result["removed"]:
        print(f"Pruned: {path}")
    return 0


def cmd_list(settings: Settings, args) -> int:
    if settings.backup_dir is None:
        print("Backups are off: set backup_dir in config.yaml.")
        return 1
    found = daily_archives(settings)
    keep = keep_set([stamp for stamp, _ in found], settings.backup_keep_daily, settings.backup_keep_weekly)
    print(f"In {settings.backup_dir}:")
    for stamp, path in reversed(found):
        env = "with .env copy" if (env_dir(settings) / f"{stamp.strftime(STAMP)}.env").exists() else "no .env copy"
        print(f"  {path.name}  {megabytes(path.stat().st_size):>8}  {keep.get(stamp, 'to prune'):7}  {env}")
    for path in sorted(settings.backup_dir.glob(f"{PREFIX}pre-rework-*.tar.gz")):
        print(f"  {path.name}  {megabytes(path.stat().st_size):>8}  kept for good")
    if not found:
        print("  no daily backups yet")
    return 0


def cmd_due(settings: Settings, args) -> int:
    """For start.sh: exit 0 when a backup is due, 1 when not, 2 when backups are off."""
    if settings.backup_dir is None:
        return 2
    age = age_hours(settings)
    if age is None:
        print("There is no backup yet.")
        return 0
    if age >= settings.backup_max_age_hours:
        print(f"The newest backup is {age / 24:.1f} days old." if age >= 48 else
              f"The newest backup is {age:.0f} h old.")
        return 0
    return 1


def newest_or(settings: Settings, given: str | None) -> Path:
    if given:
        return Path(given)
    found = daily_archives(settings)
    if not found:
        raise BackupError(f"There are no backups in {settings.backup_dir}.")
    return found[-1][1]


def cmd_verify(settings: Settings, args) -> int:
    archive = newest_or(settings, args.archive)
    result = verify(archive)
    checked = "matches its .sha256" if result["sidecar_checked"] else "no .sha256 beside it"
    print(f"{archive}: OK ({checked}; {describe(result)}).")
    return 0


def cmd_copy(settings: Settings, args) -> int:
    copied = copy_newest(settings, Path(args.destination))
    print(f"Copied to {copied}, with its .sha256; the copy matches it.")
    print("The .env copy stays in /home: if this drive is all that is left, create new keys "
          "in the Anthropic and OpenRouter consoles.")
    return 0


def cmd_restore(settings: Settings, args) -> int:
    result = restore(Path(args.archive), Path(args.to), settings, with_env=args.env)
    print(f"Restored {args.archive} into {result['target']} ({describe(result)}).")
    print(result["env"] or f"No .env restored; add --env to restore the copy from {env_dir(settings)}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Back up and restore the study assistant's data.")
    commands = parser.add_subparsers(dest="command", required=True)
    create_cmd = commands.add_parser("create", help="write one archive now, then prune old ones")
    create_cmd.add_argument("--if-older", action="store_true", help="only if the newest is older than backup_max_age_hours")
    create_cmd.add_argument("--pre-rework", action="store_true", help=f"the one-off archive of data/{PRE_REWORK}/")
    commands.add_parser("list", help="the archives, and which retention keeps each")
    commands.add_parser("due", help="exit 0 if a backup is due (for start.sh)")
    verify_cmd = commands.add_parser("verify", help="check an archive end to end, writing nothing")
    verify_cmd.add_argument("archive", nargs="?", help="the newest when left out")
    copy_cmd = commands.add_parser("copy", help="copy the newest archive to a folder, such as an external drive")
    copy_cmd.add_argument("destination")
    restore_cmd = commands.add_parser("restore", help="restore an archive into a fresh checkout")
    restore_cmd.add_argument("archive")
    restore_cmd.add_argument("--to", required=True, help="a fresh checkout of the repository")
    restore_cmd.add_argument("--env", action="store_true", help="also restore the matching .env copy")
    args = parser.parse_args(argv)
    handlers = {"create": cmd_create, "list": cmd_list, "due": cmd_due, "verify": cmd_verify,
                "copy": cmd_copy, "restore": cmd_restore}
    try:
        return handlers[args.command](load_settings(), args)
    except BackupError as error:
        print(f"Backup: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
