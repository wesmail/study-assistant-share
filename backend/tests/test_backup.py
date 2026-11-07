"""Backups: what goes in, what never does, restoring safely, and which archives are kept."""
import io
import json
import os
import sqlite3
import stat
import tarfile
from datetime import datetime, timedelta

import pytest

from app import backup
from app.backup import BackupError
from app.config import Settings

SECRET = b"STUDY_ASSISTANT_ANTHROPIC_KEY=not-a-real-key\n"


def make_db(path):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = WAL")
    for table in backup.COUNTED:
        conn.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, text TEXT)")
    conn.executemany("INSERT INTO chunks (text) VALUES (?)", [(f"chunk {i}",) for i in range(50)])
    conn.execute("INSERT INTO documents (text) VALUES ('a book')")
    conn.commit()
    return conn


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    (root / "backend" / "app").mkdir(parents=True)
    (root / "backend" / "app" / "config.py").write_text("# a checkout\n")
    data = root / "data"
    (data / "parsed" / "abc").mkdir(parents=True)
    (data / "parsed" / "abc" / "page-1.json").write_text('{"page": 1}')
    (data / "pre-rework").mkdir()
    (data / "pre-rework" / "old.sqlite").write_bytes(b"old" * 100)
    (data / "chat-busy").mkdir()
    (data / "chat-busy" / "123").write_text("busy")
    (data / "worker.lock").write_text("")
    (data / "parsed" / "abc" / "formula_crops").mkdir()          # empty: kept as a folder
    (root / "library").mkdir()
    (root / "library" / "book.pdf").write_bytes(b"%PDF-1.7 book")
    (root / "library" / "metadata.yaml").write_text("book.pdf: {}\n")
    (root / "outputs").mkdir()
    (root / "outputs" / "notes.md").write_text("# notes")
    (root / "eval" / "grading").mkdir(parents=True)
    (root / "eval" / "grading" / "round-1.json").write_text("[]")
    (root / "eval" / "results").mkdir()
    (root / "eval" / "results" / "tracked.json").write_text("{}")      # in git: not backed up
    env = root / ".env"
    env.write_bytes(SECRET)
    os.chmod(env, 0o600)
    conn = make_db(data / "app.sqlite")
    settings = Settings(library_dir=root / "library", data_dir=data, outputs_dir=root / "outputs",
                        backup_dir=tmp_path / "d-drive" / "backups", backup_local_dir=tmp_path / "home-backups")
    yield root, settings, conn
    conn.close()


def names_in(archive):
    with tarfile.open(archive) as tar:
        return set(tar.getnames())


def test_an_archive_holds_what_git_cannot_recreate(project):
    root, settings, _ = project
    result = backup.create(settings, root)
    names = names_in(result["archive"])
    assert {"data/app.sqlite", "data/parsed/abc/page-1.json", "library/book.pdf", "library/metadata.yaml",
            "outputs/notes.md", "eval/grading/round-1.json", backup.MANIFEST} == names
    assert result["database"] == {"integrity": "ok",
                                  "counts": {"documents": 1, "chunks": 50, "conversations": 0, "messages": 0,
                                             "notes": 0, "annotations": 0}}


def test_env_never_goes_into_the_archive_and_its_copy_is_private(project):
    root, settings, _ = project
    result = backup.create(settings, root)
    with open(result["archive"], "rb") as f:
        raw = f.read()
    with tarfile.open(result["archive"]) as tar:
        contents = b"".join(tar.extractfile(m).read() for m in tar.getmembers())
    assert b"not-a-real-key" not in raw and b"not-a-real-key" not in contents
    copy = backup.env_dir(settings) / f"{result['stamp']}.env"
    assert copy.read_bytes() == SECRET
    assert stat.S_IMODE(copy.stat().st_mode) == 0o600
    assert stat.S_IMODE(backup.env_dir(settings).stat().st_mode) == 0o700
    assert str(copy).startswith(str(settings.backup_local_dir))


def test_the_database_copy_is_consistent_while_a_writer_holds_a_transaction(project):
    root, settings, conn = project
    conn.execute("INSERT INTO chunks (text) VALUES ('committed')")
    conn.commit()
    conn.execute("BEGIN")
    conn.execute("INSERT INTO chunks (text) VALUES ('not yet committed')")    # left open
    result = backup.create(settings, root)
    conn.rollback()
    assert result["database"]["counts"]["chunks"] == 51
    assert result["database"]["integrity"] == "ok"


def test_the_sidecar_matches_and_verify_reads_everything_back(project):
    root, settings, _ = project
    result = backup.create(settings, root)
    archive = backup.Path(result["archive"])
    assert backup.sidecar(archive).read_text().split()[0] == backup.file_sha256(archive)
    assert backup.verify(archive)["sidecar_checked"] is True


def test_a_damaged_archive_is_refused(project):
    root, settings, _ = project
    archive = backup.Path(backup.create(settings, root)["archive"])
    data = bytearray(archive.read_bytes())
    data[len(data) // 2] ^= 0xFF
    archive.write_bytes(bytes(data))
    with pytest.raises(BackupError, match="damaged"):
        backup.verify(archive)


def test_restore_rebuilds_the_files_and_the_env_copy(project, tmp_path):
    root, settings, _ = project
    result = backup.create(settings, root)
    fresh = tmp_path / "fresh"
    (fresh / "backend" / "app").mkdir(parents=True)
    (fresh / "backend" / "app" / "config.py").write_text("# a checkout\n")
    (fresh / "library").mkdir()
    (fresh / "library" / "metadata.yaml").write_text("book.pdf: {}\n")       # tracked, the same
    restored = backup.restore(backup.Path(result["archive"]), fresh, settings, with_env=True)
    for name in ("library/book.pdf", "outputs/notes.md", "data/parsed/abc/page-1.json", "eval/grading/round-1.json"):
        assert (fresh / name).read_bytes() == (root / name).read_bytes()
    assert backup.db_facts(fresh / "data" / "app.sqlite")["counts"]["chunks"] == 50
    assert (fresh / "data" / "parsed" / "abc" / "formula_crops").is_dir()
    assert len(restored["files"]) == len(result["files"]) == 6     # the manifest isn't counted
    assert (fresh / ".env").read_bytes() == SECRET
    assert stat.S_IMODE((fresh / ".env").stat().st_mode) == 0o600
    assert "restored" in restored["env"]


def test_restore_refuses_a_folder_that_has_a_database_or_different_files(project, tmp_path):
    root, settings, _ = project
    archive = backup.Path(backup.create(settings, root)["archive"])
    with pytest.raises(BackupError, match="already exists"):
        backup.restore(archive, root, settings)
    fresh = tmp_path / "fresh"
    (fresh / "backend" / "app").mkdir(parents=True)
    (fresh / "backend" / "app" / "config.py").write_text("")
    (fresh / "library").mkdir()
    (fresh / "library" / "book.pdf").write_bytes(b"a different book")
    with pytest.raises(BackupError, match="differ from the backup"):
        backup.restore(archive, fresh, settings)
    assert (fresh / "library" / "book.pdf").read_bytes() == b"a different book"
    assert not (fresh / "data").exists()
    with pytest.raises(BackupError, match="isn't a checkout"):
        backup.restore(archive, tmp_path / "empty", settings)


@pytest.mark.parametrize("name", ["../escape.txt", "/etc/passwd", "library/../../escape.txt",
                                  "backend/app/main.py", "eval/questions.yaml", ".env"])
def test_members_outside_the_backed_up_folders_are_refused(tmp_path, name):
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for member, data in ((name, b"x"), (backup.MANIFEST, json.dumps(
                {"format": backup.FORMAT, "stamp": "x", "database": None,
                 "files": [{"path": name, "sha256": backup.sha256(b"x"), "size": 1}]}).encode())):
            info = tarfile.TarInfo(member)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with pytest.raises(BackupError, match="Refusing"):
        backup.read_archive(archive)
    assert not (tmp_path / "escape.txt").exists()


def test_a_symlink_member_is_refused(tmp_path):
    archive = tmp_path / "link.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("library/book.pdf")
        info.type, info.linkname = tarfile.SYMTYPE, "/home/someone/.env"
        tar.addfile(info)
    with pytest.raises(BackupError, match="only regular files"):
        backup.read_archive(archive)


def test_pre_rework_is_left_out_of_daily_copies_and_archived_once_by_itself(project):
    root, settings, _ = project
    daily = backup.create(settings, root)
    assert not any("pre-rework" in name for name in names_in(daily["archive"]))
    once = backup.create(settings, root, pre_rework=True, now=datetime(2026, 11, 5, 9))
    assert names_in(once["archive"]) == {"data/pre-rework/old.sqlite", backup.MANIFEST}
    assert once["env_copy"] is None
    with pytest.raises(BackupError, match="already backed up"):
        backup.create(settings, root, pre_rework=True, now=datetime(2026, 11, 6, 9))


def test_keep_set_is_seven_days_plus_four_weeks():
    start = datetime(2026, 9, 1, 12, 30)
    stamps = [start + timedelta(days=d) for d in range(60)] + [start + timedelta(days=59, hours=3)]
    keep = backup.keep_set(stamps, daily=7, weekly=4)
    newest = max(stamps)
    days = {s.date() for s, why in keep.items() if why == "daily"}
    assert len(days) == 7 and newest.date() in days
    assert newest in keep and start + timedelta(days=59) not in keep        # one per day: the newest
    weeks = {s.isocalendar()[:2] for s in keep}
    assert len(weeks) == 4                  # the last four ISO weeks, the dailies among them
    assert len(keep) == 7 + len([s for s, why in keep.items() if why == "weekly"])
    assert all(why == "weekly" for s, why in keep.items() if s.date() < newest.date() - timedelta(days=6))


def test_prune_removes_old_archives_with_their_sidecars_and_env_copies(project):
    root, settings, _ = project
    start = datetime(2026, 9, 1, 12, 30)
    for day in range(30):
        backup.create(settings, root, now=start + timedelta(days=day))
    backup.create(settings, root, pre_rework=True, now=start)
    archives = backup.daily_archives(settings)
    assert len(archives) == len(backup.keep_set([s for s, _ in archives], 7, 4)) <= 11
    kept = {s.strftime(backup.STAMP) for s, _ in archives}
    assert {p.name.removesuffix(".env") for p in backup.env_dir(settings).glob("*.env")} == kept
    assert {p.name.removesuffix(".tar.gz.sha256").removeprefix(backup.PREFIX)
            for p in settings.backup_dir.glob(f"{backup.PREFIX}2*.sha256")} == kept
    assert list(settings.backup_dir.glob(f"{backup.PREFIX}pre-rework-*.tar.gz"))


def test_a_failed_backup_leaves_no_partial_file_and_prunes_nothing(project, monkeypatch):
    root, settings, _ = project
    first = backup.create(settings, root, now=datetime(2026, 9, 1))

    def broken(*args):
        raise OSError("disk full")
    monkeypatch.setattr(backup, "snapshot_db", broken)
    with pytest.raises(OSError):
        backup.create(settings, root, now=datetime(2026, 11, 1))
    assert not list(settings.backup_dir.glob("*.partial"))
    assert backup.Path(first["archive"]).exists()


def test_copy_puts_the_newest_and_its_sidecar_on_another_drive(project, tmp_path):
    root, settings, _ = project
    backup.create(settings, root, now=datetime(2026, 11, 4))
    newest = backup.create(settings, root, now=datetime(2026, 11, 5))
    copied = backup.copy_newest(settings, tmp_path / "external")
    assert copied.name == backup.Path(newest["archive"]).name
    assert backup.check_sidecar(copied)


def test_due_follows_the_age_of_the_newest(project, capsys):
    root, settings, _ = project
    assert backup.cmd_due(settings, None) == 0
    backup.create(settings, root)
    assert backup.cmd_due(settings, None) == 1
    assert backup.cmd_due(Settings(backup_dir=None), None) == 2


def test_notes_and_highlights_are_counted_and_described(project):
    """Roadmap item 4: notes and highlights are in the database copy, and the manifest counts them."""
    root, settings, conn = project
    conn.executemany("INSERT INTO notes (text) VALUES (?)", [("a note",), ("another",)])
    conn.execute("INSERT INTO annotations (text) VALUES ('a highlight')")
    conn.commit()
    result = backup.create(settings, root)
    assert result["database"]["counts"]["notes"] == 2 and result["database"]["counts"]["annotations"] == 1
    assert "2 notes, 1 highlights" in backup.describe(result)
