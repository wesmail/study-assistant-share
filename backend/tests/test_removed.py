"""Remove, Restore and Empty in the library pane (app/removed.py): reversible until emptied."""
import os

import fitz
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db.connection import connect
from app.ingest.worker import is_ingestible, scan_library, sync_file
from app.main import create_app


def pdf_bytes(text="Hello") -> bytes:
    with fitz.open() as document:
        document.new_page().insert_text((72, 72), text)
        return document.tobytes()


@pytest.fixture
def settings(tmp_path):
    made = Settings(data_dir=tmp_path / "data", library_dir=tmp_path / "library", outputs_dir=tmp_path / "outputs")
    made.library_dir.mkdir(parents=True)
    connect(made.db_path).close()
    return made


@pytest.fixture
def client(settings, tmp_path):
    app = create_app(settings, embed=lambda texts, s: [[0.0] * 4 for _ in texts], frontend=tmp_path / "none")
    return TestClient(app, base_url="http://localhost")


@pytest.fixture
def conn(settings):
    connection = connect(settings.db_path)
    yield connection
    connection.close()


def indexed(conn, settings, name="book.pdf", text="Hello", collection=None) -> int:
    """A document as the worker leaves it once indexed, with a highlight and a note."""
    path = settings.library_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf_bytes(text))
    assert sync_file(conn, settings, path) == "new"
    doc_id = conn.execute("SELECT id FROM documents WHERE path = ?", (name,)).fetchone()[0]
    with conn:
        conn.execute("UPDATE jobs SET status = 'done' WHERE doc_id = ?", (doc_id,))
        conn.execute("UPDATE documents SET status = 'indexed', title = ?, collection = ? WHERE id = ?",
                     (f"Title of {name}", collection, doc_id))
        conn.execute("INSERT INTO annotations (doc_id, page, bbox, quote, kind, origin, status) "
                     "VALUES (?, 1, '[0,0,1,1]', 'Hello', 'highlight', 'user', 'accepted')",
                     (doc_id,))
        conn.execute("INSERT INTO notes (doc_id, doc_title, page, quote, body_md) VALUES (?, 'Title', 1, 'Hello', 'A note')",
                     (doc_id,))
    return doc_id


def status(conn, doc_id):
    row = conn.execute("SELECT status FROM documents WHERE id = ?", (doc_id,)).fetchone()
    return row[0] if row else None


def count(conn, table, doc_id):
    return conn.execute(f"SELECT count(*) FROM {table} WHERE doc_id = ?", (doc_id,)).fetchone()[0]


class TestRemove:
    def test_the_file_moves_aside_and_the_document_leaves_search_with_everything_kept(self, client, conn, settings):
        doc_id = indexed(conn, settings)
        body = (settings.library_dir / "book.pdf").read_bytes()
        jobs = count(conn, "jobs", doc_id)
        response = client.post(f"/api/documents/{doc_id}/set-aside")
        assert response.status_code == 200, response.text
        entry = response.json()["removed_as"]
        assert entry.endswith("/book.pdf")
        assert not (settings.library_dir / "book.pdf").exists()
        assert (settings.library_dir / ".removed" / entry).read_bytes() == body
        assert status(conn, doc_id) == "missing"
        assert (count(conn, "annotations", doc_id), count(conn, "notes", doc_id), count(conn, "jobs", doc_id)) == (1, 1, jobs)
        listing = client.get("/api/documents").json()
        assert [row["removed_as"] for row in listing["documents"]] == [entry]
        assert [item["entry"] for item in listing["removed"]] == [entry]

    def test_the_watcher_then_finds_nothing_to_do_and_never_reads_removed_files(self, client, conn, settings):
        doc_id = indexed(conn, settings)
        entry = client.post(f"/api/documents/{doc_id}/set-aside").json()["removed_as"]
        assert sync_file(conn, settings, settings.library_dir / "book.pdf") == "unknown"
        removed_file = settings.library_dir / ".removed" / entry
        assert not is_ingestible(removed_file, settings.library_dir)
        scan_library(conn, settings)
        assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 1

    def test_a_file_in_a_subfolder_keeps_its_path(self, client, conn, settings):
        doc_id = indexed(conn, settings, name="physics/book.pdf")
        entry = client.post(f"/api/documents/{doc_id}/set-aside").json()["removed_as"]
        assert entry.endswith("/physics/book.pdf")
        assert client.post("/api/library/removed/restore", json={"entry": entry}).status_code == 200
        assert (settings.library_dir / "physics" / "book.pdf").is_file()

    def test_refused_while_a_job_runs_or_when_the_file_is_already_gone(self, client, conn, settings):
        doc_id = indexed(conn, settings)
        with conn:
            conn.execute("UPDATE jobs SET status = 'running' WHERE doc_id = ? AND stage = 'summarize'", (doc_id,))
        assert client.post(f"/api/documents/{doc_id}/set-aside").status_code == 409
        assert (settings.library_dir / "book.pdf").is_file()
        with conn:
            conn.execute("UPDATE jobs SET status = 'done' WHERE doc_id = ?", (doc_id,))
        assert client.post(f"/api/documents/{doc_id}/set-aside").status_code == 200
        assert client.post(f"/api/documents/{doc_id}/set-aside").status_code == 409

    def test_another_site_cant_remove_a_file(self, client, conn, settings):
        doc_id = indexed(conn, settings)
        response = client.post(f"/api/documents/{doc_id}/set-aside", headers={"Origin": "http://evil.example"})
        assert response.status_code == 403
        assert (settings.library_dir / "book.pdf").is_file()


class TestRestore:
    def test_the_document_comes_back_as_it_was_with_nothing_parsed_again(self, client, conn, settings):
        doc_id = indexed(conn, settings)
        jobs = count(conn, "jobs", doc_id)
        entry = client.post(f"/api/documents/{doc_id}/set-aside").json()["removed_as"]
        response = client.post("/api/library/removed/restore", json={"entry": entry})
        assert response.status_code == 200 and response.json() == {"restored": "book.pdf"}
        assert sync_file(conn, settings, settings.library_dir / "book.pdf") == "restored"
        assert status(conn, doc_id) == "indexed"
        assert count(conn, "jobs", doc_id) == jobs and count(conn, "annotations", doc_id) == 1
        assert not (settings.library_dir / ".removed").exists() or not any((settings.library_dir / ".removed").iterdir())

    def test_a_name_taken_again_is_refused_and_both_files_stay(self, client, conn, settings):
        doc_id = indexed(conn, settings)
        entry = client.post(f"/api/documents/{doc_id}/set-aside").json()["removed_as"]
        (settings.library_dir / "book.pdf").write_bytes(b"%PDF-1.4 a new book")
        response = client.post("/api/library/removed/restore", json={"entry": entry})
        assert response.status_code == 409 and "rename or remove it first" in response.json()["detail"]
        assert (settings.library_dir / "book.pdf").read_bytes() == b"%PDF-1.4 a new book"
        assert (settings.library_dir / ".removed" / entry).is_file()

    @pytest.mark.parametrize("entry", ["../book.pdf", "a\\b/book.pdf", "/etc/passwd", "book.pdf", "x/../../y.pdf", ""])
    def test_an_entry_is_never_a_path_outside_removed(self, client, entry):
        assert client.post("/api/library/removed/restore", json={"entry": entry}).status_code in (404, 409)

    def test_a_link_planted_in_removed_is_never_followed(self, client, conn, settings, tmp_path):
        secret = tmp_path / "secret.pdf"
        secret.write_bytes(b"%PDF-1.4 secret")
        planted = settings.library_dir / ".removed" / "20260101-000000"
        planted.mkdir(parents=True)
        os.symlink(secret, planted / "secret.pdf")
        response = client.post("/api/library/removed/restore", json={"entry": "20260101-000000/secret.pdf"})
        assert response.status_code == 404
        assert not (settings.library_dir / "secret.pdf").exists()
        assert client.get("/api/library/removed/empty").json()["files"] == []


class TestEmpty:
    def test_the_preview_lists_files_and_documents_and_deletes_nothing(self, client, conn, settings):
        doc_id = indexed(conn, settings)
        entry = client.post(f"/api/documents/{doc_id}/set-aside").json()["removed_as"]
        preview = client.get("/api/library/removed/empty").json()
        assert [item["entry"] for item in preview["files"]] == [entry]
        assert preview["documents"] == [{"id": doc_id, "title": "Title of book.pdf"}]
        assert preview["notes_kept"] == 1
        assert (settings.library_dir / ".removed" / entry).is_file() and status(conn, doc_id) == "missing"

    def test_emptying_deletes_the_files_and_their_index_but_keeps_notes(self, client, conn, settings):
        doc_id = indexed(conn, settings)
        client.post(f"/api/documents/{doc_id}/set-aside")
        result = client.post("/api/library/removed/empty").json()
        assert result == {"files": 1, "documents_removed": [doc_id], "documents_kept": []}
        assert status(conn, doc_id) is None and count(conn, "annotations", doc_id) == 0
        assert conn.execute("SELECT count(*) FROM notes WHERE doc_id IS NULL AND body_md = 'A note'").fetchone()[0] == 1
        assert client.get("/api/documents").json()["removed"] == []

    def test_a_document_back_in_the_library_keeps_its_index(self, client, conn, settings):
        """Added again under another name while its old file sat in .removed: only the file goes."""
        doc_id = indexed(conn, settings)
        body = (settings.library_dir / "book.pdf").read_bytes()
        client.post(f"/api/documents/{doc_id}/set-aside")
        (settings.library_dir / "again.pdf").write_bytes(body)
        assert sync_file(conn, settings, settings.library_dir / "again.pdf") == "moved"
        result = client.post("/api/library/removed/empty").json()
        assert result["files"] == 1 and result["documents_removed"] == []
        assert status(conn, doc_id) == "indexed" and count(conn, "annotations", doc_id) == 1
        assert (settings.library_dir / "again.pdf").read_bytes() == body

    def test_nothing_outside_removed_is_ever_deleted(self, client, conn, settings):
        keep = indexed(conn, settings, name="keep.pdf", text="Keep")
        gone = indexed(conn, settings, name="gone.pdf", text="Gone")
        client.post(f"/api/documents/{gone}/set-aside")
        client.post("/api/library/removed/empty")
        assert (settings.library_dir / "keep.pdf").is_file() and status(conn, keep) == "indexed"
        assert count(conn, "annotations", keep) == 1


def test_a_removed_file_is_still_in_the_backups(client, conn, settings):
    from app.backup import files_under

    doc_id = indexed(conn, settings)
    entry = client.post(f"/api/documents/{doc_id}/set-aside").json()["removed_as"]
    assert f"library/.removed/{entry}" in [name for name, _ in files_under(settings.library_dir, "library")]
