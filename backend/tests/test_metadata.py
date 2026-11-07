"""Document metadata: what a document is called everywhere, and never its file name."""
import json
import sqlite3

import fitz
import pytest
from fastapi.testclient import TestClient

from app.chat import source_record
from app.config import Settings
from app.db.connection import connect
from app.ingest import metadata
from app.ingest.embed import serialize
from app.retrieval.search import hybrid_search

UGLY = "Simon J. D. Prince - Understanding Deep Learning-The MIT Press (2024)-55-386.pdf"


@pytest.fixture
def settings(tmp_path):
    made = Settings(data_dir=tmp_path / "data", library_dir=tmp_path / "library", outputs_dir=tmp_path / "out")
    made.library_dir.mkdir(parents=True)
    return made


def pdf(path, title="", author=""):
    document = fitz.open()
    document.new_page().insert_text((50, 72), "Words")
    document.set_metadata({"title": title, "author": author})
    document.save(path)
    document.close()


def add(conn, path, title=None, sha="s"):
    conn.execute("INSERT INTO documents (path, sha256, title, kind, status) VALUES (?, ?, ?, 'pdf', 'indexed')",
                 (path, sha, title))
    conn.commit()


def row(conn, path="a.pdf"):
    return conn.execute("SELECT * FROM documents WHERE path = ?", (path,)).fetchone()


class TestWhereItComesFrom:
    def test_the_readers_file_comes_first(self, settings):
        pdf(settings.library_dir / "a.pdf", title="PDF title", author="PDF Author")
        (settings.library_dir / "metadata.yaml").write_text(
            "a.pdf:\n  title: My title\n  author: Ada Lovelace\n  year: 1843\n  publisher: A Press\n")
        found = metadata.resolve(settings, "a.pdf", "s", metadata.read_overrides(settings.library_dir))
        assert found == {"title": "My title", "author": "Ada Lovelace", "year": 1843, "publisher": "A Press"}

    def test_the_pdfs_own_metadata_fills_what_the_file_leaves_out(self, settings):
        pdf(settings.library_dir / "a.pdf", title="PDF title", author="Ann One; Bob Two")
        (settings.library_dir / "metadata.yaml").write_text("a.pdf:\n  year: 2020\n")
        found = metadata.resolve(settings, "a.pdf", "s", metadata.read_overrides(settings.library_dir))
        assert (found["title"], found["author"], found["year"]) == ("PDF title", "Ann One, Bob Two", 2020)

    def test_then_the_title_the_parser_found(self, settings):
        pdf(settings.library_dir / "a.pdf")
        (settings.parsed_dir / "s").mkdir(parents=True)
        (settings.parsed_dir / "s" / "info.json").write_text(json.dumps({"title": "Parsed title"}))
        assert metadata.resolve(settings, "a.pdf", "s", {})["title"] == "Parsed title"

    def test_with_nothing_known_it_is_untitled_and_never_the_file_name(self, settings):
        pdf(settings.library_dir / UGLY)
        found = metadata.resolve(settings, UGLY, "s", {})
        assert found["title"] is None
        assert metadata.shown_title(found["title"]) == "Untitled document"


class TestTheLibrary:
    def test_an_old_database_gets_the_new_columns(self, tmp_path):
        old = tmp_path / "old.sqlite"
        raw = sqlite3.connect(old)
        raw.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL, "
                    "title TEXT, kind TEXT NOT NULL, pages INTEGER, status TEXT NOT NULL DEFAULT 'queued', "
                    "created_at TEXT NOT NULL DEFAULT (datetime('now')), updated_at TEXT NOT NULL DEFAULT (datetime('now')))")
        raw.commit()
        raw.close()
        columns = {item[1] for item in connect(old).execute("PRAGMA table_info(documents)")}
        assert {"author", "year", "publisher"} <= columns

    def test_refresh_replaces_a_file_name_title_from_before(self, settings):
        conn = connect(settings.db_path)
        pdf(settings.library_dir / UGLY)
        add(conn, UGLY, title=UGLY.removesuffix(".pdf"))
        (settings.library_dir / "metadata.yaml").write_text(
            f'"{UGLY}":\n  title: Understanding Deep Learning\n  author: Simon J. D. Prince\n  year: 2024\n')
        assert metadata.refresh(conn, settings) == 1
        found = row(conn, UGLY)
        assert (found["title"], found["author"], found["year"]) == ("Understanding Deep Learning", "Simon J. D. Prince", 2024)

    def test_refresh_without_any_metadata_clears_a_file_name_title(self, settings):
        conn = connect(settings.db_path)
        pdf(settings.library_dir / UGLY)
        add(conn, UGLY, title=UGLY.removesuffix(".pdf"))
        metadata.refresh(conn, settings)
        assert row(conn, UGLY)["title"] is None

    def test_the_library_list_shows_metadata_titles(self, settings):
        from app.main import create_app

        conn = connect(settings.db_path)
        pdf(settings.library_dir / UGLY)
        pdf(settings.library_dir / "x.pdf")
        add(conn, UGLY, title=UGLY, sha="a")
        add(conn, "x.pdf", sha="b")
        conn.close()
        (settings.library_dir / "metadata.yaml").write_text(
            f'"{UGLY}":\n  title: Understanding Deep Learning\n  author: Simon J. D. Prince\n  year: 2024\n')
        listed = TestClient(create_app(settings), headers={"host": "localhost"}).get("/api/documents").json()
        shown = {item["title"]: (item["author"], item["year"]) for item in listed["documents"]}
        assert shown == {"Understanding Deep Learning": ("Simon J. D. Prince", 2024), "Untitled document": (None, None)}
        assert not any(".pdf" in item["title"] for item in listed["documents"])


def test_search_results_and_sources_carry_the_metadata(settings):
    conn = connect(settings.db_path)
    conn.execute("INSERT INTO documents (id, path, sha256, title, author, year, publisher, kind, status) VALUES "
                 "(1, ?, 's', 'Understanding Deep Learning', 'Simon J. D. Prince', 2024, 'MIT Press', 'pdf', 'indexed')",
                 (UGLY,))
    conn.execute("INSERT INTO chunks (id, doc_id, kind, seq, heading_path, text, token_count, block_ids, page_start, "
                 "page_end) VALUES (1, 1, 'text', 1, '[]', 'gradient descent', 2, '[]', 3, 3)")
    conn.execute("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (1, ?)", (serialize([1.0] + [0.0] * 1023, 1024),))
    conn.commit()
    hit = hybrid_search(conn, "gradient", settings, embed=lambda texts, s: [[1.0] + [0.0] * 1023 for _ in texts])[0]
    record = source_record(1, hit)
    assert (record["doc_title"], record["author"], record["year"], record["publisher"]) == (
        "Understanding Deep Learning", "Simon J. D. Prince", 2024, "MIT Press")
    assert ".pdf" not in json.dumps(record)


def test_a_new_file_is_never_titled_by_its_name(settings):
    """The worker used to start every document off with its file name as the title."""
    from app.ingest.worker import scan_library

    pdf(settings.library_dir / UGLY)
    conn = connect(settings.db_path)
    scan_library(conn, settings)
    assert row(conn, UGLY)["title"] is None
