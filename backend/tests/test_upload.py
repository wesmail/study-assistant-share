"""The library pane's Add button: PUT /api/library/files/{name} (app/upload.py)."""
import asyncio
import io
import zipfile
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db.connection import connect
from app.ingest.worker import is_ingestible
from app.main import create_app
from app.upload import UploadError, save_upload, upload_name


def pdf_bytes(text="Hello") -> bytes:
    with fitz.open() as document:
        document.new_page().insert_text((72, 72), text)
        return document.tobytes()


def deck_bytes() -> bytes:
    from pptx import Presentation

    out = io.BytesIO()
    Presentation().save(out)
    return out.getvalue()


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


def put(client, name, body, **headers):
    return client.put(f"/api/library/files/{name}", content=body,
                      headers={"Content-Type": "application/octet-stream", **headers})


def library_files(settings) -> list[str]:
    return sorted(path.name for path in settings.library_dir.iterdir())


class TestAdding:
    def test_a_pdf_arrives_byte_for_byte_under_its_own_name(self, client, settings):
        body = pdf_bytes()
        response = put(client, "My Book (2nd ed.).pdf", body)
        assert response.status_code == 200, response.text
        assert response.json()["name"] == "My Book (2nd ed.).pdf" and response.json()["size"] == len(body)
        assert (settings.library_dir / "My Book (2nd ed.).pdf").read_bytes() == body
        assert library_files(settings) == ["My Book (2nd ed.).pdf"]      # no temporary file left

    def test_a_deck_arrives_too(self, client, settings):
        body = deck_bytes()
        assert put(client, "talk.pptx", body).status_code == 200
        assert (settings.library_dir / "talk.pptx").read_bytes() == body

    def test_the_worker_takes_the_added_file_and_never_a_temporary_one(self, client, settings):
        put(client, "paper.pdf", pdf_bytes())
        assert is_ingestible(settings.library_dir / "paper.pdf", settings.library_dir)
        assert not is_ingestible(settings.library_dir / ".upload-0123abcd.pdf", settings.library_dir)


class TestRefusing:
    @pytest.mark.parametrize("name", [".hidden.pdf", "~$lock.pptx", "notes.txt", "book.PDF.exe", "a%5Cb.pdf",
                                      "a%0Ab.pdf", "x" * 200 + ".pdf"])
    def test_names_that_arent_a_plain_pdf_or_deck(self, client, settings, name):
        response = put(client, name, pdf_bytes())
        assert response.status_code == 400, response.text
        assert library_files(settings) == []

    @pytest.mark.parametrize("raw", ["../book.pdf", "/etc/book.pdf", "sub/book.pdf", "..", "", "   "])
    def test_a_path_is_never_a_name(self, raw):
        with pytest.raises(UploadError):
            upload_name(raw)

    def test_a_file_already_there_is_never_overwritten(self, client, settings):
        (settings.library_dir / "book.pdf").write_bytes(b"%PDF-1.4 the original")
        response = put(client, "book.pdf", pdf_bytes("Another"))
        assert response.status_code == 409 and "already a file called book.pdf" in response.json()["detail"]
        assert (settings.library_dir / "book.pdf").read_bytes() == b"%PDF-1.4 the original"
        assert library_files(settings) == ["book.pdf"]

    def test_the_same_content_under_another_name(self, client, settings):
        import hashlib

        body = pdf_bytes()
        conn = connect(settings.db_path)
        conn.execute("INSERT INTO documents (path, sha256, title, kind, status) VALUES (?, ?, ?, 'pdf', 'indexed')",
                     ("old.pdf", hashlib.sha256(body).hexdigest(), "Linear Algebra"))
        conn.commit()
        conn.close()
        response = put(client, "copy.pdf", body)
        assert response.status_code == 409 and "already in the library, as Linear Algebra" in response.json()["detail"]
        assert library_files(settings) == []

    def test_too_large_stops_while_streaming_whatever_content_length_says(self, settings):
        from dataclasses import replace

        small = replace(settings, upload_max_bytes=1000)

        async def chunks():
            for _ in range(10):
                yield b"%PDF-" + b"x" * 295

        with pytest.raises(UploadError) as raised:
            asyncio.run(save_upload(small, "big.pdf", chunks(), declared_length="10"))
        assert raised.value.status == 413
        assert library_files(settings) == []

    def test_a_declared_size_over_the_limit_is_refused_before_reading(self, client, settings):
        response = put(client, "big.pdf", pdf_bytes(), **{"Content-Length": str(10**12)})
        assert response.status_code in (400, 413)
        assert library_files(settings) == []

    @pytest.mark.parametrize("name, body", [
        ("fake.pdf", b"\x89PNG\r\n\x1a\n" + b"0" * 2000),
        ("fake.pptx", b"not a zip at all"),
        ("word.pptx", None),        # a zip, but not a deck
        ("empty.pdf", b""),
    ])
    def test_content_that_isnt_what_its_name_says(self, client, settings, name, body):
        if body is None:
            out = io.BytesIO()
            with zipfile.ZipFile(out, "w") as archive:
                archive.writestr("word/document.xml", "<w:document/>")
            body = out.getvalue()
        response = put(client, name, body)
        assert response.status_code == 400, response.text
        assert library_files(settings) == []

    def test_another_site_cant_add_a_file(self, client, settings):
        response = put(client, "book.pdf", pdf_bytes(), Origin="http://evil.example")
        assert response.status_code == 403
        response = put(client, "book.pdf", pdf_bytes(), **{"Sec-Fetch-Site": "cross-site"})
        assert response.status_code == 403
        assert library_files(settings) == []
