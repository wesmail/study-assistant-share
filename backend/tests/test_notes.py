"""Notes (roadmap item 4, step 2): anchored like highlights, Markdown with LaTeX, tags, export."""
from __future__ import annotations

import io
import zipfile

import pytest
import yaml

from app import notes
from tests.test_annotations import selection, setup  # noqa: F401  (the fixture)


def passage(path, first, last, doc_id=1, page=1):
    rects, quote, _ = selection(path, first, last)
    return {"doc_id": doc_id, "page": page, "rects": rects, "quote": quote}


class TestNotes:
    def test_anchored_to_a_passage_snapped_like_a_highlight(self, setup):  # noqa: F811
        settings, conn, path = setup
        note = notes.create(conn, settings, "Proper time is $\\tau = t/\\gamma$.", ["relativity", "#time"],
                            passage(path, "clock", "particle"))
        assert note["snapped"] is True
        assert note["quote"] == "clock moving with the particle"
        assert note["page"] == 1 and len(note["rects"]) == 2
        assert note["prefix"].endswith("measured by a")
        assert note["tags"] == ["relativity", "time"]
        assert note["doc_title"] == ""  # the test document has no title

    def test_a_free_note(self, setup):  # noqa: F811
        settings, conn, _ = setup
        note = notes.create(conn, settings, "  A thought.  ")
        assert note["doc_id"] is None and note["page"] is None and note["quote"] == "" and note["rects"] == []
        assert note["body_md"] == "A thought."

    def test_refused(self, setup):  # noqa: F811
        settings, conn, path = setup
        with pytest.raises(notes.Refused, match="some text"):
            notes.create(conn, settings, "  \n")
        with pytest.raises(notes.Refused, match="At most"):
            notes.create(conn, settings, "x" * (notes.BODY_CHARS + 1))
        with pytest.raises(notes.Refused, match="tags"):
            notes.create(conn, settings, "x", [f"t{i}" for i in range(notes.MAX_TAGS + 1)])
        with pytest.raises(notes.Refused, match="no page"):
            notes.create(conn, settings, "x", [], {**passage(path, "clock", "particle"), "page": 7})

    def test_tags_as_obsidian_takes_them(self):
        assert notes.clean_tags(["#Lorentz boost", "lorentz-boost", " x/y ", "a,b!", ""]) == ["Lorentz-boost", "x/y", "ab"]

    def test_edit_filter_and_delete(self, setup):  # noqa: F811
        settings, conn, path = setup
        first = notes.create(conn, settings, "About the clock", ["clocks"], passage(path, "clock", "particle"))
        second = notes.create(conn, settings, "Invariance 50% of the time", ["lorentz"])
        assert [n["id"] for n in notes.listing(conn)] == [second["id"], first["id"]]
        assert [n["id"] for n in notes.listing(conn, doc_id=1)] == [first["id"]]
        assert [n["id"] for n in notes.listing(conn, query="clock particle")] == [first["id"]]  # body and quote
        assert [n["id"] for n in notes.listing(conn, query="LORENTZ")] == [second["id"]]  # tags, any case
        assert [n["id"] for n in notes.listing(conn, query="50%")] == [second["id"]]
        assert notes.listing(conn, query="5_%") == []  # LIKE wildcards are literal
        changed = notes.update(conn, first["id"], body_md="Edited", tags=["a b"])
        assert changed["body_md"] == "Edited" and changed["tags"] == ["a-b"]
        assert changed["quote"] == first["quote"]
        notes.delete(conn, first["id"])
        with pytest.raises(notes.NotFound):
            notes.get(conn, first["id"])
        with pytest.raises(notes.NotFound):
            notes.update(conn, first["id"], body_md="x")

    def test_a_note_outlives_its_book(self, setup):  # noqa: F811
        settings, conn, path = setup
        conn.execute("UPDATE documents SET title = 'Relativity' WHERE id = 1")
        note = notes.create(conn, settings, "Keep me", [], passage(path, "clock", "particle"))
        conn.execute("DELETE FROM documents WHERE id = 1")
        kept = notes.get(conn, note["id"])
        assert kept["doc_id"] is None and kept["doc_title"] == "Relativity"
        assert kept["quote"] == "clock moving with the particle" and kept["page"] == 1


class TestExport:
    def test_obsidian_math(self):
        text = "Inline \\(a^2\\) and display \\[\\int f\\] and $b$, but `\\(code\\)` and\n```\n\\[x\\]\n```"
        assert notes.dollar_math(text) == "Inline $a^2$ and display $$\\int f$$ and $b$, but `\\(code\\)` and\n```\n\\[x\\]\n```"

    def test_one_file_per_document(self, setup):  # noqa: F811
        settings, conn, path = setup
        conn.execute("UPDATE documents SET title = 'Relativity: A/B', author = 'Einstein', year = 1905 WHERE id = 1")
        notes.create(conn, settings, "Second, on the second line: \\(\\tau\\)", ["time"], passage(path, "It", "transformations."))
        notes.create(conn, settings, "First", ["clocks", "time"], passage(path, "clock", "particle"))
        notes.create(conn, settings, "Free", ["misc"])
        files = notes.export_files(conn)
        assert set(files) == {"Relativity- A-B.md", "Notes without a document.md"}
        text = files["Relativity- A-B.md"]
        head, body = text.split("---\n", 2)[1:]
        front = yaml.safe_load(head)
        assert front["title"] == "Relativity: A/B" and front["authors"] == "Einstein" and front["year"] == 1905
        assert front["source"] == "book.pdf" and front["tags"] == ["study-notes", "clocks", "time"] and front["notes"] == 2
        assert "## p. 1 · note" in body and "> clock moving with the particle" in body
        assert "$\\tau$" in body and "#clocks #time" in body
        assert body.index("First") < body.index("Second")  # by page, then down the page

    def test_a_removed_books_notes_keep_its_title(self, setup):  # noqa: F811
        settings, conn, path = setup
        conn.execute("UPDATE documents SET title = 'Relativity' WHERE id = 1")
        notes.create(conn, settings, "Keep me", [], passage(path, "clock", "particle"))
        conn.execute("DELETE FROM documents WHERE id = 1")
        text = notes.export_files(conn)["Notes without a document.md"]
        assert "> clock moving with the particle\n> — Relativity" in text and "## p. 1" in text

    def test_zip_through_the_api(self, setup):  # noqa: F811
        from tests.test_chat import FakeOllama, client_for

        settings, conn, path = setup
        client = client_for(settings, FakeOllama())
        made = client.post("/api/notes", json={"body_md": "Note $x$", "tags": ["t"],
                                               "passage": passage(path, "clock", "particle")})
        assert made.status_code == 200 and made.json()["snapped"] is True
        note_id = made.json()["id"]
        assert client.post("/api/notes", json={"body_md": " "}).status_code == 409
        assert client.post("/api/notes", json={"body_md": "x", "passage": {**passage(path, "clock", "particle"), "doc_id": 9}}).status_code == 404
        assert [n["id"] for n in client.get("/api/notes", params={"doc_id": 1}).json()["notes"]] == [note_id]
        assert client.get("/api/notes", params={"q": "nothing"}).json()["notes"] == []
        assert client.patch(f"/api/notes/{note_id}", json={"tags": ["u v"]}).json()["tags"] == ["u-v"]
        response = client.get("/api/notes/export")
        assert response.headers["content-type"] == "application/zip"
        assert "attachment" in response.headers["content-disposition"]
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert archive.namelist() == ["Study notes/book.pdf.md"]
            assert "Note $x$" in archive.read("Study notes/book.pdf.md").decode()
        assert client.delete(f"/api/notes/{note_id}").json() == {"ok": True}
        assert client.delete(f"/api/notes/{note_id}").status_code == 404
