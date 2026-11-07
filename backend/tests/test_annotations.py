"""Highlights (roadmap item 4, step 1): snapped to PyMuPDF's words, saved, changed, removed."""
from __future__ import annotations

import hashlib

import fitz
import pytest

from app import annotations
from app.config import Settings
from app.db.connection import connect

LINES = ["The proper time is the time measured by a clock", "moving with the particle along its world line.",
         "It is invariant under Lorentz transformations."]


@pytest.fixture
def setup(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    pdf = fitz.open()
    page = pdf.new_page(width=400, height=300)
    for number, line in enumerate(LINES):
        page.insert_text((40, 60 + 20 * number), line, fontsize=11)
    pdf.new_page(width=400, height=300).insert_text((40, 60), "Second page.", fontsize=11)
    pdf.save(library / "book.pdf")
    settings = Settings(data_dir=tmp_path / "data", library_dir=library, outputs_dir=tmp_path / "outputs")
    conn = connect(settings.db_path)
    conn.execute("INSERT INTO documents (id, path, sha256, kind, pages, status) VALUES (1, 'book.pdf', 'x', 'pdf', 2, 'indexed')")
    conn.commit()
    yield settings, conn, library / "book.pdf"
    conn.close()


def words_of(path, page=1):
    with fitz.open(path) as pdf:
        return pdf[page - 1].get_text("words", sort=False)


def selection(path, first: str, last: str, pad: float = 2.0):
    """Boxes like pdf.js gives: a bit taller than the words, one per line, from `first` to `last`."""
    words = words_of(path)
    start = next(i for i, w in enumerate(words) if w[4] == first)
    end = next(i for i, w in enumerate(words) if w[4] == last and i >= start)
    chosen = words[start:end + 1]
    lines = {}
    for w in chosen:
        lines.setdefault((w[5], w[6]), []).append(w)
    rects = [[min(w[0] for w in ws), min(w[1] for w in ws) - pad, max(w[2] for w in ws), max(w[3] for w in ws) + pad]
             for ws in lines.values()]
    return rects, "\n".join(" ".join(w[4] for w in ws) for ws in lines.values()), chosen


class TestSnap:
    def test_snaps_a_two_line_selection_to_the_words(self, setup):
        settings, conn, path = setup
        rects, quote, chosen = selection(path, "clock", "particle")
        saved = annotations.create(conn, settings, 1, 1, rects, quote)
        assert saved["snapped"] is True
        assert saved["quote"] == "clock moving with the particle"
        assert len(saved["rects"]) == 2
        # The boxes are the words' own, not the taller selection boxes.
        assert saved["rects"][0][1] == pytest.approx(chosen[0][1], abs=0.01)
        assert saved["rects"][0][3] == pytest.approx(chosen[0][3], abs=0.01)
        assert saved["prefix"].endswith("measured by a")
        assert saved["suffix"].startswith("along its world")
        assert saved["bbox"][0] == pytest.approx(min(r[0] for r in saved["rects"]))

    def test_a_word_cut_in_half_is_taken_whole(self, setup):
        settings, conn, path = setup
        rects, _, chosen = selection(path, "proper", "time")
        word = chosen[0]
        rects[0][0] = word[0] + (word[2] - word[0]) * 0.3  # starts inside "proper"
        saved = annotations.create(conn, settings, 1, 1, rects, "oper time")
        assert saved["snapped"] is True and saved["quote"] == "proper time"

    def test_more_words_than_selected_dont_snap(self):
        assert annotations.reads_like(["proper", "time"], "oper ti")
        assert not annotations.reads_like(["the", "proper", "time", "is"], "proper time")
        assert not annotations.reads_like(["proper"], "proper time is invariant")
        assert annotations.reads_like(["ﬁeld", "theory"], "field theory")

    def test_glyphs_read_differently_snap_when_the_boxes_agree(self, setup):
        # pdf.js reads the floor brackets of the Sparse Transformers paper as "b" and "c".
        settings, conn, path = setup
        rects, _, _ = selection(path, "It", "transformations.")
        saved = annotations.create(conn, settings, 1, 1, rects, "It is invarianb under Lorentc transformationc.")
        assert saved["snapped"] is True and saved["quote"] == "It is invariant under Lorentz transformations."
        # The same text over the wrong boxes isn't snapped.
        elsewhere, _, _ = selection(path, "The", "by")
        wrong = annotations.create(conn, settings, 1, 1, elsewhere, "It is invarianb under Lorentc transformationc.")
        assert wrong["snapped"] is False

    def test_boxes_iou(self):
        assert annotations.boxes_iou([(0, 0, 10, 10)], [(0, 0, 10, 10)]) == 1.0
        assert annotations.boxes_iou([(0, 0, 10, 10)], [(5, 0, 15, 10)]) == pytest.approx(1 / 3)
        assert annotations.boxes_iou([(0, 0, 10, 10)], [(20, 0, 30, 10)]) == 0.0

    def test_accents_are_left_out_of_the_comparison(self):
        # MML p. 60: the tilde of B̃ is a word of its own on the line above, in front.
        assert annotations.reads_like(["˜", "B", ":", "V", "→V", "that", "maps", "coordi-"], "B : V → V that maps coordi-")
        # Rubbia p. 18: combining arrows over W.
        assert annotations.reads_like(["∂µ", "⃗W", "ν"], "∂µ W ν")
        assert annotations.squeeze("é ˜ x⃗") == "ex"

    def test_ligatures_and_spaces_dont_stop_a_snap(self):
        assert annotations.similarity("deﬁnition of\nproper time", "definition of proper time") == 1.0
        assert annotations.similarity("propertime", "proper time") == 1.0

    def test_keeps_the_browser_boxes_when_the_words_dont_match(self, setup):
        settings, conn, path = setup
        rects, _, _ = selection(path, "clock", "particle")
        saved = annotations.create(conn, settings, 1, 1, rects, "something else entirely")
        assert saved["snapped"] is False
        assert saved["quote"] == "something else entirely"
        assert saved["rects"] == [[round(v, 2) for v in r] for r in rects]
        assert saved["prefix"] == "" and saved["suffix"] == ""

    def test_boxes_are_clamped_to_the_page(self, setup):
        settings, conn, _ = setup
        saved = annotations.create(conn, settings, 1, 1, [[-50, -10, 1000, 20]], "nothing here")
        assert saved["rects"] == [[0, 0, 400, 20]]

    def test_the_original_file_is_never_written(self, setup):
        settings, conn, path = setup
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        rects, quote, _ = selection(path, "It", "transformations.")
        annotations.create(conn, settings, 1, 1, rects, quote)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before


class TestRefused:
    def test_bad_input(self, setup):
        settings, conn, _ = setup
        box = [[40, 40, 100, 70]]
        with pytest.raises(annotations.Refused, match="colour"):
            annotations.create(conn, settings, 1, 1, box, "x", color="purple")
        with pytest.raises(annotations.Refused, match="selected text"):
            annotations.create(conn, settings, 1, 1, box, "  \n ")
        with pytest.raises(annotations.Refused, match="no page 3"):
            annotations.create(conn, settings, 1, 3, box, "x")
        with pytest.raises(annotations.Refused, match="four numbers"):
            annotations.create(conn, settings, 1, 1, [[1, 2, 3]], "x")
        with pytest.raises(annotations.Refused, match="outside the page"):
            annotations.create(conn, settings, 1, 1, [[500, 500, 600, 600]], "x")
        with pytest.raises(annotations.NotFound):
            annotations.create(conn, settings, 9, 1, box, "x")

    def test_a_path_outside_the_library(self, setup):
        settings, conn, _ = setup
        conn.execute("INSERT INTO documents (id, path, sha256, kind, pages, status) VALUES (2, '../x.pdf', 'y', 'pdf', 1, 'indexed')")
        with pytest.raises(annotations.NotFound):
            annotations.create(conn, settings, 2, 1, [[0, 0, 10, 10]], "x")


class TestChangeAndRemove:
    def test_colour_note_status_then_delete(self, setup):
        settings, conn, path = setup
        rects, quote, _ = selection(path, "proper", "time")
        saved = annotations.create(conn, settings, 1, 1, rects, quote)
        changed = annotations.update(conn, saved["id"], color="green", body="see eq. 2.3")
        assert changed["color"] == "green" and changed["body"] == "see eq. 2.3"
        assert annotations.update(conn, saved["id"], body="")["body"] is None
        with pytest.raises(annotations.Refused):
            annotations.update(conn, saved["id"], status="maybe")
        assert [a["id"] for a in annotations.listing(conn, 1)] == [saved["id"]]
        annotations.delete(conn, saved["id"])
        assert annotations.listing(conn, 1) == []
        with pytest.raises(annotations.NotFound):
            annotations.delete(conn, saved["id"])

    def test_listed_by_page(self, setup):
        settings, conn, path = setup
        annotations.create(conn, settings, 1, 2, [[38, 45, 120, 65]], "Second page.")
        rects, quote, _ = selection(path, "proper", "time")
        annotations.create(conn, settings, 1, 1, rects, quote)
        assert [a["page"] for a in annotations.listing(conn, 1)] == [1, 2]

    def test_removing_the_document_removes_its_highlights(self, setup):
        settings, conn, path = setup
        rects, quote, _ = selection(path, "proper", "time")
        annotations.create(conn, settings, 1, 1, rects, quote)
        conn.execute("DELETE FROM documents WHERE id = 1")
        assert conn.execute("SELECT count(*) FROM annotations").fetchone()[0] == 0


class TestThroughTheApi:
    def test_create_list_change_delete(self, setup):
        from tests.test_chat import FakeOllama, client_for

        settings, conn, path = setup
        client = client_for(settings, FakeOllama())
        rects, quote, _ = selection(path, "clock", "particle")
        created = client.post("/api/annotations", json={"doc_id": 1, "page": 1, "rects": rects, "quote": quote})
        assert created.status_code == 200 and created.json()["snapped"] is True
        annotation_id = created.json()["id"]
        assert [a["id"] for a in client.get("/api/annotations", params={"doc_id": 1}).json()["annotations"]] == [annotation_id]
        assert client.patch(f"/api/annotations/{annotation_id}", json={"color": "blue"}).json()["color"] == "blue"
        assert client.patch(f"/api/annotations/{annotation_id}", json={"color": "red"}).status_code == 409
        assert client.post("/api/annotations", json={"doc_id": 1, "page": 0, "rects": rects, "quote": quote}).status_code == 422
        assert client.post("/api/annotations", json={"doc_id": 5, "page": 1, "rects": rects, "quote": quote}).status_code == 404
        assert client.delete(f"/api/annotations/{annotation_id}").json() == {"ok": True}
        assert client.delete(f"/api/annotations/{annotation_id}").status_code == 404


def test_an_old_table_gets_the_new_columns(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE annotations (id INTEGER PRIMARY KEY, doc_id INTEGER, page INTEGER, bbox TEXT,"
                " quote TEXT, kind TEXT, color TEXT, body TEXT, origin TEXT, status TEXT, created_at TEXT)")
    old.close()
    conn = connect(path)
    assert {"rects", "prefix", "suffix"} <= {row[1] for row in conn.execute("PRAGMA table_info(annotations)")}
    conn.close()


class TestLineBox:
    def test_a_tall_glyph_is_trimmed_to_the_line(self):
        # "O(n p√n) with": the root sign's box reaches up into the line above.
        boxes = [(10, 100, 20, 110), (22, 100, 30, 110), (32, 88, 45, 111), (47, 100, 60, 110)]
        assert annotations.line_box(boxes) == (10, 100, 60, 110)

    def test_a_stacked_fraction_keeps_its_parts(self):
        # MML p. 167, "dc/da = 1 = dc/db": numerators and denominators are ordinary words at other heights.
        boxes = [(207, 138, 218, 149), (206, 153, 218, 164), (223, 146, 231, 156), (234, 146, 240, 156),
                 (243, 146, 251, 156), (255, 138, 267, 149), (255, 153, 267, 164)]
        assert annotations.line_box(boxes) == (206, 138, 267, 164)
