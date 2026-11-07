"""Decks in the library (roadmap item 7, step 3): app/ingest/deck.py and the worker with a deck.

LibreOffice and Docling are faked here: the render is a PDF with one page per slide made by
PyMuPDF, the slides are a dict like render.py's, and the parser writes a sidecar. The real
thing runs in test_office.py and in the gate's ingestion of the two decks.
"""
import json
from types import SimpleNamespace

import fitz
import pytest

from app.config import Settings
from app.db.connection import connect
from app.ingest import deck
from app.ingest.chunker import Block
from app.ingest.worker import STAGES, Worker, scan_library

SLIDES = {
    "title": "PowerPoint Presentation", "author": "Template Author", "created": None,
    "slides": [
        {"slide": 1, "hidden": False, "title": "About me", "lines": ["About me", "Private CV text here"],
         "equations": [], "notes": "My private notes"},
        {"slide": 2, "hidden": False, "title": "Kinematic Fitting",
         "lines": ["Kinematic Fitting", "Track Parameterization:", "y = ( 1 p , θ, φ, R, Z)"],
         "equations": ["y = ( 1 p , θ, φ, R, Z)"], "notes": ""},
        {"slide": 3, "hidden": False, "title": "Array model", "lines": ["Array model", "Spatial attention"],
         "equations": [], "notes": "We simulate a network of 16 stations around the site."},
    ],
}


def block(order, kind, text, page, **meta):
    return Block(order=order, kind=kind, text=text, page=page, meta=meta)


class TestTitle:
    def test_a_template_title_is_rejected_and_its_author_with_it(self):
        assert deck.deck_title(SLIDES) == ("About me", False)

    def test_the_first_kept_slide_when_slide_one_is_excluded(self):
        assert deck.deck_title(SLIDES, {1}) == ("Kinematic Fitting", False)

    def test_a_core_title_on_the_slides_is_kept(self):
        slides = {**SLIDES, "title": "Kinematic fitting"}
        assert deck.deck_title(slides) == ("Kinematic fitting", True)

    def test_a_placeholder_in_another_language_is_rejected(self):
        slides = {**SLIDES, "title": "Hier steht der zweizeilige Titel der Präsentation"}
        assert deck.deck_title(slides)[1] is False


class TestSlideChunks:
    def blocks(self):
        return [
            block(0, "paragraph", "About me Private CV text here", 1),
            block(1, "figure", "", 1, image_path="figures/cv.png"),
            block(2, "paragraph", "Kinematic Fitting", 2, bbox=[1, 2, 3, 4]),
            block(3, "equation", "$$\ny = (1/p, \\theta, \\phi, R, Z)\n$$", 2),
            block(6, "equation", "$$\nx^{2}\n$$", 2, unverified=True, crop="crops/7.png", text_layer="x2"),
            block(4, "figure", "Missing mass plot", 2, image_path="figures/mm.png"),
            block(5, "paragraph", "Array model", 3),
        ]

    def test_one_chunk_per_slide_with_its_text_formulas_and_blocks(self):
        chunks = deck.slide_chunks(self.blocks(), SLIDES, "Talk")
        slide2 = next(c for c in chunks if c.kind == "text" and c.page_start == 2)
        assert slide2.heading_path == ["Talk", "Kinematic Fitting"]
        assert "Slide 2\nKinematic Fitting\nTrack Parameterization:" in slide2.text
        assert "$$\ny = (1/p, \\theta, \\phi, R, Z)\n$$" in slide2.text and "$$$$" not in slide2.text
        assert "[unverified formula, see page 2]\n$$\nx^{2}\n$$" in slide2.text
        assert slide2.block_orders == [2, 3, 6]
        assert slide2.meta == {"slide": 2, "unverified_formulas": [
            {"page": 2, "crop": "crops/7.png", "text_layer": "x2"}]}

    def test_speaker_notes_get_their_own_chunk_on_their_slide(self):
        chunks = deck.slide_chunks(self.blocks(), SLIDES, "Talk")
        notes = [c for c in chunks if c.meta.get("speaker_notes")]
        assert [c.page_start for c in notes] == [1, 3]
        assert notes[1].heading_path == ["Talk", "Array model", "Speaker notes"]
        assert "16 stations" in notes[1].text and notes[1].block_orders == []

    def test_figures_as_for_a_book(self):
        chunks = deck.slide_chunks(self.blocks(), SLIDES, "Talk")
        figures = [c for c in chunks if c.kind == "figure"]
        assert [(c.page_start, c.meta["image_path"]) for c in figures] == [(2, "figures/mm.png")]

    def test_an_excluded_slide_leaves_nothing(self):
        chunks = deck.slide_chunks(self.blocks(), SLIDES, "Talk", excluded={1})
        assert all(c.page_start != 1 for c in chunks)
        assert not any("CV" in c.text or "private notes" in c.text for c in chunks)
        assert deck.kept_blocks(self.blocks(), {1}) == self.blocks()[2:]


def test_pdf_path_for_a_book_and_a_deck(tmp_path):
    settings = Settings(library_dir=tmp_path / "lib", data_dir=tmp_path / "data")
    book = {"path": "a.pdf", "kind": "pdf", "sha256": "aa"}
    talk = {"path": "talks/b.pptx", "kind": "pptx", "sha256": "bb"}
    assert deck.pdf_path(settings, _Row(book)) == tmp_path / "lib" / "a.pdf"
    assert deck.pdf_path(settings, _Row(talk)) == tmp_path / "data" / "renders" / "bb.pdf"


class _Row(dict):
    """Like sqlite3.Row: indexable by name, with keys()."""


# --- With the worker ---------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_real_models(monkeypatch):
    import app.ingest.summarize as tree

    def fake_ask(settings, model, think=False, timeout=900, busy=None):
        return lambda prompt: "A summary."
    monkeypatch.setattr(tree, "ollama_ask", fake_ask)
    monkeypatch.setattr(tree, "ollama_stream_ask", fake_ask)

    def no_describer(ctx, rows):
        raise AssertionError("a test reached the figure describer")
    monkeypatch.setattr("app.ingest.worker.default_describe_figures", no_describer)


@pytest.fixture
def settings(tmp_path):
    (tmp_path / "library").mkdir()
    return Settings(library_dir=tmp_path / "library", data_dir=tmp_path / "data", embed_batch=4)


@pytest.fixture
def conn(settings):
    connection = connect(settings.db_path)
    yield connection
    connection.close()


@pytest.fixture
def fake_office(monkeypatch):
    """render_pdf writes a PDF with one page per slide; read_slides returns SLIDES."""
    from app import office
    calls = {"render": 0, "slides": 0}

    async def render_pdf(settings, source, target, pages=None):
        calls["render"] += 1
        pdf = fitz.open()
        for slide in SLIDES["slides"][:pages or len(SLIDES["slides"])]:
            pdf.new_page(width=960, height=540).insert_text((72, 72), " ".join(slide["lines"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        pdf.save(target)
        return pdf.page_count

    async def read_slides(settings, source):
        calls["slides"] += 1
        return SLIDES
    monkeypatch.setattr(office, "render_pdf", render_pdf)
    monkeypatch.setattr(office, "read_slides", read_slides)
    return calls


def deck_parser(pdf, out_dir):
    """Docling on the render: a paragraph per slide, a figure on slides 1 and 2."""
    with fitz.open(pdf) as rendered:
        pages = rendered.page_count
    records = []
    for page in range(1, pages + 1):
        records.append({"order": len(records), "kind": "paragraph", "text": f"Render text {page}",
                        "page": page, "bbox": [10, 20, 30, 40]})
        if page <= 2:
            records.append({"order": len(records), "kind": "figure", "text": "", "page": page,
                            "image_path": f"figures/p{page}.png", "bbox": [50, 60, 70, 80]})
    (out_dir / "blocks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return {"title": None, "pages": pages}


def embed(texts):
    return [[1.0] + [0.0] * 1023 for _ in texts]


def add_deck(settings, name="talk.pptx", exclude=None, title=None):
    (settings.library_dir / name).write_bytes(b"PK\x03\x04 a deck")
    if exclude or title:
        entry = {}
        if title:
            entry["title"] = title
        if exclude:
            entry["exclude_pages"] = exclude
        import yaml
        (settings.library_dir / "metadata.yaml").write_text(yaml.safe_dump({name: entry}, allow_unicode=True),
                                                            encoding="utf-8")


def ingest(conn, settings):
    scan_library(conn, settings)
    Worker(conn, settings, deck_parser, embed).drain()
    return conn.execute("SELECT * FROM documents").fetchone()


def test_a_deck_goes_through_as_its_render(conn, settings, fake_office):
    add_deck(settings)
    row = ingest(conn, settings)
    assert row["kind"] == "pptx" and row["status"] == "indexed" and row["pages"] == 3
    assert [r[0] for r in conn.execute("SELECT status FROM jobs WHERE doc_id = ? ORDER BY stage_order",
                                       (row["id"],))] == ["done"] * len(STAGES)
    assert deck.render_path(settings, row["sha256"]).is_file()
    assert (settings.parsed_dir / row["sha256"] / "slides.json").is_file()
    assert row["title"] == "About me"      # the template title rejected, the first slide's taken
    assert row["author"] is None           # and the template's author with it
    texts = [r["text"] for r in conn.execute("SELECT text FROM chunks WHERE kind = 'text' ORDER BY seq")]
    assert any("Track Parameterization:" in t for t in texts)
    assert any("Speaker notes, slide 3" in t and "16 stations" in t for t in texts)
    assert (settings.library_dir / "talk.pptx").read_bytes() == b"PK\x03\x04 a deck"


def test_an_excluded_slide_is_nowhere(conn, settings, fake_office):
    add_deck(settings, exclude=[1], title="My talk")
    row = ingest(conn, settings)
    assert row["title"] == "My talk"
    for table in ("blocks", "figures"):
        assert not conn.execute(f"SELECT 1 FROM {table} WHERE page = 1").fetchone(), table
    assert not conn.execute("SELECT 1 FROM chunks WHERE page_start = 1 OR page_end = 1").fetchone()
    every = " ".join(r[0] for r in conn.execute("SELECT text FROM chunks"))
    assert "CV" not in every and "private notes" not in every and "Render text 1" not in every
    hits = conn.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'private'").fetchall()
    assert hits == []
    assert conn.execute("SELECT count(*) FROM figures").fetchone()[0] == 1      # slide 2's only


def test_a_render_that_doesnt_match_the_slides_fails_the_parse(conn, settings, fake_office, monkeypatch):
    from app import office

    async def short_render(settings, source, target):
        pdf = fitz.open()
        pdf.new_page()
        target.parent.mkdir(parents=True, exist_ok=True)
        pdf.save(target)
        return 1
    monkeypatch.setattr(office, "render_pdf", short_render)
    add_deck(settings)
    scan_library(conn, settings)
    Worker(conn, settings, deck_parser, embed).drain()
    parse = conn.execute("SELECT status, error FROM jobs WHERE stage = 'parse'").fetchone()
    assert parse["status"] == "failed" and "1 pages for 3 slides" in parse["error"]


def test_the_viewer_gets_the_render(conn, settings, fake_office, tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    add_deck(settings)
    row = ingest(conn, settings)
    client = TestClient(create_app(settings, embed=lambda texts, s: embed(texts), frontend=tmp_path / "none"),
                        base_url="http://localhost")
    response = client.get(f"/api/documents/{row['id']}/file")
    assert response.status_code == 200 and response.headers["content-type"] == "application/pdf"
    assert response.content == deck.render_path(settings, row["sha256"]).read_bytes()


def test_hosted_models_are_refused_on_excluded_pages(conn, settings, fake_office):
    add_deck(settings, exclude=[1])
    row = ingest(conn, settings)
    hosted, local = SimpleNamespace(provider="anthropic"), SimpleNamespace(provider="ollama")
    assert "kept away from hosted models" in deck.refuse_excluded(conn, settings, hosted, row["id"], 1, 2)
    assert deck.refuse_excluded(conn, settings, hosted, row["id"], 2, 3) is None
    assert deck.refuse_excluded(conn, settings, local, row["id"], 1, 1) is None
    assert deck.refuse_excluded(conn, settings, SimpleNamespace(provider="openrouter"), row["id"], 1, 1)


def test_notes_on_an_excluded_page_are_not_searched(conn, settings, fake_office):
    add_deck(settings, exclude=[1])
    row = ingest(conn, settings)
    with conn:
        for page in (1, 2):
            conn.execute("INSERT INTO notes (doc_id, page, body_md, tags) VALUES (?, ?, ?, '[]')",
                         (row["id"], page, f"note on page {page}"))
    kept_out = deck.excluded_note_ids(conn, settings)
    pages = {r["id"]: r["page"] for r in conn.execute("SELECT id, page FROM notes")}
    assert {pages[i] for i in kept_out} == {1}


def test_details_form_keeps_exclude_pages_as_a_list(conn, settings, fake_office):
    import yaml
    from app import library
    add_deck(settings, exclude=[1], title="My talk")
    row = ingest(conn, settings)
    library.save_details(conn, settings, row, {"title": "Renamed talk", "author": "Me", "year": 2023},
                         lambda texts, s: embed(texts))
    entry = yaml.safe_load((settings.library_dir / "metadata.yaml").read_text())["talk.pptx"]
    assert entry["exclude_pages"] == [1] and entry["title"] == "Renamed talk"
    assert library.suggestion(settings, row)["title"] == "Renamed talk"


def test_explain_and_suggestions_refuse_hosted_on_an_excluded_page(conn, settings, fake_office, tmp_path,
                                                                     monkeypatch):
    """Through the API: nothing of a kept-out page reaches a hosted model, and a page next
    to it still works (the model call itself is faked, so nothing leaves here either)."""
    from fastapi.testclient import TestClient
    from app import explain, main, suggest
    add_deck(settings, exclude=[1])
    row = ingest(conn, settings)
    sent = []

    async def fake_explain(settings, option, text, *args, **kwargs):
        sent.append(("explain", text))
        return {"items": []}

    async def fake_suggest(conn, settings, option, doc_id, first, last, *args, **kwargs):
        sent.append(("suggest", first, last))
        return {"suggestions": []}
    monkeypatch.setattr(explain, "explain", fake_explain)
    monkeypatch.setattr(suggest, "suggest", fake_suggest)
    client = TestClient(main.create_app(settings, embed=lambda texts, s: embed(texts), frontend=tmp_path / "none"),
                        base_url="http://localhost")
    passage = {"doc_id": row["id"], "page": 1, "rects": [[72, 60, 300, 80]], "quote": "About me"}
    refused = client.post("/api/explain", json={"passage": passage, "option": "deep", "kind": "code"})
    assert refused.status_code == 409 and "kept away from hosted models" in refused.json()["detail"]
    refused = client.post("/api/suggestions", json={"doc_id": row["id"], "first_page": 1, "last_page": 2,
                                                    "instruction": "highlight key ideas"})
    assert refused.status_code == 409
    assert sent == []
    allowed = client.post("/api/suggestions", json={"doc_id": row["id"], "first_page": 2, "last_page": 3,
                                                    "instruction": "highlight key ideas"})
    assert allowed.status_code == 200 and sent == [("suggest", 2, 3)]


def test_highlights_follow_an_edited_deck_and_go_into_the_annotated_copy(conn, settings, fake_office,
                                                                          monkeypatch, tmp_path):
    """A highlight on slide 3; then a slide is put in front, so that text is on slide 4 of the
    new render. Re-anchoring (after the index stage) moves it; the annotated copy is written
    from the render, into outputs/annotated, never the library."""
    from app import annotated, annotations, office
    settings = Settings(**{**settings.__dict__, "outputs_dir": tmp_path / "outputs"})
    edited = {"title": SLIDES["title"], "author": "", "created": None,
              "slides": [{"slide": 1, "hidden": False, "title": "New first slide", "lines": ["New first slide"],
                          "equations": [], "notes": ""}]
              + [{**s, "slide": s["slide"] + 1} for s in SLIDES["slides"]]}

    def slides_for(source):
        return edited if b"v2" in source.read_bytes() else SLIDES

    async def render_pdf(settings, source, target):
        pdf = fitz.open()
        for slide in slides_for(source)["slides"]:
            pdf.new_page(width=960, height=540).insert_text((72, 72), " ".join(slide["lines"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        pdf.save(target)
        return pdf.page_count

    async def read_slides(settings, source):
        return slides_for(source)
    monkeypatch.setattr(office, "render_pdf", render_pdf)
    monkeypatch.setattr(office, "read_slides", read_slides)

    add_deck(settings)
    row = ingest(conn, settings)
    with fitz.open(deck.render_path(settings, row["sha256"])) as pdf:
        box = pdf[2].search_for("Spatial attention")[0]
    made = annotations.create(conn, settings, row["id"], 3, [[box.x0, box.y0, box.x1, box.y1]], "Spatial attention")
    assert made["snapped"] and made["page"] == 3

    (settings.library_dir / "talk.pptx").write_bytes(b"PK\x03\x04 a deck v2")
    scan_library(conn, settings)
    Worker(conn, settings, deck_parser, embed).drain()
    moved = conn.execute("SELECT * FROM annotations WHERE id = ?", (made["id"],)).fetchone()
    assert moved["page"] == 4 and moved["anchor_status"] in ("ok", "moved")
    new_sha = conn.execute("SELECT sha256 FROM documents WHERE id = ?", (row["id"],)).fetchone()[0]
    assert new_sha != row["sha256"] and deck.render_path(settings, row["sha256"]).is_file()  # old render kept

    copy = annotated.write_copy(conn, settings, row["id"])
    out = annotated.copy_path(settings, row["id"])
    assert copy["highlights"] == 1 and out.parent == tmp_path / "outputs" / "annotated"
    with fitz.open(out) as pdf:
        assert [len(list(page.annots())) for page in pdf] == [0, 0, 0, 1]
    assert (settings.library_dir / "talk.pptx").read_bytes() == b"PK\x03\x04 a deck v2"


def _searchable(conn, settings, words):
    from app.retrieval.search import search_chunks
    hits = search_chunks(conn, words, settings, embed=lambda texts, s: embed(texts))
    return [hit for hit in hits if hit.page_start == 1 or "private" in hit.text.lower() or "Render text 1" in hit.text]


def test_a_page_excluded_after_ingestion_is_gone_before_the_next_search(conn, settings, fake_office):
    """Review R2-S1: exclude_pages added to an indexed deck left the page searchable, so it
    could be given to Deep or Qwen. The reader edits metadata.yaml while the app runs."""
    add_deck(settings, title="My talk")
    ingest(conn, settings)
    assert _searchable(conn, settings, "private CV notes")        # before: slide 1 is found
    add_deck(settings, exclude=[1], title="My talk")              # the hand edit
    assert _searchable(conn, settings, "private CV notes") == []
    for table in ("blocks", "figures"):
        assert not conn.execute(f"SELECT 1 FROM {table} WHERE page = 1").fetchone(), table
    assert not conn.execute("SELECT 1 FROM chunks WHERE page_start <= 1 AND page_end >= 1").fetchone()
    assert conn.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'private'").fetchall() == []
    assert conn.execute("SELECT count(*) FROM chunk_vec WHERE chunk_id NOT IN (SELECT id FROM chunks)").fetchone()[0] == 0
    assert conn.execute("SELECT 1 FROM chunks WHERE page_start = 2").fetchone()      # the rest stays


def test_a_page_excluded_while_the_app_was_closed_is_gone_at_start(conn, settings, fake_office, tmp_path):
    from app.main import create_app
    add_deck(settings, title="My talk")
    ingest(conn, settings)
    add_deck(settings, exclude=[1], title="My talk")
    create_app(settings, embed=lambda texts, s: embed(texts), frontend=tmp_path / "none")
    assert not conn.execute("SELECT 1 FROM chunks WHERE page_start <= 1 AND page_end >= 1").fetchone()


@pytest.mark.parametrize("broken", [
    "talk.pptx:\n  title: \"My talk\n  exclude_pages: [1]\n",        # an unclosed quote
    "talk.pptx:\n  exclude_pages: one\n",                            # not a page number
    "talk.pptx: exclude_pages [1]\n",                                 # details not under the name
])
def test_a_broken_metadata_file_fails_closed_and_the_app_still_starts(conn, settings, fake_office, tmp_path, broken):
    """Review R2-C7: one typo stopped the app and the worker. Read as empty, it would also drop
    exclude_pages, so nothing that depends on it goes ahead: the reader is told instead."""
    from fastapi.testclient import TestClient
    from app.ingest import metadata
    from app.main import create_app

    add_deck(settings, exclude=[1], title="My talk")
    row = ingest(conn, settings)
    conn.execute("INSERT INTO notes (body_md, tags, doc_id, page, quote) VALUES ('on slide 2', '[]', ?, 2, 'q')", (row["id"],))
    conn.commit()
    (settings.library_dir / "metadata.yaml").write_text(broken, encoding="utf-8")
    with pytest.raises(metadata.MetadataError):
        metadata.read_overrides(settings.library_dir)

    client = TestClient(create_app(settings, embed=lambda texts, s: embed(texts), frontend=tmp_path / "none"),
                        base_url="http://localhost")
    listing = client.get("/api/documents").json()
    assert "metadata.yaml" in listing["metadata_problem"] and len(listing["documents"]) == 1
    hosted, local = SimpleNamespace(provider="anthropic"), SimpleNamespace(provider="ollama")
    assert "no page goes to a hosted model" in deck.refuse_excluded(conn, settings, hosted, row["id"], 2, 2)
    assert deck.refuse_excluded(conn, settings, local, row["id"], 2, 2) is None
    assert deck.excluded_note_ids(conn, settings) == {r[0] for r in conn.execute("SELECT id FROM notes")}
    (settings.library_dir / "other.pptx").write_bytes(b"PK\x03\x04 another deck")
    scan_library(conn, settings)
    assert Worker(conn, settings, deck_parser, embed).drain() == 0          # waits, attempts untouched
    assert conn.execute("SELECT max(attempts) FROM jobs WHERE status = 'pending'").fetchone()[0] == 0
