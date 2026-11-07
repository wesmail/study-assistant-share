"""Original figures in generated files: a source's figure goes in as its crop, under a name the
app gives it, never redrawn from its description (bug report, Nov 7, 2026)."""
import asyncio
from pathlib import Path

import fitz
import pytest

from app.config import Settings
from app.llm.prompts import TOOLS_PROMPT
from app.retrieval.search import Hit
from app.tools import documents
from app.tools.documents import ContentError, Deck, Slide
from app.tools.executor import REDRAW_RULE, Pending, ToolBox, figure_name, reader_reason, run_summary
from app.tools.sandbox import Completed

SHA = "abc123"


def png(path: Path, width=40, height=30, shade=90) -> Path:
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height), False)
    pixmap.set_rect(pixmap.irect, (shade, 120, 200))
    path.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(str(path))
    return path


@pytest.fixture
def settings(tmp_path):
    made = Settings(data_dir=tmp_path / "data", library_dir=tmp_path / "library", outputs_dir=tmp_path / "outputs")
    made.outputs_dir.mkdir(parents=True)
    return made


def hit(figures, chunk_id=1) -> Hit:
    return Hit(chunk_id=chunk_id, doc_id=1, doc_path="paper.pdf", doc_title="Paper", heading_path=[], text="Text",
               page_start=13, page_end=13, block_ids=[], meta={}, score=1.0, figures=figures)


def figure(settings, figure_id, caption, page=13, image="figures/p13-1.png", exists=True) -> dict:
    if exists:
        png(settings.parsed_dir / SHA / image, shade=figure_id % 250)
    return {"figure_id": figure_id, "chunk_id": 1, "page": page, "caption": caption, "image_path": image,
            "sha256": SHA, "url": "", "score": None}


def box(settings, hits) -> ToolBox:
    return ToolBox(settings, lambda query: [], {}, hits)


class TestTheNames:
    def test_each_figure_of_the_sources_has_a_name_mapped_to_its_crop(self, settings):
        five = figure(settings, 176, "Figure 5: Phase error over time.", image="figures/p13-1.png")
        two = figure(settings, 171, "Figure 2: NCC over the rollout.", page=6, image="figures/p6-1.png")
        files = box(settings, [hit([five]), hit([two], 2)]).figure_files()
        assert files == {"source-figure-176.png": settings.parsed_dir / SHA / "figures/p13-1.png",
                         "source-figure-171.png": settings.parsed_dir / SHA / "figures/p6-1.png"}

    def test_a_crop_that_isnt_on_disk_isnt_offered(self, settings):
        gone = figure(settings, 9, "Figure 1: Gone.", exists=False)
        assert box(settings, [hit([gone])]).figure_files() == {}
        assert box(settings, [hit([gone])]).figure_menu() == ""

    def test_the_menu_lists_label_page_source_and_the_rule(self, settings):
        five = figure(settings, 176, "Figure 5: Phase error over time.")
        menu = box(settings, [hit([]), hit([five, five], 2)]).figure_menu()
        assert "- Figure 5, page 13, from [S2]: ![Figure 5](source-figure-176.png)" in menu
        assert menu.count("source-figure-176.png") == 1
        assert REDRAW_RULE in menu and "Approximate redraw, not the original figure" in menu

    @pytest.mark.parametrize("caption, name", [("Figure 5: Phase.", "Figure 5"), ("Fig. 3b shows", "Fig. 3b"),
                                               ("Phase error over time", "A figure"), ("", "A figure")])
    def test_a_figure_is_called_what_its_caption_calls_it(self, caption, name):
        assert figure_name(caption) == name

    @pytest.mark.parametrize("tool, listed", [("make_pdf", True), ("make_docx", True), ("make_pptx", True),
                                              ("make_markdown", False)])
    def test_starting_a_file_lists_the_figures_it_can_take(self, settings, tool, listed):
        five = figure(settings, 176, "Figure 5: Phase error over time.")
        outcome = asyncio.run(box(settings, [hit([five])]).call(tool, {"filename": "summary"}))
        assert outcome.pending is not None
        assert ("source-figure-176.png" in outcome.text) is listed

    def test_the_tools_prompt_forbids_presenting_a_redraw_as_the_figure(self):
        assert "Never redraw a figure from its description" in TOOLS_PROMPT
        assert '"Approximate redraw, not the original figure" written inside the image itself' in TOOLS_PROMPT


class TestTheCopy:
    def test_a_figure_name_brings_the_original_crop_byte_for_byte(self, settings, tmp_path):
        crop = png(settings.parsed_dir / SHA / "figures/p13-1.png")
        work = tmp_path / "work"
        work.mkdir()
        text = documents.localize_images("![Figure 5](source-figure-176.png)", settings.outputs_dir, work, 10**7,
                                         {"source-figure-176.png": crop})
        assert text == "![Figure 5](images/source-figure-176.png)"
        assert (work / "images" / "source-figure-176.png").read_bytes() == crop.read_bytes()

    @pytest.mark.parametrize("name", ["source-figure-999.png", "figure_5_description", "../source-figure-176.png"])
    def test_a_name_the_app_didnt_give_stops_the_build(self, settings, tmp_path, name):
        crop = png(settings.parsed_dir / SHA / "figures/p13-1.png")
        with pytest.raises(ContentError, match="isn't a plot this app made or an original figure"):
            documents.localize_images(f"![x]({name})", settings.outputs_dir, tmp_path, 10**7,
                                      {"source-figure-176.png": crop})

    def test_a_figure_name_is_never_read_from_outputs(self, settings, tmp_path):
        """outputs/ is where plots land; a file there can't pose as an original figure."""
        png(settings.outputs_dir / "source-figure-176.png", shade=10)
        crop = png(settings.parsed_dir / SHA / "figures/p13-1.png", shade=200)
        work = tmp_path / "work"
        work.mkdir()
        documents.localize_images("![F](source-figure-176.png)", settings.outputs_dir, work, 10**7,
                                  {"source-figure-176.png": crop})
        assert (work / "images" / "source-figure-176.png").read_bytes() == crop.read_bytes()

    def test_a_deck_keeps_a_figure_name_and_never_swaps_it_for_a_plot(self, settings):
        crop = png(settings.parsed_dir / SHA / "figures/p13-1.png")
        deck = Deck(slides=[Slide(layout="figure_text", title="T", image="source-figure-176.png")])
        figures = {"source-figure-176.png": crop}
        documents.resolve_images(deck, {"plot.png": "plot-1a2b3c.png"}, figures)
        assert documents.drop_missing_images(deck, settings.outputs_dir, figures) == []
        assert deck.slides[0].image == "source-figure-176.png"


class TestWhatTheReaderSees:
    def test_a_failed_build_says_why_in_the_chat(self, settings):
        outcome = asyncio.run(box(settings, []).build(Pending("make_pdf", "s"), "# S\n\n![F](figure_5_description)\n"))
        assert not outcome.ok
        assert outcome.summary == ("make_pdf failed: The image 'figure_5_description' isn't a plot this app made "
                                   "or an original figure from the sources")

    def test_the_reason_is_the_first_sentence_and_short(self):
        assert reader_reason(ContentError("No slides. Write ## lines.")) == "No slides"
        assert len(reader_reason(ContentError("x" * 500))) == 161

    @pytest.mark.parametrize("code, line", [(0, "Ran the code"), (1, "The code failed (exit code 1)")])
    def test_a_run_says_what_happened_not_exit_code_0(self, code, line):
        done = Completed(code, "", "", 1.0, False, False)
        assert run_summary(done, 30)[1] == line
