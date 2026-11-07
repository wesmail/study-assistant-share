"""What runs inside the image, tested here on text written for these tests.

sympy_check.py and the slide layout need only libraries the app already has. Anything
that needs TeX, pandoc or a container is in test_sandbox_docker.py.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

# This checkout's own .env, wherever it lives: what an attack would try to read.
ENV_FILE = str(Path(__file__).resolve().parents[2] / ".env")

SANDBOX = Path(__file__).resolve().parents[2] / "sandbox"
sys.path.append(str(SANDBOX))               # they import each other, as they do in the image


def load(name):
    spec = importlib.util.spec_from_file_location(f"sandbox_{name}", SANDBOX / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sympy_check = load("sympy_check")
build = load("build")
deck = load("deck")
measure = load("measure_deck")
theme = load("theme")


class TestVerdicts:
    def verdict(self, lhs, rhs, positive=()):
        return sympy_check.check({"lhs": lhs, "rhs": rhs, "positive": list(positive)})

    def test_a_true_derivative(self):
        assert self.verdict("diff(sin(x)**2, x)", "sin(2*x)")["verdict"] == "true"

    def test_a_false_integral_gives_the_real_value(self):
        result = self.verdict("integrate(exp(-x**2), (x, 0, oo))", "pi/2")
        assert result["verdict"] == "false"
        assert result["lhs"] == "sqrt(pi)/2"

    def test_the_gaussian_integral_needs_sigma_positive(self):
        lhs = "integrate(exp(-x**2/(2*sigma**2)), (x, -oo, oo))"
        assert self.verdict(lhs, "sigma*sqrt(2*pi)", ["sigma"])["verdict"] == "true"
        assert self.verdict(lhs, "sigma*sqrt(2*pi)")["verdict"] == "undecided"

    def test_the_sigmoid_derivative(self):
        s = "1/(1 + exp(-x))"
        assert self.verdict(f"diff({s}, x)", f"({s})*(1 - {s})")["verdict"] == "true"

    def test_a_false_claim_with_a_free_symbol(self):
        assert self.verdict("diff(x**3, x)", "2*x**2")["verdict"] == "false"

    def test_caret_means_power(self):
        assert self.verdict("x^2", "x*x")["verdict"] == "true"

    def test_an_identity(self):
        assert self.verdict("cosh(x)**2 - sinh(x)**2", "1")["verdict"] == "true"

    def test_a_note_says_how_far_apart_the_sides_are(self):
        assert "differ" in self.verdict("x + 1", "x")["note"]


class TestEquations:
    def test_each_equation_gets_its_own_page_and_line(self):
        source = deck.equation_source(["a = b", "c\n= d"])
        assert "\\[ a = b \\]\n\\clearpage\n\\[ c = d \\]" in source

    @pytest.mark.parametrize("environment", ["align", "align*", "gather", "equation"])
    def test_a_display_environment_isnt_wrapped_in_another(self, environment):
        latex = f"\\begin{{{environment}}} a \\end{{{environment}}}"
        assert latex in deck.equation_source([latex]) and "\\[ " not in deck.equation_source([latex])

    def test_an_error_line_names_the_page(self):
        source = deck.equation_source(["a", "b", "c"])
        line = source.split("\n").index("\\[ b \\]") + 1
        done = type("Done", (), {"stderr": f"error: equations.tex:{line}: Undefined control sequence"})
        assert deck.failing_page(done, "", source) == 2

    def test_no_line_number_means_no_guess(self):
        done = type("Done", (), {"stderr": "error: something"})
        assert deck.failing_page(done, "", "x") is None


class TestDerivationRows:
    BLOCK = r"\begin{aligned} a &= b \\ &= c \\ &= d \end{aligned}"

    def test_each_row_is_the_whole_block_with_the_others_hidden(self):
        rows = deck.derivation_rows(self.BLOCK)
        assert len(rows) == 3
        assert r"a &= b" in rows[0] and r"\phantom{= c}" in rows[0]
        assert r"\phantom{a}" in rows[1] and r"&= c" in rows[1]

    def test_a_matrix_inside_a_row_isnt_split(self):
        block = r"\begin{aligned} A &= \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \\ &= B \end{aligned}"
        assert len(deck.derivation_rows(block)) == 2

    def test_a_single_equation_stays_whole(self):
        assert deck.derivation_rows("a = b") == ["a = b"]

    def test_too_many_rows_stay_as_one_picture(self):
        block = "\\begin{aligned}" + " \\\\ ".join(["a &= b"] * 9) + "\\end{aligned}"
        assert deck.derivation_rows(block) == [block]

    def test_only_an_animated_slide_is_split(self):
        slides = [{"equations": [self.BLOCK]}, {"equations": [self.BLOCK], "animate": True}]
        assert [len(group) for _, group in deck.equation_groups(slides)] == [1, 3]


class TestPictures:
    def test_images_become_data_uris_so_pandocs_sandbox_needs_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(build, "WORK", tmp_path)
        (tmp_path / "images").mkdir()
        (tmp_path / "images" / "plot.png").write_bytes(b"\x89PNG")
        assert build.inline_images("![a](images/plot.png)") == "![a](data:image/png;base64,iVBORw==)"

    @pytest.mark.parametrize("text", [f"![a]({ENV_FILE})", "![a](../.env)", "![a](images/../x)"])
    def test_any_other_path_is_left_for_pandoc_to_refuse(self, tmp_path, monkeypatch, text):
        monkeypatch.setattr(build, "WORK", tmp_path)
        assert build.inline_images(text) == text

    def test_a_link_in_the_folder_isnt_read(self, tmp_path, monkeypatch):
        monkeypatch.setattr(build, "WORK", tmp_path)
        (tmp_path / "images").mkdir()
        secret = tmp_path / "secret"
        secret.write_text("KEY=value")
        (tmp_path / "images" / "plot.png").symlink_to(secret)
        assert build.inline_images("![a](images/plot.png)") == "![a](images/plot.png)"


def picture(path: Path, width: int, height: int) -> Path:
    from PIL import Image

    Image.new("RGBA", (width, height), (20, 20, 20, 255)).save(path)
    return path


A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
SLIDES = {
    "title": {"title": "A deck", "subtitle": "A subtitle"},
    "bullets_equation": {"title": "T", "bullets": ["One", "Two"], "equations": ["a"], "footer": "Sources: Book, p. 1",
                         "notes": "Say it"},
    "big_equation": {"title": "T", "equations": ["a", "b"], "caption": "A caption"},
    "figure_text": {"title": "T", "bullets": ["One"], "image": "plot.png", "caption": "A caption"},
    "two_columns": {"title": "T", "left_heading": "L", "right_heading": "R", "left_bullets": ["a", "b"],
                    "right_bullets": ["c", "d"]},
    "summary": {"title": "T", "bullets": ["One", "Two", "Three"]},
}


def slide_of(layout: str, tmp_path: Path, monkeypatch, **changes):
    """A real deck with one slide in the layout, and plain pictures for its equations."""
    monkeypatch.setattr(deck, "WORK", tmp_path)
    (tmp_path / "images").mkdir(exist_ok=True)
    picture(tmp_path / "images" / "plot.png", 800, 600)
    made, layouts = deck.themed_deck()
    content = {**SLIDES[layout], "layout": layout, **changes}
    equations = [[picture(tmp_path / f"e{index}.png", 1800, 200)] for index in range(len(content.get("equations", [])))]
    deck.add_slide(made, layouts, 1, content, equations, [])
    return made, made.slides[0]


def overlapping(slide):
    shapes = [(s.left, s.top, s.left + s.width, s.top + s.height, s.name) for s in slide.shapes if s.name != "panel"]
    for index, one in enumerate(shapes):
        for other in shapes[index + 1:]:
            if one[0] < other[2] and other[0] < one[2] and one[1] < other[3] and other[1] < one[3]:
                return one[4], other[4]
    return None


class TestTheme:
    def themed(self):
        return deck.themed_deck()[0]

    def theme_xml(self, made) -> str:
        from pptx.opc.constants import RELATIONSHIP_TYPE as RT

        return made.slide_master.part.part_related_by(RT.THEME).blob.decode()

    def test_the_deck_has_exactly_the_six_named_layouts(self):
        assert sorted(layout.name for layout in self.themed().slide_layouts) == sorted(theme.LAYOUTS)

    def test_the_palette_and_fonts_are_in_the_files_theme(self):
        xml = self.theme_xml(self.themed())
        assert theme.ACCENT in xml and theme.INK in xml
        assert f'typeface="{theme.HEADING_FONT}"' in xml and f'typeface="{theme.BODY_FONT}"' in xml

    def test_a_layout_doesnt_restyle_bullets_behind_the_themes_back(self):
        """The default section header forced grey capitals on the summary slide."""
        for layout in self.themed().slide_layouts:
            if layout.name in ("bullets_equation", "summary", "figure_text", "big_equation"):
                for shape in layout.placeholders:
                    assert len(shape._element.txBody.find(f"{A}lstStyle")) == 0

    @pytest.mark.parametrize("layout", theme.LAYOUTS)
    def test_every_layout_lays_out_with_nothing_overlapping(self, layout, tmp_path, monkeypatch):
        _, slide = slide_of(layout, tmp_path, monkeypatch)
        assert overlapping(slide) is None
        assert slide.slide_layout.name == layout

    @pytest.mark.parametrize("layout", theme.LAYOUTS)
    def test_nothing_runs_off_the_slide(self, layout, tmp_path, monkeypatch):
        made, slide = slide_of(layout, tmp_path, monkeypatch)
        for shape in slide.shapes:
            assert shape.left >= 0 and shape.top >= 0
            assert shape.left + shape.width <= made.slide_width and shape.top + shape.height <= made.slide_height

    @pytest.mark.parametrize("layout", ["bullets_equation", "summary", "two_columns"])
    def test_text_stays_in_front_of_its_panel(self, layout, tmp_path, monkeypatch):
        _, slide = slide_of(layout, tmp_path, monkeypatch)
        names = [shape.name for shape in slide.shapes]
        panels = [index for index, name in enumerate(names) if name == "panel"]
        text = [index for index, shape in enumerate(slide.shapes) if shape.is_placeholder]
        if layout != "bullets_equation":
            assert max(panels) < max(text)

    def test_an_equation_grows_to_fill_its_frame_up_to_a_limit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(deck, "WORK", tmp_path)
        made, layouts = deck.themed_deck()
        small = [[picture(tmp_path / "small.png", 900, 100)]]
        deck.add_slide(made, layouts, 1, {"layout": "big_equation", "title": "T", "equations": ["a"]}, small, [])
        shape = next(s for s in made.slides[0].shapes if s.name == "equation 1")
        assert shape.width / deck.EMU == pytest.approx(900 / deck.EQUATION_DPI * deck.BIG_SCALE, abs=0.02)

    def test_a_wide_equation_shrinks_to_fit_and_says_so(self, tmp_path, monkeypatch):
        monkeypatch.setattr(deck, "WORK", tmp_path)
        made, layouts = deck.themed_deck()
        warnings = []
        wide = [[picture(tmp_path / "wide.png", 6000, 100)]]
        deck.add_slide(made, layouts, 3, {"layout": "big_equation", "title": "T", "equations": ["a"]}, wide, warnings)
        shape = next(s for s in made.slides[0].shapes if s.name == "equation 1")
        assert shape.width / deck.EMU <= deck.CONTENT
        assert "slide 3" in warnings[0]

    def test_the_equation_keeps_its_latex_as_alt_text(self, tmp_path, monkeypatch):
        _, slide = slide_of("bullets_equation", tmp_path, monkeypatch, equations=["a = b"])
        shape = next(s for s in slide.shapes if s.name == "equation 1")
        assert shape._element.nvPicPr.cNvPr.get("descr") == "a = b"

    def test_the_slide_number_is_a_live_field(self, tmp_path, monkeypatch):
        _, slide = slide_of("summary", tmp_path, monkeypatch)
        shape = next(s for s in slide.shapes if s.name == "slide number")
        assert shape._element.find(f".//{A}fld").get("type") == "slidenum"

    def test_a_missing_picture_stops_the_build_and_says_which_slide(self, tmp_path, monkeypatch, capsys):
        with pytest.raises(SystemExit):
            slide_of("figure_text", tmp_path, monkeypatch, image=ENV_FILE)
        assert "slide 1" in json.loads(capsys.readouterr().out)["error"]


class TestScripts:
    @pytest.mark.parametrize("text, parts", [
        ("Near M_Z and Γ_Z here", [("Near M", 0), ("Z", deck.SUBSCRIPT), (" and Γ", 0), ("Z", deck.SUBSCRIPT), (" here", 0)]),
        ("The term D_Z(s) sets it", [("The term D", 0), ("Z", deck.SUBSCRIPT), ("(s) sets it", 0)]),
        ("x^n grows", [("x", 0), ("n", deck.SUPERSCRIPT), (" grows", 0)]),
        ("Plain text", [("Plain text", 0)]),
        ("run_python and gd_iterates stay", [("run_python and gd_iterates stay", 0)]),
    ])
    def test_real_subscripts_for_what_unicode_lacks(self, text, parts):
        assert deck.script_parts(text) == parts

    @pytest.mark.parametrize("text, parts", [
        ("x_(k+1) next", [("x", 0), ("k+1", deck.SUBSCRIPT), (" next", 0)]),
        ("T_(μν) here", [("T", 0), ("μν", deck.SUBSCRIPT), (" here", 0)]),
        ("e^(-x)", [("e", 0), ("-x", deck.SUPERSCRIPT)]),
    ])
    def test_a_group_that_unicode_couldnt_write_is_set_as_real_script(self, text, parts):
        assert deck.script_parts(text) == parts

    @pytest.mark.parametrize("capital", list("ZNKMΓΣ"))
    def test_a_capital_subscript_keeps_its_case(self, capital):
        assert deck.script_parts(f"M_{capital}") == [("M", 0), (capital, deck.SUBSCRIPT)]

    def test_notes_get_real_subscripts_too(self, tmp_path, monkeypatch):
        _, slide = slide_of("summary", tmp_path, monkeypatch, notes="Near M_Z.\nSecond line")
        frame = slide.notes_slide.notes_text_frame
        runs = frame.paragraphs[0].runs
        assert [run.text for run in runs] == ["Near M", "Z", "."] and len(frame.paragraphs) == 2
        assert runs[1]._r.get_or_add_rPr().get("baseline") == str(deck.SUBSCRIPT)

    def test_titles_subtitles_and_column_heads_get_them(self, tmp_path, monkeypatch):
        _, slide = slide_of("two_columns", tmp_path, monkeypatch, title="The width Γ_Z", left_heading="M_Z")
        assert [run.text for run in slide.shapes.title.text_frame.paragraphs[0].runs] == ["The width Γ", "Z"]
        head = deck.body_placeholders(slide)[0]
        assert [run.text for run in head.text_frame.paragraphs[0].runs] == ["M", "Z"]

    def test_a_subscript_is_a_real_run(self, tmp_path, monkeypatch):
        _, slide = slide_of("summary", tmp_path, monkeypatch, bullets=["Near M_Z"])
        body = deck.body_placeholders(slide)[0]
        runs = body.text_frame.paragraphs[0].runs
        assert [run.text for run in runs] == ["Near M", "Z"]
        assert runs[1]._r.get_or_add_rPr().get("baseline") == str(deck.SUBSCRIPT)


class TestFigures:
    def picture_named(self, slide, name="image"):
        return next(s for s in slide.shapes if s.name == name)

    def wide(self, tmp_path):
        picture(tmp_path / "images" / "plot.png", 1200, 400)

    def test_a_tall_figure_sits_beside_the_text(self, tmp_path, monkeypatch):
        _, slide = slide_of("figure_text", tmp_path, monkeypatch)
        assert self.picture_named(slide).left > deck.inches(deck.MARGIN + deck.FIGURE_TEXT_WIDTH)
        assert overlapping(slide) is None

    def test_a_wide_figure_takes_the_full_width_with_text_under_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr(deck, "WORK", tmp_path)
        (tmp_path / "images").mkdir()
        self.wide(tmp_path)
        made, layouts = deck.themed_deck()
        content = {**SLIDES["figure_text"], "layout": "figure_text", "bullets": ["One", "Two"]}
        deck.add_slide(made, layouts, 1, content, [], [])
        slide = made.slides[0]
        image = self.picture_named(slide)
        assert image.width > deck.inches(8) and overlapping(slide) is None      # far past the 6.3 in column
        assert deck.body_placeholders(slide)[0].top >= image.top + image.height

    def test_a_wide_figure_keeps_its_equation(self, tmp_path, monkeypatch):
        """Review R2-C8: the full-width branch never placed the slide's equations."""
        monkeypatch.setattr(deck, "WORK", tmp_path)
        (tmp_path / "images").mkdir()
        self.wide(tmp_path)
        made, layouts = deck.themed_deck()
        content = {**SLIDES["figure_text"], "layout": "figure_text", "bullets": ["One", "Two"],
                   "equations": ["E = mc^2"]}
        deck.add_slide(made, layouts, 1, content, [[picture(tmp_path / "e0.png", 1800, 200)]], [])
        slide = made.slides[0]
        image, equation = self.picture_named(slide), self.picture_named(slide, "equation 1")
        middle = image.left + image.width / 2                   # centred on the slide, not in a column
        assert abs(middle - deck.inches(deck.MARGIN + deck.CONTENT / 2)) < deck.inches(0.05)
        assert image.top + image.height <= deck.body_placeholders(slide)[0].top and overlapping(slide) is None
        assert equation.top >= deck.body_placeholders(slide)[0].top + deck.body_placeholders(slide)[0].height
        assert equation.top + equation.height <= deck.inches(deck.BODY_BOTTOM)

    def test_a_long_caption_gets_a_box_tall_enough(self, tmp_path, monkeypatch):
        caption = "A caption long enough that it has to wrap onto a second line under this figure, certainly"
        _, slide = slide_of("figure_text", tmp_path, monkeypatch, caption=caption)
        assert self.picture_named(slide, "caption").height > deck.inches(0.4)


class TestSummary:
    def test_the_panel_is_as_tall_as_its_points(self, tmp_path, monkeypatch):
        _, slide = slide_of("summary", tmp_path, monkeypatch, bullets=["One", "Two"])
        panel = next(s for s in slide.shapes if s.name == "panel")
        body = deck.body_placeholders(slide)[0]
        assert panel.height < deck.inches(deck.BODY_BOTTOM - deck.BODY_TOP) / 2
        assert panel.top <= body.top and body.top + body.height <= panel.top + panel.height

    def test_more_points_make_a_taller_panel(self, tmp_path, monkeypatch):
        short = next(s for s in slide_of("summary", tmp_path, monkeypatch, bullets=["One"])[1].shapes if s.name == "panel")
        tall = next(s for s in slide_of("summary", tmp_path, monkeypatch,
                                         bullets=["One", "Two", "Three", "Four"])[1].shapes if s.name == "panel")
        assert tall.height > short.height


class TestMotion:
    def test_every_slide_fades(self, tmp_path, monkeypatch):
        _, slide = slide_of("summary", tmp_path, monkeypatch)
        assert slide._element.find(f"{P}transition").find(f"{P}fade") is not None

    def test_the_transition_comes_before_the_timing_as_the_schema_requires(self, tmp_path, monkeypatch):
        _, slide = slide_of("summary", tmp_path, monkeypatch, animate=True)
        tags = [child.tag.split("}")[1] for child in slide._element]
        assert tags.index("clrMapOvr") < tags.index("transition") < tags.index("timing")

    def test_no_clicks_unless_asked(self, tmp_path, monkeypatch):
        _, slide = slide_of("summary", tmp_path, monkeypatch)
        assert slide._element.find(f"{P}timing") is None

    def clicks(self, slide):
        return slide._element.findall(f".//{P}cTn[@nodeType='clickEffect']")

    def test_each_bullet_is_one_click(self, tmp_path, monkeypatch):
        _, slide = slide_of("summary", tmp_path, monkeypatch, animate=True)
        assert len(self.clicks(slide)) == 3
        assert [item.get("st") for item in slide._element.findall(f".//{P}pRg")] == ["0", "1", "2"]

    def test_bullets_come_before_the_equation(self, tmp_path, monkeypatch):
        _, slide = slide_of("bullets_equation", tmp_path, monkeypatch, animate=True)
        targets = slide._element.findall(f".//{P}spTgt")
        assert len(targets) == 3
        assert targets[0].find(f"{P}txEl") is not None and targets[2].find(f"{P}txEl") is None

    def test_both_columns_build_left_then_right(self, tmp_path, monkeypatch):
        _, slide = slide_of("two_columns", tmp_path, monkeypatch, animate=True)
        shapes = [target.get("spid") for target in slide._element.findall(f".//{P}spTgt")]
        assert len(shapes) == 4 and shapes[0] == shapes[1] and shapes[2] == shapes[3] and shapes[0] != shapes[2]

    def test_timing_ids_are_unique_because_powerpoint_rejects_duplicates(self):
        from lxml import etree

        root = etree.fromstring(deck.timing_xml([(3, 0), (3, 1), (5, None)]).encode())
        ids = [item.get("id") for item in root.iter(f"{P}cTn")]
        assert len(ids) == len(set(ids))

    def test_only_text_shapes_get_a_paragraph_build(self):
        from lxml import etree

        root = etree.fromstring(deck.timing_xml([(3, 0), (5, None)]).encode())
        assert [item.get("spid") for item in root.iter(f"{P}bldP")] == ["3"]


class TestMeasuring:
    def test_a_slide_that_fits_is_clean(self, tmp_path, monkeypatch):
        made, _ = slide_of("bullets_equation", tmp_path, monkeypatch)
        made.save(tmp_path / "a.pptx")
        assert measure.overflowing(str(tmp_path / "a.pptx")) == []

    def test_text_too_big_for_its_box_is_caught(self, tmp_path, monkeypatch):
        made, slide = slide_of("summary", tmp_path, monkeypatch)
        body = deck.body_placeholders(slide)[0]
        body.height = int(0.3 * deck.EMU)
        made.save(tmp_path / "a.pptx")
        assert "needs" in measure.overflowing(str(tmp_path / "a.pptx"))[0]


def test_the_outline_and_the_builder_agree_on_field_names():
    from app.tools.documents import Slide

    used = {"layout", "title", "subtitle", "bullets", "equations", "image", "caption", "left_heading",
            "right_heading", "left_bullets", "right_bullets", "animate", "notes", "footer"}
    assert used <= set(Slide.model_fields)
    source = (SANDBOX / "deck.py").read_text()
    for field in used:
        assert f'"{field}"' in source, field
