"""Merging two readings of a page, with fake readers and a real small PDF."""
import fitz

from app.ingest.formula_layer import page_trace
import pytest

from app.ingest.chunker import DROPPED_EQUATION
from app.ingest.formula_layer import Glyph
from app.ingest.formula_merge import (
    glyph_mismatch,
    group_boxes,
    linked,
    merge_document,
    save_crops,
)


def always(formulas):
    return [True] * len(formulas)


def eq(order, text, top, bottom, page=1, **extra):
    box = [100.0, float(top), 400.0, float(bottom)]
    return {"order": order, "kind": "equation", "text": f"$$\n{text}\n$$", "level": 0,
            "page": page, "bbox": box, "bboxes": [{"page": page, "bbox": box}],
            "docling_ref": f"#/texts/{order}", **extra}


def para(order, text, top, bottom, page=1):
    box = [100.0, float(top), 400.0, float(bottom)]
    return {"order": order, "kind": "paragraph", "text": text, "level": 0, "page": page,
            "bbox": box, "bboxes": [{"page": page, "bbox": box}], "docling_ref": None}


def formula(latex, top, bottom):
    return {"latex": latex, "bbox": [100.0, float(top), 500.0, float(bottom)]}


def merge(records, formulas, check=always):
    merged, stats = merge_document(records, None, {1: formulas} if formulas else {}, check)
    return merged, stats


def latex_of(record):
    return record["text"].removeprefix("$$\n").removesuffix("\n$$")


# --- Grouping ----------------------------------------------------------------


@pytest.mark.parametrize("a, b, expected", [
    ([0, 0, 100, 20], [0, 5, 100, 25], True),
    ([0, 0, 100, 20], [0, 18, 100, 40], False),      # only grazing
    ([0, 0, 100, 20], [0, 30, 100, 50], False),
    ([0, 0, 100, 20], [200, 0, 300, 20], False),     # same height, other column
    ([0, 0, 100, 100], [0, 40, 100, 50], True),      # one line of a tall block
])
def test_linked(a, b, expected):
    assert linked(a, b) is expected


def test_one_tall_box_links_several_lines():
    groups = group_boxes({7: [0, 0, 100, 60]}, [[0, 0, 100, 20], [0, 30, 100, 50]])
    assert [(g.docling, sorted(g.granite)) for g in groups] == [([7], [0, 1])]


def test_unmatched_boxes_stay_alone():
    groups = group_boxes({1: [0, 0, 100, 20]}, [[0, 300, 100, 320]])
    assert sorted((g.docling, g.granite) for g in groups) == [([], [0]), ([1], [])]


# --- Merging -----------------------------------------------------------------


def test_granite_replaces_docling_in_place():
    records = [para(0, "Before.", 10, 20), eq(1, "x=2 \\tag{1.1}", 30, 50), para(2, "After.", 60, 70)]
    merged, stats = merge(records, [formula("y=2", 32, 48)])
    assert [r["kind"] for r in merged] == ["paragraph", "equation", "paragraph"]
    assert latex_of(merged[1]).startswith("y=2")
    assert merged[1]["formula_source"] == "granite"
    assert [r["order"] for r in merged] == [0, 1, 2]
    assert stats["granite"] == 1 and stats["docling"] == 0


def test_a_formula_granite_missed_keeps_docling():
    merged, stats = merge([eq(0, "x=2", 30, 50)], [])
    assert latex_of(merged[0]) == "x=2"
    assert merged[0]["formula_source"] == "docling"
    assert stats["docling_only"] == 1


def test_a_formula_only_granite_saw_lands_where_it_belongs():
    records = [para(0, "Top.", 10, 20), para(1, "Bottom.", 90, 100)]
    merged, stats = merge(records, [formula("z=3", 40, 60)])
    assert [r["text"][:6] for r in merged] == ["Top.", "$$\nz=3", "Bottom"]
    assert stats["granite_only"] == 1


def test_a_formula_at_the_end_of_a_page_stays_on_that_page():
    records = [para(0, "One.", 10, 20), para(1, "Two.", 10, 20, page=2)]
    merged, _ = merge(records, [formula("z=3", 700, 720)])
    assert [r["page"] for r in merged] == [1, 1, 2]


def test_several_docling_blocks_become_one_granite_formula():
    records = [eq(0, "a", 30, 45), eq(1, "b", 46, 60), para(2, "After.", 70, 80)]
    merged, _ = merge(records, [formula("a \\\\ b", 30, 60)])
    assert [latex_of(r) for r in merged[:-1]] == ["a \\\\ b"]
    assert merged[-1]["text"] == "After."


def test_one_docling_block_splits_into_granite_lines_in_order():
    merged, _ = merge([eq(0, "a b", 30, 70)], [formula("b=2", 52, 68), formula("a=1", 32, 48)])
    assert [latex_of(r) for r in merged] == ["a=1", "b=2"]


def test_a_junk_granite_reading_falls_back_to_docling():
    merged, _ = merge([eq(0, "x=2", 30, 50)], [formula(r"x \\ \intertext{a}", 30, 50)])
    assert latex_of(merged[0]) == "x=2"
    assert merged[0]["formula_source"] == "docling"


def test_a_reading_katex_rejects_falls_back_to_docling():
    check = lambda formulas: ["bad" not in f for f in formulas]
    merged, _ = merge([eq(0, "x=2", 30, 50)], [formula(r"\bad{x}", 30, 50)], check)
    assert latex_of(merged[0]) == "x=2"
    assert merged[0]["compile_ok"] is True


def test_when_nothing_compiles_the_reading_stays_and_says_so():
    merged, stats = merge([eq(0, "x=2", 30, 50)], [formula("y=2", 30, 50)],
                          lambda formulas: [False] * len(formulas))
    assert merged[0]["kind"] == "equation" and merged[0]["compile_ok"] is False
    assert latex_of(merged[0]) == "y=2"
    assert stats["compile_failed"] == 1


def test_junk_from_both_readers_is_dropped_but_not_lost():
    junk = r"\vdots \, , \, " * 9
    merged, stats = merge([eq(0, junk, 30, 50)], [formula(r"x \intertext{word}", 30, 50)])
    assert merged[0]["kind"] == DROPPED_EQUATION
    assert merged[0]["drop_reason"] == "prose"
    assert merged[0]["bbox"][1] == 30
    assert stats["dropped"] == 1 and stats["drop_reasons"] == {"prose": 1}


def test_a_lone_number_block_is_dropped():
    merged, _ = merge([eq(0, "(2 . 2 8)", 30, 50)], [])
    assert merged[0]["kind"] == DROPPED_EQUATION
    assert merged[0]["drop_reason"] == "number"


def test_an_unrecognised_formula_with_no_reading_is_dropped():
    record = eq(0, "% [formula not recognised — page 1]", 30, 50, formula_unrecognised=True)
    assert merge([record], [])[0][0]["kind"] == DROPPED_EQUATION


def test_a_fragment_inside_a_formula_is_absorbed():
    records = [eq(0, "x", 30, 60), para(1, "̸", 40, 44), para(2, "After.", 70, 80)]
    merged, stats = merge(records, [formula("x\\neq y", 30, 60)])
    assert [r["kind"] for r in merged] == ["equation", "paragraph"]
    assert stats["absorbed"] == 1


def test_prose_inside_a_formula_box_is_kept():
    words = "This sentence is real prose that must never disappear from the document."
    records = [eq(0, "x", 30, 60), para(1, words, 40, 50)]
    merged, _ = merge(records, [formula("x", 30, 60)])
    assert merged[-1]["text"] == words


def test_other_block_kinds_pass_through():
    records = [{"order": 0, "kind": "heading", "text": "1 Intro", "level": 1, "page": None},
               para(1, "Body.", 10, 20), eq(2, "x", 30, 50)]
    merged, _ = merge(records, [])
    assert merged[0]["text"] == "1 Intro"


def test_docling_number_survives_when_there_is_no_text_layer():
    merged, _ = merge([eq(0, "x=2 \\tag{3.1}", 30, 50, equation_number="3.1")], [formula("y", 30, 50)])
    assert latex_of(merged[0]) == "y \\tag{3.1}"
    assert merged[0]["equation_numbers"] == ["3.1"]


# --- A real page -------------------------------------------------------------


@pytest.fixture
def pdf_page():
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((110, 100), "x", fontname="hebo", fontsize=12)
    page.insert_text((125, 100), "=y", fontname="helv", fontsize=12)
    page.insert_text((500, 100), "(4.20)", fontname="helv", fontsize=12)
    yield page
    document.close()


def test_fonts_and_numbers_reach_the_merged_formula(pdf_page):
    records = [eq(0, "x=y", 85, 105)]
    merged, stats = merge_document(records, lambda _: pdf_page, {1: [formula("x=y", 85, 105)]}, always)
    assert latex_of(merged[0]) == r"\mathbf{x}=y \tag{4.20}"
    assert merged[0]["equation_numbers"] == ["4.20"]
    assert stats["numbers_placed"] == 1 and stats["numbers_unplaced"] == 0


def test_a_number_the_model_copied_is_not_printed_twice(pdf_page):
    merged, _ = merge_document([eq(0, "x", 85, 105)], lambda _: pdf_page,
                               {1: [formula(r"x=y \quad(4 . 2 0)", 85, 105)]}, always)
    assert latex_of(merged[0]) == r"\mathbf{x}=y \tag{4.20}"


def test_a_lone_number_block_does_not_take_the_number_from_its_equation(pdf_page):
    records = [eq(0, "x=y", 85, 105), {**eq(1, "(4 . 2 0)", 88, 102), "bbox": [495.0, 88.0, 540.0, 102.0]}]
    merged, stats = merge_document(records, lambda _: pdf_page, {1: []}, always)
    assert merged[0]["kind"] == "equation"
    assert merged[0]["equation_numbers"] == ["4.20"]
    assert merged[1]["kind"] == DROPPED_EQUATION
    assert stats["lone_numbers"] == 1


def test_a_number_with_no_formula_is_counted(pdf_page):
    merged, stats = merge_document([para(0, "Text.", 300, 320)], lambda _: pdf_page, {1: []}, always)
    assert stats["numbers_unplaced"] == 1


# --- The glyph check ---------------------------------------------------------


def printed(text):
    return [Glyph(char, None) for char in text]


@pytest.mark.parametrize("latex, page, numbers", [
    ("x+y=2", "x+y=2", []),                                   # operators aren't counted
    (r"\boldsymbol{x}=\alpha_{1}", "x=α1", []),
    (r"\frac{\partial f}{\partial x}", "∂f∂x", []),
    (r"\left[x\right]", "[x]", []),
    (r"\begin{bmatrix}1\end{bmatrix}", "[1]", []),
    (r"\binom{N}{y}", "(Ny)", []),
    (r"\hat{x}", "ˆx", []),
    (r"\widehat{x}", "⌢x", []),
    (r"p \propto q", "p∝q", []),
    (r"a \leq b", "a⩽b", []),                                  # one sign, two shapes
    (r"a \neq b", "a≠b", []),
    (r"a \implies b", "a=⇒b", []),                             # a long arrow is two glyphs
    (r"a \iff b", "a⇐=⇒b", []),
    (r"a \leftarrow b", "a←−b", []),
    (r"x", "x\ufffd", []),                                      # a glyph nobody can name
    (r"x \cdots y", "x···y", []),                              # dots aren't counted
    (r"\log x \tag{2.1}", "logx(2.1)", ["2.1"]),               # the box reaches the number
    (r"f(x) \tag{2.1}", "f(x)(2.1)", ["2.1"]),                 # and its brackets are forgiven
    (r"\log x \tag{2.1}", "logx", ["2.1"]),                    # it stops short of the number
    (r"x_{2} \tag{2.1}", "x2(2.1)", ["2.1"]),                  # a digit shared with the number
    (r"\text{define} x", "de\ufb01nex", []),                    # a ligature is two letters
    (r"\phi", "ϕ", []),                                       # a variant glyph of one letter
    (r"\hbar x", "ħx", []),                                    # not a counted symbol
    (r"\begin{bmatrix}1 \\ 2\end{bmatrix}", "[12]", []),       # environment names aren't letters
    (r"\boldsymbol{x}^{\top}y", "xTy", []),                     # a transpose printed as a letter
    (r"\boldsymbol{x}^{\top}y", "x⊤y", []),                     # a transpose printed as a sign
    (r"x^{\top}y \tag{4.1}", "xTy(4.1)", ["4.1"]),
])
def test_a_formula_with_the_printed_symbols_passes(latex, page, numbers):
    assert glyph_mismatch(latex, printed(page), numbers) is None


@pytest.mark.parametrize("latex, page", [
    (r"\begin{bmatrix}2 \\ 0\end{bmatrix}", "22"),               # a digit misread
    (r"\begin{bmatrix}1 \\ 2\end{bmatrix}", "123"),              # a row lost
    (r"\begin{bmatrix}4 \\ 8 \\ 8\end{bmatrix}", "48"),          # a row invented
    (r"x_{3}", "2x3"),                                          # a coefficient lost
    (r"x", ""),                                                 # no text layer to compare
    (r"\triangle q x", "≜x"),                                   # a real misreading of one sign
    (r"x^{\top}", "xT2"),                                       # the transpose excuses only itself
])
def test_a_formula_with_other_symbols_fails(latex, page):
    assert glyph_mismatch(latex, printed(page), []) == "glyphs"


@pytest.mark.parametrize("latex, page", [
    (r"p \infty q", "p∝q"),                                     # proportional-to misread
    (r"\widetilde{m}", "⌣m"),                                   # an accent misread
    (r"a \begin{bmatrix}x\end{bmatrix}", "a[[x]]"),             # an outer bracket lost
    (r"x_{\langle i}", "x\\i"),                                 # set-minus misread
    (r"x", "⏟x"),                                               # an underbrace lost
    (r"[x]", "x"),                                              # a bracket invented
    (r"a = b", "a<b"),
])
def test_a_formula_with_other_marks_fails(latex, page):
    assert glyph_mismatch(latex, printed(page), []) == "signs"


def test_letters_are_compared_even_beside_an_unnamed_glyph():
    assert glyph_mismatch("x", printed("y\ufffd"), []) == "glyphs"


def test_a_box_with_an_unnamed_glyph_is_counted(monkeypatch, pdf_page):
    from app.ingest import formula_merge

    monkeypatch.setattr(formula_merge, "page_trace", lambda _: printed("x=y\ufffd"))
    monkeypatch.setattr(formula_merge, "glyphs_in", lambda trace, _box: trace)
    _, stats = merge_document([eq(0, "x=y", 85, 105)], lambda _: pdf_page,
                              {1: [formula("x=y", 85, 105)]}, always)
    assert stats["signs_unchecked"] == 1


def test_only_the_printed_number_is_forgiven():
    assert glyph_mismatch("x", printed("x(2.1)7"), ["2.1"]) == "glyphs"


def test_a_reading_that_matches_the_page_is_verified(pdf_page):
    merged, stats = merge_document([eq(0, "x=z", 85, 105)], lambda _: pdf_page,
                                   {1: [formula("x=y", 85, 105)]}, always)
    assert "unverified" not in merged[0]
    assert merged[0]["formula_source"] == "granite"
    assert stats["unverified"] == 0 and stats["swapped"] == 0


def test_a_wrong_reading_gives_way_to_one_that_matches_the_page(pdf_page):
    merged, stats = merge_document([eq(0, "x=y", 85, 105)], lambda _: pdf_page,
                                   {1: [formula("x=z", 85, 105)]}, always)
    assert merged[0]["formula_source"] == "docling"
    assert "unverified" not in merged[0]
    assert merged[0]["alt_latex"].startswith(r"\mathbf{x}=z")
    assert stats["swapped"] == 1 and stats["docling"] == 1


def test_when_no_reading_matches_the_best_one_stays_unverified(pdf_page):
    merged, stats = merge_document([eq(0, "x=w", 85, 105)], lambda _: pdf_page,
                                   {1: [formula("x=z", 85, 105)]}, always)
    assert merged[0]["kind"] == "equation"
    assert merged[0]["formula_source"] == "granite"
    assert merged[0]["unverified"] is True
    assert merged[0]["unverified_reason"] == "glyphs"
    assert stats["unverified"] == 1 and stats["unverified_reasons"] == {"glyphs": 1}


def test_a_wrong_operator_is_unverified_with_its_own_reason(pdf_page):
    merged, stats = merge_document([eq(0, "x+y", 85, 105)], lambda _: pdf_page,
                                   {1: [formula("x<y", 85, 105)]}, always)
    assert merged[0]["unverified_reason"] == "signs"
    assert stats["unverified_reasons"] == {"signs": 1}


def test_a_wrong_operator_gives_way_to_a_reading_with_the_right_one(pdf_page):
    merged, stats = merge_document([eq(0, "x=y", 85, 105)], lambda _: pdf_page,
                                   {1: [formula("x<y", 85, 105)]}, always)
    assert merged[0]["formula_source"] == "docling"
    assert "unverified" not in merged[0]
    assert stats["swapped"] == 1


def test_a_reading_that_does_not_compile_is_unverified_even_if_it_matches(pdf_page):
    merged, stats = merge_document([eq(0, "x=y", 85, 105)], lambda _: pdf_page,
                                   {1: [formula("x=y", 85, 105)]}, lambda found: [False] * len(found))
    assert merged[0]["unverified_reason"] == "compile"
    assert stats["compile_failed"] == 1


def test_a_compiling_reading_beats_a_matching_one_that_does_not_compile(pdf_page):
    only_w_compiles = lambda found: ["w" in latex for latex in found]
    merged, _ = merge_document([eq(0, "x=w", 85, 105)], lambda _: pdf_page,
                               {1: [formula("x=y", 85, 105)]}, only_w_compiles)
    assert merged[0]["formula_source"] == "docling"
    assert merged[0]["unverified_reason"] == "glyphs"


def test_without_a_text_layer_nothing_can_be_verified():
    merged, stats = merge([eq(0, "x", 10, 20)], [formula("x", 10, 20)])
    assert merged[0]["unverified_reason"] == "glyphs"
    assert stats["unverified"] == 1


def test_a_dropped_formula_is_not_also_unverified():
    junk = r"\text{and this is plainly a whole sentence of prose}"
    merged, stats = merge([eq(0, junk, 10, 20)], [formula(junk, 10, 20)])
    assert merged[0]["kind"] == DROPPED_EQUATION
    assert "unverified" not in merged[0]
    assert stats["unverified"] == 0


# --- Crops -------------------------------------------------------------------


def test_the_crop_box_covers_both_readers(pdf_page):
    record = eq(0, "x=y", 85, 105)
    merged, _ = merge_document([record], lambda _: pdf_page, {1: [formula("x=y", 90, 102)]}, always)
    assert merged[0]["bbox"] == [100.0, 90.0, 500.0, 102.0]
    assert merged[0]["crop_bbox"] == [100.0, 85.0, 500.0, 105.0]


def test_a_crop_is_saved_for_an_unverified_formula_only(pdf_page, tmp_path):
    records = [eq(0, "x=z", 85, 105), eq(1, "x=y", 85, 105)]
    records[0]["unverified"], records[0]["crop_bbox"] = True, [100.0, 85.0, 500.0, 105.0]
    records[1]["crop_bbox"] = [100.0, 85.0, 500.0, 105.0]
    save_crops(records, lambda _: pdf_page, tmp_path / "formula_crops")
    assert records[0]["crop"] == "formula_crops/000000.png"
    assert "crop" not in records[1]
    assert [p.name for p in (tmp_path / "formula_crops").iterdir()] == ["000000.png"]
    image = fitz.Pixmap(str(tmp_path / "formula_crops" / "000000.png"))
    assert image.width > (500 - 100) * 200 / 72            # the padding is there


def test_an_unverified_formula_keeps_the_symbols_printed_in_its_box(pdf_page, tmp_path):
    record = {"order": 3, "page": 1, "unverified": True, "crop_bbox": [0.0, 0.0, 600.0, 800.0]}
    save_crops([record], lambda _: pdf_page, tmp_path / "formula_crops")
    trace = page_trace(pdf_page)
    assert trace and record["text_layer"].replace(" ", "") == "".join(glyph.char for glyph in trace).replace(" ", "")


def test_a_rerun_replaces_the_old_crops(pdf_page, tmp_path):
    directory = tmp_path / "formula_crops"
    directory.mkdir()
    (directory / "000099.png").write_bytes(b"left over")
    save_crops([], lambda _: pdf_page, directory)
    assert list(directory.iterdir()) == []


def test_a_box_off_the_page_falls_back_to_the_whole_page(pdf_page, tmp_path):
    record = {"order": 3, "page": 1, "unverified": True, "crop_bbox": [900.0, 900.0, 950.0, 950.0]}
    save_crops([record], lambda _: pdf_page, tmp_path / "formula_crops")
    image = fitz.Pixmap(str(tmp_path / record["crop"]))
    assert image.width >= 600 * 200 / 72 - 1
