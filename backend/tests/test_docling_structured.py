"""Script fixes and the blocks.jsonl sidecar, on a hand-built DoclingDocument."""
import pytest
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
)

from app.ingest.chunker import chunk_blocks, load_blocks_jsonl
from app.ingest.docling_structured import (
    build_enriched_markdown,
    build_vlm_converter,
    clean_inline_math,
    clean_latex,
    equation_tag,
    infer_heading_levels,
)

HEIGHT = 800.0


def prov(page=1, top=700.0, bottom=680.0):
    """A bottom-left box, as Docling reports: `top` is the larger y value."""
    return ProvenanceItem(
        page_no=page,
        bbox=BoundingBox(l=50, t=top, r=500, b=bottom, coord_origin=CoordOrigin.BOTTOMLEFT),
        charspan=(0, 1),
    )


@pytest.fixture
def doc():
    document = DoclingDocument(name="test")
    document.add_page(page_no=1, size=Size(width=600, height=HEIGHT))
    document.add_page(page_no=2, size=Size(width=600, height=HEIGHT))
    return document


def build(doc, tmp_path, **kwargs):
    return build_enriched_markdown(doc, tmp_path, **kwargs)


def blocks_of(doc, tmp_path, **kwargs):
    return build(doc, tmp_path, **kwargs)[3]


# --- Fix 1: ListItem before TextItem --------------------------------


def test_list_items_keep_their_bullet(doc, tmp_path):
    group = doc.add_list_group()
    doc.add_list_item("First point", parent=group, prov=prov())
    markdown, *_, blocks = build(doc, tmp_path)
    assert "- First point" in markdown
    assert blocks[0]["kind"] == "list_item"


def test_enumerated_list_keeps_its_marker(doc, tmp_path):
    group = doc.add_list_group()
    doc.add_list_item("Second step", enumerated=True, marker="2.", parent=group, prov=prov())
    assert "2. Second step" in build(doc, tmp_path)[0]


def test_enumerated_item_with_an_empty_marker_falls_back_to_a_bullet(doc, tmp_path):
    group = doc.add_list_group()
    doc.add_list_item("No marker", enumerated=True, marker="", parent=group, prov=prov())
    assert "- No marker" in build(doc, tmp_path)[0]


# --- Fix 2: TITLE as a heading --------------------------------------


def test_title_renders_as_a_top_level_heading(doc, tmp_path):
    doc.add_title("A Book About Fields", prov=prov())
    markdown, *_, blocks = build(doc, tmp_path)
    assert "\n# A Book About Fields\n" in markdown
    assert (blocks[0]["kind"], blocks[0]["level"]) == ("heading", 1)
    assert blocks[0]["text"] == "A Book About Fields"


# --- Fix 3: the inline-math guard -----------------------------------


@pytest.mark.parametrize("text", [
    "a 2 × 2 matrix",
    "Take a 2 × 2 matrix.",
    "If A = B, then a 2 × 2 matrix is enough.",
    "This is a 2 x 2 block with n = 4.",
    "Plan B 2 weeks later.",
])
def test_ordinary_text_is_not_rewritten(text):
    assert clean_inline_math(text, True) == text


@pytest.mark.parametrize("text", [
    "so that x 2 = 4",                     # x_2 on the page: a plain subscript
    "take x 1 , x 2 , x 3 as the three inputs, and set λ 1 = 0",
    "( x 1 , x 2 , x 3 , x 4 ) = (5 , 3 , -7 , 9)",
    "where e 2 ≈ 1",
])
def test_a_flattened_subscript_is_never_turned_into_a_superscript(text):
    """The text layer has lost the height of the 2, so x_2 and x^2 look alike."""
    assert clean_inline_math(text, True) == text


def test_the_line_element_is_still_repaired():
    assert clean_inline_math("the interval d s 2", True) == "the interval ds^2"


@pytest.mark.parametrize("raw, expected", [
    (r"\sigma ( a ) \triangle q \frac { 1 } { 2 }", r"\sigma(a)\triangleq \frac{1}{2}"),
    (r"x \triangle q y", r"x \triangleq y"),
    (r"\triangle ABC", r"\triangle ABC"),                    # a real triangle stays one
    (r"\triangle qr", r"\triangle qr"),                      # and so does a longer name
])
def test_a_split_triangleq_is_rejoined(raw, expected):
    assert clean_latex(raw) == expected


def test_inline_cleanup_can_be_disabled():
    assert clean_inline_math("x 2 = 4", False) == "x 2 = 4"


# --- clean_latex -------------------------------------------------------------


def test_spaced_latex_is_tightened():
    assert clean_latex(r"\frac { 1 } { 2 }") == r"\frac{1}{2}"


def test_a_broken_command_is_repaired():
    assert clean_latex(r"\ frac { 1 } { 2 }") == r"\frac{1}{2}"


@pytest.mark.parametrize("latex,expected", [
    (r"a , \ \ W _ { 1 } = 0", r"a , \ \ W_{1}= 0"),
    (r"x \ y", r"x \ y"),
    (r"x \ { y } z", r"x \ {y}z"),
    (r"x \ [ y ] z", r"x \ [y]z"),
    (r"x \ ( y ) z", r"x \ (y)z"),
])
def test_a_control_space_is_never_merged_into_the_next_token(latex, expected):
    assert clean_latex(latex) == expected


def test_a_row_break_is_not_mistaken_for_a_control_space():
    assert clean_latex(r"a \\ b") == r"a \\ b"


def test_escaped_braces_survive():
    assert clean_latex(r"\{ 0 , 1 \}") == r"\{0 , 1 \}"


def test_empty_latex():
    assert clean_latex("") == ""


# --- Equation numbers --------------------------------------------------------


@pytest.mark.parametrize("orig,latex,expected", [
    ("h ∈ { 0 , 1 } , (2)", r"h \in \{0,1\},", "2"),
    ("E = mc 2 (2.14)", "E=mc^2", "2.14"),
    ("E = mc 2 ( 2 . 14 )", "E=mc^2", "2.14"),
    ("x = 1 (A.3)", "x=1", "A.3"),
    ("x = 1 (12a)", "x=1", "12a"),
    ("no number here", "x=1", None),
    ("", "x=1", None),
])
def test_equation_number_comes_from_the_raw_text(orig, latex, expected):
    assert equation_tag(orig, latex) == expected


def test_a_bracket_that_belongs_to_the_formula_is_not_a_number():
    assert equation_tag("y = f ( 2 )", "y = f ( 2 )") is None


def test_equation_block_carries_the_tag(doc, tmp_path):
    doc.add_formula(r"E = m c ^ { 2 }", orig="E = mc 2 (2.14)", prov=prov())
    block = blocks_of(doc, tmp_path)[0]
    assert block["kind"] == "equation"
    assert block["equation_number"] == "2.14"
    assert block["text"] == "$$\nE = m c^{2} \\tag{2.14}\n$$"


def test_unrecognised_formula_is_flagged(doc, tmp_path):
    doc.add_formula("", prov=prov(page=2))
    block = blocks_of(doc, tmp_path)[0]
    assert block["formula_unrecognised"] is True
    assert "page 2" in block["text"]


# --- Heading levels ----------------------------------------------------------


def test_numbering_sets_the_level():
    assert infer_heading_levels(["3 Method", "3.1 Setup", "4.4.1 Phase drift."]) == [1, 2, 3]


def test_unnumbered_heading_sits_below_the_last_numbered_one():
    headings = ["2 Related Work", "Rollout.", "Transformers.", "4.1 Results", "Failure tail"]
    assert infer_heading_levels(headings) == [1, 2, 2, 2, 3]


def test_front_matter_and_back_matter_stay_top_level():
    headings = ["Title", "Abstract", "1 Intro", "6 Conclusion", "References"]
    assert infer_heading_levels(headings) == [1, 1, 1, 1, 1]


def test_appendix_letters_are_top_level_when_a_series_exists():
    assert infer_heading_levels(["5 End", "A Protocol", "B Analysis"]) == [1, 1, 1]
    assert infer_heading_levels(["A.1 Details"]) == [2]


def test_a_title_starting_with_a_is_not_an_appendix():
    assert infer_heading_levels(["1.1 Basics", "A Simple Example"]) == [2, 3]


def test_chapter_words_and_german_names():
    headings = ["Kapitel 2", "2.1 Lösungen", "Literaturverzeichnis"]
    assert infer_heading_levels(headings) == [1, 2, 1]


def test_a_leading_year_or_dimension_is_not_a_section_number():
    assert infer_heading_levels(["1905 and after", "3D Reconstruction"]) == [1, 1]


def test_docling_hierarchy_is_kept_when_it_has_one(doc, tmp_path):
    doc.add_heading("Unnumbered top", level=1, prov=prov())
    doc.add_heading("Unnumbered lower", level=3, prov=prov())
    assert [b["level"] for b in blocks_of(doc, tmp_path)] == [1, 3]


def test_flat_docling_levels_are_replaced_by_inference(doc, tmp_path):
    doc.add_heading("3 Method", level=1, prov=prov())
    doc.add_heading("3.1 Setup", level=1, prov=prov())
    markdown, *_, blocks = build(doc, tmp_path)
    assert "\n## 3.1 Setup\n" in markdown
    assert [b["level"] for b in blocks] == [1, 2]


# --- Sidecar -----------------------------------------------------------------


def test_bbox_is_flipped_to_a_top_left_origin(doc, tmp_path):
    doc.add_text(DocItemLabel.TEXT, "Hello.", prov=prov(top=700, bottom=680))
    block = blocks_of(doc, tmp_path)[0]
    assert block["page"] == 1
    assert block["bbox"] == [50, HEIGHT - 700, 500, HEIGHT - 680]
    assert block["bbox"][1] < block["bbox"][3]


def test_every_body_block_carries_its_docling_reference(doc, tmp_path):
    item = doc.add_text(DocItemLabel.TEXT, "Hello.", prov=prov())
    assert blocks_of(doc, tmp_path)[0]["docling_ref"] == item.self_ref


def test_block_split_across_pages_keeps_both_regions(doc, tmp_path):
    item = doc.add_text(DocItemLabel.TEXT, "Long paragraph.", prov=prov(page=1))
    item.prov.append(prov(page=2, top=750, bottom=740))
    block = blocks_of(doc, tmp_path)[0]
    assert [r["page"] for r in block["bboxes"]] == [1, 2]
    assert block["page"] == 1


def test_order_is_gapless_and_follows_the_document(doc, tmp_path):
    doc.add_heading("1 One", level=1, prov=prov())
    doc.add_text(DocItemLabel.TEXT, "Body.", prov=prov())
    doc.add_formula("x", prov=prov())
    doc.add_code("print(1)", prov=prov())
    blocks = blocks_of(doc, tmp_path)
    assert [b["order"] for b in blocks] == list(range(4))
    assert [b["kind"] for b in blocks] == ["heading", "paragraph", "equation", "code"]


def test_block_text_matches_what_the_markdown_contains(doc, tmp_path):
    doc.add_text(DocItemLabel.TEXT, "Body text.", prov=prov())
    doc.add_formula("x + y", prov=prov())
    doc.add_code("print(1)", prov=prov())
    markdown, *_, blocks = build(doc, tmp_path)
    for block in blocks:
        assert block["text"] in markdown


def test_code_block_is_fenced_with_its_language(doc, tmp_path):
    doc.add_code("print(1)", code_language="Python", prov=prov())
    block = blocks_of(doc, tmp_path)[0]
    assert block["text"] == "```Python\nprint(1)\n```"
    assert block["language"] == "Python"


def test_footnotes_become_endnotes_with_their_own_position(doc, tmp_path):
    doc.add_text(DocItemLabel.TEXT, "Before.", prov=prov())
    doc.add_text(DocItemLabel.FOOTNOTE, "A side remark.", prov=prov(page=2))
    doc.add_text(DocItemLabel.TEXT, "After.", prov=prov())
    blocks = blocks_of(doc, tmp_path, footnotes="endnotes")
    assert [b["kind"] for b in blocks] == ["paragraph", "paragraph", "heading", "footnote"]
    assert blocks[-1]["page"] == 2


def test_asset_index_stays_out_of_the_sidecar(doc, tmp_path):
    doc.add_picture(prov=prov())
    markdown, *_, blocks = build(doc, tmp_path)
    assert "Asset Reference Index" in markdown
    assert [b["kind"] for b in blocks] == ["figure"]


def test_figure_without_a_caption_has_no_retrievable_text(doc, tmp_path):
    doc.add_picture(prov=prov())
    block = blocks_of(doc, tmp_path)[0]
    assert block["text"] == ""
    assert block["image_path"] is None     # the test document has no pixels


def test_sidecar_round_trips_through_the_chunker(doc, tmp_path):
    import json

    doc.add_heading("3 Method", level=1, prov=prov())
    doc.add_heading("3.1 Setup", level=1, prov=prov())
    doc.add_text(DocItemLabel.TEXT, "The energy is given by:", prov=prov(page=1))
    doc.add_formula("E = m", orig="E = m (1)", prov=prov(page=2))
    doc.add_text(DocItemLabel.TEXT, "where m is the mass.", prov=prov(page=2))
    blocks = blocks_of(doc, tmp_path)

    path = tmp_path / "blocks.jsonl"
    path.write_text("\n".join(json.dumps(b) for b in blocks), encoding="utf-8")
    chunks = chunk_blocks(load_blocks_jsonl(path))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.heading_path == ["3 Method", "3.1 Setup"]
    assert "The energy is given by:" in chunk.text
    assert "\\tag{1}" in chunk.text
    assert "where m is the mass." in chunk.text
    assert (chunk.page_start, chunk.page_end) == (1, 2)


def test_vlm_converter_keeps_the_models_latex():
    """Backend text swaps the model's LaTeX for the flattened PDF text layer."""
    converter = build_vlm_converter("granite")
    options = converter.format_to_options[InputFormat.PDF].pipeline_options
    assert options.force_backend_text is False
