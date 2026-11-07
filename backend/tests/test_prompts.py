"""The answer prompt, and how sources and history go into it."""
import pytest

from app.chat import regions_of, unverified_crops
from app.db.connection import connect
from app.ingest.chunker import unverified_note
from app.llm.prompts import (
    NO_TEXT_LAYER,
    REMINDER,
    SYSTEM_PROMPT,
    TEXT_LAYER_LABEL,
    build_messages,
    format_sources,
    page_label,
    tools_wanted,
    text_for_model,
)
from app.retrieval.search import Hit


def hit(text="Body", start=3, end=3, title="Book", meta=None, doc_id=1) -> Hit:
    return Hit(chunk_id=1, doc_id=doc_id, doc_path="a.pdf", doc_title=title, heading_path=[], text=text,
               page_start=start, page_end=end, block_ids=[], meta=meta or {}, score=1.0)


def test_sources_are_numbered_from_one():
    text = format_sources([hit("First"), hit("Second")])
    assert text.index("[S1] Book, p. 3\nFirst") < text.index("[S2] Book, p. 3\nSecond")


def test_page_labels():
    assert page_label(hit(start=3, end=3)) == "p. 3"
    assert page_label(hit(start=3, end=5)) == "pp. 3-5"
    assert page_label(hit(start=None, end=None)) == "page unknown"


MARKED = f"Intro\n\n{unverified_note(42)}\n$$\ng_z = \\bar k\n$$\n\nOutro"
SYMBOLS = {"page": 42, "crop": "c.png", "text_layer": "gZ=k1"}


def test_the_model_gets_the_symbols_not_the_flagged_latex():
    content = build_messages("Q", [hit(MARKED, meta={"unverified_formulas": [SYMBOLS]})], [])[-1]["content"]
    assert f"{unverified_note(42)}\n{TEXT_LAYER_LABEL} gZ=k1" in content
    assert "g_z" not in content and "\\bar" not in content


def test_the_text_around_a_flagged_formula_stays():
    assert text_for_model(MARKED, [SYMBOLS]).startswith("Intro\n\n")
    assert text_for_model(MARKED, [SYMBOLS]).endswith("\n\nOutro")


def test_unflagged_formulas_keep_their_latex():
    text = f"$$ a = b $$\n{MARKED}\n$$ c = d $$"
    out = text_for_model(text, [SYMBOLS])
    assert "$$ a = b $$" in out and "$$ c = d $$" in out and "g_z" not in out


def test_several_flagged_formulas_keep_their_order():
    text = f"{unverified_note(1)}\n$$ a $$\nand\n{unverified_note(2)}\n$$ b $$"
    out = text_for_model(text, [{"text_layer": "first"}, {"text_layer": "second"}])
    assert out.index("first") < out.index("second")
    assert f"{unverified_note(2)}\n{TEXT_LAYER_LABEL} second" in out


def test_flagged_latex_never_gets_through_without_symbols():
    for formulas in ([], [{"text_layer": ""}], [{"page": 42}]):
        out = text_for_model(MARKED, formulas)
        assert "g_z" not in out and unverified_note(42) in out and NO_TEXT_LAYER in out


def test_the_stored_hit_keeps_its_latex():
    flagged = hit(MARKED, meta={"unverified_formulas": [SYMBOLS]})
    format_sources([flagged])
    assert "g_z" in flagged.text


def test_the_prompt_quotes_the_exact_marker():
    assert unverified_note("N") in SYSTEM_PROMPT


def test_the_prompt_says_how_to_treat_flattened_symbols():
    assert TEXT_LAYER_LABEL in SYSTEM_PROMPT
    assert "Don't write that formula out as fact" in SYSTEM_PROMPT


def test_the_prompt_says_to_cite_only_given_numbers():
    assert "never invent one" in SYSTEM_PROMPT


def test_the_prompt_rules_out_formatted_citations():
    assert "never bold or italic" in SYSTEM_PROMPT


def test_the_prompt_asks_for_explicit_math_delimiters():
    assert "$$...$$" in SYSTEM_PROMPT


def test_the_system_prompt_comes_first_and_the_question_last():
    messages = build_messages("Why?", [hit()], [{"role": "user", "content": "Earlier"}])
    assert [message["role"] for message in messages] == ["system", "user", "user"]
    assert "Question: Why?" in messages[-1]["content"]


def test_the_rules_are_repeated_after_the_sources():
    content = build_messages("Why?", [hit("A source")], [])[-1]["content"]
    assert content.endswith(REMINDER)
    assert content.index("A source") < content.index(REMINDER)


def test_the_reminder_covers_citations_and_unverified_formulas():
    assert "[S1]" in REMINDER and unverified_note("N") in REMINDER


def test_earlier_answers_lose_their_citations():
    history = [{"role": "assistant", "content": "It holds [S2]."}]
    assert build_messages("Q", [hit()], history)[1]["content"] == "It holds."


def test_crops_come_from_every_source_with_cited_ones_first():
    marked = {"unverified_formulas": [{"page": 7, "crop": "formula_crops/000003.png"}]}
    hits = [hit(meta=marked, doc_id=4), hit(), hit(meta=marked, doc_id=5)]
    assert [(c["source"], c["doc_id"], c["cited"]) for c in unverified_crops([3], hits)] == [
        (3, 5, True), (1, 4, False)]


def test_a_formula_in_two_sources_is_shown_once():
    marked = {"unverified_formulas": [{"page": 7, "crop": "formula_crops/000003.png"}]}
    crops = unverified_crops([2], [hit(meta=marked), hit(meta=marked)])
    assert [(crop["source"], crop["cited"]) for crop in crops] == [(2, True)]


def test_a_formula_without_a_stored_crop_is_skipped():
    assert unverified_crops([1], [hit(meta={"unverified_formulas": [{"page": 7, "crop": None}]})]) == []


def test_regions_fall_back_to_the_single_box(tmp_path):
    conn = connect(tmp_path / "a.sqlite")
    conn.execute("INSERT INTO documents (id, path, sha256, kind) VALUES (1, 'a', 'x', 'pdf')")
    conn.execute('INSERT INTO blocks (id, doc_id, "order", kind, page, bbox, text) '
                 "VALUES (1, 1, 1, 'paragraph', 2, '[1, 2, 3, 4]', 't')")
    conn.execute('INSERT INTO blocks (id, doc_id, "order", kind, page, bboxes_json, text) '
                 """VALUES (2, 1, 2, 'equation', 2, '[{"page": 3, "bbox": [5, 6, 7, 8]}]', 'e')""")
    conn.execute('INSERT INTO blocks (id, doc_id, "order", kind, text) VALUES (3, 1, 3, \'heading\', \'h\')')
    assert regions_of(conn, [1, 2, 3]) == [
        {"page": 2, "bbox": [1, 2, 3, 4]}, {"page": 3, "bbox": [5, 6, 7, 8]}]


def test_no_blocks_no_regions(tmp_path):
    assert regions_of(connect(tmp_path / "a.sqlite"), []) == []


def test_the_model_is_told_it_sees_only_passages():
    assert "the passages I was given don't mention X" in SYSTEM_PROMPT
    assert "Never say that the library" in SYSTEM_PROMPT


def test_the_rule_is_repeated_after_the_sources():
    text = build_messages("Q?", [], [])[-1]["content"]
    assert text.index("Question: Q?") < text.index("never the library")


def test_finding_nothing_is_not_put_as_a_fact_about_the_library():
    text = build_messages("Q?", [], [])[-1]["content"]
    assert "in the library" not in text and "returned no passages" in text


class TestToolsWanted:
    @pytest.mark.parametrize("question, tools", [
        ("Make a 5-slide deck on the Cholesky decomposition from MML.", ["make_pptx"]),
        ("Export this answer as a PDF.", ["make_pdf"]),
        ("Make a one-page cheat sheet on the SVD, as a PDF.", ["make_pdf"]),
        ("Write out the derivation in a PDF.", ["make_pdf"]),
        ("Erstelle ein kurzes PDF auf Deutsch.", ["make_pdf"]),
        ("Export this answer as a Word document.", ["make_docx"]),
        ("Save study notes as a Markdown file.", ["make_markdown"]),
        ("Check with SymPy that this holds.", ["sympy_check"]),
        ("Run this code and plot.\n\n```python\nprint(1)\n```", ["run_python"]),
        ("Make a deck on gradient descent. Run code that plots it.", ["run_python", "make_pptx"]),
    ])
    def test_a_request_names_its_tool(self, question, tools):
        assert tools_wanted(question) == tools

    @pytest.mark.parametrize("question", [
        "What is the Cholesky decomposition, and when does it exist?",
        "What's the pdf of a Gaussian?",
        "Derive the pdf of the sum of two uniform variables.",
        "How do I sample from a pdf?",
        "Why does the loss function run off to infinity?",
    ])
    def test_an_ordinary_question_gets_no_nudge(self, question):
        assert tools_wanted(question) == []

    def test_the_nudge_comes_after_the_question_only_with_tools_on(self):
        question = "Make a deck on eigenvalues."
        assert "make_pptx" in build_messages(question, [], [], tools=True)[-1]["content"]
        assert "make_pptx" not in build_messages(question, [], [])[-1]["content"]
