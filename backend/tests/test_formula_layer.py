"""Restoring what the text layer knows: alphabets and equation numbers."""
import json
import subprocess
from pathlib import Path

import pytest

from app.ingest.formula_layer import (
    BLACKBOARD,
    BOLD_ITALIC,
    BOLD_UPRIGHT,
    CALLIGRAPHIC,
    Glyph,
    Word,
    UNNAMED,
    assign_numbers,
    glyph_marks,
    glyph_symbols,
    glyphs_in,
    has_unnamed_glyphs,
    latex_marks,
    latex_symbols,
    page_trace,
    sign_of_glyph_name,
    page_words,
    font_style,
    junk_reason,
    printed_numbers,
    restore_styles,
    strip_number,
    tag_for,
    text_layer,
    wrap_alignment,
)


def glyphs(*runs):
    """glyphs(("xy", BOLD_ITALIC), ("=", None)) -> one Glyph per character."""
    return [Glyph(char, style) for text, style in runs for char in text]


@pytest.mark.parametrize("font, style", [
    ("EYUVZO+CMMIB10", BOLD_ITALIC),
    ("CMMIB7", BOLD_ITALIC),
    ("YKQKAE+CMBX10", BOLD_UPRIGHT),
    ("LMRoman10-Bold", BOLD_UPRIGHT),
    ("HVESJX+CharterBT-Bold", BOLD_UPRIGHT),
    ("dsrom10", BLACKBOARD),
    ("ABCDEF+MSBM10", BLACKBOARD),
    ("YUTRNV+CMSY10", CALLIGRAPHIC),
    ("CMMI10", None),
    ("CMR10", None),
    ("CharterBT-Roman", None),
    ("LMRoman10-Regular", None),
])
def test_font_names_map_to_alphabets(font, style):
    assert font_style(font) == style


def test_bold_letters_are_restored():
    latex = r"\text{tr}(x y^{\top})=y^{\top}x"
    page = glyphs(("tr", None), ("xyyx", BOLD_ITALIC))
    assert restore_styles(latex, page) == (
        r"\text{tr}(\boldsymbol{x} \boldsymbol{y}^{\top})=\boldsymbol{y}^{\top}\boldsymbol{x}"
    )


def test_a_bold_and_a_plain_instance_of_one_letter_stay_apart():
    latex = r"SRR(y,\hat{y})=\sum_{i}y_{i}"
    page = glyphs(("SRR", None), ("yy", BOLD_UPRIGHT), ("i", None), ("yi", None))
    assert restore_styles(latex, page) == (
        r"SRR(\mathbf{y},\hat{\mathbf{y}})=\sum_{i}y_{i}"
    )


def test_a_symbol_that_does_not_align_uses_its_letter_only_if_unambiguous():
    # The model wrote A where the page has an unreadable glyph, then a later A.
    latex = r"A+b+c+d+A"
    page = glyphs(("bcd", None), ("A", BOLD_UPRIGHT))
    assert restore_styles(latex, page) == r"\mathbf{A}+b+c+d+\mathbf{A}"


def test_greek_letters_are_restored_and_variants_match():
    assert restore_styles(r"\phi+\lambda", glyphs(("ϕλ", BOLD_ITALIC))) == (
        r"\boldsymbol{\phi}+\boldsymbol{\lambda}"
    )


def test_upright_bold_greek_uses_boldsymbol():
    assert restore_styles(r"\Sigma", glyphs(("Σ", BOLD_UPRIGHT))) == r"\boldsymbol{\Sigma}"


def test_blackboard_and_calligraphic_capitals():
    latex = r"\mathcal{L}+L+E[x]\in R"
    page = glyphs(("L", CALLIGRAPHIC), ("L", CALLIGRAPHIC), ("E", BLACKBOARD), ("x", None),
                  ("R", BLACKBOARD))
    assert restore_styles(latex, page) == r"\mathcal{L}+\mathcal{L}+\mathbb{E}[x]\in \mathbb{R}"


def test_calligraphic_fonts_never_style_lowercase_or_digits():
    assert restore_styles(r"x+1", glyphs(("x1", CALLIGRAPHIC))) == r"x+1"


def test_styled_groups_are_left_alone_but_still_align():
    latex = r"\mathbf{X}+\boldsymbol{\phi}+a"
    page = glyphs(("Xφ", BOLD_UPRIGHT), ("a", None))
    assert restore_styles(latex, page) == latex


def test_a_bold_word_in_text_becomes_one_group():
    latex = r"\text{Softmax}[x]"
    page = glyphs(("Softmax", BOLD_UPRIGHT), ("x", BOLD_ITALIC))
    assert restore_styles(latex, page) == r"\mathbf{Softmax}[\boldsymbol{x}]"


def test_plain_text_stays_text():
    assert restore_styles(r"\text{and}\ x", glyphs(("and", None), ("x", None))) == r"\text{and}\ x"


@pytest.mark.parametrize("latex", [
    r"\begin{bmatrix}x\\y\end{bmatrix}",
    r"\begin{array}{r l}x&y\end{array}",
])
def test_environment_names_and_column_specs_are_not_symbols(latex):
    restored = restore_styles(latex, glyphs(("xy", BOLD_ITALIC)))
    assert restored.startswith(latex[:latex.index("x")])
    assert r"\boldsymbol{x}" in restored and r"\boldsymbol{y}" in restored


def test_operator_letters_are_expected_but_never_styled():
    assert restore_styles(r"\log x", glyphs(("log", None), ("x", BOLD_ITALIC))) == (
        r"\log \boldsymbol{x}"
    )


def test_a_bold_operator_stays_an_operator():
    assert restore_styles(r"\max_{m}x", glyphs(("max", BOLD_UPRIGHT), ("m", None), ("x", None))) == (
        r"\mathop{\mathbf{max}}_{m}x"
    )


def test_digits_can_be_bold():
    assert restore_styles(r"x+0", glyphs(("x", None), ("0", BOLD_UPRIGHT))) == r"x+\mathbf{0}"


def test_latex_symbols_are_what_a_pdf_would_print():
    latex = r"\boldsymbol{x}_{1}=\sin\alpha + \text{for} \tag{2.1}"
    assert latex_symbols(latex) == list("x1sinαfor")


def test_glyph_symbols_keep_letters_digits_and_greek_only():
    glyphs = [Glyph(char, None) for char in "x=\ufb01ϕ2∑"]
    assert glyph_symbols(glyphs) == ["x", "f", "i", "φ", "2"]


@pytest.mark.parametrize("latex, marks", [
    (r"\left[\frac{1}{2}\right]\tag{1.1}", "[]"),                  # a tag prints nothing here
    (r"\begin{bmatrix}1 \\ -2\end{bmatrix}", "[−]"),
    (r"\begin{pmatrix}1\end{pmatrix}", "()"),
    (r"\begin{cases}a\end{cases}", "{"),                            # one brace only
    (r"\begin{array}{r|l}a\end{array}", ""),                        # a column spec prints nothing
    (r"a \\[2pt] b", ""),                                           # nor does a line skip
    (r"\hspace{-1pt}x", ""),
    (r"\{x\}", "{}"),
    (r"p(x)\propto\sum_{i}\hat{\theta}^{\prime}", "()∝∑ˆ′"),
    (r"x\leq y\neq z", "≤=\u0338"),
    (r"\underbrace{x}_{a}", "⏟"),
    (r"\text{non-negative}", "−"),
    (r"x\,y\!z\;w\:v", ""),                                         # spacing isn't punctuation
    (r"a \cdot b \cdots c", ""),                                     # dots aren't counted
])
def test_latex_marks(latex, marks):
    assert "".join(latex_marks(latex)) == marks


def test_glyph_marks_share_the_vocabulary_of_the_latex():
    glyphs = [Glyph(char, None) for char in "x−⩽⌢∗a"]
    assert glyph_marks(glyphs) == ["−", "≤", "ˆ", "∗"]


@pytest.mark.parametrize("page, marks", [
    ("a=⇒b", "⇒"),
    ("a⇐=⇒b", "⇔"),
    ("a−→b", "→"),
    ("a=b⇒c", "=⇒"),                     # a letter between them: two separate signs
])
def test_a_long_arrow_is_one_sign(page, marks):
    assert "".join(glyph_marks([Glyph(char, None) for char in page])) == marks


@pytest.mark.parametrize("name, sign", [
    ("bracketleftbigg", "["), ("bracketrightBigg", "]"), ("parenleftbig", "("), ("braceleftBig", "{"),
    ("parenlefttp", "("), ("parenleftex", ""), ("parenleftbt", ""),   # a tall one counts once
    ("summationdisplay", "∑"), ("summationtext", "∑"), ("productdisplay", "∏"),
    ("integraldisplay", "∫"), ("radicalbig", "√"), ("hatwide", "ˆ"), ("tildewider", "˜"),
    ("bracehtipupleft", "⏟"), ("bracehtipdownright", ""), ("vextendsingle", ""), ("unknown", ""),
])
def test_sign_of_glyph_name(name, sign):
    assert sign_of_glyph_name(name) == sign


class FakeParent:
    def extract_font(self, xref):
        return ("CMEX10", "n/a", "Type1", b"")


class FakePage:
    parent = FakeParent()

    def __init__(self, spans):
        self._spans = spans

    def get_texttrace(self):
        return self._spans

    def get_fonts(self):
        return [(1, "n/a", "Type1", "ABCDEF+CMEX10", "F1", "")]


def span(*chars, font="CMEX10"):
    return {"type": 0, "font": font, "chars": list(chars)}


def test_a_tall_bracket_counts_once_in_the_trace():
    pieces = [(0xF8EE, 1, (10.0, 5.0), (10, 5, 15, 15)), (0xF8EF, 2, (10.0, 15.0), (10, 15, 15, 25)),
              (0xF8F0, 3, (10.0, 25.0), (10, 25, 15, 35))]
    assert [glyph.char for glyph in page_trace(FakePage([span(*pieces)]))] == ["["]


def test_a_delimiter_counts_if_its_origin_or_its_middle_is_in_the_box():
    hanging = Glyph("[", None, 10.0, 90.0, 110.0)      # origin above the box, middle inside
    above_both = Glyph("[", None, 10.0, 90.0, 95.0)
    assert glyphs_in([hanging, above_both], [0.0, 100.0, 50.0, 200.0]) == [hanging]


def test_a_letter_counts_by_its_origin_alone():
    letter = Glyph("x", None, 10.0, 90.0)
    assert glyphs_in([letter], [0.0, 100.0, 50.0, 200.0]) == []


def test_a_glyph_with_no_name_stays_marked_as_unknown():
    trace = page_trace(FakePage([span((0xFFFD, 7, (10.0, 5.0), (10, 5, 15, 15)))]))
    assert [glyph.char for glyph in trace] == [UNNAMED]
    assert has_unnamed_glyphs(trace)


def test_a_tag_is_not_read_as_symbols():
    assert restore_styles(r"x \tag{1.2}", glyphs(("x", BOLD_ITALIC), ("12", None))) == (
        r"\boldsymbol{x} \tag{1.2}"
    )


def test_a_formula_that_does_not_match_the_page_is_left_alone():
    assert restore_styles(r"a+b+c+d", glyphs(("wxyz", BOLD_ITALIC))) == r"a+b+c+d"


def test_nothing_to_compare_means_no_change():
    assert restore_styles(r"x", []) == r"x"
    assert restore_styles(r"+", glyphs(("x", BOLD_ITALIC))) == r"+"


# --- Equation numbers --------------------------------------------------------


def line(*pieces, y=100.0):
    return [Word(text, x0, y, x1, y + 10) for text, x0, x1 in pieces]


def test_a_number_alone_at_the_end_of_a_line_is_found():
    words = line(("x", 100, 110), ("=", 112, 118), ("1", 120, 126), ("(4.20)", 300, 330))
    assert [w.text for w in printed_numbers(words, 400)] == ["(4.20)"]


@pytest.mark.parametrize("pieces", [
    [("=", 200, 206), ("f", 208, 212), ("(2)", 214, 228)],       # no gap: it belongs to f
    [("see", 100, 120), ("equation", 122, 160), ("(3.1)", 164, 190), ("again", 194, 230)],
    [("(4.20)", 20, 50)],                                         # left half of the page
    [("(iv)", 300, 320)],
])
def test_other_brackets_are_not_numbers(pieces):
    assert printed_numbers(line(*pieces), 400) == []


@pytest.mark.parametrize("text", ["(9)", "(2.28)", "(13.21)", "(4.67a)", "(A.3)"])
def test_number_shapes(text):
    assert len(printed_numbers(line((text, 300, 340)), 400)) == 1


def test_a_margin_note_citing_a_number_is_not_a_number():
    words = [Word("(2.28)", 300, 100, 330, 110), Word("(2.28)", 340, 150, 362, 157)]
    assert [w.y0 for w in printed_numbers(words, 400)] == [100]


def test_a_number_goes_to_the_box_it_sits_in():
    numbers = [Word("(2.1)", 300, 110, 330, 120), Word("(2.2)", 300, 150, 330, 160)]
    boxes = [(50, 100, 290, 130), (50, 140, 290, 170)]
    assert assign_numbers(boxes, numbers) == [["2.1"], ["2.2"]]


def test_a_number_between_overlapping_boxes_goes_to_the_nearest_centre():
    number = [Word("(2.1)", 300, 118, 330, 126)]
    assert assign_numbers([(50, 100, 290, 124), (50, 118, 290, 170)], number) == [["2.1"], []]


def test_a_block_with_two_lines_keeps_both_numbers():
    numbers = [Word("(4.67b)", 300, 150, 330, 160), Word("(4.67a)", 300, 110, 330, 120)]
    assert assign_numbers([(50, 100, 290, 170)], numbers) == [["4.67a", "4.67b"]]


def test_a_number_outside_every_box_is_dropped():
    assert assign_numbers([(50, 100, 290, 130)], [Word("(2.1)", 300, 200, 330, 210)]) == [[]]


@pytest.mark.parametrize("latex", [
    r"x=1\quad(1 2 . 1 1)",
    r"x=1 \quad \quad(1 2 . 1 1)",
    r"x=1 & &(1 2 . 1 1)",
    r"x=1 \text{(12.11)}",
    r"x=1 \tag{12.11}",
    r"x=1 \tag{1 2 . 1 1}",
])
def test_a_copied_number_is_removed_however_it_is_spaced(latex):
    assert strip_number(latex, "12.11").strip() == "x=1"


def test_a_bracket_that_belongs_to_the_formula_survives():
    assert strip_number(r"f(2)=1", "3.1") == r"f(2)=1"
    assert strip_number(r"(1+x)^{2}", "1") == r"(1+x)^{2}"


def test_tags():
    assert tag_for([]) == ""
    assert tag_for(["4.20"]) == r" \tag{4.20}"
    assert tag_for(["4.67a", "4.67b"]) == r" \tag{4.67a, 4.67b}"


# --- A real PDF --------------------------------------------------------------


def test_fonts_and_numbers_are_read_from_a_pdf():
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    page = document.new_page(width=400, height=200)
    page.insert_text((60, 100), "x", fontname="hebo", fontsize=12)
    page.insert_text((75, 100), "+y", fontname="helv", fontsize=12)
    page.insert_text((320, 100), "(4.20)", fontname="helv", fontsize=12)

    box = [50, 85, 390, 105]
    assert restore_styles("x+y", glyphs_in(page_trace(page), box)) == r"\mathbf{x}+y"
    words = page_words(page)
    assert assign_numbers([tuple(box)], printed_numbers(words, 400)) == [["4.20"]]


# --- Bare alignment ----------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
NODE = ROOT / ".venv" / "bin" / "node"
KATEX = ROOT / "frontend" / "node_modules" / "katex"


def compiles(latex: str) -> bool:
    if not (NODE.exists() and KATEX.exists()):
        pytest.skip("KaTeX is not installed; run npm ci in frontend/")
    result = subprocess.run(
        [str(NODE), str(ROOT / "eval" / "katex_check.mjs")],
        input=json.dumps([latex]), capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)[0]["ok"]


@pytest.mark.parametrize("latex", [
    r"A & = B",
    r"A & = B \\ & = C",
    r"(x+y)C & = \begin{bmatrix}1 & 2\\3 & 4\end{bmatrix}",
    r"x = 1 & & y = 2",
])
def test_a_bare_ampersand_is_wrapped_and_compiles(latex):
    assert not compiles(latex)
    wrapped = wrap_alignment(latex)
    assert wrapped.startswith(r"\begin{aligned}") and wrapped.endswith(r"\end{aligned}")
    assert compiles(wrapped)
    assert compiles(wrapped + r" \tag{4.67a, 4.67b}")


@pytest.mark.parametrize("latex", [
    r"\begin{bmatrix}1 & 2\end{bmatrix}",
    r"\begin{array}{r l}x & y\end{array}+\begin{pmatrix}1 & 0\end{pmatrix}",
    r"\begin{aligned}a & = b\end{aligned}",
    r"\text{R\&D}+x",
])
def test_an_ampersand_that_is_already_valid_is_left_alone(latex):
    assert wrap_alignment(latex) == latex


@pytest.mark.parametrize("latex, tidied", [
    (r"x=1 \\", "x=1"),
    (r"x=1. \\ ", "x=1."),
    (r"x=1\, \quad", "x=1"),
    (r"x=1", "x=1"),
])
def test_a_trailing_line_break_is_removed(latex, tidied):
    assert wrap_alignment(latex) == tidied


# --- Junk --------------------------------------------------------------------


@pytest.mark.parametrize("latex", [
    r"E=mc^{2}",
    r"\frac{\partial f}{\partial x}\Big|_{x=0}=1 \tag{3.1}",
    r"\begin{bmatrix}0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0\end{bmatrix}",
    r"S T F T(x) \quad \text{for all}\ x",
    r"f(x)=\begin{cases}1 & \text{if }x>0\\0 & \text{otherwise}\end{cases}",
    r"\mathcal{L}_{h o r i z o n}=\sum_{h}\beta^{h}\ell_{h}",
    r"\sum_{i=1}^{n}x_{i}+\sum_{j=1}^{n}y_{j}+\sum_{k=1}^{n}z_{k}",
    r"x_{1}+x_{2}+x_{3}+x_{4}+x_{5}+x_{6}+x_{7}+x_{8}+x_{9}",
])
def test_real_formulas_are_kept(latex):
    assert junk_reason(latex) is None


@pytest.mark.parametrize("latex, reason", [
    (r"(2 . 2 8)", "number"),
    (r"(6 . 8 4 a)", "number"),
    (r"\tag{2.28}", "number"),
    (r"L[\phi]=\sum_{i}\ell_{i}. \\ \intertext{l e d}\int", "prose"),
    (r"x=1 \\ o p t i m i z a t i o n", "prose"),
    (r"x=1 \\ \text{returns the element-wise maximum of the vectors}", "prose"),
    (r"\{1\}\\ " + r"\vdots \cdot \quad " * 40, "repeat"),
    (r"x=0 9 9 9 9 9 9 9 9 9 9 9", "repeat"),
    (r"{1}{0}" * 12, "repeat"),
    (r"\vdots \, , \, " * 9, "repeat"),
    ("x_{1}" + "+".join(f"a_{{{i}}}" for i in range(600)), "length"),
])
def test_junk_is_recognised(latex, reason):
    assert junk_reason(latex) == reason


def run(text: str, start: float = 0.0, step: float = 5.0) -> list[Glyph]:
    return [Glyph(char, None, start + i * step, 0.0) for i, char in enumerate(text)]


def test_the_text_layer_keeps_the_printed_symbols_in_drawing_order():
    assert text_layer(run("gZ(ce)")) == "gZ(ce)"


def test_a_jump_backwards_in_the_text_layer_starts_a_new_group():
    assert text_layer(run("ab", start=50) + run("cd", start=10)) == "ab cd"


def test_a_slash_moves_from_before_its_letter_to_after_it():
    assert text_layer(run("\u0338k1")) == "k\u03381"


def test_a_box_with_no_glyphs_has_no_text_layer():
    assert text_layer([]) == ""
