"""The streaming citation filter, fed in every way a model can split its text."""
import time

import pytest

from app.llm.citations import CitationFilter, strip_citations


def run(text: str, provided=(1, 2, 3), size: int | None = None) -> tuple[str, CitationFilter]:
    citations = CitationFilter(set(provided))
    pieces = [text] if size is None else [text[i:i + size] for i in range(0, len(text), size)]
    shown = "".join(citations.feed(piece) for piece in pieces) + citations.flush()
    return shown, citations


@pytest.mark.parametrize("size", [None, 1, 2, 3, 5])
@pytest.mark.parametrize("text, expected", [
    ("Plain text.", "Plain text."),
    ("Claim [S1].", "Claim [S1]."),
    ("Claim [S9].", "Claim."),
    ("One [S1] two [S9] three [S2].", "One [S1] two three [S2]."),
    ("Both [S1, S2].", "Both [S1][S2]."),
    ("Both [S1,S2,S3]", "Both [S1][S2][S3]"),
    ("Mixed [S1, S9].", "Mixed [S1]."),
    ("Short [S1, 2].", "Short [S1][S2]."),
    ("Semi [S1; S2].", "Semi [S1][S2]."),
    ("Spaced [S 2].", "Spaced [S2]."),
    ("Two digits [S12].", "Two digits."),
    ("Italic [*S2*].", "Italic [S2]."),
    ("Bold [**S2**].", "Bold [S2]."),
    ("(See S1, p. 196–197.)", "(See [S1], p. 196–197.)"),
    ("[See S1, p. 13]", "[See [S1], p. 13]"),
    ("(S3, p. 197, describes it.)", "([S3], p. 197, describes it.)"),
    ("(S2, equations 4.14–4.15)", "([S2], equations 4.14–4.15)"),
    ("as in (S1, Fig. 3).", "as in ([S1], Fig. 3)."),
    ("See S1 and S3.", "See [S1][S3]."),
    ("see S1, S2 for more", "see [S1][S2] for more"),
    ("Ends with See S2", "Ends with See [S2]"),
    ("See S9 for it.", " for it."),
    ("See S1 and S9.", "See [S1]."),
    ("See S1 and more.", "See [S1] and more."),
    ("See the book.", "See the book."),
    ("Seeds grow. Sets too.", "Seeds grow. Sets too."),
    ("See", "See"),
    ("See ", "See "),
    ("See S", "See S"),
    ("Such as S1 alone", "Such as S1 alone"),
    ("Math (S_1, p) stays.", "Math (S_1, p) stays."),
    ("Zero width [\u200bS2].", "Zero width [S2]."),
    ("Zero width invented [\u200bS9].", "Zero width invented."),
    ("Lenticular 【S2】.", "Lenticular [S2]."),
    ("Lenticular invented 【S9】.", "Lenticular invented."),
    ("Round (S2).", "Round [S2]."),
    ("Round list (S1, S3).", "Round list [S1][S3]."),
    ("Round invented (S9).", "Round invented."),
    ("Range (S1–S5) stays.", "Range (S1–S5) stays."),
    ("Plain (see above) stays.", "Plain (see above) stays."),
    ("Function f(x) stays.", "Function f(x) stays."),
    ("Ends with (", "Ends with ("),
    ("Ends with (S", "Ends with (S"),
    ("Ends with (S1", "Ends with (S1"),
    ("Math $E[S_1]$ stays.", "Math $E[S_1]$ stays."),
    ("Math $f(S_1, S_2)$ stays.", "Math $f(S_1, S_2)$ stays."),
    ("Math $P(S_1)$ stays.", "Math $P(S_1)$ stays."),
    ("Math \\(g(S_2)\\) stays.", "Math \\(g(S_2)\\) stays."),
    ("Set [S_1, S_2] stays.", "Set [S_1, S_2] stays."),
    ("Italic list [*S1*, *S2*].", "Italic list [S1][S2]."),
    ("Italic invented [*S9*].", "Italic invented."),
    ("Starred [*note*] stays.", "Starred [*note*] stays."),
    ("Spaced [ 0, 1 ] stays.", "Spaced [ 0, 1 ] stays."),
    ("Interval [0, 1] stays.", "Interval [0, 1] stays."),
    ("Bracket [a] stays.", "Bracket [a] stays."),
    ("Word [See this].", "Word [See this]."),
    ("Set [S] stays.", "Set [S] stays."),
    ("Index x[1] stays.", "Index x[1] stays."),
    ("Ends with [", "Ends with ["),
    ("Ends with [S", "Ends with [S"),
    ("Ends with [S1", "Ends with [S1"),
    ("[S1] first", "[S1] first"),
    ("[S9] first", " first"),
    ("Double [[S1]]", "Double [[S1]]"),
    ("Spaces   kept  here", "Spaces   kept  here"),
    ("Trailing space ", "Trailing space "),
])
def test_text_comes_out_the_same_however_it_is_split(text, expected, size):
    assert run(text, size=size)[0] == expected


def test_cited_ids_are_listed_once_in_order_of_first_use():
    assert run("[S2] then [S1] then [S2]")[1].cited == [2, 1]


def test_dropped_ids_are_listed_once():
    assert run("[S7] and [S7, S8]")[1].dropped == [7, 8]


def test_a_partial_citation_is_held_back_not_shown():
    citations = CitationFilter({1})
    assert citations.feed("Claim [S") == "Claim"
    assert citations.feed("1") == ""
    assert citations.feed("] done") == " [S1] done"


@pytest.mark.parametrize("held", ["See S1,", "See S1, S", "See S1 a", "See S1 and", "See S1 and S", "See S1, S2 an"])
def test_an_unfinished_see_list_is_held_back(held):
    assert CitationFilter({1, 2}).feed("x " + held) == "x"


def test_a_long_see_near_miss_is_decided_at_once():
    """Review R2-C1: spaces in "See S1      ..." could split many ways (0.19 s at 18 items)."""
    near_miss = "See S1" + "".join(f", S{n}" for n in range(2, 40)) + " " * 30 + "and the rest."
    started = time.perf_counter()
    shown, _ = run(near_miss, provided=range(1, 40), size=1)
    assert shown.startswith("See [S1]") and time.perf_counter() - started < 0.5


def test_an_invented_id_is_never_visible_at_any_point():
    citations = CitationFilter({1})
    shown = [citations.feed(piece) for piece in ("x [", "S", "9", "]", ".")]
    assert not any("9" in text or "S" in text for text in shown)


def test_with_no_sources_every_citation_goes():
    assert run("Claim [S1].", provided=())[0] == "Claim."


def test_strip_citations_cleans_old_answers():
    assert strip_citations("Claim [S1] and [S2, S3] and [*S4*].") == "Claim and and."


def filtered(text, provided=(1, 2, 3), pieces=None):
    citations = CitationFilter(set(provided))
    out = ""
    for piece in pieces or [text]:
        out += citations.feed(piece)
    return out + citations.flush(), citations


@pytest.mark.parametrize("written, kept, cited", [
    (r"x \tag{S2}", r"x \tag{S2}", [2]),
    (r"x \tag{S1, S3}", r"x \tag{S1, S3}", [1, 3]),
    (r"x \tag{ S2 }", r"x \tag{S2}", [2]),
    (r"x \tag*{S2}", r"x \tag{S2}", [2]),
    (r"x \tag{S1:4.106}", r"x \tag{S1: 4.106}", [1]),
    (r"x \tag{S1: one-standard-error rule}", r"x \tag{S1: one-standard-error rule}", [1]),
])
def test_a_tag_naming_a_source_counts_as_a_citation(written, kept, cited):
    out, citations = filtered(written)
    assert out == kept and citations.cited == cited


def test_an_invented_source_in_a_tag_is_removed():
    out, citations = filtered(r"x \tag{S9}")
    assert out == "x " and citations.dropped == [9] and citations.cited == []


def test_an_invented_source_keeps_the_rest_of_its_tag():
    out, citations = filtered(r"x \tag{S1, S9}")
    assert out == r"x \tag{S1}" and citations.dropped == [9]


def test_an_invented_source_leaves_a_real_equation_label():
    assert filtered(r"x \tag{S9: 4.106}")[0] == r"x \tag{4.106}"


@pytest.mark.parametrize("text", [
    r"E = mc^2 \tag{4.106}", r"x \tag{1}", r"x \tag{Saddle point}", r"\theta + \tau \times \text{S1}",
    r"\tan x", r"a \\ b", "\\",
])
def test_ordinary_math_is_left_alone(text):
    out, citations = filtered(text)
    assert out == text and citations.cited == []


@pytest.mark.parametrize("pieces", [
    ["x \\", "tag{S2}"], ["x \\t", "ag{S2}"], ["x \\ta", "g{S", "2}"], ["x \\tag", "{", "S2", "}"], list(r"x \tag{S2}"),
])
def test_a_tag_split_across_tokens_is_still_read(pieces):
    out, citations = filtered("", pieces=pieces)
    assert out == r"x \tag{S2}" and citations.cited == [2]


def test_an_unfinished_tag_at_the_end_is_released_as_written():
    assert filtered(r"x \tag{S2")[0] == r"x \tag{S2"


def test_stripping_removes_a_tag_but_keeps_a_real_label():
    assert strip_citations(r"x \tag{S2} y") == "x  y"
    assert strip_citations(r"x \tag{S1: 4.106}") == r"x \tag{4.106}"
    assert strip_citations(r"x \tag{4.106}") == r"x \tag{4.106}"

