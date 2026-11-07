"""Chunker tests, grouped by the chunking rule each one protects."""
import random
import re

from dataclasses import replace

import pytest

from app.ingest.chunker import (
    MAX_TOKENS,
    DROPPED_EQUATION,
    TOKEN_SCALE,
    Block,
    blocks_from_markdown,
    chunk_blocks,
    chunk_markdown,
    continues_equation,
    estimate_tokens,
    split_sentences,
)


def prose(words: int, seed: str = "word") -> str:
    """A paragraph of about `words` estimated tokens, in ten-word sentences."""
    sentences = []
    for i in range(0, int(words / TOKEN_SCALE), 11):
        sentences.append(f"{seed.capitalize()}{i} " + " ".join([seed] * 9) + ".")
    return " ".join(sentences)


def kinds(markdown: str) -> list[str]:
    return [b.kind for b in blocks_from_markdown(markdown)]


def body(chunk) -> str:
    return chunk.text.split("›\n\n", 1)[-1]


# --- Block scanner: $$ blocks ------------------------------------------------


def test_multiline_display_math_is_one_block():
    md = "Intro.\n\n$$\na = b\n\n+ c\n$$\n\nOutro."
    blocks = blocks_from_markdown(md)
    assert [b.kind for b in blocks] == ["paragraph", "equation", "paragraph"]
    assert blocks[1].text == "$$\na = b\n\n+ c\n$$"


def test_single_line_display_math():
    assert kinds("$$ E = mc^2 $$") == ["equation"]


def test_opening_delimiter_sharing_a_line_with_content():
    assert kinds("$$ a = b\n+ c $$\n\nafter") == ["equation", "paragraph"]


def test_bracket_display_math():
    assert kinds("\\[\nx^2\n\\]\n\nafter") == ["equation", "paragraph"]


def test_markdown_lookalikes_inside_math_stay_math():
    md = "$$\n# not a heading\n| a | b |\n- x\n$$"
    assert kinds(md) == ["equation"]


def test_unterminated_display_math_does_not_swallow_the_document():
    md = "$$ broken\n\n# Next section\n\nMore text."
    blocks = blocks_from_markdown(md)
    assert "heading" in [b.kind for b in blocks]
    assert blocks[-1].text == "More text."


def test_adjacent_display_blocks_stay_separate():
    assert kinds("$$a$$\n$$b$$") == ["equation", "equation"]


# --- Block scanner: code fences ----------------------------------------------


def test_fence_protects_headings_math_and_blank_lines():
    md = "```python\n# comment\n\n$$ not math $$\n| not | table |\n```\n\nAfter."
    blocks = blocks_from_markdown(md)
    assert [b.kind for b in blocks] == ["code", "paragraph"]
    assert "# comment" in blocks[0].text


def test_tilde_fence():
    assert kinds("~~~\n```\nstill code\n~~~\nafter") == ["code", "paragraph"]


def test_longer_fence_contains_shorter_fence():
    md = "````md\n```\ninner\n```\n````\nafter"
    assert kinds(md) == ["code", "paragraph"]


def test_closing_fence_must_not_carry_an_info_string():
    md = "```\ncode\n```python\nmore code\n```\nafter"
    blocks = blocks_from_markdown(md)
    assert [b.kind for b in blocks] == ["code", "paragraph"]
    assert "more code" in blocks[0].text


def test_unterminated_fence_runs_to_the_end():
    assert kinds("```\ncode\n\n# inside") == ["code"]


# --- Block scanner: tables, lists, figures -----------------------------------


def test_pipe_table_is_one_block():
    md = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\nAfter."
    blocks = blocks_from_markdown(md)
    assert [b.kind for b in blocks] == ["table", "paragraph"]
    assert blocks[0].text.count("\n") == 3


def test_table_without_leading_pipes():
    assert kinds("a | b\n---|---\n1 | 2") == ["table"]


def test_table_cells_containing_inline_math_and_escaped_pipes():
    md = "| $a|b$ | x \\| y |\n|---|---|\n| 1 | 2 |"
    assert kinds(md) == ["table"]


def test_a_pipe_in_prose_is_not_a_table():
    assert kinds("The norm |x| is positive.") == ["paragraph"]


def test_paragraph_stops_at_an_adjacent_structure():
    md = "Text directly above\n$$\nx\n$$\nand below"
    assert kinds(md) == ["paragraph", "equation", "paragraph"]


def test_list_items_are_separate_blocks():
    assert kinds("- one\n- two\n1. three") == ["list_item"] * 3


def test_image_line_is_a_figure():
    blocks = blocks_from_markdown("![Figure 1: spectrum](figures/f1.png)")
    assert blocks[0].kind == "figure"
    assert blocks[0].meta["image_path"] == "figures/f1.png"


# --- Sentences ---------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "See Fig. 2 for details.",
    "Use a basis, e.g. Legendre polynomials.",
    "As shown in Eq. 3 and Ref. 4.",
    "Following Einstein et al. 1935 here.",
    "Nach Abb. 3 gilt dies, z.B. Lösungen der Gleichung.",
    "Written by A. Einstein alone.",
    "The value $x = 1. Then y$ stays inline.",
    "Call `a. B` here.",
])
def test_sentence_splitter_keeps_these_whole(text):
    assert split_sentences(text) == [text]


def test_sentence_splitter_splits_ordinary_prose():
    assert split_sentences("First one. Second one! Third? Fourth.") == [
        "First one.", "Second one!", "Third?", "Fourth."]


def test_sentence_may_start_with_inline_math():
    assert split_sentences("It holds. $x$ is real.") == ["It holds.", "$x$ is real."]


# --- Rule: split at H1-H3 ----------------------------------------------------


def test_each_h1_to_h3_starts_a_new_chunk():
    md = "# A\n\nAlpha text.\n\n## B\n\nBeta text.\n\n### C\n\nGamma text."
    assert [body(c) for c in chunk_markdown(md)] == ["Alpha text.", "Beta text.", "Gamma text."]


def test_h4_stays_inline_and_keeps_its_section():
    chunks = chunk_markdown("## B\n\nBefore.\n\n#### Detail\n\nAfter.")
    assert len(chunks) == 1
    assert "Detail" in chunks[0].text


def test_h4_is_not_stranded_from_the_following_paragraph():
    md = f"## B\n\n{prose(560)}\n\n#### Detail\n\n{prose(200, 'other')}"
    for chunk in chunk_markdown(md):
        if "Detail" in chunk.text:
            assert "other" in chunk.text


def test_a_heading_inside_a_code_fence_does_not_split():
    assert len(chunk_markdown("## A\n\n```\n# fake\n```\n\nText.")) == 1


def test_heading_without_body_makes_no_chunk_but_stays_in_the_path():
    chunks = chunk_markdown("# Ch\n\n## Sec\n\nBody.")
    assert len(chunks) == 1
    assert chunks[0].heading_path == ["Ch", "Sec"]


# --- Rule: 300-600 tokens ----------------------------------------------------


def test_long_section_packs_within_limits():
    chunks = chunk_markdown("# H\n\n" + "\n\n".join(prose(120, f"w{i}x") for i in range(30)))
    assert len(chunks) > 3
    assert all(c.token_count <= MAX_TOKENS for c in chunks)
    assert all(c.token_count >= 300 for c in chunks)


def test_short_final_chunk_is_absorbed_when_it_fits():
    md = "\n\n".join(prose(150, f"w{i}x") for i in range(4))
    tail = "Last short sentence here."
    chunks = chunk_markdown(md + "\n\n" + tail, max_tokens=600)
    assert tail in chunks[-1].text
    assert chunks[-1].token_count > 300


def test_paragraph_over_the_limit_splits_between_sentences():
    chunks = chunk_markdown(prose(1500))
    assert len(chunks) >= 3
    for chunk in chunks:
        assert chunk.token_count <= MAX_TOKENS
        assert chunk.text.rstrip().endswith(".")


# --- Rule: never split inside $$, code or tables -----------------------------


@pytest.mark.parametrize("atom", [
    "$$\n" + " + ".join(f"x_{{{i}}}" for i in range(250)) + "\n$$",
    "```python\n" + "\n".join(f"value_{i} = {i}" for i in range(250)) + "\n```",
    "| a | b |\n|---|---|\n" + "\n".join(f"| {i} | {i * 2} |" for i in range(200)),
])
def test_oversized_atomic_block_stays_whole(atom):
    assert estimate_tokens(atom) > MAX_TOKENS
    md = f"{prose(100)}\n\n{atom}\n\n{prose(100, 'other')}"
    holders = [c for c in chunk_markdown(md) if atom in c.text]
    assert len(holders) == 1


def test_chunks_never_end_inside_structure_in_a_mixed_document():
    rng = random.Random(1)
    atoms = {
        "equation": lambda i: f"$$\nf_{i}(x) = \\sum_k a_k x^k \\\\\n\ng_{i}(x) = 0\n$$",
        "code": lambda i: f"```python\n# step {i}\n\nprint({i})\n```",
        "table": lambda i: f"| k{i} | v |\n|---|---|\n| 1 | 2 |",
    }
    parts, expected = ["# Doc"], []
    for i in range(300):
        choice = rng.choice(["prose", "prose", "equation", "code", "table", "heading"])
        if choice == "prose":
            parts.append(prose(rng.randint(10, 200), f"w{i}x"))
        elif choice == "heading":
            parts.append(f"{'#' * rng.randint(1, 4)} Section {i}")
        else:
            atom = atoms[choice](i)
            parts.append(atom)
            expected.append(atom)
    chunks = chunk_markdown("\n\n".join(parts))
    for atom in expected:
        assert sum(atom in c.text for c in chunks) == 1, atom
    for chunk in chunks:
        assert len(re.findall(r"^\$\$$", chunk.text, re.M)) % 2 == 0
        assert len(re.findall(r"^```", chunk.text, re.M)) % 2 == 0
        assert chunk.token_count <= MAX_TOKENS


# --- Rule: keep an equation with its context --------------------------------


def test_lead_in_equation_and_where_sentence_stay_together():
    filler = prose(570)
    md = (f"{filler} The energy follows from the action:\n\n$$\nE = mc^2\n$$\n\n"
          f"where $m$ is the mass. {prose(200, 'other')}")
    holders = [c for c in chunk_markdown(md) if "E = mc^2" in c.text]
    assert len(holders) == 1
    assert "The energy follows from the action:" in holders[0].text
    assert "where $m$ is the mass." in holders[0].text


def test_only_the_last_sentence_is_pulled_in_not_the_paragraph():
    md = f"{prose(590)} Lead-in sentence here:\n\n$$\nx\n$$"
    chunks = chunk_markdown(md)
    holder = next(c for c in chunks if "$$" in c.text)
    assert "Lead-in sentence here:" in holder.text
    assert len(chunks) == 2


def test_unrelated_paragraph_after_an_equation_is_not_glued():
    md = f"{prose(300)} Lead:\n\n$$\nx\n$$\n\nAn unrelated claim follows here. {prose(400)}"
    chunks = chunk_markdown(md)
    assert "$$" in chunks[0].text
    assert "unrelated claim" not in chunks[0].text


@pytest.mark.parametrize("text,expected", [
    ("where $x$ is real", True),
    ("Where $x$ is real", True),
    ("with $k$ the wave number", True),
    ("wobei $x$ reell ist", True),
    ("and therefore the result holds", True),
    ("Therefore the result holds", False),
    ("Withdrawal is not a continuation", False),
])
def test_continuation_detection(text, expected):
    assert continues_equation(text) is expected


def test_equation_run_between_two_lead_ins_breaks_between_equations_when_too_long():
    eq = "$$\n" + " + ".join(f"y_{{{i}}}" for i in range(60)) + "\n$$"
    md = "Consider this chain:\n\n" + "\n\n".join([eq] * 12)
    chunks = chunk_markdown(md)
    assert len(chunks) > 1
    assert all(c.token_count <= MAX_TOKENS for c in chunks)
    assert all(c.text.count("$$") % 2 == 0 for c in chunks)
    assert "Consider this chain:" in chunks[0].text


def test_figure_between_lead_in_and_equation_does_not_break_the_glue():
    md = f"{prose(590)} Lead-in:\n\n![Figure 1](f.png)\n\n$$\nx\n$$"
    holder = next(c for c in chunk_markdown(md) if c.kind == "text" and "$$" in c.text)
    assert "Lead-in:" in holder.text


# --- Rule: heading-path prefix ----------------------------------------------


def test_prefix_uses_the_full_heading_path():
    chunks = chunk_markdown("# Ch. 2\n\n## Lorentz transformations\n\nBody text.")
    assert chunks[0].text == "Ch. 2 › Lorentz transformations ›\n\nBody text."


def test_sibling_heading_replaces_the_old_one_and_clears_deeper_levels():
    md = "# A\n\n## B\n\n### C\n\nOne.\n\n## D\n\nTwo."
    assert chunk_markdown(md)[-1].heading_path == ["A", "D"]


def test_heading_level_gap_is_tolerated():
    assert chunk_markdown("# A\n\n### C\n\nOne.")[0].heading_path == ["A", "C"]


def test_prefix_counts_towards_the_token_limit():
    title = " ".join(["heading"] * 100)
    md = f"# {title}\n\n" + "\n\n".join(prose(150, f"w{i}x") for i in range(10))
    assert all(c.token_count <= MAX_TOKENS for c in chunk_markdown(md))


# --- Rule: figures are their own chunks --------------------------------------


def test_figure_with_text_becomes_its_own_chunk_with_metadata():
    blocks = [
        Block(0, "heading", "Results", level=1),
        Block(1, "paragraph", "Some text.", page=3),
        Block(2, "figure", "Figure 1. A spectrum.", page=4, meta={"image_path": "fig/1.png"}),
    ]
    chunks = chunk_blocks(blocks)
    assert [c.kind for c in chunks] == ["text", "figure"]
    figure = chunks[1]
    assert figure.page_start == 4
    assert figure.meta["image_path"] == "fig/1.png"
    assert figure.text == "Results ›\n\nFigure 1. A spectrum."


def test_figure_without_any_text_is_not_indexed():
    assert chunk_blocks([Block(0, "figure", "", meta={"image_path": "f.png"})]) == []


# --- Rule: one-sentence overlap ----------------------------------------------


def test_neighbours_share_exactly_one_sentence():
    chunks = chunk_markdown("\n\n".join(prose(120, f"w{i}x") for i in range(12)))
    assert len(chunks) >= 2
    for left, right in zip(chunks, chunks[1:]):
        last = split_sentences(left.text.split("\n\n")[-1])[-1]
        assert right.text.startswith(last)
        assert right.text.count(last) == 1


def test_the_repeated_sentence_doesnt_set_the_page_or_the_focus():
    """Review R2-C4: a chunk whose own text starts on page 2 opened at page 1, where the
    sentence repeated from the chunk before sits (59 of 1497 chunks in the library)."""
    blocks = [Block(i, "paragraph", prose(250, f"p{i}x"), page=i + 1) for i in range(6)]
    chunks = chunk_blocks(blocks)
    assert len(chunks) >= 3
    for left, right in zip(chunks, chunks[1:]):
        last = split_sentences(left.text.split("\n\n")[-1])[-1]
        assert right.text.startswith(last)                      # still read for context
        own_text = right.text[len(last):]
        own_pages = [block.page for block in blocks if f" p{block.order}x " in own_text]
        assert right.page_start == min(own_pages)


def test_no_overlap_when_a_chunk_ends_in_an_equation():
    eq = "$$\nz\n$$"
    md = f"{prose(590)} Lead:\n\n{eq}\n\nAn unrelated claim. {prose(500, 'other')}"
    chunks = chunk_markdown(md)
    holder = chunks.index(next(c for c in chunks if eq in c.text))
    assert chunks[holder].text.endswith(eq)
    assert eq not in chunks[holder + 1].text


def test_no_overlap_across_a_section_boundary():
    md = f"# A\n\n{prose(50)} Final of A.\n\n# B\n\n{prose(50, 'other')}"
    assert "Final of A." not in chunk_markdown(md)[1].text


def test_oversized_block_is_not_pushed_over_further_by_overlap():
    atom = "$$\n" + " + ".join(f"x_{{{i}}}" for i in range(95)) + "\n$$"
    assert 550 < estimate_tokens(atom) <= MAX_TOKENS
    md = f"{prose(590)} Closing sentence.\n\nNext unrelated line.\n\n{atom}"
    assert all(c.token_count <= MAX_TOKENS for c in chunk_markdown(md))


# --- Provenance --------------------------------------------------------------


def test_page_range_and_block_orders_cover_every_contributing_block():
    blocks = [
        Block(0, "paragraph", "On page five:", page=5),
        Block(1, "equation", "$$\nx\n$$", page=6),
        Block(2, "paragraph", "where $x$ is real.", page=6),
    ]
    chunk = chunk_blocks(blocks)[0]
    assert (chunk.page_start, chunk.page_end) == (5, 6)
    assert chunk.block_orders == [0, 1, 2]


def test_blocks_without_pages_give_a_null_range():
    chunk = chunk_blocks([Block(0, "paragraph", "x")])[0]
    assert (chunk.page_start, chunk.page_end) == (None, None)


def test_every_sentence_of_the_input_survives():
    source = [f"Sentence number{i} is here." for i in range(400)]
    chunks = chunk_markdown(" ".join(source))
    joined = "\n".join(c.text for c in chunks)
    assert all(s in joined for s in source)


def test_empty_input():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  ") == []


def test_seq_is_reading_order():
    chunks = chunk_markdown("# A\n\nOne.\n\n# B\n\nTwo.")
    assert [c.seq for c in chunks] == [0, 1]


def test_a_dropped_equation_never_reaches_a_chunk():
    blocks = [
        Block(order=0, kind="paragraph", text="Before the equation."),
        Block(order=1, kind=DROPPED_EQUATION, text="$$\\vdots \\vdots \\vdots$$", page=3),
        Block(order=2, kind="paragraph", text="After the equation."),
    ]
    chunks = chunk_blocks(blocks)
    assert len(chunks) == 1
    assert "vdots" not in chunks[0].text
    assert chunks[0].block_orders == [0, 2]


def unverified(order, page=42):
    return Block(order=order, kind="equation", text="$$\nx=z\n$$", page=page,
                 meta={"unverified": True, "crop": f"formula_crops/{order:06d}.png"})


def test_an_unverified_formula_carries_a_warning_above_its_latex():
    blocks = [Block(order=0, kind="paragraph", text="Before the equation:"), unverified(1)]
    chunk = chunk_blocks(blocks)[0]
    assert "[unverified formula, see page 42]\n$$\nx=z\n$$" in chunk.text


def test_the_warning_goes_in_the_chunk_and_not_in_the_block():
    block = unverified(1)
    chunk_blocks([block])
    assert block.text == "$$\nx=z\n$$"


def test_a_verified_formula_carries_no_warning():
    blocks = [Block(order=0, kind="equation", text="$$\nx=y\n$$", page=42)]
    chunk = chunk_blocks(blocks)[0]
    assert "unverified" not in chunk.text
    assert chunk.meta == {}


def test_a_chunk_lists_the_crops_of_its_unverified_formulas():
    blocks = [unverified(0, page=7), Block(order=1, kind="paragraph", text="and so on."), unverified(2, page=8)]
    assert chunk_blocks(blocks)[0].meta == {"unverified_formulas": [
        {"page": 7, "crop": "formula_crops/000000.png", "text_layer": ""},
        {"page": 8, "crop": "formula_crops/000002.png", "text_layer": ""},
    ]}


def test_a_chunk_carries_each_unverified_formulas_symbols():
    block = replace(unverified(0, page=7), meta={**unverified(0, page=7).meta, "text_layer": "gZ=k1"})
    assert chunk_blocks([block])[0].meta["unverified_formulas"][0]["text_layer"] == "gZ=k1"


def test_the_warning_counts_towards_the_token_budget():
    plain = Block(order=0, kind="equation", text="$$\nx=z\n$$", page=42)
    assert chunk_blocks([unverified(0)])[0].token_count > chunk_blocks([plain])[0].token_count
