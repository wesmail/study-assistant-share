"""Heading paths from the PDF outline (app.ingest.outline)."""
from __future__ import annotations

import fitz
import pytest

from app.ingest.chunker import Block, Chunk, _prefix
from app.ingest.outline import (
    Entry, agreement, merged, new_paths, normalized, outline_chunks, path_at, place, reprefixed,
)

# A small book shaped like MML's: Part > chapter > section in the outline; the parsed
# headings put the chapter title under the previous chapter's last section.
OUTLINE = [Entry(1, "Part I Foundations", 1), Entry(2, "1 Linear Algebra", 1),
           Entry(3, "1.1 Matrices", 2), Entry(3, "1.2 Further Reading", 3),
           Entry(2, "2 Geometry", 4), Entry(3, "2.1 Norms", 5)]
BLOCKS = [(0, 1, "heading", "Linear Algebra"), (1, 1, "paragraph", "intro"),
          (2, 2, "heading", "1.1 Matrices"), (3, 2, "heading", "Example 1.1"), (4, 2, "paragraph", "x"),
          (5, 3, "heading", "1.2 Further Reading"), (6, 3, "paragraph", "y"),
          (7, 4, "heading", "Geometry"), (8, 4, "paragraph", "z"),
          (9, 5, "heading", "2.1 Norms"), (10, 5, "paragraph", "w")]
PARSED = [(1, ["Part II Is about Learning", "Linear Algebra"]),
          (4, ["Part II Is about Learning", "1.1 Matrices", "Example 1.1"]),
          (6, ["Part II Is about Learning", "1.2 Further Reading"]),
          (8, ["Part II Is about Learning", "1.2 Further Reading", "Geometry"]),
          (10, ["Expansion along row j", "2.1 Norms"])]


def test_normalized_ignores_numbering_markup_and_spacing():
    assert normalized("4.3 Cholesky Decomposition") == normalized("Cholesky  decomposition")
    assert normalized("26.10 W[sup(+)]W[sup(−)] Production") == normalized("26.10 W + W -Production")
    assert normalized("I Foundations") == normalized("Foundations")
    assert normalized("Introduction") == "introduction"     # a leading "I" word isn't a numeral


def test_entries_start_at_their_heading_in_the_text():
    placed = place(OUTLINE, BLOCKS)
    assert [entry.order for entry in placed] == [0, 0, 2, 5, 7, 9]


def test_an_entry_without_a_heading_starts_at_its_page():
    placed = place([Entry(1, "Chapter Without Heading", 3)], BLOCKS)
    assert placed[0].order == 5


def test_a_repeated_title_cannot_pull_an_entry_backwards():
    blocks = [(0, 1, "heading", "Exercises"), (1, 9, "heading", "Exercises")]
    placed = place([Entry(1, "A", 1), Entry(1, "Exercises", 9)], blocks)
    assert placed[1].order == 1


def test_path_follows_the_outline():
    placed = place(OUTLINE, BLOCKS)
    assert path_at(placed, 8) == ["Part I Foundations", "2 Geometry"]
    assert path_at(placed, 6) == ["Part I Foundations", "1 Linear Algebra", "1.2 Further Reading"]


def test_deeper_parsed_headings_survive_when_the_parsed_path_agrees():
    placed = place(OUTLINE, BLOCKS)
    paths = new_paths(placed, PARSED)
    assert paths[1] == ["Part I Foundations", "1 Linear Algebra", "1.1 Matrices", "Example 1.1"]
    # The stale "1.2 Further Reading" and the chapter title Docling put under it are gone.
    assert paths[3] == ["Part I Foundations", "2 Geometry"]
    assert paths[4] == ["Part I Foundations", "2 Geometry", "2.1 Norms"]


def test_a_tail_under_a_different_section_is_dropped():
    titles = {normalized(e.title) for e in OUTLINE}
    outline = ["Part I Foundations", "1 Linear Algebra", "1.2 Further Reading"]
    assert merged(["X", "1.1 Matrices", "Example 1.1"], outline, titles) == outline


def test_before_the_first_entry_the_parsed_path_stays():
    assert merged(["Contents"], [], set()) == ["Contents"]


def test_agreement_is_low_for_misplaced_chapters_and_high_for_chapter_labels():
    placed = place(OUTLINE, BLOCKS)
    assert agreement(placed, PARSED) < 0.5
    good = [(4, ["Chapter 1", "1.1 Matrices"]), (10, ["Part I Foundations", "2 Geometry", "2.1 Norms"])]
    assert agreement(placed, good) == pytest.approx((2 / 3 + 1) / 2)


def test_reprefixed_swaps_only_the_heading_prefix():
    text = _prefix(["Old", "1.1 Matrices"]) + "body text"
    assert reprefixed(text, ["Old", "1.1 Matrices"], ["New"]) == _prefix(["New"]) + "body text"
    with pytest.raises(ValueError):
        reprefixed("no prefix here", ["Old"], ["New"])


def _pdf_with_outline(path, toc):
    document = fitz.open()
    for _ in range(5):
        document.new_page()
    document.set_toc(toc)
    document.save(path)


def test_outline_chunks_rewrites_a_disagreeing_document(tmp_path):
    pdf = tmp_path / "book.pdf"
    _pdf_with_outline(pdf, [[e.level, e.title, e.page] for e in OUTLINE])
    blocks = [Block(order, kind, text, page) for order, page, kind, text in BLOCKS]
    chunks = [Chunk(kind="text", seq=i, heading_path=path, text=_prefix(path) + "body", token_count=1,
                    block_orders=[first], page_start=None, page_end=None, meta={})
              for i, (first, path) in enumerate(PARSED)]
    out = outline_chunks(chunks, blocks, pdf)
    assert out[3].heading_path == ["Part I Foundations", "2 Geometry"]
    assert out[3].text == _prefix(["Part I Foundations", "2 Geometry"]) + "body"


def test_outline_chunks_leaves_an_agreeing_or_unreadable_document(tmp_path):
    pdf = tmp_path / "book.pdf"
    _pdf_with_outline(pdf, [[e.level, e.title, e.page] for e in OUTLINE])
    blocks = [Block(order, kind, text, page) for order, page, kind, text in BLOCKS]
    path = ["Part I Foundations", "2 Geometry", "2.1 Norms"]
    chunks = [Chunk(kind="text", seq=0, heading_path=path, text=_prefix(path) + "body", token_count=1,
                    block_orders=[10], page_start=None, page_end=None, meta={})]
    assert outline_chunks(chunks, blocks, pdf)[0].heading_path == path
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    assert outline_chunks(chunks, blocks, tmp_path / "broken.pdf")[0].heading_path == path


def test_a_bookmark_on_a_kept_out_page_names_nothing(tmp_path):
    """Review R2-S2: the title of a kept-out page's bookmark would head the chunks after it."""
    pdf = tmp_path / "book.pdf"
    _pdf_with_outline(pdf, [[e.level, e.title, e.page] for e in OUTLINE])
    blocks = [Block(order, kind, text, page) for order, page, kind, text in BLOCKS if page != 1]
    chunks = [Chunk(kind="text", seq=i, heading_path=path, text=_prefix(path) + "body", token_count=1,
                    block_orders=[first], page_start=None, page_end=None, meta={})
              for i, (first, path) in enumerate(PARSED) if first > 1]
    out = outline_chunks(chunks, blocks, pdf, excluded={1})
    every = " ".join(" ".join(chunk.heading_path) + chunk.text for chunk in out)
    assert "Part I Foundations" not in every and "1 Linear Algebra" not in every
    assert out[-1].heading_path[-1] == "2.1 Norms"      # later bookmarks still apply
