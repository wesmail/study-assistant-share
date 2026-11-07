"""eval/office/deck_anchors.py never samples, or keeps text of, pages kept out in metadata.yaml."""
import importlib.util
import random
from pathlib import Path

import fitz

SCRIPT = Path(__file__).resolve().parents[2] / "eval" / "office" / "deck_anchors.py"
spec = importlib.util.spec_from_file_location("deck_anchors", SCRIPT)
deck_anchors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deck_anchors)


def three_pages(path: Path) -> Path:
    with fitz.open() as pdf:
        for number in (1, 2, 3):
            page = pdf.new_page()
            for line in range(6):
                page.insert_text((72, 72 + 20 * line), f"page{number} alpha beta gamma delta epsilon line{line}")
        pdf.save(path)
    return path


def test_excluded_pages_are_never_sampled(tmp_path):
    pdf = three_pages(tmp_path / "deck.pdf")
    everything = deck_anchors.passages(pdf, random.Random(1), 100)
    assert {item["page"] for item in everything} == {1, 2, 3}      # the control: page 1 is sampled
    kept = deck_anchors.passages(pdf, random.Random(1), 100, excluded={1})
    assert kept and {item["page"] for item in kept} == {2, 3}
    assert not any("page1" in item["quote"] or any(w[4] == "page1" for w in item["words"]) for item in kept)


def test_excluded_pages_are_blanked_for_alignment(tmp_path):
    pdf = three_pages(tmp_path / "deck.pdf")
    pages = deck_anchors.page_texts(pdf, {1})
    assert len(pages) == 3
    assert pages[0].words == [] and pages[0].text == ""
    assert "page2" in pages[1].text
