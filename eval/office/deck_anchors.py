"""Gate item 8 of roadmap item 7: highlights on a deck's real LibreOffice render.

    python eval/office/deck_anchors.py DECK OUT_DIR [--seed N] [--count 20]

1. Renders DECK (read only) and an edited copy made in OUT_DIR: a new slide put in front, and
   one word changed on another slide, as a reader editing a talk would.
2. Picks COUNT passages at random (seeded) from the original render's text layer: 3 to 8
   words on one line, from slides with text. Each becomes a selection (the union of its
   words' boxes, as the viewer sends) and is snapped as a highlight is (annotations.snap).
3. Finds each again in the edited render as re-anchoring a deck does (anchors.page_map, then
   anchors.locate): it should be on the next slide, its words read alike (>= 0.9).

Pages kept out in library/metadata.yaml (`exclude_pages`) are never sampled, and their text is
blanked in both renders before alignment and search, so no quote of them can reach the output.

Prints and saves the counts. Nothing is written to the library or to data/.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import fitz  # noqa: E402
from pptx import Presentation  # noqa: E402

from app import office  # noqa: E402
from app.anchors import Page, locate, page_map, placed  # noqa: E402
from app.annotations import snap, union  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.ingest.deck import excluded_pages  # noqa: E402

def edited_copy(deck: Path, target: Path) -> str:
    """A copy with a new first slide, and one word changed in the longest text run of the old
    slide 4 (a run, so the change can't be split across formatting). Returns what changed."""
    presentation = Presentation(deck)
    layout = presentation.slide_layouts[min(1, len(presentation.slide_layouts) - 1)]
    slide = presentation.slides.add_slide(layout)
    if slide.shapes.title is not None:
        slide.shapes.title.text = "A slide added in front"
    ids = presentation.slides._sldIdLst
    ids.insert(0, ids[-1])          # move the new slide to the front
    runs = [run for shape in presentation.slides[4].shapes if shape.has_text_frame
            for paragraph in shape.text_frame.paragraphs for run in paragraph.runs]
    run = max(runs, key=lambda r: len(r.text.split()))
    words = run.text.split()
    before = run.text
    words[1] = "edited"
    run.text = " ".join(words)
    presentation.save(target)
    return f"new slide 5 (old 4): {before!r} -> {run.text!r}"


def passages(pdf_path: Path, rng: random.Random, count: int, excluded: set[int] = frozenset()) -> list[dict]:
    """COUNT runs of 3-8 words on one line, from different places of the render, never from a
    page in `excluded`."""
    found = []
    with fitz.open(pdf_path) as pdf:
        for number, page in enumerate(pdf, 1):
            if number in excluded:
                continue
            words = page.get_text("words", sort=False)
            lines: dict[tuple[int, int], list[int]] = {}
            for index, word in enumerate(words):
                lines.setdefault((word[5], word[6]), []).append(index)
            for indices in lines.values():
                if len(indices) >= 4:
                    found.append((number, words, indices))
    rng.shuffle(found)
    chosen = []
    for number, words, indices in found[:count]:
        size = rng.randint(3, min(8, len(indices)))
        start = rng.randint(0, len(indices) - size)
        picked = [words[i] for i in indices[start:start + size]]
        chosen.append({"page": number, "quote": " ".join(w[4] for w in picked),
                       "rects": [list(union([w[:4] for w in picked]))], "words": words})
    return chosen


def page_texts(pdf_path: Path, blank: set[int]) -> list[Page]:
    """Each page's words, with the pages in `blank` left empty."""
    with fitz.open(pdf_path) as pdf:
        return [Page.of([] if number in blank else page.get_text("words", sort=False))
                for number, page in enumerate(pdf, 1)]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("deck", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    settings = load_settings()
    excluded = excluded_pages(settings, args.deck.name)
    shifted = {page + 1 for page in excluded}       # the edited copy has one slide in front
    args.out.mkdir(parents=True, exist_ok=True)
    original, edited = args.out / "original.pdf", args.out / "edited.pdf"
    changed = edited_copy(args.deck, args.out / "edited.pptx")
    await office.render_pdf(settings, args.deck, original)
    await office.render_pdf(settings, args.out / "edited.pptx", edited)
    pages = page_texts(edited, shifted)
    mapped = page_map(page_texts(original, excluded), pages)

    rows = []
    for item in passages(original, random.Random(args.seed), args.count, excluded):
        snapped = snap([tuple(r) for r in item["rects"]], item["quote"], item["words"])
        row = {"page": item["page"], "quote": item["quote"], "snapped": snapped is not None}
        if snapped:
            row["snapped_quote_equal"] = snapped["quote"] == item["quote"]
            start = mapped.get(item["page"], item["page"])     # as reanchor_document does for a deck
            candidate, why = locate(pages, start, snapped["quote"], snapped["prefix"], snapped["suffix"])
            row["found"] = why
            if candidate:
                new = placed(pages[candidate.page - 1], candidate)
                row.update(new_page=new["page"], new_quote=new["quote"], similarity=round(candidate.similarity, 3))
        rows.append(row)
    moved = [r for r in rows if r.get("new_page") == r["page"] + 1 and r.get("similarity", 0) >= 0.9]
    summary = {"deck": args.deck.name, "edit": changed, "seed": args.seed, "excluded_pages": sorted(excluded),
               "passages": len(rows),
               "snapped": sum(r["snapped"] for r in rows),
               "snapped_quote_equal": sum(r.get("snapped_quote_equal", False) for r in rows),
               "found_on_next_slide": len(moved)}
    (args.out / "anchors.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    print(json.dumps(summary))
    for r in rows:
        if r not in moved:
            print("  NOT MOVED:", r)


if __name__ == "__main__":
    asyncio.run(main())
