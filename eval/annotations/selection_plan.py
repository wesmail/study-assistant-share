"""Gate item 1 of roadmap item 4: the 20 selections the browser makes, chosen from PyMuPDF's words.

    .venv/bin/python eval/annotations/selection_plan.py             eval/annotations/selections.json
    .venv/bin/python eval/annotations/selection_plan.py --set heldout    eval/annotations/selections-heldout.json
    .venv/bin/python eval/annotations/selection_plan.py --set heldout2   eval/annotations/selections-heldout2.json

The held-out sets have other pages and other seeds. "heldout" was added after the first gate
run, when the snapping rule changed (glyphs pdf.js reads differently); "heldout2" after
"heldout", when it changed again (accents left out of the comparison). Each checks the rule
on selections it wasn't made on.

Each selection is a drag from the first word to the last, made in the real viewer by
frontend/e2e/annotations.mjs. The words come from the file through PyMuPDF, so the expected
boxes don't depend on pdf.js. Kinds: "line" (3 to 8 words in a line), "lines" (from the middle
of a line to the middle of the next), "midword" (the drag starts inside a word: the highlight
should take the whole word) and "equation" (a line with mathematical symbols). Pages: doc 6 is
two-column; the others' pages include their most equation-heavy ones.
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("selections.json")
MATH = re.compile(r"[=∑∫∂√±×·≤≥∞αβγδεθλμνπρσφψωΣΩΔ∇^_]")
SEED = 20261106
HELDOUT_SEED = 20261107
HELDOUT2_SEED = 20261108
HELDOUT2 = [
    (6, 1, "lines"), (6, 8, "line"), (6, 10, "midword"), (6, 8, "equation"),
    (1, 2, "line"), (1, 10, "equation"), (1, 16, "lines"),
    (5, 17, "equation"), (5, 25, "lines"), (5, 51, "midword"),
    (4, 55, "equation"), (4, 26, "lines"), (4, 33, "line"),
    (2, 171, "equation"), (2, 173, "lines"), (2, 80, "midword"), (2, 230, "line"),
    (3, 312, "equation"), (3, 64, "lines"), (3, 150, "midword"),
]
HELDOUT = [
    (6, 3, "lines"), (6, 7, "line"), (6, 6, "equation"), (6, 9, "midword"),
    (1, 5, "equation"), (1, 9, "line"), (1, 14, "lines"),
    (5, 18, "equation"), (5, 27, "midword"), (5, 60, "lines"),
    (4, 15, "equation"), (4, 40, "line"), (4, 52, "lines"),
    (2, 154, "equation"), (2, 172, "midword"), (2, 60, "line"), (2, 211, "lines"),
    (3, 321, "equation"), (3, 3, "midword"), (3, 200, "lines"),
]
PLAN = [  # (doc id, page, kind)
    (6, 2, "line"), (6, 2, "lines"), (6, 5, "midword"), (6, 4, "equation"),
    (1, 3, "line"), (1, 7, "lines"), (1, 12, "midword"),
    (5, 26, "equation"), (5, 31, "lines"), (5, 48, "line"),
    (4, 27, "equation"), (4, 12, "midword"), (4, 5, "lines"),
    (2, 167, "equation"), (2, 152, "line"), (2, 30, "lines"), (2, 100, "midword"),
    (3, 62, "equation"), (3, 60, "lines"), (3, 117, "line"),
]


def lines_of(page: fitz.Page) -> list[list[tuple]]:
    """Body lines with at least 5 words, away from the running head and foot."""
    found: dict[tuple[int, int], list[tuple]] = {}
    for word in page.get_text("words", sort=False):
        found.setdefault((word[5], word[6]), []).append(word)
    height = page.rect.height
    return [words for words in found.values()
            if len(words) >= 5 and 0.1 * height < words[0][1] < 0.9 * height]


def choose(rng: random.Random, page: fitz.Page, kind: str) -> dict:
    lines = lines_of(page)
    keys = list(dict.fromkeys((w[5], w[6]) for line in lines for w in line))
    if kind == "equation":
        candidates = [line for line in lines if sum(bool(MATH.search(w[4])) for w in line) >= 2]
        line = rng.choice(candidates or lines)
        words = line
    elif kind == "lines":
        pairs = [(a, b) for a, b in zip(lines, lines[1:]) if a[0][5] == b[0][5]]  # same block
        a, b = rng.choice(pairs)
        words = a[len(a) // 2:] + b[:max(2, len(b) // 2)]
    else:
        line = rng.choice(lines)
        size = rng.randint(3, min(8, len(line)))
        start = rng.randint(0, len(line) - size)
        words = line[start:start + size]
    if kind == "midword":
        long_first = [w for w in words if len(w[4]) >= 4]
        if len(words[0][4]) < 4 and long_first:
            words = words[words.index(long_first[0]):]
    return {
        "kind": kind,
        "words": [{"text": w[4], "box": [round(v, 2) for v in w[:4]], "line": [w[5], w[6]]} for w in words],
        "start": [words[0][0] + (0.4 * (words[0][2] - words[0][0]) if kind == "midword" else 0.6),
                  (words[0][1] + words[0][3]) / 2],
        "end": [words[-1][2] - 0.6, (words[-1][1] + words[-1][3]) / 2],
        "_keys": len(keys),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=("gate", "heldout", "heldout2"), default="gate")
    chosen_set = parser.parse_args().set
    plan, seed, prefix = {"gate": (PLAN, SEED, "s"), "heldout": (HELDOUT, HELDOUT_SEED, "h"),
                          "heldout2": (HELDOUT2, HELDOUT2_SEED, "k")}[chosen_set]
    out = OUT if chosen_set == "gate" else OUT.with_name(f"selections-{chosen_set}.json")
    rng = random.Random(seed)
    conn = sqlite3.connect(f"file:{ROOT / 'data' / 'app.sqlite'}?mode=ro", uri=True)
    paths = dict(conn.execute("SELECT id, path FROM documents"))
    selections = []
    for number, (doc_id, page_number, kind) in enumerate(plan, 1):
        with fitz.open(ROOT / "library" / paths[doc_id]) as pdf:
            chosen = choose(rng, pdf[page_number - 1], kind)
        chosen.pop("_keys")
        selections.append({"id": f"{prefix}{number:02d}", "doc_id": doc_id, "page": page_number, **chosen})
        print(f"{selections[-1]['id']} doc {doc_id} p. {page_number:3d} {kind:8s} {' '.join(w['text'] for w in chosen['words'])[:90]}")
    out.write_text(json.dumps({"seed": seed, "selections": selections}, ensure_ascii=False, indent=1) + "\n")
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
