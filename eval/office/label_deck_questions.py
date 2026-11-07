"""Point eval/deck_questions.yaml's needed facts at the current chunk ids (roadmap item 7).

    STUDY_ASSISTANT_CONFIG=<the gate's config> python eval/office/label_deck_questions.py

Chunk ids change whenever the decks are chunked again. Each fact keeps its slides and quotes,
which don't: its chunk is the text chunk of the same deck, on one of the fact's slides, that
holds every quote (squeezed, as run_eval.py compares). The `chunks: [...]` lines are rewritten
in place, so the file's comments and layout stay. Stops if a fact matches no chunk.
"""
from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "eval"))

from app.config import load_settings  # noqa: E402
from run_eval import DECK_QUESTIONS, squeezed  # noqa: E402


def main() -> None:
    settings = load_settings()
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    text = DECK_QUESTIONS.read_text(encoding="utf-8")
    found = []
    for question in yaml.safe_load(text)["questions"]:
        doc = question["expected_sources"][0]["doc"]
        rows = conn.execute("SELECT c.id, c.text, c.page_start FROM chunks c JOIN documents d ON d.id = c.doc_id "
                            "WHERE d.path = ? AND c.kind = 'text' ORDER BY c.id", (doc,)).fetchall()
        for fact in question["needed_facts"]:
            ids = [r["id"] for r in rows if r["page_start"] in fact["pages"]
                   and all(squeezed(quote) in squeezed(r["text"]) for quote in fact["quotes"])]
            if not ids:
                raise SystemExit(f"{question['id']}: no chunk on {fact['pages']} holds {fact['quotes']}")
            print(question["id"], fact["pages"], ids)
            found.append(ids)
    lines, position = text.split("\n"), 0
    for index, line in enumerate(lines):
        match = re.match(r"(\s+)chunks: \[.*\]$", line)
        if match:
            lines[index] = f"{match.group(1)}chunks: {found[position]}"
            position += 1
    if position != len(found):
        raise SystemExit(f"{position} chunks lines for {len(found)} facts")
    DECK_QUESTIONS.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
