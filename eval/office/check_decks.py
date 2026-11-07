"""Gate item 4 of roadmap item 7: decks in the library, after ingestion.

    STUDY_ASSISTANT_CONFIG=... python eval/office/check_decks.py [--out eval/results/deck-ingest.json]

For every .pptx document:
  - every slide is a page: the render's page count equals the deck's slide count;
  - the slides' words are in the chunks: python-pptx's words (slides.json, kept slides only,
    notes not counted) found in the document's chunks, as multisets of alphanumeric words
    after NFKC and case folding; the gate wants >= 98%;
  - every speaker note can be found and is cited to its slide: a keyword search for the
    note's first sentence returns its chunk, on that slide's page;
  - the title isn't the file name;
  - excluded pages are nowhere: no block, chunk, figure or summary on them, and none of
    their words that appear nowhere else in the deck is in the index.
Runs on a copy of the database only (the config's data_dir).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import fitz  # noqa: E402

from app.config import load_settings  # noqa: E402
from app.db.connection import connect  # noqa: E402
from app.ingest import deck  # noqa: E402
from app.retrieval.search import fts_query  # noqa: E402

WORD = re.compile(r"\w+")


def words(text: str) -> collections.Counter:
    return collections.Counter(WORD.findall(unicodedata.normalize("NFKC", text).casefold()))


def check(conn, settings, doc) -> dict:
    slides = deck.read_slides(settings.parsed_dir / doc["sha256"])
    excluded = deck.excluded_pages(settings, doc["path"])
    with fitz.open(deck.render_path(settings, doc["sha256"])) as pdf:
        pages = pdf.page_count
    chunks = conn.execute("SELECT id, kind, text, page_start, page_end, meta FROM chunks WHERE doc_id = ?",
                          (doc["id"],)).fetchall()
    have = words(" ".join(c["text"] for c in chunks))
    kept = [s for s in slides["slides"] if s["slide"] not in excluded]
    want = words(" ".join(" ".join(s["lines"]) for s in kept))
    matched = sum(min(n, have[w]) for w, n in want.items())
    missing = sorted((want - have).elements())

    notes = []
    for slide in kept:
        note = (slide.get("notes") or "").strip()
        if not note:
            continue
        first = re.split(r"(?<=[.!?])\s", note)[0][:200]
        query = fts_query(first)
        found = conn.execute("SELECT c.id, c.page_start, c.meta FROM chunk_fts f JOIN chunks c ON c.id = f.rowid "
                             "WHERE chunk_fts MATCH ? AND c.doc_id = ? ORDER BY rank LIMIT 3",
                             (query, doc["id"])).fetchall() if query else []
        on_slide = [r for r in found if r["page_start"] == slide["slide"]
                    and json.loads(r["meta"] or "{}").get("speaker_notes")]
        notes.append({"slide": slide["slide"], "found": bool(on_slide), "query": first[:60]})

    leaks = {}
    if excluded:
        place = ",".join("?" * len(excluded))
        args = [doc["id"], *sorted(excluded)]
        leaks["blocks"] = conn.execute(f"SELECT count(*) FROM blocks WHERE doc_id = ? AND page IN ({place})", args).fetchone()[0]
        leaks["figures"] = conn.execute(f"SELECT count(*) FROM figures WHERE doc_id = ? AND page IN ({place})", args).fetchone()[0]
        leaks["chunks"] = conn.execute(
            f"SELECT count(*) FROM chunks WHERE doc_id = ? AND (page_start IN ({place}) OR page_end IN ({place}))",
            [doc["id"], *sorted(excluded), *sorted(excluded)]).fetchone()[0]
        only_there = words(" ".join(" ".join(s["lines"]) + " " + (s.get("notes") or "")
                                    for s in slides["slides"] if s["slide"] in excluded)) - want
        distinctive = [w for w in only_there if len(w) >= 4 and not w.isdigit()]
        # Exact words, not the keyword index (it stems: "investigate" matched "Investigation" on
        # slide 19), in the chunks made from the deck's own text; the title heads every chunk.
        # Summaries are the local model's own wording of the kept slides, so a common word there
        # proves nothing: they're listed for a look, and the excluded page's capitalised words
        # (names, places) must appear nowhere at all.
        # Figure descriptions are a model's wording too (Sonnet's, of the kept slides' figures:
        # "the GSI logo" on slides 3-20 is not slide 1's "GSI 2021-"), so they go with summaries.
        source = words(" ".join(c["text"] for c in chunks if c["kind"] == "text"))
        summaries = words(" ".join(c["text"] for c in chunks if c["kind"] in ("summary", "figure")))
        title = words(doc["title"] or "")
        indexed = sorted(w for w in distinctive if source[w] and not title[w])
        leaks["in_summaries_or_figures_to_look_at"] = sorted(w for w in distinctive if summaries[w] and not source[w])
        names = {w.casefold() for s in slides["slides"] if s["slide"] in excluded
                 for w in WORD.findall(" ".join(s["lines"])) if w[:1].isupper() and len(w) >= 3} - set(want)
        leaks["names_checked"] = len(names)
        leaks["names_in_slide_text"] = sorted(n for n in names if source[n])
        leaks["names_in_summaries_or_figures"] = sorted(n for n in names if summaries[n] and not source[n])
        leaks["distinctive_words_checked"] = len(distinctive)
        leaks["distinctive_words_indexed"] = indexed

    return {
        "doc": doc["path"], "title": doc["title"], "title_is_file_name": (doc["title"] or "").strip()
        in {doc["path"], Path(doc["path"]).stem},
        "slides": len(slides["slides"]), "pages": pages, "excluded": sorted(excluded),
        "kept_slides": len(kept), "chunks": collections.Counter(c["kind"] for c in chunks),
        "slide_words": sum(want.values()), "coverage": round(matched / max(1, sum(want.values())), 4),
        "missing": missing[:40], "notes": notes, "excluded_leaks": leaks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "eval/results/deck-ingest.json")
    args = parser.parse_args()
    settings = load_settings()
    conn = connect(settings.db_path, apply_schema=False)
    rows = [check(conn, settings, doc) for doc in
            conn.execute("SELECT * FROM documents WHERE kind = 'pptx' ORDER BY id").fetchall()]
    args.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in rows:
        notes_ok = sum(n["found"] for n in row["notes"])
        print(f"{row['doc']}: {row['pages']} pages for {row['slides']} slides; title {row['title']!r}"
              f" (file name: {row['title_is_file_name']}); chunks {dict(row['chunks'])}")
        print(f"   slide words in chunks {row['coverage']:.1%} of {row['slide_words']}; missing {row['missing'][:15]}")
        print(f"   speaker notes found on their slide: {notes_ok}/{len(row['notes'])}"
              + "".join(f"\n      MISS slide {n['slide']}: {n['query']}" for n in row["notes"] if not n["found"]))
        if row["excluded"]:
            print(f"   excluded {row['excluded']}: {row['excluded_leaks']}")


if __name__ == "__main__":
    main()
