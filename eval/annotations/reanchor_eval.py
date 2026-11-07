"""Gate item 6 of roadmap item 4: anchors found again after a book is re-ingested.

    .venv/bin/python eval/annotations/reanchor_eval.py --work DIR [--doc 2606.10868v2.pdf]

Everything happens in DIR (a new folder, outside the project): its own library and data, so
the real ones are never touched. Steps:

1. The document is copied into DIR/library and ingested from nothing by the real worker,
   up to the index stage (the late stages, figure descriptions and summaries, aren't needed).
2. 20 anchors are made on it, 10 highlights and 10 notes, each a run of words picked with a
   fixed seed from PyMuPDF's words (one line, or the end of a line and the start of the next),
   snapped as a selection in the viewer would be.
3. The file is replaced by a copy with one page inserted after page INSERT_AFTER. That page
   also carries, word for word, the quotes of two anchors (one before the insertion, one after):
   only the words around them tell the right place from the copy.
4. The worker is run again, as after any changed file: sync_file sees the new hash, the stages
   run up to index, and the worker re-anchors (app/anchors.py).

Right: the anchor is on its old page (+1 after the insertion) with its old boxes (IoU >= 0.8)
and quote. Unanchored: flagged. Wrong: anchored anywhere else. Gate: at least 95% right, the
rest unanchored, none wrong.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app import annotations, notes  # noqa: E402
from app.config import Settings, load_settings  # noqa: E402
from app.db.connection import connect  # noqa: E402
from app.ingest import worker as worker_module  # noqa: E402

SEED = 20261109
INSERT_AFTER = 5
ANCHORS = 20
RESULTS = ROOT / "eval" / "results"


def ingest_until_searchable(conn, settings: Settings, doc_id: int) -> float:
    """Run the worker's jobs for this document's current file up to the index stage."""
    started = time.monotonic()
    worker = worker_module.Worker(conn, settings)
    while True:
        sha = conn.execute("SELECT sha256 FROM documents WHERE id = ?", (doc_id,)).fetchone()[0]
        index = conn.execute("SELECT status FROM jobs WHERE doc_id = ? AND sha256 = ? AND stage = ?",
                             (doc_id, sha, worker_module.SEARCHABLE_AFTER)).fetchone()
        if index and index[0] == "done":
            return time.monotonic() - started
        job = worker.next_job()
        if job is None or job["stage"] in worker_module.LATE_STAGES:
            raise SystemExit(f"stuck before index: {dict(job) if job else None}")
        worker.run_one()
        failed = conn.execute("SELECT stage, error FROM jobs WHERE doc_id = ? AND status = 'failed'", (doc_id,)).fetchone()
        if failed:
            raise SystemExit(f"{failed['stage']} failed: {failed['error'][-500:]}")


def pick_anchors(path: Path, rng: random.Random) -> list[dict]:
    """20 runs of words: pages spread over the document, body lines only."""
    picked = []
    with fitz.open(path) as pdf:
        pages = list(range(1, pdf.page_count + 1))
        for number in range(ANCHORS):
            page_number = pages[number % len(pages)] if number < len(pages) else rng.choice(pages)
            page = pdf[page_number - 1]
            lines: dict[tuple, list] = {}
            for word in page.get_text("words", sort=False):
                lines.setdefault((word[5], word[6]), []).append(word)
            height = page.rect.height
            body = [words for words in lines.values() if len(words) >= 6 and 0.1 * height < words[0][1] < 0.9 * height]
            line = rng.choice(body)
            following = [words for key, words in lines.items() if words is not line and key[0] == line[0][5]
                         and key[1] == line[0][6] + 1 and len(words) >= 3]
            if following and rng.random() < 0.4:
                chosen = [line[len(line) // 2:], following[0][:3]]
            else:
                size = rng.randint(4, min(10, len(line)))
                start = rng.randint(0, len(line) - size)
                chosen = [line[start:start + size]]
            rects = [[min(w[0] for w in ws), min(w[1] for w in ws), max(w[2] for w in ws), max(w[3] for w in ws)]
                     for ws in chosen]
            picked.append({"page": page_number, "rects": rects,
                           "quote": " ".join(w[4] for ws in chosen for w in ws)})
    return picked


def changed_copy(original: Path, out: Path, duplicates: list[str]) -> None:
    """The document with a page inserted after INSERT_AFTER that repeats `duplicates` word for word."""
    with fitz.open(original) as pdf:
        size = pdf[0].rect
        page = pdf.new_page(INSERT_AFTER, width=size.width, height=size.height)
        lines = ["Erratum and addendum (inserted page).", "The following lines repeat text from elsewhere:"]
        lines += duplicates + ["End of the inserted page."]
        y = 90
        for line in lines:
            page.insert_textbox(fitz.Rect(72, y, size.width - 72, y + 40), line, fontsize=10)
            y += 44
        pdf.save(out)


def iou(a: list[list[float]], b: list[list[float]]) -> float:
    def area(r):
        return max(0, r[2] - r[0]) * max(0, r[3] - r[1])
    common = sum(area([max(x[0], y[0]), max(x[1], y[1]), min(x[2], y[2]), min(x[3], y[3])]) for x in a for y in b)
    total = sum(map(area, a)) + sum(map(area, b)) - common
    return common / total if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--doc", default="2606.10868v2.pdf")
    parser.add_argument("--save", default="reanchor-1")
    args = parser.parse_args()
    work = args.work.resolve()
    if work.is_relative_to(ROOT) or work.exists():
        raise SystemExit("--work must be a new folder outside the project")
    library, data = work / "library", work / "data"
    library.mkdir(parents=True)
    shutil.copy2(ROOT / "library" / args.doc, library / args.doc)
    base = load_settings()
    settings = Settings(**{**{f: getattr(base, f) for f in base.__dataclass_fields__},
                           "library_dir": library, "data_dir": data, "outputs_dir": work / "outputs"})
    conn = connect(settings.db_path)
    print("ingesting the original", flush=True)
    assert worker_module.sync_file(conn, settings, library / args.doc) == "new"
    doc_id = conn.execute("SELECT id FROM documents").fetchone()[0]
    first = ingest_until_searchable(conn, settings, doc_id)
    print(f"  searchable after {first:.0f} s", flush=True)

    rng = random.Random(SEED)
    made = []
    for number, pick in enumerate(pick_anchors(library / args.doc, rng)):
        if number % 2 == 0:
            saved = annotations.create(conn, settings, doc_id, pick["page"], pick["rects"], pick["quote"])
            made.append({"table": "annotations", **saved})
        else:
            saved = notes.create(conn, settings, f"Test note {number}", [],
                                 {"doc_id": doc_id, "page": pick["page"], "rects": pick["rects"], "quote": pick["quote"]})
            made.append({"table": "notes", **saved})
    print(f"  {len(made)} anchors, {sum(1 for m in made if m.get('snapped'))} snapped", flush=True)
    before = [m for m in made if m["page"] <= INSERT_AFTER]
    after = [m for m in made if m["page"] > INSERT_AFTER]
    # One trap before the insertion and one after; picked as objects, since highlights and notes
    # have ids of their own (reanchor-1 matched by id alone and put both traps before it).
    # The after-trap is within reach of the inserted page (old page at most INSERT_AFTER + MAX_SHIFT),
    # or the search never sees the copy (reanchor-2's was on p. 17, out of reach).
    from app.anchors import MAX_SHIFT
    reachable = [m for m in after if m["page"] <= INSERT_AFTER + MAX_SHIFT]
    traps = [rng.choice(before), rng.choice(reachable)] if before and reachable else []
    duplicates = [m["quote"] for m in traps]

    changed = work / "changed.pdf"
    changed_copy(library / args.doc, changed, duplicates)
    shutil.copy2(changed, library / args.doc)
    print("re-ingesting the changed file", flush=True)
    assert worker_module.sync_file(conn, settings, library / args.doc) == "changed"
    second = ingest_until_searchable(conn, settings, doc_id)
    print(f"  searchable again after {second:.0f} s", flush=True)

    rows = []
    for m in made:
        now = conn.execute(f"SELECT * FROM {m['table']} WHERE id = ?", (m["id"],)).fetchone()
        expected_page = m["page"] + (1 if m["page"] > INSERT_AFTER else 0)
        overlap = iou(json.loads(now["rects"]), m["rects"]) if now["page"] == expected_page else 0.0
        status = now["anchor_status"] or "ok"
        verdict = ("unanchored" if status == "unanchored"
                   else "right" if now["page"] == expected_page and overlap >= 0.8 else "wrong")
        rows.append({"table": m["table"], "id": m["id"], "page_before": m["page"], "page_after": now["page"],
                     "expected": expected_page, "status": status, "iou": round(overlap, 3), "verdict": verdict,
                     "trap": any(m is t for t in traps), "quote": m["quote"]})
        print(f"{m['table'][:5]} {m['id']:3d} p.{m['page']:2d} -> p.{now['page']:2d} (want {expected_page:2d}) "
              f"{status:10s} IoU {overlap:.2f} {verdict}{'  [repeated on the inserted page]' if rows[-1]['trap'] else ''}")
    summary = {"anchors": len(rows), "right": sum(r["verdict"] == "right" for r in rows),
               "unanchored": sum(r["verdict"] == "unanchored" for r in rows),
               "wrong": sum(r["verdict"] == "wrong" for r in rows),
               "traps_right": sum(r["trap"] and r["verdict"] == "right" for r in rows),
               "ingest_seconds": [round(first), round(second)]}
    summary["met"] = summary["right"] >= 0.95 * len(rows) and summary["wrong"] == 0
    print(json.dumps(summary, indent=1))
    (RESULTS / f"{args.save}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
