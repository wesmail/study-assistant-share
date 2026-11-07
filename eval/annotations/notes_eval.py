"""Gate item 5 of roadmap item 4: do answers find and cite the reader's notes?

Runs only on a copy of data/ (STUDY_ASSISTANT_CONFIG pointing at a config whose data_dir is
not the project's data/); it refuses otherwise, since it writes notes.

    STUDY_ASSISTANT_CONFIG=... .venv/bin/python eval/annotations/notes_eval.py setup
    ... notes_eval.py ask --option fast --name notes-fast [--max-usd 1.2]
    ... notes_eval.py score --name notes-fast
    ... notes_eval.py recall          the 20 book questions' chunk recall, with the notes present

`setup` writes the ten notes of notes_testset.yaml, each anchored to its quote as PyMuPDF
finds it on the page, snapped like a note from the viewer. `ask` asks each note's question as
the app does (tools mode, search proposals off, web off) and saves eval/results/answers-NAME.jsonl.
`score`: the note among the sources, and cited. Gate: in sources for at least 9 of 10; cited in
at least 8 of 10, in Deep and in Fast.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import fitz
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "eval"))

from app import notes  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.db.connection import connect  # noqa: E402
from app.ingest.embed import embed_texts  # noqa: E402

TESTSET = Path(__file__).with_name("notes_testset.yaml")
RESULTS = ROOT / "eval" / "results"


def safe_settings():
    settings = load_settings()
    if settings.data_dir.resolve() == (ROOT / "data").resolve():
        raise SystemExit("Refused: this writes notes, so it runs only on a copy of data/ (STUDY_ASSISTANT_CONFIG)")
    return settings


def items() -> list[dict]:
    return yaml.safe_load(TESTSET.read_text(encoding="utf-8"))["notes"]


def cmd_setup(args) -> None:
    settings = safe_settings()
    conn = connect(settings.db_path)
    if conn.execute("SELECT count(*) FROM notes").fetchone()[0]:
        raise SystemExit("The copy has notes already: set up a fresh copy")
    made = {}
    for item in items():
        passage = None
        if item.get("doc"):
            doc_id = conn.execute("SELECT id FROM documents WHERE path = ?", (item["doc"],)).fetchone()[0]
            with fitz.open(settings.library_dir / item["doc"]) as pdf:
                rects = [list(rect) for rect in pdf[item["page"] - 1].search_for(item["quote"])]
            if not rects:
                raise SystemExit(f"{item['id']}: quote not found on p. {item['page']}")
            passage = {"doc_id": doc_id, "page": item["page"], "rects": rects, "quote": item["quote"]}
        note = notes.create(conn, settings, item["body"].strip(), item["tags"], passage)
        made[item["id"]] = note["id"]
        print(f"{item['id']}: note {note['id']}, snapped {note['snapped']}, quote {note['quote'][:60]!r}")
    print(f"indexed {notes.index_notes(conn, settings, embed_texts)} vectors")
    (RESULTS / "notes-testset-ids.json").write_text(json.dumps(made, indent=1) + "\n")


def cmd_ask(args) -> None:
    from run_eval import ask

    configured = safe_settings()
    option = next(o for o in configured.answer_options if o.id == args.option)
    settings = option.apply(configured)
    note_ids = json.loads((RESULTS / "notes-testset-ids.json").read_text())
    path = RESULTS / f"answers-{args.name}.jsonl"
    done = {json.loads(line)["id"] for line in path.read_text().splitlines()} if path.exists() else set()
    spent = sum(json.loads(line).get("cost_usd") or 0 for line in path.read_text().splitlines()) if path.exists() else 0.0
    with open(path, "a", encoding="utf-8") as out:
        for item in items():
            if item["id"] in done:
                continue
            if args.max_usd is not None and spent >= args.max_usd:
                print(f"Stopped at ${spent:.2f}: the limit is ${args.max_usd:.2f}")
                break
            record = asyncio.run(ask({"id": item["id"], "question": item["question"]}, settings, option.provider,
                                     f"eval:{args.name}", option.vision, tools=True, propose=False))
            if record.get("error"):
                print(f"{item['id']}: FAILED, not saved: {record['error']}")
                continue
            record["note_id"] = note_ids[item["id"]]
            spent += record.get("cost_usd") or 0.0
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            numbers = note_sources(record)
            print(f"{item['id']}: note source {numbers}, cited {record.get('citations')}, "
                  f"{record['total_seconds']:.0f}s, ${record.get('cost_usd') or 0:.3f}", flush=True)


def note_sources(record: dict) -> list[int]:
    """The source numbers that are this question's note: from the first search, or a tool search."""
    return [s["id"] for s in record.get("sources", []) + record.get("more_sources", [])
            if s.get("note_id") == record["note_id"]]


def cmd_score(args) -> None:
    records = [json.loads(line) for line in (RESULTS / f"answers-{args.name}.jsonl").read_text().splitlines()]
    rows = []
    for record in records:
        numbers = note_sources(record)
        rows.append({"id": record["id"], "in_sources": bool(numbers),
                     "cited": bool(set(numbers) & set(record.get("citations") or [])),
                     "other_notes": sorted({s.get("note_id") for s in record.get("sources", [])
                                            if s.get("kind") == "note" and s.get("note_id") != record["note_id"]}),
                     "seconds": round(record["total_seconds"]), "cost_usd": record.get("cost_usd")})
        print(rows[-1])
    summary = {"questions": len(rows), "in_sources": sum(r["in_sources"] for r in rows),
               "cited": sum(r["cited"] for r in rows),
               "cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 3)}
    summary["met"] = summary["in_sources"] >= 9 and summary["cited"] >= 8 and len(rows) == 10
    print(summary)
    (RESULTS / f"notes-{args.name}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")


def cmd_recall(args) -> None:
    import run_eval

    safe_settings()
    conn = connect(load_settings().db_path, apply_schema=False)
    count = conn.execute("SELECT count(*) FROM notes").fetchone()[0]
    print(f"{count} notes present")
    run_eval.RESULTS = Path(args.out)       # recall.json goes there, not over the committed one
    run_eval.RESULTS.mkdir(parents=True, exist_ok=True)
    run_eval.cmd_recall(argparse.Namespace(k=6, context=False, questions=None, figures=False, dev=False,
                                           web=False, include_stale=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup").set_defaults(run=cmd_setup)
    ask = commands.add_parser("ask")
    ask.add_argument("--option", required=True)
    ask.add_argument("--name", required=True)
    ask.add_argument("--max-usd", type=float)
    ask.set_defaults(run=cmd_ask)
    score = commands.add_parser("score")
    score.add_argument("--name", required=True)
    score.set_defaults(run=cmd_score)
    recall = commands.add_parser("recall")
    recall.add_argument("--out", required=True, help="a folder for recall.json, not eval/results")
    recall.set_defaults(run=cmd_recall)
    args = parser.parse_args()
    args.run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
