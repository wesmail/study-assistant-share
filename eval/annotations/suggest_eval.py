"""Gate item 3 of roadmap item 4: suggested highlights from an instruction.

    .venv/bin/python eval/annotations/suggest_eval.py run --work DIR --option deep --name sug-deep-1 [--max-usd 1.3]
    ... run --work DIR --option balanced --name sug-balanced-1
    ... pack --name sug-1 --runs sug-deep-1 sug-balanced-1 --seed S     blind pack for graders
    ... score --name sug-1

DIR is a scratch folder outside the project (made on first use): a library with copies of the
PDFs the tasks need, the grammar test page, and a database of its own, so the real ones are
never touched. Each task asks `option` through app/suggest.py, as the viewer's Suggest does, and
saves the suggestions (quote as located, note, page) and the quotes not found. The quote check:
the share of the model's quotes found in the text (gate: at least 90%).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

import fitz
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app import suggest  # noqa: E402
from app.config import Settings, load_settings  # noqa: E402
from app.db.connection import connect  # noqa: E402

TASKS = Path(__file__).with_name("suggest_tasks.yaml")
RESULTS = ROOT / "eval" / "results"
GRADING = ROOT / "eval" / "grading"


def load() -> dict:
    return yaml.safe_load(TASKS.read_text(encoding="utf-8"))


def work_settings(work: Path) -> Settings:
    base = load_settings()
    return Settings(**{**{f: getattr(base, f) for f in base.__dataclass_fields__},
                       "library_dir": work / "library", "data_dir": work / "data", "outputs_dir": work / "outputs"})


def prepare(work: Path) -> Settings:
    """The scratch library and database, made once."""
    work = work.resolve()
    if work.is_relative_to(ROOT):
        raise SystemExit("--work must be outside the project")
    settings = work_settings(work)
    if (work / "data" / "app.sqlite").exists():
        return settings
    (work / "library").mkdir(parents=True)
    tasks = load()["tasks"]
    conn = connect(settings.db_path)
    for name in dict.fromkeys(task["doc"] for task in tasks):
        target = work / "library" / name
        if name == "grammar-test.pdf":
            task = next(t for t in tasks if t["doc"] == name)
            pdf = fitz.open()
            page = pdf.new_page(width=595, height=842)
            y = 72
            for paragraph in task["text"]:
                height = 20 + 14 * (len(paragraph) // 80 + 1)
                page.insert_textbox(fitz.Rect(72, y, 523, y + height), " ".join(paragraph.split()), fontsize=11)
                y += height + 16
            pdf.save(target)
            title = "Grammar test page"
        else:
            shutil.copy2(ROOT / "library" / name, target)
            title = None
        sha = hashlib.sha256(target.read_bytes()).hexdigest()
        real = None
        if title is None:
            source = connect(ROOT / "data" / "app.sqlite", apply_schema=False)
            real = source.execute("SELECT title FROM documents WHERE path = ?", (name,)).fetchone()
            source.close()
        conn.execute("INSERT INTO documents (path, sha256, title, kind, status) VALUES (?, ?, ?, 'pdf', 'indexed')",
                     (name, sha, title or (real["title"] if real else name)))
    conn.commit()
    conn.close()
    return settings


def cmd_run(args) -> None:
    settings = prepare(args.work)
    option = next(o for o in settings.answer_options if o.id == args.option)
    data = load()
    path = RESULTS / f"suggest-{args.name}.jsonl"
    done = {json.loads(line)["id"]: json.loads(line) for line in path.read_text().splitlines()} if path.exists() else {}
    spent = sum(record["cost_usd"] for record in done.values())
    conn = connect(settings.db_path)
    with open(path, "a", encoding="utf-8") as out:
        for task in data["tasks"]:
            if task["id"] in done:
                continue
            if args.max_usd is not None and spent >= args.max_usd:
                print(f"Stopped at ${spent:.2f}: the limit is ${args.max_usd:.2f}")
                break
            doc_id = conn.execute("SELECT id FROM documents WHERE path = ?", (task["doc"],)).fetchone()[0]
            instruction = task.get("instruction", data["study_instruction"])
            started = time.monotonic()
            try:
                result = asyncio.run(suggest.suggest(conn, option.apply(settings), option, doc_id, task["first"],
                                                     task["last"], instruction))
            except Exception as error:      # noqa: BLE001 - recorded, the run goes on
                print(f"{task['id']}: FAILED {error}")
                continue
            record = {"id": task["id"], "option": args.option, "doc": task["doc"], "pages": [task["first"], task["last"]],
                      "instruction": instruction, "asked": result["asked"], "located": len(result["suggestions"]),
                      "not_found": result["not_found"], "cost_usd": result["cost_usd"],
                      "seconds": round(time.monotonic() - started, 1), "batch": result["batch"],
                      "suggestions": [{"id": s["id"], "page": s["page"], "quote": s["quote"], "note": s["body"] or ""}
                                      for s in result["suggestions"]]}
            spent += record["cost_usd"]
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{task['id']}: {record['located']}/{record['asked']} quotes found, {record['seconds']} s, "
                  f"${record['cost_usd']:.3f} (run ${spent:.2f})", flush=True)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    asked = sum(r["asked"] for r in records)
    located = sum(r["located"] for r in records)
    print(f"quote check: {located}/{asked} = {located / max(asked, 1):.1%} found; ${spent:.2f}")


def pages_text(work: Path, doc: str, first: int, last: int) -> str:
    with fitz.open(work / "library" / doc) as pdf:
        return "\n\n".join(f"--- Page {n} ---\n{pdf[n - 1].get_text().strip()}" for n in range(first, last + 1))


GRADER_BRIEF = """You are grading suggested highlights that an assistant made in study documents, following an
instruction. Items come from different sources in a shuffled order; don't try to tell them apart.
Read this whole file and no other file.

For every item you get the instruction, the text of the pages, and the numbered suggestions: each a quote
from the pages and a short note.

Judge each suggestion: "fits" = true if the quote is a place the instruction asks for AND its note is right
(a correct correction, a true statement of what is uncited, a correct and useful point to remember);
otherwise false, with a few words why.

Then judge the set as a whole:
- Study items ("Highlight the key definitions and results"): "worth" = true if a student would be glad to
  have these highlights for studying: they catch the key definitions and results of the pages, without
  many trivial or wrong ones. Missing one minor point is fine; missing the main result is not.
- Grammar items: the item lists the mistakes planted on purpose. "planted_found" = how many of them a
  suggestion marks with a right correction; "false_flags" = suggestions marking something that isn't a
  mistake. "worth" = true if planted_found >= 4 and false_flags <= 1.
- Citation items ("Highlight claims without a citation"): "worth" = true if most flagged claims are really
  claims made without a citation in the text, none of them clearly cited, and the notes are right. A
  sentence of the paper's own setup or a statement of common knowledge counts as a claim only if it asserts
  something checkable.

Write valid JSON to {grades}:
[{{"item": "...", "worth": true, "planted_found": null, "false_flags": null,
   "suggestions": [{{"n": 1, "fits": true, "why": "short, for false ones"}}], "comment": "one sentence"}}]
This batch has {count} items."""


def cmd_pack(args) -> None:
    data = load()
    tasks = {task["id"]: task for task in data["tasks"]}
    work = args.work.resolve()
    folder = GRADING / args.name
    if folder.exists():
        raise SystemExit(f"{folder} exists: a new pack needs a new name")
    rng = random.Random(args.seed)
    items, key = [], {}
    for run in args.runs:
        for line in (RESULTS / f"suggest-{run}.jsonl").read_text().splitlines():
            record = json.loads(line)
            task = tasks[record["id"]]
            item_id = f"{rng.randrange(16 ** 6):06x}"
            key[item_id] = {"run": run, "task": record["id"], "suggestions": len(record["suggestions"])}
            planted = ""
            if task.get("planted"):
                planted = "MISTAKES PLANTED ON PURPOSE (the answer key):\n" + "\n".join(
                    f"- \"{p['wrong']}\" -> {p['right']}" for p in task["planted"]) + "\n\n"
            numbered = "\n".join(f"{n}. (p. {s['page']}) QUOTE: \"{s['quote']}\"\n   NOTE: {s['note']}"
                                 for n, s in enumerate(record["suggestions"], 1)) or "(no suggestions)"
            items.append((item_id, f"=== ITEM {item_id}\nINSTRUCTION: {record['instruction']}\n\n{planted}"
                                   f"PAGES:\n{pages_text(work, task['doc'], task['first'], task['last'])}\n\n"
                                   f"SUGGESTIONS:\n{numbered}\n=== END OF ITEM {item_id}"))
    rng.shuffle(items)
    folder.mkdir(parents=True)
    for batch in range(1, args.batches + 1):
        chosen = items[batch - 1::args.batches]
        brief = GRADER_BRIEF.format(grades=folder / f"grades-{batch}.json", count=len(chosen))
        (folder / f"batch-{batch}.txt").write_text(brief + "\n\n" + "\n\n".join(text for _, text in chosen), encoding="utf-8")
    (folder / "key.json").write_text(json.dumps({"runs": args.runs, "seed": args.seed, "items": key}, indent=1))
    print(f"{len(items)} items in {args.batches} batches in {folder}")


def cmd_score(args) -> None:
    folder = GRADING / args.name
    key = json.loads((folder / "key.json").read_text())
    grades = {g["item"]: g for path in sorted(folder.glob("grades-*.json")) for g in json.loads(path.read_text())}
    missing = sorted(set(key["items"]) - set(grades))
    if missing:
        raise SystemExit(f"no grades for {missing}")
    summary: dict = {}
    rows = []
    for item_id, entry in key["items"].items():
        grade = grades[item_id]
        fits = sum(s["fits"] for s in grade["suggestions"])
        rows.append({"run": entry["run"], "task": entry["task"], "worth": grade["worth"], "fits": fits,
                     "suggestions": entry["suggestions"], "planted_found": grade.get("planted_found"),
                     "false_flags": grade.get("false_flags"), "comment": grade.get("comment", "")})
    for run in key["runs"]:
        mine = [r for r in rows if r["run"] == run]
        study = [r for r in mine if r["task"].startswith("s")]
        summary[run] = {"study_worth": sum(r["worth"] for r in study), "study_sets": len(study),
                        "suggestions_fit": sum(r["fits"] for r in mine), "suggestions": sum(r["suggestions"] for r in mine),
                        "tasks": {r["task"]: {"worth": r["worth"], "planted_found": r["planted_found"],
                                              "false_flags": r["false_flags"], "fits": f"{r['fits']}/{r['suggestions']}",
                                              "comment": r["comment"]}
                                  for r in mine if not r["task"].startswith("s")}}
    print(json.dumps(summary, indent=1))
    (RESULTS / f"suggest-grades-{args.name}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--option", required=True)
    run.add_argument("--name", required=True)
    run.add_argument("--max-usd", type=float)
    run.set_defaults(run=cmd_run)
    pack = commands.add_parser("pack")
    pack.add_argument("--work", type=Path, required=True)
    pack.add_argument("--name", required=True)
    pack.add_argument("--runs", nargs="+", required=True)
    pack.add_argument("--seed", type=int, required=True)
    pack.add_argument("--batches", type=int, default=3)
    pack.set_defaults(run=cmd_pack)
    score = commands.add_parser("score")
    score.add_argument("--name", required=True)
    score.set_defaults(run=cmd_score)
    args = parser.parse_args()
    args.run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
