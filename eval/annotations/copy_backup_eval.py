"""Gate items 2 and 8 of roadmap item 4: annotated copies, and notes and highlights in backups.

    .venv/bin/python eval/annotations/copy_backup_eval.py --work DIR --accept RUN [--save NAME]

DIR is the scratch folder of suggest_eval.py (its own library copies and database, with the
suggestions of the runs). Steps:

1. The reader's own highlights: the 20 gate selections of step 1 (eval/annotations/selections.json),
   made again here from their PyMuPDF words, and five notes on passages.
2. Before anything is accepted: an annotated copy of every document. No AI suggestion may be in
   any of them (nothing written before acceptance).
3. Every suggestion of run RUN accepted; the copies made again. Checked: each original's sha256
   unchanged (against the project's library too); the same page count; every accepted
   highlight in the copy as a highlight annotation on its page whose words read like its quote
   (at least 95% similar) and whose comment is its note word for word; every note as an underline; pdf.js opens each copy and counts the same
   annotations; nothing written under the library.
4. A real backup of DIR (app/backup.py, archive in DIR/backups), restored into a fresh folder:
   the counts of notes and highlights in the manifest and in the restored database match, and
   the annotated copies come back byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

from app import annotated, annotations, backup, notes  # noqa: E402
from app.annotations import area, overlap, similarity  # noqa: E402
from app.db.connection import connect  # noqa: E402
from suggest_eval import prepare, work_settings  # noqa: E402

RESULTS = ROOT / "eval" / "results"
SELECTIONS = Path(__file__).with_name("selections.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words_under(page: fitz.Page, annot: fitz.Annot) -> str:
    """The words a highlight covers, by the rule snapping uses: at least half of a word's box
    under its quads. (A word's centre inside a quad, the first rule, misses a root sign's or a
    subscript's words.)"""
    points = annot.vertices or []
    quads = [tuple(fitz.Quad(points[i:i + 4]).rect) for i in range(0, len(points), 4)]
    found = [w for w in page.get_text("words", sort=False)
             if area(w[:4]) > 0 and sum(overlap(w[:4], quad) for quad in quads) >= 0.5 * area(w[:4])]
    return " ".join(w[4] for w in found)


def pdfjs_counts(files: list[Path]) -> dict:
    script = ROOT / "frontend" / "e2e" / "pdfjs_annotations.mjs"
    done = subprocess.run(["node", str(script), *map(str, files)], cwd=ROOT / "frontend", capture_output=True,
                          text=True, check=True, env={"PATH": f"{ROOT / '.venv' / 'bin'}:/usr/bin:/bin"})
    return json.loads(done.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--accept", required=True, help="the suggest_eval run whose suggestions are accepted")
    parser.add_argument("--save", default="copy-backup-1")
    parser.add_argument("--selections-result", type=Path,
                        help="a frontend/e2e/annotations.mjs result: highlights from what the browser sent")
    args = parser.parse_args()
    work = args.work.resolve()
    settings = prepare(work) if not (work / "data" / "app.sqlite").exists() else work_settings(work)
    conn = connect(settings.db_path)
    docs = {row["path"]: row["id"] for row in conn.execute("SELECT id, path FROM documents")}
    report: dict = {}

    # 0. A fresh folder: the run's suggestions saved again through app/suggest.py's own save (each
    # quote located in the PDF), without asking the model again.
    if not conn.execute("SELECT 1 FROM annotations WHERE origin = 'ai' LIMIT 1").fetchone():
        from app import suggest
        resaved = []
        for line in (RESULTS / f"suggest-{args.accept}.jsonl").read_text().splitlines():
            record = json.loads(line)
            items = [suggest.Suggestion(page=s["page"], quote=s["quote"], note=s["note"]) for s in record["suggestions"]]
            result = suggest.save(conn, settings, docs[record["doc"]], items, record["instruction"],
                                  tuple(record["pages"]))
            resaved.append({"task": record["id"], "batch": result["batch"], "saved": len(result["suggestions"]),
                            "of": len(items)})
        (work / "batches.json").write_text(json.dumps(resaved, indent=1))
        report["resaved_suggestions"] = {"saved": sum(r["saved"] for r in resaved), "of": sum(r["of"] for r in resaved)}

    # 1. The reader's own highlights and notes.
    if not conn.execute("SELECT 1 FROM annotations WHERE origin = 'user' LIMIT 1").fetchone():
        real = connect(ROOT / "data" / "app.sqlite", apply_schema=False)
        real_paths = dict(real.execute("SELECT id, path FROM documents").fetchall())
        real.close()
        made = 0
        # What the browser sent in a gate-1 run, if given: a real selection's boxes and text. Otherwise
        # the planned words' line boxes, which on a line with a stacked fraction also cover its other
        # parts, so snapping (rightly) refuses them (copy-backup-2: two such).
        sent = {}
        if args.selections_result:
            sent = {r["id"]: r["sent"] for r in json.loads(args.selections_result.read_text())["results"] if r.get("sent")}
        for number, selection in enumerate(json.loads(SELECTIONS.read_text())["selections"]):
            path = real_paths[selection["doc_id"]]
            if path not in docs:
                continue
            if selection["id"] in sent:
                rects, quote = sent[selection["id"]]["rects"], sent[selection["id"]]["quote"]
            else:
                lines: dict = {}
                for word in selection["words"]:
                    lines.setdefault(tuple(word["line"]), []).append(word["box"])
                rects = [[min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
                         for boxes in lines.values()]
                quote = " ".join(w["text"] for w in selection["words"])
            if number % 4 == 3:
                notes.create(conn, settings, f"Test note on {quote[:30]}: $x^2$", ["test"],
                             {"doc_id": docs[path], "page": selection["page"], "rects": rects, "quote": quote})
            else:
                annotations.create(conn, settings, docs[path], selection["page"], rects, quote,
                                   color=("yellow", "green", "pink")[number % 3])
            made += 1
        report["made_from_selections"] = made

    library_before = {path: sha(settings.library_dir / path) for path in docs}
    project_library = {path: sha(ROOT / "library" / path) for path in docs if (ROOT / "library" / path).exists()}

    # 2. Before acceptance: no suggestion in any copy.
    before = {}
    for path, doc_id in docs.items():
        result = annotated.write_copy(conn, settings, doc_id)
        with fitz.open(annotated.copy_path(settings, doc_id)) as pdf:
            ai = sum(1 for page in pdf for a in page.annots() if a.info.get("title", "").startswith("AI"))
        before[path] = {"ai_annotations": ai, "suggested_left_out": result["suggested_left_out"]}
    report["before_acceptance"] = {"ai_annotations_in_copies": sum(v["ai_annotations"] for v in before.values()),
                                   "suggestions_left_out": sum(v["suggested_left_out"] for v in before.values())}

    # 3. Accept one run's suggestions; make the copies again; check them.
    batches = ([entry["batch"] for entry in json.loads((work / "batches.json").read_text())] if (work / "batches.json").exists()
               else [json.loads(line)["batch"] for line in (RESULTS / f"suggest-{args.accept}.jsonl").read_text().splitlines()])
    accepted = 0
    for batch in batches:
        accepted += conn.execute("UPDATE annotations SET status = 'accepted' WHERE batch = ? AND status = 'suggested'",
                                 (batch,)).rowcount
    conn.commit()
    report["accepted_suggestions"] = accepted
    rows, copies = [], []
    for path, doc_id in docs.items():
        written = annotated.write_copy(conn, settings, doc_id)
        out = annotated.copy_path(settings, doc_id)
        copies.append(out)
        with fitz.open(out) as pdf, fitz.open(settings.library_dir / path) as original:
            same_pages = pdf.page_count == original.page_count
            on_page = {}
            for page in pdf:
                on_page[page.number + 1] = [(a.type[1], a.info.get("title", ""), words_under(page, a), a.info.get("content", ""))
                                            for a in page.annots()]
            for row in conn.execute("SELECT * FROM annotations WHERE doc_id = ? AND status = 'accepted'", (doc_id,)):
                found = [(similarity(words, row["quote"]), content == (row["body"] or ""))
                         for kind, title, words, content in on_page.get(row["page"], []) if kind == "Highlight"]
                best = max((sim for sim, _ in found), default=0.0)
                # Its note (a suggestion's, or the reader's comment) as the annotation's comment, word for word.
                comment = any(sim >= 0.95 and same for sim, same in found)
                rows.append({"doc": path, "id": row["id"], "origin": row["origin"], "page": row["page"],
                             "kind": "highlight", "similarity": round(best, 3), "comment": comment,
                             "ok": best >= 0.95 and comment and same_pages})
            for row in conn.execute("SELECT * FROM notes WHERE doc_id = ? AND page IS NOT NULL", (doc_id,)):
                found = [similarity(words, row["quote"]) for kind, title, words, _ in on_page.get(row["page"], [])
                         if kind == "Underline" and title == "Your note"]
                best = max(found, default=0.0)
                rows.append({"doc": path, "id": row["id"], "origin": "note", "page": row["page"], "kind": "note",
                             "similarity": round(best, 3), "ok": best >= 0.95 and same_pages})
        report.setdefault("copies", {})[path] = written
    seen = pdfjs_counts(copies)
    pdfjs_ok = all("error" not in seen[str(c)] for c in copies)
    pdfjs_total = sum(n for c in copies for page in seen[str(c)].get("annotations", {}).values()
                      for kind, n in page.items() if kind in ("Highlight", "Underline"))
    expected_total = sum(1 for r in rows)
    originals_same = all(sha(settings.library_dir / p) == h for p, h in library_before.items())
    project_same = all(sha(ROOT / "library" / p) == h for p, h in project_library.items())
    in_library = [str(p) for p in settings.library_dir.rglob("*annotated*")]
    report["copy_check"] = {
        "anchors": len(rows), "ok": sum(r["ok"] for r in rows),
        "ai_notes_as_comments": sum(1 for r in rows if r["origin"] == "ai" and r["comment"]),
        "ai_highlights": sum(1 for r in rows if r["origin"] == "ai"),
        "below_95": [r for r in rows if not r["ok"]],
        "originals_unchanged": originals_same, "project_library_unchanged": project_same,
        "pdfjs_opened_all": pdfjs_ok, "pdfjs_annotations": pdfjs_total, "expected_annotations": expected_total,
        "files_in_library": in_library,
    }

    # 4. Backup and restore.
    fake_root = work / "fake-root"
    fake_root.mkdir(exist_ok=True)
    backup_settings = replace(settings, backup_dir=work / "backups", backup_local_dir=work / "backup-local")
    made = backup.create(backup_settings, fake_root)
    target = work / f"restored-{made['stamp']}"
    (target / "backend" / "app").mkdir(parents=True)
    (target / "backend" / "app" / "config.py").write_text("# placeholder: restore() wants a checkout\n")
    backup.restore(Path(made["archive"]), target, backup_settings)
    restored = connect(target / "data" / "app.sqlite", apply_schema=False)
    counts = {table: restored.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in ("notes", "annotations")}
    source = {table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in ("notes", "annotations")}
    copies_back = all(sha(target / "outputs" / "annotated" / c.name) == sha(c) for c in copies)
    report["backup"] = {"archive": made["archive"], "manifest_counts": {k: made["database"]["counts"][k] for k in source},
                        "source_counts": source, "restored_counts": counts, "annotated_copies_restored": copies_back,
                        "met": counts == source == {k: made["database"]["counts"][k] for k in source} and copies_back}
    report["gate_2_met"] = (report["before_acceptance"]["ai_annotations_in_copies"] == 0
                            and report["copy_check"]["ok"] == len(rows) and originals_same and project_same
                            and pdfjs_ok and pdfjs_total == expected_total and not in_library)
    print(json.dumps({k: v for k, v in report.items() if k != "copies"}, indent=1))
    (RESULTS / f"{args.save}.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
