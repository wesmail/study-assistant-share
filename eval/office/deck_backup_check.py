"""Gate item 8 of roadmap item 7: a deck's highlights, notes, annotated copies and render
survive a backup and a restore.

    STUDY_ASSISTANT_CONFIG=<a config on a COPY of data/> python eval/office/deck_backup_check.py WORK_DIR

On the copy: a highlight on WE-DPG26 slide 12 and a note on Alberta slide 19 (made as the
viewer makes them, snapped to the render's words), and an annotated copy of each deck. Then
app/backup.py's own create (into WORK_DIR, with a project root that has no .env, so no key is
copied) and restore (into a fresh folder under WORK_DIR), and a comparison: the database's
counts, every file under data/renders and the decks' parse folders, and the annotated copies,
byte for byte. Refuses to run unless data_dir is a scratch copy.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import fitz  # noqa: E402

from app import annotated, annotations, backup, notes  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.db.connection import connect  # noqa: E402
from app.ingest import deck  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(folder: Path) -> dict[str, str]:
    return {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def selection(settings, doc, page: int, text: str) -> dict:
    with fitz.open(deck.render_path(settings, doc["sha256"])) as pdf:
        box = pdf[page - 1].search_for(text)[0]
    return {"doc_id": doc["id"], "page": page, "rects": [[box.x0, box.y0, box.x1, box.y1]], "quote": text}


def counts(db: Path) -> dict:
    with sqlite3.connect(db) as conn:
        return {table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                for table in ("documents", "chunks", "notes", "annotations")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    settings = load_settings()
    if "scratchpad" not in str(settings.data_dir) or "scratchpad" not in str(settings.outputs_dir):
        raise SystemExit(f"Refusing: {settings.data_dir} / {settings.outputs_dir} aren't scratch copies")
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    settings = dataclasses.replace(settings, backup_dir=work / "archives", backup_local_dir=work / "local")

    conn = connect(settings.db_path, apply_schema=False)
    docs = {row["path"]: row for row in conn.execute("SELECT * FROM documents WHERE kind = 'pptx'")}
    dpg, alb = docs["WE-DPG26.pptx"], docs["Alberta.pptx"]
    highlight = annotations.create(conn, settings, dpg["id"], 12, **{k: v for k, v in selection(
        settings, dpg, 12, "SeismoGPT: Implications for the ET").items() if k in ("rects", "quote")})
    note = notes.create(conn, settings, "Cross section from the CMS angular distribution; systematics 5%, 2%, 7%.", ["hades"],
                        selection(settings, alb, 19, "Production Cross Section"))
    conn.commit()
    copies = {row["path"]: annotated.write_copy(conn, settings, row["id"]) for row in (dpg, alb)}
    conn.close()
    made = {"highlight": {k: highlight[k] for k in ("id", "page", "quote", "snapped")},
            "note": {k: note[k] for k in ("id", "page", "quote")}, "copies": copies}

    fake_root = work / "root"          # no .env here, so no key is copied anywhere
    fake_root.mkdir()
    archive = backup.create(settings, root=fake_root)
    target = work / "restored"
    (target / "backend" / "app").mkdir(parents=True)
    shutil.copy(ROOT / "backend" / "app" / "config.py", target / "backend" / "app" / "config.py")
    restored = backup.restore(Path(archive["archive"]), target, settings)

    before_db, after_db = counts(settings.db_path), counts(target / "data" / "app.sqlite")
    renders = tree(settings.data_dir / "renders") == tree(target / "data" / "renders")
    parsed = all(tree(settings.parsed_dir / row["sha256"]) == tree(target / "data" / "parsed" / row["sha256"])
                 for row in (dpg, alb))
    copies_equal = tree(settings.outputs_dir / "annotated") == tree(target / "outputs" / "annotated")
    with fitz.open(target / "outputs" / "annotated" / f"{dpg['id']}-annotated.pdf") as pdf:
        on_slide_12 = len(list(pdf[11].annots()))
    result = {"made": made, "archive_files": len(archive["files"]), "env_copy": archive["env_copy"],
              "counts_before": before_db, "counts_after": after_db, "counts_equal": before_db == after_db,
              "renders_equal": renders, "deck_parse_folders_equal": parsed, "annotated_copies_equal": copies_equal,
              "highlight_in_restored_copy_slide_12": on_slide_12, "restore": {k: v for k, v in restored.items()
                                                                             if k in ("files", "written", "skipped")}}
    (work / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(result, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
